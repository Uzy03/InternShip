#!/usr/bin/env python3
"""
町丁の重心座標を作成するスクリプト

GMLファイルから町丁の重心座標を抽出し、CSVファイルとして保存します。
"""

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_column_names(gdf):
    """
    GMLファイルの列名を自動推定して、県/市/区/町名の列を特定する
    """
    columns = gdf.columns.tolist()
    
    # 候補となる列名パターン
    pref_patterns = ['N03_001', 'PREF_NAME', 'PREF_CD', '都道府県名', '県名', 'PREF']
    city_patterns = ['N03_002', 'CITY_NAME', 'CITY_CD', '市区町村名', '市名', '区名', 'CITY']
    ward_patterns = ['N03_003', 'WARD_NAME', 'WARD_CD', '区名', '町名', 'S_AREA']
    town_patterns = ['N03_004', 'TOWN_NAME', 'TOWN_CD', '町丁目名', '町名', '丁目名', 'S_NAME']
    
    detected = {}
    
    # 県名列の検出（PREF_NAMEを優先）
    if 'PREF_NAME' in columns:
        detected['pref'] = 'PREF_NAME'
    else:
        for pattern in pref_patterns:
            for col in columns:
                if pattern in col.upper() or col.upper() in pattern:
                    detected['pref'] = col
                    break
            if 'pref' in detected:
                break
    
    # 市名列の検出（CITY_NAMEを優先）
    if 'CITY_NAME' in columns:
        detected['city'] = 'CITY_NAME'
    else:
        for pattern in city_patterns:
            for col in columns:
                if pattern in col.upper() or col.upper() in pattern:
                    detected['city'] = col
                    break
            if 'city' in detected:
                break
    
    # 区名列の検出
    for pattern in ward_patterns:
        for col in columns:
            if pattern in col.upper() or col.upper() in pattern:
                detected['ward'] = col
                break
        if 'ward' in detected:
            break
    
    # 町名列の検出（S_NAMEを優先）
    if 'S_NAME' in columns:
        detected['town'] = 'S_NAME'
    else:
        for pattern in town_patterns:
            for col in columns:
                if pattern in col.upper() or col.upper() in pattern:
                    detected['town'] = col
                    break
            if 'town' in detected:
                break
    
    return detected


def filter_by_location(gdf, detected_cols, pref=None, city=None, ward=None):
    """
    指定された県/市/区でフィルタリング
    """
    filtered_gdf = gdf.copy()
    
    if pref and 'pref' in detected_cols:
        filtered_gdf = filtered_gdf[filtered_gdf[detected_cols['pref']].str.contains(pref, na=False)]
        logger.info(f"県名フィルタ適用: {pref} -> {len(filtered_gdf)}件")
    
    if city and 'city' in detected_cols:
        filtered_gdf = filtered_gdf[filtered_gdf[detected_cols['city']].str.contains(city, na=False)]
        logger.info(f"市名フィルタ適用: {city} -> {len(filtered_gdf)}件")
    
    if ward and 'ward' in detected_cols:
        filtered_gdf = filtered_gdf[filtered_gdf[detected_cols['ward']].str.contains(ward, na=False)]
        logger.info(f"区名フィルタ適用: {ward} -> {len(filtered_gdf)}件")
    
    return filtered_gdf


def filter_by_town_list(gdf, detected_cols, town_list_file):
    """
    町丁目リストでフィルタリング
    """
    if not town_list_file or not Path(town_list_file).exists():
        return gdf
    
    try:
        town_df = pd.read_csv(town_list_file)
        if 'town' not in town_df.columns:
            logger.warning(f"町丁目リストファイルに'town'列が見つかりません: {town_list_file}")
            return gdf
        
        town_names = set(town_df['town'].dropna().astype(str))
        
        if 'town' in detected_cols:
            filtered_gdf = gdf[gdf[detected_cols['town']].astype(str).isin(town_names)]
            logger.info(f"町丁目リストフィルタ適用: {len(town_names)}件 -> {len(filtered_gdf)}件")
            return filtered_gdf
        else:
            logger.warning("町名列が見つからないため、町丁目リストフィルタをスキップします")
            return gdf
            
    except Exception as e:
        logger.error(f"町丁目リストファイルの読み込みエラー: {e}")
        return gdf


def add_town_id(gdf, detected_cols, master_file):
    """
    マスターファイルからtown_idを付与
    """
    if not master_file or not Path(master_file).exists():
        logger.info("マスターファイルが指定されていないか存在しません。town列をIDとして使用します。")
        if 'town' in detected_cols:
            gdf['town_id'] = gdf[detected_cols['town']].astype(str)
        else:
            gdf['town_id'] = gdf.index.astype(str)
        return gdf
    
    try:
        master_df = pd.read_csv(master_file)
        if 'town_id' not in master_df.columns or 'town' not in master_df.columns:
            logger.warning(f"マスターファイルに'town_id'または'town'列が見つかりません: {master_file}")
            if 'town' in detected_cols:
                gdf['town_id'] = gdf[detected_cols['town']].astype(str)
            else:
                gdf['town_id'] = gdf.index.astype(str)
            return gdf
        
        # 町名でマージ
        if 'town' in detected_cols:
            merged_gdf = gdf.merge(
                master_df[['town_id', 'town']], 
                left_on=detected_cols['town'], 
                right_on='town', 
                how='left'
            )
            # マッチしなかった場合は元の町名を使用
            merged_gdf['town_id'] = merged_gdf['town_id'].fillna(merged_gdf[detected_cols['town']].astype(str))
            logger.info(f"マスターファイルからID付与: {len(merged_gdf)}件")
            return merged_gdf
        else:
            logger.warning("町名列が見つからないため、マスターファイルのID付与をスキップします")
            if 'town' in detected_cols:
                gdf['town_id'] = gdf[detected_cols['town']].astype(str)
            else:
                gdf['town_id'] = gdf.index.astype(str)
            return gdf
            
    except Exception as e:
        logger.error(f"マスターファイルの読み込みエラー: {e}")
        if 'town' in detected_cols:
            gdf['town_id'] = gdf[detected_cols['town']].astype(str)
        else:
            gdf['town_id'] = gdf.index.astype(str)
        return gdf


def normalize_town_name_for_panel(town_name: str) -> str:
    """町丁名をpanel_raw.csvの形式に正規化（漢数字を半角数字に変換）"""
    if pd.isna(town_name):
        return town_name
    
    town_name = str(town_name)
    
    # 漢数字を半角数字に変換（「九」は固有名詞の一部なので除外）
    kanji_to_num = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '十': '10'
    }
    
    for kanji, num in kanji_to_num.items():
        town_name = town_name.replace(kanji, num)
    
    # 全角数字を半角数字に変換
    town_name = town_name.replace('０', '0').replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4')
    town_name = town_name.replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9')
    
    return town_name


def normalize_town_name_alternative(town_name: str) -> str:
    """町丁名を別の正規化方法で変換（より柔軟なマッチング）"""
    if pd.isna(town_name):
        return town_name
    
    town_name = str(town_name)
    
    # 漢数字を半角数字に変換（すべての漢数字を変換）
    kanji_to_num = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
    }
    
    for kanji, num in kanji_to_num.items():
        town_name = town_name.replace(kanji, num)
    
    # 全角数字を半角数字に変換
    town_name = town_name.replace('０', '0').replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4')
    town_name = town_name.replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9')
    
    return town_name

def extract_centroids(gdf, detected_cols):
    """
    重心座標を抽出
    """
    # 座標系をWGS84 (EPSG:4326) に変換
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # 重心を計算
    centroids = gdf.geometry.centroid
    
    # 座標を抽出
    gdf['lon'] = centroids.x
    gdf['lat'] = centroids.y
    
    # 出力用のDataFrameを作成
    output_columns = ['town_id', 'lon', 'lat']
    
    # 町名も含める場合（panel_raw.csvの形式に正規化）
    if 'town' in detected_cols:
        output_columns.append('town')
        gdf['town'] = gdf[detected_cols['town']].astype(str).apply(normalize_town_name_for_panel)
    
    result_df = gdf[output_columns].copy()
    
    # 重複を除去
    result_df = result_df.drop_duplicates(subset=['town_id'])
    
    # 欠損値を除去
    result_df = result_df.dropna(subset=['lon', 'lat'])
    
    logger.info(f"重心座標抽出完了: {len(result_df)}件")
    
    return result_df


def filter_by_panel_towns(result_df, panel_towns_file):
    """
    panel_raw.csvの町丁名でフィルタリング
    """
    if not panel_towns_file or not Path(panel_towns_file).exists():
        logger.warning(f"panel_raw.csvファイルが見つかりません: {panel_towns_file}")
        return result_df
    
    try:
        panel_df = pd.read_csv(panel_towns_file)
        if 'town' not in panel_df.columns:
            logger.warning(f"panel_raw.csvに'town'列が見つかりません")
            return result_df
        
        panel_towns = set(panel_df['town'].dropna().astype(str))
        
        # 正規化された町丁名でフィルタリング
        filtered_df = result_df[result_df['town'].isin(panel_towns)]
        logger.info(f"panel_raw.csvでフィルタ適用: {len(panel_towns)}件 -> {len(filtered_df)}件")
        
        # マッチしなかった町丁名をログ出力
        matched_towns = set(filtered_df['town'])
        unmatched_towns = panel_towns - matched_towns
        if unmatched_towns:
            logger.warning(f"マッチしなかった町丁名（最初の10件）: {list(unmatched_towns)[:10]}")
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"panel_raw.csvの読み込みエラー: {e}")
        return result_df


def search_unmatched_towns_with_alternative_normalization(gdf, detected_cols, panel_towns_file, matched_towns):
    """
    マッチしなかった町丁を別の正規化方法で再検索
    """
    if not panel_towns_file or not Path(panel_towns_file).exists():
        return pd.DataFrame()
    
    try:
        panel_df = pd.read_csv(panel_towns_file)
        if 'town' not in panel_df.columns:
            return pd.DataFrame()
        
        panel_towns = set(panel_df['town'].dropna().astype(str))
        unmatched_towns = panel_towns - matched_towns
        
        if not unmatched_towns:
            logger.info("マッチしなかった町丁はありません")
            return pd.DataFrame()
        
        logger.info(f"マッチしなかった町丁を別の正規化方法で再検索: {len(unmatched_towns)}件")
        
        # GMLファイルの町丁名を別の正規化方法で変換
        gdf_copy = gdf.copy()
        if 'town' in detected_cols:
            gdf_copy['town_alt'] = gdf_copy[detected_cols['town']].astype(str).apply(normalize_town_name_alternative)
        
        # 座標系をWGS84 (EPSG:4326) に変換
        if gdf_copy.crs != 'EPSG:4326':
            gdf_copy = gdf_copy.to_crs('EPSG:4326')
        
        # 重心を計算
        centroids = gdf_copy.geometry.centroid
        gdf_copy['lon'] = centroids.x
        gdf_copy['lat'] = centroids.y
        
        # マッチしなかった町丁で再検索
        additional_matches = gdf_copy[gdf_copy['town_alt'].isin(unmatched_towns)]
        
        if len(additional_matches) > 0:
            # 出力用のDataFrameを作成
            output_columns = ['town_id', 'lon', 'lat', 'town']
            additional_matches['town'] = additional_matches['town_alt']
            additional_matches['town_id'] = additional_matches[detected_cols['town']].astype(str)
            
            result_df = additional_matches[output_columns].copy()
            
            # 重複を除去
            result_df = result_df.drop_duplicates(subset=['town_id'])
            
            # 欠損値を除去
            result_df = result_df.dropna(subset=['lon', 'lat'])
            
            logger.info(f"別の正規化方法で追加マッチ: {len(result_df)}件")
            
            # 追加マッチした町丁名をログ出力
            additional_matched_towns = set(result_df['town'])
            logger.info(f"追加マッチした町丁名: {list(additional_matched_towns)}")
            
            return result_df
        else:
            logger.info("別の正規化方法でもマッチしませんでした")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"別の正規化方法での再検索エラー: {e}")
        return pd.DataFrame()


def search_unmatched_towns_with_flexible_matching(gdf, detected_cols, panel_towns_file, matched_towns):
    """
    マッチしなかった町丁を柔軟なマッチング方法で再検索
    """
    if not panel_towns_file or not Path(panel_towns_file).exists():
        return pd.DataFrame()
    
    try:
        panel_df = pd.read_csv(panel_towns_file)
        if 'town' not in panel_df.columns:
            return pd.DataFrame()
        
        panel_towns = set(panel_df['town'].dropna().astype(str))
        unmatched_towns = panel_towns - matched_towns
        
        if not unmatched_towns:
            logger.info("マッチしなかった町丁はありません")
            return pd.DataFrame()
        
        logger.info(f"マッチしなかった町丁を柔軟なマッチング方法で再検索: {len(unmatched_towns)}件")
        
        # 座標系をWGS84 (EPSG:4326) に変換
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # 重心を計算
        centroids = gdf.geometry.centroid
        gdf['lon'] = centroids.x
        gdf['lat'] = centroids.y
        
        additional_matches = []
        
        # 各マッチしなかった町丁に対して柔軟なマッチングを試行
        for target_town in unmatched_towns:
            # パターン1: 部分マッチング（町名の基本部分で検索）
            base_name = target_town.replace('丁目', '').replace('町', '')
            matching_towns = gdf[gdf[detected_cols['town']].str.contains(base_name, na=False)]
            
            if len(matching_towns) > 0:
                # 最も近い町丁を選択（文字列の類似度で判定）
                best_match = None
                best_similarity = 0
                
                for _, row in matching_towns.iterrows():
                    gml_town = str(row[detected_cols['town']])
                    # 簡単な類似度計算（共通文字数 / 最大文字数）
                    common_chars = len(set(target_town) & set(gml_town))
                    max_chars = max(len(target_town), len(gml_town))
                    similarity = common_chars / max_chars if max_chars > 0 else 0
                    
                    if similarity > best_similarity and similarity > 0.5:  # 50%以上の類似度
                        best_similarity = similarity
                        best_match = row
                
                if best_match is not None:
                    # マッチした町丁を追加
                    match_data = {
                        'town_id': str(best_match[detected_cols['town']]),
                        'lon': best_match['lon'],
                        'lat': best_match['lat'],
                        'town': target_town  # 元の町丁名を使用
                    }
                    additional_matches.append(match_data)
                    logger.info(f"柔軟マッチング成功: {target_town} -> {best_match[detected_cols['town']]} (類似度: {best_similarity:.2f})")
        
        if additional_matches:
            result_df = pd.DataFrame(additional_matches)
            
            # 重複を除去
            result_df = result_df.drop_duplicates(subset=['town_id'])
            
            # 欠損値を除去
            result_df = result_df.dropna(subset=['lon', 'lat'])
            
            logger.info(f"柔軟なマッチング方法で追加マッチ: {len(result_df)}件")
            
            # 追加マッチした町丁名をログ出力
            additional_matched_towns = set(result_df['town'])
            logger.info(f"追加マッチした町丁名: {list(additional_matched_towns)}")
            
            return result_df
        else:
            logger.info("柔軟なマッチング方法でもマッチしませんでした")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"柔軟なマッチング方法での再検索エラー: {e}")
        return pd.DataFrame()


def search_unmatched_towns_with_partial_matching(gdf, detected_cols, panel_towns_file, matched_towns):
    """
    マッチしなかった町丁を部分マッチング方法で再検索
    例: 元三町1丁目 -> 元三町のデータを使用
    """
    if not panel_towns_file or not Path(panel_towns_file).exists():
        return pd.DataFrame()
    
    try:
        panel_df = pd.read_csv(panel_towns_file)
        if 'town' not in panel_df.columns:
            return pd.DataFrame()
        
        panel_towns = set(panel_df['town'].dropna().astype(str))
        unmatched_towns = panel_towns - matched_towns
        
        if not unmatched_towns:
            logger.info("マッチしなかった町丁はありません")
            return pd.DataFrame()
        
        logger.info(f"マッチしなかった町丁を部分マッチング方法で再検索: {len(unmatched_towns)}件")
        
        # 座標系をWGS84 (EPSG:4326) に変換
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # 重心を計算
        centroids = gdf.geometry.centroid
        gdf['lon'] = centroids.x
        gdf['lat'] = centroids.y
        
        additional_matches = []
        
        # 部分マッチングのルールを定義
        partial_matching_rules = {
            '元三町': ['元三町1丁目', '元三町2丁目', '元三町3丁目', '元三町4丁目', '元三町5丁目'],
            '八島町': ['八島1丁目', '八島2丁目'],
            '野口町': ['野口1丁目', '野口2丁目', '野口3丁目', '野口4丁目'],
            '龍田町弓削': ['龍田弓削1丁目', '龍田弓削2丁目'],
            '三郎': ['三郎1丁目', '三郎2丁目'],
            '二本木': ['二本木1丁目', '二本木2丁目', '二本木3丁目', '二本木4丁目', '二本木5丁目'],
            '八幡': ['八幡1丁目', '八幡2丁目', '八幡3丁目', '八幡4丁目', '八幡5丁目', '八幡6丁目', '八幡7丁目', '八幡8丁目', '八幡9丁目', '八幡10丁目', '八幡11丁目'],
            '八反田': ['八反田1丁目', '八反田2丁目', '八反田3丁目'],
            '八景水谷': ['八景水谷1丁目', '八景水谷2丁目', '八景水谷3丁目', '八景水谷4丁目'],
            '十禅寺': ['十禅寺1丁目', '十禅寺2丁目', '十禅寺3丁目'],
            '島崎': ['島崎1丁目']
        }
        
        # 各ルールに対して部分マッチングを実行
        for base_town, target_towns in partial_matching_rules.items():
            # マッチしなかった町丁の中で、このルールに該当するものを抽出
            matching_targets = [town for town in target_towns if town in unmatched_towns]
            
            if matching_targets:
                # GMLファイルでベースとなる町丁を検索
                base_matches = gdf[gdf[detected_cols['town']].str.contains(base_town, na=False)]
                
                if len(base_matches) > 0:
                    # 最初に見つかったベース町丁を使用
                    base_match = base_matches.iloc[0]
                    
                    # 各ターゲット町丁に対してベース町丁のデータを使用
                    for target_town in matching_targets:
                        match_data = {
                            'town_id': f"{base_match[detected_cols['town']]}_{target_town}",  # 一意のIDを生成
                            'lon': base_match['lon'],
                            'lat': base_match['lat'],
                            'town': target_town  # 元の町丁名を使用
                        }
                        additional_matches.append(match_data)
                        logger.info(f"部分マッチング成功: {target_town} -> {base_match[detected_cols['town']]}")
        
        if additional_matches:
            result_df = pd.DataFrame(additional_matches)
            
            # 重複を除去
            result_df = result_df.drop_duplicates(subset=['town_id'])
            
            # 欠損値を除去
            result_df = result_df.dropna(subset=['lon', 'lat'])
            
            logger.info(f"部分マッチング方法で追加マッチ: {len(result_df)}件")
            
            # 追加マッチした町丁名をログ出力
            additional_matched_towns = set(result_df['town'])
            logger.info(f"追加マッチした町丁名: {list(additional_matched_towns)}")
            
            return result_df
        else:
            logger.info("部分マッチング方法でもマッチしませんでした")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"部分マッチング方法での再検索エラー: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='町丁の重心座標を作成')
    parser.add_argument('--geo', required=True, help='GMLファイルのパス')
    parser.add_argument('--pref', help='県名でフィルタ')
    parser.add_argument('--city', help='市名でフィルタ')
    parser.add_argument('--ward', help='区名でフィルタ')
    parser.add_argument('--town-list', help='町丁目リストCSVファイル（列名: town）')
    parser.add_argument('--master', help='マスターファイルCSV（列名: town_id, town）')
    parser.add_argument('--panel-towns', help='panel_raw.csvファイル（列名: town）でフィルタ')
    parser.add_argument('--output', default='subject3/data/processed/town_centroids.csv', 
                       help='出力ファイルパス')
    
    args = parser.parse_args()
    
    # GMLファイルの存在確認
    geo_path = Path(args.geo)
    if not geo_path.exists():
        logger.error(f"GMLファイルが見つかりません: {args.geo}")
        sys.exit(1)
    
    try:
        # GMLファイルを読み込み
        logger.info(f"GMLファイルを読み込み中: {args.geo}")
        gdf = gpd.read_file(args.geo)
        logger.info(f"読み込み完了: {len(gdf)}件")
        
        # 列名を自動推定
        detected_cols = detect_column_names(gdf)
        logger.info(f"検出された列名: {detected_cols}")
        
        # フィルタリング
        filtered_gdf = filter_by_location(gdf, detected_cols, args.pref, args.city, args.ward)
        
        if args.town_list:
            filtered_gdf = filter_by_town_list(filtered_gdf, detected_cols, args.town_list)
        
        # town_idを付与
        result_gdf = add_town_id(filtered_gdf, detected_cols, args.master)
        
        # 重心座標を抽出
        result_df = extract_centroids(result_gdf, detected_cols)
        
        # panel_raw.csvでフィルタリング
        if args.panel_towns:
            result_df = filter_by_panel_towns(result_df, args.panel_towns)
            
            # マッチしなかった町丁を別の正規化方法で再検索
            matched_towns = set(result_df['town']) if 'town' in result_df.columns else set()
            additional_matches = search_unmatched_towns_with_alternative_normalization(
                filtered_gdf, detected_cols, args.panel_towns, matched_towns
            )
            
            # 追加マッチした町丁を結果に追加
            if len(additional_matches) > 0:
                result_df = pd.concat([result_df, additional_matches], ignore_index=True)
                matched_towns = set(result_df['town'])  # 更新されたマッチした町丁セット
                logger.info(f"別の正規化方法後のマッチ数: {len(result_df)}件")
            
            # 柔軟なマッチング方法で再検索
            flexible_matches = search_unmatched_towns_with_flexible_matching(
                filtered_gdf, detected_cols, args.panel_towns, matched_towns
            )
            
            # 柔軟なマッチングで追加マッチした町丁を結果に追加
            if len(flexible_matches) > 0:
                result_df = pd.concat([result_df, flexible_matches], ignore_index=True)
                matched_towns = set(result_df['town'])  # 更新されたマッチした町丁セット
                logger.info(f"柔軟マッチング後のマッチ数: {len(result_df)}件")
            
            # 部分マッチング方法で再検索
            partial_matches = search_unmatched_towns_with_partial_matching(
                filtered_gdf, detected_cols, args.panel_towns, matched_towns
            )
            
            # 部分マッチングで追加マッチした町丁を結果に追加
            if len(partial_matches) > 0:
                result_df = pd.concat([result_df, partial_matches], ignore_index=True)
                logger.info(f"最終的なマッチ数: {len(result_df)}件")
        
        # 出力ディレクトリを作成
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSVファイルとして保存（UTF-8-SIG）
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"出力完了: {output_path}")
        logger.info(f"出力件数: {len(result_df)}件")
        
        # 先頭5行を表示
        print("\n=== 出力ファイルの先頭5行 ===")
        print(result_df.head().to_string(index=False))
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
