#!/usr/bin/env python3
"""
町丁の一致をチェックするスクリプト

town_centroids.csvの町丁がpanel_raw.csvの全町丁と一致するかを調べます。
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_town_centroids(centroids_file):
    """
    town_centroids.csvを読み込み、町丁名のセットを返す
    """
    try:
        df = pd.read_csv(centroids_file)
        if 'town' not in df.columns:
            logger.error(f"town_centroids.csvに'town'列が見つかりません")
            return set()
        
        towns = set(df['town'].dropna().astype(str))
        logger.info(f"town_centroids.csvから{len(towns)}件の町丁を読み込みました")
        return towns
    except Exception as e:
        logger.error(f"town_centroids.csvの読み込みエラー: {e}")
        return set()


def load_panel_towns(panel_file):
    """
    panel_raw.csvを読み込み、町丁名のセットを返す
    """
    try:
        df = pd.read_csv(panel_file)
        if 'town' not in df.columns:
            logger.error(f"panel_raw.csvに'town'列が見つかりません")
            return set()
        
        towns = set(df['town'].dropna().astype(str))
        logger.info(f"panel_raw.csvから{len(towns)}件の町丁を読み込みました")
        return towns
    except Exception as e:
        logger.error(f"panel_raw.csvの読み込みエラー: {e}")
        return set()


def compare_towns(centroids_towns, panel_towns):
    """
    町丁の一致をチェックし、結果を表示
    """
    logger.info("=== 町丁一致チェック結果 ===")
    
    # 完全一致チェック
    if centroids_towns == panel_towns:
        logger.info("✅ 完全一致: town_centroids.csvとpanel_raw.csvの町丁が完全に一致しています")
        return True
    
    # 差分の詳細分析
    only_in_centroids = centroids_towns - panel_towns
    only_in_panel = panel_towns - centroids_towns
    common_towns = centroids_towns & panel_towns
    
    logger.info(f"共通の町丁数: {len(common_towns)}")
    logger.info(f"centroidsのみの町丁数: {len(only_in_centroids)}")
    logger.info(f"panelのみの町丁数: {len(only_in_panel)}")
    
    # マッチ率の計算
    if len(panel_towns) > 0:
        match_rate = len(common_towns) / len(panel_towns) * 100
        logger.info(f"マッチ率: {match_rate:.2f}%")
    
    # 詳細な差分を表示
    if only_in_centroids:
        logger.warning(f"town_centroids.csvのみに存在する町丁（最初の20件）:")
        for i, town in enumerate(sorted(only_in_centroids)[:20]):
            logger.warning(f"  {i+1:2d}. {town}")
        if len(only_in_centroids) > 20:
            logger.warning(f"  ... 他{len(only_in_centroids) - 20}件")
    
    if only_in_panel:
        logger.warning(f"panel_raw.csvのみに存在する町丁（最初の20件）:")
        for i, town in enumerate(sorted(only_in_panel)[:20]):
            logger.warning(f"  {i+1:2d}. {town}")
        if len(only_in_panel) > 20:
            logger.warning(f"  ... 他{len(only_in_panel) - 20}件")
    
    # 類似町丁名の検索（部分マッチ）
    if only_in_panel:
        logger.info("=== 類似町丁名の検索 ===")
        similar_matches = find_similar_towns(only_in_panel, centroids_towns)
        if similar_matches:
            logger.info("類似する町丁名が見つかりました:")
            for panel_town, similar_centroid in similar_matches.items():
                logger.info(f"  panel: {panel_town} -> centroid: {similar_centroid}")
    
    return False


def find_similar_towns(panel_towns, centroids_towns, threshold=0.7):
    """
    類似する町丁名を検索
    """
    similar_matches = {}
    
    for panel_town in panel_towns:
        best_match = None
        best_similarity = 0
        
        for centroid_town in centroids_towns:
            # 簡単な類似度計算（共通文字数 / 最大文字数）
            common_chars = len(set(panel_town) & set(centroid_town))
            max_chars = max(len(panel_town), len(centroid_town))
            similarity = common_chars / max_chars if max_chars > 0 else 0
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = centroid_town
        
        if best_match:
            similar_matches[panel_town] = best_match
    
    return similar_matches


def save_comparison_report(centroids_towns, panel_towns, output_file):
    """
    比較結果をCSVファイルに保存
    """
    try:
        # 差分の詳細データを作成
        only_in_centroids = centroids_towns - panel_towns
        only_in_panel = panel_towns - centroids_towns
        common_towns = centroids_towns & panel_towns
        
        # 結果をDataFrameにまとめる
        results = []
        
        # 共通の町丁
        for town in sorted(common_towns):
            results.append({
                'town': town,
                'status': 'common',
                'description': '両方に存在'
            })
        
        # centroidsのみの町丁
        for town in sorted(only_in_centroids):
            results.append({
                'town': town,
                'status': 'centroids_only',
                'description': 'town_centroids.csvのみ'
            })
        
        # panelのみの町丁
        for town in sorted(only_in_panel):
            results.append({
                'town': town,
                'status': 'panel_only',
                'description': 'panel_raw.csvのみ'
            })
        
        # CSVファイルに保存
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"比較結果を保存しました: {output_file}")
        
        # サマリー情報も保存
        summary_file = str(output_file).replace('.csv', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("町丁一致チェック結果サマリー\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"共通の町丁数: {len(common_towns)}\n")
            f.write(f"centroidsのみの町丁数: {len(only_in_centroids)}\n")
            f.write(f"panelのみの町丁数: {len(only_in_panel)}\n")
            f.write(f"合計町丁数 (centroids): {len(centroids_towns)}\n")
            f.write(f"合計町丁数 (panel): {len(panel_towns)}\n")
            
            if len(panel_towns) > 0:
                match_rate = len(common_towns) / len(panel_towns) * 100
                f.write(f"マッチ率: {match_rate:.2f}%\n")
            
            f.write(f"\n完全一致: {'Yes' if centroids_towns == panel_towns else 'No'}\n")
        
        logger.info(f"サマリー情報を保存しました: {summary_file}")
        
    except Exception as e:
        logger.error(f"比較結果の保存エラー: {e}")


def main():
    parser = argparse.ArgumentParser(description='町丁の一致をチェック')
    parser.add_argument('--centroids', 
                       default='subject3/data/processed/town_centroids.csv',
                       help='town_centroids.csvファイルのパス')
    parser.add_argument('--panel', 
                       default='subject3/data/processed/panel_raw.csv',
                       help='panel_raw.csvファイルのパス')
    parser.add_argument('--output', 
                       default='subject3/data/processed/town_comparison_report.csv',
                       help='比較結果の出力ファイルパス')
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    centroids_path = Path(args.centroids)
    panel_path = Path(args.panel)
    
    if not centroids_path.exists():
        logger.error(f"town_centroids.csvが見つかりません: {args.centroids}")
        sys.exit(1)
    
    if not panel_path.exists():
        logger.error(f"panel_raw.csvが見つかりません: {args.panel}")
        sys.exit(1)
    
    try:
        # 町丁データを読み込み
        logger.info("町丁データを読み込み中...")
        centroids_towns = load_town_centroids(centroids_path)
        panel_towns = load_panel_towns(panel_path)
        
        if not centroids_towns or not panel_towns:
            logger.error("町丁データの読み込みに失敗しました")
            sys.exit(1)
        
        # 町丁の一致をチェック
        is_consistent = compare_towns(centroids_towns, panel_towns)
        
        # 結果をファイルに保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_comparison_report(centroids_towns, panel_towns, output_path)
        
        # 終了コードを設定
        if is_consistent:
            logger.info("✅ 町丁の一致チェック完了: 完全一致")
            sys.exit(0)
        else:
            logger.warning("⚠️ 町丁の一致チェック完了: 不一致あり")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
