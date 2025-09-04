#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
町丁データマージツール
マージルールに基づいて町丁データを統合し、100%の一貫性を持つデータを作成
"""

import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from collections import defaultdict

def normalize_town_name(town_name):
    """町丁名を正規化"""
    if pd.isna(town_name):
        return town_name
    
    # 区名を削除
    town_name = re.sub(r'（[東西南北中央区]+）', '', str(town_name))
    
    # 空白を削除
    town_name = re.sub(r'[\s　\n\r\t]+', '', str(town_name))
    
    # 前後の空白を削除
    town_name = town_name.strip()
    
    return town_name

def identify_merged_towns():
    """合併町（2009年編入）を特定"""
    merged_towns = {
        '城南町系': [
            '城南町阿高', '城南町隈庄', '城南町舞原', '城南町藤山', '城南町島田',
            '城南町六田', '城南町宮地', '城南町築地', '城南町碇', '城南町色出',
            '城南町舟島', '城南町平野', '城南町坂野', '城南町富応', '城南町上古閑',
            '城南町古閑', '城南町陳内', '城南町小岩瀬', '城南町阿高', '城南町平井',
            '城南町大町', '城南町豊岡', '城南町山本', '城南町今吉野', '城南町千町',
            '城南町沈目', '城南町赤見', '城南町大井', '城南町杉島', '城南町御船手',
            '城南町西田尻', '城南町南田尻', '城南町後古閑', '城南町出水', '城南町小野',
            '城南町正清', '城南町碇', '城南町色出', '城南町石川', '城南町米塚',
            '城南町東阿高', '城南町辺田野', '城南町舞尾', '城南町円台寺', '城南町永',
            '城南町塚原', '城南町鐙田', '城南町清藤', '城南町釈迦堂', '城南町鈴麦',
            '城南町荻迫', '城南町田底', '城南町鰐瀬', '城南町伊知坊', '城南町榎津',
            '城南町味取', '城南町宮原', '城南町平原', '城南町下宮地', '城南町丹生宮',
            '城南町今吉野', '城南町六田', '城南町出水', '城南町千町', '城南町坂野',
            '城南町塚原', '城南町宮地', '城南町島田', '城南町東阿高', '城南町永',
            '城南町沈目', '城南町碇', '城南町築地', '城南町舞原', '城南町藤山',
            '城南町赤見', '城南町阿高', '城南町陳内', '城南町隈庄', '城南町高',
            '城南町鰐瀬'
        ],
        '富合町系': [
            '富合町志々水', '富合町平原', '富合町御船手', '富合町新', '富合町木原',
            '富合町杉島', '富合町榎津', '富合町清藤', '富合町田尻', '富合町硴江',
            '富合町莎崎', '富合町菰江', '富合町西田尻', '富合町釈迦堂', '富合町上杉',
            '富合町南田尻', '富合町古閑', '富合町国町', '富合町大町', '富合町小岩瀬',
            '富合町廻江'
        ],
        '植木町系': [
            '植木町滴水', '植木町一木', '植木町平原', '植木町亀甲', '植木町岩野',
            '植木町今藤', '植木町伊知坊', '植木町内', '植木町円台寺', '植木町古閑',
            '植木町味取', '植木町大井', '植木町大和', '植木町宮原', '植木町富応',
            '植木町小野', '植木町山本', '植木町平井', '植木町平野', '植木町広住',
            '植木町後古閑', '植木町投刀塚', '植木町有泉', '植木町木留', '植木町植木',
            '植木町正清', '植木町清水', '植木町田底', '植木町石川', '植木町米塚',
            '植木町舞尾', '植木町舟島', '植木町色出', '植木町荻迫', '植木町豊岡',
            '植木町豊田', '植木町轟', '植木町辺田野', '植木町那知', '植木町鈴麦',
            '植木町鐙田', '植木町鞍掛', '植木町上古閑'
        ]
    }
    
    # フラットなリストに変換
    all_merged_towns = []
    for category, towns in merged_towns.items():
        all_merged_towns.extend(towns)
    
    return all_merged_towns

def get_merge_year(era_year):
    """年号を西暦に変換"""
    if era_year.startswith('H'):
        year = int(era_year[1:3])
        return 1988 + year  # 平成元年は1989年
    elif era_year.startswith('R'):
        year = int(era_year[1:3])
        return 2018 + year  # 令和元年は2019年
    else:
        return None

def merge_town_data():
    """町丁データをマージして一貫性のあるデータを作成"""
    try:
        print("=== 町丁データマージ開始 ===")
        
        # 入力・出力ディレクトリ
        input_dir = Path("Data_csv")
        output_dir = Path("Preprocessed_Data_csv")
        
        if not input_dir.exists():
            print(f"✗ 入力ディレクトリが見つかりません: {input_dir}")
            return False
        
        # 出力ディレクトリを作成
        output_dir.mkdir(exist_ok=True)
        print(f"出力先: {output_dir.absolute()}")
        
        # CSVファイルを検索
        csv_files = list(input_dir.glob("*.csv"))
        if not csv_files:
            print("CSVファイルが見つかりませんでした")
            return False
        
        # 合併町を特定
        merged_towns = identify_merged_towns()
        print(f"合併町数: {len(merged_towns)}")
        
        # 各ファイルを処理
        processed_files = []
        
        for csv_file in sorted(csv_files):
            print(f"\n処理中: {csv_file.name}")
            
            # ファイル名から年月度を抽出
            year_month = csv_file.stem
            merge_year = get_merge_year(year_month)
            
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 町丁名を正規化
            df['町丁名_正規化'] = df.iloc[:, 0].apply(normalize_town_name)
            
            # 合併町の処理
            if merge_year and merge_year < 2009:
                # 2009年以前の場合、合併町はNaNまたは0で補完
                for town in merged_towns:
                    if town in df['町丁名_正規化'].values:
                        # 既存の行を更新
                        mask = df['町丁名_正規化'] == town
                        if mask.any():
                            # 人口データ列を0で埋める
                            for col in df.columns[1:]:
                                if '人口' in col or '計' in col or '男' in col or '女' in col:
                                    df.loc[mask, col] = 0
            
            # 正規化された町丁名で列名を変更
            df = df.rename(columns={df.columns[0]: '町丁名'})
            df['町丁名'] = df['町丁名_正規化']
            df = df.drop('町丁名_正規化', axis=1)
            
            # 重複する町丁名を統合（区制導入による重複を解決）
            df = df.groupby('町丁名').agg({
                '町丁名': 'first',
                **{col: 'sum' if '人口' in col or '計' in col or '男' in col or '女' in col else 'first' 
                   for col in df.columns[1:]}
            }).reset_index(drop=True)
            
            # 出力ファイル名
            output_file = output_dir / f"{year_month}_merged.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✓ 処理完了: {output_file.name} ({len(df)}町丁)")
            processed_files.append((year_month, output_file, len(df)))
        
        # マージ後の一貫性チェック
        print(f"\n=== マージ後の一貫性チェック ===")
        
        # 全ファイルの町丁名を収集
        all_towns = set()
        year_town_data = {}
        
        for year_month, file_path, town_count in processed_files:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            towns = set(df['町丁名'].dropna())
            all_towns.update(towns)
            year_town_data[year_month] = towns
            print(f"{year_month}: {len(towns)}町丁")
        
        # 一貫性をチェック
        consistent_towns = set.intersection(*year_town_data.values())
        inconsistent_towns = all_towns - consistent_towns
        
        print(f"\n全期間で確認された町丁数: {len(all_towns)}")
        print(f"一貫性のある町丁数: {len(consistent_towns)}")
        print(f"一貫性のない町丁数: {len(inconsistent_towns)}")
        
        if inconsistent_towns:
            print(f"\n一貫性のない町丁:")
            for town in sorted(inconsistent_towns):
                appearances = [year for year, towns in year_town_data.items() if town in towns]
                print(f"  {town}: {len(appearances)}/{len(year_town_data)}回出現")
        
        # 一貫性率を計算
        consistency_rate = len(consistent_towns) / len(all_towns) * 100
        print(f"\n一貫性率: {consistency_rate:.1f}%")
        
        if consistency_rate >= 99.5:
            print("✓ ほぼ100%の一貫性が達成されました！")
        else:
            print("⚠️  さらなる調整が必要です")
        
        # 統合された町丁リストを保存
        consistent_towns_list = sorted(list(consistent_towns))
        towns_file = output_dir / "consistent_towns_list.txt"
        
        with open(towns_file, 'w', encoding='utf-8') as f:
            f.write("=== 一貫性のある町丁一覧 ===\n")
            f.write(f"総数: {len(consistent_towns_list)}\n\n")
            for i, town in enumerate(consistent_towns_list, 1):
                f.write(f"{i:3d}. {town}\n")
        
        print(f"\n✓ 一貫性のある町丁リストを保存: {towns_file}")
        
        # サマリーファイルを作成
        summary_file = output_dir / "merge_summary.csv"
        summary_data = []
        
        for year_month, file_path, town_count in processed_files:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            consistent_count = len(set(df['町丁名'].dropna()) & consistent_towns)
            summary_data.append({
                '年度': year_month,
                '総町丁数': town_count,
                '一貫性のある町丁数': consistent_count,
                '一貫性率': consistent_count / town_count * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✓ サマリーファイルを保存: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ マージ処理に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=== 町丁データマージツール ===")
    
    print(f"\nこのツールは以下の処理を実行します:")
    print(f"1. 町丁名の正規化（区名削除、空白削除）")
    print(f"2. 合併町の処理（2009年以前は0人口で補完）")
    print(f"3. 区制導入による重複の統合")
    print(f"4. 100%の一貫性を持つデータの生成")
    
    confirm = input(f"\n処理を開始しますか？ (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        merge_town_data()
    else:
        print("処理をキャンセルしました")

if __name__ == "__main__":
    main()
