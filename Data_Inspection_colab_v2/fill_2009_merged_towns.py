#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2009年合併町の2009年以前のデータをNaNで埋めるスクリプト
城南町系、富合町系、植木町系の2009年以前のデータをNaNに設定
"""

import pandas as pd
import re
from pathlib import Path
import numpy as np

def get_merge_year(era_year):
    """日本年号を西暦に変換"""
    # ファイル名から年月を正しく抽出
    if era_year.startswith('H'):
        # H10-04 → H10 → 1988 + 10 = 1998
        year_match = re.search(r'H(\d+)', era_year)
        if year_match:
            year = int(year_match.group(1))
            return 1988 + year
    elif era_year.startswith('R'):
        # R02-04 → R02 → 2018 + 2 = 2020
        year_match = re.search(r'R(\d+)', era_year)
        if year_match:
            year = int(year_match.group(1))
            return 2018 + year
    return None

def identify_2009_merged_towns():
    """2009年に合併した町丁を特定"""
    merged_towns = {
        # 城南町系（旧下益城郡城南町）
        '城南町系': [
            '城南町阿高', '城南町隈庄', '城南町舞原', '城南町上安永', '城南町下安永',
            '城南町七瀬', '城南町宮地', '城南町吉野', '城南町赤峰', '城南町坂野',
            '城南町小岩', '城南町出田', '城南町藤山', '城南町野口', '城南町塚原',
            '城南町福藤', '城南町島田', '城南町永', '城南町六嘉', '城南町鰐瀬',
            '城南町手水', '城南町保田', '城南町馬渡', '城南町豊原', '城南町上',
            '城南町下', '城南町中', '城南町西', '城南町東', '城南町南', '城南町北'
        ],
        
        # 富合町系（旧下益城郡富合町）
        '富合町系': [
            '富合町志々水', '富合町平原', '富合町御船手', '富合町砂取', '富合町上木葉',
            '富合町下木葉', '富合町木原', '富合町菰江', '富合町西田尻', '富合町釈迦堂',
            '富合町小岩瀬', '富合町大野', '富合町小野', '富合町田尻', '富合町上田尻',
            '富合町下田尻', '富合町田尻中', '富合町田尻西', '富合町田尻東', '富合町田尻南',
            '富合町田尻北', '富合町田尻上', '富合町田尻下', '富合町田尻中', '富合町田尻西',
            '富合町田尻東', '富合町田尻南', '富合町田尻北', '富合町田尻上', '富合町田尻下'
        ],
        
        # 植木町系（旧鹿本郡植木町）
        '植木町系': [
            '植木町一木', '植木町滴水', '植木町平原', '植木町古閑', '植木町上古閑',
            '植木町後古閑', '植木町味取', '植木町大井', '植木町大和', '植木町宮原',
            '植木町富応', '植木町小野', '植木町山本', '植木町岩野', '植木町平井',
            '植木町平野', '植木町広住', '植木町投刀塚', '植木町有泉', '植木町木留',
            '植木町植木', '植木町正清', '植木町清水', '植木町田底', '植木町石川',
            '植木町米塚', '植木町舞尾', '植木町舟島', '植木町色出', '植木町荻迫',
            '植木町豊岡', '植木町豊田', '植木町轟', '植木町辺田野', '植木町那知',
            '植木町鈴麦', '植木町鐙田', '植木町鞍掛', '植木町伊知坊', '植木町内',
            '植木町円台寺', '植木町今藤', '植木町亀甲'
        ]
    }
    
    # フラットなリストに変換
    all_merged_towns = []
    for category, towns in merged_towns.items():
        all_merged_towns.extend(towns)
    
    return all_merged_towns

def fill_2009_merged_towns_with_nan():
    """2009年合併町の2009年以前のデータをNaNで埋める"""
    print("=== 2009年合併町の2009年以前データをNaNで埋める処理開始 ===")
    
    # ディレクトリ設定
    input_dir = Path("Preprocessed_Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # 2009年合併町を特定
    merged_towns = identify_2009_merged_towns()
    print(f"✓ 2009年合併町を特定: {len(merged_towns)}町丁")
    
    # CSVファイルを処理
    csv_files = list(input_dir.glob("*_consistent.csv"))
    csv_files.sort()
    
    print(f"✓ 処理対象ファイル数: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\n処理中: {csv_file.name}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # ファイル名から年月を抽出
            filename = csv_file.stem
            if '_consistent' in filename:
                filename = filename.replace('_consistent', '')
            
            # 年月を西暦に変換
            merge_year = get_merge_year(filename)
            if merge_year is None:
                print(f"⚠️  年月を特定できません: {filename}")
                continue
            
            print(f"  {filename} → 西暦{merge_year}年")
            
            # 2009年以前の場合、合併町のデータをNaNで埋める
            if merge_year < 2009:
                print(f"  → 2009年以前のため、{len(merged_towns)}町丁をNaNで埋めます")
                
                # 合併町の行を特定してNaNで埋める
                for town in merged_towns:
                    # 町丁名の列（通常は最初の列）を検索
                    town_rows = df[df.iloc[:, 0] == town]
                    if not town_rows.empty:
                        # 人口データの列（2列目以降）をNaNで埋める
                        for col_idx in range(1, len(df.columns)):
                            df.iloc[town_rows.index, col_idx] = np.nan
                
                # 出力ファイル名
                output_filename = f"{filename}_nan_filled.csv"
                output_path = output_dir / output_filename
                
                # 保存
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"  ✓ 保存完了: {output_filename}")
                
            else:
                print(f"  → 2009年以降のため、変更なし")
                # 2009年以降はそのまま保存
                output_filename = f"{filename}_nan_filled.csv"
                output_path = output_dir / output_filename
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"  ✓ 保存完了: {output_filename}")
                
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print(f"\n=== 処理完了 ===")
    print(f"✓ 全ファイルの処理が完了しました")
    print(f"✓ 出力先: {output_dir}")

def main():
    """メイン処理"""
    fill_2009_merged_towns_with_nan()

if __name__ == "__main__":
    main()
