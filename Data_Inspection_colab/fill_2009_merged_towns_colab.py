#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2009年合併町NaN埋めスクリプト（Google Colab版）
2009年に合併した町丁の2009年以前のデータをNaNで埋める
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def get_merge_year(era_year):
    """日本年号を西暦に変換"""
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
    merged_towns = [
        # 城南町系（旧下益城郡城南町）
        '城南町阿高', '城南町隈庄', '城南町舞原', '城南町上', '城南町下',
        # 富合町系（旧下益城郡富合町）
        '富合町志々水', '富合町平原', '富合町御船手', '富合町砂原', '富合町清藤',
        # 植木町系（旧鹿本郡植木町）
        '植木町滴水', '植木町一木', '植木町平原', '植木町田底', '植木町豊岡'
    ]
    return merged_towns

def fill_2009_merged_towns():
    """2009年合併町の2009年以前のデータをNaNで埋める"""
    print("=== ステップ4: 2009年合併町NaN埋め処理開始 ===")
    
    input_dir = Path("Preprocessed_Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # 2009年合併町を特定
    merged_towns = identify_2009_merged_towns()
    print(f"✓ 2009年合併町数: {len(merged_towns)}")
    
    for i, town in enumerate(merged_towns):
        print(f"  {i+1}. {town}")
    
    # 統合済みファイルを取得
    csv_files = list(input_dir.glob("*_consistent.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ 統合済みファイルが見つかりません")
        print("先にmerge_town_data_colab.pyを実行してください")
        return False
    
    print(f"\n✓ 処理対象ファイル数: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\n処理中: {csv_file.name}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # ファイル名から年月を抽出
            era_year = csv_file.stem.replace('_consistent', '')  # 例: H10-04
            merge_year = get_merge_year(era_year)
            
            if merge_year is None:
                print(f"  ⚠️  年月を特定できませんでした: {era_year}")
                continue
            
            print(f"  → 対象年度: {era_year} (西暦{merge_year}年)")
            
            # 2009年以前のファイルの場合、合併町の人口をNaNで埋める
            if merge_year < 2009:
                changes_made = 0
                
                # 町丁名の列（最初の列）を取得
                town_col = df.iloc[:, 0]
                
                for idx, town_name in enumerate(town_col):
                    if pd.isna(town_name):
                        continue
                    
                    town_name_str = str(town_name).strip()
                    
                    # 2009年合併町かチェック
                    if town_name_str in merged_towns:
                        # 人口列をNaNで埋める
                        for col_idx in range(1, len(df.columns)):
                            if pd.notna(df.iloc[idx, col_idx]) and str(df.iloc[idx, col_idx]).strip() != '':
                                df.iloc[idx, col_idx] = np.nan
                        changes_made += 1
                        print(f"  → NaN埋め: {town_name_str}")
                
                if changes_made > 0:
                    print(f"  → 変更件数: {changes_made}件")
                else:
                    print(f"  → 変更なし")
            else:
                print(f"  → 2009年以降のため処理不要")
            
            # 出力ファイル名
            output_filename = f"{csv_file.stem.replace('_consistent', '')}_nan_filled.csv"
            output_path = output_dir / output_filename
            
            # 保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存完了: {output_filename}")
            
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print("\n✓ 2009年合併町NaN埋め処理完了")
    return True

def main():
    """メイン処理"""
    print("=== 2009年合併町NaN埋め開始（Google Colab版） ===")
    
    # 2009年合併町NaN埋め
    if not fill_2009_merged_towns():
        return
    
    print("\n=== ステップ4完了 ===")
    print("✓ 2009年合併町NaN埋めが完了しました")
    print("✓ 次のステップ: 区制町丁の統一処理")
    print("\n次のスクリプトを実行してください:")
    print("python unify_district_towns_colab.py")

if __name__ == "__main__":
    main()
