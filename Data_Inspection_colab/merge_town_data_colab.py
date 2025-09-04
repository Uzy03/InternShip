#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
町丁データ統合スクリプト（Google Colab版）
町丁名の正規化、2009年合併町の処理、区制町丁の統一を実行
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def normalize_town_name(town_name):
    """町丁名を正規化（区名除去、空白除去）"""
    if pd.isna(town_name):
        return town_name
    
    # 区名を除去: （中央区）、（東区）、（西区）、（南区）、（北区）
    normalized = re.sub(r'（[東南西北中央]区）', '', str(town_name))
    
    # 空白を除去
    normalized = normalized.strip()
    
    return normalized

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

def is_valid_town_name(town_name):
    """有効な町丁名かどうかを判定"""
    if pd.isna(town_name):
        return False
    
    town_name = str(town_name).strip()
    
    # 除外すべきパターン
    exclude_patterns = [
        # 年齢区分
        r'^\d+[〜～]\d+歳$',
        r'^\d+歳以上$',
        r'^\d+歳$',
        r'^計$',
        r'^男$',
        r'^女$',
        r'^備考$',
        
        # ヘッダー・タイトル
        r'人口統計表',
        r'町丁別の年齢５歳刻み一覧表',
        r'年齢区分',
        r'町丁別',
        r'一覧表',
        r'現在',
        r'人',
        r'口',
        
        # 日付
        r'平成\d+年\d+月\d+日現在',
        r'令和\d+年\d+月\d+日現在',
        r'^\d+年\d+月\d+日現在$',
        
        # その他の不要なデータ
        r'^Column\d+$',
        r'^Unnamed:\d+$',
        r'^\s*$',  # 空白のみ
        
        # 数字のみ
        r'^\d+$',
        
        # 特殊文字のみ
        r'^[^\w\s\u4e00-\u9fff]+$'
    ]
    
    for pattern in exclude_patterns:
        if re.match(pattern, town_name):
            return False
    
    # 有効な町丁名の条件
    if (len(town_name) == 0 or 
        len(town_name) > 50 or
        re.match(r'^\d+$', town_name) or
        not re.search(r'[\u4e00-\u9fff]', town_name)):
        return False
    
    return True

def extract_valid_towns(df):
    """データフレームから有効な町丁名を抽出"""
    valid_towns = []
    
    for i, row in df.iterrows():
        town_name = str(row.iloc[0]) if len(row) > 0 else ""
        if is_valid_town_name(town_name):
            valid_towns.append(town_name.strip())
    
    return valid_towns

def merge_town_data():
    """町丁データの統合と前処理を実行"""
    print("=== ステップ3: 町丁データ統合と前処理開始 ===")
    
    csv_dir = Path("Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # CSVファイルを取得
    csv_files = list(csv_dir.glob("*.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ CSVファイルが見つかりません")
        return False
    
    print(f"✓ 処理対象ファイル数: {len(csv_files)}")
    
    # 2009年合併町を特定
    merged_towns = identify_2009_merged_towns()
    print(f"✓ 2009年合併町数: {len(merged_towns)}")
    
    # 全ファイルの有効な町丁名を収集
    all_valid_towns = set()
    file_towns = {}
    
    for csv_file in csv_files:
        print(f"読み込み中: {csv_file.name}")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        valid_towns = extract_valid_towns(df)
        file_towns[csv_file.name] = valid_towns
        all_valid_towns.update(valid_towns)
    
    print(f"✓ 全有効町丁数: {len(all_valid_towns)}")
    
    # 各ファイルを処理
    for csv_file in csv_files:
        print(f"\n処理中: {csv_file.name}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # ファイル名から年月を抽出
            era_year = csv_file.stem  # 例: H10-04
            merge_year = get_merge_year(era_year)
            
            # 町丁名の列（最初の列）を取得
            town_col = df.iloc[:, 0]
            
            # 町丁名の正規化と2009年合併町の処理
            changes_made = 0
            
            for idx, town_name in enumerate(town_col):
                if pd.isna(town_name):
                    continue
                
                original_name = str(town_name).strip()
                
                # 有効な町丁名かチェック
                if not is_valid_town_name(original_name):
                    continue
                
                # 町丁名を正規化
                normalized_name = normalize_town_name(original_name)
                
                # 2009年合併町の処理
                if normalized_name in merged_towns and merge_year and merge_year < 2009:
                    # 2009年以前の合併町は人口を0に設定
                    for col_idx in range(1, len(df.columns)):
                        if pd.notna(df.iloc[idx, col_idx]) and str(df.iloc[idx, col_idx]).strip() != '':
                            df.iloc[idx, col_idx] = 0
                    changes_made += 1
                    print(f"  → 2009年合併町処理: {normalized_name} (人口を0に設定)")
                
                # 正規化された町丁名を設定
                if original_name != normalized_name:
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 正規化: {original_name} → {normalized_name}")
            
            if changes_made > 0:
                print(f"  → 変更件数: {changes_made}件")
            
            # 出力ファイル名
            output_filename = f"{csv_file.stem}_consistent.csv"
            output_path = output_dir / output_filename
            
            # 保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存完了: {output_filename}")
            
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print("\n✓ 町丁データ統合処理完了")
    return True

def main():
    """メイン処理"""
    print("=== 町丁データ統合開始（Google Colab版） ===")
    
    # 町丁データ統合
    if not merge_town_data():
        return
    
    print("\n=== ステップ3完了 ===")
    print("✓ 町丁データ統合が完了しました")
    print("✓ 次のステップ: 2009年合併町のNaN埋め処理")
    print("\n次のスクリプトを実行してください:")
    print("python fill_2009_merged_towns_colab.py")

if __name__ == "__main__":
    main()
