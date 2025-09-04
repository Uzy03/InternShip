#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100%一貫性のあるCSVファイルから町丁名のリストを抽出するスクリプト（修正版）
数値や空の値を町丁名として抽出しないように修正
"""

import os
import pandas as pd
from pathlib import Path
import re

def is_valid_town_name(value):
    """
    値が有効な町丁名かどうかを判定する
    
    Args:
        value: 判定対象の値
    
    Returns:
        bool: 有効な町丁名の場合True
    """
    if pd.isna(value):
        return False
    
    value_str = str(value).strip()
    
    # 空文字列の場合は除外
    if not value_str:
        return False
    
    # 以下の値を除外
    exclude_values = {
        "総数", "人口統計表", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "χ", "【秘匿】", "含みます", "含めています"
    }
    
    if value_str in exclude_values:
        return False
    
    # 数値のみの場合は除外
    if value_str.isdigit():
        return False
    
    # 数値で始まる場合は除外
    if re.match(r'^\d', value_str):
        return False
    
    # 特殊文字のみの場合は除外
    if re.match(r'^[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+$', value_str):
        return False
    
    return True

def extract_town_names_from_csv_files(directory_path):
    """
    指定されたディレクトリ内のCSVファイルから町丁名を抽出する
    
    Args:
        directory_path (str): CSVファイルが格納されているディレクトリのパス
    
    Returns:
        set: 抽出された町丁名のセット
    """
    town_names = set()
    
    # ディレクトリ内のCSVファイルを取得
    csv_files = list(Path(directory_path).glob("*.csv"))
    
    if not csv_files:
        print(f"指定されたディレクトリ {directory_path} にCSVファイルが見つかりませんでした。")
        return town_names
    
    print(f"処理対象のCSVファイル数: {len(csv_files)}")
    
    for csv_file in csv_files:
        try:
            print(f"処理中: {csv_file.name}")
            
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, header=None)
            
            # 1列目のデータを取得（町丁名が含まれる列）
            first_column = df.iloc[:, 0]
            
            # 町丁名を抽出
            for value in first_column:
                if is_valid_town_name(value):
                    town_names.add(str(value).strip())
                    
        except Exception as e:
            print(f"エラー: {csv_file.name} の処理中にエラーが発生しました: {e}")
            continue
    
    return town_names

def main():
    """メイン処理"""
    # 対象ディレクトリのパス
    target_directory = "Data_Inspection/100_Percent_Consistent_Data/CSV_Files"
    
    print("町丁名の抽出を開始します...")
    print(f"対象ディレクトリ: {target_directory}")
    print("-" * 50)
    
    # 町丁名を抽出
    town_names = extract_town_names_from_csv_files(target_directory)
    
    print("-" * 50)
    print(f"抽出された町丁名の総数: {len(town_names)}")
    print("\n町丁名の一覧:")
    print("=" * 50)
    
    # 町丁名をソートして表示
    for i, town_name in enumerate(sorted(town_names), 1):
        print(f"{i:3d}. {town_name}")
    
    print("=" * 50)
    
    # 結果をファイルに保存
    output_file = "extracted_town_names_fixed.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"抽出された町丁名の総数: {len(town_names)}\n")
        f.write("=" * 50 + "\n")
        for i, town_name in enumerate(sorted(town_names), 1):
            f.write(f"{i:3d}. {town_name}\n")
    
    print(f"\n結果を {output_file} に保存しました。")

if __name__ == "__main__":
    main()
