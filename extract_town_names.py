#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100%一貫性のあるCSVファイルから町丁名のリストを抽出するスクリプト
"""

import os
import pandas as pd
from pathlib import Path

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
            
            # 町丁名を抽出（「総数」や空の行を除外）
            for value in first_column:
                if pd.notna(value) and value != "総数" and value != "人口統計表" and str(value).strip():
                    # 町丁名として追加
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
    output_file = "extracted_town_names.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"抽出された町丁名の総数: {len(town_names)}\n")
        f.write("=" * 50 + "\n")
        for i, town_name in enumerate(sorted(town_names), 1):
            f.write(f"{i:3d}. {town_name}\n")
    
    print(f"\n結果を {output_file} に保存しました。")

if __name__ == "__main__":
    main()
