#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前処理状況調査スクリプト（Google Colab版）
各ステップの実行状況とファイルの状態を調査
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def investigate_file_status():
    """各ステップで生成されたファイルの状況を調査"""
    print("=== 前処理状況調査開始 ===")
    
    # 各ディレクトリの状況を確認
    directories = [
        "Data_csv",
        "Preprocessed_Data_csv"
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        print(f"\n=== {dir_name} の状況 ===")
        
        if not dir_path.exists():
            print(f"❌ ディレクトリが存在しません: {dir_name}")
            continue
        
        # ファイル一覧を取得
        files = list(dir_path.glob("*"))
        files.sort()
        
        print(f"✓ ファイル数: {len(files)}")
        
        if len(files) == 0:
            print("  ⚠️  ファイルが存在しません")
            continue
        
        # ファイルの種類別に分類
        file_types = {}
        for file in files:
            if file.is_file():
                suffix = file.suffix
                if suffix == '.csv':
                    if '_consistent' in file.stem:
                        file_types['consistent'] = file_types.get('consistent', 0) + 1
                    elif '_nan_filled' in file.stem:
                        file_types['nan_filled'] = file_types.get('nan_filled', 0) + 1
                    elif '_unified' in file.stem:
                        file_types['unified'] = file_types.get('unified', 0) + 1
                    elif '_100percent_final' in file.stem:
                        file_types['100percent_final'] = file_types.get('100percent_final', 0) + 1
                    else:
                        file_types['basic_csv'] = file_types.get('basic_csv', 0) + 1
                elif suffix == '.txt':
                    file_types['text'] = file_types.get('text', 0) + 1
                elif suffix == '.png':
                    file_types['image'] = file_types.get('image', 0) + 1
                else:
                    file_types['other'] = file_types.get('other', 0) + 1
        
        # ファイルタイプ別の状況を表示
        for file_type, count in file_types.items():
            print(f"  {file_type}: {count}件")
        
        # 最初の数件のファイル名を表示
        print(f"\n  最初の5件のファイル:")
        for i, file in enumerate(files[:5]):
            print(f"    {i+1}. {file.name}")
        
        if len(files) > 5:
            print(f"    ... 他 {len(files) - 5}件")

def investigate_csv_content():
    """CSVファイルの内容を調査"""
    print("\n=== CSVファイル内容調査 ===")
    
    csv_dir = Path("Preprocessed_Data_csv")
    if not csv_dir.exists():
        print("❌ Preprocessed_Data_csvディレクトリが存在しません")
        return
    
    # 最終ファイルを探す
    final_files = list(csv_dir.glob("*_100percent_final.csv"))
    
    if not final_files:
        print("❌ 100%一貫性達成済みファイルが見つかりません")
        print("ステップ6が正しく実行されていない可能性があります")
        return
    
    print(f"✓ 100%一貫性達成済みファイル数: {len(final_files)}")
    
    # 最初のファイルの内容を確認
    sample_file = final_files[0]
    print(f"\n=== サンプルファイル: {sample_file.name} ===")
    
    try:
        df = pd.read_csv(sample_file, encoding='utf-8-sig')
        print(f"✓ 読み込み成功")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        
        # 最初の列（町丁名）の最初の10件を表示
        print(f"\n  町丁名（最初の10件）:")
        for i, town_name in enumerate(df.iloc[:10, 0]):
            print(f"    {i+1}. {town_name}")
        
        # 区名付きの町丁があるかチェック
        district_towns = []
        for town_name in df.iloc[:, 0]:
            if pd.notna(town_name) and '（' in str(town_name) and '区）' in str(town_name):
                district_towns.append(str(town_name))
        
        if district_towns:
            print(f"\n  ⚠️  区名付き町丁が残存しています:")
            for town in district_towns[:5]:
                print(f"    - {town}")
            if len(district_towns) > 5:
                print(f"    ... 他 {len(district_towns) - 5}件")
        else:
            print(f"\n  ✅ 区名付き町丁は見つかりませんでした")
            
    except Exception as e:
        print(f"✗ ファイル読み込みエラー: {e}")

def investigate_processing_steps():
    """各ステップの実行状況を調査"""
    print("\n=== 処理ステップ実行状況調査 ===")
    
    csv_dir = Path("Preprocessed_Data_csv")
    if not csv_dir.exists():
        print("❌ Preprocessed_Data_csvディレクトリが存在しません")
        return
    
    # 各ステップのファイルを確認
    steps = [
        ("ステップ3: 町丁データ統合", "*_consistent.csv"),
        ("ステップ4: 2009年合併町NaN埋め", "*_nan_filled.csv"),
        ("ステップ5: 区制町丁統一", "*_unified.csv"),
        ("ステップ6: 100%一貫性達成", "*_100percent_final.csv")
    ]
    
    for step_name, pattern in steps:
        files = list(csv_dir.glob(pattern))
        if files:
            print(f"✅ {step_name}: {len(files)}件のファイルが存在")
        else:
            print(f"❌ {step_name}: ファイルが見つかりません")

def main():
    """メイン処理"""
    print("=== 前処理状況調査開始（Google Colab版） ===")
    
    # ファイル状況の調査
    investigate_file_status()
    
    # CSVファイル内容の調査
    investigate_csv_content()
    
    # 処理ステップの実行状況調査
    investigate_processing_steps()
    
    print("\n=== 調査完了 ===")
    print("上記の結果を確認して、問題点を特定してください")

if __name__ == "__main__":
    main()
