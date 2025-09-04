#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2009年合併町をNaNで埋めた後の町丁一貫性率を計算するスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def load_csv_file(file_path):
    """CSVファイルから町丁名を抽出"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        town_names = []
        
        # 最初の列から町丁名を抽出
        for i, row in df.iterrows():
            town_name = str(row.iloc[0]) if len(row) > 0 else ""
            if (town_name and 
                not any(char.isdigit() for char in town_name) and
                not any(keyword in town_name for keyword in ['年齢', '区分', '計', '男', '女', '備考', '人口統計表', '町丁別', '一覧表', '現在', '人', '口']) and
                len(town_name.strip()) > 1 and
                not town_name.strip().startswith('Column')):
                town_names.append(town_name.strip())
        
        return town_names
    except Exception as e:
        print(f"エラー: {file_path} - {e}")
        return []

def analyze_town_consistency():
    """町丁一貫性を分析"""
    print("=== 2009年合併町をNaNで埋めた後の町丁一貫性分析 ===")
    
    # ディレクトリ設定
    csv_dir = Path("../Preprocessed_Data_csv")
    
    # NaNで埋めたファイルを取得
    csv_files = list(csv_dir.glob("*_nan_filled.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ NaNで埋めたファイルが見つかりません")
        return
    
    print(f"✓ 分析対象ファイル数: {len(csv_files)}")
    
    # 各ファイルの町丁名を取得
    all_towns = set()
    file_towns = {}
    
    for csv_file in csv_files:
        print(f"読み込み中: {csv_file.name}")
        towns = load_csv_file(csv_file)
        file_towns[csv_file.name] = set(towns)
        all_towns.update(towns)
    
    print(f"\n✓ 全町丁数: {len(all_towns)}")
    
    # 各町丁の出現回数をカウント
    town_consistency = {}
    total_files = len(csv_files)
    
    for town in all_towns:
        count = sum(1 for towns in file_towns.values() if town in towns)
        town_consistency[town] = count
    
    # 一貫性のある町丁（全ファイルに出現）を特定
    consistent_towns = [town for town, count in town_consistency.items() if count == total_files]
    inconsistent_towns = [town for town, count in town_consistency.items() if count < total_files]
    
    # 一貫性率を計算
    consistency_rate = (len(consistent_towns) / len(all_towns)) * 100
    
    print(f"\n=== 分析結果 ===")
    print(f"全町丁数: {len(all_towns)}")
    print(f"一貫性のある町丁数: {len(consistent_towns)}")
    print(f"一貫性のない町丁数: {len(inconsistent_towns)}")
    print(f"町丁一貫性率: {consistency_rate:.1f}%")
    
    # 一貫性のない町丁の詳細
    if inconsistent_towns:
        print(f"\n=== 一貫性のない町丁（出現回数順） ===")
        sorted_inconsistent = sorted(inconsistent_towns, key=lambda x: town_consistency[x], reverse=True)
        
        for town in sorted_inconsistent[:20]:  # 上位20件を表示
            count = town_consistency[town]
            missing_files = [fname for fname, towns in file_towns.items() if town not in towns]
            print(f"{town}: {count}/{total_files}回出現")
            if len(missing_files) <= 5:
                print(f"  欠損年度: {', '.join(missing_files)}")
            else:
                print(f"  欠損年度: {len(missing_files)}ファイル")
    
    # 結果を保存
    output_dir = Path("../Preprocessed_Data_csv")
    
    # 一貫性のある町丁リスト
    consistent_file = output_dir / "consistent_towns_after_nan_fill.txt"
    with open(consistent_file, 'w', encoding='utf-8') as f:
        f.write("=== 2009年合併町をNaNで埋めた後の一貫性のある町丁 ===\n")
        f.write(f"一貫性率: {consistency_rate:.1f}%\n")
        f.write(f"一貫性のある町丁数: {len(consistent_towns)}\n")
        f.write(f"全町丁数: {len(all_towns)}\n\n")
        for town in sorted(consistent_towns):
            f.write(f"{town}\n")
    
    # 一貫性分析結果
    analysis_file = output_dir / "town_consistency_after_nan_fill.csv"
    analysis_data = []
    for town, count in town_consistency.items():
        missing_files = [fname for fname, towns in file_towns.items() if town not in towns]
        analysis_data.append({
            '町丁名': town,
            '出現回数': count,
            '全ファイル数': total_files,
            '一貫性': count == total_files,
            '欠損ファイル数': len(missing_files),
            '欠損ファイル': ', '.join(missing_files) if len(missing_files) <= 3 else f"{len(missing_files)}ファイル"
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(analysis_file, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 結果保存完了 ===")
    print(f"✓ 一貫性のある町丁リスト: {consistent_file}")
    print(f"✓ 詳細分析結果: {analysis_file}")
    
    return consistency_rate, consistent_towns, all_towns

def main():
    """メイン処理"""
    consistency_rate, consistent_towns, all_towns = analyze_town_consistency()
    
    if consistency_rate > 80:
        print(f"\n🎉 素晴らしい！町丁一貫性率が{consistency_rate:.1f}%に達しました！")
    elif consistency_rate > 60:
        print(f"\n👍 良好な結果です！町丁一貫性率が{consistency_rate:.1f}%です。")
    else:
        print(f"\n⚠️  さらなる改善が必要です。現在の一貫性率: {consistency_rate:.1f}%")

if __name__ == "__main__":
    main()
