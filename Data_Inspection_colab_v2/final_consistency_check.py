#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区制町丁統一後の最終的な町丁一貫性率を計算するスクリプト
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

def analyze_final_consistency():
    """最終的な町丁一貫性を分析"""
    print("=== 区制町丁統一後の最終的な町丁一貫性分析 ===")
    
    # ディレクトリ設定
    csv_dir = Path("Preprocessed_Data_csv")
    
    # 統一済みファイルを取得
    csv_files = list(csv_dir.glob("*_unified.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ 統一済みファイルが見つかりません")
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
    
    print(f"\n=== 最終分析結果 ===")
    print(f"全町丁数: {len(all_towns)}")
    print(f"一貫性のある町丁数: {len(consistent_towns)}")
    print(f"一貫性のない町丁数: {len(inconsistent_towns)}")
    print(f"最終町丁一貫性率: {consistency_rate:.1f}%")
    
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
    
    # 改善の進捗を表示
    print(f"\n=== 改善の進捗 ===")
    print(f"初期状態: 17.8%")
    print(f"2009年合併町NaN埋め後: 90.5%")
    print(f"区制町丁統一後: {consistency_rate:.1f}%")
    print(f"総合改善幅: +{consistency_rate - 17.8:.1f}%")
    
    # 結果を保存
    output_dir = Path("Preprocessed_Data_csv")
    
    # 最終的な一貫性のある町丁リスト
    final_consistent_file = output_dir / "final_consistent_towns.txt"
    with open(final_consistent_file, 'w', encoding='utf-8') as f:
        f.write("=== 最終的な一貫性のある町丁 ===\n")
        f.write(f"最終一貫性率: {consistency_rate:.1f}%\n")
        f.write(f"一貫性のある町丁数: {len(consistent_towns)}\n")
        f.write(f"全町丁数: {len(all_towns)}\n")
        f.write(f"改善の進捗:\n")
        f.write(f"  初期状態: 17.8%\n")
        f.write(f"  2009年合併町NaN埋め後: 90.5%\n")
        f.write(f"  区制町丁統一後: {consistency_rate:.1f}%\n")
        f.write(f"  総合改善幅: +{consistency_rate - 17.8:.1f}%\n\n")
        for town in sorted(consistent_towns):
            f.write(f"{town}\n")
    
    # 最終分析結果
    final_analysis_file = output_dir / "final_town_consistency_analysis.csv"
    final_analysis_data = []
    for town, count in town_consistency.items():
        missing_files = [fname for fname, towns in file_towns.items() if town not in towns]
        final_analysis_data.append({
            '町丁名': town,
            '出現回数': count,
            '全ファイル数': total_files,
            '一貫性': count == total_files,
            '欠損ファイル数': len(missing_files),
            '欠損ファイル': ', '.join(missing_files) if len(missing_files) <= 3 else f"{len(missing_files)}ファイル"
        })
    
    final_analysis_df = pd.DataFrame(final_analysis_data)
    final_analysis_df.to_csv(final_analysis_file, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 最終結果保存完了 ===")
    print(f"✓ 最終一貫性町丁リスト: {final_consistent_file}")
    print(f"✓ 最終詳細分析結果: {final_analysis_file}")
    
    return consistency_rate, consistent_towns, all_towns

def main():
    """メイン処理"""
    consistency_rate, consistent_towns, all_towns = analyze_final_consistency()
    
    if consistency_rate > 95:
        print(f"\n🎉 素晴らしい！最終町丁一貫性率が{consistency_rate:.1f}%に達しました！")
        print(f"目標の95%を大幅に上回っています！")
    elif consistency_rate > 90:
        print(f"\n🎯 優秀な結果です！最終町丁一貫性率が{consistency_rate:.1f}%です。")
        print(f"95%の目標まであと{95 - consistency_rate:.1f}%です。")
    elif consistency_rate > 80:
        print(f"\n👍 良好な結果です！最終町丁一貫性率が{consistency_rate:.1f}%です。")
        print(f"さらなる改善の余地があります。")
    else:
        print(f"\n⚠️  さらなる改善が必要です。現在の一貫性率: {consistency_rate:.1f}%")

if __name__ == "__main__":
    main()
