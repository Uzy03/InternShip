#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
町丁一貫性チェックツール
Data_csv/配下の全年度データで町丁が一貫しているかを調査
"""

import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def load_csv_file(file_path):
    """CSVファイルを読み込み、町丁名を抽出"""
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # ファイル名から年月度を抽出
        filename = Path(file_path).stem
        year_month = filename
        
        # 町丁名を抽出（年齢区分列から町丁名を探す）
        town_names = []
        
        for i, row in df.iterrows():
            # 年齢区分列の内容を確認
            age_col = str(row.iloc[0]) if len(row) > 0 else ""
            
            # 町丁名の特徴：年齢区分ではない文字列（数字や年齢表記でない）
            if (age_col and 
                not any(char.isdigit() for char in age_col) and
                not any(keyword in age_col for keyword in ['年齢', '区分', '計', '男', '女', '備考', '人口統計表', '町丁別', '一覧表', '現在', '人', '口']) and
                len(age_col.strip()) > 1 and
                not age_col.strip().startswith('Column')):
                
                town_names.append(age_col.strip())
        
        return year_month, town_names
        
    except Exception as e:
        print(f"✗ ファイル読み込みエラー {file_path}: {e}")
        return None, []

def analyze_town_consistency():
    """全年度データの町丁一貫性を分析"""
    try:
        print("=== 町丁一貫性分析開始 ===")
        
        # Data_csvディレクトリのパス（Colab用）
        csv_dir = Path("Data_csv")
        if not csv_dir.exists():
            print(f"✗ Data_csvディレクトリが見つかりません: {csv_dir}")
            return False
        
        # CSVファイルを検索
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            print("CSVファイルが見つかりませんでした")
            return False
        
        print(f"処理対象ファイル数: {len(csv_files)}")
        
        # 各ファイルの町丁名を読み込み
        town_data = {}
        all_towns = set()
        
        for csv_file in sorted(csv_files):
            year_month, town_names = load_csv_file(csv_file)
            if year_month and town_names:
                town_data[year_month] = town_names
                all_towns.update(town_names)
                print(f"✓ {year_month}: {len(town_names)}町丁")
        
        if not town_data:
            print("データが読み込めませんでした")
            return False
        
        # 全町丁の一覧
        all_towns_list = sorted(list(all_towns))
        print(f"\n全期間で確認された町丁数: {len(all_towns_list)}")
        
        # 各年度の町丁出現状況を分析
        print(f"\n=== 町丁出現状況の詳細分析 ===")
        
        # 年度別町丁数
        year_town_counts = {}
        for year_month, towns in town_data.items():
            year_town_counts[year_month] = len(towns)
        
        # 町丁別出現年度数
        town_appearance_count = defaultdict(int)
        for towns in town_data.values():
            for town in towns:
                town_appearance_count[town] += 1
        
        # 分析結果を表示
        print(f"\n年度別町丁数:")
        for year_month in sorted(year_town_counts.keys()):
            print(f"  {year_month}: {year_town_counts[year_month]}町丁")
        
        print(f"\n町丁別出現回数:")
        for town in sorted(all_towns_list):
            count = town_appearance_count[town]
            status = "✓" if count == len(town_data) else f"⚠️ ({count}/{len(town_data)})"
            print(f"  {town}: {status}")
        
        # 一貫性の分析
        print(f"\n=== 一貫性分析結果 ===")
        
        # 全年度で一貫して出現する町丁
        consistent_towns = [town for town, count in town_appearance_count.items() 
                          if count == len(town_data)]
        
        # 一部の年度でしか出現しない町丁
        inconsistent_towns = [town for town, count in town_appearance_count.items() 
                            if count < len(town_data)]
        
        print(f"全年度で一貫して出現する町丁: {len(consistent_towns)}町丁")
        print(f"一貫性のない町丁: {len(inconsistent_towns)}町丁")
        
        if inconsistent_towns:
            print(f"\n一貫性のない町丁の詳細:")
            for town in sorted(inconsistent_towns):
                count = town_appearance_count[town]
                missing_years = []
                for year_month, towns in town_data.items():
                    if town not in towns:
                        missing_years.append(year_month)
                print(f"  {town}: {count}/{len(town_data)}回出現")
                print(f"    欠損年度: {', '.join(missing_years)}")
        
        # 年度別の町丁変化を分析
        print(f"\n=== 年度別町丁変化分析 ===")
        
        # 町丁の増減を追跡
        town_changes = {}
        years = sorted(town_data.keys())
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            prev_towns = set(town_data[prev_year])
            curr_towns = set(town_data[curr_year])
            
            added = curr_towns - prev_towns
            removed = prev_towns - curr_towns
            
            if added or removed:
                town_changes[f"{prev_year}→{curr_year}"] = {
                    'added': list(added),
                    'removed': list(removed)
                }
        
        if town_changes:
            print(f"町丁の変化が確認された年度:")
            for change_period, changes in town_changes.items():
                if changes['added']:
                    print(f"  {change_period}: +{len(changes['added'])}町丁追加")
                    for town in changes['added']:
                        print(f"    + {town}")
                if changes['removed']:
                    print(f"  {change_period}: -{len(changes['removed'])}町丁削除")
                    for town in changes['removed']:
                        print(f"    - {town}")
        else:
            print("町丁の変化は確認されませんでした")
        
        # 結果をCSVファイルに保存
        output_file = csv_dir / "town_consistency_analysis.csv"
        
        # 分析結果をDataFrameにまとめる
        analysis_data = []
        for town in all_towns_list:
            appearances = []
            for year_month in sorted(town_data.keys()):
                appearances.append("✓" if town in town_data[year_month] else "✗")
            
            row = [town, town_appearance_count[town], len(town_data)] + appearances
            analysis_data.append(row)
        
        # 列名を設定
        columns = ['町丁名', '出現回数', '総年度数'] + sorted(town_data.keys())
        analysis_df = pd.DataFrame(analysis_data, columns=columns)
        
        # CSVに保存
        analysis_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 分析結果を保存しました: {output_file}")
        
        # 一貫性の統計
        consistency_rate = len(consistent_towns) / len(all_towns_list) * 100
        print(f"\n=== 一貫性統計 ===")
        print(f"全町丁数: {len(all_towns_list)}")
        print(f"一貫性のある町丁数: {len(consistent_towns)}")
        print(f"一貫性率: {consistency_rate:.1f}%")
        
        if consistency_rate < 100:
            print(f"⚠️  一部の町丁で一貫性がありません")
        else:
            print(f"✓ すべての町丁で完全な一貫性が確認されました")
        
        return True
        
    except Exception as e:
        print(f"✗ 分析に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_visualization():
    """町丁一貫性の可視化を作成"""
    try:
        print("\n=== 可視化の作成 ===")
        
        csv_dir = Path("Data_csv")
        csv_files = list(csv_dir.glob("*.csv"))
        
        if not csv_files:
            print("CSVファイルが見つかりません")
            return False
        
        # 各ファイルの町丁数を集計
        year_town_counts = {}
        for csv_file in sorted(csv_files):
            year_month = csv_file.stem
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 町丁数をカウント（年齢区分列から町丁名を抽出）
            town_count = 0
            for i, row in df.iterrows():
                age_col = str(row.iloc[0]) if len(row) > 0 else ""
                if (age_col and 
                    not any(char.isdigit() for char in age_col) and
                    not any(keyword in age_col for keyword in ['年齢', '区分', '計', '男', '女', '備考', '人口統計表', '町丁別', '一覧表', '現在', '人', '口']) and
                    len(age_col.strip()) > 1 and
                    not age_col.strip().startswith('Column')):
                    town_count += 1
            
            year_town_counts[year_month] = town_count
        
        # グラフを作成
        plt.figure(figsize=(15, 8))
        
        # 年度別町丁数の推移
        years = sorted(year_town_counts.keys())
        counts = [year_town_counts[year] for year in years]
        
        plt.subplot(2, 1, 1)
        plt.plot(range(len(years)), counts, marker='o', linewidth=2, markersize=6)
        plt.title('年度別町丁数の推移', fontsize=14, fontweight='bold')
        plt.xlabel('年度')
        plt.ylabel('町丁数')
        plt.xticks(range(len(years)), years, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 町丁数の変化率
        plt.subplot(2, 1, 2)
        changes = []
        for i in range(1, len(counts)):
            change_rate = ((counts[i] - counts[i-1]) / counts[i-1]) * 100
            changes.append(change_rate)
        
        plt.bar(range(len(changes)), changes, alpha=0.7, color='skyblue')
        plt.title('年度間町丁数変化率 (%)', fontsize=14, fontweight='bold')
        plt.xlabel('年度間')
        plt.ylabel('変化率 (%)')
        
        # x軸ラベルを設定
        change_labels = []
        for i in range(1, len(years)):
            change_labels.append(f"{years[i-1]}→{years[i]}")
        plt.xticks(range(len(changes)), change_labels, rotation=45)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # グラフを保存
        output_file = Path("Preprocessed_Data_csv") / "town_consistency_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 可視化を保存しました: {output_file}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"✗ 可視化の作成に失敗: {e}")
        return False

def main():
    """メイン処理"""
    print("=== 町丁一貫性チェックツール ===")
    
    print(f"\n処理方法を選択してください:")
    print(f"  1. 町丁一貫性分析")
    print(f"  2. 可視化作成")
    print(f"  3. 両方実行")
    
    choice = input("選択 (1-3): ").strip()
    
    if choice == "1":
        analyze_town_consistency()
    elif choice == "2":
        create_visualization()
    elif choice == "3":
        analyze_town_consistency()
        create_visualization()
    else:
        print("無効な選択です")

if __name__ == "__main__":
    main()
