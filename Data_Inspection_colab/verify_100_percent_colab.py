#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100%一貫性達成の最終検証スクリプト（Google Colab版）
最終統一後の町丁一貫性率を検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

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

def verify_100_percent_final():
    """最終解決後の100%町丁一貫性率達成の検証"""
    print("=== 最終検証: 100%町丁一貫性率達成の検証開始 ===")
    
    csv_dir = Path("Preprocessed_Data_csv")
    
    # 最終解決済みファイルを取得
    csv_files = list(csv_dir.glob("*_100percent_final.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ 最終解決済みファイルが見つかりません")
        print("先にfinal_unification_colab.pyを実行してください")
        return False
    
    print(f"✓ 検証対象ファイル数: {len(csv_files)}")
    
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
    
    print(f"\n=== 最終検証結果 ===")
    print(f"全町丁数: {len(all_towns)}")
    print(f"一貫性のある町丁数: {len(consistent_towns)}")
    print(f"一貫性のない町丁数: {len(inconsistent_towns)}")
    print(f"最終町丁一貫性率: {consistency_rate:.1f}%")
    
    # 100%達成の判定
    if consistency_rate == 100.0:
        print(f"\n🎉🎉🎉 おめでとうございます！ 🎉🎉🎉")
        print(f"町丁一貫性率100%を達成しました！")
        print(f"完璧なデータセットが完成しました！")
    elif consistency_rate > 99.5:
        print(f"\n🎯 ほぼ完璧です！")
        print(f"町丁一貫性率{consistency_rate:.1f}%で、99.5%を上回っています！")
    elif consistency_rate > 99.0:
        print(f"\n👍 素晴らしい結果です！")
        print(f"町丁一貫性率{consistency_rate:.1f}%で、99%を上回っています！")
    else:
        print(f"\n⚠️  さらなる改善が必要です。")
        print(f"現在の一貫性率: {consistency_rate:.1f}%")
    
    # 一貫性のない町丁がある場合の詳細
    if inconsistent_towns:
        print(f"\n=== 一貫性のない町丁（出現回数順） ===")
        sorted_inconsistent = sorted(inconsistent_towns, key=lambda x: town_consistency[x], reverse=True)
        
        for town in sorted_inconsistent[:10]:  # 上位10件を表示
            count = town_consistency[town]
            missing_files = [fname for fname, towns in file_towns.items() if town not in towns]
            print(f"{town}: {count}/{total_files}回出現")
            if len(missing_files) <= 5:
                print(f"  欠損年度: {', '.join(missing_files)}")
            else:
                print(f"  欠損年度: {len(missing_files)}ファイル")
    
    # 結果を保存
    output_dir = Path("Preprocessed_Data_csv")
    
    # 最終検証結果
    verification_file = output_dir / "100_percent_final_verification_result.txt"
    with open(verification_file, 'w', encoding='utf-8') as f:
        f.write("=== 最終解決後の100%町丁一貫性率達成の検証結果 ===\n")
        f.write(f"最終一貫性率: {consistency_rate:.1f}%\n")
        f.write(f"一貫性のある町丁数: {len(consistent_towns)}\n")
        f.write(f"全町丁数: {len(all_towns)}\n")
        f.write(f"一貫性のない町丁数: {len(inconsistent_towns)}\n\n")
        
        if consistency_rate == 100.0:
            f.write("🎉 100%達成！完璧なデータセットです！\n\n")
        else:
            f.write(f"⚠️  100%未達成。現在{consistency_rate:.1f}%\n\n")
        
        f.write("=== 一貫性のある町丁 ===\n")
        for town in sorted(consistent_towns):
            f.write(f"{town}\n")
    
    print(f"\n=== 最終検証結果保存完了 ===")
    print(f"✓ 最終検証結果: {verification_file}")
    
    return consistency_rate, consistent_towns, all_towns

def create_final_visualization():
    """最終結果の可視化"""
    print("\n=== 最終結果の可視化 ===")
    
    try:
        # データ読み込み
        analysis_file = Path("Preprocessed_Data_csv/town_consistency_analysis.csv")
        if analysis_file.exists():
            df = pd.read_csv(analysis_file, encoding='utf-8-sig')
            
            # 最終結果の可視化
            plt.figure(figsize=(10, 6))
            
            # 一貫性の分布
            plt.subplot(1, 2, 1)
            consistency_counts = df['一貫性'].value_counts()
            colors = ['#ff6b6b', '#4ecdc4']
            plt.pie(consistency_counts.values, labels=['一貫性なし', '一貫性あり'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('町丁一貫性の分布', fontsize=14, fontweight='bold')
            
            # 出現回数の分布
            plt.subplot(1, 2, 2)
            plt.hist(df['出現回数'], bins=20, alpha=0.7, edgecolor='black', color='#4ecdc4')
            plt.xlabel('出現回数')
            plt.ylabel('町丁数')
            plt.title('町丁出現回数の分布', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存
            output_dir = Path("Preprocessed_Data_csv")
            plot_file = output_dir / "final_consistency_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ 最終可視化完了: {plot_file}")
            
            plt.show()
        else:
            print("⚠️  分析結果ファイルが見つかりません")
            
    except Exception as e:
        print(f"⚠️  可視化中にエラー: {e}")

def main():
    """メイン処理"""
    print("=== 100%一貫性達成の最終検証開始（Google Colab版） ===")
    
    # 最終検証
    consistency_rate, consistent_towns, all_towns = verify_100_percent_final()
    
    # 最終可視化
    create_final_visualization()
    
    print("\n=== 全処理完了 ===")
    if consistency_rate == 100.0:
        print("🎉🎉🎉 おめでとうございます！ 🎉🎉🎉")
        print("町丁一貫性率100%を達成しました！")
        print("完璧なデータセットが完成しました！")
    else:
        print(f"現在の一貫性率: {consistency_rate:.1f}%")
        print("さらなる改善が必要な場合があります。")
    
    print("\n=== プロジェクト完了 ===")
    print("✓ 全6ステップの前処理が完了しました")
    print("✓ 出力先: Preprocessed_Data_csv/")
    print("✓ 最終ファイル: *_100percent_final.csv")

if __name__ == "__main__":
    main()
