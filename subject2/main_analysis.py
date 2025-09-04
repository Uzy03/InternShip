#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メイン分析実行スクリプト
データ処理から人口変動分析まで一連の流れを実行する
"""

import os
import sys
from pathlib import Path

def main():
    """メイン実行関数"""
    print("=== 町丁別人口データ分析プロジェクト ===")
    print("Subject 2: 人口増減の要因特定")
    print()
    
    # 1. データ処理の実行
    print("Phase 1: データ整形")
    print("-" * 40)
    
    try:
        from data_processor import PopulationDataProcessor
        processor = PopulationDataProcessor()
        time_series_df, summary_df = processor.process_all()
        print("✓ データ整形完了")
    except Exception as e:
        print(f"✗ データ整形でエラーが発生: {e}")
        return
    
    print()
    
    # 2. 人口変動分析の実行
    print("Phase 2: 人口変動分析")
    print("-" * 40)
    
    try:
        from analysis.population_analysis import PopulationChangeAnalyzer
        analyzer = PopulationChangeAnalyzer()
        results = analyzer.run_analysis()
        print("✓ 人口変動分析完了")
    except Exception as e:
        print(f"✗ 人口変動分析でエラーが発生: {e}")
        return
    
    print()
    
    # 3. 結果サマリーの表示
    print("=== 分析結果サマリー ===")
    print(f"対象期間: {summary_df['年度'].min()}年 〜 {summary_df['年度'].max()}年")
    print(f"対象町丁数: {len(time_series_df)}地区")
    print(f"総年度数: {len(summary_df)}年")
    print()
    
    print("大きな人口変化の上位10件:")
    major_changes = results['major_changes']
    if len(major_changes) > 0:
        top_changes = major_changes.head(10)
        for _, change in top_changes.iterrows():
            print(f"  {change['町丁名']} ({change['年度']}年): {change['変化率(%)']:.1f}% ({change['変化タイプ']})")
    else:
        print("  大きな人口変化は検出されませんでした")
    
    print()
    print("=== 分析完了 ===")
    print("出力ファイルは 'processed_data/' ディレクトリに保存されました")
    
    # 出力ファイル一覧
    output_files = [
        "population_time_series.csv",
        "population_summary.csv", 
        "population_changes.csv",
        "population_anomalies.csv",
        "major_population_changes.csv",
        "population_change_summary.csv"
    ]
    
    print("\n生成されたファイル:")
    for file in output_files:
        file_path = f"processed_data/{file}"
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({file_size:,} bytes)")
        else:
            print(f"  ✗ {file} (未生成)")

if __name__ == "__main__":
    main()
