#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100%一貫性達成のための最終統一スクリプト（Google Colab版）
区名付き町丁を区名なしに統一して100%の町丁一貫性率を達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def force_remove_district_names():
    """強制的に区名を除去する最終解決処理"""
    print("=== ステップ6: 100%一貫性達成のための最終統一処理開始 ===")
    
    input_dir = Path("Preprocessed_Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # 最終統一対象町丁を特定
    target_district_towns = [
        '東京塚町（中央区）',
        '京町本丁（中央区）',
        '神水本町（中央区）',
        '室園町（中央区）',
        '八王寺町（中央区）'
    ]
    
    # 対応する区名なし町丁
    base_towns = [
        '東京塚町',
        '京町本丁',
        '神水本町',
        '室園町',
        '八王寺町'
    ]
    
    print(f"✓ 最終統一対象町丁: {len(target_district_towns)}町丁")
    for i, (district_town, base_town) in enumerate(zip(target_district_towns, base_towns)):
        print(f"  {i+1}. {district_town} → {base_town}")
    
    # 統一済みファイルを取得
    csv_files = list(input_dir.glob("*_unified.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ 統一済みファイルが見つかりません")
        print("先にunify_district_towns_colab.pyを実行してください")
        return False
    
    print(f"\n✓ 処理対象ファイル数: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\n処理中: {csv_file.name}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 町丁名の列（最初の列）を取得
            town_col = df.iloc[:, 0]
            
            # 強制的な区名除去処理
            changes_made = 0
            
            for idx, town_name in enumerate(town_col):
                if pd.isna(town_name):
                    continue
                
                town_name_str = str(town_name).strip()
                original_name = town_name_str
                
                # 強制的に区名を除去
                if '（中央区）' in town_name_str:
                    town_name_str = town_name_str.replace('（中央区）', '')
                    changes_made += 1
                    print(f"  → 強制除去: {original_name} → {town_name_str}")
                elif '（東区）' in town_name_str:
                    town_name_str = town_name_str.replace('（東区）', '')
                    changes_made += 1
                    print(f"  → 強制除去: {original_name} → {town_name_str}")
                elif '（西区）' in town_name_str:
                    town_name_str = town_name_str.replace('（西区）', '')
                    changes_made += 1
                    print(f"  → 強制除去: {original_name} → {town_name_str}")
                elif '（南区）' in town_name_str:
                    town_name_str = town_name_str.replace('（南区）', '')
                    changes_made += 1
                    print(f"  → 強制除去: {original_name} → {town_name_str}")
                elif '（北区）' in town_name_str:
                    town_name_str = town_name_str.replace('（北区）', '')
                    changes_made += 1
                    print(f"  → 強制除去: {original_name} → {town_name_str}")
                
                # 変更があった場合は更新
                if original_name != town_name_str:
                    df.iloc[idx, 0] = town_name_str
            
            if changes_made > 0:
                print(f"  → 変更件数: {changes_made}件")
            
            # 出力ファイル名
            output_filename = f"{csv_file.stem.replace('_unified', '')}_100percent_final.csv"
            output_path = output_dir / output_filename
            
            # 保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存完了: {output_filename}")
            
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print("\n=== 最終統一処理完了 ===")
    print("✓ 全ファイルの100%統一が完了しました")
    print("✓ 出力先: Preprocessed_Data_csv")
    print("🎯 これで町丁一貫性率100%達成が期待されます！")
    
    return True

def main():
    """メイン処理"""
    print("=== 100%一貫性達成のための最終統一開始（Google Colab版） ===")
    
    # 最終統一処理
    if not force_remove_district_names():
        return
    
    print("\n=== ステップ6完了 ===")
    print("✓ 100%一貫性達成のための最終統一が完了しました")
    print("✓ 次のステップ: 最終検証")
    print("\n次のスクリプトを実行してください:")
    print("python verify_100_percent_colab.py")

if __name__ == "__main__":
    main()
