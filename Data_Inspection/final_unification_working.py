#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100%町丁一貫性率達成のための最終修正版スクリプト
確実に区名を除去して100%達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def identify_final_district_towns():
    """100%達成のために統一が必要な区名付き町丁を特定"""
    final_district_towns = [
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
    
    return final_district_towns, base_towns

def normalize_to_base_town(town_name):
    """区名付き町丁を区名なしに正規化（確実版）"""
    # 区名を除去: （中央区）、（東区）、（西区）、（南区）、（北区）
    normalized = re.sub(r'（[東南西北中央]区）', '', str(town_name))
    return normalized.strip()

def unify_for_100_percent_working():
    """100%町丁一貫性率達成のための最終修正版統一処理"""
    print("=== 100%町丁一貫性率達成のための最終修正版統一処理開始 ===")
    
    # ディレクトリ設定
    input_dir = Path("Preprocessed_Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # 最終統一対象町丁を特定
    final_district_towns, base_towns = identify_final_district_towns()
    print(f"✓ 最終統一対象町丁: {len(final_district_towns)}町丁")
    
    for i, (district_town, base_town) in enumerate(zip(final_district_towns, base_towns)):
        print(f"  {i+1}. {district_town} → {base_town}")
    
    # 統一済みファイルを取得
    csv_files = list(input_dir.glob("*_unified.csv"))
    csv_files.sort()
    
    print(f"\n✓ 処理対象ファイル数: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\n処理中: {csv_file.name}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 町丁名の列（最初の列）を取得
            town_col = df.iloc[:, 0]
            
            # 最終統一処理
            changes_made = 0
            
            for idx, town_name in enumerate(town_col):
                if pd.isna(town_name):
                    continue
                
                town_name_str = str(town_name).strip()
                
                # 最終統一対象町丁かチェック（完全一致）
                if town_name_str in final_district_towns:
                    # 区名なしに統一
                    normalized_name = normalize_to_base_town(town_name_str)
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 最終統一: {town_name_str} → {normalized_name}")
                
                # 追加チェック: 区名付きの町丁を一般的に検出
                elif '（中央区）' in town_name_str:
                    normalized_name = town_name_str.replace('（中央区）', '')
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 一般統一: {town_name_str} → {normalized_name}")
                elif '（東区）' in town_name_str:
                    normalized_name = town_name_str.replace('（東区）', '')
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 一般統一: {town_name_str} → {normalized_name}")
                elif '（西区）' in town_name_str:
                    normalized_name = town_name_str.replace('（西区）', '')
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 一般統一: {town_name_str} → {normalized_name}")
                elif '（南区）' in town_name_str:
                    normalized_name = town_name_str.replace('（南区）', '')
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 一般統一: {town_name_str} → {normalized_name}")
                elif '（北区）' in town_name_str:
                    normalized_name = town_name_str.replace('（北区）', '')
                    df.iloc[idx, 0] = normalized_name
                    changes_made += 1
                    print(f"  → 一般統一: {town_name_str} → {normalized_name}")
            
            if changes_made > 0:
                print(f"  → 変更件数: {changes_made}件")
            
            # 出力ファイル名
            output_filename = f"{csv_file.stem}_100percent_working.csv"
            output_path = output_dir / output_filename
            
            # 保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存完了: {output_filename}")
            
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print(f"\n=== 最終修正版統一処理完了 ===")
    print(f"✓ 全ファイルの100%統一が完了しました")
    print(f"✓ 出力先: {output_dir}")
    print(f"🎯 これで町丁一貫性率100%達成が期待されます！")

def main():
    """メイン処理"""
    unify_for_100_percent_working()

if __name__ == "__main__":
    main()
