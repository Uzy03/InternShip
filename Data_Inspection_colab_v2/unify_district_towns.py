#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区制導入による町丁名の不整合を解決するスクリプト
区名付き町丁と区名なし町丁を統一して、最終的な一貫性率を向上
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def identify_district_town_pairs():
    """区制導入による町丁名のペアを特定"""
    district_pairs = {
        # 室園町系
        '室園町': ['室園町（中央区）', '室園町（北区）'],
        
        # 八王寺町系
        '八王寺町': ['八王寺町（南区）', '八王寺町（中央区）'],
        
        # 東京塚町系
        '東京塚町': ['東京塚町（中央区）', '東京塚町（東区）'],
        
        # 神水本町系
        '神水本町': ['神水本町（東区）', '神水本町（中央区）'],
        
        # 津浦町系
        '津浦町': ['津浦町（北区）', '津浦町（西区）'],
        
        # 京町本丁系
        '京町本丁': ['京町本丁（西区）', '京町本丁（中央区）']
    }
    
    # フラットなリストに変換
    all_district_towns = []
    for base_town, district_towns in district_pairs.items():
        all_district_towns.extend(district_towns)
        all_district_towns.append(base_town)
    
    return district_pairs, all_district_towns

def normalize_town_name(town_name):
    """町丁名を正規化（区名を除去）"""
    # 区名を除去: （中央区）、（東区）、（西区）、（南区）、（北区）
    normalized = re.sub(r'（[東南西北中央]区）', '', town_name)
    return normalized.strip()

def unify_district_towns():
    """区制導入による町丁名の不整合を解決"""
    print("=== 区制導入による町丁名の不整合解決処理開始 ===")
    
    # ディレクトリ設定
    input_dir = Path("Preprocessed_Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # 区制町丁ペアを特定
    district_pairs, all_district_towns = identify_district_town_pairs()
    print(f"✓ 区制町丁ペアを特定: {len(district_pairs)}組")
    
    # NaNで埋めたファイルを取得
    csv_files = list(input_dir.glob("*_nan_filled.csv"))
    csv_files.sort()
    
    print(f"✓ 処理対象ファイル数: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\n処理中: {csv_file.name}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 町丁名の列（最初の列）を取得
            town_col = df.iloc[:, 0]
            
            # 区制町丁の統一処理
            unified_towns = set()
            rows_to_remove = []
            
            for idx, town_name in enumerate(town_col):
                if pd.isna(town_name):
                    continue
                
                town_name = str(town_name).strip()
                
                # 区制町丁かチェック
                if town_name in all_district_towns:
                    normalized_name = normalize_town_name(town_name)
                    
                    # 既に統一済みかチェック
                    if normalized_name in unified_towns:
                        # 重複する区制町丁は削除対象
                        rows_to_remove.append(idx)
                        print(f"  → 重複削除: {town_name} → {normalized_name}")
                    else:
                        # 最初の出現時は正規化して統一
                        df.iloc[idx, 0] = normalized_name
                        unified_towns.add(normalized_name)
                        print(f"  → 統一: {town_name} → {normalized_name}")
            
            # 重複行を削除
            if rows_to_remove:
                df = df.drop(df.index[rows_to_remove]).reset_index(drop=True)
                print(f"  → 重複行を削除: {len(rows_to_remove)}行")
            
            # 出力ファイル名
            output_filename = f"{csv_file.stem}_unified.csv"
            output_path = output_dir / output_filename
            
            # 保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存完了: {output_filename}")
            
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print(f"\n=== 処理完了 ===")
    print(f"✓ 全ファイルの区制町丁統一が完了しました")
    print(f"✓ 出力先: {output_dir}")

def main():
    """メイン処理"""
    unify_district_towns()

if __name__ == "__main__":
    main()
