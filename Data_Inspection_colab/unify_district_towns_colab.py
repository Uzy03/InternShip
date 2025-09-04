#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区制町丁統一スクリプト（Google Colab版）
区制導入による町丁名の不整合を解決
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def identify_district_town_pairs():
    """区制町丁のペアを特定"""
    district_pairs = {
        # 中央区
        '室園町': ['室園町（中央区）'],
        '東京塚町': ['東京塚町（中央区）'],
        '京町本丁': ['京町本丁（中央区）'],
        '神水本町': ['神水本町（中央区）'],
        '八王寺町': ['八王寺町（中央区）'],
        
        # 東区
        '池田３丁目': ['池田３丁目（東区）'],
        '出水４丁目': ['出水４丁目（東区）'],
        '江津２丁目': ['江津２丁目（東区）'],
        '帯山４丁目': ['帯山４丁目（東区）'],
        '湖東１丁目': ['湖東１丁目（東区）'],
        '三郎１丁目': ['三郎１丁目（東区）'],
        '平成１丁目': ['平成１丁目（東区）'],
        '平成２丁目': ['平成２丁目（東区）'],
        '保田窪２丁目': ['保田窪２丁目（東区）'],
        
        # 西区
        '池田３丁目': ['池田３丁目（西区）'],
        '横手１丁目': ['横手１丁目（西区）'],
        '横手２丁目': ['横手２丁目（西区）'],
        '横手３丁目': ['横手３丁目（西区）'],
        
        # 南区
        '平成１丁目': ['平成１丁目（南区）'],
        '平成２丁目': ['平成２丁目（南区）'],
        
        # 北区
        '池田３丁目': ['池田３丁目（北区）'],
        '黒髪７丁目': ['黒髪７丁目（北区）']
    }
    
    # すべての区制町丁を収集
    all_district_towns = set()
    for base_town, district_variants in district_pairs.items():
        all_district_towns.update(district_variants)
    
    return district_pairs, all_district_towns

def normalize_town_name(town_name):
    """区制町丁名を正規化（区名除去）"""
    if pd.isna(town_name):
        return town_name
    
    # 区名を除去: （中央区）、（東区）、（西区）、（南区）、（北区）
    normalized = re.sub(r'（[東南西北中央]区）', '', str(town_name))
    
    return normalized.strip()

def unify_district_towns():
    """区制導入による町丁名の不整合を解決"""
    print("=== ステップ5: 区制町丁統一処理開始 ===")
    
    input_dir = Path("Preprocessed_Data_csv")
    output_dir = Path("Preprocessed_Data_csv")
    
    # 区制町丁ペアを特定
    district_pairs, all_district_towns = identify_district_town_pairs()
    print(f"✓ 区制町丁ペア数: {len(district_pairs)}組")
    
    # NaNで埋めたファイルを取得
    csv_files = list(input_dir.glob("*_nan_filled.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ NaN埋め済みファイルが見つかりません")
        print("先にfill_2009_merged_towns_colab.pyを実行してください")
        return False
    
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
            output_filename = f"{csv_file.stem.replace('_nan_filled', '')}_unified.csv"
            output_path = output_dir / output_filename
            
            # 保存
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 保存完了: {output_filename}")
            
        except Exception as e:
            print(f"✗ エラー: {csv_file.name} - {e}")
            continue
    
    print("\n✓ 区制町丁統一処理完了")
    return True

def main():
    """メイン処理"""
    print("=== 区制町丁統一開始（Google Colab版） ===")
    
    # 区制町丁統一
    if not unify_district_towns():
        return
    
    print("\n=== ステップ5完了 ===")
    print("✓ 区制町丁統一が完了しました")
    print("✓ 次のステップ: 100%一貫性達成のための最終統一")
    print("\n次のスクリプトを実行してください:")
    print("python final_unification_colab.py")

if __name__ == "__main__":
    main()
