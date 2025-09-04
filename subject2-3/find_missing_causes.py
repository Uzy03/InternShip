#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原因がまだ埋まっていない行をリストとして表示するスクリプト
"""

import pandas as pd
import os

def find_missing_causes():
    """原因がまだ埋まっていない行をリストとして表示"""
    
    # CSVファイルを読み込み
    file_path = "subject2-3/manual_investigation_targets.csv"
    
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return
    
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 原因列が空またはNaNの行を抽出
        missing_causes = df[df['原因'].isna() | (df['原因'] == '')]
        
        # シンプルに表示
        for idx, row in missing_causes.iterrows():
            print(f"{row['町丁名']},{row['年度']},{row['変化率(%)']},{row['変化タイプ']},{row['変化の大きさ']},{row['前年人口']},{row['人口増減']},")
            
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    find_missing_causes()
