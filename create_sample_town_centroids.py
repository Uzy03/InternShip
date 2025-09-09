#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
町丁名から仮想的な経緯度データを作成するスクリプト
既存のCSVファイルから町丁名を抽出し、熊本市内の範囲でランダムな経緯度を生成
"""

import pandas as pd
import numpy as np
import os

def create_sample_town_centroids():
    # 既存のCSVファイルから町丁名を取得
    csv_path = "subject3/data/processed/features_l4.csv"
    df = pd.read_csv(csv_path)
    
    # 町丁名の一覧を取得（重複を除去）
    towns = df['town'].unique()
    print(f"町丁数: {len(towns)}")
    
    # 熊本市の大まかな範囲（経緯度）
    # 熊本市の中心付近: 130.7414, 32.7898
    center_lon = 130.7414
    center_lat = 32.7898
    
    # 範囲を設定（約10km四方）
    lon_range = 0.1  # 経度の範囲
    lat_range = 0.1  # 緯度の範囲
    
    # ランダムな経緯度を生成
    np.random.seed(42)  # 再現性のため
    lons = np.random.uniform(center_lon - lon_range/2, center_lon + lon_range/2, len(towns))
    lats = np.random.uniform(center_lat - lat_range/2, center_lat + lat_range/2, len(towns))
    
    # 町丁IDを生成（連番）
    town_ids = [f"{i+1:03d}" for i in range(len(towns))]
    
    # データフレームを作成
    result_df = pd.DataFrame({
        'town_id': town_ids,
        'town': towns,
        'lon': lons,
        'lat': lats
    })
    
    # 出力ディレクトリを作成
    output_dir = "subject3/src/layer5/data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # CSVファイルに保存
    output_path = os.path.join(output_dir, "town_centroids.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"町丁セントロイドCSVを作成しました: {output_path}")
    print(f"町丁数: {len(result_df)}")
    print("\n先頭5行:")
    print(result_df.head())
    
    return output_path

if __name__ == "__main__":
    create_sample_town_centroids()
