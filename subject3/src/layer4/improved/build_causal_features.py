# -*- coding: utf-8 -*-
# src/layer4/improved/build_causal_features.py
"""
因果関係学習用の特徴量構築
- 生のイベントデータを直接使用
- 時空間的相互作用特徴量を追加
- LightGBMが真の因果関係を学習できるようにする

設計:
- 入力: features_panel.csv, events_matrix_signed.csv
- 出力: features_causal.csv (因果関係学習用特徴量)
- 特徴量:
  1. 生のイベントデータ (event_*_t, event_*_t1)
  2. 時空間ラグ特徴量 (近隣町丁のイベント影響)
  3. 時系列ラグ特徴量 (過去のイベント影響)
  4. 相互作用特徴量 (イベント間の相互作用)
  5. 既存の人口・マクロ特徴量
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from scipy.spatial.distance import cdist

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from src.common.spatial import calculate_spatial_lags_simple, detect_cols_to_lag
from src.common.feature_gate import drop_excluded_columns

# パス設定（improved/配下から相対パス）
P_PANEL = "../../../data/processed/features_panel.csv"
P_EVENTS = "../../../data/processed/events_matrix_signed.csv"
P_CENTROIDS = "../../../data/processed/town_centroids.csv"
P_OUTPUT = "../../../data/processed/features_causal.csv"

# イベントタイプの定義
EVENT_TYPES = {
    "housing", "commercial", "transit", "policy_boundary", 
    "public_edu_medical", "employment", "disaster"
}

def load_data():
    """データの読み込み"""
    print("[因果学習] データを読み込み中...")
    
    panel = pd.read_csv(P_PANEL)
    events = pd.read_csv(P_EVENTS)
    centroids = pd.read_csv(P_CENTROIDS) if Path(P_CENTROIDS).exists() else None
    
    print(f"[因果学習] パネルデータ: {len(panel)}行")
    print(f"[因果学習] イベントデータ: {len(events)}行")
    if centroids is not None:
        print(f"[因果学習] 重心データ: {len(centroids)}行")
    
    return panel, events, centroids

def create_raw_event_features(panel, events):
    """生のイベント特徴量を作成"""
    print("[因果学習] 生のイベント特徴量を作成中...")
    
    # パネルとイベントを結合
    df = panel.merge(events, on=["town", "year"], how="left")
    
    # イベント列を特定
    event_cols = [col for col in events.columns if col.startswith("event_")]
    print(f"[因果学習] イベント列数: {len(event_cols)}")
    
    # 欠損値を0で埋める
    for col in event_cols:
        df[col] = df[col].fillna(0.0)
    
    return df, event_cols

def create_temporal_lag_features(df, event_cols):
    """時系列ラグ特徴量を作成"""
    print("[因果学習] 時系列ラグ特徴量を作成中...")
    
    df = df.sort_values(["town", "year"]).copy()
    
    # 各町丁で時系列ラグを計算
    for col in event_cols:
        # 1年ラグ
        df[f"{col}_lag1"] = df.groupby("town")[col].shift(1)
        # 2年ラグ
        df[f"{col}_lag2"] = df.groupby("town")[col].shift(2)
        # 3年ラグ
        df[f"{col}_lag3"] = df.groupby("town")[col].shift(3)
        
        # 移動平均
        df[f"{col}_ma2"] = df.groupby("town")[col].rolling(2).mean().reset_index(0, drop=True)
        df[f"{col}_ma3"] = df.groupby("town")[col].rolling(3).mean().reset_index(0, drop=True)
        
        # 累積効果
        df[f"{col}_cumsum"] = df.groupby("town")[col].cumsum()
        
        # 変化率
        df[f"{col}_pct_change"] = df.groupby("town")[col].pct_change()
    
    return df

def create_spatial_lag_features(df, centroids, event_cols):
    """空間ラグ特徴量を作成"""
    print("[因果学習] 空間ラグ特徴量を作成中...")
    
    if centroids is None:
        print("[因果学習] 重心データが利用できないため、空間ラグをスキップします")
        return df
    
    # 空間ラグの計算
    df = calculate_spatial_lags_simple(
        df, 
        centroids, 
        event_cols, 
        town_col="town", 
        year_col="year", 
        k_neighbors=5
    )
    
    return df

def create_interaction_features(df, event_cols):
    """相互作用特徴量を作成"""
    print("[因果学習] 相互作用特徴量を作成中...")
    
    # イベントタイプ別の相互作用
    for event_type in EVENT_TYPES:
        # 増加・減少の組み合わせ
        inc_cols = [col for col in event_cols if f"event_{event_type}_inc" in col]
        dec_cols = [col for col in event_cols if f"event_{event_type}_dec" in col]
        
        if inc_cols and dec_cols:
            # 増加と減少の相互作用
            for inc_col in inc_cols:
                for dec_col in dec_cols:
                    df[f"{inc_col}_x_{dec_col}"] = df[inc_col] * df[dec_col]
    
    # 異なるイベントタイプ間の相互作用
    event_type_pairs = [
        ("housing", "commercial"),
        ("housing", "employment"),
        ("commercial", "employment"),
        ("disaster", "housing"),
        ("disaster", "commercial"),
    ]
    
    for type1, type2 in event_type_pairs:
        type1_cols = [col for col in event_cols if f"event_{type1}_" in col]
        type2_cols = [col for col in event_cols if f"event_{type2}_" in col]
        
        if type1_cols and type2_cols:
            # 主要な組み合わせのみ作成
            main_col1 = f"event_{type1}_inc_t"
            main_col2 = f"event_{type2}_inc_t"
            
            if main_col1 in df.columns and main_col2 in df.columns:
                df[f"{type1}_x_{type2}_interaction"] = df[main_col1] * df[main_col2]
    
    return df

def create_intensity_features(df, event_cols):
    """強度特徴量を作成"""
    print("[因果学習] 強度特徴量を作成中...")
    
    # 各年のイベント総数
    df["total_events_t"] = df[[col for col in event_cols if col.endswith("_t")]].sum(axis=1)
    df["total_events_t1"] = df[[col for col in event_cols if col.endswith("_t1")]].sum(axis=1)
    
    # イベントタイプ別の強度
    for event_type in EVENT_TYPES:
        type_cols = [col for col in event_cols if f"event_{event_type}_" in col]
        if type_cols:
            df[f"{event_type}_intensity"] = df[type_cols].sum(axis=1)
    
    # 正のイベントと負のイベントの分離
    inc_cols = [col for col in event_cols if "_inc_" in col]
    dec_cols = [col for col in event_cols if "_dec_" in col]
    
    if inc_cols:
        df["positive_events_t"] = df[[col for col in inc_cols if col.endswith("_t")]].sum(axis=1)
        df["positive_events_t1"] = df[[col for col in inc_cols if col.endswith("_t1")]].sum(axis=1)
    
    if dec_cols:
        df["negative_events_t"] = df[[col for col in dec_cols if col.endswith("_t")]].sum(axis=1)
        df["negative_events_t1"] = df[[col for col in dec_cols if col.endswith("_t1")]].sum(axis=1)
    
    return df

def create_regime_features(df):
    """レジーム特徴量を作成"""
    print("[因果学習] レジーム特徴量を作成中...")
    
    # 時代区分
    df["era_pre2010"] = (df["year"] < 2010).astype(int)
    df["era_2010_2015"] = ((df["year"] >= 2010) & (df["year"] < 2015)).astype(int)
    df["era_2015_2020"] = ((df["year"] >= 2015) & (df["year"] < 2020)).astype(int)
    df["era_post2020"] = (df["year"] >= 2020).astype(int)
    
    # コロナ期間
    df["era_covid"] = ((df["year"] >= 2020) & (df["year"] <= 2022)).astype(int)
    
    return df

def create_population_features(df):
    """人口関連特徴量を作成"""
    print("[因果学習] 人口関連特徴量を作成中...")
    
    # 人口の対数変換
    if "pop_total" in df.columns:
        df["log_pop_total"] = np.log1p(df["pop_total"])
        df["sqrt_pop_total"] = np.sqrt(df["pop_total"])
    
    # 人口密度（簡易版）
    if "pop_total" in df.columns:
        df["pop_density"] = df["pop_total"] / 1000  # 千人あたり
    
    # 年齢構成の特徴量
    age_cols = [col for col in df.columns if "歳" in col and col != "100歳以上"]
    if age_cols:
        # 若年層比率
        young_cols = [col for col in age_cols if any(age in col for age in ["0〜4歳", "5〜9歳", "10〜14歳", "15〜19歳"])]
        if young_cols:
            df["young_ratio"] = df[young_cols].sum(axis=1) / df["pop_total"]
        
        # 高齢者比率
        elderly_cols = [col for col in age_cols if any(age in col for age in ["65〜69歳", "70〜74歳", "75〜79歳", "80〜84歳", "85〜89歳", "90〜94歳", "95〜99歳"])]
        if elderly_cols:
            df["elderly_ratio"] = df[elderly_cols].sum(axis=1) / df["pop_total"]
    
    return df

def main():
    """メイン処理"""
    print("[因果学習] 因果関係学習用特徴量を構築中...")
    
    # データ読み込み
    panel, events, centroids = load_data()
    
    # 生のイベント特徴量を作成
    df, event_cols = create_raw_event_features(panel, events)
    
    # 時系列ラグ特徴量を作成
    df = create_temporal_lag_features(df, event_cols)
    
    # 空間ラグ特徴量を作成
    df = create_spatial_lag_features(df, centroids, event_cols)
    
    # 相互作用特徴量を作成
    df = create_interaction_features(df, event_cols)
    
    # 強度特徴量を作成
    df = create_intensity_features(df, event_cols)
    
    # レジーム特徴量を作成
    df = create_regime_features(df)
    
    # 人口関連特徴量を作成
    df = create_population_features(df)
    
    # 欠損値処理
    df = df.fillna(0.0)
    
    # 保存
    df.to_csv(P_OUTPUT, index=False)
    print(f"[因果学習] 特徴量を保存しました: {P_OUTPUT}")
    print(f"[因果学習] 総特徴量数: {len(df.columns)}")
    print(f"[因果学習] 生のイベント特徴量数: {len(event_cols)}")
    
    # 特徴量の概要を表示
    print("\n[因果学習] 特徴量の概要:")
    print(f"- 生のイベント特徴量: {len(event_cols)}個")
    print(f"- 時系列ラグ特徴量: {len([col for col in df.columns if '_lag' in col or '_ma' in col or '_cumsum' in col or '_pct_change' in col])}個")
    print(f"- 空間ラグ特徴量: {len([col for col in df.columns if col.startswith('ring')])}個")
    print(f"- 相互作用特徴量: {len([col for col in df.columns if '_x_' in col or '_interaction' in col])}個")
    print(f"- 強度特徴量: {len([col for col in df.columns if 'intensity' in col or 'total_events' in col or 'positive_events' in col or 'negative_events' in col])}個")
    print(f"- レジーム特徴量: {len([col for col in df.columns if col.startswith('era_')])}個")
    print(f"- 人口関連特徴量: {len([col for col in df.columns if 'log_pop' in col or 'sqrt_pop' in col or 'pop_density' in col or 'ratio' in col])}個")

if __name__ == "__main__":
    main()