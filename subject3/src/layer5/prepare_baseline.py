# -*- coding: utf-8 -*-
# src/layer5/prepare_baseline.py
"""
基準年の土台準備：features_panel.csvから指定町丁・基準年のベース値を取得
出力: 単一行DataFrame（キーとベース値を含む）

設計:
- 入力: features_panel.csv, town, base_year
- 出力: 単一行DataFrame（town, year=base_year, pop_total, lag_d1, lag_d2, ...）
- 見つからない場合：最近接年（<= base_year）から安全コピー、無い列は NaN
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# パス設定
P_FEATURES_PANEL = "../../data/processed/features_panel.csv"

# L4で必要な最小ラグ列（choose_features()から抽出）
REQUIRED_LAG_COLS = [
    "lag_d1", "lag_d2", "ma2_delta", "town_ma5", "town_std5", "town_trend5"
]

# その他の重要列
OTHER_IMPORTANT_COLS = [
    "pop_total", "male", "female", "city_pop", "city_growth_log"
]

def find_baseline_data(panel_df: pd.DataFrame, town: str, base_year: int) -> pd.DataFrame:
    """指定町丁・基準年のベースデータを取得"""
    # 該当町丁のデータをフィルタ
    town_data = panel_df[panel_df["town"] == town].copy()
    
    if len(town_data) == 0:
        raise ValueError(f"町丁 '{town}' のデータが見つかりません")
    
    # 基準年またはそれ以前の最新年を取得
    available_years = town_data[town_data["year"] <= base_year]["year"].tolist()
    
    if not available_years:
        # 基準年以前のデータがない場合、最も近い年を使用
        available_years = town_data["year"].tolist()
        if not available_years:
            raise ValueError(f"町丁 '{town}' にデータがありません")
    
    # 基準年またはそれ以前の最新年を選択
    target_year = max(available_years)
    
    # 該当年のデータを取得
    baseline = town_data[town_data["year"] == target_year].copy()
    
    if len(baseline) == 0:
        raise ValueError(f"町丁 '{town}' の年 {target_year} のデータが見つかりません")
    
    # 複数行ある場合は最初の行を選択
    baseline = baseline.iloc[0:1].copy()
    
    # 年を基準年に更新
    baseline["year"] = base_year
    
    return baseline

def ensure_required_columns(baseline: pd.DataFrame, panel_df: pd.DataFrame, town: str) -> pd.DataFrame:
    """必要な列が存在することを確認し、不足分はNaNで埋める"""
    # 全列のリスト
    all_cols = set(panel_df.columns)
    
    # 不足している列を特定
    missing_cols = []
    for col in REQUIRED_LAG_COLS + OTHER_IMPORTANT_COLS:
        if col not in baseline.columns:
            missing_cols.append(col)
    
    # 不足列をNaNで追加
    for col in missing_cols:
        baseline[col] = np.nan
    
    return baseline

def calculate_lag_features(baseline: pd.DataFrame, panel_df: pd.DataFrame, town: str) -> pd.DataFrame:
    """ラグ特徴を計算（可能な場合）"""
    town_data = panel_df[panel_df["town"] == town].sort_values("year")
    
    if len(town_data) < 2:
        print(f"[WARN] 町丁 '{town}' のデータが不足（{len(town_data)}行）のため、ラグ特徴を計算できません")
        return baseline
    
    # delta_peopleが存在するかチェック
    if "delta_people" not in town_data.columns:
        if "pop_total" in town_data.columns:
            # delta_peopleを計算
            town_data = town_data.copy()
            town_data["delta_people"] = town_data["pop_total"].diff()
        else:
            print(f"[WARN] 町丁 '{town}' にdelta_peopleもpop_totalもないため、ラグ特徴を計算できません")
            return baseline
    
    # ラグ特徴の計算
    if "lag_d1" in baseline.columns and baseline["lag_d1"].isna().all():
        # 最新のdelta_peopleをlag_d1に設定
        latest_delta = town_data["delta_people"].iloc[-1]
        if not pd.isna(latest_delta):
            baseline["lag_d1"] = latest_delta
    
    if "lag_d2" in baseline.columns and baseline["lag_d2"].isna().all():
        # 2番目に新しいdelta_peopleをlag_d2に設定
        if len(town_data) >= 2:
            second_latest_delta = town_data["delta_people"].iloc[-2]
            if not pd.isna(second_latest_delta):
                baseline["lag_d2"] = second_latest_delta
    
    # 移動平均の計算
    if "ma2_delta" in baseline.columns and baseline["ma2_delta"].isna().all():
        if len(town_data) >= 2:
            ma2 = town_data["delta_people"].rolling(window=2).mean().iloc[-1]
            if not pd.isna(ma2):
                baseline["ma2_delta"] = ma2
    
    # 町丁レベルの統計（可能な場合）
    if "town_ma5" in baseline.columns and baseline["town_ma5"].isna().all():
        if len(town_data) >= 5:
            town_ma5 = town_data["delta_people"].rolling(window=5).mean().iloc[-1]
            if not pd.isna(town_ma5):
                baseline["town_ma5"] = town_ma5
    
    if "town_std5" in baseline.columns and baseline["town_std5"].isna().all():
        if len(town_data) >= 5:
            town_std5 = town_data["delta_people"].rolling(window=5).std().iloc[-1]
            if not pd.isna(town_std5):
                baseline["town_std5"] = town_std5
    
    if "town_trend5" in baseline.columns and baseline["town_trend5"].isna().all():
        if len(town_data) >= 5:
            # 5年間の線形トレンドを計算
            years = town_data["year"].iloc[-5:].values
            deltas = town_data["delta_people"].iloc[-5:].values
            valid_mask = ~pd.isna(deltas)
            
            if valid_mask.sum() >= 3:  # 最低3点必要
                years_valid = years[valid_mask]
                deltas_valid = deltas[valid_mask]
                
                # 線形回帰の傾きを計算
                if len(years_valid) > 1:
                    slope = np.polyfit(years_valid, deltas_valid, 1)[0]
                    baseline["town_trend5"] = slope
    
    return baseline

def prepare_baseline(town: str, base_year: int) -> pd.DataFrame:
    """基準年の土台データを準備"""
    # features_panel.csvを読み込み
    panel_df = pd.read_csv(P_FEATURES_PANEL)
    
    # 基本データの取得
    baseline = find_baseline_data(panel_df, town, base_year)
    
    # 必要な列の確保
    baseline = ensure_required_columns(baseline, panel_df, town)
    
    # ラグ特徴の計算
    baseline = calculate_lag_features(baseline, panel_df, town)
    
    return baseline

def main(town: str, base_year: int) -> None:
    """メイン処理"""
    print(f"[L5] 基準年データを準備中: {town}, {base_year}")
    
    # ベースラインデータの準備
    baseline = prepare_baseline(town, base_year)
    
    # 結果の表示
    print(f"[L5] ベースラインデータを取得しました:")
    print(f"  町丁: {baseline['town'].iloc[0]}")
    print(f"  年: {baseline['year'].iloc[0]}")
    print(f"  人口: {baseline['pop_total'].iloc[0] if 'pop_total' in baseline.columns else 'N/A'}")
    
    # 非NaN列の表示
    non_nan_cols = baseline.columns[~baseline.isna().all()].tolist()
    print(f"  非NaN列数: {len(non_nan_cols)}")
    
    # 重要な列の状態確認
    important_cols = ["pop_total", "lag_d1", "lag_d2", "ma2_delta"]
    for col in important_cols:
        if col in baseline.columns:
            value = baseline[col].iloc[0]
            status = "✓" if not pd.isna(value) else "✗"
            print(f"  {col}: {value} {status}")
    
    return baseline

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("使用方法: python prepare_baseline.py <town> <base_year>")
        sys.exit(1)
    
    town = sys.argv[1]
    base_year = int(sys.argv[2])
    
    baseline = main(town, base_year)
