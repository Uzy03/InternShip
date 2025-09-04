# -*- coding: utf-8 -*-
# src/layer5/intervals.py
"""
予測区間（PI）の推定
入力: l4_per_year_metrics.csv（RMSE列がある前提）・予測する年
出力: pi95 = [ŷ - 1.96*σ, ŷ + 1.96*σ]

設計:
- 予測年の RMSE が無ければ：最近接年 or aggregate RMSE（l4_cv_metrics.json）を fallback
- h>1 の σ は √h 倍の拡大（独立残差近似）
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional

# パス設定
P_PER_YEAR_METRICS = "../../data/processed/l4_per_year_metrics.csv"
P_CV_METRICS = "../../data/processed/l4_cv_metrics.json"

def load_per_year_metrics() -> pd.DataFrame:
    """年別メトリクスを読み込み"""
    if not Path(P_PER_YEAR_METRICS).exists():
        print(f"[WARN] {P_PER_YEAR_METRICS} が見つかりません")
        return pd.DataFrame()
    
    df = pd.read_csv(P_PER_YEAR_METRICS)
    
    if "RMSE" not in df.columns:
        print(f"[WARN] {P_PER_YEAR_METRICS} にRMSE列がありません")
        return pd.DataFrame()
    
    return df

def load_cv_metrics() -> dict:
    """CVメトリクスを読み込み"""
    if not Path(P_CV_METRICS).exists():
        print(f"[WARN] {P_CV_METRICS} が見つかりません")
        return {}
    
    with open(P_CV_METRICS, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_rmse_for_year(per_year_df: pd.DataFrame, target_year: int) -> Optional[float]:
    """指定年のRMSEを取得"""
    if per_year_df.empty:
        return None
    
    # 完全一致を探す
    exact_match = per_year_df[per_year_df["year"] == target_year]
    if len(exact_match) > 0:
        return exact_match["RMSE"].iloc[0]
    
    # 最近接年を探す
    per_year_df = per_year_df.dropna(subset=["RMSE"])
    if len(per_year_df) == 0:
        return None
    
    # 年差の絶対値でソート
    per_year_df = per_year_df.copy()
    per_year_df["year_diff"] = np.abs(per_year_df["year"] - target_year)
    closest = per_year_df.loc[per_year_df["year_diff"].idxmin()]
    
    print(f"[L5] 年 {target_year} のRMSEが見つからないため、最近接年 {closest['year']} の値 {closest['RMSE']:.2f} を使用")
    return closest["RMSE"]

def get_aggregate_rmse(cv_metrics: dict) -> Optional[float]:
    """CVメトリクスから集約RMSEを取得"""
    if not cv_metrics or "aggregate" not in cv_metrics:
        return None
    
    aggregate = cv_metrics["aggregate"]
    if "RMSE" in aggregate:
        return aggregate["RMSE"]
    
    return None

def calculate_prediction_interval(prediction: float, rmse: float, horizon: int) -> Tuple[float, float]:
    """予測区間の計算"""
    # h>1 の場合は √h 倍の拡大
    sigma = rmse * np.sqrt(horizon)
    
    # 95%信頼区間（1.96σ）
    margin = 1.96 * sigma
    
    lower = prediction - margin
    upper = prediction + margin
    
    return lower, upper

def pi95_delta(year: int, per_year_df: pd.DataFrame, cv_metrics: dict) -> Tuple[float, float]:
    """Δ用の予測区間"""
    rmse = find_rmse_for_year(per_year_df, year)
    if rmse is None:
        rmse = get_aggregate_rmse(cv_metrics)
    if rmse is None:
        rmse = 50.0  # デフォルト値
    
    sigma = rmse
    margin = 1.96 * sigma
    return (-margin, margin)  # Δ用は±の範囲

def pi95_pop(pop_hat: float, year: int, horizon: int, per_year_df: pd.DataFrame, cv_metrics: dict) -> Tuple[float, float]:
    """人口用の予測区間"""
    rmse = find_rmse_for_year(per_year_df, year)
    if rmse is None:
        rmse = get_aggregate_rmse(cv_metrics)
    if rmse is None:
        rmse = 50.0  # デフォルト値
    
    sigma = rmse * np.sqrt(horizon)  # 独立残差近似
    margin = 1.96 * sigma
    return (pop_hat - margin, pop_hat + margin)

def estimate_intervals(predictions: List[float], years: List[int], 
                      base_year: int) -> List[Tuple[float, float]]:
    """予測区間の推定"""
    print("[L5] 予測区間を推定中...")
    
    # 年別メトリクスの読み込み
    per_year_df = load_per_year_metrics()
    
    # CVメトリクスの読み込み
    cv_metrics = load_cv_metrics()
    
    intervals = []
    
    for i, (pred, year) in enumerate(zip(predictions, years)):
        horizon = year - base_year
        
        # 該当年のRMSEを取得
        rmse = find_rmse_for_year(per_year_df, year)
        
        # 年別RMSEが取得できない場合は集約RMSEを使用
        if rmse is None:
            rmse = get_aggregate_rmse(cv_metrics)
            if rmse is not None:
                print(f"[L5] 年別RMSEが取得できないため、集約RMSE {rmse:.2f} を使用")
        
        # RMSEが全く取得できない場合はデフォルト値を使用
        if rmse is None:
            rmse = 50.0  # デフォルト値（適宜調整）
            print(f"[WARN] RMSEが取得できないため、デフォルト値 {rmse} を使用")
        
        # 予測区間の計算
        lower, upper = calculate_prediction_interval(pred, rmse, horizon)
        intervals.append((lower, upper))
        
        print(f"  年 {year} (h={horizon}): RMSE={rmse:.2f}, PI95=[{lower:.1f}, {upper:.1f}]")
    
    return intervals

def main(predictions: List[float], years: List[int], base_year: int) -> List[Tuple[float, float]]:
    """メイン処理"""
    return estimate_intervals(predictions, years, base_year)

if __name__ == "__main__":
    # テスト用
    test_predictions = [45.2, 32.5, 12.0]
    test_years = [2026, 2027, 2028]
    test_base_year = 2025
    
    intervals = main(test_predictions, test_years, test_base_year)
    print(f"予測区間: {intervals}")
