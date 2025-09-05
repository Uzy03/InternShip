# -*- coding: utf-8 -*-
# src/layer5/build_future_features.py
"""
将来特徴の合成：L4互換の特徴合成（期待効果・マクロ・時代フラグ）
出力: data/processed/l5_future_features.csv

設計:
- 入力: prepare_baseline()のベース1行, l5_future_events.csv, effects_coefficients.csv, features_panel.csv, シナリオのmacros/manual_delta
- 方針: L4の合成ロジックを"将来年に対して"トレース。補間しない（NaNはNaNのまま・∞→NaNのみ置換）
- 期待効果: effects_coefficients.csvの係数を使用
- マクロ: 外国人人口の成長率を適用
- 時代フラグ: era_covid, era_post2022等を将来年にも適用
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json

# パス設定
P_FEATURES_PANEL = "../../data/processed/features_panel.csv"
P_FUTURE_EVENTS = "../../data/processed/l5_future_events.csv"
P_EFFECTS_COEF = "../../output/effects_coefficients.csv"
P_EFFECTS_COEF_RATE = "../../output/effects_coefficients_rate.csv"
P_OUTPUT = "../../data/processed/l5_future_features.csv"

# イベントタイプの定義
EVENT_TYPES = {
    "housing", "commercial", "transit", "policy_boundary", 
    "public_edu_medical", "employment", "disaster"
}

def load_effects_coefficients() -> pd.DataFrame:
    """効果係数を読み込み"""
    coef_df = pd.read_csv(P_EFFECTS_COEF)
    
    # 列名の確認と調整
    required_cols = ["event_var", "beta_t", "beta_t1"]
    missing_cols = [col for col in required_cols if col not in coef_df.columns]
    if missing_cols:
        raise ValueError(f"effects_coefficients.csv に列不足: {missing_cols}")
    
    return coef_df

def load_effects_coefficients_rate() -> pd.DataFrame:
    """率効果係数を読み込み"""
    if not Path(P_EFFECTS_COEF_RATE).exists():
        raise FileNotFoundError(f"率効果係数ファイルが見つかりません: {P_EFFECTS_COEF_RATE}")
    
    coef_df = pd.read_csv(P_EFFECTS_COEF_RATE)
    
    # 列名の確認と調整
    required_cols = ["event_var", "beta"]
    missing_cols = [col for col in required_cols if col not in coef_df.columns]
    if missing_cols:
        raise ValueError(f"effects_coefficients_rate.csv に列不足: {missing_cols}")
    
    return coef_df

def coef_rate(event_type: str, direction: str, horizon: int, coef_df: pd.DataFrame) -> float:
    """率係数を取得（単位統一）"""
    event_var = f"{event_type}_{direction}"
    r = coef_df[coef_df["event_var"] == event_var]
    if r.empty:
        return 0.0
    
    v = float(r["beta"].iloc[0])
    
    # 係数の単位を確認（%表記なら小数に変換）
    COEF_IS_PERCENT = True  # effects_coefficients_rate.csvが%表記の場合True
    
    if COEF_IS_PERCENT:
        v = v / 100.0  # % → 小数
    
    # デバッグログ
    print(f"[L5] coef_rate({event_type}_{direction}, h={horizon})={v:.6f}")
    
    return v

def calculate_expected_effects_rate(future_events: pd.DataFrame, effects_coef_rate: pd.DataFrame, 
                                   manual_delta: Dict[str, float], manual_delta_rate: Dict[str, float],
                                   base_population: float, base_year: int) -> pd.DataFrame:
    """期待効果の計算（率ベース、hごとに正しい年配置）"""
    result = future_events[["town", "year"]].copy()
    
    # 各年でexp_rate_all_h{h}を初期化
    for horizon in [1, 2, 3]:
        result[f"exp_rate_all_h{horizon}"] = 0.0
    
    # 各年ごとに期待効果を計算（率ベース）
    for i, (_, row) in enumerate(future_events.iterrows()):
        year = row["year"]
        horizon = year - base_year
        
        if horizon not in [1, 2, 3]:
            continue
        
        exp_col = f"exp_rate_all_h{horizon}"
        exp_rate = 0.0
        
        # 各イベントタイプの効果を計算（率ベース）
        for event_type in EVENT_TYPES:
            # 率係数の取得
            inc_coef = coef_rate(event_type, "inc", horizon, effects_coef_rate)
            dec_coef = coef_rate(event_type, "dec", horizon, effects_coef_rate)
            
            # 仕様通りの年割付（率ベース）
            if horizon == 1:
                # h=1: Σ v_t(type) * coef_rate(type,inc/dec)
                event_t_col = f"event_{event_type}_t"
                if event_t_col in future_events.columns and not pd.isna(row[event_t_col]):
                    v_t = float(row[event_t_col])
                    exp_rate += v_t * inc_coef  # increase効果
                    exp_rate += v_t * dec_coef  # decrease効果
            
            elif horizon == 2:
                # h=2: Σ v_t(type) * coef_rate(type,h2) + Σ v_t1(type) * coef_rate(type,h1)
                # v_t効果
                event_t_col = f"event_{event_type}_t"
                if event_t_col in future_events.columns and not pd.isna(row[event_t_col]):
                    v_t = float(row[event_t_col])
                    exp_rate += v_t * inc_coef  # increase効果
                    exp_rate += v_t * dec_coef  # decrease効果
                
                # v_t1効果（前年のt1が当年のh2に効く）
                prev_year = year - 1
                prev_row = future_events[future_events["year"] == prev_year]
                if len(prev_row) > 0:
                    prev_row = prev_row.iloc[0]
                    event_t1_col = f"event_{event_type}_t1"
                    if event_t1_col in future_events.columns and not pd.isna(prev_row[event_t1_col]):
                        v_t1 = float(prev_row[event_t1_col])
                        # h=1の係数を使用
                        inc_coef_h1 = coef_rate(event_type, "inc", 1, effects_coef_rate)
                        dec_coef_h1 = coef_rate(event_type, "dec", 1, effects_coef_rate)
                        exp_rate += v_t1 * inc_coef_h1  # increase効果
                        exp_rate += v_t1 * dec_coef_h1  # decrease効果
            
            elif horizon == 3:
                # h=3: Σ v_t(type) * coef_rate(type,h3) + Σ v_t1(type) * coef_rate(type,h2)
                # v_t効果
                event_t_col = f"event_{event_type}_t"
                if event_t_col in future_events.columns and not pd.isna(row[event_t_col]):
                    v_t = float(row[event_t_col])
                    exp_rate += v_t * inc_coef  # increase効果
                    exp_rate += v_t * dec_coef  # decrease効果
                
                # v_t1効果（前年のt1が当年のh3に効く）
                prev_year = year - 1
                prev_row = future_events[future_events["year"] == prev_year]
                if len(prev_row) > 0:
                    prev_row = prev_row.iloc[0]
                    event_t1_col = f"event_{event_type}_t1"
                    if event_t1_col in future_events.columns and not pd.isna(prev_row[event_t1_col]):
                        v_t1 = float(prev_row[event_t1_col])
                        # h=2の係数を使用
                        inc_coef_h2 = coef_rate(event_type, "inc", 2, effects_coef_rate)
                        dec_coef_h2 = coef_rate(event_type, "dec", 2, effects_coef_rate)
                        exp_rate += v_t1 * inc_coef_h2  # increase効果
                        exp_rate += v_t1 * dec_coef_h2  # decrease効果
        
        # manual_deltaの加算（率化）
        manual_key = f"h{horizon}"
        add_rate = 0.0
        
        # manual_delta_rate（% → 小数）
        if manual_key in manual_delta_rate:
            add_rate += float(manual_delta_rate[manual_key]) / 100.0
        
        # manual_delta（人数 → 率）
        manual_people = 0.0
        if manual_key in manual_delta:
            manual_people = float(manual_delta[manual_key])
            add_rate += manual_people / max(base_population, 1.0)
        
        exp_rate += add_rate
        
        # 安全弁：レートをクリップ（±100%以内）
        def clip_rate(x, lim=1.0):
            return max(-lim, min(lim, x))
        
        safe_rate = clip_rate(exp_rate, 1.0)
        if safe_rate != exp_rate:
            print(f"[L5][WARN] exp_rate clipped in build: {exp_rate:.4f} -> {safe_rate:.4f} (year={year}, h={horizon})")
        
        result.at[i, exp_col] = safe_rate
        
        # 手動人数を別列に保存（デバッグ専用）
        manual_people_col = f"manual_people_h{horizon}"
        result[manual_people_col] = 0.0
        result.at[i, manual_people_col] = manual_people
    
    # post-2022相互作用の計算（率ベース）
    for horizon in [1, 2, 3]:
        exp_col = f"exp_rate_all_h{horizon}"
        post_col = f"exp_rate_all_h{horizon}_post2022"
        result[post_col] = 0.0
        
        for i, (_, row) in enumerate(future_events.iterrows()):
            year = row["year"]
            if year >= 2023:  # post-2022
                result.at[i, post_col] = result.at[i, exp_col]
    
    # デバッグ出力: 各年のexp_rate_all_h1/2/3を表示
    print("[L5] 期待効果の健全性チェック（率ベース）:")
    for i, (_, row) in enumerate(future_events.iterrows()):
        year = row["year"]
        horizon = year - base_year
        if horizon in [1, 2, 3]:
            exp_h1 = result.at[i, "exp_rate_all_h1"]
            exp_h2 = result.at[i, "exp_rate_all_h2"]
            exp_h3 = result.at[i, "exp_rate_all_h3"]
            print(f"  年 {year} (h={horizon}): exp_rate_all_h1={exp_h1:.4f}, exp_rate_all_h2={exp_h2:.4f}, exp_rate_all_h3={exp_h3:.4f}")
    
    return result

def calculate_macro_features(future_events: pd.DataFrame, baseline: pd.DataFrame, 
                           macros: Dict[str, Any]) -> pd.DataFrame:
    """マクロ特徴の計算（主に外国人人口）"""
    result = future_events[["town", "year"]].copy()
    
    # ベース年の外国人人口を取得
    foreign_base = None
    if "foreign_population" in baseline.columns:
        foreign_base = baseline["foreign_population"].iloc[0]
    
    if pd.isna(foreign_base):
        print("[WARN] ベース年の外国人人口が不明のため、マクロ特徴を計算できません")
        for col in ["foreign_population", "foreign_change", "foreign_pct_change", 
                   "foreign_log", "foreign_ma3"]:
            result[col] = np.nan
        return result
    
    # 外国人人口成長率の取得
    growth_rates = {}
    if "macros" in macros and "foreign_population_growth_pct" in macros["macros"]:
        growth_rates = macros["macros"]["foreign_population_growth_pct"]
    
    # 各年の外国人人口を計算
    foreign_pop = foreign_base
    result["foreign_population"] = foreign_base
    
    for i, row in future_events.iterrows():
        year = row["year"]
        base_year = baseline["year"].iloc[0]
        year_offset = year - base_year
        
        if year_offset > 0:
            growth_key = f"h{year_offset}"
            if growth_key in growth_rates:
                growth_rate = growth_rates[growth_key]
                foreign_pop = foreign_pop * (1 + growth_rate)
                result.loc[i, "foreign_population"] = foreign_pop
            else:
                result.loc[i, "foreign_population"] = np.nan
    
    # 派生特徴の計算
    result["foreign_change"] = result["foreign_population"].diff()
    result["foreign_pct_change"] = result["foreign_change"] / result["foreign_population"].shift(1)
    result["foreign_log"] = np.log(np.maximum(result["foreign_population"], 1))
    
    # 移動平均（直近3年）
    result["foreign_ma3"] = result["foreign_population"].rolling(window=3, min_periods=1).mean()
    
    return result

def calculate_era_features(future_events: pd.DataFrame) -> pd.DataFrame:
    """時代フラグの計算"""
    result = future_events[["town", "year"]].copy()
    
    # era_covid: 2020-2022
    result["era_covid"] = ((result["year"] >= 2020) & (result["year"] <= 2022)).astype(int)
    
    # era_post2022: 2023以降
    result["era_post2022"] = (result["year"] >= 2023).astype(int)
    
    # era_pre2013: 2013以前
    result["era_pre2013"] = (result["year"] <= 2013).astype(int)
    
    # era_post2009: 2009以降
    result["era_post2009"] = (result["year"] >= 2009).astype(int)
    
    # era_post2013: 2013以降
    result["era_post2013"] = (result["year"] >= 2013).astype(int)
    
    return result

def calculate_interaction_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """相互作用特徴の計算"""
    result = features_df.copy()
    
    # 外国人マクロ × COVID期
    if "foreign_population" in result.columns and "era_covid" in result.columns:
        result["foreign_population_covid"] = result["foreign_population"] * result["era_covid"]
        result["foreign_change_covid"] = result["foreign_change"] * result["era_covid"]
        result["foreign_pct_change_covid"] = result["foreign_pct_change"] * result["era_covid"]
        result["foreign_log_covid"] = result["foreign_log"] * result["era_covid"]
        result["foreign_ma3_covid"] = result["foreign_ma3"] * result["era_covid"]
    
    # 外国人マクロ × post2022
    if "foreign_population" in result.columns and "era_post2022" in result.columns:
        result["foreign_population_post2022"] = result["foreign_population"] * result["era_post2022"]
        result["foreign_change_post2022"] = result["foreign_change"] * result["era_post2022"]
        result["foreign_pct_change_post2022"] = result["foreign_pct_change"] * result["era_post2022"]
        result["foreign_log_post2022"] = result["foreign_log"] * result["era_post2022"]
        result["foreign_ma3_post2022"] = result["foreign_ma3"] * result["era_post2022"]
    
    # 期待効果 × 時代フラグ
    for horizon in [1, 2, 3]:
        exp_col = f"exp_all_h{horizon}"
        if exp_col in result.columns:
            if "era_covid" in result.columns:
                result[f"{exp_col}_covid"] = result[exp_col] * result["era_covid"]
            if "era_post2022" in result.columns:
                result[f"{exp_col}_post2022"] = result[exp_col] * result["era_post2022"]
    
    return result

def carry_forward_features(future_events: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """ベースラインから将来年への特徴の持ち越し"""
    result = future_events[["town", "year"]].copy()
    
    # 持ち越す特徴（将来更新しない）
    carry_forward_cols = [
        "town_trend5", "pop_total", "male", "female", "city_pop", "city_growth_log",
        "town_ma5", "town_std5"  # 追加
    ]
    
    for col in carry_forward_cols:
        if col in baseline.columns:
            baseline_value = baseline[col].iloc[0]
            result[col] = baseline_value
        else:
            result[col] = np.nan
    
    return result

def calculate_individual_expected_effects(features_df: pd.DataFrame, effects_coef_rate: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """個別カテゴリの期待効果を計算（L4互換）"""
    result = features_df.copy()
    
    # 各イベントタイプと方向の組み合わせで期待効果を計算
    for event_type in EVENT_TYPES:
        for direction in ['inc', 'dec']:
            event_var = f"{event_type}_{direction}"
            
            # 率係数を取得
            inc_coef = coef_rate(event_type, "inc", 1, effects_coef_rate)
            dec_coef = coef_rate(event_type, "dec", 1, effects_coef_rate)
            
            # 各horizonで期待効果を計算
            for horizon in [1, 2, 3]:
                exp_col = f"exp_{event_var}_h{horizon}"
                result[exp_col] = 0.0
                
                # 減衰率を適用
                decay = 1.0
                if horizon == 2:
                    decay = 0.5
                elif horizon == 3:
                    decay = 0.25
                
                # 率ベースの期待効果を計算（簡略化）
                if direction == "inc":
                    result[exp_col] = inc_coef * decay
                else:
                    result[exp_col] = dec_coef * decay
    
    return result

def generate_legacy_exp_columns(features_df: pd.DataFrame) -> pd.DataFrame:
    """従来のexp_all_h*列を生成（後方互換性）"""
    result = features_df.copy()
    
    # exp_rate_all_h*からexp_all_h*を生成（人数ベースに変換）
    for horizon in [1, 2, 3]:
        rate_col = f"exp_rate_all_h{horizon}"
        legacy_col = f"exp_all_h{horizon}"
        
        if rate_col in result.columns:
            # 率を人数に変換（簡易的に1000人を基準とする）
            result[legacy_col] = result[rate_col] * 1000.0
        else:
            result[legacy_col] = 0.0
    
    return result

def calculate_regime_interactions(features_df: pd.DataFrame) -> pd.DataFrame:
    """レジーム相互作用を計算（L4互換）"""
    result = features_df.copy()
    
    # レジームフラグを追加
    result["era_post2009"] = (result["year"] >= 2009).astype(int)
    result["era_post2013"] = (result["year"] >= 2013).astype(int)
    result["era_pre2013"] = (result["year"] < 2013).astype(int)
    result["era_covid"] = ((result["year"] >= 2020) & (result["year"] <= 2022)).astype(int)
    result["era_post2022"] = (result["year"] >= 2023).astype(int)
    
    # 期待効果とレジームの相互作用
    for horizon in [1, 2, 3]:
        # 率ベース
        rate_col = f"exp_rate_all_h{horizon}"
        if rate_col in result.columns:
            result[f"{rate_col}_pre2013"] = result[rate_col] * result["era_pre2013"]
            result[f"{rate_col}_post2013"] = result[rate_col] * result["era_post2013"]
            result[f"{rate_col}_post2009"] = result[rate_col] * result["era_post2009"]
            result[f"{rate_col}_covid"] = result[rate_col] * result["era_covid"]
            result[f"{rate_col}_post2022"] = result[rate_col] * result["era_post2022"]
        
        # 従来の人数ベース
        legacy_col = f"exp_all_h{horizon}"
        if legacy_col in result.columns:
            result[f"{legacy_col}_pre2013"] = result[legacy_col] * result["era_pre2013"]
            result[f"{legacy_col}_post2013"] = result[legacy_col] * result["era_post2013"]
            result[f"{legacy_col}_post2009"] = result[legacy_col] * result["era_post2009"]
            result[f"{legacy_col}_covid"] = result[legacy_col] * result["era_covid"]
            result[f"{legacy_col}_post2022"] = result[legacy_col] * result["era_post2022"]
    
    return result

def calculate_macro_features_l4_style(features_df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """マクロ特徴を計算（L4互換）"""
    result = features_df.copy()
    
    # 簡易的なマクロ特徴（将来年では実際の値は不明）
    result["macro_delta"] = np.nan
    result["macro_ma3"] = np.nan
    result["macro_shock"] = np.nan
    result["macro_excl"] = np.nan
    
    return result

def calculate_town_trend_features(features_df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """町丁トレンド特徴を計算（L4互換）"""
    result = features_df.copy()
    
    # ベースラインから町丁トレンド特徴を取得
    if "town_trend5" in baseline.columns:
        result["town_trend5"] = baseline["town_trend5"].iloc[0]
    else:
        result["town_trend5"] = np.nan
    
    if "town_ma5" in baseline.columns:
        result["town_ma5"] = baseline["town_ma5"].iloc[0]
    else:
        result["town_ma5"] = np.nan
    
    if "town_std5" in baseline.columns:
        result["town_std5"] = baseline["town_std5"].iloc[0]
    else:
        result["town_std5"] = np.nan
    
    # ラグ特徴（将来年では不明）
    result["lag_d1"] = np.nan
    result["lag_d2"] = np.nan
    result["ma2_delta"] = np.nan
    
    return result

def add_missing_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """不足している特徴を追加（L4互換）"""
    result = features_df.copy()
    
    # L4で使用されたが不足している特徴（率ベース対応）
    missing_features = [
        # レジーム相互作用（率ベース）
        "exp_rate_all_h1_pre2013", "exp_rate_all_h1_post2013", "exp_rate_all_h1_post2009",
        "exp_rate_all_h2_pre2013", "exp_rate_all_h2_post2013", "exp_rate_all_h2_post2009", 
        "exp_rate_all_h3_pre2013", "exp_rate_all_h3_post2013", "exp_rate_all_h3_post2009",
        "exp_rate_all_h1_covid", "exp_rate_all_h2_covid", "exp_rate_all_h3_covid",
        # ラグ特徴
        "lag_d1", "lag_d2", "ma2_delta",
        # マクロ特徴
        "macro_delta", "macro_ma3", "macro_shock", "macro_excl",
        # 町丁トレンド
        "town_trend5", "town_ma5", "town_std5",
        # 個別カテゴリの期待効果（率ベース）
        "exp_housing_inc_h1", "exp_housing_inc_h2", "exp_housing_inc_h3",
        "exp_housing_dec_h1", "exp_housing_dec_h2", "exp_housing_dec_h3",
        "exp_commercial_inc_h1", "exp_commercial_inc_h2", "exp_commercial_inc_h3",
        "exp_commercial_dec_h1", "exp_commercial_dec_h2", "exp_commercial_dec_h3",
        "exp_transit_inc_h1", "exp_transit_inc_h2", "exp_transit_inc_h3",
        "exp_transit_dec_h1", "exp_transit_dec_h2", "exp_transit_dec_h3",
        "exp_public_edu_medical_inc_h1", "exp_public_edu_medical_inc_h2", "exp_public_edu_medical_inc_h3",
        "exp_public_edu_medical_dec_h1", "exp_public_edu_medical_dec_h2", "exp_public_edu_medical_dec_h3",
        "exp_employment_inc_h1", "exp_employment_inc_h2", "exp_employment_inc_h3",
        "exp_employment_dec_h1", "exp_employment_dec_h2", "exp_employment_dec_h3",
        "exp_disaster_inc_h1", "exp_disaster_inc_h2", "exp_disaster_inc_h3",
        "exp_disaster_dec_h1", "exp_disaster_dec_h2", "exp_disaster_dec_h3",
        # 従来のexp_all_h*（後方互換性のため）
        "exp_all_h1", "exp_all_h2", "exp_all_h3",
        "exp_all_h1_pre2013", "exp_all_h1_post2013", "exp_all_h1_post2009",
        "exp_all_h2_pre2013", "exp_all_h2_post2013", "exp_all_h2_post2009", 
        "exp_all_h3_pre2013", "exp_all_h3_post2013", "exp_all_h3_post2009",
        "exp_all_h1_covid", "exp_all_h2_covid", "exp_all_h3_covid",
        "exp_all_h1_post2022", "exp_all_h2_post2022", "exp_all_h3_post2022"
    ]
    
    for feature in missing_features:
        if feature not in result.columns:
            result[feature] = np.nan
    
    return result

def build_future_features(baseline: pd.DataFrame, future_events: pd.DataFrame, 
                         scenario: Dict[str, Any]) -> pd.DataFrame:
    """将来特徴の合成"""
    print("[L5] 将来特徴を構築中...")
    
    # シナリオで指定された町丁のみを処理
    target_town = scenario.get("town")
    if target_town:
        future_events = future_events[future_events["town"] == target_town].copy()
        print(f"[L5] 町丁 '{target_town}' のデータを処理中...")
    
    # 効果係数の読み込み（率ベース）
    effects_coef_rate = load_effects_coefficients_rate()
    
    # 期待効果の計算（率ベース）
    base_year = scenario.get("base_year", 2025)
    base_population = float(baseline["pop_total"].iloc[0])
    expected_effects = calculate_expected_effects_rate(
        future_events, effects_coef_rate, 
        scenario.get("manual_delta", {}),
        scenario.get("manual_delta_rate", {}),
        base_population, base_year
    )
    
    # マクロ特徴の計算
    macro_features = calculate_macro_features(future_events, baseline, scenario)
    
    # 時代フラグの計算
    era_features = calculate_era_features(future_events)
    
    # 特徴の持ち越し
    carry_features = carry_forward_features(future_events, baseline)
    
    # 特徴の結合
    features_df = future_events[["town", "year"]].copy()
    
    # 各特徴群を結合
    for df in [expected_effects, macro_features, era_features, carry_features]:
        for col in df.columns:
            if col not in ["town", "year"]:
                features_df[col] = df[col]
    
    # 相互作用特徴の計算
    features_df = calculate_interaction_features(features_df)
    
    # 個別カテゴリの期待効果を計算（L4互換）
    features_df = calculate_individual_expected_effects(features_df, effects_coef_rate, base_year)
    
    # 従来のexp_all_h*列を生成（後方互換性）
    features_df = generate_legacy_exp_columns(features_df)
    
    # レジーム相互作用を計算（L4互換）
    features_df = calculate_regime_interactions(features_df)
    
    # マクロ特徴を計算（L4互換）
    features_df = calculate_macro_features_l4_style(features_df, base_year)
    
    # 町丁トレンド特徴を計算（L4互換）
    features_df = calculate_town_trend_features(features_df, baseline)
    
    # 不足している特徴を追加
    features_df = add_missing_features(features_df)
    
    # ∞をNaNに置換
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    return features_df

def main(baseline_path: str, future_events_path: str, scenario_path: str) -> None:
    """メイン処理"""
    # データの読み込み
    baseline = pd.read_csv(baseline_path)
    future_events = pd.read_csv(future_events_path)
    
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)
    
    # 将来特徴の構築
    future_features = build_future_features(baseline, future_events, scenario)
    
    # 出力ディレクトリの作成
    Path(P_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    future_features.to_csv(P_OUTPUT, index=False)
    print(f"[L5] 将来特徴を保存しました: {P_OUTPUT}")
    print(f"[L5] 行数: {len(future_features)}, 列数: {len(future_features.columns)}")
    
    # デバッグ情報
    non_nan_cols = future_features.columns[~future_features.isna().all()].tolist()
    print(f"[L5] 非NaN列数: {len(non_nan_cols)}")
    
    # 期待効果の確認（率ベース）
    for horizon in [1, 2, 3]:
        exp_col = f"exp_rate_all_h{horizon}"
        if exp_col in future_features.columns:
            non_zero = (future_features[exp_col] != 0).sum()
            print(f"  {exp_col}: {non_zero} 行が非ゼロ")
    
    # 健全性ログ（率ベース）
    print(f"[L5] exp_rate_all_h1/2/3 nonzero rows = "
          f"{(future_features['exp_rate_all_h1']!=0).sum()}/"
          f"{(future_features['exp_rate_all_h2']!=0).sum()}/"
          f"{(future_features['exp_rate_all_h3']!=0).sum()}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("使用方法: python build_future_features.py <baseline.csv> <future_events.csv> <scenario.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
