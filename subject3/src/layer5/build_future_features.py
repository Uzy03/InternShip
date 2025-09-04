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

def calculate_expected_effects(future_events: pd.DataFrame, effects_coef: pd.DataFrame, 
                             manual_delta: Dict[str, float], base_year: int) -> pd.DataFrame:
    """期待効果の計算（hごとに正しい年配置）"""
    result = future_events[["town", "year"]].copy()
    
    # 各年でexp_all_h{h}を初期化
    for horizon in [1, 2, 3]:
        result[f"exp_all_h{horizon}"] = 0.0
    
    # 各年ごとに期待効果を計算
    for i, (_, row) in enumerate(future_events.iterrows()):
        year = row["year"]
        horizon = year - base_year
        
        if horizon not in [1, 2, 3]:
            continue
        
        exp_col = f"exp_all_h{horizon}"
        exp_value = 0.0
        
        # 各イベントタイプの効果を計算
        for event_type in EVENT_TYPES:
            # 係数の取得
            inc_coef = effects_coef[effects_coef["event_var"] == f"{event_type}_inc"]
            dec_coef = effects_coef[effects_coef["event_var"] == f"{event_type}_dec"]
            
            if len(inc_coef) == 0 and len(dec_coef) == 0:
                continue
            
            # 仕様通りの年割付
            if horizon == 1:
                # h=1: Σ v_t(type) * coef(type,h1)
                event_t_col = f"event_{event_type}_t"
                if event_t_col in future_events.columns and not pd.isna(row[event_t_col]):
                    if len(inc_coef) > 0:
                        inc_effect = row[event_t_col] * inc_coef["beta_t"].iloc[0]
                        exp_value += inc_effect
                    
                    if len(dec_coef) > 0:
                        dec_effect = row[event_t_col] * dec_coef["beta_t"].iloc[0]
                        exp_value += dec_effect
            
            elif horizon == 2:
                # h=2: Σ v_t(type) * coef(type,h2) + Σ v_t1(type) * coef(type,h1)
                # v_t効果
                event_t_col = f"event_{event_type}_t"
                if event_t_col in future_events.columns and not pd.isna(row[event_t_col]):
                    if len(inc_coef) > 0:
                        inc_effect = row[event_t_col] * inc_coef["beta_t"].iloc[0]
                        exp_value += inc_effect
                    
                    if len(dec_coef) > 0:
                        dec_effect = row[event_t_col] * dec_coef["beta_t"].iloc[0]
                        exp_value += dec_effect
                
                # v_t1効果（前年のt1が当年のh2に効く）
                # 前年の行を取得
                prev_year = year - 1
                prev_row = future_events[future_events["year"] == prev_year]
                if len(prev_row) > 0:
                    prev_row = prev_row.iloc[0]
                    event_t1_col = f"event_{event_type}_t1"
                    if event_t1_col in future_events.columns and not pd.isna(prev_row[event_t1_col]):
                        if len(inc_coef) > 0:
                            inc_effect = prev_row[event_t1_col] * inc_coef["beta_t1"].iloc[0]
                            exp_value += inc_effect
                        
                        if len(dec_coef) > 0:
                            dec_effect = prev_row[event_t1_col] * dec_coef["beta_t1"].iloc[0]
                            exp_value += dec_effect
            
            elif horizon == 3:
                # h=3: Σ v_t(type) * coef(type,h3) + Σ v_t1(type) * coef(type,h2)
                # v_t効果
                event_t_col = f"event_{event_type}_t"
                if event_t_col in future_events.columns and not pd.isna(row[event_t_col]):
                    if len(inc_coef) > 0:
                        inc_effect = row[event_t_col] * inc_coef["beta_t"].iloc[0]
                        exp_value += inc_effect
                    
                    if len(dec_coef) > 0:
                        dec_effect = row[event_t_col] * dec_coef["beta_t"].iloc[0]
                        exp_value += dec_effect
                
                # v_t1効果（前年のt1が当年のh3に効く）
                # 前年の行を取得
                prev_year = year - 1
                prev_row = future_events[future_events["year"] == prev_year]
                if len(prev_row) > 0:
                    prev_row = prev_row.iloc[0]
                    event_t1_col = f"event_{event_type}_t1"
                    if event_t1_col in future_events.columns and not pd.isna(prev_row[event_t1_col]):
                        if len(inc_coef) > 0:
                            inc_effect = prev_row[event_t1_col] * inc_coef["beta_t1"].iloc[0]
                            exp_value += inc_effect
                        
                        if len(dec_coef) > 0:
                            dec_effect = prev_row[event_t1_col] * dec_coef["beta_t1"].iloc[0]
                            exp_value += dec_effect
        
        # 減衰率の適用（h2, h3）
        if horizon == 2:
            exp_value *= 0.5  # DECAY_H2
        elif horizon == 3:
            exp_value *= 0.25  # DECAY_H3
        
        # manual_deltaの加算
        manual_key = f"h{horizon}"
        if manual_key in manual_delta:
            exp_value += manual_delta[manual_key]
        
        result.at[i, exp_col] = exp_value
    
    # post-2022相互作用の計算
    for horizon in [1, 2, 3]:
        exp_col = f"exp_all_h{horizon}"
        post_col = f"exp_all_h{horizon}_post2022"
        result[post_col] = 0.0
        
        for i, (_, row) in enumerate(future_events.iterrows()):
            year = row["year"]
            if year >= 2023:  # post-2022
                result.at[i, post_col] = result.at[i, exp_col]
    
    # デバッグ出力: 各年のexp_all_h1/2/3を表示
    print("[L5] 期待効果の健全性チェック:")
    for i, (_, row) in enumerate(future_events.iterrows()):
        year = row["year"]
        horizon = year - base_year
        if horizon in [1, 2, 3]:
            exp_h1 = result.at[i, "exp_all_h1"]
            exp_h2 = result.at[i, "exp_all_h2"]
            exp_h3 = result.at[i, "exp_all_h3"]
            print(f"  年 {year} (h={horizon}): exp_all_h1={exp_h1:.2f}, exp_all_h2={exp_h2:.2f}, exp_all_h3={exp_h3:.2f}")
    
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

def add_missing_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """不足している特徴を追加（NaNで埋める）"""
    result = features_df.copy()
    
    # L4で使用されたが不足している特徴
    missing_features = [
        "exp_all_h1_pre2013", "exp_all_h1_post2013", "exp_all_h1_post2009",
        "exp_all_h2_pre2013", "exp_all_h2_post2013", "exp_all_h2_post2009", 
        "exp_all_h3_pre2013", "exp_all_h3_post2013", "exp_all_h3_post2009",
        "lag_d1", "lag_d2", "ma2_delta",
        "macro_delta", "macro_ma3", "macro_shock", "macro_excl"
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
    
    # 効果係数の読み込み
    effects_coef = load_effects_coefficients()
    
    # 期待効果の計算
    base_year = scenario.get("base_year", 2025)
    expected_effects = calculate_expected_effects(future_events, effects_coef, 
                                                scenario.get("manual_delta", {}), base_year)
    
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
    
    # 期待効果の確認
    for horizon in [1, 2, 3]:
        exp_col = f"exp_all_h{horizon}"
        if exp_col in future_features.columns:
            non_zero = (future_features[exp_col] != 0).sum()
            print(f"  {exp_col}: {non_zero} 行が非ゼロ")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("使用方法: python build_future_features.py <baseline.csv> <future_events.csv> <scenario.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
