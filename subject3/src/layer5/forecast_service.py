# -*- coding: utf-8 -*-
# src/layer5/forecast_service.py
"""
予測本体：L4モデルに将来特徴を投入して人口予測と寄与分解を実行
出力: 予測結果JSON

設計:
- 入力: l4_model.joblib, l5_future_features.csv, ベース人口 pop_base
- 出力: JSON形式の予測結果
- アルゴリズム: L4モデルに l5_future_features を投入 → 各年の Δ人口（delta_hat） を得る
- 人口パス: pop_{t+1} = pop_t + delta_hat_{h1} → pop_{t+2} = pop_{t+1} + delta_hat_{h2} → …
- 寄与分解: exp, macro, inertia, other の簡易分解
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from intervals import estimate_intervals

# パス設定
P_MODEL = "../../models/l4_model.joblib"
P_FUTURE_FEATURES = "../../data/processed/l5_future_features.csv"
P_EFFECTS_COEF = "../../output/effects_coefficients.csv"

def load_model() -> Any:
    """L4モデルを読み込み"""
    if not Path(P_MODEL).exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {P_MODEL}")
    
    return joblib.load(P_MODEL)

def load_future_features() -> pd.DataFrame:
    """将来特徴を読み込み"""
    if not Path(P_FUTURE_FEATURES).exists():
        raise FileNotFoundError(f"将来特徴ファイルが見つかりません: {P_FUTURE_FEATURES}")
    
    return pd.read_csv(P_FUTURE_FEATURES)

def load_effects_coefficients() -> pd.DataFrame:
    """効果係数を読み込み"""
    if not Path(P_EFFECTS_COEF).exists():
        raise FileNotFoundError(f"効果係数ファイルが見つかりません: {P_EFFECTS_COEF}")
    
    return pd.read_csv(P_EFFECTS_COEF)

def choose_features(df: pd.DataFrame) -> List[str]:
    """L4の実際に使用された特徴を選択"""
    # L4で実際に使用された特徴リスト（l4_cv_metrics.jsonから）
    l4_features = [
        "pop_total",
        "exp_all_h1",
        "exp_all_h2", 
        "exp_all_h3",
        "era_post2009",
        "era_post2013",
        "era_pre2013",
        "exp_all_h1_pre2013",
        "exp_all_h1_post2013",
        "exp_all_h1_post2009",
        "exp_all_h2_pre2013",
        "exp_all_h2_post2013",
        "exp_all_h2_post2009",
        "exp_all_h3_pre2013",
        "exp_all_h3_post2013",
        "exp_all_h3_post2009",
        "lag_d1",
        "lag_d2",
        "ma2_delta",
        "era_covid",
        "macro_delta",
        "macro_ma3",
        "macro_shock",
        "town_trend5",
        "town_ma5",
        "town_std5",
        "macro_excl",
        "foreign_population",
        "foreign_change",
        "foreign_pct_change",
        "foreign_log",
        "foreign_ma3",
        "foreign_population_covid",
        "foreign_change_covid",
        "foreign_pct_change_covid",
        "foreign_log_covid",
        "foreign_ma3_covid",
        "era_post2022",
        "exp_all_h1_post2022",
        "exp_all_h2_post2022",
        "exp_all_h3_post2022",
        "foreign_population_post2022",
        "foreign_change_post2022",
        "foreign_pct_change_post2022",
        "foreign_log_post2022",
        "foreign_ma3_post2022"
    ]
    
    # データフレームに存在し、数値型の特徴のみを選択
    keep = []
    for feature in l4_features:
        if feature in df.columns and np.issubdtype(df[feature].dtype, np.number):
            keep.append(feature)
    
    return keep

def predict_delta_population_sequential(model: Any, features_df: pd.DataFrame, base_population: float) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    """人口変化量を逐次予測（ラグ更新付き）"""
    # 特徴選択
    feature_cols = choose_features(features_df)
    
    # 年順でソート
    features_df = features_df.sort_values("year").copy()
    
    # 予測結果の格納
    delta_predictions = []
    population_path = []
    contributions = []
    
    # 初期状態
    pop = base_population
    prev_deltas = []  # 直近のΔを先頭に積む
    
    # 使う列グループを定義
    EXP_COLS = [c for c in feature_cols if c.startswith("exp_all_h")]
    EXP_POST_COLS = [c for c in feature_cols if c.startswith("exp_all_h") and c.endswith("_post2022")]
    MACRO_COLS = [c for c in feature_cols if c.startswith(("foreign_", "macro_"))]
    INERTIA_COLS = [c for c in feature_cols if c.startswith(("lag_", "town_ma", "town_std", "town_trend"))]
    
    def zero_cols(Xrow, cols):
        X0 = Xrow.copy()
        for c in cols:
            if c in X0.columns:
                X0[c] = 0.0
        return X0
    
    for i, (_, row) in enumerate(features_df.iterrows()):
        # --- 直前の予測でラグ系を更新 ---
        if len(prev_deltas) >= 1 and "lag_d1" in features_df.columns:
            features_df.iloc[i, features_df.columns.get_loc("lag_d1")] = prev_deltas[0]
        if len(prev_deltas) >= 2 and "lag_d2" in features_df.columns:
            features_df.iloc[i, features_df.columns.get_loc("lag_d2")] = prev_deltas[1]
        if "ma2_delta" in features_df.columns:
            v = np.mean(prev_deltas[:2]) if len(prev_deltas) >= 1 else np.nan
            features_df.iloc[i, features_df.columns.get_loc("ma2_delta")] = v
        
        # 特徴行列の準備
        Xrow = features_df.iloc[i:i+1, features_df.columns.get_indexer(feature_cols)].copy()
        Xrow.columns = feature_cols
        Xrow = Xrow.replace([np.inf, -np.inf], np.nan)
        
        # ① GBM の「exp 抜き」予測
        X_noexp = zero_cols(Xrow.copy(), EXP_COLS + EXP_POST_COLS)
        y_noexp = float(model.predict(X_noexp)[0])
        
        # ② 期待効果（人ベース）をそのまま加算
        exp_terms = 0.0
        for c in EXP_COLS + EXP_POST_COLS:
            if c in Xrow.columns:
                val = Xrow[c].values[0]
                if not pd.isna(val):
                    exp_terms += float(val)
        
        
        # ③ 最終Δ予測 = 「GBM（exp抜き）」＋「期待効果」
        delta_hat = y_noexp + exp_terms
        
        # ④ 人口パス更新
        pop = pop + delta_hat
        prev_deltas.insert(0, delta_hat)
        prev_deltas = prev_deltas[:8]  # 必要分だけ保持（安全）
        
        # ⑤ 寄与分解（ノックアウト基準を y_noexp に合わせる）
        contrib = {}
        contrib["exp"] = exp_terms
        
        # macro 寄与 = y_noexp - y_without_macro
        y_wo_macro = float(model.predict(zero_cols(X_noexp.copy(), MACRO_COLS))[0])
        contrib["macro"] = y_noexp - y_wo_macro
        
        # inertia 寄与 = y_wo_macro - y_without_inertia
        y_wo_inertia = float(model.predict(zero_cols(X_noexp.copy(), INERTIA_COLS))[0])
        contrib["inertia"] = y_wo_macro - y_wo_inertia
        
        # other = 残差
        contrib["other"] = delta_hat - (contrib["exp"] + contrib["macro"] + contrib["inertia"])
        
        # 結果格納
        delta_predictions.append(delta_hat)
        population_path.append(pop)
        contributions.append(contrib)
    
    return delta_predictions, population_path, contributions

def calculate_contribution_knockout(model: Any, X: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """グループ・ノックアウトによる寄与分解"""
    # グループ定義（厳密化）
    GROUPS = {
        "exp": list(set([c for c in feature_cols if c.startswith("exp_all_h")] + [c for c in feature_cols if c.startswith("exp_all_h") and c.endswith("_post2022")])),
        "macro": [c for c in feature_cols if c.startswith(("foreign_", "macro_"))],
        "inertia": [c for c in feature_cols if c.startswith(("lag_", "town_ma", "town_std", "town_trend"))],
    }
    
    def predict_with_knockout(Xrow: pd.DataFrame, group_cols: List[str]) -> float:
        X0 = Xrow.copy()
        # 効果を取り除くので、イベント由来は0、連続特徴は基準値（安全に0）に
        for col in group_cols:
            if col in X0.columns:
                X0[col] = 0.0
        return float(model.predict(X0)[0])
    
    # 完全予測
    full = float(model.predict(X)[0])
    
    # 各グループの寄与を計算
    contrib = {}
    rem = full
    
    for group, cols in GROUPS.items():
        # 実際に存在する列のみを対象
        existing_cols = [c for c in cols if c in X.columns]
        if existing_cols:
            y_wo = predict_with_knockout(X.copy(), existing_cols)
            contrib[group] = full - y_wo
            rem -= contrib[group]
        else:
            contrib[group] = 0.0
    
    contrib["other"] = rem
    
    return contrib

def calculate_population_path(base_population: float, delta_predictions: List[float]) -> List[float]:
    """人口パスの計算"""
    population_path = [base_population]
    current_pop = base_population
    
    for delta in delta_predictions:
        current_pop += delta
        population_path.append(current_pop)
    
    return population_path[1:]  # ベース年を除く

def create_forecast_result(town: str, base_year: int, horizons: List[int], 
                          delta_predictions: List[float], population_path: List[float],
                          contributions: List[Dict[str, float]], 
                          prediction_intervals: List[Tuple[float, float]]) -> Dict[str, Any]:
    """予測結果JSONの作成"""
    from intervals import pi95_delta, pi95_pop, load_per_year_metrics, load_cv_metrics
    
    # メトリクスデータの読み込み
    per_year_df = load_per_year_metrics()
    cv_metrics = load_cv_metrics()
    
    result = {
        "town": town,
        "base_year": base_year,
        "horizons": horizons,
        "path": []
    }
    
    for i, horizon in enumerate(horizons):
        year = base_year + horizon
        delta_hat = delta_predictions[i]
        pop_hat = population_path[i]
        contrib = contributions[i]
        
        # Δ用と人口用の予測区間を計算
        pi_delta = pi95_delta(year, per_year_df, cv_metrics)
        pi_pop = pi95_pop(pop_hat, year, horizon, per_year_df, cv_metrics)
        
        path_entry = {
            "year": year,
            "delta_hat": round(delta_hat, 1),
            "pop_hat": round(pop_hat, 1),
            "pi95_delta": [round(pi_delta[0], 1), round(pi_delta[1], 1)],
            "pi95_pop": [round(pi_pop[0], 1), round(pi_pop[1], 1)],
            "contrib": {
                "exp": round(contrib["exp"], 1),
                "macro": round(contrib["macro"], 1),
                "inertia": round(contrib["inertia"], 1),
                "other": round(contrib["other"], 1)
            }
        }
        
        result["path"].append(path_entry)
    
    return result

def forecast_population(town: str, base_year: int, horizons: List[int], 
                       base_population: float) -> Dict[str, Any]:
    """人口予測の実行"""
    print(f"[L5] 人口予測を実行中: {town}, {base_year}, horizons={horizons}")
    
    # モデルとデータの読み込み
    model = load_model()
    features_df = load_future_features()
    effects_coef = load_effects_coefficients()
    
    # 該当町丁のデータをフィルタ
    town_features = features_df[features_df["town"] == town].copy()
    if len(town_features) == 0:
        raise ValueError(f"町丁 '{town}' の将来特徴が見つかりません")
    
    # 予測年でフィルタ
    target_years = [base_year + h for h in horizons]
    town_features = town_features[town_features["year"].isin(target_years)].copy()
    
    if len(town_features) == 0:
        raise ValueError(f"町丁 '{town}' の予測年 {target_years} の特徴が見つかりません")
    
    # 年順でソート
    town_features = town_features.sort_values("year")
    
    # 逐次予測（ラグ更新付き）
    delta_predictions, population_path, contributions = predict_delta_population_sequential(
        model, town_features, base_population)
    
    # 予測区間の計算
    prediction_intervals = estimate_intervals(delta_predictions, target_years, base_year)
    
    # デバッグ出力の保存
    save_debug_outputs(town, town_features, delta_predictions, contributions, base_year)
    
    # 結果の作成
    result = create_forecast_result(town, base_year, horizons, delta_predictions, 
                                  population_path, contributions, prediction_intervals)
    
    return result

def save_debug_outputs(town: str, features_df: pd.DataFrame, delta_predictions: List[float], 
                      contributions: List[Dict[str, float]], base_year: int) -> None:
    """デバッグ出力の保存"""
    # 特徴デバッグ
    debug_features = features_df[["year", "exp_all_h1", "exp_all_h2", "exp_all_h3", 
                                 "lag_d1", "lag_d2", "foreign_population", "foreign_change"]].copy()
    debug_features["horizon"] = debug_features["year"] - base_year
    debug_features.to_csv(f"../../data/processed/l5_debug_features_{town.replace(' ', '_')}.csv", index=False)
    
    # 寄与デバッグ
    debug_contrib = pd.DataFrame({
        "year": features_df["year"],
        "horizon": features_df["year"] - base_year,
        "delta_full": delta_predictions,
        "contrib_exp": [c["exp"] for c in contributions],
        "contrib_macro": [c["macro"] for c in contributions],
        "contrib_inertia": [c["inertia"] for c in contributions],
        "contrib_other": [c["other"] for c in contributions]
    })
    debug_contrib.to_csv(f"../../data/processed/l5_debug_contrib_{town.replace(' ', '_')}.csv", index=False)
    
    print(f"[L5] デバッグ出力を保存: l5_debug_features_{town.replace(' ', '_')}.csv, l5_debug_contrib_{town.replace(' ', '_')}.csv")

def main(town: str, base_year: int, horizons: List[int], base_population: float) -> Dict[str, Any]:
    """メイン処理"""
    return forecast_population(town, base_year, horizons, base_population)

if __name__ == "__main__":
    # テスト用
    test_town = "九品寺5丁目"
    test_base_year = 2025
    test_horizons = [1, 2, 3]
    test_base_pop = 7000.0
    
    result = main(test_town, test_base_year, test_horizons, test_base_pop)
    print(json.dumps(result, ensure_ascii=False, indent=2))
