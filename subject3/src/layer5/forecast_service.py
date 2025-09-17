# -*- coding: utf-8 -*-
# src/layer5/forecast_service.py
print(f"[L5] forecast_service path: {__file__}")
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
from typing import Dict, List, Any, Tuple, Optional
from intervals import estimate_intervals
import sys
import os
import re
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.feature_gate import load_feature_list, align_features_for_inference, get_feature_statistics

# ログ設定
logger = logging.getLogger(__name__)

# 列名を厳密にパース
COL_RE = re.compile(r"^exp_(?P<cat>.+)_(?P<dir>inc|dec)_h(?P<h>[123])$")

def parse_exp_col(col: str):
    """列名を厳密にパース（public_edu_medicalなど下線を含むカテゴリに対応）"""
    m = COL_RE.match(col)
    if not m:
        return None
    cat = m.group("cat")                  # 例: "housing" / "public_edu_medical"
    dir_ = m.group("dir")                 # "inc" or "dec"
    h = int(m.group("h"))                 # 1,2,3
    return cat, dir_, h

# パス設定
P_MODEL = "../../models/l4_model_no_macro.joblib"  # no_macroモデルを使用
P_FUTURE_FEATURES = "../../data/processed/l5_future_features.csv"
P_EFFECTS_COEF = "../../output/effects_coefficients.csv"

def load_model() -> Any:
    """L4モデルを読み込み（no_macro対応）"""
    if not Path(P_MODEL).exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {P_MODEL}")
    
    try:
        model = joblib.load(P_MODEL)
        print(f"[L5] no_macroモデルを読み込みました: {P_MODEL}")
        return model
    except Exception as e:
        raise RuntimeError(f"モデルの読み込みに失敗: {e}")

def load_future_features() -> pd.DataFrame:
    """将来特徴を読み込み"""
    if not Path(P_FUTURE_FEATURES).exists():
        raise FileNotFoundError(f"将来特徴ファイルが見つかりません: {P_FUTURE_FEATURES}")
    
    return pd.read_csv(P_FUTURE_FEATURES)

def load_effects_coefficients() -> pd.DataFrame:
    """効果係数を読み込み（Colab対応）"""
    # 複数のパスを試す
    possible_paths = [
        P_EFFECTS_COEF,
        "../../output/effects_coefficients.csv",
        "output/effects_coefficients.csv",
        "subject3-5/output/effects_coefficients.csv",
        "subject3-4/output/effects_coefficients.csv",
        "subject3-3/output/effects_coefficients.csv"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"[L5] 効果係数ファイルを読み込み: {path}")
            return pd.read_csv(path)
    
    # effects_coefficients.csvが見つからない場合は、effects_coefficients_rate.csvから生成
    print(f"[L5] effects_coefficients.csvが見つかりません。rateファイルから生成します...")
    try:
        # effects_coefficients_rate.csvを読み込み
        rate_paths = [
            "../../output/effects_coefficients_rate.csv",
            "output/effects_coefficients_rate.csv",
            "subject3-5/output/effects_coefficients_rate.csv",
            "subject3-4/output/effects_coefficients_rate.csv",
            "subject3-3/output/effects_coefficients_rate.csv"
        ]
        
        rate_df = None
        for path in rate_paths:
            if Path(path).exists():
                print(f"[L5] 率効果係数ファイルを読み込み: {path}")
                rate_df = pd.read_csv(path)
                break
        
        if rate_df is None:
            raise FileNotFoundError("effects_coefficients_rate.csvも見つかりません")
        
        # effects_coefficients.csvの形式に変換
        effects_df = rate_df.copy()
        if 'beta' in effects_df.columns:
            effects_df['beta_t'] = effects_df['beta']
            effects_df['beta_t1'] = effects_df['beta']  # 同じ値を使用
            effects_df = effects_df.drop('beta', axis=1)
        
        print(f"[L5] effects_coefficients.csvを生成しました（{len(effects_df)}行）")
        return effects_df
        
    except Exception as e:
        print(f"[WARN] rateファイルからの生成に失敗: {e}")
        # 空のDataFrameを返す（エラーを回避）
        return pd.DataFrame(columns=['event_var', 'beta_t', 'beta_t1'])

# 係数辞書（グローバル変数として保持）
COEF_RATE = {}

def load_coef_rate_dict(effects_coef_rate: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """係数を辞書形式で読み込み（キーベースアクセス用）"""
    coef_dict = {}
    
    for _, row in effects_coef_rate.iterrows():
        event_var = row["event_var"]
        beta = float(row["beta"])
        
        # 係数の単位を確認（%表記なら小数に変換）
        COEF_IS_PERCENT = True  # effects_coefficients_rate.csvが%表記の場合True
        
        if COEF_IS_PERCENT:
            beta = beta / 100.0  # % → 小数
        
        # 全horizonに同じ係数を適用（簡略化）
        coef_dict[event_var] = {1: beta, 2: beta, 3: beta}
    
    return coef_dict

def coef_rate_for(cat: str, dir_: str, h: int) -> float:
    """係数取得の一元化（キーベースアクセス）"""
    key = f"{cat}_{dir_}"   # 例: "housing_inc"
    if key not in COEF_RATE:
        logger.warning(f"係数が見つかりません: {key}")
        return 0.0
    return COEF_RATE[key].get(h, 0.0)

def compute_exp_rate_terms(row) -> Dict[int, float]:
    """exp_rate_termsの計算をexp_rate_all_h*列から取得"""
    # {1: rate_h1, 2: rate_h2, 3: rate_h3}
    out = {1: 0.0, 2: 0.0, 3: 0.0}
    
    # exp_rate_all_h*列から直接取得（既に係数×値の合計が計算済み）
    for h in [1, 2, 3]:
        col = f"exp_rate_all_h{h}"
        if col in row and not pd.isna(row[col]):
            out[h] = float(row[col])
    
    return out

def load_model_features() -> Optional[List[str]]:
    """学習時に使用された特徴量リストを読み込み（no_macro対応）"""
    # no_macroモデルの場合は専用の特徴量リストを使用
    feature_list_path = Path("../../src/layer4/feature_reduced_training/feature_list_no_macro.json")
    
    if not feature_list_path.exists():
        print(f"[WARN] no_macro特徴量リストファイルが見つかりません: {feature_list_path}")
        # フォールバック: アブレーション研究の結果から特徴量リストを生成
        print(f"[L5] アブレーション研究から特徴量リストを生成中...")
        return generate_no_macro_features()
    
    try:
        return load_feature_list(str(feature_list_path))
    except Exception as e:
        print(f"[WARN] 特徴量リストファイルの読み込みに失敗: {e}")
        return generate_no_macro_features()

def generate_no_macro_features() -> List[str]:
    """no_macroモデル用の特徴量リストを生成（アブレーション研究の結果から）"""
    try:
        # アブレーション研究の結果から特徴量リストを読み込み
        ablation_path = Path("../../data/processed/ablation_study/ablation_metrics_no_macro.json")
        if ablation_path.exists():
            with open(ablation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                remaining_features = data.get('remaining_features', [])
                if remaining_features:
                    print(f"[L5] アブレーション研究から特徴量リストを読み込み: {len(remaining_features)}列")
                    return remaining_features
        
        # フォールバック: 手動でno_macroの特徴量リストを定義
        print(f"[L5] フォールバック: 手動でno_macro特徴量リストを生成")
        return [
            "era_covid", "era_post2009", "era_post2013", "era_post2022", "era_pre2013",
            "foreign_change", "foreign_change_covid", "foreign_change_post2022",
            "foreign_log", "foreign_log_covid", "foreign_log_post2022",
            "foreign_ma3", "foreign_ma3_covid", "foreign_ma3_post2022",
            "foreign_pct_change", "foreign_pct_change_covid", "foreign_pct_change_post2022",
            "foreign_population", "foreign_population_covid", "foreign_population_post2022",
            "lag_d1", "lag_d2", "ma2_delta", "pop_total",
            "ring1_exp_commercial_inc_h1", "ring1_exp_commercial_inc_h2", "ring1_exp_commercial_inc_h3",
            "ring1_exp_disaster_dec_h1", "ring1_exp_disaster_dec_h2", "ring1_exp_disaster_dec_h3",
            "ring1_exp_disaster_inc_h1", "ring1_exp_disaster_inc_h2", "ring1_exp_disaster_inc_h3",
            "ring1_exp_employment_inc_h1", "ring1_exp_employment_inc_h2", "ring1_exp_employment_inc_h3",
            "ring1_exp_housing_dec_h1", "ring1_exp_housing_dec_h2", "ring1_exp_housing_dec_h3",
            "ring1_exp_housing_inc_h1", "ring1_exp_housing_inc_h2", "ring1_exp_housing_inc_h3",
            "ring1_exp_public_edu_medical_dec_h1", "ring1_exp_public_edu_medical_dec_h2",
            "town_ma5", "town_std5", "town_trend5"
        ]
    except Exception as e:
        print(f"[WARN] no_macro特徴量リストの生成に失敗: {e}")
        return None

def choose_features(df: pd.DataFrame, model_features: Optional[List[str]] = None) -> List[str]:
    """L4の実際に使用された特徴を選択（率ベース対応）"""
    if model_features is not None:
        # 学習時に使用された特徴量のリストを使用
        keep = []
        for feature in model_features:
            if feature in df.columns and np.issubdtype(df[feature].dtype, np.number):
                keep.append(feature)
        return keep
    
    # フォールバック: 従来の方法
    # L4で実際に使用された特徴リスト（率ベースに更新）
    l4_features = [
        "pop_total",
        "exp_rate_all_h1",  # 率ベースに変更
        "exp_rate_all_h2", 
        "exp_rate_all_h3",
        "era_post2009",
        "era_post2013",
        "era_pre2013",
        "exp_rate_all_h1_pre2013",  # 率ベースに変更
        "exp_rate_all_h1_post2013",
        "exp_rate_all_h1_post2009",
        "exp_rate_all_h2_pre2013",
        "exp_rate_all_h2_post2013",
        "exp_rate_all_h2_post2009",
        "exp_rate_all_h3_pre2013",
        "exp_rate_all_h3_post2013",
        "exp_rate_all_h3_post2009",
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
        "exp_rate_all_h1_post2022",  # 率ベースに変更
        "exp_rate_all_h2_post2022",
        "exp_rate_all_h3_post2022",
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
    
    # ring1_*特徴量も追加（学習時と同じように）
    for col in df.columns:
        if col.startswith("ring1_") and np.issubdtype(df[col].dtype, np.number):
            keep.append(col)
    
    return keep

def predict_delta_population_sequential(model: Any, features_df: pd.DataFrame, base_population: float, model_features: Optional[List[str]] = None, base_year: int = None) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    """人口変化量を逐次予測（ラグ更新付き）"""
    # base_yearが渡されていない場合は、features_dfから推定
    if base_year is None:
        years = sorted(features_df["year"].unique())
        base_year = min(years)
        print(f"[L5] base_yearが指定されていません。推定値 {base_year} を使用します。")
    
    # パススルー列の定義（期待効果関連列は常に保持）
    PASSTHRU_PATTERNS = [
        r'^exp_rate_all_h\d+$',  # exp_rate_all_h1, exp_rate_all_h2, exp_rate_all_h3
        r'^manual_people_h\d+$'  # manual_people_h1, manual_people_h2, manual_people_h3
    ]
    
    # 特徴量の整列（学習時と同じ列順・欠損0埋め）
    if model_features is not None:
        print(f"[L5] 学習時の特徴量リストを使用: {len(model_features)}列")
        features_df_aligned = align_features_for_inference(features_df, model_features)
        
        # パススルー列を追加（モデル特徴量に含まれていなくても保持）
        import re
        for col in features_df.columns:
            for pattern in PASSTHRU_PATTERNS:
                if re.match(pattern, col) and col not in features_df_aligned.columns:
                    features_df_aligned[col] = features_df[col]
                    print(f"[L5] パススルー列を追加: {col}")
        
        # 統計情報を表示
        stats = get_feature_statistics(features_df, model_features)
        print(f"[L5] 特徴量統計: カバレッジ={stats['feature_coverage']:.2f}, 欠損={len(stats['missing_features'])}, 余分={len(stats['extra_features'])}")
        
        feature_cols = model_features
    else:
        print("[L5][WARN] 学習時の特徴量リストが利用できません。フォールバックを使用します。")
        feature_cols = choose_features(features_df, model_features)
        features_df_aligned = features_df
    
    # 年順でソート
    features_df_aligned = features_df_aligned.sort_values("year").copy()
    
    # 予測結果の格納
    delta_predictions = []
    population_path = []
    contributions = []
    debug_rows = []  # デバッグ用の詳細情報
    
    # 初期状態
    pop = base_population
    prev_deltas = []  # 直近のΔを先頭に積む
    
    # 使う列グループを定義（率ベース対応、二重加算回避）
    # パススルー列から動的に取得（features_df_alignedに含まれる列を使用）
    EXP_RATE_COLS = [c for c in features_df_aligned.columns if c.startswith("exp_rate_all_h") and not c.endswith("_post2022")]
    # EXP_RATE_POST_COLS は使わない（二重加算を避けるため）
    MACRO_COLS = [c for c in feature_cols if c.startswith(("foreign_", "macro_"))]
    INERTIA_COLS = [c for c in feature_cols if c.startswith(("lag_", "town_ma", "town_std", "town_trend"))]
    
    def zero_cols(Xrow, cols):
        X0 = Xrow.copy()
        for c in cols:
            if c in X0.columns:
                X0[c] = 0.0
        return X0
    
    for i, (_, row) in enumerate(features_df_aligned.iterrows()):
        # --- 直前の予測でラグ系を更新 ---
        if len(prev_deltas) >= 1 and "lag_d1" in features_df_aligned.columns:
            features_df_aligned.iloc[i, features_df_aligned.columns.get_loc("lag_d1")] = prev_deltas[0]
        if len(prev_deltas) >= 2 and "lag_d2" in features_df_aligned.columns:
            features_df_aligned.iloc[i, features_df_aligned.columns.get_loc("lag_d2")] = prev_deltas[1]
        if "ma2_delta" in features_df_aligned.columns:
            v = np.mean(prev_deltas[:2]) if len(prev_deltas) >= 1 else np.nan
            features_df_aligned.iloc[i, features_df_aligned.columns.get_loc("ma2_delta")] = v
        
        # 特徴行列の準備（整列済みデータを使用）
        Xrow = features_df_aligned.iloc[i:i+1, features_df_aligned.columns.get_indexer(feature_cols)].copy()
        Xrow.columns = feature_cols
        Xrow = Xrow.replace([np.inf, -np.inf], np.nan)
        
        # ① GBM の「exp 抜き」予測
        X_noexp = zero_cols(Xrow.copy(), EXP_RATE_COLS)
        y_noexp = float(model.predict(X_noexp)[0])
        
        # ② 期待効果（率ベース）を人数に変換して加算（二重加算回避）
        exp_rate_terms = 0.0
        for c in EXP_RATE_COLS:
            if c in Xrow.columns:
                val = Xrow[c].values[0]
                if not pd.isna(val):
                    exp_rate_terms += float(val)
        
        # 安全弁：レートをクリップ（±50%以内）
        MAX_RATE = 0.5  # ±50%
        raw_rate = exp_rate_terms
        safe_rate = max(-MAX_RATE, min(MAX_RATE, raw_rate))
        if safe_rate != raw_rate:
            year = features_df.iloc[i]["year"]
            print(f"[L5][WARN] exp_rate clipped: {raw_rate:.4f} -> {safe_rate:.4f} (year={year})")
        
        # 率 → 人数変換（動的母数）
        base_for_rate = max(pop, 1.0)
        exp_people_from_rate = safe_rate * base_for_rate
        
        # 手動人数（もし build で別列に保持しているなら拾う）
        manual_people_cols = [c for c in Xrow.columns if c.startswith("manual_people_h")]
        exp_people_manual = 0.0
        if manual_people_cols:
            for c in manual_people_cols:
                val = Xrow[c].values[0]
                if not pd.isna(val):
                    exp_people_manual += float(val)
        
        # 総和（これを足し戻しに使う）
        exp_people = exp_people_from_rate + exp_people_manual
        
        # ③ 最終Δ予測 = 「GBM（exp抜き）」＋「期待効果（人数）」
        delta_hat = y_noexp + exp_people
        
        # ④ 人口パス更新
        pop = pop + delta_hat
        prev_deltas.insert(0, delta_hat)
        prev_deltas = prev_deltas[:8]  # 必要分だけ保持（安全）
        
        # ⑤ 寄与分解（ノックアウト基準を y_noexp に合わせる）
        contrib = {}
        contrib["exp"] = exp_people  # 人数ベース
        
        # macro 寄与 = y_noexp - y_without_macro
        y_wo_macro = float(model.predict(zero_cols(X_noexp.copy(), MACRO_COLS))[0])
        contrib["macro"] = y_noexp - y_wo_macro
        
        # inertia 寄与 = y_wo_macro - y_without_inertia
        y_wo_inertia = float(model.predict(zero_cols(X_noexp.copy(), INERTIA_COLS))[0])
        contrib["inertia"] = y_wo_macro - y_wo_inertia
        
        # other = 残差
        contrib["other"] = delta_hat - (contrib["exp"] + contrib["macro"] + contrib["inertia"])
        
        # デバッグ行（CSV出力用）
        year = features_df_aligned.iloc[i]["year"]
        horizon = year - base_year  # horizonを計算
        
        debug_rows.append({
            "year": year,
            "delta_noexp": y_noexp,
            "exp_rate_terms": exp_rate_terms,
            "exp_rate_terms_clipped": safe_rate,
            "base_pop_for_rate": base_for_rate,
            "exp_people_from_rate": exp_people_from_rate,
            "exp_people_manual": exp_people_manual,
            "exp_people_total": exp_people,
            "delta_hat": delta_hat,
            "pop_after": pop,
            "lag_d1": features_df.iloc[i]["lag_d1"] if "lag_d1" in features_df.columns else np.nan,
            "lag_d2": features_df.iloc[i]["lag_d2"] if "lag_d2" in features_df.columns else np.nan
        })
        
        # 詳細ログ出力（期待効果の計算過程）
        print(f"[L5] 年 {year} (h={horizon}): exp_rate_terms={exp_rate_terms:.6f}, "
              f"exp_rate_terms_clipped={safe_rate:.6f}, base_for_rate={base_for_rate:.1f}, "
              f"exp_people_from_rate={exp_people_from_rate:.1f}, exp_people_manual={exp_people_manual:.1f}, "
              f"exp_people_total={exp_people:.1f}, y_noexp={y_noexp:.1f}, delta_hat={delta_hat:.1f}")
        
        # 結果格納
        delta_predictions.append(delta_hat)
        population_path.append(pop)
        contributions.append(contrib)
    
    return delta_predictions, population_path, contributions, debug_rows

def calculate_contribution_knockout(model: Any, X: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """グループ・ノックアウトによる寄与分解（率ベース対応）"""
    # グループ定義（率ベース対応）
    GROUPS = {
        "exp": list(set([c for c in feature_cols if c.startswith("exp_rate_all_h")] + [c for c in feature_cols if c.startswith("exp_rate_all_h") and c.endswith("_post2022")])),
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

def _build_result_with_explain(
    town: str,
    baseline_year: int,
    years: list[int],
    forecast_payload: dict,           # year -> {'delta': float, 'pop': float, ...}
    features_by_year: dict,           # year -> {'exp_rate_terms': float, ...} を含む
    base_pop_for_rate: dict,          # year -> float（率換算に使う母数、通常は前年人口）
    manual_add_by_h: dict             # {1: float, 2: float, 3: float}
):
    result = {
        "town": town,
        "baseline_year": baseline_year,
        "forecast": {},
        "explain": {},
    }
    for y in sorted(years):
        h = int(y - baseline_year)
        # 既存予測をそのまま載せる
        result["forecast"][y] = forecast_payload[y]

        # --- explain（率→人数換算 + 手動 + 復元）---
        exp_rate_terms = float(features_by_year[y].get("exp_rate_terms", 0.0))
        base = float(base_pop_for_rate.get(y, 0.0))
        exp_people_from_rate = exp_rate_terms * base
        exp_people_manual = float(manual_add_by_h.get(h, 0.0))
        exp_people_total = exp_people_from_rate + exp_people_manual
        delta_hat = float(forecast_payload[y].get("delta", 0.0))
        delta_noexp = delta_hat - exp_people_total

        result["explain"][y] = {
            "exp_rate_terms": exp_rate_terms,
            "base_pop_for_rate": base,
            "exp_people_from_rate": float(exp_people_from_rate),
            "exp_people_manual": float(exp_people_manual),
            "exp_people_total": float(exp_people_total),
            "delta_noexp": float(delta_noexp),
            "delta_hat": float(delta_hat),
            "exp_people_by_category": features_by_year[y].get("exp_people_by_category", {}),
        }
    return result

def create_forecast_result(town: str, base_year: int, horizons: List[int], 
                          delta_predictions: List[float], population_path: List[float],
                          contributions: List[Dict[str, float]], 
                          prediction_intervals: List[Tuple[float, float]],
                          explain_data: Optional[Dict] = None, base_population: float = 0.0) -> Dict[str, Any]:
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
        
        # exp_people_totalを決定論から取得（SHAP集計ではなく）
        # 期待効果はモデル外の決定論で足し戻す（二重計上を防ぐ）
        exp_people_total = 0.0
        delta_noexp = 0.0
        
        # explain_dataから決定論の値を取得（文字列キーと整数キーの両方に対応）
        if explain_data and "explain" in explain_data:
            explain_year = explain_data["explain"].get(str(year), explain_data["explain"].get(year, {}))
            exp_people_total = float(explain_year.get("exp_people_total", 0.0))
            # delta_noexpはモデルの予測値（y_noexp）を使用
            delta_noexp = float(explain_year.get("delta_noexp", 0.0))
            print(f"[L5] 年 {year}: explain_dataから取得 - exp_people_total={exp_people_total:.1f}, delta_noexp={delta_noexp:.1f}")
        else:
            print(f"[L5] 年 {year}: explain_dataが利用できません - exp_people_total=0.0")
            delta_noexp = 0.0
        
        # 寄与辞書のexpを決定論値で上書き（SHAPではなく決定論値で上書き）
        contrib["exp"] = exp_people_total
        
        # keep recomposition for logging only
        delta_hat = float(delta_noexp) + float(exp_people_total)
        print(f"[L5] 年 {year}: delta_hat調整 - delta_hat={delta_hat:.1f} (= {delta_noexp:.1f} + {exp_people_total:.1f})")
        
        # DO NOT override result with delta_hat here.
        # Use the final (eventful) delta computed in [apply].
        final_delta = float(delta_predictions[i])  # イベント込みの最終値を使用
        
        # 人口パスを最終deltaで再計算
        if i == 0:
            # 最初の年は基準人口 + 最終delta
            pop_hat_adjusted = base_population + final_delta
        else:
            # 2年目以降は前年の調整後人口 + 最終delta
            prev_pop = result["path"][i-1]["pop_hat"]
            pop_hat_adjusted = prev_pop + final_delta
        
        # Δ用と人口用の予測区間を計算
        pi_delta = pi95_delta(year, per_year_df, cv_metrics)
        pi_pop = pi95_pop(pop_hat_adjusted, year, horizon, per_year_df, cv_metrics)
        
        # サマリー寄与（contrib）の整合化（otherを残差で再計算）
        # 最終Δに整合させる
        exp_total = float(exp_people_total)
        macro = float(contrib["macro"])
        inertia = float(contrib["inertia"])
        other = final_delta - (exp_total + macro + inertia)
        
        path_entry = {
            "year": year,
            "delta_hat": round(final_delta, 1),  # 最終Δ（イベント込み）を使用
            "pop_hat": round(pop_hat_adjusted, 1),
            "pi95_delta": [round(pi_delta[0], 1), round(pi_delta[1], 1)],
            "pi95_pop": [round(pi_pop[0], 1), round(pi_pop[1], 1)],
            "contrib": {
                "exp": round(exp_total, 1),  # 決定論の期待効果を使用
                "macro": round(macro, 1),  # モデル寄与
                "inertia": round(inertia, 1),  # モデル寄与
                "other": round(other, 1)  # 残差（最終Δに整合）
            }
        }
        
        result["path"].append(path_entry)
    
    return result

def create_basic_prediction(town: str, base_year: int, horizons: List[int], 
                           base_population: float, manual_add_by_h: dict = None) -> Dict[str, Any]:
    """
    将来特徴が見つからない場合の基本予測（イベントなし）
    """
    
    # 手動加算のデフォルト値設定
    manual_add_by_h = manual_add_by_h or {1: 0.0, 2: 0.0, 3: 0.0}
    
    # 基本予測：人口変化は0、人口はベース人口を維持
    target_years = [base_year + h for h in horizons]
    delta_predictions = [0.0] * len(horizons)  # 変化なし
    population_path = [base_population] * len(horizons)  # ベース人口を維持
    
    # 寄与分解：すべて0
    contributions = []
    for i, year in enumerate(target_years):
        contrib = {
            "exp": 0.0,
            "macro": 0.0,
            "inertia": 0.0,
            "other": 0.0
        }
        contributions.append(contrib)
    
    # 予測区間：変化なしなので信頼区間も0
    prediction_intervals = {
        "pi95_delta": [[0.0, 0.0]] * len(horizons),
        "pi95_pop": [[base_population, base_population]] * len(horizons)
    }
    
    # 結果を整形
    result = {
        "town": town,
        "base_year": base_year,
        "horizons": horizons,
        "path": []
    }
    
    for i, year in enumerate(target_years):
        path_entry = {
            "year": year,
            "delta_hat": delta_predictions[i],
            "pop_hat": population_path[i],
            "contrib": contributions[i],
            "pi95_delta": prediction_intervals["pi95_delta"][i],
            "pi95_pop": prediction_intervals["pi95_pop"][i]
        }
        result["path"].append(path_entry)
    
    # explain機能の準備（基本予測では空）
    explain = {}
    for i, year in enumerate(target_years):
        h = i + 1
        explain[year] = {
            "exp_rate_terms": 0.0,
            "base_pop_for_rate": base_population,
            "exp_people_from_rate": 0.0,
            "exp_people_manual": float(manual_add_by_h.get(h, 0.0)),
            "exp_people_total": float(manual_add_by_h.get(h, 0.0)),
            "delta_noexp": 0.0,
            "delta_hat": 0.0
        }
    
    result["explain"] = explain
    
    return result

def create_normal_prediction_without_features(town: str, base_year: int, horizons: List[int], 
                                            base_population: float, manual_add_by_h: dict = None) -> Dict[str, Any]:
    """
    将来特徴なしでも通常の人口予測を実行（イベントなしの通常予測）
    """
    # 手動加算のデフォルト値設定
    manual_add_by_h = manual_add_by_h or {1: 0.0, 2: 0.0, 3: 0.0}
    
    # モデルを読み込み
    model = load_model()
    model_features = load_model_features()
    
    # 将来特徴なしでも、モデルの基本予測を実行
    # 空の特徴量で予測を試行
    target_years = [base_year + h for h in horizons]
    
    # 空の特徴量データフレームを作成
    empty_features = pd.DataFrame({
        'town': [town] * len(horizons),
        'year': target_years
    })
    
    # モデルが期待する特徴量を0で埋める
    if model_features:
        for feature in model_features:
            if feature not in empty_features.columns:
                empty_features[feature] = 0.0
    
    # 逐次予測を実行
    try:
        delta_predictions, population_path, contributions, debug_rows = predict_delta_population_sequential(
            model, empty_features, base_population, model_features, base_year)
        
        # 予測区間の計算
        prediction_intervals = estimate_intervals(delta_predictions, target_years, base_year)
        
        # 結果を整形
        result = {
            "town": town,
            "base_year": base_year,
            "horizons": horizons,
            "path": []
        }
        
        for i, year in enumerate(target_years):
            path_entry = {
                "year": year,
                "delta_hat": delta_predictions[i],
                "pop_hat": population_path[i],
                "contrib": contributions[i],
                "pi95_delta": prediction_intervals["pi95_delta"][i],
                "pi95_pop": prediction_intervals["pi95_pop"][i]
            }
            result["path"].append(path_entry)
        
        # explain機能の準備
        explain = {}
        for i, year in enumerate(target_years):
            h = i + 1
            explain[year] = {
                "exp_rate_terms": 0.0,
                "base_pop_for_rate": base_population,
                "exp_people_from_rate": 0.0,
                "exp_people_manual": float(manual_add_by_h.get(h, 0.0)),
                "exp_people_total": float(manual_add_by_h.get(h, 0.0)),
                "delta_noexp": delta_predictions[i],
                "delta_hat": delta_predictions[i]
            }
        
        result["explain"] = explain
        return result
        
    except Exception as e:
        # 通常予測も失敗した場合は基本予測にフォールバック
        return create_basic_prediction(town, base_year, horizons, base_population, manual_add_by_h)

def forecast_population(town: str, base_year: int, horizons: List[int], 
                       base_population: float, debug_output_dir: str = None, 
                       manual_add_by_h: dict = None, apply_event_to_prediction: bool = True) -> Dict[str, Any]:
    """人口予測の実行"""
    
    # 手動加算のデフォルト値設定
    manual_add_by_h = manual_add_by_h or {1: 0.0, 2: 0.0, 3: 0.0}
    
    # モデルとデータの読み込み
    model = load_model()
    features_df = load_future_features()
    effects_coef = load_effects_coefficients()
    
    # 係数辞書を読み込み（グローバル変数に設定）
    global COEF_RATE
    COEF_RATE = load_coef_rate_dict(effects_coef)
    
    # 学習時に使用された特徴量リストを読み込み
    model_features = load_model_features()
    
    # 該当町丁のデータをフィルタ
    town_features = features_df[features_df["town"] == town].copy()
    if len(town_features) == 0:
        # 将来特徴が見つからない場合は、基本予測（イベントなし）を実行
        # ただし、イベントが発生していない町丁の場合は通常の人口予測を試行
        if apply_event_to_prediction:
            return create_basic_prediction(town, base_year, horizons, base_population, manual_add_by_h)
        else:
            # イベントなしの通常予測を試行（将来特徴なしでも）
            return create_normal_prediction_without_features(town, base_year, horizons, base_population, manual_add_by_h)
    
    # 予測年でフィルタ
    target_years = [base_year + h for h in horizons]
    town_features = town_features[town_features["year"].isin(target_years)].copy()
    
    if len(town_features) == 0:
        # 予測年の特徴が見つからない場合も基本予測を実行
        return create_basic_prediction(town, base_year, horizons, base_population, manual_add_by_h)
    
    # 年順でソート
    town_features = town_features.sort_values("year")
    
    # 逐次予測（ラグ更新付き）
    delta_predictions, population_path, contributions, debug_rows = predict_delta_population_sequential(
        model, town_features, base_population, model_features, base_year)
    
    # 予測区間の計算
    prediction_intervals = estimate_intervals(delta_predictions, target_years, base_year)
    
    # explain機能の準備
    # forecast_payload の作成
    forecast_payload = {}
    for i, year in enumerate(target_years):
        forecast_payload[year] = {
            "delta": delta_predictions[i],
            "pop": population_path[i]
        }
    
    # features_by_year の作成（列×正しい係数の合計で計算）
    features_by_year = {}
    for i, year in enumerate(target_years):
        row = town_features.iloc[i]
        h = i + 1  # 2026→h1, 2027→h2, 2028→h3
        
        # 列×正しい係数の合計でexp_rate_termsを計算
        exp_rate_terms_by_h = compute_exp_rate_terms(row)
        exp_rate_terms = exp_rate_terms_by_h.get(h, 0.0)
        
        # カテゴリ別の人数寄与を計算（デバッグ可視化のため）
        people_by_cat = {}
        base = base_population if i == 0 else (base_population + sum(delta_predictions[:i]))
        
        for col, val in row.items():
            parsed = parse_exp_col(col)
            if not parsed or val in (None, 0):
                continue
            cat, dir_, col_h = parsed
            if col_h != h:  # 該当するhorizonのみ
                continue
            coef = coef_rate_for(cat, dir_, col_h)
            rate_contrib = float(val) * float(coef)
            people_contrib = rate_contrib * float(base)
            k = f"{cat}_{dir_}_h{col_h}"
            people_by_cat[k] = people_by_cat.get(k, 0.0) + people_contrib
        
        features_by_year[year] = {
            "exp_rate_terms": exp_rate_terms,
            "exp_people_by_category": people_by_cat
        }
        
        # デバッグログ（今回の症状を即検知する）
        logger.info("[CHECK] h=%d base=%.1f exp_rate=%.6f people=%.1f by_cat=%s",
                    h, base, exp_rate_terms, exp_rate_terms*base, 
                    json.dumps(people_by_cat, ensure_ascii=False))
        print(f"[explain] year={year} (h={h}): exp_rate_terms={exp_rate_terms:.6f}, base={base:.1f}, people={exp_rate_terms*base:.1f}")
        print(f"[explain] by_category: {people_by_cat}")
        
        # 追加のクイック自己診断ログ
        print(f"[apply-check] {year}: v_inc={row.get('event_housing_inc_t', 0.0):.6f}/{row.get('event_housing_inc_t1', 0.0):.6f}, "
              f"v_dec={row.get('event_housing_dec_t', 0.0):.6f}/{row.get('event_housing_dec_t1', 0.0):.6f}, "
              f"exp_rate_h*={exp_rate_terms_by_h.get(1, 0.0):.6f},{exp_rate_terms_by_h.get(2, 0.0):.6f},{exp_rate_terms_by_h.get(3, 0.0):.6f}")
    
    # base_pop_for_rate の作成（前年人口）
    base_pop_for_rate = {}
    current_pop = base_population
    for i, year in enumerate(target_years):
        base_pop_for_rate[year] = current_pop
        if i < len(population_path):
            current_pop = population_path[i]
    
    # explain機能を統合
    explain_result = _build_result_with_explain(
        town=town,
        baseline_year=base_year,
        years=target_years,
        forecast_payload=forecast_payload,
        features_by_year=features_by_year,
        base_pop_for_rate=base_pop_for_rate,
        manual_add_by_h=manual_add_by_h,
    )
    
    # イベント効果を予測値に適用（explain_resultベース）
    print("[L5] イベント効果を予測値に適用中...")
    
    # 1) 年別のイベント寄与（人）
    exp_people_total_by_year = {}
    for i, year in enumerate(target_years):
        h = i + 1
        rate = features_by_year[year]["exp_rate_terms"]  # ここは 2) で作った exp_rate_all_* の合算
        base = base_pop_for_rate[year]
        from_rate = rate * base
        manual = float(manual_add_by_h.get(f"h{h}", 0.0))
        exp_people_total_by_year[year] = from_rate + manual
    
    # 予測直後の配列 → 年→値へ
    delta_noexp_by_year = {year: float(delta_predictions[i]) for i, year in enumerate(target_years)}
    
    # 2) Δ(イベント適用後)
    delta_with_event_by_year = {y: delta_noexp_by_year[y] + exp_people_total_by_year[y] for y in target_years}
    
    # 3) 人口パス（イベント適用後）
    pop_eventful = []
    p = base_population
    for y in target_years:
        p = p + delta_with_event_by_year[y]
        pop_eventful.append(p)
    
    # 4) サマリー出力に使う値を差し替え
    use_event = any(abs(exp_people_total_by_year[y]) > 1e-9 for y in target_years)
    delta_path_for_summary = [delta_with_event_by_year[y] if use_event else delta_noexp_by_year[y] for y in target_years]
    pop_path_for_summary = pop_eventful if use_event else delta_predictions
    
    # 以降の出力に使う配列を置き換え
    delta_predictions = delta_path_for_summary
    population_path = pop_path_for_summary
    
    print("[apply] y_noexp:", delta_noexp_by_year)
    print("[apply] exp_people_total:", exp_people_total_by_year)
    print("[apply] delta_with_event:", delta_with_event_by_year)
    print("[apply] pop_eventful:", pop_eventful)
    
    # 結果の作成（従来の形式、explain_dataを含む）
    result = create_forecast_result(town, base_year, horizons, delta_predictions, 
                                  population_path, contributions, prediction_intervals, explain_result, base_population)
    
    # 結果にexplainを追加
    result["explain"] = explain_result["explain"]
    
    # デバッグ出力の保存（explain統合後）
    save_debug_outputs_from_explain(town, explain_result["explain"], target_years, base_year, debug_output_dir)
    
    return result

def save_debug_outputs_from_explain(town: str, explain_dict: Dict, target_years: List[int], 
                                   base_year: int, debug_output_dir: str = None) -> None:
    """explainデータから直接デバッグCSVを保存"""
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    # 出力ディレクトリの設定
    if debug_output_dir is None:
        debug_output_dir = "../../data/processed"
    
    debug_output_path = Path(debug_output_dir)
    debug_output_path.mkdir(parents=True, exist_ok=True)
    
    # 町名の安全化
    safe_town = str(town).strip().replace(" ", "_")
    
    # explainデータからCSV用の行配列を作成
    debug_rows = []
    max_residual = 0.0
    
    for year in sorted(target_years):
        year_data = explain_dict.get(str(year), explain_dict.get(year, {}))
        
        # 各値を取得（floatで明示キャスト）
        exp_rate_terms = float(year_data.get("exp_rate_terms", 0.0))
        base_pop_for_rate = float(year_data.get("base_pop_for_rate", 0.0))
        exp_people_from_rate = float(year_data.get("exp_people_from_rate", 0.0))
        exp_people_manual = float(year_data.get("exp_people_manual", 0.0))
        exp_people_total = float(year_data.get("exp_people_total", 0.0))
        delta_noexp = float(year_data.get("delta_noexp", 0.0))
        delta_hat = float(year_data.get("delta_hat", 0.0))
        
        # 復元誤差の計算
        residual = delta_hat - (delta_noexp + exp_people_total)
        max_residual = max(max_residual, abs(residual))
        
        debug_rows.append({
            "year": int(year),  # 英語の列名に統一
            "exp_rate_terms": exp_rate_terms,
            "base_pop_for_rate": base_pop_for_rate,
            "exp_people_from_rate": exp_people_from_rate,
            "exp_people_manual": exp_people_manual,
            "exp_people_total": exp_people_total,
            "delta_noexp": delta_noexp,
            "delta_hat": delta_hat,
            "residual_error": residual
        })
        
        # 最初の年のログ出力
        if year == target_years[0]:
            print(f"[explain] year={year}: rate={exp_rate_terms:.6f}, base={base_pop_for_rate:.1f}, "
                  f"people_from_rate={exp_people_from_rate:.1f}, manual={exp_people_manual:.1f}, "
                  f"total={exp_people_total:.1f}")
    
    # 復元誤差の警告
    if max_residual > 1e-6:
        print(f"[WARN] 復元誤差が大きすぎます: {max_residual:.2e}")
    
    # DataFrame化して保存
    debug_detail = pd.DataFrame(debug_rows)
    debug_detail_path = debug_output_path / f"l5_debug_detail_{safe_town}.csv"
    debug_detail.to_csv(debug_detail_path, index=False)
    
    print(f"Saved debug detail: {debug_detail_path} (rows={len(debug_detail)})")

def save_debug_outputs(town: str, features_df: pd.DataFrame, delta_predictions: List[float], 
                      contributions: List[Dict[str, float]], population_path: List[float], 
                      base_population: float, base_year: int, debug_rows: List[Dict], 
                      debug_output_dir: str = None) -> None:
    """デバッグ出力の保存"""
    # 出力ディレクトリの設定
    if debug_output_dir is None:
        debug_output_dir = "../../data/processed"
    
    debug_output_path = Path(debug_output_dir)
    debug_output_path.mkdir(parents=True, exist_ok=True)
    
    # 特徴デバッグ（率ベース対応、除外列対応）
    debug_cols = ["year"]
    optional_cols = ["exp_rate_all_h1", "exp_rate_all_h2", "exp_rate_all_h3", 
                     "lag_d1", "lag_d2", "foreign_population", "foreign_change"]
    
    # 存在する列のみを選択
    available_cols = [col for col in optional_cols if col in features_df.columns]
    debug_cols.extend(available_cols)
    
    debug_features = features_df[debug_cols].copy()
    debug_features["horizon"] = debug_features["year"] - base_year
    
    # 除外された列がある場合は0で埋める
    for col in optional_cols:
        if col not in features_df.columns:
            debug_features[col] = 0.0
    
    debug_features.to_csv(debug_output_path / f"l5_debug_features_{town.replace(' ', '_')}.csv", index=False)
    
    # 寄与デバッグ（詳細版）
    # 配列の長さを統一
    n_rows = len(features_df)
    
    debug_contrib = pd.DataFrame({
        "year": features_df["year"].values,
        "horizon": (features_df["year"] - base_year).values,
        "delta_full": delta_predictions,
        "contrib_exp": [c["exp"] for c in contributions],
        "contrib_macro": [c["macro"] for c in contributions],
        "contrib_inertia": [c["inertia"] for c in contributions],
        "contrib_other": [c["other"] for c in contributions],
        # 率関連のデバッグ情報
        "exp_rate_all_h1": features_df["exp_rate_all_h1"].values if "exp_rate_all_h1" in features_df.columns else np.full(n_rows, np.nan),
        "exp_rate_all_h2": features_df["exp_rate_all_h2"].values if "exp_rate_all_h2" in features_df.columns else np.full(n_rows, np.nan),
        "exp_rate_all_h3": features_df["exp_rate_all_h3"].values if "exp_rate_all_h3" in features_df.columns else np.full(n_rows, np.nan),
        "base_pop_for_rate": [max(pop, 1.0) for pop in population_path],  # population_pathの長さに合わせる
        "lag_d1": features_df["lag_d1"].values if "lag_d1" in features_df.columns else np.full(n_rows, np.nan),
        "lag_d2": features_df["lag_d2"].values if "lag_d2" in features_df.columns else np.full(n_rows, np.nan)
    })
    debug_contrib.to_csv(debug_output_path / f"l5_debug_contrib_{town.replace(' ', '_')}.csv", index=False)
    
    # 詳細デバッグ（内訳透明化）
    debug_detail = pd.DataFrame(debug_rows)
    debug_detail.to_csv(debug_output_path / f"l5_debug_detail_{town.replace(' ', '_')}.csv", index=False)
    
    print(f"[L5] デバッグ出力を保存: {debug_output_path}/l5_debug_features_{town.replace(' ', '_')}.csv, {debug_output_path}/l5_debug_contrib_{town.replace(' ', '_')}.csv, {debug_output_path}/l5_debug_detail_{town.replace(' ', '_')}.csv")

def run_scenario(scenario: Dict[str, Any], out_path: str = None) -> Dict[str, Any]:
    """
    シナリオを実行して予測結果を返す（全地域表示対応）
    
    Args:
        scenario: シナリオ辞書（town, base_year, horizons, events, macros, manual_deltaを含む）
        out_path: 出力パス（オプション）
    
    Returns:
        辞書形式の予測結果（必須キー: town, baseline_year, horizons, results）
        results: 年ごと配列。各要素は year, delta, pop, contrib, pi を含む
    """
    # シナリオから基本パラメータを抽出
    town = scenario.get("town")
    base_year = scenario.get("base_year", 2025)
    horizons = scenario.get("horizons", [1, 2, 3])
    manual_delta = scenario.get("manual_delta", {})
    
    if not town:
        raise ValueError("シナリオにtownが指定されていません")
    
    # 手動加算パラメータを準備
    manual_add = {
        1: float(manual_delta.get("h1", 0.0)),
        2: float(manual_delta.get("h2", 0.0)),
        3: float(manual_delta.get("h3", 0.0))
    }
    
    # ベース人口を取得（シナリオから、またはデフォルト値）
    base_population = scenario.get("base_population", 0.0)
    
    # 予測実行
    result = forecast_population(
        town=town,
        base_year=base_year,
        horizons=horizons,
        base_population=base_population,
        debug_output_dir=out_path,
        manual_add_by_h=manual_add
    )
    
    # 返り値契約に従って結果を整形
    formatted_result = {
        "town": result["town"],
        "baseline_year": result["base_year"],
        "horizons": result["horizons"],
        "results": []
    }
    
    # 各年の結果をresults配列に追加
    for path_entry in result["path"]:
        year = path_entry["year"]
        delta = path_entry["delta_hat"]
        pop = path_entry["pop_hat"]
        contrib = path_entry["contrib"]
        pi_delta = path_entry.get("pi95_delta", [0.0, 0.0])
        pi_pop = path_entry.get("pi95_pop", [0.0, 0.0])
        
        # Δの整合性チェック
        delta_sum = contrib["exp"] + contrib["macro"] + contrib["inertia"] + contrib["other"]
        delta_diff = abs(delta - delta_sum)
        
        if delta_diff > 1e-6:
            logger.warning(f"[L5] Δの整合性チェック失敗: town={town}, year={year}, "
                          f"delta={delta:.6f}, sum={delta_sum:.6f}, diff={delta_diff:.6f}")
        else:
            logger.info(f"[L5] Δの整合性チェックOK: town={town}, year={year}, "
                       f"delta={delta:.6f}, sum={delta_sum:.6f}")
        
        year_result = {
            "year": year,
            "delta": round(delta, 1),
            "pop": round(pop, 1),
            "contrib": {
                "exp": round(contrib["exp"], 1),
                "macro": round(contrib["macro"], 1),
                "inertia": round(contrib["inertia"], 1),
                "other": round(contrib["other"], 1)
            },
            "pi": {
                "delta_low": round(pi_delta[0], 1),
                "delta_high": round(pi_delta[1], 1),
                "pop_low": round(pi_pop[0], 1),
                "pop_high": round(pi_pop[1], 1)
            }
        }
        
        formatted_result["results"].append(year_result)
    
    # explain連携の維持（既存のl5_debug_detail_*.csvの生成を継続）
    if "explain" in result:
        formatted_result["explain"] = result["explain"]
    
    return formatted_result

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
