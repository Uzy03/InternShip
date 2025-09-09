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
from scipy.spatial.distance import cdist
import sys
import os
# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from src.common.spatial import calculate_spatial_lags_simple, detect_cols_to_lag
from src.common.feature_gate import drop_excluded_columns

# パス設定
P_FEATURES_PANEL = "../../data/processed/features_panel.csv"
P_FUTURE_EVENTS = "../../data/processed/l5_future_events.csv"
P_EFFECTS_COEF = "../../output/effects_coefficients.csv"
P_EFFECTS_COEF_RATE = "../../output/effects_coefficients_rate.csv"
P_OUTPUT = "../../data/processed/l5_future_features.csv"

# 空間ラグ設定
SPATIAL_CENTROIDS_CSV = "../../data/processed/town_centroids.csv"
SPATIAL_TOWN_COL = "town"  # 町名の列名
SPATIAL_JOIN_COL = "town"  # 結合キーの列名（town_id または town）

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

def load_spatial_centroids() -> pd.DataFrame:
    """空間重心データを読み込み"""
    if not Path(SPATIAL_CENTROIDS_CSV).exists():
        print(f"[L5][WARN] 空間重心ファイルが見つかりません: {SPATIAL_CENTROIDS_CSV}")
        return None
    
    centroids_df = pd.read_csv(SPATIAL_CENTROIDS_CSV)
    
    # 必要な列の確認
    required_cols = ["lon", "lat", SPATIAL_JOIN_COL]
    missing_cols = [col for col in required_cols if col not in centroids_df.columns]
    if missing_cols:
        print(f"[L5][WARN] 空間重心ファイルに列不足: {missing_cols}")
        return None
    
    print(f"[L5] 空間重心データを読み込み: {len(centroids_df)}件")
    return centroids_df

def normalize_town_name(town_name: str) -> str:
    """町丁名を正規化（重心データの形式に合わせる）"""
    if pd.isna(town_name):
        return town_name
    
    town_name = str(town_name)
    
    # 漢数字を半角数字に変換（「九」は固有名詞の一部なので除外）
    kanji_to_num = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '十': '10'
    }
    
    for kanji, num in kanji_to_num.items():
        town_name = town_name.replace(kanji, num)
    
    # 全角数字を半角数字に変換
    town_name = town_name.replace('０', '0').replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4')
    town_name = town_name.replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9')
    
    return town_name

def calculate_spatial_lags(future_events: pd.DataFrame, centroids_df: pd.DataFrame) -> pd.DataFrame:
    """空間ラグ特徴を計算"""
    if centroids_df is None:
        print("[L5][WARN] 空間重心データが利用できないため、空間ラグをスキップします")
        return future_events
    
    result = future_events.copy()
    
    # 各町丁の座標を取得（正規化された町丁名で）
    town_coords = {}
    for _, row in centroids_df.iterrows():
        town_key = normalize_town_name(row[SPATIAL_JOIN_COL])
        town_coords[town_key] = (row['lon'], row['lat'])
    
    print(f"[L5] 正規化後の町丁名（最初の5件）: {list(town_coords.keys())[:5]}")
    
    # 距離行列を計算
    towns = list(town_coords.keys())
    coords = np.array([town_coords[town] for town in towns])
    distances = cdist(coords, coords, metric='euclidean')
    
    # 距離行列をDataFrameに変換
    dist_df = pd.DataFrame(distances, index=towns, columns=towns)
    
    # 各町丁について空間ラグを計算
    for i, (_, row) in enumerate(future_events.iterrows()):
        town = normalize_town_name(row[SPATIAL_TOWN_COL])
        
        if town not in town_coords:
            print(f"[L5][WARN] 町丁 '{town}' の座標が見つかりません")
            continue
        
        # 距離による重み付け（近いほど重い）
        distances_to_town = dist_df[town]
        
        # リング1: 最も近い5つの町丁（自分を除く）
        ring1_towns = distances_to_town.nsmallest(6).index[1:6]  # 自分を除く
        
        # リング2: 次の5つの町丁
        ring2_towns = distances_to_town.nsmallest(11).index[6:11]
        
        # リング3: 次の10の町丁
        ring3_towns = distances_to_town.nsmallest(21).index[11:21]
        
        # 各リングの特徴を計算（例：人口密度、イベント数など）
        # ここでは簡易的に距離の逆数を重みとして使用
        for ring_num, ring_towns in enumerate([ring1_towns, ring2_towns, ring3_towns], 1):
            if len(ring_towns) > 0:
                # 距離の逆数を重みとして使用
                weights = 1.0 / (distances_to_town[ring_towns] + 1e-6)  # ゼロ除算を防ぐ
                weights = weights / weights.sum()  # 正規化
                
                # 例：人口密度の空間ラグ（実際のデータに応じて調整）
                ring_col = f"ring{ring_num}_pop_density"
                result.loc[i, ring_col] = 0.0  # デフォルト値
                
                # 例：イベント数の空間ラグ
                ring_col = f"ring{ring_num}_event_count"
                result.loc[i, ring_col] = 0.0  # デフォルト値
    
    print(f"[L5] 空間ラグ特徴を計算完了: {len(result)}件")
    return result

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

def _event_col(event_type: str, effect_direction: str, h: int) -> str:
    """イベント列名を生成（列選択で方向を表現）"""
    dir_key = "inc" if effect_direction == "increase" else "dec"
    return f"exp_{event_type}_{dir_key}_h{h}"

def _sum_rate_for_h(Xrow, h: int, effects_coef_rate: pd.DataFrame) -> float:
    """horizon hの期待効果率を計算（係数×該当列の総和、符号そのまま）"""
    families = ["housing", "commercial", "public_edu_medical", "employment", "transit", "disaster", "policy_boundary"]
    total = 0.0
    
    for fam in families:
        # inc と dec を別々に処理（両方加算しない）
        for dir_key in ("inc", "dec"):
            col = f"exp_{fam}_{dir_key}_h{h}"
            val = float(Xrow.get(col, 0.0))
            if val == 0.0:
                continue
            coef = coef_rate(fam, dir_key, h, effects_coef_rate)
            contrib = coef * val
            total += contrib  # 符号そのまま加算
            
            # デバッグログ（クイック自己診断用）
            print(f"[apply-check] {fam}_{dir_key}_h{h}: val={val:.6f}, coef={coef:.6f}, contrib={contrib:.6f}")
    
    return total

def calculate_expected_effects_rate(future_events: pd.DataFrame, effects_coef_rate: pd.DataFrame, 
                                   manual_delta: Dict[str, float], manual_delta_rate: Dict[str, float],
                                   base_population: float, base_year: int) -> pd.DataFrame:
    """期待効果の計算（率ベース、hごとに正しい年配置）"""
    result = future_events[["town", "year"]].copy()
    
    # 個別イベント列を初期化
    for event_type in EVENT_TYPES:
        for direction in ["inc", "dec"]:
            for horizon in [1, 2, 3]:
                col = _event_col(event_type, direction, horizon)
                result[col] = 0.0
    
    # 各年でexp_rate_all_h{h}を初期化
    for horizon in [1, 2, 3]:
        result[f"exp_rate_all_h{horizon}"] = 0.0
    
    # 各年ごとに個別イベント列を埋める（列選択で方向を表現）
    for i, (_, row) in enumerate(future_events.iterrows()):
        year = row["year"]
        horizon = year - base_year
        
        if horizon not in [1, 2, 3]:
            continue
        
        # 各イベントタイプの効果を個別列に設定（方向別列から読み取り）
        for event_type in EVENT_TYPES:
            # 方向別のイベント強度を取得（無ければ 0）
            v_t_inc = row.get(f"event_{event_type}_inc_t", 0.0) or 0.0
            v_t1_inc = row.get(f"event_{event_type}_inc_t1", 0.0) or 0.0
            v_t_dec = row.get(f"event_{event_type}_dec_t", 0.0) or 0.0
            v_t1_dec = row.get(f"event_{event_type}_dec_t1", 0.0) or 0.0
            
            # 古い形式（方向なし）へのフォールバック（片方向のみ採用）
            if (v_t_inc == v_t_dec == 0.0):
                v_t = row.get(f"event_{event_type}_t", 0.0) or 0.0
                v_t1 = row.get(f"event_{event_type}_t1", 0.0) or 0.0
                # デフォルトは「increase」を優先（必要なら scenario 側で dec を入れる）
                v_t_inc, v_t1_inc = v_t, v_t1
            
            # 仕様通りの年割付（個別列ベース）
            if horizon == 1:
                # h=1: v_t(type) を inc/dec 列に設定
                inc_col = _event_col(event_type, "increase", horizon)
                dec_col = _event_col(event_type, "decrease", horizon)
                result.at[i, inc_col] = v_t_inc
                result.at[i, dec_col] = v_t_dec
            
            elif horizon == 2:
                # h=2: v_t(type) + v_t1(type) を inc/dec 列に設定
                inc_col = _event_col(event_type, "increase", horizon)
                dec_col = _event_col(event_type, "decrease", horizon)
                result.at[i, inc_col] = v_t_inc + v_t1_inc
                result.at[i, dec_col] = v_t_dec + v_t1_dec
            
            elif horizon == 3:
                # h=3: v_t(type) + v_t1(type) を inc/dec 列に設定
                inc_col = _event_col(event_type, "increase", horizon)
                dec_col = _event_col(event_type, "decrease", horizon)
                result.at[i, inc_col] = v_t_inc + v_t1_inc
                result.at[i, dec_col] = v_t_dec + v_t1_dec
        
        # manual_deltaの加算（個別列に設定）
        manual_key = f"h{horizon}"
        
        # manual_delta_rate（% → 小数）
        manual_rate = 0.0
        if manual_key in manual_delta_rate:
            manual_rate += float(manual_delta_rate[manual_key]) / 100.0
        
        # manual_delta（人数 → 率）
        manual_people = 0.0
        if manual_key in manual_delta:
            manual_people = float(manual_delta[manual_key])
            manual_rate += manual_people / max(base_population, 1.0)
        
        # 手動効果を全イベントタイプのinc/dec列に加算
        if manual_rate != 0.0:
            for event_type in EVENT_TYPES:
                inc_col = _event_col(event_type, "increase", horizon)
                dec_col = _event_col(event_type, "decrease", horizon)
                result.at[i, inc_col] += manual_rate
                result.at[i, dec_col] += manual_rate
        
        # 手動人数を別列に保存（デバッグ専用）
        manual_people_col = f"manual_people_h{horizon}"
        result[manual_people_col] = 0.0
        result.at[i, manual_people_col] = manual_people
    
    # exp_rate_all_h{1,2,3}を係数×該当列の総和で計算（符号そのまま）
    for horizon in [1, 2, 3]:
        exp_col = f"exp_rate_all_h{horizon}"
        result[exp_col] = result.apply(lambda row: _sum_rate_for_h(row, horizon, effects_coef_rate), axis=1)
        
        # 符号を保持（クリップしない）
        print(f"[L5] exp_rate_all_h{horizon} 計算完了（符号保持）")
    
    # post-2022相互作用の計算（率ベース）
    for horizon in [1, 2, 3]:
        exp_col = f"exp_rate_all_h{horizon}"
        post_col = f"exp_rate_all_h{horizon}_post2022"
        result[post_col] = 0.0
        
        for i, (_, row) in enumerate(future_events.iterrows()):
            year = row["year"]
            if year >= 2023:  # post-2022
                result.at[i, post_col] = result.at[i, exp_col]
    
    # デバッグ出力: 各年のexp_rate_all_h1/2/3を表示（符号保持で確認）
    print("[L5] 期待効果の健全性チェック（率ベース）:")
    for i, (_, row) in enumerate(future_events.iterrows()):
        year = row["year"]
        horizon = year - base_year
        if horizon in [1, 2, 3]:
            # 符号保持で確認
            exp_h1 = result.iloc[i][f"exp_rate_all_h1"]
            exp_h2 = result.iloc[i][f"exp_rate_all_h2"]
            exp_h3 = result.iloc[i][f"exp_rate_all_h3"]
            print(f"  年 {year} (h={horizon}): exp_rate_all_h1={exp_h1:+.4f}, exp_rate_all_h2={exp_h2:+.4f}, exp_rate_all_h3={exp_h3:+.4f}")
    
    # exp_rate_terms列を追加（年次合計）
    result['exp_rate_terms'] = result[['exp_rate_all_h1', 'exp_rate_all_h2', 'exp_rate_all_h3']].sum(axis=1).fillna(0.0)
    print(f"[L5] exp_rate_terms の例: {result['exp_rate_terms'].head(3).tolist()}")
    
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
    
    # 空間重心データの読み込み
    centroids_df = load_spatial_centroids()
    
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
    
    # 空間ラグ特徴を計算（共通モジュールを使用）
    if centroids_df is not None:
        print("[L5] 空間ラグ特徴を計算中...")
        
        # ラグ対象列の自動検出
        cols_to_lag = detect_cols_to_lag(features_df)
        print(f"[L5] ラグ対象列: {cols_to_lag[:10]}...")  # 最初の10列を表示
        
        # 空間ラグの計算
        features_df = calculate_spatial_lags_simple(
            features_df, 
            centroids_df, 
            cols_to_lag, 
            town_col="town", 
            year_col="year", 
            k_neighbors=5
        )
        
        # 生成されたring1_*列の確認
        ring1_cols = [col for col in features_df.columns if col.startswith('ring1_')]
        print(f"[L5] 生成されたring1_*列数: {len(ring1_cols)}")
        if ring1_cols:
            print(f"[L5] ring1_*列の例: {ring1_cols[:5]}")
    else:
        print("[L5][WARN] 重心データが利用できないため、空間ラグをスキップします")
    
    # 不足している特徴を追加
    features_df = add_missing_features(features_df)
    
    # exp_rate_terms列を追加（年次合計）
    features_df['exp_rate_terms'] = (
        features_df[["exp_rate_all_h1", "exp_rate_all_h2", "exp_rate_all_h3"]]
        .sum(axis=1)
        .fillna(0.0)
    )
    
    # ∞をNaNに置換
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # デバッグログ（exp_rate_termsの確認）
    print(f"[L5] exp_rate_terms サンプル: {features_df['exp_rate_terms'].head(3).tolist()}")
    
    # ゲート前の完全版データフレームを返す（CSV保存用）
    return features_df

def build_future_features_full(baseline: pd.DataFrame, future_events: pd.DataFrame, scenario: Dict[str, Any]) -> pd.DataFrame:
    """将来特徴の構築（完全版 - ゲート適用前）"""
    # 基本構造の作成
    features_df = future_events[["town", "year"]].copy()
    
    # 各特徴の計算
    features_df = calculate_expected_effects(future_events, baseline, scenario)
    features_df = calculate_macro_features(future_events, baseline, scenario.get("macros", {}))
    features_df = calculate_town_trend_features(future_events, baseline)
    
    # 空間ラグ特徴の計算
    centroids_path = "../../data/processed/town_centroids.csv"
    if os.path.exists(centroids_path):
        centroids_df = pd.read_csv(centroids_path)
        print(f"[L5] 空間重心データを読み込み: {len(centroids_df)}件")
        
        # ラグ対象列を検出
        cols_to_lag = detect_cols_to_lag(features_df)
        print(f"[L5] ラグ対象列: {cols_to_lag[:10]}...")
        
        # 空間ラグを計算
        features_df = calculate_spatial_lags_simple(
            features_df, centroids_df, cols_to_lag, 
            town_col="town", year_col="year", k_neighbors=5
        )
        
        # 生成されたring1_*列の確認
        ring1_cols = [col for col in features_df.columns if col.startswith('ring1_')]
        print(f"[L5] 生成されたring1_*列数: {len(ring1_cols)}")
        if ring1_cols:
            print(f"[L5] ring1_*列の例: {ring1_cols[:5]}")
    else:
        print("[L5][WARN] 重心データが利用できないため、空間ラグをスキップします")
    
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
    
    # 将来特徴の構築（完全版 - ゲート前）
    future_features = build_future_features(baseline, future_events, scenario)
    
    # 出力ディレクトリの作成
    Path(P_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    
    # 完全版をCSV保存（exp_rate_terms含む）
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
    
    # exp_rate_termsの確認
    if 'exp_rate_terms' in future_features.columns:
        non_zero_terms = (future_features['exp_rate_terms'] != 0).sum()
        print(f"[L5] exp_rate_terms nonzero rows = {non_zero_terms}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("使用方法: python build_future_features.py <baseline.csv> <future_events.csv> <scenario.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
