# -*- coding: utf-8 -*-
# src/layer4/build_expected_effects.py
"""
L3の係数とイベント行列から、L4用の「期待効果特徴」を作る。
出力: data/processed/features_l4.csv

設計:
- 入力:
  - data/processed/features_panel.csv           … パネル（既存）
  - data/processed/events_matrix_signed.csv     … 方向付きイベント（inc/dec, t/t1）
  - data/processed/effects_coefficients.csv     … L3 出力（12カテゴリ）
- 期待効果:
  h=1:  β_t  * event_*_t  + β_t1 * event_*_t1
  h=2:  DECAY_H2 * (上と同じ)
  h=3:  DECAY_H3 * (上と同じ)
  （単純減衰。必要なら係数を調整してください）
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os
# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from src.common.spatial import calculate_spatial_lags_simple, detect_cols_to_lag
from src.common.feature_gate import drop_excluded_columns

# ---- ハードコードパス ----
P_PANEL  = "subject3/data/processed/features_panel.csv"
P_EVENTS = "subject3/data/processed/events_matrix_signed.csv"
P_COEF   = "subject3/output/effects_coefficients.csv"
P_OUT    = "subject3/data/processed/features_l4.csv"
P_CENTROIDS = "subject3/data/processed/town_centroids.csv"

# ---- 減衰率（任意に調整可）----
DECAY_H2 = 0.5
DECAY_H3 = 0.25

def main():
    panel  = pd.read_csv(P_PANEL).sort_values(["town","year"])
    events = pd.read_csv(P_EVENTS)
    coef   = pd.read_csv(P_COEF)

    # 必須列チェック
    need = {"event_var","beta_t","beta_t1"}
    if not need.issubset(coef.columns):
        raise ValueError(f"effects_coefficients.csv に列不足: {need - set(coef.columns)}")

    # inc/dec×(t,t1)列だけを採用
    ev_cols = [c for c in events.columns if c.startswith("event_") and ("_inc_" in c or "_dec_" in c)]
    if not ev_cols:
        raise ValueError("events_matrix_signed.csv に inc/dec 列が見つかりません。")

    # 期待効果の作成
    # event_var は "disaster_inc" のような形を想定
    # イベント列は "event_disaster_inc_t", "event_disaster_inc_t1"
    expected = events[["town","year"]].copy()
    for ev in sorted(coef["event_var"].unique()):
        row = coef[coef["event_var"]==ev].iloc[0]
        b_t  = float(row["beta_t"])
        b_t1 = float(row["beta_t1"])
        col_t  = f"event_{ev}_t"
        col_t1 = f"event_{ev}_t1"
        # 列が無い場合は0扱い
        x_t  = events[col_t]  if col_t  in events.columns else 0.0
        x_t1 = events[col_t1] if col_t1 in events.columns else 0.0

        eff_h1 = b_t * x_t + b_t1 * x_t1
        eff_h2 = (DECAY_H2) * eff_h1
        eff_h3 = (DECAY_H3) * eff_h1

        expected[f"exp_{ev}_h1"] = eff_h1
        expected[f"exp_{ev}_h2"] = eff_h2
        expected[f"exp_{ev}_h3"] = eff_h3

    # 合計（全カテゴリの合算効果）も用意
    for h in [1,2,3]:
        hcols = [c for c in expected.columns if c.startswith("exp_") and c.endswith(f"_h{h}")]
        expected[f"exp_all_h{h}"] = expected[hcols].sum(axis=1)

    # パネルにマージして保存
    out = panel.merge(expected, on=["town","year"], how="left")
    out = out.fillna(0.0)
    
    # ==== ここから追記（保存直前） ====
    
    # 年度レジーム・フラグ
    out["era_post2009"]  = (out["year"] >= 2010).astype(int)   # 合併後期
    out["era_post2013"]  = (out["year"] >= 2013).astype(int)   # 政令市化後
    out["era_pre2013"]   = (out["year"] <  2013).astype(int)
    
    # 期待効果 × レジーム のインタラクション（全カテゴリ合算に限定して過学習を抑制）
    for h in [1,2,3]:
        base = f"exp_all_h{h}"
        if base in out.columns:
            out[f"{base}_pre2013"]  = out[base] * out["era_pre2013"]
            out[f"{base}_post2013"] = out[base] * out["era_post2013"]
            out[f"{base}_post2009"] = out[base] * out["era_post2009"]
    
    # 既に無ければターゲットとラグを用意（NaNはL4側で落とす）
    out = out.sort_values(["town","year"])
    if "delta_people" not in out.columns and "pop_total" in out.columns:
        out["delta_people"] = out.groupby("town")["pop_total"].diff()
    if "lag_d1" not in out.columns:
        out["lag_d1"] = out.groupby("town")["delta_people"].shift(1)
        out["lag_d2"] = out.groupby("town")["delta_people"].shift(2)
        out["rate_d1"] = out["lag_d1"] / np.maximum(100.0, out.groupby("town")["pop_total"].shift(1))
        out["ma2_delta"] = out.groupby("town")["delta_people"].rolling(2).mean().reset_index(0, drop=True)
    
    # --- ポスト2020（COVID期）ダミー & 期待効果との相互作用 ---
    out["era_covid"] = ((out["year"] >= 2020) & (out["year"] <= 2022)).astype(int)

    for h in [1,2,3]:
        base = f"exp_all_h{h}"
        if base in out.columns:
            out[f"{base}_covid"] = out[base] * out["era_covid"]

    # --- マクロショックの抽出（年合計Δの乖離） ---
    yr = out.groupby("year")["delta_people"].sum().reset_index().rename(columns={"delta_people":"macro_delta"})
    yr["macro_ma3"] = yr["macro_delta"].rolling(3, min_periods=1).mean()
    yr["macro_shock"] = yr["macro_delta"] - yr["macro_ma3"]  # その年だけの異常分
    out = out.merge(yr, on="year", how="left")

    # --- 町丁の直近5年トレンド（先読み防止に1期シフト） ---
    out = out.sort_values(["town","year"])
    def _slope(v):
        import numpy as np
        x = np.arange(len(v))
        if np.isfinite(v).sum() < 3: return np.nan
        A = np.vstack([x, np.ones(len(v))]).T
        m,_ = np.linalg.lstsq(A, np.nan_to_num(v), rcond=None)[0]
        return m
    out["town_trend5"] = out.groupby("town")["delta_people"].rolling(5, min_periods=3).apply(_slope, raw=False).reset_index(0, drop=True).shift(1)

    # 町丁の過去傾向（直近5年の移動平均・分散）→ 1期先読み防止のためシフト
    out["town_ma5"]  = out.groupby("town")["delta_people"].rolling(5, min_periods=2).mean().reset_index(0, drop=True).shift(1)
    out["town_std5"] = out.groupby("town")["delta_people"].rolling(5, min_periods=2).std().reset_index(0, drop=True).shift(1)

    out["macro_excl"] = out["macro_delta"] - out["delta_people"].fillna(0.0)
    
    # ===================== ここから追記：外国人マクロの正規化＋NaN方針込み =====================
    from pathlib import Path

    # 設定（必要に応じて切り替え可）
    FOREIGN_CSV_CANDIDATES = [
        "subject3/data/external/kumamoto_foreign_population_clean.csv",   # プロジェクト側の所定パス
        "/mnt/data/kumamoto_foreign_population_norm.csv",  # 手元確認用（存在すれば使う）
    ]
    FOREIGN_YEAR_MIN = int(out["year"].min()) if "year" in out.columns else 1999
    FOREIGN_YEAR_MAX = int(out["year"].max()) if "year" in out.columns else 2025

    # NaN埋め方針（"none"｜"ffill"｜"bfill"｜"zero"）
    # ※推奨は "none"（木モデルは NaN を自然に扱える & 0は誤含意になりやすい）
    FOREIGN_IMPUTE = "none"

    def _detect_col(cols, cands):
        for c in cols:
            if c in cands:
                return c
        return None

    def load_foreign_population_csv(path: str, year_min: int, year_max: int, impute: str = "none") -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        year_col = _detect_col(df.columns, ("year", "年度", "西暦", "year_ad"))
        pop_col  = _detect_col(df.columns, ("foreign_population", "外国人数", "外国人住民", "value", "count", "人数"))
        if year_col is None or pop_col is None:
            raise ValueError(f"[foreign] '{path}' に year/foreign_population 相当の列が見つかりません。")

        tmp = pd.DataFrame({
            "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
            "foreign_population": pd.to_numeric(df[pop_col], errors="coerce")
        }).dropna(subset=["year"]).sort_values("year").drop_duplicates("year", keep="last")

        # フルイヤーフレーム（例：1999..2025）
        full = pd.DataFrame({"year": np.arange(year_min, year_max + 1, dtype=int)})
        merged = full.merge(tmp, on="year", how="left")

        # --- NaN埋めポリシー ---
        if impute == "ffill":
            merged["foreign_population"] = merged["foreign_population"].ffill()
        elif impute == "bfill":
            merged["foreign_population"] = merged["foreign_population"].bfill()
        elif impute == "zero":
            merged["foreign_population"] = merged["foreign_population"].fillna(0.0)
        elif impute == "none":
            # 何もしない（NaNのまま）。木モデルで自然に分岐処理させるのが安全。
            pass
        else:
            raise ValueError(f"[foreign] unknown impute policy: {impute}")

        # --- 派生特徴（安全な演算） ---
        # 変化量
        merged["foreign_change"] = merged["foreign_population"].diff()

        # 伸び率（0除算は NaN に）
        denom = merged["foreign_population"].shift(1)
        denom = denom.where(~(denom == 0), np.nan)
        merged["foreign_pct_change"] = merged["foreign_change"] / denom

        # 対数（負値ガード）
        merged["foreign_log"] = np.log1p(merged["foreign_population"].clip(lower=0))

        # 3年移動平均（欠損は保持）
        merged["foreign_ma3"] = merged["foreign_population"].rolling(3, min_periods=1).mean()

        # 簡易QCログ
        n_nan = int(merged["foreign_population"].isna().sum())
        print(f"[foreign] loaded '{path}', years={year_min}..{year_max}, impute={impute}, NaN(pop)={n_nan}")

        return merged

    # 候補パスから最初に存在するCSVを使う
    _foreign_csv = next((p for p in FOREIGN_CSV_CANDIDATES if Path(p).exists()), None)
    if _foreign_csv is None:
        print("[foreign] WARN: 外国人住民CSVが見つかりません。追加特徴はスキップします。")
    else:
        fp = load_foreign_population_csv(
            _foreign_csv, year_min=FOREIGN_YEAR_MIN, year_max=FOREIGN_YEAR_MAX, impute=FOREIGN_IMPUTE
        )
        out = out.merge(fp, on="year", how="left")

    # ===================== 追記ここまで =====================
    
    # === 相互作用: 外国人マクロ × COVID期 ===
    # era_covid が未定義なら作る（2020〜2022）
    if "era_covid" not in out.columns:
        out["era_covid"] = ((out["year"] >= 2020) & (out["year"] <= 2022)).astype(int)

    for col in ["foreign_population", "foreign_change", "foreign_pct_change", "foreign_log", "foreign_ma3"]:
        if col in out.columns:
            out[f"{col}_covid"] = out[col] * out["era_covid"]
    
    # ===== post-2022 レジーム相互作用（軽パッチ） =====
    # 2023年以降を「回復局面」としてフラグ化
    if "era_post2022" not in out.columns:
        out["era_post2022"] = (out["year"] >= 2023).astype(int)

    # exp_all_h* × era_post2022
    for h in [1, 2, 3]:
        base = f"exp_all_h{h}"
        if base in out.columns:
            out[f"{base}_post2022"] = out[base] * out["era_post2022"]

    # 外国人マクロ × era_post2022
    _foreign_cols = ["foreign_population", "foreign_change", "foreign_pct_change", "foreign_log", "foreign_ma3"]
    for col in _foreign_cols:
        if col in out.columns:
            out[f"{col}_post2022"] = out[col] * out["era_post2022"]

    # （方針）NaNは埋めない：木モデルが自然に分岐で処理する
    # ===============================================
    
    # ==== 追記ここまで ====
    
    # ==== 空間ラグ特徴の追加 ====
    print("[L4] 空間ラグ特徴を計算中...")
    
    # 重心データの読み込み
    if Path(P_CENTROIDS).exists():
        centroids_df = pd.read_csv(P_CENTROIDS)
        print(f"[L4] 重心データを読み込み: {len(centroids_df)}件")
        
        # ラグ対象列の自動検出
        cols_to_lag = detect_cols_to_lag(out)
        print(f"[L4] ラグ対象列: {cols_to_lag[:10]}...")  # 最初の10列を表示
        
        # 空間ラグの計算（処理時間短縮のため、主要な列のみに限定）
        # 全列だと時間がかかりすぎるため、主要な期待効果列のみに限定
        main_cols_to_lag = [col for col in cols_to_lag if col.startswith('exp_all_') or col.startswith('exp_')]
        if len(main_cols_to_lag) > 20:  # 20列を超える場合は上位20列のみ
            main_cols_to_lag = main_cols_to_lag[:20]
        
        print(f"[L4] 空間ラグ対象列を制限: {len(main_cols_to_lag)}列（全{len(cols_to_lag)}列から）")
        
        out = calculate_spatial_lags_simple(
            out, 
            centroids_df, 
            main_cols_to_lag, 
            town_col="town", 
            year_col="year", 
            k_neighbors=5
        )
        
        # 生成されたring1_*列の確認
        ring1_cols = [col for col in out.columns if col.startswith('ring1_')]
        print(f"[L4] 生成されたring1_*列数: {len(ring1_cols)}")
        if ring1_cols:
            print(f"[L4] ring1_*列の例: {ring1_cols[:5]}")
    else:
        print(f"[L4][WARN] 重心データが見つかりません: {P_CENTROIDS}")
    
    # ==== 空間ラグ特徴の追加ここまで ====
    
    # ==== 特徴量ゲート適用（期待効果を除外） ====
    print("[L4] 特徴量ゲートを適用中...")
    out_kept, removed_cols = drop_excluded_columns(out)
    print(f"[L4] 除外された列数: {len(removed_cols)}")
    if removed_cols:
        print(f"[L4] 除外された列の例: {removed_cols[:10]}...")
    print(f"[L4] 残存列数: {len(out_kept.columns)}")
    
    # 除外後のデータで保存
    Path(P_OUT).parent.mkdir(parents=True, exist_ok=True)
    out_kept.to_csv(P_OUT, index=False)
    print(f"[L4] features_l4.csv saved: rows={len(out_kept)}, cols={len(out_kept.columns)}")

if __name__ == "__main__":
    main()
