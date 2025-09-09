# -*- coding: utf-8 -*-
# src/layer4/train_lgbm.py
"""
L4 予測モデル: LightGBM（なければ sklearn HistGBR にフォールバック）
- 目的変数: delta_people
- 時系列CV（年でスプリット: expanding window）
- 出力:
  - data/processed/l4_predictions.csv
  - data/processed/l4_cv_metrics.json
  - models/l4_model.joblib
  - data/processed/l4_feature_importance.csv
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.feature_gate import select_feature_columns, save_feature_list

# パス
P_FEAT = "subject3/data/processed/features_l4.csv"
P_PREDS = "subject3/data/processed/l4_predictions.csv"
P_METR  = "subject3/data/processed/l4_cv_metrics.json"
P_FIMP  = "subject3/data/processed/l4_feature_importance.csv"
P_MODEL = "subject3/models/l4_model.joblib"

TARGET = "delta_people"     # 既存のターゲット列を使用
ID_KEYS = ["town","year"]

# === ファイル冒頭の定数付近に追加 ===
USE_HUBER = True     # 外れ値が目立つ年に強い推奨オプション
HUBER_ALPHA = 0.9

# かなり大きめにして EarlyStopping に任せる
N_ESTIMATORS = 20000
LEARNING_RATE = 0.01
EARLY_STOPPING_ROUNDS = 600
LOG_EVERY_N = 200

# LightGBM が無ければ HistGradientBoosting にフォールバック
USE_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    USE_LGBM = False
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# choose_features関数は削除（select_feature_columnsを使用）

def time_series_folds(years, min_train_years=20, test_window=1, last_n_tests=None):
    """
    expanding window: 最初の min_train_years を学習、その次の年を検証。
    以降、テスト年を1つずつ進め、最大 last_n_tests 回。
    """
    ys = sorted(years)
    folds = []
    for i in range(min_train_years, len(ys)):
        train_years = ys[:i]
        test_years  = ys[i:i+test_window]
        folds.append((set(train_years), set(test_years)))
        if last_n_tests is not None and len(folds) >= last_n_tests:
            break
    return folds

def metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # 古いsklearn対応
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, np.abs(y_true)))))  # 0割回避
    r2   = float(r2_score(y_true, y_pred))
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

def main():
    df = pd.read_csv(P_FEAT).sort_values(ID_KEYS)

    # --- 目的変数の整備 ---
    # なければ作成（各町丁で差分 → 先頭年は NaN になる）
    if TARGET not in df.columns:
        if "pop_total" not in df.columns:
            raise ValueError("features_l4.csv に delta_people も pop_total もありません。")
        df[TARGET] = df.groupby("town")["pop_total"].diff()

    # NaN/∞ を除去（学習の y に NaN は厳禁）
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df[TARGET].isna()].copy()

    # 型の安定化
    if df["year"].dtype.kind not in "iu":
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)

    # y を作った直後あたりに診断ログ
    print("[L4] rows after y-dropna:", len(df))
    print("[L4] years present:", sorted(df["year"].unique().tolist())[:5], "...", sorted(df["year"].unique().tolist())[-5:])

    # 年ごとのサンプル数を出力（早期診断用）
    year_counts = df.groupby("year").size().reset_index(name="n")
    year_counts.to_csv("subject3/data/processed/l4_year_counts.csv", index=False)

    # 特徴量選択（ゲート機能を使用）
    Xcols = select_feature_columns(df)
    print(f"[L4] 選択された特徴量数: {len(Xcols)}")
    
    # 特徴量リストを保存
    feature_list_path = "subject3/src/layer4/feature_list.json"
    save_feature_list(Xcols, feature_list_path)
    
    years = sorted(df["year"].unique().tolist())
    print(f"[L4] year span: {years[0]}..{years[-1]} (#years={len(years)})")
    if len(years) < 21:
        raise RuntimeError("年数が20未満のため、要件（20年以上）を満たせません。features_l4.csv の年レンジを確認してください。")

    folds = time_series_folds(years, min_train_years=20, test_window=1, last_n_tests=None)
    print(f"[L4] #folds={len(folds)}  first_train_len={len(sorted(list(folds[0][0])))}  first_test={sorted(list(folds[0][1]))}")

    all_preds = []
    fold_metrics = []

    # 学習器
    if USE_LGBM:
        base_params = dict(
            objective=("huber" if USE_HUBER else "regression_l1"),
            alpha=(HUBER_ALPHA if USE_HUBER else None),
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.3,
            num_leaves=63, min_child_samples=20,
            random_state=42, n_jobs=-1
        )
    else:
        base_params = dict(
            max_depth=None, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.0,
            random_state=42
        )

    # サンプル重み: 規模×時間減衰（最近をやや重視）
    time_decay = 0.0  # ← まず無効化して様子を見る
    def make_weights(s):
        w_pop  = np.sqrt(np.maximum(1.0, s["pop_total"].values)) if "pop_total" in s.columns else np.ones(len(s))
        w_time = (1.0 + time_decay) ** (s["year"].values - s["year"].min())
        w = w_pop * w_time
        w[~np.isfinite(w)] = 1.0
        return w

    # 時系列CV
    for fi, (train_years, test_years) in enumerate(folds, 1):
        tr = df[df["year"].isin(train_years)].copy()
        te = df[df["year"].isin(test_years)].copy()

        # ---- X の ∞ を NaN に置換（NaNは残す） ----
        tr_X = tr[Xcols].replace([np.inf, -np.inf], np.nan)
        te_X = te[Xcols].replace([np.inf, -np.inf], np.nan)
        sw = make_weights(tr)

        # 学習データが空になっていないか（安全装置）
        if len(tr_X) == 0 or len(te_X) == 0:
            print(f"[WARN] fold {fi}: empty train/test after NaN filtering. skip.")
            continue

        # 学習用 y を軽くウィンズライズ（上下0.5%）
        y_tr = tr[TARGET].values
        ql, qh = np.quantile(y_tr, [0.005, 0.995])
        y_tr = np.clip(y_tr, ql, qh)

        if USE_LGBM:
            model = lgb.LGBMRegressor(**base_params)
            model.fit(
                tr_X, y_tr,                     # ← y_tr は（あれば）Winsorized学習ラベル
                sample_weight=sw,
                eval_set=[(te_X, te[TARGET].values)],
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(LOG_EVERY_N)
                ]
            )
            pred = model.predict(te_X, num_iteration=model.best_iteration_)
            # fold1 のモデルを保存（本番用に使い回しやすい）
            if fi == len(folds):
                Path(Path(P_MODEL).parent).mkdir(parents=True, exist_ok=True)
                joblib.dump(model, P_MODEL)
        else:
            model = HistGradientBoostingRegressor(**base_params)
            model.fit(tr_X, y_tr)   # HistGBR は sample_weight 省略でも可
            pred = model.predict(te_X)
            if fi == len(folds):
                Path(Path(P_MODEL).parent).mkdir(parents=True, exist_ok=True)
                joblib.dump(model, P_MODEL)

        m = metrics(te[TARGET].values, pred)
        m["fold"]   = fi
        m["train_years"] = {
            "len": len(train_years),
            "first": int(min(train_years)),
            "last": int(max(train_years))
        }
        m["test_years"]  = sorted(list(test_years))
        fold_metrics.append(m)

        te_out = te[ID_KEYS + [TARGET]].copy()
        te_out["y_pred"] = pred
        te_out["fold"]   = fi
        all_preds.append(te_out)

    preds = pd.concat(all_preds, axis=0, ignore_index=True)
    Path(Path(P_PREDS).parent).mkdir(parents=True, exist_ok=True)
    preds.to_csv(P_PREDS, index=False)

    agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    out = {"folds": fold_metrics, "aggregate": agg, "features": Xcols, "use_lightgbm": USE_LGBM}
    Path(Path(P_METR).parent).mkdir(parents=True, exist_ok=True)
    Path(P_METR).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 特徴量重要度
    if USE_LGBM:
        fi = pd.DataFrame({"feature": Xcols, "gain": model.booster_.feature_importance(importance_type="gain")})
        fi = fi.sort_values("gain", ascending=False)
        fi.to_csv(P_FIMP, index=False)
    else:
        # HistGBRは直接の重要度取得が難しいため省略（必要ならPermutationで）
        pd.DataFrame({"feature": Xcols}).to_csv(P_FIMP, index=False)

    # === 末尾の保存処理の直前/直後に追記 ===
    # 予測の年別誤差テーブル
    per_year = preds.groupby("year").apply(
        lambda g: pd.Series({
            "MAE": float(np.mean(np.abs(g["delta_people"] - g["y_pred"]))),
            "MedianAE": float(np.median(np.abs(g["delta_people"] - g["y_pred"]))),
            "RMSE": float(np.sqrt(np.mean((g["delta_people"] - g["y_pred"])**2))),
            "n": int(len(g))
        })
    ).reset_index()
    per_year.to_csv("subject3/data/processed/l4_per_year_metrics.csv", index=False)

    # 上位外れ行
    preds["abs_err"] = (preds["delta_people"] - preds["y_pred"]).abs()
    preds["signed_err"] = preds["y_pred"] - preds["delta_people"]
    preds.sort_values("abs_err", ascending=False).head(200).to_csv(
        "subject3/data/processed/l4_top_errors.csv", index=False
    )

    # fold別メトリクスもCSVで保存
    pd.DataFrame(fold_metrics).to_csv("subject3/data/processed/l4_fold_metrics.csv", index=False)
    print("[L4] extras saved: l4_per_year_metrics.csv, l4_top_errors.csv, l4_fold_metrics.csv")

    print(f"[L4] Done. preds -> {P_PREDS}, metrics -> {P_METR}, model -> {P_MODEL}")

if __name__ == "__main__":
    main()
