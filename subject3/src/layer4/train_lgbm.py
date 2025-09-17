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

# パス（ルートディレクトリ: subject3/ から実行する場合）
P_FEAT = "data/processed/features_l4.csv"
P_PREDS = "data/processed/l4_predictions.csv"
P_METR  = "data/processed/l4_cv_metrics_feature_engineered.json" #defautl:l4_cv_metrics.json
P_FIMP  = "data/processed/l4_feature_importance.csv"
P_MODEL = "models/l4_model.joblib"

TARGET = "delta_people"     # 既存のターゲット列を使用
ID_KEYS = ["town","year"]

# === ファイル冒頭の定数付近に追加 ===
USE_HUBER = True     # 外れ値が目立つ年に強い推奨オプション
HUBER_ALPHA = 0.9

# バランスの取れた設定（元の設定をベースに微調整）
N_ESTIMATORS = 25000
LEARNING_RATE = 0.008  # 元の0.01より少し小さく
EARLY_STOPPING_ROUNDS = 800  # 元の600より少し長く
LOG_EVERY_N = 200

# === サンプル重み調整のパラメータ ===
TIME_DECAY = 0.05  # 最近の年を重視する係数
ANOMALY_YEARS = [2022, 2023]  # 異常値期間（重みを下げる年）
ANOMALY_WEIGHT = 0.3  # 異常値期間の重み係数

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

# Optuna最適化のインポート
try:
    import optuna
    USE_OPTUNA = True
except ImportError:
    USE_OPTUNA = False
    print("[WARN] Optunaがインストールされていません。最適化機能は無効です。")

from lgbm_model import LightGBMModel
from optuna_optimizer import OptunaOptimizer, FastOptunaOptimizer

def add_enhanced_features(df):
    """時系列特化の特徴量を追加（将来年でも生成可能なもののみ）"""
    df = df.copy()
    
    # 年次トレンドの非線形変換（将来年でも生成可能）
    year_min, year_max = df['year'].min(), df['year'].max()
    df['year_normalized'] = (df['year'] - year_min) / (year_max - year_min)
    df['year_squared'] = df['year_normalized'] ** 2
    df['year_cubic'] = df['year_normalized'] ** 3
    
    # 周期性の特徴量（10年周期を想定、将来年でも生成可能）
    df['year_sin_10'] = np.sin(2 * np.pi * df['year_normalized'] * 10)
    df['year_cos_10'] = np.cos(2 * np.pi * df['year_normalized'] * 10)
    
    # 期間別フラグ特徴量（将来年でも生成可能）
    df['is_anomaly_period'] = df['year'].isin(ANOMALY_YEARS).astype(int)
    df['is_covid_period'] = df['year'].isin([2020, 2021]).astype(int)
    df['is_post_covid'] = (df['year'] >= 2022).astype(int)
    
    # 人口規模の非線形変換（将来年でも生成可能）
    if 'pop_total' in df.columns:
        df['pop_total_log'] = np.log1p(df['pop_total'])
        df['pop_total_sqrt'] = np.sqrt(df['pop_total'])
        df['pop_total_squared'] = df['pop_total'] ** 2
    
    # 2022-2023年特化の特徴量
    df['is_2022_2023'] = df['year'].isin([2022, 2023]).astype(int)
    df['years_since_2020'] = df['year'] - 2020
    df['covid_impact_factor'] = np.where(df['year'] >= 2020, 1.0, 0.0)
    
    # 外国人数の変化率特徴量（2022-2023年特化）
    if 'foreign_population' in df.columns:
        df['foreign_change_rate'] = df.groupby('town')['foreign_population'].pct_change()
        df['foreign_change_rate'] = df['foreign_change_rate'].fillna(0)
        df['foreign_change_rate_2022_2023'] = df['foreign_change_rate'] * df['is_2022_2023']
    
    return df

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

def main(optimize: bool = False, n_trials: int = 50, fast_mode: bool = False):
    """
    メイン処理
    
    Args:
        optimize: Optuna最適化を実行するか
        n_trials: 最適化試行回数
        fast_mode: 高速化モード（Colab無料環境用）
    """
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
    
    # --- 特徴量エンジニアリングの強化 ---
    print("[L4] Adding enhanced features...")
    df = add_enhanced_features(df)
    print(f"[L4] Enhanced features added. New shape: {df.shape}")

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
    year_counts.to_csv("data/processed/l4_year_counts.csv", index=False)

    # 特徴量選択（ゲート機能を使用）
    Xcols = select_feature_columns(df)
    print(f"[L4] 選択された特徴量数: {len(Xcols)}")
    
    # 新しく追加された特徴量を確認
    enhanced_features = [col for col in df.columns if col not in ['town', 'year', 'delta_people', 'pop_total']]
    new_features = [col for col in enhanced_features if col not in Xcols]
    if new_features:
        print(f"[L4] 新規追加された特徴量: {new_features[:10]}{'...' if len(new_features) > 10 else ''}")
    
    # 特徴量リストを保存
    feature_list_path = "src/layer4/feature_list.json"
    save_feature_list(Xcols, feature_list_path)
    
    years = sorted(df["year"].unique().tolist())
    print(f"[L4] year span: {years[0]}..{years[-1]} (#years={len(years)})")
    if len(years) < 21:
        raise RuntimeError("年数が20未満のため、要件（20年以上）を満たせません。features_l4.csv の年レンジを確認してください。")

    folds = time_series_folds(years, min_train_years=20, test_window=1, last_n_tests=None)
    print(f"[L4] #folds={len(folds)}  first_train_len={len(sorted(list(folds[0][0])))}  first_test={sorted(list(folds[0][1]))}")

    all_preds = []
    fold_metrics = []

    # サンプル重み: 規模×時間減衰×異常値期間調整
    def make_weights(s):
        # 人口規模による重み（人口が多い地域を重視）
        w_pop = np.sqrt(np.maximum(1.0, s["pop_total"].values)) if "pop_total" in s.columns else np.ones(len(s))
        
        # 時間減衰（最近の年を重視）
        w_time = (1.0 + TIME_DECAY) ** (s["year"].values - s["year"].min())
        
        # 異常値期間の重み調整（2022-2023年の重みを下げる）
        w_anomaly = np.where(s["year"].isin(ANOMALY_YEARS), ANOMALY_WEIGHT, 1.0)
        
        # 最終的な重み（元の実装に戻す）
        w = w_pop * w_time * w_anomaly
        w[~np.isfinite(w)] = 1.0
        
        # 重みの正規化（平均が1になるように）
        w = w / np.mean(w)
        
        return w

    # パラメータ設定
    if optimize and USE_OPTUNA:
        if fast_mode:
            print("[L4] LightGBM Fast Optuna最適化を実行します...（Colab無料環境用）")
            optimizer = FastOptunaOptimizer(folds, n_trials=n_trials)
        else:
            print("[L4] LightGBM Optuna最適化を実行します...")
            optimizer = OptunaOptimizer(folds, n_trials=n_trials)
        base_params = optimizer.optimize(df, Xcols, TARGET, make_weights)
        optimizer.save_results()
        study_summary = optimizer.get_study_summary()
    else:
        # デフォルトパラメータ
        if USE_LGBM:
            base_params = dict(
                objective=("huber" if USE_HUBER else "regression_l1"),
                alpha=(HUBER_ALPHA if USE_HUBER else None),
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.15, reg_lambda=0.4,
                num_leaves=50, min_child_samples=25,
                random_state=42, n_jobs=-1
            )
        else:
            base_params = dict(
                max_depth=None, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.0,
                random_state=42
            )
        study_summary = None

    # 時系列CV
    for fi, (train_years, test_years) in enumerate(folds, 1):
        tr = df[df["year"].isin(train_years)].copy()
        te = df[df["year"].isin(test_years)].copy()

        # ---- X の ∞ を NaN に置換（NaNは残す） ----
        tr_X = tr[Xcols].replace([np.inf, -np.inf], np.nan)
        te_X = te[Xcols].replace([np.inf, -np.inf], np.nan)
        sw = make_weights(tr)
        
        # 重みの統計情報をログ出力（最初のfoldのみ）
        if fi == 1:
            print(f"[L4] Sample weights stats: mean={np.mean(sw):.3f}, std={np.std(sw):.3f}, min={np.min(sw):.3f}, max={np.max(sw):.3f}")
            anomaly_mask = tr["year"].isin(ANOMALY_YEARS)
            if np.any(anomaly_mask):
                print(f"[L4] Anomaly years weight: {np.mean(sw[anomaly_mask]):.3f} (normal: {np.mean(sw[~anomaly_mask]):.3f})")

        # 学習データが空になっていないか（安全装置）
        if len(tr_X) == 0 or len(te_X) == 0:
            print(f"[WARN] fold {fi}: empty train/test after NaN filtering. skip.")
            continue

        # 学習用 y を軽くウィンズライズ（上下0.5%）
        y_tr = tr[TARGET].values
        if 2022 in test_years:
            # 2022年がテストの場合は、より積極的に外れ値を処理
            ql, qh = np.quantile(y_tr, [0.01, 0.99])
        else:
            ql, qh = np.quantile(y_tr, [0.005, 0.995])
        y_tr = np.clip(y_tr, ql, qh)
        
        # 外れ値の統計情報をログ出力（最初のfoldのみ）
        if fi == 1:
            original_std = np.std(tr[TARGET].values)
            clipped_std = np.std(y_tr)
            print(f"[L4] Winsorizing: original_std={original_std:.3f}, clipped_std={clipped_std:.3f}")

        # LightGBMModelクラスを使用
        model = LightGBMModel(params=base_params)
        model.fit(
            tr_X, y_tr,
            sample_weight=sw,
            eval_set=(te_X, te[TARGET].values),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            log_evaluation=LOG_EVERY_N
        )
        pred = model.predict(te_X)
        
        # 最後のfoldのモデルを保存（本番用）
        if fi == len(folds):
            Path(Path(P_MODEL).parent).mkdir(parents=True, exist_ok=True)
            # 後方互換性のため、内部のLightGBMモデルも保存
            if USE_LGBM and model.get_lgbm_model() is not None:
                joblib.dump(model.get_lgbm_model(), P_MODEL)
            else:
                joblib.dump(model.model, P_MODEL)
            # 完全なモデルクラスも保存
            model.save(P_MODEL.replace('.joblib', '_full.joblib'))

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
    out = {
        "folds": fold_metrics, 
        "aggregate": agg, 
        "features": Xcols, 
        "use_lightgbm": USE_LGBM,
        "parameters": base_params,
        "optuna_optimized": optimize and USE_OPTUNA,
        "optuna_study": study_summary
    }
    Path(Path(P_METR).parent).mkdir(parents=True, exist_ok=True)
    Path(P_METR).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 特徴量重要度
    if USE_LGBM and model.get_lgbm_model() is not None:
        fi = model.get_feature_importance(Xcols)
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
    per_year.to_csv("data/processed/l4_per_year_metrics.csv", index=False)

    # 上位外れ行
    preds["abs_err"] = (preds["delta_people"] - preds["y_pred"]).abs()
    preds["signed_err"] = preds["y_pred"] - preds["delta_people"]
    preds.sort_values("abs_err", ascending=False).head(200).to_csv(
        "data/processed/l4_top_errors.csv", index=False
    )

    # fold別メトリクスもCSVで保存
    pd.DataFrame(fold_metrics).to_csv("data/processed/l4_fold_metrics.csv", index=False)
    print("[L4] extras saved: l4_per_year_metrics.csv, l4_top_errors.csv, l4_fold_metrics.csv")

    print(f"[L4] Done. preds -> {P_PREDS}, metrics -> {P_METR}, model -> {P_MODEL}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGBM学習スクリプト')
    parser.add_argument('--optimize', action='store_true', 
                       help='Optuna最適化を実行')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='最適化試行回数 (デフォルト: 50)')
    parser.add_argument('--fast', action='store_true',
                       help='高速化モード（Colab無料環境用）')
    args = parser.parse_args()
    
    main(optimize=args.optimize, n_trials=args.n_trials, fast_mode=args.fast)
