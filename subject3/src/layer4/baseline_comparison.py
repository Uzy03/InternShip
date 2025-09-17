# -*- coding: utf-8 -*-
# src/layer4/baseline_comparison.py
"""
ベースライン比較スクリプト
- 昨年比（ランダムウォーク型）ベースライン
- 移動平均ベースライン
- LightGBMの結果と比較
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.feature_gate import select_feature_columns

# ファイルパス
P_FEAT = "data/processed/features_l4.csv"
P_LGBM_METRICS = "data/processed/l4_cv_metrics.json"
P_OUTPUT = "data/processed/baseline_comparison.json"

TARGET = "delta_people"
ID_KEYS = ["town", "year"]

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
    """評価指標を計算"""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, np.abs(y_true)))))
    r2 = float(r2_score(y_true, y_pred))
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

def random_walk_baseline(df, folds):
    """
    昨年比（ランダムウォーク型）ベースライン
    各町丁で前年のdelta_peopleをそのまま予測値とする
    """
    print("[ベースライン] 昨年比（ランダムウォーク型）を実行中...")
    
    all_preds = []
    fold_metrics = []
    
    for fi, (train_years, test_years) in enumerate(folds, 1):
        # テストデータを取得
        test_data = df[df["year"].isin(test_years)].copy()
        
        # 各町丁で前年のdelta_peopleを予測値とする
        predictions = []
        for _, row in test_data.iterrows():
            town = row['town']
            year = row['year']
            
            # 前年のデータを取得
            prev_year_data = df[(df['town'] == town) & (df['year'] == year - 1)]
            
            if len(prev_year_data) > 0:
                pred = prev_year_data[TARGET].iloc[0]
            else:
                # 前年データがない場合は0とする
                pred = 0.0
            
            predictions.append(pred)
        
        test_data['y_pred'] = predictions
        all_preds.append(test_data[ID_KEYS + [TARGET] + ['y_pred']].copy())
        
        # メトリクス計算
        m = metrics(test_data[TARGET].values, np.array(predictions))
        m["fold"] = fi
        m["train_years"] = {
            "len": len(train_years),
            "first": int(min(train_years)),
            "last": int(max(train_years))
        }
        m["test_years"] = sorted(list(test_years))
        fold_metrics.append(m)
    
    # 全予測結果を結合
    preds = pd.concat(all_preds, axis=0, ignore_index=True)
    
    # 集計メトリクス
    agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    
    return {
        "folds": fold_metrics,
        "aggregate": agg,
        "predictions": preds
    }

def moving_average_baseline(df, folds, window=3):
    """
    移動平均ベースライン
    各町丁で過去window年のdelta_peopleの平均を予測値とする
    """
    print(f"[ベースライン] 移動平均（{window}年）を実行中...")
    
    all_preds = []
    fold_metrics = []
    
    for fi, (train_years, test_years) in enumerate(folds, 1):
        # テストデータを取得
        test_data = df[df["year"].isin(test_years)].copy()
        
        # 各町丁で過去window年の平均を予測値とする
        predictions = []
        for _, row in test_data.iterrows():
            town = row['town']
            year = row['year']
            
            # 過去window年のデータを取得
            past_years = [year - i for i in range(1, window + 1)]
            past_data = df[(df['town'] == town) & (df['year'].isin(past_years))]
            
            if len(past_data) > 0:
                pred = past_data[TARGET].mean()
            else:
                # 過去データがない場合は0とする
                pred = 0.0
            
            predictions.append(pred)
        
        test_data['y_pred'] = predictions
        all_preds.append(test_data[ID_KEYS + [TARGET] + ['y_pred']].copy())
        
        # メトリクス計算
        m = metrics(test_data[TARGET].values, np.array(predictions))
        m["fold"] = fi
        m["train_years"] = {
            "len": len(train_years),
            "first": int(min(train_years)),
            "last": int(max(train_years))
        }
        m["test_years"] = sorted(list(test_years))
        fold_metrics.append(m)
    
    # 全予測結果を結合
    preds = pd.concat(all_preds, axis=0, ignore_index=True)
    
    # 集計メトリクス
    agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    
    return {
        "folds": fold_metrics,
        "aggregate": agg,
        "predictions": preds
    }

def ema_baseline(df, folds, alpha=0.5):
    """
    EMA（単純指数平滑）ベースライン
    各町丁で人口変化率のEMAを計算し、前年人口に適用して予測
    """
    print(f"[ベースライン] EMA（α={alpha}）を実行中...")
    
    all_preds = []
    fold_metrics = []
    epsilon = 10.0  # ゼロ割対策の閾値
    
    for fi, (train_years, test_years) in enumerate(folds, 1):
        # テストデータを取得
        test_data = df[df["year"].isin(test_years)].copy()
        
        # 各町丁でEMAを計算して予測
        predictions = []
        for _, row in test_data.iterrows():
            town = row['town']
            year = row['year']
            
            # 訓練期間のデータを取得（年順にソート）
            train_data = df[(df['town'] == town) & (df['year'].isin(train_years))].sort_values('year')
            
            if len(train_data) < 2:
                # 訓練データが不足している場合は0とする
                pred = 0.0
            else:
                # 人口データを取得
                pop_data = train_data['pop_total'].values
                
                # 変化率を計算（ゼロ割対策付き）
                rates = []
                for i in range(1, len(pop_data)):
                    prev_pop = max(pop_data[i-1], epsilon)
                    rate = (pop_data[i] - pop_data[i-1]) / prev_pop
                    rates.append(rate)
                
                if len(rates) == 0:
                    pred = 0.0
                else:
                    # EMAを計算
                    ema_rate = rates[0]  # 初期値
                    for rate in rates[1:]:
                        ema_rate = alpha * rate + (1 - alpha) * ema_rate
                    
                    # 前年人口に適用して予測
                    prev_pop = pop_data[-1]
                    pred = prev_pop * ema_rate
            
            predictions.append(pred)
        
        test_data['y_pred'] = predictions
        all_preds.append(test_data[ID_KEYS + [TARGET] + ['y_pred']].copy())
        
        # メトリクス計算
        m = metrics(test_data[TARGET].values, np.array(predictions))
        m["fold"] = fi
        m["train_years"] = {
            "len": len(train_years),
            "first": int(min(train_years)),
            "last": int(max(train_years))
        }
        m["test_years"] = sorted(list(test_years))
        fold_metrics.append(m)
    
    # 全予測結果を結合
    preds = pd.concat(all_preds, axis=0, ignore_index=True)
    
    # 集計メトリクス
    agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    
    return {
        "folds": fold_metrics,
        "aggregate": agg,
        "predictions": preds
    }

def drift_naive_baseline(df, folds, k=3):
    """
    ドリフト付きナイーブベースライン
    各町丁で直近k年の平均増分を計算し、前年人口に加算して予測
    （MovingAverageとの違い：直近k年のみを使用、重み付けなし）
    """
    print(f"[ベースライン] ドリフト付きナイーブ（k={k}年）を実行中...")
    
    all_preds = []
    fold_metrics = []
    
    for fi, (train_years, test_years) in enumerate(folds, 1):
        # テストデータを取得
        test_data = df[df["year"].isin(test_years)].copy()
        
        # 各町丁でドリフト付きナイーブを計算して予測
        predictions = []
        for _, row in test_data.iterrows():
            town = row['town']
            year = row['year']
            
            # 訓練期間のデータを取得（年順にソート）
            train_data = df[(df['town'] == town) & (df['year'].isin(train_years))].sort_values('year')
            
            if len(train_data) < k + 1:
                # 訓練データが不足している場合は0とする
                pred = 0.0
            else:
                # 人口データを取得
                pop_data = train_data['pop_total'].values
                
                # 直近k年の平均増分を計算
                deltas = []
                for i in range(len(pop_data) - k, len(pop_data)):
                    if i > 0:
                        delta = pop_data[i] - pop_data[i-1]
                        deltas.append(delta)
                
                if len(deltas) == 0:
                    pred = 0.0
                else:
                    # 平均増分を計算
                    avg_delta = np.mean(deltas)
                    
                    # 前年人口に平均増分を加算した結果の増分を予測
                    # つまり、平均増分をそのまま予測値とする（delta_peopleの予測）
                    pred = avg_delta
            
            predictions.append(pred)
        
        test_data['y_pred'] = predictions
        all_preds.append(test_data[ID_KEYS + [TARGET] + ['y_pred']].copy())
        
        # メトリクス計算
        m = metrics(test_data[TARGET].values, np.array(predictions))
        m["fold"] = fi
        m["train_years"] = {
            "len": len(train_years),
            "first": int(min(train_years)),
            "last": int(max(train_years))
        }
        m["test_years"] = sorted(list(test_years))
        fold_metrics.append(m)
    
    # 全予測結果を結合
    preds = pd.concat(all_preds, axis=0, ignore_index=True)
    
    # 集計メトリクス
    agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    
    return {
        "folds": fold_metrics,
        "aggregate": agg,
        "predictions": preds
    }

def load_lightgbm_results():
    """LightGBMの結果を読み込み"""
    print("[ベースライン] LightGBMの結果を読み込み中...")
    
    with open(P_LGBM_METRICS, 'r', encoding='utf-8') as f:
        lgbm_results = json.load(f)
    
    return lgbm_results

def compare_methods(lgbm_results, random_walk_results, moving_avg_results, ema_results, drift_naive_results):
    """5つの手法の結果を比較"""
    print("[ベースライン] 結果を比較中...")
    
    comparison = {
        "methods": {
            "LightGBM": {
                "aggregate": lgbm_results["aggregate"],
                "description": "LightGBM機械学習モデル"
            },
            "RandomWalk": {
                "aggregate": random_walk_results["aggregate"],
                "description": "昨年比（ランダムウォーク型）ベースライン"
            },
            "MovingAverage": {
                "aggregate": moving_avg_results["aggregate"],
                "description": "移動平均（3年）ベースライン"
            },
            "EMA": {
                "aggregate": ema_results["aggregate"],
                "description": "EMA（単純指数平滑）ベースライン"
            },
            "DriftNaive": {
                "aggregate": drift_naive_results["aggregate"],
                "description": "ドリフト付きナイーブ（3年）ベースライン"
            }
        },
        "comparison_summary": {
            "best_mae": min([
                ("LightGBM", lgbm_results["aggregate"]["MAE"]),
                ("RandomWalk", random_walk_results["aggregate"]["MAE"]),
                ("MovingAverage", moving_avg_results["aggregate"]["MAE"]),
                ("EMA", ema_results["aggregate"]["MAE"]),
                ("DriftNaive", drift_naive_results["aggregate"]["MAE"])
            ], key=lambda x: x[1]),
            "best_rmse": min([
                ("LightGBM", lgbm_results["aggregate"]["RMSE"]),
                ("RandomWalk", random_walk_results["aggregate"]["RMSE"]),
                ("MovingAverage", moving_avg_results["aggregate"]["RMSE"]),
                ("EMA", ema_results["aggregate"]["RMSE"]),
                ("DriftNaive", drift_naive_results["aggregate"]["RMSE"])
            ], key=lambda x: x[1]),
            "best_mape": min([
                ("LightGBM", lgbm_results["aggregate"]["MAPE"]),
                ("RandomWalk", random_walk_results["aggregate"]["MAPE"]),
                ("MovingAverage", moving_avg_results["aggregate"]["MAPE"]),
                ("EMA", ema_results["aggregate"]["MAPE"]),
                ("DriftNaive", drift_naive_results["aggregate"]["MAPE"])
            ], key=lambda x: x[1]),
            "best_r2": max([
                ("LightGBM", lgbm_results["aggregate"]["R2"]),
                ("RandomWalk", random_walk_results["aggregate"]["R2"]),
                ("MovingAverage", moving_avg_results["aggregate"]["R2"]),
                ("EMA", ema_results["aggregate"]["R2"]),
                ("DriftNaive", drift_naive_results["aggregate"]["R2"])
            ], key=lambda x: x[1])
        }
    }
    
    # 改善率を計算
    lgbm_mae = lgbm_results["aggregate"]["MAE"]
    rw_mae = random_walk_results["aggregate"]["MAE"]
    ma_mae = moving_avg_results["aggregate"]["MAE"]
    ema_mae = ema_results["aggregate"]["MAE"]
    dn_mae = drift_naive_results["aggregate"]["MAE"]
    
    comparison["improvement_vs_baselines"] = {
        "LightGBM_vs_RandomWalk_MAE": {
            "improvement_pct": ((rw_mae - lgbm_mae) / rw_mae) * 100,
            "description": "LightGBMのMAE改善率（RandomWalk比）"
        },
        "LightGBM_vs_MovingAverage_MAE": {
            "improvement_pct": ((ma_mae - lgbm_mae) / ma_mae) * 100,
            "description": "LightGBMのMAE改善率（MovingAverage比）"
        },
        "LightGBM_vs_EMA_MAE": {
            "improvement_pct": ((ema_mae - lgbm_mae) / ema_mae) * 100,
            "description": "LightGBMのMAE改善率（EMA比）"
        },
        "LightGBM_vs_DriftNaive_MAE": {
            "improvement_pct": ((dn_mae - lgbm_mae) / dn_mae) * 100,
            "description": "LightGBMのMAE改善率（DriftNaive比）"
        }
    }
    
    return comparison

def main():
    """メイン処理"""
    print("=== ベースライン比較を開始 ===")
    
    # データ読み込み
    df = pd.read_csv(P_FEAT).sort_values(ID_KEYS)
    
    # 目的変数の整備
    if TARGET not in df.columns:
        if "pop_total" not in df.columns:
            raise ValueError("features_l4.csv に delta_people も pop_total もありません。")
        df[TARGET] = df.groupby("town")["pop_total"].diff()
    
    # NaN/∞ を除去
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df[TARGET].isna()].copy()
    
    # 年データの準備
    years = sorted(df["year"].unique().tolist())
    print(f"[ベースライン] 年範囲: {years[0]}..{years[-1]} (#years={len(years)})")
    
    if len(years) < 21:
        raise RuntimeError("年数が20未満のため、要件（20年以上）を満たせません。")
    
    # クロスバリデーションの設定
    folds = time_series_folds(years, min_train_years=20, test_window=1, last_n_tests=None)
    print(f"[ベースライン] #folds={len(folds)}")
    
    # 各ベースラインを実行
    random_walk_results = random_walk_baseline(df, folds)
    moving_avg_results = moving_average_baseline(df, folds, window=3)
    ema_results = ema_baseline(df, folds, alpha=0.5)
    drift_naive_results = drift_naive_baseline(df, folds, k=3)
    
    # LightGBMの結果を読み込み
    lgbm_results = load_lightgbm_results()
    
    # 結果を比較
    comparison = compare_methods(lgbm_results, random_walk_results, moving_avg_results, ema_results, drift_naive_results)
    
    # 結果を保存
    Path(Path(P_OUTPUT).parent).mkdir(parents=True, exist_ok=True)
    with open(P_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 結果を表示
    print("\n=== 比較結果 ===")
    print(f"LightGBM MAE: {lgbm_results['aggregate']['MAE']:.4f}")
    print(f"RandomWalk MAE: {random_walk_results['aggregate']['MAE']:.4f}")
    print(f"MovingAverage MAE: {moving_avg_results['aggregate']['MAE']:.4f}")
    print(f"EMA MAE: {ema_results['aggregate']['MAE']:.4f}")
    print(f"DriftNaive MAE: {drift_naive_results['aggregate']['MAE']:.4f}")
    
    print(f"\nLightGBM RMSE: {lgbm_results['aggregate']['RMSE']:.4f}")
    print(f"RandomWalk RMSE: {random_walk_results['aggregate']['RMSE']:.4f}")
    print(f"MovingAverage RMSE: {moving_avg_results['aggregate']['RMSE']:.4f}")
    print(f"EMA RMSE: {ema_results['aggregate']['RMSE']:.4f}")
    print(f"DriftNaive RMSE: {drift_naive_results['aggregate']['RMSE']:.4f}")
    
    print(f"\nLightGBM R2: {lgbm_results['aggregate']['R2']:.4f}")
    print(f"RandomWalk R2: {random_walk_results['aggregate']['R2']:.4f}")
    print(f"MovingAverage R2: {moving_avg_results['aggregate']['R2']:.4f}")
    print(f"EMA R2: {ema_results['aggregate']['R2']:.4f}")
    print(f"DriftNaive R2: {drift_naive_results['aggregate']['R2']:.4f}")
    
    print(f"\n=== 改善率 ===")
    print(f"LightGBM vs RandomWalk MAE改善率: {comparison['improvement_vs_baselines']['LightGBM_vs_RandomWalk_MAE']['improvement_pct']:.2f}%")
    print(f"LightGBM vs MovingAverage MAE改善率: {comparison['improvement_vs_baselines']['LightGBM_vs_MovingAverage_MAE']['improvement_pct']:.2f}%")
    print(f"LightGBM vs EMA MAE改善率: {comparison['improvement_vs_baselines']['LightGBM_vs_EMA_MAE']['improvement_pct']:.2f}%")
    print(f"LightGBM vs DriftNaive MAE改善率: {comparison['improvement_vs_baselines']['LightGBM_vs_DriftNaive_MAE']['improvement_pct']:.2f}%")
    
    print(f"\n結果を保存しました: {P_OUTPUT}")

if __name__ == "__main__":
    main()
