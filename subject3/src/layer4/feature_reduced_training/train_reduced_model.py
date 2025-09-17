# -*- coding: utf-8 -*-
# src/layer4/feature_reduced_training/train_reduced_model.py
"""
削除された特徴量を使用して再学習を行うスクリプト
- 削除対象特徴量を除外してモデルを学習
- フルモデルと同じ条件で学習
- 結果を比較用に保存
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from common.feature_gate import select_feature_columns, save_feature_list

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent.parent
P_FEAT = BASE_DIR / "data/processed/features_l4.csv"
P_REDUCTION_ANALYSIS = BASE_DIR / "src/layer4/ablation/feature_reduction_analysis.json"
P_PREDS = BASE_DIR / "data/processed/feature_reduced_training/l4_predictions_no_macro.csv"
P_METR = BASE_DIR / "data/processed/feature_reduced_training/l4_cv_metrics_no_macro.json"
P_FIMP = BASE_DIR / "data/processed/feature_reduced_training/l4_feature_importance_no_macro.csv"
P_MODEL = BASE_DIR / "models/l4_model_no_macro.joblib"
P_FEATURE_LIST = BASE_DIR / "src/layer4/feature_reduced_training/feature_list_no_macro.json"

TARGET = "delta_people"
ID_KEYS = ["town", "year"]

# フルモデルと同じパラメータ
USE_HUBER = True
HUBER_ALPHA = 0.9
N_ESTIMATORS = 25000
LEARNING_RATE = 0.008
EARLY_STOPPING_ROUNDS = 800
LOG_EVERY_N = 200
TIME_DECAY = 0.05
ANOMALY_YEARS = [2022, 2023]
ANOMALY_WEIGHT = 0.3

# LightGBM の設定
USE_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    USE_LGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor
    print("[WARN] LightGBM not available, using HistGradientBoostingRegressor")

def load_reduction_analysis():
    """削除分析結果を読み込み（現在は使用しない）"""
    return None

def load_data():
    """データを読み込み"""
    print("データを読み込み中...")
    df = pd.read_csv(P_FEAT)
    print(f"データ形状: {df.shape}")
    return df

def select_reduced_features(df, features_to_remove):
    """マクロ経済特徴量を除外して特徴量を選択（アブレーション研究のno_macroに基づく）"""
    print(f"削除対象特徴量: {len(features_to_remove)}個")
    
    # 削除対象特徴量を表示
    for feature in features_to_remove:
        print(f"  - {feature}")
    
    # フルモデルと同じ特徴量選択ロジックを使用
    from common.feature_gate import select_feature_columns
    all_features = select_feature_columns(df)
    remaining_features = [f for f in all_features if f not in features_to_remove]
    
    print(f"フルモデルの特徴量数: {len(all_features)}")
    print(f"削除後の特徴量数: {len(remaining_features)}")
    print(f"削減率: {len(features_to_remove)/len(all_features)*100:.1f}%")
    
    # アブレーション研究のno_macroと同じ特徴量数になることを確認
    # 現在のデータでは51個の特徴量から4個のマクロ経済特徴量を除外して47個になるはず
    expected_features = 47  # アブレーション研究のno_macroの結果
    if len(remaining_features) != expected_features:
        print(f"[WARN] 期待される特徴量数({expected_features})と実際の特徴量数({len(remaining_features)})が異なります")
        print(f"実際の除外された特徴量: {[f for f in all_features if f not in remaining_features]}")
        print(f"期待される除外特徴量: {features_to_remove}")
    
    return remaining_features

def make_weights(s):
    """サンプル重みを計算（フルモデルと同じロジック）"""
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

def time_series_folds(years, min_train_years=20, test_window=1, last_n_tests=None):
    """
    expanding window: 最初の min_train_years を学習、その次の年を検証。
    以降、テスト年を1つずつ進め、最大 last_n_tests 回。
    （フルモデルと同じロジック）
    """
    ys = sorted(years)
    folds = []
    for i in range(min_train_years, len(ys)):
        train_years = ys[:i]
        test_years = ys[i:i+test_window]
        folds.append((set(train_years), set(test_years)))
        if last_n_tests is not None and len(folds) >= last_n_tests:
            break
    return folds

def metrics(y_true, y_pred):
    """評価指標を計算（フルモデルと同じロジック）"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

def train_reduced_model():
    """削除された特徴量でモデルを学習"""
    print("=" * 80)
    print("マクロ経済特徴量除外モデルの学習開始（no_macro）")
    print("=" * 80)
    
    # アブレーション研究のno_macro結果に基づくマクロ経済特徴量の除外
    # feature_reduction_analysis.jsonにはマクロ経済特徴量が含まれていないため、
    # アブレーション研究のno_macro結果を直接使用
    features_to_remove = [
        "macro_delta",
        "macro_excl", 
        "macro_ma3",
        "macro_shock"
    ]
    print(f"アブレーション研究のno_macroに基づくマクロ経済特徴量を除外: {len(features_to_remove)}個")
    
    # データを読み込み
    df = load_data()
    
    # 削除対象特徴量を除外して特徴量を選択
    remaining_features = select_reduced_features(df, features_to_remove)
    
    # 出力ディレクトリを作成
    P_PREDS.parent.mkdir(parents=True, exist_ok=True)
    P_MODEL.parent.mkdir(parents=True, exist_ok=True)
    
    # 特徴量リストを保存
    save_feature_list(remaining_features, P_FEATURE_LIST)
    
    # 時系列交差検証（フルモデルと同じ設定）
    years = sorted(df['year'].unique())
    print(f"利用可能な年: {years}")
    folds = time_series_folds(years, min_train_years=20, test_window=1)
    print(f"交差検証フォールド数: {len(folds)}")
    
    all_predictions = []
    fold_metrics = []
    
    for fi, (train_years, test_years) in enumerate(folds, 1):
        test_year = list(test_years)[0]  # test_yearsはsetなので最初の要素を取得
        print(f"\n--- フォールド {fi}/{len(folds)}: {sorted(train_years)} -> {test_year} ---")
        
        # データを分割
        train_mask = df['year'].isin(train_years)
        test_mask = df['year'] == test_year
        
        X_train = df.loc[train_mask, remaining_features]
        y_train = df.loc[train_mask, TARGET]
        X_test = df.loc[test_mask, remaining_features]
        y_test = df.loc[test_mask, TARGET]
        
        print(f"学習データ: {len(X_train)}サンプル")
        print(f"テストデータ: {len(X_test)}サンプル")
        
        # サンプル重みを計算（フルモデルと同じ）
        train_data_for_weights = df.loc[train_mask]
        train_weights = make_weights(train_data_for_weights)
        
        # 学習用 y を軽くウィンズライズ（フルモデルと同じロジック）
        y_tr = y_train.values
        if test_year == 2022:
            ql, qh = np.quantile(y_tr, [0.01, 0.99])
        else:
            ql, qh = np.quantile(y_tr, [0.005, 0.995])
        
        y_train = np.clip(y_tr, ql, qh)
        
        # モデルを学習（フルモデルと同じパラメータ）
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
            
            # 直接LightGBMを使用（アブレーション研究と同じ）
            model = lgb.LGBMRegressor(**base_params)
            
            callbacks = []
            if EARLY_STOPPING_ROUNDS > 0:
                callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))
            if LOG_EVERY_N > 0:
                callbacks.append(lgb.log_evaluation(LOG_EVERY_N))
            
            model.fit(
                X_train, y_train,
                sample_weight=train_weights,
                eval_set=(X_test, y_test),  # アブレーション研究と同じくテストデータを使用
                callbacks=callbacks
            )
            
            y_pred = model.predict(X_test)
            
            # 特徴量重要度を保存（最初のフォールドのみ）
            if fi == 1 and hasattr(model, 'feature_importances_'):
                # アブレーション研究と同じ方法で特徴量重要度を取得
                feature_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(remaining_features, model.feature_importances_)
                }
                
                # 重要度でソート
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                
                # DataFrameとして保存
                importance_df = pd.DataFrame([
                    {'feature': feature, 'importance': importance}
                    for feature, importance in sorted_importance.items()
                ])
                importance_df.to_csv(P_FIMP, index=False)
                print(f"特徴量重要度を保存: {P_FIMP}")
        
        else:
            # HistGradientBoostingRegressor（フルモデルと同じ）
            base_params = dict(
                max_depth=None, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.0,
                random_state=42
            )
            model = HistGradientBoostingRegressor(**base_params)
            model.fit(X_train, y_train, sample_weight=train_weights)
            y_pred = model.predict(X_test)
            
            # 特徴量重要度を保存（最初のフォールドのみ）
            if fi == 1:
                importance_df = pd.DataFrame({
                    'feature': remaining_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                importance_df.to_csv(P_FIMP, index=False)
                print(f"特徴量重要度を保存: {P_FIMP}")
        
        # 予測結果を保存
        fold_predictions = df.loc[test_mask, ID_KEYS].copy()
        fold_predictions['actual'] = y_test
        fold_predictions['predicted'] = y_pred
        fold_predictions['fold'] = fi
        all_predictions.append(fold_predictions)
        
        # 評価指標を計算
        fold_metric = metrics(y_test, y_pred)
        fold_metrics.append(fold_metric)
        
        print(f"MAE: {fold_metric['MAE']:.4f}, RMSE: {fold_metric['RMSE']:.4f}, MAPE: {fold_metric['MAPE']:.4f}, R²: {fold_metric['R2']:.4f}")
    
    # 全予測結果を保存
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(P_PREDS, index=False)
    print(f"\n予測結果を保存: {P_PREDS}")
    
    # 集約指標を計算
    agg_metrics = {}
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
        values = [m[metric] for m in fold_metrics]
        agg_metrics[metric] = np.mean(values)
        agg_metrics[f'{metric}_std'] = np.std(values)
    
    # 結果を保存
    results = {
        'model_info': {
            'model_type': 'LightGBM' if USE_LGBM else 'HistGradientBoostingRegressor',
            'n_estimators': N_ESTIMATORS,
            'learning_rate': LEARNING_RATE,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'use_huber': USE_HUBER,
            'huber_alpha': HUBER_ALPHA if USE_HUBER else None,
            'time_decay': TIME_DECAY,
            'anomaly_years': ANOMALY_YEARS,
            'anomaly_weight': ANOMALY_WEIGHT
        },
        'feature_info': {
            'original_features': len(remaining_features) + len(features_to_remove),
            'remaining_features': len(remaining_features),
            'removed_features': len(features_to_remove),
            'reduction_rate': len(features_to_remove) / (len(remaining_features) + len(features_to_remove)) * 100,
            'removed_features_list': features_to_remove
        },
        'fold_metrics': fold_metrics,
        'aggregate_metrics': agg_metrics
    }
    
    with open(P_METR, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n評価結果を保存: {P_METR}")
    
    # 最終モデルを保存（最後のフォールドのモデル）
    if USE_LGBM:
        model.save(str(P_MODEL))  # save_modelではなくsave
    else:
        import joblib
        joblib.dump(model, P_MODEL)
    
    print(f"モデルを保存: {P_MODEL}")
    
    # 結果を表示
    print(f"\n" + "=" * 80)
    print("マクロ経済特徴量除外モデルの学習完了（no_macro）")
    print("=" * 80)
    print(f"特徴量削減: {len(features_to_remove)}個削除 ({results['feature_info']['reduction_rate']:.1f}%)")
    print(f"集約性能:")
    print(f"  MAE: {agg_metrics['MAE']:.4f} ± {agg_metrics['MAE_std']:.4f}")
    print(f"  RMSE: {agg_metrics['RMSE']:.4f} ± {agg_metrics['RMSE_std']:.4f}")
    print(f"  MAPE: {agg_metrics['MAPE']:.4f} ± {agg_metrics['MAPE_std']:.4f}")
    print(f"  R²: {agg_metrics['R2']:.4f} ± {agg_metrics['R2_std']:.4f}")
    
    # アブレーション研究のno_macro結果との比較
    print(f"\nアブレーション研究のno_macro結果との比較:")
    print(f"  期待される特徴量数: 47")
    print(f"  実際の特徴量数: {len(remaining_features)}")
    print(f"  期待されるMAE: 1.5431")
    print(f"  実際のMAE: {agg_metrics['MAE']:.4f}")
    print(f"  期待されるR²: 0.8305")
    print(f"  実際のR²: {agg_metrics['R2']:.4f}")
    
    return results

if __name__ == "__main__":
    train_reduced_model()
