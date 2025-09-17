# -*- coding: utf-8 -*-
# src/layer4/improved/train_causal_lgbm.py
"""
因果関係学習用LightGBMモデル
- 生のイベントデータを直接使用
- 真の因果関係を学習
- 従来の期待効果ベースモデルと性能比較

設計:
- 入力: features_causal.csv
- 出力: 因果関係学習済みモデル
- 特徴量選択: 生のイベントデータ + 時空間特徴量 + 相互作用特徴量
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from src.common.feature_gate import select_feature_columns, save_feature_list

# LightGBM のインポート
try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_LGBM = False
    print("[WARN] LightGBMが利用できません。HistGradientBoostingを使用します。")

# パス設定（improved/配下から相対パス）
P_FEATURES = "../../../data/processed/features_causal.csv"
P_PREDICTIONS = "../../../data/processed/l4_causal_predictions.csv"
P_METRICS = "../../../data/processed/l4_causal_metrics.json"
P_FEATURE_IMPORTANCE = "../../../data/processed/l4_causal_feature_importance.csv"
P_MODEL = "../../../models/l4_causal_model.joblib"
P_FEATURE_LIST = "causal_feature_list.json"

# ターゲット変数
TARGET = "delta_people"
ID_KEYS = ["town", "year"]

# モデルパラメータ
N_ESTIMATORS = 30000
LEARNING_RATE = 0.005
EARLY_STOPPING_ROUNDS = 1000
LOG_EVERY_N = 500

# 時系列CV設定
N_SPLITS = 5
TEST_SIZE = 3  # 最後の3年をテスト用に確保

def load_data():
    """データの読み込み"""
    print("[因果学習] データを読み込み中...")
    
    if not Path(P_FEATURES).exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {P_FEATURES}")
    
    df = pd.read_csv(P_FEATURES)
    print(f"[因果学習] データ形状: {df.shape}")
    
    return df

def select_causal_features(df):
    """因果関係学習用の特徴量を選択"""
    print("[因果学習] 因果関係学習用特徴量を選択中...")
    
    # 除外する列
    exclude_cols = [
        "town", "year", TARGET,
        "pop_total", "male", "female",  # 生の人口データ（ターゲットとの相関が高い）
        "delta", "growth_pct", "growth_log",  # ターゲットと直接関連
        "city_pop", "city_growth_log",  # 市全体のデータ
    ]
    
    # 除外パターン
    exclude_patterns = [
        "^exp_",  # 従来の期待効果特徴量
        "^ring1_exp_",  # 従来の空間期待効果特徴量
        "_roll_",  # 移動平均（ターゲットとの相関が高い）
        "_accel",  # 加速度（ターゲットとの相関が高い）
        "_cum3",  # 累積（ターゲットとの相関が高い）
        "_trend_",  # トレンド（ターゲットとの相関が高い）
    ]
    
    # 特徴量を選択
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        # 除外パターンをチェック
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern.startswith("^"):
                if col.startswith(pattern[1:]):
                    should_exclude = True
                    break
            else:
                if pattern in col:
                    should_exclude = True
                    break
        
        if not should_exclude:
            feature_cols.append(col)
    
    print(f"[因果学習] 選択された特徴量数: {len(feature_cols)}")
    
    # 特徴量のカテゴリ別カウント
    event_features = [col for col in feature_cols if col.startswith("event_")]
    lag_features = [col for col in feature_cols if any(suffix in col for suffix in ["_lag", "_ma", "_cumsum", "_pct_change"])]
    spatial_features = [col for col in feature_cols if col.startswith("ring")]
    interaction_features = [col for col in feature_cols if "_x_" in col or "_interaction" in col]
    intensity_features = [col for col in feature_cols if any(keyword in col for keyword in ["intensity", "total_events", "positive_events", "negative_events"])]
    regime_features = [col for col in feature_cols if col.startswith("era_")]
    population_features = [col for col in feature_cols if any(keyword in col for keyword in ["log_pop", "sqrt_pop", "pop_density", "ratio"])]
    
    print(f"[因果学習] 特徴量カテゴリ:")
    print(f"  - 生のイベント特徴量: {len(event_features)}個")
    print(f"  - 時系列ラグ特徴量: {len(lag_features)}個")
    print(f"  - 空間ラグ特徴量: {len(spatial_features)}個")
    print(f"  - 相互作用特徴量: {len(interaction_features)}個")
    print(f"  - 強度特徴量: {len(intensity_features)}個")
    print(f"  - レジーム特徴量: {len(regime_features)}個")
    print(f"  - 人口関連特徴量: {len(population_features)}個")
    
    return feature_cols

def create_time_series_cv(df):
    """時系列CVを作成"""
    print("[因果学習] 時系列CVを作成中...")
    
    # 年でソート
    df = df.sort_values("year")
    
    # 年別のデータ数を確認
    year_counts = df["year"].value_counts().sort_index()
    print(f"[因果学習] 年別データ数: {dict(year_counts)}")
    
    # 時系列CVの設定
    years = sorted(df["year"].unique())
    n_years = len(years)
    
    # テスト期間を最後の3年に設定
    test_years = years[-TEST_SIZE:]
    train_years = years[:-TEST_SIZE]
    
    print(f"[因果学習] 訓練期間: {min(train_years)}-{max(train_years)} ({len(train_years)}年)")
    print(f"[因果学習] テスト期間: {min(test_years)}-{max(test_years)} ({len(test_years)}年)")
    
    return train_years, test_years

def train_model(X_train, y_train, X_val, y_val):
    """モデルを訓練"""
    print("[因果学習] モデルを訓練中...")
    
    if USE_LGBM:
        # LightGBMのパラメータ
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': LEARNING_RATE,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        # データセット作成
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 訓練
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=N_ESTIMATORS,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(LOG_EVERY_N)]
        )
        
    else:
        # HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=EARLY_STOPPING_ROUNDS,
            random_state=42
        )
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """モデルを評価"""
    print("[因果学習] モデルを評価中...")
    
    # 予測
    y_pred = model.predict(X_test)
    
    # メトリクス計算
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[因果学習] 評価結果:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    # 特徴量重要度
    if USE_LGBM:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'feature_importance': feature_importance
    }

def main():
    """メイン処理"""
    print("[因果学習] 因果関係学習用LightGBMモデルを訓練中...")
    
    # データ読み込み
    df = load_data()
    
    # 特徴量選択
    feature_cols = select_causal_features(df)
    
    # データ準備
    X = df[feature_cols].fillna(0.0)
    y = df[TARGET].fillna(0.0)
    
    # 時系列CV
    train_years, test_years = create_time_series_cv(df)
    
    # 訓練・テストデータの分割
    train_mask = df["year"].isin(train_years)
    test_mask = df["year"].isin(test_years)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"[因果学習] 訓練データ: {X_train.shape}")
    print(f"[因果学習] テストデータ: {X_test.shape}")
    
    # 検証データの分割（訓練データの最後の20%）
    n_train = len(X_train)
    n_val = int(n_train * 0.2)
    
    X_train_split = X_train.iloc[:-n_val]
    X_val_split = X_train.iloc[-n_val:]
    y_train_split = y_train.iloc[:-n_val]
    y_val_split = y_train.iloc[-n_val:]
    
    # モデル訓練
    model = train_model(X_train_split, y_train_split, X_val_split, y_val_split)
    
    # モデル評価
    results = evaluate_model(model, X_test, y_test, feature_cols)
    
    # 結果保存
    print("[因果学習] 結果を保存中...")
    
    # 予測結果
    predictions = df[test_mask].copy()
    predictions['predicted_delta_people'] = model.predict(X_test)
    predictions['residual'] = predictions[TARGET] - predictions['predicted_delta_people']
    predictions.to_csv(P_PREDICTIONS, index=False)
    
    # メトリクス
    metrics = {
        'mae': results['mae'],
        'mse': results['mse'],
        'rmse': results['rmse'],
        'r2': results['r2'],
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'train_years': train_years,
        'test_years': test_years
    }
    
    with open(P_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 特徴量重要度
    results['feature_importance'].to_csv(P_FEATURE_IMPORTANCE, index=False)
    
    # モデル保存
    joblib.dump(model, P_MODEL)
    
    # 特徴量リスト保存
    feature_list = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'target': TARGET,
        'model_type': 'causal_learning'
    }
    
    with open(P_FEATURE_LIST, 'w') as f:
        json.dump(feature_list, f, indent=2)
    
    print(f"[因果学習] 訓練完了!")
    print(f"[因果学習] モデル保存先: {P_MODEL}")
    print(f"[因果学習] 予測結果保存先: {P_PREDICTIONS}")
    print(f"[因果学習] メトリクス保存先: {P_METRICS}")
    print(f"[因果学習] 特徴量重要度保存先: {P_FEATURE_IMPORTANCE}")
    
    # トップ10の重要特徴量を表示
    print("\n[因果学習] トップ10の重要特徴量:")
    for i, (_, row) in enumerate(results['feature_importance'].head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

if __name__ == "__main__":
    main()