# -*- coding: utf-8 -*-
# src/layer4/feature_ablation_study.py
"""
特徴量アブレーション研究スクリプト
- 特定の特徴量カテゴリを除外して学習
- フルモデルとの性能比較
- イベント効果の検証（外国人人数、空間ラグ等）
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.feature_gate import select_feature_columns, save_feature_list

# パス設定
P_FEAT = "data/processed/features_l4.csv"
TARGET = "delta_people"
ID_KEYS = ["town","year"]

# フルモデルと同じパラメータ設定
USE_HUBER = True
HUBER_ALPHA = 0.9
N_ESTIMATORS = 25000
LEARNING_RATE = 0.008
EARLY_STOPPING_ROUNDS = 800
LOG_EVERY_N = 200

# サンプル重み調整のパラメータ
TIME_DECAY = 0.05
ANOMALY_YEARS = [2022, 2023]
ANOMALY_WEIGHT = 0.3

# LightGBM が無ければ HistGradientBoosting にフォールバック
USE_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    USE_LGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor

# 特徴量カテゴリの定義
FEATURE_CATEGORIES = {
    'foreign': {
        'pattern': r'^foreign_',
        'description': '外国人人数関連特徴量',
        'features': []
    },
    'spatial_commercial': {
        'pattern': r'^ring1_exp_commercial_',
        'description': '空間ラグ（商業施設）',
        'features': []
    },
    'spatial_disaster': {
        'pattern': r'^ring1_exp_disaster_',
        'description': '空間ラグ（災害）',
        'features': []
    },
    'spatial_employment': {
        'pattern': r'^ring1_exp_employment_',
        'description': '空間ラグ（雇用）',
        'features': []
    },
    'spatial_housing': {
        'pattern': r'^ring1_exp_housing_',
        'description': '空間ラグ（住宅）',
        'features': []
    },
    'spatial_public': {
        'pattern': r'^ring1_exp_public_',
        'description': '空間ラグ（公共施設）',
        'features': []
    },
    'macro': {
        'pattern': r'^macro_',
        'description': 'マクロ経済指標',
        'features': []
    },
    'town_level': {
        'pattern': r'^town_',
        'description': '町レベル統計',
        'features': []
    },
    'temporal': {
        'pattern': r'^(lag_|ma2_)',
        'description': '時系列特徴量',
        'features': []
    },
    'era_flags': {
        'pattern': r'^era_',
        'description': '期間フラグ',
        'features': []
    }
}

def categorize_features(all_features):
    """特徴量をカテゴリ別に分類"""
    categorized_features = {}
    
    for category, config in FEATURE_CATEGORIES.items():
        config['features'] = []
        pattern = config['pattern']
        
        for feature in all_features:
            if feature == 'pop_total':  # 基本人口統計は除外
                continue
            if pattern == r'^foreign_':
                if 'foreign' in feature:
                    config['features'].append(feature)
            elif pattern == r'^ring1_exp_commercial_':
                if 'commercial' in feature:
                    config['features'].append(feature)
            elif pattern == r'^ring1_exp_disaster_':
                if 'disaster' in feature:
                    config['features'].append(feature)
            elif pattern == r'^ring1_exp_employment_':
                if 'employment' in feature:
                    config['features'].append(feature)
            elif pattern == r'^ring1_exp_housing_':
                if 'housing' in feature:
                    config['features'].append(feature)
            elif pattern == r'^ring1_exp_public_':
                if 'public' in feature:
                    config['features'].append(feature)
            elif pattern == r'^macro_':
                if feature.startswith('macro_'):
                    config['features'].append(feature)
            elif pattern == r'^town_':
                if feature.startswith('town_'):
                    config['features'].append(feature)
            elif pattern == r'^(lag_|ma2_)':
                if feature.startswith(('lag_', 'ma2_')):
                    config['features'].append(feature)
            elif pattern == r'^era_':
                if feature.startswith('era_'):
                    config['features'].append(feature)
    
    return FEATURE_CATEGORIES

def metrics(y_true, y_pred):
    """評価指標を計算（フルモデルと同じ）"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # 古いsklearn対応
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, np.abs(y_true)))))  # 0割回避
    r2   = float(r2_score(y_true, y_pred))
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

def exclude_features_by_category(features, exclude_categories):
    """指定されたカテゴリの特徴量を除外"""
    excluded_features = []
    
    for category in exclude_categories:
        if category in FEATURE_CATEGORIES:
            excluded_features.extend(FEATURE_CATEGORIES[category]['features'])
    
    remaining_features = [f for f in features if f not in excluded_features]
    
    return remaining_features, excluded_features

# 高速アブレーション関数を削除（科学的に不正確なため）

def run_ablation_study(df, Xcols, exclude_categories, study_name):
    """特徴量アブレーション研究（科学的に正確な方法）"""
    print(f"\n=== アブレーション研究: {study_name} ===")
    print("特徴量を除外してLightGBMを再学習します（約10-30分）")
    
    # 除外する特徴量を決定
    remaining_features, excluded_features = exclude_features_by_category(Xcols, exclude_categories)
    
    print(f"除外カテゴリ: {', '.join(exclude_categories)}")
    print(f"除外特徴量数: {len(excluded_features)}")
    print(f"残存特徴量数: {len(remaining_features)}")
    print(f"除外特徴量: {excluded_features[:5]}{'...' if len(excluded_features) > 5 else ''}")
    
    # 学習データの準備
    df_clean = df.copy()
    df_clean[TARGET] = pd.to_numeric(df_clean[TARGET], errors="coerce")
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean[~df_clean[TARGET].isna()].copy()
    
    # 時系列CVの設定
    years = sorted(df_clean["year"].unique().tolist())
    min_train_years = 20
    test_window = 1
    
    def time_series_folds(years, min_train_years, test_window):
        ys = sorted(years)
        folds = []
        for i in range(min_train_years, len(ys)):
            train_years = ys[:i]
            test_years = ys[i:i+test_window]
            folds.append((set(train_years), set(test_years)))
        return folds
    
    folds = time_series_folds(years, min_train_years, test_window)
    
    # サンプル重み関数（フルモデルと同じ）
    def make_weights(s):
        # 人口規模による重み（人口が多い地域を重視）
        w_pop = np.sqrt(np.maximum(1.0, s["pop_total"].values)) if "pop_total" in s.columns else np.ones(len(s))
        
        # 時間減衰（最近の年を重視）
        w_time = (1.0 + TIME_DECAY) ** (s["year"].values - s["year"].min())
        
        # 異常値期間の重み調整（2022-2023年の重みを下げる）
        w_anomaly = np.where(s["year"].isin(ANOMALY_YEARS), ANOMALY_WEIGHT, 1.0)
        
        # 最終的な重み
        w = w_pop * w_time * w_anomaly
        w[~np.isfinite(w)] = 1.0
        
        # 重みの正規化（平均が1になるように）
        w = w / np.mean(w)
        
        return w

    # フルモデルと同じパラメータ設定
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
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # 出力ディレクトリを準備
    output_dir = Path("data/processed/ablation_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_preds = []
    fold_metrics = []
    
    for fi, (train_years, test_years) in enumerate(folds, 1):
        print(f"フォールド {fi}/{len(folds)} を処理中...")
        
        tr = df_clean[df_clean["year"].isin(train_years)].copy()
        te = df_clean[df_clean["year"].isin(test_years)].copy()
        
        # 特徴量の準備
        tr_X = tr[remaining_features].replace([np.inf, -np.inf], np.nan)
        te_X = te[remaining_features].replace([np.inf, -np.inf], np.nan)
        sw = make_weights(tr)
        
        # 重みの統計情報をログ出力（最初のfoldのみ）
        if fi == 1:
            print(f"[アブレーション] Sample weights stats: mean={np.mean(sw):.3f}, std={np.std(sw):.3f}, min={np.min(sw):.3f}, max={np.max(sw):.3f}")
        
        if len(tr_X) == 0 or len(te_X) == 0:
            continue
        
        # 学習用 y を軽くウィンズライズ（フルモデルと同じ）
        y_tr = tr[TARGET].values
        if 2022 in test_years:
            ql, qh = np.quantile(y_tr, [0.01, 0.99])
        else:
            ql, qh = np.quantile(y_tr, [0.005, 0.995])
        y_tr = np.clip(y_tr, ql, qh)
        
        # モデル学習（フルモデルと同じ設定）
        if USE_LGBM:
            model = lgb.LGBMRegressor(**base_params)
            
            callbacks = []
            if EARLY_STOPPING_ROUNDS > 0:
                callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))
            if LOG_EVERY_N > 0:
                callbacks.append(lgb.log_evaluation(LOG_EVERY_N))
            
            model.fit(
                tr_X, y_tr,
                sample_weight=sw,
                eval_set=(te_X, te[TARGET].values),
                callbacks=callbacks
            )
        else:
            model = HistGradientBoostingRegressor(**base_params)
            model.fit(tr_X, y_tr, sample_weight=sw)
        
        # 予測
        if USE_LGBM and hasattr(model, 'best_iteration_'):
            pred = model.predict(te_X, num_iteration=model.best_iteration_)
        else:
            pred = model.predict(te_X)
        
        # メトリクス計算（4指標すべて）
        m = metrics(te[TARGET].values, pred)
        m['fold'] = fi
        m['train_years'] = len(train_years)
        m['test_year'] = list(test_years)[0]
        fold_metrics.append(m)
        
        # 予測結果を保存
        te_out = te[ID_KEYS + [TARGET]].copy()
        te_out["y_pred"] = pred
        te_out["fold"] = fi
        all_preds.append(te_out)
        
        # モデルを保存（最初のfoldのみ）
        if fi == 1:
            import joblib
            
            # モデルファイルを保存
            model_path = output_dir / f"ablation_model_{study_name}.joblib"
            joblib.dump(model, model_path)
            print(f"  モデル保存: {model_path}")
            
            # 特徴量重要度を保存（LightGBMの場合のみ）
            if USE_LGBM and hasattr(model, 'feature_importances_'):
                # 特徴量重要度を取得（NumPy型をPython型に変換）
                feature_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(remaining_features, model.feature_importances_)
                }
                
                # 重要度でソート
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                
                # 重み情報を保存
                weights_info = {
                    'fold': int(fi),
                    'feature_importance': sorted_importance,
                    'top_10_features': dict(list(sorted_importance.items())[:10]),
                    'total_features': int(len(remaining_features)),
                    'model_params': {
                        k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                        for k, v in base_params.items()
                    }
                }
                
                # 重みファイルを保存
                weights_path = output_dir / f"ablation_model_weights_{study_name}.json"
                with open(weights_path, 'w', encoding='utf-8') as f:
                    json.dump(weights_info, f, ensure_ascii=False, indent=2)
                
                print(f"  モデル重み: {weights_path}")
    
    # 結果の集計
    preds_df = pd.concat(all_preds, axis=0, ignore_index=True)
    agg_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    
    # 結果を保存
    
    # 予測結果を保存
    preds_path = output_dir / f"ablation_predictions_{study_name}.csv"
    preds_df.to_csv(preds_path, index=False)
    
    # メトリクスを保存（NumPy型をPython型に変換）
    def convert_numpy_types(obj):
        """NumPy型をPython型に変換する"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    metrics_path = output_dir / f"ablation_metrics_{study_name}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        metrics_data = {
            'study_name': study_name,
            'excluded_categories': exclude_categories,
            'excluded_features': excluded_features,
            'remaining_features': remaining_features,
            'fold_metrics': convert_numpy_types(fold_metrics),
            'aggregate_metrics': convert_numpy_types(agg_metrics),
            'n_features_used': int(len(remaining_features)),
            'n_features_excluded': int(len(excluded_features)),
            'method': 'ablation_study'
        }
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    
    print(f"結果を保存しました:")
    print(f"  予測結果: {preds_path}")
    print(f"  メトリクス: {metrics_path}")
    
    return agg_metrics, excluded_features

def compare_with_full_model(ablation_metrics, full_model_metrics):
    """フルモデルとの性能比較（4指標すべて）"""
    comparison = {}
    
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
        if metric in ablation_metrics and metric in full_model_metrics:
            ablation_val = ablation_metrics[metric]
            full_val = full_model_metrics[metric]
            
            if metric == 'R2':
                # R2の場合は改善率を計算
                improvement = ablation_val - full_val
                improvement_pct = (improvement / abs(full_val)) * 100 if full_val != 0 else 0
            else:
                # MAE, RMSE, MAPEの場合は悪化率を計算
                degradation = ablation_val - full_val
                degradation_pct = (degradation / full_val) * 100 if full_val != 0 else 0
                improvement = -degradation
                improvement_pct = -degradation_pct
            
            comparison[metric] = {
                'full_model': full_val,
                'ablation_model': ablation_val,
                'difference': improvement,
                'improvement_pct': improvement_pct
            }
    
    return comparison

def main():
    """メイン処理"""
    print("=== 特徴量アブレーション研究を開始 ===")
    
    # データ読み込み
    df = pd.read_csv(P_FEAT).sort_values(ID_KEYS)
    
    # 目的変数の整備
    if TARGET not in df.columns:
        if "pop_total" not in df.columns:
            raise ValueError("features_l4.csv に delta_people も pop_total もありません。")
        df[TARGET] = df.groupby("town")["pop_total"].diff()
    
    # 特徴量選択
    Xcols = select_feature_columns(df)
    print(f"全特徴量数: {len(Xcols)}")
    
    # 特徴量をカテゴリ別に分類
    categorize_features(Xcols)
    
    # カテゴリ別特徴量数を表示（時系列特徴量は除外）
    print("\n=== 特徴量カテゴリ別統計 ===")
    for category, config in FEATURE_CATEGORIES.items():
        if config['features'] and category != 'temporal':  # 時系列特徴量は除外
            print(f"{config['description']:<20}: {len(config['features']):2d}個")
    
    print(f"{'時系列特徴量':<20}: {len(FEATURE_CATEGORIES['temporal']['features']):2d}個 (必須 - アブレーション対象外)")
    
    # フルモデルの結果を読み込み（既存の結果がある場合）
    full_model_metrics = None
    try:
        with open('data/processed/l4_cv_metrics_feature_engineered.json', 'r', encoding='utf-8') as f:
            full_results = json.load(f)
            full_model_metrics = full_results['aggregate']
            print(f"\nフルモデルの性能: MAE={full_model_metrics['MAE']:.4f}, RMSE={full_model_metrics['RMSE']:.4f}, R2={full_model_metrics['R2']:.4f}")
    except FileNotFoundError:
        print("\nフルモデルの結果が見つかりません。先にtrain_lgbm.pyを実行してください。")
        return
    
    # アブレーション研究を実行（時系列特徴量は除外）
    studies = [
        {
            'name': 'no_foreign',
            'categories': ['foreign'],
            'description': '外国人人数関連特徴量を除外'
        },
        {
            'name': 'no_spatial',
            'categories': ['spatial_commercial', 'spatial_disaster', 'spatial_employment', 'spatial_housing', 'spatial_public'],
            'description': '全空間ラグ特徴量を除外'
        },
        {
            'name': 'no_macro',
            'categories': ['macro'],
            'description': 'マクロ経済指標を除外'
        }
    ]
    
    results = {}
    
    for study in studies:
        study_name = study['name']
        exclude_categories = study['categories']
        
        # アブレーション研究を実行
        ablation_metrics, excluded_features = run_ablation_study(
            df, Xcols, exclude_categories, study_name
        )
        
        # フルモデルとの比較
        comparison = compare_with_full_model(ablation_metrics, full_model_metrics)
        
        results[study_name] = {
            'description': study['description'],
            'excluded_features': excluded_features,
            'ablation_metrics': ablation_metrics,
            'comparison': comparison
        }
        
        # 結果を表示
        print(f"\n--- {study['description']} ---")
        print(f"MAE: {ablation_metrics['MAE']:.4f} (フルモデル比: {comparison['MAE']['improvement_pct']:+.2f}%)")
        print(f"RMSE: {ablation_metrics['RMSE']:.4f} (フルモデル比: {comparison['RMSE']['improvement_pct']:+.2f}%)")
        print(f"MAPE: {ablation_metrics['MAPE']:.4f} (フルモデル比: {comparison['MAPE']['improvement_pct']:+.2f}%)")
        print(f"R2: {ablation_metrics['R2']:.4f} (フルモデル比: {comparison['R2']['improvement_pct']:+.2f}%)")
    
    # 全体の結果を保存
    output_dir = Path("data/processed/ablation_study")
    summary_path = output_dir / "ablation_study_summary.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'full_model_metrics': full_model_metrics,
            'studies': results,
            'feature_categories': {k: {'description': v['description'], 'features': v['features']} 
                                 for k, v in FEATURE_CATEGORIES.items()}
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== アブレーション研究完了 ===")
    print(f"結果サマリー: {summary_path}")
    
    # 結果の要約を表示
    print("\n=== 性能影響の要約 ===")
    for study_name, result in results.items():
        mae_impact = result['comparison']['MAE']['improvement_pct']
        mape_impact = result['comparison']['MAPE']['improvement_pct']
        print(f"{result['description']:<25}: MAE {mae_impact:+.2f}%, MAPE {mape_impact:+.2f}%")

if __name__ == "__main__":
    # 3つのアブレーション研究をすべて実行
    print("特徴量アブレーション研究を実行します")
    main()
