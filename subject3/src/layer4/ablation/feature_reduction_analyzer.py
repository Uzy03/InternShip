#!/usr/bin/env python3
"""
特徴量削減分析器

Colabの結果から、特徴量の重要度が低いものを特定し、
アブレーション研究の3つのカテゴリに該当する特徴量をリストアップする。
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent.parent
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "processed" / "l4_feature_importance.csv"
COLAB_RESULTS_DIR = BASE_DIR.parent / "colab"

# アブレーション研究のカテゴリ定義
ABLATION_CATEGORIES = {
    'foreign': [
        'foreign_change', 'foreign_change_covid', 'foreign_change_post2022',
        'foreign_log', 'foreign_log_covid', 'foreign_log_post2022',
        'foreign_ma3', 'foreign_ma3_covid', 'foreign_ma3_post2022',
        'foreign_pct_change', 'foreign_pct_change_covid', 'foreign_pct_change_post2022',
        'foreign_population', 'foreign_population_covid', 'foreign_population_post2022',
        'foreign_change_rate', 'foreign_change_rate_2022_2023'
    ],
    'spatial': [
        'ring1_exp_commercial_inc_h1', 'ring1_exp_commercial_inc_h2', 'ring1_exp_commercial_inc_h3',
        'ring1_exp_disaster_dec_h1', 'ring1_exp_disaster_dec_h2', 'ring1_exp_disaster_dec_h3',
        'ring1_exp_disaster_inc_h1', 'ring1_exp_disaster_inc_h2', 'ring1_exp_disaster_inc_h3',
        'ring1_exp_employment_inc_h1', 'ring1_exp_employment_inc_h2', 'ring1_exp_employment_inc_h3',
        'ring1_exp_housing_dec_h1', 'ring1_exp_housing_dec_h2', 'ring1_exp_housing_dec_h3',
        'ring1_exp_housing_inc_h1', 'ring1_exp_housing_inc_h2', 'ring1_exp_housing_inc_h3',
        'ring1_exp_public_edu_medical_dec_h1', 'ring1_exp_public_edu_medical_dec_h2'
    ],
    'macro': [
        'macro_delta', 'macro_excl', 'macro_ma3', 'macro_shock'
    ],
    'temporal': [
        'lag_d1', 'lag_d2', 'ma2_delta'
    ],
    'era': [
        'era_covid', 'era_post2009', 'era_post2013', 'era_post2022', 'era_pre2013'
    ],
    'town': [
        'town_ma5', 'town_std5', 'town_trend5'
    ]
}

def load_feature_importance() -> pd.DataFrame:
    """特徴量重要度データを読み込む"""
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return df.sort_values('importance', ascending=False)

def load_colab_results() -> Dict[str, Dict]:
    """Colabの結果を読み込む"""
    results = {}
    
    # フルモデルの結果
    full_model_path = COLAB_RESULTS_DIR / "l4_cv_metrics_feature_engineered.json"
    if full_model_path.exists():
        with open(full_model_path, 'r', encoding='utf-8') as f:
            results['full_model'] = json.load(f)
    
    # アブレーション研究の結果
    ablation_files = [
        "ablation_metrics_no_foreign.json",
        "ablation_metrics_no_spatial.json", 
        "ablation_metrics_no_macro.json"
    ]
    
    for file_name in ablation_files:
        file_path = COLAB_RESULTS_DIR / file_name
        if file_path.exists():
            study_name = file_name.replace("ablation_metrics_", "").replace(".json", "")
            with open(file_path, 'r', encoding='utf-8') as f:
                results[study_name] = json.load(f)
    
    return results

def categorize_features(features: List[str]) -> Dict[str, List[str]]:
    """特徴量をカテゴリに分類"""
    categorized = {category: [] for category in ABLATION_CATEGORIES.keys()}
    uncategorized = []
    
    for feature in features:
        found_category = False
        for category, category_features in ABLATION_CATEGORIES.items():
            if feature in category_features:
                categorized[category].append(feature)
                found_category = True
                break
        
        if not found_category:
            uncategorized.append(feature)
    
    categorized['uncategorized'] = uncategorized
    return categorized

def identify_low_importance_features(df: pd.DataFrame, threshold_percentile: float = 20) -> List[str]:
    """アブレーション対象特徴量の中から重要度が低い特徴量を特定"""
    # アブレーション対象特徴量のみを抽出
    ablation_features = []
    for category_features in ABLATION_CATEGORIES.values():
        ablation_features.extend(category_features)
    
    ablation_df = df[df['feature'].isin(ablation_features)]
    
    if len(ablation_df) == 0:
        return []
    
    # アブレーション対象特徴量の中で重要度が低いものを特定
    threshold = np.percentile(ablation_df['importance'], threshold_percentile)
    low_importance = ablation_df[ablation_df['importance'] <= threshold]
    
    return low_importance['feature'].tolist()

def analyze_feature_reduction_candidates():
    """特徴量削減候補を分析"""
    print("=" * 80)
    print("特徴量削減分析")
    print("=" * 80)
    
    # 特徴量重要度を読み込み
    print("\n📊 特徴量重要度データを読み込み中...")
    df_importance = load_feature_importance()
    print(f"総特徴量数: {len(df_importance)}")
    
    # Colabの結果を読み込み
    print("\n📊 Colabの結果を読み込み中...")
    colab_results = load_colab_results()
    
    if 'full_model' in colab_results:
        full_model_metrics = colab_results['full_model']['aggregate']
        print(f"フルモデル性能:")
        print(f"  MAE: {full_model_metrics['MAE']:.4f}")
        print(f"  RMSE: {full_model_metrics['RMSE']:.4f}")
        print(f"  MAPE: {full_model_metrics['MAPE']:.4f}")
        print(f"  R²: {full_model_metrics['R2']:.4f}")
    
    # アブレーション研究の性能を表示
    print(f"\n📊 アブレーション研究の性能:")
    for study_name, study_data in colab_results.items():
        if study_name != 'full_model' and 'aggregate_metrics' in study_data:
            metrics = study_data['aggregate_metrics']
            print(f"{study_name}:")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.4f}")
            print(f"  R²: {metrics['R2']:.4f}")
    
    # アブレーション対象特徴量の中で重要度が低い特徴量を特定
    print(f"\n🔍 アブレーション対象特徴量の中で重要度が低い特徴量を分析中...")
    low_importance_features = identify_low_importance_features(df_importance, threshold_percentile=20)
    print(f"アブレーション対象特徴量の中で重要度下位20%の特徴量数: {len(low_importance_features)}")
    
    # 特徴量をカテゴリに分類
    print(f"\n📋 特徴量をカテゴリに分類中...")
    categorized = categorize_features(df_importance['feature'].tolist())
    
    for category, features in categorized.items():
        if category != 'uncategorized':
            print(f"{category}: {len(features)}個")
    
    # 削減候補を特定
    print(f"\n🎯 削減候補の分析:")
    
    reduction_candidates = {
        'low_importance_by_category': {},
        'zero_importance': [],
        'recommended_reduction': {}
    }
    
    # アブレーション対象特徴量の中で重要度0の特徴量を削除対象に
    ablation_features = []
    for category_features in ABLATION_CATEGORIES.values():
        ablation_features.extend(category_features)
    
    ablation_df = df_importance[df_importance['feature'].isin(ablation_features)]
    zero_importance = ablation_df[ablation_df['importance'] == 0.0]['feature'].tolist()
    reduction_candidates['zero_importance'] = zero_importance
    
    print(f"アブレーション対象特徴量の中で重要度0の特徴量（削除対象）: {len(zero_importance)}個")
    for feature in zero_importance:
        print(f"  - {feature}")
    
    # カテゴリ別の低重要度特徴量
    for category, category_features in ABLATION_CATEGORIES.items():
        category_low_importance = []
        for feature in low_importance_features:
            if feature in category_features:
                category_low_importance.append(feature)
        
        if category_low_importance:
            reduction_candidates['low_importance_by_category'][category] = category_low_importance
            print(f"\n{category}カテゴリの低重要度特徴量 ({len(category_low_importance)}個):")
            for feature in category_low_importance:
                importance = df_importance[df_importance['feature'] == feature]['importance'].iloc[0]
                print(f"  - {feature}: {importance:.2f}")
    
    # 推奨削減戦略
    print(f"\n💡 推奨削減戦略:")
    
    # 1. アブレーション対象特徴量の中で重要度0の特徴量を削除
    if zero_importance:
        print(f"1. アブレーション対象特徴量の中で重要度0の特徴量を削除: {len(zero_importance)}個")
        for feature in zero_importance:
            print(f"   - {feature}")
    
    # 2. 各カテゴリで重要度が低い特徴量を削除
    print(f"\n2. カテゴリ別低重要度特徴量の削除:")
    for category, low_features in reduction_candidates['low_importance_by_category'].items():
        if low_features:
            print(f"   {category}: {len(low_features)}個")
            for feature in low_features[:3]:  # 上位3個のみ表示
                importance = df_importance[df_importance['feature'] == feature]['importance'].iloc[0]
                print(f"     - {feature}: {importance:.2f}")
            if len(low_features) > 3:
                print(f"     ... 他{len(low_features)-3}個")
    
    # 削減後の特徴量数を計算
    features_to_remove = set(zero_importance)
    for low_features in reduction_candidates['low_importance_by_category'].values():
        features_to_remove.update(low_features)
    
    remaining_features = len(df_importance) - len(features_to_remove)
    print(f"\n📈 削減効果:")
    print(f"現在の特徴量数: {len(df_importance)}")
    print(f"削除する特徴量数: {len(features_to_remove)}")
    print(f"削減後の特徴量数: {remaining_features}")
    print(f"削減率: {len(features_to_remove)/len(df_importance)*100:.1f}%")
    
    # 結果をJSONファイルに保存
    output_data = {
        'analysis_summary': {
            'total_features': len(df_importance),
            'features_to_remove': len(features_to_remove),
            'remaining_features': remaining_features,
            'reduction_rate': len(features_to_remove)/len(df_importance)*100
        },
        'zero_importance_features': zero_importance,
        'low_importance_by_category': reduction_candidates['low_importance_by_category'],
        'all_features_to_remove': list(features_to_remove),
        'remaining_features_list': [f for f in df_importance['feature'].tolist() if f not in features_to_remove],
        'note': 'アブレーション研究で対象となった特徴量の中から重要度が低いものを削除対象とした'
    }
    
    output_path = BASE_DIR / "src" / "layer4" / "ablation" / "feature_reduction_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 分析結果を保存しました: {output_path}")
    
    return output_data

def main():
    """メイン処理"""
    try:
        result = analyze_feature_reduction_candidates()
        print(f"\n🎉 特徴量削減分析が完了しました！")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
