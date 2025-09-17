# -*- coding: utf-8 -*-
# src/layer4/improved/compare_causal_vs_traditional.py
"""
因果関係学習モデル vs 従来の期待効果ベースモデルの性能比較

設計:
- 因果関係学習モデル: 生のイベントデータ + 時空間特徴量
- 従来モデル: 事前計算された期待効果特徴量
- 比較指標: MAE, RMSE, R², 特徴量重要度
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# パス設定
P_CAUSAL_METRICS = "../../../data/processed/l4_causal_metrics.json"
P_TRADITIONAL_METRICS = "../../../data/processed/l4_cv_metrics_feature_engineered.json"
P_CAUSAL_IMPORTANCE = "../../../data/processed/l4_causal_feature_importance.csv"
P_TRADITIONAL_IMPORTANCE = "../../../data/processed/l4_feature_importance.csv"
P_OUTPUT = "causal_vs_traditional_comparison.json"
P_PLOT_OUTPUT = "causal_vs_traditional_comparison.png"

def load_metrics():
    """メトリクスを読み込み"""
    print("[比較] メトリクスを読み込み中...")
    
    # 因果関係学習モデルのメトリクス
    if Path(P_CAUSAL_METRICS).exists():
        with open(P_CAUSAL_METRICS, 'r') as f:
            causal_metrics = json.load(f)
    else:
        print(f"[WARN] 因果関係学習モデルのメトリクスが見つかりません: {P_CAUSAL_METRICS}")
        causal_metrics = None
    
    # 従来モデルのメトリクス
    if Path(P_TRADITIONAL_METRICS).exists():
        with open(P_TRADITIONAL_METRICS, 'r') as f:
            traditional_metrics = json.load(f)
    else:
        print(f"[WARN] 従来モデルのメトリクスが見つかりません: {P_TRADITIONAL_METRICS}")
        traditional_metrics = None
    
    return causal_metrics, traditional_metrics

def load_feature_importance():
    """特徴量重要度を読み込み"""
    print("[比較] 特徴量重要度を読み込み中...")
    
    # 因果関係学習モデルの特徴量重要度
    if Path(P_CAUSAL_IMPORTANCE).exists():
        causal_importance = pd.read_csv(P_CAUSAL_IMPORTANCE)
    else:
        print(f"[WARN] 因果関係学習モデルの特徴量重要度が見つかりません: {P_CAUSAL_IMPORTANCE}")
        causal_importance = None
    
    # 従来モデルの特徴量重要度
    if Path(P_TRADITIONAL_IMPORTANCE).exists():
        traditional_importance = pd.read_csv(P_TRADITIONAL_IMPORTANCE)
    else:
        print(f"[WARN] 従来モデルの特徴量重要度が見つかりません: {P_TRADITIONAL_IMPORTANCE}")
        traditional_importance = None
    
    return causal_importance, traditional_importance

def compare_metrics(causal_metrics, traditional_metrics):
    """メトリクスを比較"""
    print("[比較] メトリクスを比較中...")
    
    if causal_metrics is None or traditional_metrics is None:
        print("[WARN] メトリクスの比較をスキップします（データ不足）")
        return None
    
    # 比較結果
    comparison = {
        'causal_model': {
            'mae': causal_metrics.get('mae', None),
            'rmse': causal_metrics.get('rmse', None),
            'r2': causal_metrics.get('r2', None),
            'n_features': causal_metrics.get('n_features', None),
            'n_train_samples': causal_metrics.get('n_train_samples', None),
            'n_test_samples': causal_metrics.get('n_test_samples', None)
        },
        'traditional_model': {
            'mae': traditional_metrics.get('mae', None),
            'rmse': traditional_metrics.get('rmse', None),
            'r2': traditional_metrics.get('r2', None),
            'n_features': traditional_metrics.get('n_features', None),
            'n_train_samples': traditional_metrics.get('n_train_samples', None),
            'n_test_samples': traditional_metrics.get('n_test_samples', None)
        }
    }
    
    # 改善率の計算
    if (causal_metrics.get('mae') and traditional_metrics.get('mae')):
        mae_improvement = (traditional_metrics['mae'] - causal_metrics['mae']) / traditional_metrics['mae'] * 100
        comparison['improvement'] = {
            'mae_improvement_pct': mae_improvement,
            'rmse_improvement_pct': (traditional_metrics['rmse'] - causal_metrics['rmse']) / traditional_metrics['rmse'] * 100 if causal_metrics.get('rmse') and traditional_metrics.get('rmse') else None,
            'r2_improvement_pct': (causal_metrics['r2'] - traditional_metrics['r2']) / abs(traditional_metrics['r2']) * 100 if causal_metrics.get('r2') and traditional_metrics.get('r2') else None
        }
    
    return comparison

def analyze_feature_importance(causal_importance, traditional_importance):
    """特徴量重要度を分析"""
    print("[比較] 特徴量重要度を分析中...")
    
    if causal_importance is None or traditional_importance is None:
        print("[WARN] 特徴量重要度の分析をスキップします（データ不足）")
        return None
    
    # トップ20の特徴量を取得
    causal_top20 = causal_importance.head(20)
    traditional_top20 = traditional_importance.head(20)
    
    # 特徴量のカテゴリ分析
    def categorize_features(df):
        categories = {
            'raw_events': len([f for f in df['feature'] if f.startswith('event_') and not any(suffix in f for suffix in ['_lag', '_ma', '_cumsum', '_pct_change'])]),
            'temporal_lags': len([f for f in df['feature'] if any(suffix in f for suffix in ['_lag', '_ma', '_cumsum', '_pct_change'])]),
            'spatial_lags': len([f for f in df['feature'] if f.startswith('ring')]),
            'interactions': len([f for f in df['feature'] if '_x_' in f or '_interaction' in f]),
            'intensity': len([f for f in df['feature'] if any(keyword in f for keyword in ['intensity', 'total_events', 'positive_events', 'negative_events'])]),
            'regime': len([f for f in df['feature'] if f.startswith('era_')]),
            'population': len([f for f in df['feature'] if any(keyword in f for keyword in ['log_pop', 'sqrt_pop', 'pop_density', 'ratio'])]),
            'expected_effects': len([f for f in df['feature'] if f.startswith('exp_') or f.startswith('ring1_exp_')]),
            'other': 0
        }
        
        # その他の特徴量を計算
        categorized_count = sum(categories.values())
        categories['other'] = len(df) - categorized_count
        
        return categories
    
    causal_categories = categorize_features(causal_top20)
    traditional_categories = categorize_features(traditional_top20)
    
    return {
        'causal_top20_categories': causal_categories,
        'traditional_top20_categories': traditional_categories,
        'causal_top20_features': causal_top20['feature'].tolist(),
        'traditional_top20_features': traditional_top20['feature'].tolist()
    }

def create_comparison_plot(comparison, feature_analysis):
    """比較プロットを作成"""
    print("[比較] 比較プロットを作成中...")
    
    if comparison is None:
        print("[WARN] 比較プロットの作成をスキップします（データ不足）")
        return
    
    # プロット設定
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('因果関係学習モデル vs 従来モデル 比較', fontsize=16, fontweight='bold')
    
    # 1. メトリクス比較
    ax1 = axes[0, 0]
    metrics = ['mae', 'rmse', 'r2']
    causal_values = [comparison['causal_model'].get(m, 0) for m in metrics]
    traditional_values = [comparison['traditional_model'].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, causal_values, width, label='因果関係学習', alpha=0.8)
    ax1.bar(x + width/2, traditional_values, width, label='従来モデル', alpha=0.8)
    
    ax1.set_xlabel('メトリクス')
    ax1.set_ylabel('値')
    ax1.set_title('性能メトリクス比較')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 特徴量数比較
    ax2 = axes[0, 1]
    models = ['因果関係学習', '従来モデル']
    n_features = [comparison['causal_model'].get('n_features', 0), 
                  comparison['traditional_model'].get('n_features', 0)]
    
    bars = ax2.bar(models, n_features, alpha=0.8, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('特徴量数')
    ax2.set_title('使用特徴量数比較')
    ax2.grid(True, alpha=0.3)
    
    # バーの上に値を表示
    for bar, value in zip(bars, n_features):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(n_features),
                f'{value}', ha='center', va='bottom')
    
    # 3. 特徴量カテゴリ比較（因果関係学習モデル）
    if feature_analysis:
        ax3 = axes[1, 0]
        causal_cats = feature_analysis['causal_top20_categories']
        categories = list(causal_cats.keys())
        values = list(causal_cats.values())
        
        ax3.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
        ax3.set_title('因果関係学習モデル\nトップ20特徴量のカテゴリ分布')
    
    # 4. 特徴量カテゴリ比較（従来モデル）
    if feature_analysis:
        ax4 = axes[1, 1]
        traditional_cats = feature_analysis['traditional_top20_categories']
        categories = list(traditional_cats.keys())
        values = list(traditional_cats.values())
        
        ax4.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
        ax4.set_title('従来モデル\nトップ20特徴量のカテゴリ分布')
    
    plt.tight_layout()
    plt.savefig(P_PLOT_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"[比較] プロットを保存しました: {P_PLOT_OUTPUT}")

def main():
    """メイン処理"""
    print("[比較] 因果関係学習モデル vs 従来モデル の比較を開始...")
    
    # メトリクス読み込み
    causal_metrics, traditional_metrics = load_metrics()
    
    # 特徴量重要度読み込み
    causal_importance, traditional_importance = load_feature_importance()
    
    # メトリクス比較
    comparison = compare_metrics(causal_metrics, traditional_metrics)
    
    # 特徴量重要度分析
    feature_analysis = analyze_feature_importance(causal_importance, traditional_importance)
    
    # 結果をまとめる
    results = {
        'comparison': comparison,
        'feature_analysis': feature_analysis,
        'summary': {
            'causal_model_available': causal_metrics is not None,
            'traditional_model_available': traditional_metrics is not None,
            'causal_importance_available': causal_importance is not None,
            'traditional_importance_available': traditional_importance is not None
        }
    }
    
    # 結果保存
    with open(P_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[比較] 比較結果を保存しました: {P_OUTPUT}")
    
    # 比較プロット作成
    create_comparison_plot(comparison, feature_analysis)
    
    # 結果表示
    if comparison:
        print("\n[比較] 性能比較結果:")
        print(f"因果関係学習モデル:")
        print(f"  - MAE: {comparison['causal_model']['mae']:.4f}")
        print(f"  - RMSE: {comparison['causal_model']['rmse']:.4f}")
        print(f"  - R²: {comparison['causal_model']['r2']:.4f}")
        print(f"  - 特徴量数: {comparison['causal_model']['n_features']}")
        
        print(f"\n従来モデル:")
        print(f"  - MAE: {comparison['traditional_model']['mae']:.4f}")
        print(f"  - RMSE: {comparison['traditional_model']['rmse']:.4f}")
        print(f"  - R²: {comparison['traditional_model']['r2']:.4f}")
        print(f"  - 特徴量数: {comparison['traditional_model']['n_features']}")
        
        if 'improvement' in comparison:
            print(f"\n改善率:")
            print(f"  - MAE改善: {comparison['improvement']['mae_improvement_pct']:.2f}%")
            print(f"  - RMSE改善: {comparison['improvement']['rmse_improvement_pct']:.2f}%")
            print(f"  - R²改善: {comparison['improvement']['r2_improvement_pct']:.2f}%")
    
    if feature_analysis:
        print(f"\n[比較] 特徴量重要度分析:")
        print(f"因果関係学習モデル トップ20特徴量カテゴリ:")
        for cat, count in feature_analysis['causal_top20_categories'].items():
            print(f"  - {cat}: {count}個")
        
        print(f"\n従来モデル トップ20特徴量カテゴリ:")
        for cat, count in feature_analysis['traditional_top20_categories'].items():
            print(f"  - {cat}: {count}個")
    
    print(f"\n[比較] 比較完了!")

if __name__ == "__main__":
    main()
