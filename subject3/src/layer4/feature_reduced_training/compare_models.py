# -*- coding: utf-8 -*-
# src/layer4/feature_reduced_training/compare_models.py
"""
フルモデルと削除特徴量モデルの性能を比較するスクリプト
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent.parent
P_FULL_METRICS = BASE_DIR / "data/processed/l4_cv_metrics_feature_engineered.json"
P_REDUCED_METRICS = BASE_DIR / "data/processed/feature_reduced_training/l4_cv_metrics_reduced.json"
P_OUTPUT = BASE_DIR / "data/processed/feature_reduced_training/model_comparison.json"

def load_metrics(file_path):
    """評価指標を読み込み"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_improvement_rate(full_value, reduced_value, metric_name):
    """改善率を計算"""
    if metric_name in ['MAE', 'RMSE', 'MAPE']:
        # 低い方が良い指標
        return (full_value - reduced_value) / full_value * 100
    else:
        # 高い方が良い指標（R²）
        return (reduced_value - full_value) / abs(full_value) * 100

def compare_models():
    """モデルの性能を比較"""
    print("=" * 80)
    print("フルモデル vs 削除特徴量モデルの比較")
    print("=" * 80)
    
    # 評価指標を読み込み
    print("評価指標を読み込み中...")
    full_metrics = load_metrics(P_FULL_METRICS)
    reduced_metrics = load_metrics(P_REDUCED_METRICS)
    
    # 集約指標を取得
    full_agg = full_metrics['aggregate_metrics']
    reduced_agg = reduced_metrics['aggregate_metrics']
    
    print("\n📊 性能比較:")
    print("-" * 80)
    
    metrics_to_compare = ['MAE', 'RMSE', 'MAPE', 'R2']
    comparison_results = {}
    
    for metric in metrics_to_compare:
        full_value = full_agg[metric]
        reduced_value = reduced_agg[metric]
        improvement = calculate_improvement_rate(full_value, reduced_value, metric)
        
        comparison_results[metric] = {
            'full_model': full_value,
            'reduced_model': reduced_value,
            'improvement_rate': improvement,
            'better_model': 'reduced' if (metric in ['MAE', 'RMSE', 'MAPE'] and improvement > 0) or (metric == 'R2' and improvement > 0) else 'full'
        }
        
        print(f"{metric:4s}: フル={full_value:8.4f}, 削除={reduced_value:8.4f}, 改善率={improvement:+6.2f}%")
    
    # 特徴量情報を表示
    print(f"\n📈 特徴量情報:")
    print("-" * 80)
    full_features = full_metrics.get('feature_info', {}).get('total_features', 'N/A')
    reduced_features = reduced_metrics['feature_info']['remaining_features']
    removed_features = reduced_metrics['feature_info']['removed_features']
    reduction_rate = reduced_metrics['feature_info']['reduction_rate']
    
    print(f"フルモデル特徴量数: {full_features}")
    print(f"削除モデル特徴量数: {reduced_features}")
    print(f"削除特徴量数: {removed_features}")
    print(f"削減率: {reduction_rate:.1f}%")
    
    # 削除された特徴量を表示
    removed_features_list = reduced_metrics['feature_info']['removed_features_list']
    print(f"\n🗑️ 削除された特徴量 ({len(removed_features_list)}個):")
    for feature in removed_features_list:
        print(f"  - {feature}")
    
    # 結果を保存
    output_data = {
        'comparison_summary': {
            'full_model_features': full_features,
            'reduced_model_features': reduced_features,
            'removed_features': removed_features,
            'reduction_rate': reduction_rate,
            'removed_features_list': removed_features_list
        },
        'performance_comparison': comparison_results,
        'full_model_metrics': full_agg,
        'reduced_model_metrics': reduced_agg
    }
    
    # 出力ディレクトリを作成
    P_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(P_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 比較結果を保存: {P_OUTPUT}")
    
    # 総合評価
    print(f"\n🎯 総合評価:")
    print("-" * 80)
    
    better_count = sum(1 for result in comparison_results.values() if result['better_model'] == 'reduced')
    total_metrics = len(comparison_results)
    
    if better_count > total_metrics / 2:
        print("✅ 削除特徴量モデルの方が優れている")
        print(f"   {better_count}/{total_metrics}の指標で改善")
    elif better_count < total_metrics / 2:
        print("❌ フルモデルの方が優れている")
        print(f"   削除モデルは{better_count}/{total_metrics}の指標でのみ改善")
    else:
        print("🤝 両モデルが同等の性能")
        print(f"   {better_count}/{total_metrics}の指標で改善")
    
    # 削減効果の評価
    print(f"\n📊 削減効果の評価:")
    if reduction_rate > 20:
        print(f"✅ 大幅な特徴量削減を実現 ({reduction_rate:.1f}%)")
    elif reduction_rate > 10:
        print(f"👍 適度な特徴量削減を実現 ({reduction_rate:.1f}%)")
    else:
        print(f"⚠️ 特徴量削減は限定的 ({reduction_rate:.1f}%)")
    
    return comparison_results

def plot_comparison():
    """比較結果をプロット"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # データを読み込み
        full_metrics = load_metrics(P_FULL_METRICS)
        reduced_metrics = load_metrics(P_REDUCED_METRICS)
        
        full_agg = full_metrics['aggregate_metrics']
        reduced_agg = reduced_metrics['aggregate_metrics']
        
        # プロットデータを準備
        metrics_names = ['MAE', 'RMSE', 'MAPE', 'R²']
        full_values = [full_agg[m] for m in ['MAE', 'RMSE', 'MAPE', 'R2']]
        reduced_values = [reduced_agg[m] for m in ['MAE', 'RMSE', 'MAPE', 'R2']]
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 性能比較バープロット
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax1.bar(x - width/2, full_values, width, label='フルモデル', alpha=0.8)
        ax1.bar(x + width/2, reduced_values, width, label='削除モデル', alpha=0.8)
        
        ax1.set_xlabel('評価指標')
        ax1.set_ylabel('値')
        ax1.set_title('モデル性能比較')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 改善率プロット
        improvements = []
        for i, metric in enumerate(['MAE', 'RMSE', 'MAPE', 'R2']):
            full_val = full_values[i]
            reduced_val = reduced_values[i]
            if metric in ['MAE', 'RMSE', 'MAPE']:
                improvement = (full_val - reduced_val) / full_val * 100
            else:
                improvement = (reduced_val - full_val) / abs(full_val) * 100
            improvements.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax2.bar(metrics_names, improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('評価指標')
        ax2.set_ylabel('改善率 (%)')
        ax2.set_title('削除モデルの改善率')
        ax2.grid(True, alpha=0.3)
        
        # 値のラベルを追加
        for i, v in enumerate(improvements):
            ax2.text(i, v + (1 if v >= 0 else -1), f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = BASE_DIR / "data/processed/feature_reduced_training/model_comparison_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 比較プロットを保存: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib/seabornが利用できないため、プロットをスキップします")

if __name__ == "__main__":
    comparison_results = compare_models()
    plot_comparison()
