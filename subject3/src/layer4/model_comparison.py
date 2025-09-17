#!/usr/bin/env python3
"""
フルモデルとアブレーション研究の性能比較

JSONファイルを読み込んで、フルモデルと各アブレーション研究の性能を比較し、
視覚的に表示する。
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ABLATION_DIR = DATA_DIR / "ablation_study"

# ファイルパス
FULL_MODEL_PATH = DATA_DIR / "l4_cv_metrics_feature_engineered.json"

# アブレーション研究の定義
ABLATION_STUDIES = {
    'no_foreign': {
        'name': '外国人人数関連除外',
        'description': '外国人人数関連特徴量（15個）を除外',
        'categories': ['foreign']
    },
    'no_spatial': {
        'name': '空間ラグ除外', 
        'description': '空間ラグ特徴量（20個）を除外',
        'categories': ['spatial_commercial', 'spatial_disaster', 'spatial_employment', 'spatial_housing', 'spatial_public']
    },
    'no_macro': {
        'name': 'マクロ経済除外',
        'description': 'マクロ経済指標（4個）を除外', 
        'categories': ['macro']
    },
    'no_temporal': {
        'name': '時系列特徴除外',
        'description': '時系列特徴量（3個）を除外',
        'categories': ['temporal']
    }
}

# メトリクス名の日本語対応
METRIC_NAMES = {
    'MAE': 'MAE (Mean Absolute Error)',
    'RMSE': 'RMSE (Root Mean Squared Error)', 
    'MAPE': 'MAPE (Mean Absolute Percentage Error)',
    'R2': 'R² (Coefficient of Determination)'
}

def load_json(file_path: Path) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: {file_path} が見つかりません")
        return {}
    except json.JSONDecodeError as e:
        print(f"エラー: {file_path} のJSON解析に失敗: {e}")
        return {}

def load_full_model_metrics() -> Dict[str, Any]:
    """フルモデルのメトリクスを読み込む"""
    return load_json(FULL_MODEL_PATH)

def load_ablation_metrics(study_name: str) -> Dict[str, Any]:
    """アブレーション研究のメトリクスを読み込む"""
    file_path = ABLATION_DIR / f"ablation_metrics_{study_name}.json"
    return load_json(file_path)

def load_ablation_weights(study_name: str) -> Dict[str, Any]:
    """アブレーション研究の重みを読み込む"""
    file_path = ABLATION_DIR / f"ablation_model_weights_{study_name}.json"
    return load_json(file_path)

def extract_aggregate_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """集約メトリクスを抽出する"""
    if 'aggregate' in data:
        return data['aggregate']
    elif 'aggregate_metrics' in data:
        return data['aggregate_metrics']
    else:
        return {}

def calculate_improvement(full_metrics: Dict[str, float], ablation_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """改善率を計算する"""
    improvements = {}
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in full_metrics and metric in ablation_metrics:
            # MAE, RMSE, MAPEは小さい方が良い
            change = ablation_metrics[metric] - full_metrics[metric]
            improvement_pct = (change / full_metrics[metric]) * 100
            improvements[metric] = {
                'change': change,
                'improvement_pct': improvement_pct,
                'direction': '悪化' if change > 0 else '改善'
            }
    
    for metric in ['R2']:
        if metric in full_metrics and metric in ablation_metrics:
            # R2は大きい方が良い
            change = ablation_metrics[metric] - full_metrics[metric]
            improvement_pct = (change / full_metrics[metric]) * 100
            improvements[metric] = {
                'change': change,
                'improvement_pct': improvement_pct,
                'direction': '改善' if change > 0 else '悪化'
            }
    
    return improvements

def create_comparison_table(full_metrics: Dict[str, float], ablation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """比較テーブルを作成する"""
    data = []
    
    for study_name, study_info in ABLATION_STUDIES.items():
        if study_name in ablation_results:
            ablation_metrics = ablation_results[study_name]['metrics']
            improvements = ablation_results[study_name]['improvements']
            
            row = {
                '研究名': study_info['name'],
                '説明': study_info['description'],
                '除外特徴量数': ablation_results[study_name].get('n_excluded', 0)
            }
            
            # 各メトリクスの値と改善率を追加
            for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
                if metric in ablation_metrics:
                    row[f'{metric}_値'] = f"{ablation_metrics[metric]:.4f}"
                    if metric in improvements:
                        pct = improvements[metric]['improvement_pct']
                        direction = improvements[metric]['direction']
                        row[f'{metric}_改善率'] = f"{pct:+.2f}% ({direction})"
                    else:
                        row[f'{metric}_改善率'] = "N/A"
                else:
                    row[f'{metric}_値'] = "N/A"
                    row[f'{metric}_改善率'] = "N/A"
            
            data.append(row)
    
    return pd.DataFrame(data)

def plot_metrics_comparison(full_metrics: Dict[str, float], ablation_results: Dict[str, Dict[str, Any]]):
    """メトリクス比較のグラフを作成する"""
    # データ準備
    studies = []
    metrics_data = {metric: [] for metric in ['MAE', 'RMSE', 'MAPE', 'R2']}
    
    for study_name, study_info in ABLATION_STUDIES.items():
        if study_name in ablation_results:
            studies.append(study_info['name'])
            ablation_metrics = ablation_results[study_name]['metrics']
            
            for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
                if metric in ablation_metrics:
                    metrics_data[metric].append(ablation_metrics[metric])
                else:
                    metrics_data[metric].append(np.nan)
    
    # フルモデルの値を追加
    studies.insert(0, 'フルモデル')
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
        if metric in full_metrics:
            metrics_data[metric].insert(0, full_metrics[metric])
        else:
            metrics_data[metric].insert(0, np.nan)
    
    # グラフ作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('フルモデル vs アブレーション研究 性能比較', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for idx, (metric, values) in enumerate(metrics_data.items()):
        ax = axes[idx // 2, idx % 2]
        
        # バープロット
        bars = ax.bar(range(len(studies)), values, color=colors[:len(studies)], alpha=0.7)
        
        # フルモデルのバーを強調
        if len(bars) > 0:
            bars[0].set_color('#1f77b4')
            bars[0].set_alpha(1.0)
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(2)
        
        ax.set_title(f'{METRIC_NAMES[metric]}', fontweight='bold')
        ax.set_xlabel('研究')
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(studies)))
        ax.set_xticklabels(studies, rotation=45, ha='right')
        
        # 値をバーの上に表示
        for i, (bar, value) in enumerate(zip(bars, values)):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # グリッドを追加
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_feature_importance(ablation_weights: Dict[str, Dict[str, Any]]):
    """アブレーション研究の特徴量重要度の比較グラフを作成する"""
    # 各アブレーション研究の特徴量重要度を取得
    ablation_top20 = {}
    for study_name, weights_info in ablation_weights.items():
        if weights_info and 'top_10_features' in weights_info:
            ablation_top20[study_name] = weights_info['top_10_features']
    
    if not ablation_top20:
        print("アブレーション研究の重み情報が不足しています")
        return
    
    # 共通特徴量を特定
    common_features = None
    for study_top10 in ablation_top20.values():
        if common_features is None:
            common_features = set(study_top10.keys())
        else:
            common_features = common_features.intersection(set(study_top10.keys()))
    
    if not common_features:
        print("共通特徴量が見つかりません")
        return
    
    # トップ5の共通特徴量を選択
    top_common_features = list(common_features)[:5]
    
    if not top_common_features:
        print("十分な共通特徴量が見つかりません")
        return
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # データ準備
    x = np.arange(len(top_common_features))
    width = 0.2
    
    # 各アブレーション研究の重要度
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (study_name, study_top10) in enumerate(ablation_top20.items()):
        importances = [study_top10.get(feature, 0) for feature in top_common_features]
        bars = ax.bar(x + i*width, importances, width, 
                     label=ABLATION_STUDIES[study_name]['name'], 
                     color=colors[i % len(colors)], alpha=0.7)
    
    ax.set_xlabel('特徴量')
    ax.set_ylabel('重要度')
    ax.set_title('アブレーション研究の特徴量重要度比較（共通特徴量）', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(top_common_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'ablation_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_improvement_rates(ablation_results: Dict[str, Dict[str, Any]]):
    """改善率のグラフを作成する"""
    # データ準備
    studies = []
    improvement_data = {metric: [] for metric in ['MAE', 'RMSE', 'MAPE', 'R2']}
    
    for study_name, study_info in ABLATION_STUDIES.items():
        if study_name in ablation_results:
            studies.append(study_info['name'])
            improvements = ablation_results[study_name]['improvements']
            
            for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
                if metric in improvements:
                    improvement_data[metric].append(improvements[metric]['improvement_pct'])
                else:
                    improvement_data[metric].append(0)
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(studies))
    width = 0.2
    
    for idx, (metric, values) in enumerate(improvement_data.items()):
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=METRIC_NAMES[metric], alpha=0.8)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            if value != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value > 0 else -3),
                       f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top', fontsize=9)
    
    ax.set_xlabel('アブレーション研究')
    ax.set_ylabel('改善率 (%)')
    ax.set_title('フルモデルに対する改善率', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(studies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'model_comparison_improvements.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_comparison(full_metrics: Dict[str, float], ablation_results: Dict[str, Dict[str, Any]]):
    """詳細な比較結果を表示する"""
    print("=" * 80)
    print("フルモデル vs アブレーション研究 詳細比較")
    print("=" * 80)
    
    # フルモデルの性能
    print("\n📊 フルモデルの性能:")
    for metric, value in full_metrics.items():
        if metric in METRIC_NAMES:
            print(f"  {METRIC_NAMES[metric]}: {value:.4f}")
    
    print("\n" + "=" * 80)
    
    # 各アブレーション研究の結果
    for study_name, study_info in ABLATION_STUDIES.items():
        if study_name in ablation_results:
            print(f"\n🔬 {study_info['name']}")
            print(f"   説明: {study_info['description']}")
            
            ablation_metrics = ablation_results[study_name]['metrics']
            improvements = ablation_results[study_name]['improvements']
            n_excluded = ablation_results[study_name].get('n_excluded', 0)
            
            print(f"   除外特徴量数: {n_excluded}個")
            print("   性能:")
            
            for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
                if metric in ablation_metrics and metric in improvements:
                    value = ablation_metrics[metric]
                    improvement = improvements[metric]
                    pct = improvement['improvement_pct']
                    direction = improvement['direction']
                    
                    print(f"     {METRIC_NAMES[metric]}: {value:.4f} ({pct:+.2f}% {direction})")
                elif metric in ablation_metrics:
                    value = ablation_metrics[metric]
                    print(f"     {METRIC_NAMES[metric]}: {value:.4f}")
            
            print("-" * 60)

def save_comparison_summary(full_metrics: Dict[str, float], ablation_results: Dict[str, Dict[str, Any]]):
    """比較結果をJSONファイルに保存する"""
    summary = {
        'full_model_metrics': full_metrics,
        'ablation_studies': {},
        'summary_statistics': {
            'total_studies': len(ablation_results),
            'studies_with_improvement': 0,
            'studies_with_degradation': 0
        }
    }
    
    for study_name, study_info in ABLATION_STUDIES.items():
        if study_name in ablation_results:
            ablation_data = ablation_results[study_name]
            
            summary['ablation_studies'][study_name] = {
                'name': study_info['name'],
                'description': study_info['description'],
                'metrics': ablation_data['metrics'],
                'improvements': ablation_data['improvements'],
                'n_excluded_features': ablation_data.get('n_excluded', 0)
            }
            
            # 統計情報を更新
            has_improvement = any(
                imp['improvement_pct'] > 0 and metric in ['R2'] or 
                imp['improvement_pct'] < 0 and metric in ['MAE', 'RMSE', 'MAPE']
                for metric, imp in ablation_data['improvements'].items()
            )
            
            if has_improvement:
                summary['summary_statistics']['studies_with_improvement'] += 1
            else:
                summary['summary_statistics']['studies_with_degradation'] += 1
    
    # ファイルに保存
    output_path = BASE_DIR / 'model_comparison_summary.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 比較結果を保存しました: {output_path}")

def main():
    """メイン処理"""
    print("フルモデルとアブレーション研究の性能比較を開始します...")
    
    # フルモデルのメトリクスを読み込み
    print("\n📖 フルモデルのメトリクスを読み込み中...")
    full_model_data = load_full_model_metrics()
    if not full_model_data:
        print("❌ フルモデルのメトリクスファイルが見つかりません")
        return
    
    full_metrics = extract_aggregate_metrics(full_model_data)
    print(f"✅ フルモデルのメトリクスを読み込み完了")
    
    # 各アブレーション研究のメトリクスを読み込み
    print("\n📖 アブレーション研究のメトリクスを読み込み中...")
    ablation_results = {}
    ablation_weights = {}
    
    for study_name in ABLATION_STUDIES.keys():
        ablation_data = load_ablation_metrics(study_name)
        if ablation_data:
            ablation_metrics = extract_aggregate_metrics(ablation_data)
            improvements = calculate_improvement(full_metrics, ablation_metrics)
            
            ablation_results[study_name] = {
                'metrics': ablation_metrics,
                'improvements': improvements,
                'n_excluded': ablation_data.get('n_features_excluded', 0)
            }
            print(f"✅ {ABLATION_STUDIES[study_name]['name']} のメトリクスを読み込み完了")
        else:
            print(f"⚠️ {ABLATION_STUDIES[study_name]['name']} のメトリクスファイルが見つかりません")
        
        # アブレーション研究の重みを読み込み
        weights_data = load_ablation_weights(study_name)
        if weights_data:
            ablation_weights[study_name] = weights_data
            print(f"✅ {ABLATION_STUDIES[study_name]['name']} の重みを読み込み完了")
        else:
            print(f"⚠️ {ABLATION_STUDIES[study_name]['name']} の重みファイルが見つかりません")
    
    if not ablation_results:
        print("❌ アブレーション研究のメトリクスファイルが見つかりません")
        return
    
    # 比較結果を表示
    print_detailed_comparison(full_metrics, ablation_results)
    
    # 比較テーブルを作成・表示
    print("\n📊 比較テーブル:")
    comparison_table = create_comparison_table(full_metrics, ablation_results)
    print(comparison_table.to_string(index=False))
    
    # グラフを作成
    print("\n📈 グラフを作成中...")
    try:
        plot_metrics_comparison(full_metrics, ablation_results)
        plot_improvement_rates(ablation_results)
        
        # アブレーション研究の特徴量重要度の比較グラフ
        if ablation_weights:
            plot_ablation_feature_importance(ablation_weights)
        
        print("✅ グラフの作成が完了しました")
    except Exception as e:
        print(f"⚠️ グラフの作成でエラーが発生しました: {e}")
    
    # 結果を保存
    save_comparison_summary(full_metrics, ablation_results)
    
    print("\n🎉 性能比較が完了しました！")

if __name__ == "__main__":
    main()
