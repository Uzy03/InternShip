# -*- coding: utf-8 -*-
# src/layer4/performance_comparison.py
"""
性能比較システム
- ベースライン性能の記録
- 改善案の性能比較
- 2022-2023年特化の分析
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceComparator:
    """性能比較クラス"""
    
    def __init__(self, results_dir: str = "data/processed"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_file = self.results_dir / "performance_comparison.json"
        self.load_comparison_data()
    
    def load_comparison_data(self):
        """比較データを読み込み"""
        if self.comparison_file.exists():
            with open(self.comparison_file, 'r', encoding='utf-8') as f:
                self.comparison_data = json.load(f)
        else:
            self.comparison_data = {
                "baseline": {},
                "improvements": {},
                "summary": {}
            }
    
    def save_comparison_data(self):
        """比較データを保存"""
        with open(self.comparison_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_data, f, ensure_ascii=False, indent=2)
    
    def record_baseline(self, metrics_file: str = "l4_cv_metrics.json", baseline_name: str = "original_baseline"):
        """ベースライン性能を記録"""
        metrics_path = self.results_dir / metrics_file
        if not metrics_path.exists():
            print(f"[ERROR] メトリクスファイルが見つかりません: {metrics_path}")
            return
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        # ベースライン性能を記録（上書きしない）
        if "baselines" not in self.comparison_data:
            self.comparison_data["baselines"] = {}
        
        self.comparison_data["baselines"][baseline_name] = {
            "aggregate_metrics": metrics_data["aggregate"],
            "fold_metrics": metrics_data["folds"],
            "features_count": len(metrics_data["features"]),
            "parameters": metrics_data.get("parameters", {}),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 2022-2023年の性能を特別に記録
        fold_2022_2023 = [f for f in metrics_data["folds"] if 2022 in f.get("test_years", []) or 2023 in f.get("test_years", [])]
        if fold_2022_2023:
            self.comparison_data["baselines"][baseline_name]["2022_2023_metrics"] = {
                "R2": np.mean([f["R2"] for f in fold_2022_2023]),
                "MAE": np.mean([f["MAE"] for f in fold_2022_2023]),
                "RMSE": np.mean([f["RMSE"] for f in fold_2022_2023]),
                "MAPE": np.mean([f["MAPE"] for f in fold_2022_2023])
            }
        
        self.save_comparison_data()
        print("[Performance] ベースライン性能を記録しました")
    
    def record_improvement(self, improvement_name: str, metrics_file: str = "l4_cv_metrics.json"):
        """改善案の性能を記録"""
        metrics_path = self.results_dir / metrics_file
        if not metrics_path.exists():
            print(f"[ERROR] メトリクスファイルが見つかりません: {metrics_path}")
            return
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        
        # 改善案の性能を記録
        self.comparison_data["improvements"][improvement_name] = {
            "aggregate_metrics": metrics_data["aggregate"],
            "fold_metrics": metrics_data["folds"],
            "features_count": len(metrics_data["features"]),
            "parameters": metrics_data.get("parameters", {}),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 2022-2023年の性能を特別に記録
        fold_2022_2023 = [f for f in metrics_data["folds"] if 2022 in f.get("test_years", []) or 2023 in f.get("test_years", [])]
        if fold_2022_2023:
            self.comparison_data["improvements"][improvement_name]["2022_2023_metrics"] = {
                "R2": np.mean([f["R2"] for f in fold_2022_2023]),
                "MAE": np.mean([f["MAE"] for f in fold_2022_2023]),
                "RMSE": np.mean([f["RMSE"] for f in fold_2022_2023]),
                "MAPE": np.mean([f["MAPE"] for f in fold_2022_2023])
            }
        
        self.save_comparison_data()
        print(f"[Performance] 改善案 '{improvement_name}' の性能を記録しました")
    
    def compare_performance(self, reference_baseline: str = "original_baseline") -> Dict[str, Any]:
        """性能比較を実行"""
        if "baselines" not in self.comparison_data or reference_baseline not in self.comparison_data["baselines"]:
            print(f"[ERROR] ベースライン '{reference_baseline}' が記録されていません")
            return {}
        
        baseline = self.comparison_data["baselines"][reference_baseline]["aggregate_metrics"]
        baseline_2022_2023 = self.comparison_data["baselines"][reference_baseline].get("2022_2023_metrics", {})
        
        comparison_results = {
            "baseline": baseline,
            "baseline_2022_2023": baseline_2022_2023,
            "improvements": {}
        }
        
        for name, improvement in self.comparison_data["improvements"].items():
            improvement_metrics = improvement["aggregate_metrics"]
            improvement_2022_2023 = improvement.get("2022_2023_metrics", {})
            
            # 全体性能の比較
            overall_comparison = {}
            for metric in ["R2", "MAE", "RMSE", "MAPE"]:
                if metric in baseline and metric in improvement_metrics:
                    baseline_val = baseline[metric]
                    improvement_val = improvement_metrics[metric]
                    if metric == "R2":
                        # R2は高い方が良い
                        improvement_pct = ((improvement_val - baseline_val) / baseline_val) * 100
                    else:
                        # MAE, RMSE, MAPEは低い方が良い
                        improvement_pct = ((baseline_val - improvement_val) / baseline_val) * 100
                    
                    overall_comparison[metric] = {
                        "baseline": baseline_val,
                        "improvement": improvement_val,
                        "improvement_pct": improvement_pct
                    }
            
            # 2022-2023年性能の比較
            year_2022_2023_comparison = {}
            if baseline_2022_2023 and improvement_2022_2023:
                for metric in ["R2", "MAE", "RMSE", "MAPE"]:
                    if metric in baseline_2022_2023 and metric in improvement_2022_2023:
                        baseline_val = baseline_2022_2023[metric]
                        improvement_val = improvement_2022_2023[metric]
                        if metric == "R2":
                            improvement_pct = ((improvement_val - baseline_val) / baseline_val) * 100
                        else:
                            improvement_pct = ((baseline_val - improvement_val) / baseline_val) * 100
                        
                        year_2022_2023_comparison[metric] = {
                            "baseline": baseline_val,
                            "improvement": improvement_val,
                            "improvement_pct": improvement_pct
                        }
            
            comparison_results["improvements"][name] = {
                "overall": overall_comparison,
                "2022_2023": year_2022_2023_comparison
            }
        
        # サマリーを生成
        self.comparison_data["summary"] = comparison_results
        self.save_comparison_data()
        
        return comparison_results
    
    def print_comparison_summary(self, reference_baseline: str = "original_baseline"):
        """比較結果のサマリーを表示"""
        if not self.comparison_data.get("summary"):
            print("[ERROR] 比較結果がありません。compare_performance()を先に実行してください。")
            return
        
        summary = self.comparison_data["summary"]
        
        print("\n" + "="*60)
        print("性能比較サマリー")
        print("="*60)
        
        # ベースライン性能
        baseline = summary["baseline"]
        print(f"\n[ベースライン]")
        print(f"R²: {baseline['R2']:.4f}")
        print(f"MAE: {baseline['MAE']:.2f}")
        print(f"RMSE: {baseline['RMSE']:.2f}")
        print(f"MAPE: {baseline['MAPE']:.4f}")
        
        # 2022-2023年性能
        if summary.get("baseline_2022_2023"):
            baseline_2022_2023 = summary["baseline_2022_2023"]
            print(f"\n[ベースライン 2022-2023年]")
            print(f"R²: {baseline_2022_2023['R2']:.4f}")
            print(f"MAE: {baseline_2022_2023['MAE']:.2f}")
            print(f"RMSE: {baseline_2022_2023['RMSE']:.2f}")
            print(f"MAPE: {baseline_2022_2023['MAPE']:.4f}")
        
        # 改善案の比較
        for name, improvement in summary["improvements"].items():
            print(f"\n[{name}]")
            
            # 全体性能
            overall = improvement["overall"]
            print("  全体性能:")
            for metric, data in overall.items():
                print(f"    {metric}: {data['baseline']:.4f} → {data['improvement']:.4f} ({data['improvement_pct']:+.2f}%)")
            
            # 2022-2023年性能
            if improvement.get("2022_2023"):
                year_2022_2023 = improvement["2022_2023"]
                print("  2022-2023年性能:")
                for metric, data in year_2022_2023.items():
                    print(f"    {metric}: {data['baseline']:.4f} → {data['improvement']:.4f} ({data['improvement_pct']:+.2f}%)")
    
    def get_best_improvement(self) -> str:
        """最も効果的な改善案を取得"""
        if not self.comparison_data.get("summary"):
            return ""
        
        best_improvement = ""
        best_r2_improvement = 0
        
        for name, improvement in self.comparison_data["summary"]["improvements"].items():
            r2_improvement = improvement["overall"].get("R2", {}).get("improvement_pct", 0)
            if r2_improvement > best_r2_improvement:
                best_r2_improvement = r2_improvement
                best_improvement = name
        
        return best_improvement

def main():
    """メイン実行"""
    comparator = PerformanceComparator()
    
    # ベースラインを記録
    comparator.record_baseline()
    
    # 比較を実行
    comparator.compare_performance()
    
    # サマリーを表示
    comparator.print_comparison_summary()

if __name__ == "__main__":
    main()
