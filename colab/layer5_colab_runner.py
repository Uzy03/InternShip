# -*- coding: utf-8 -*-
"""
Colab用レイヤー5実行スクリプト
CLIに依存せず、Colabで直接実行可能

使用方法:
1. 必要なファイルをColabにアップロード
2. このスクリプトを実行
3. シナリオ辞書を直接指定して実行
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# プロジェクトルートをパスに追加
project_root = "/content/インターンシップ本課題_地域科学研究所"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "subject3", "src"))

# レイヤー5のモジュールをインポート
try:
    from subject3.src.layer5.scenario_to_events import scenario_to_events
    from subject3.src.layer5.prepare_baseline import prepare_baseline
    from subject3.src.layer5.build_future_features import build_future_features
    from subject3.src.layer5.forecast_service import forecast_population
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なファイルがアップロードされているか確認してください")
    sys.exit(1)

class Layer5ColabRunner:
    """Colab用レイヤー5実行クラス"""
    
    def __init__(self, project_root: str = "/content/インターンシップ本課題_地域科学研究所"):
        self.project_root = project_root
        self.output_dir = os.path.join(project_root, "subject3", "data", "processed")
        
        # 出力ディレクトリの作成
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # パス設定
        self.p_baseline = os.path.join(self.output_dir, "l5_baseline.csv")
        self.p_future_events = os.path.join(self.output_dir, "l5_future_events.csv")
        self.p_future_features = os.path.join(self.output_dir, "l5_future_features.csv")
    
    def run_scenario(self, scenario: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """シナリオを実行して予測結果を返す"""
        print(f"[L5] シナリオを実行中: {scenario.get('town', 'Unknown')}")
        
        town = scenario["town"]
        base_year = scenario["base_year"]
        horizons = scenario["horizons"]
        
        print(f"[L5] 町丁: {town}, 基準年: {base_year}, 予測期間: {horizons}")
        
        try:
            # Step 1: 将来イベント行列の生成
            print("\n[L5] Step 1: 将来イベント行列を生成中...")
            future_events = scenario_to_events(scenario)
            future_events.to_csv(self.p_future_events, index=False)
            print(f"[L5] 将来イベント行列を保存: {self.p_future_events}")
            
            # Step 2: 基準年データの準備
            print("\n[L5] Step 2: 基準年データを準備中...")
            baseline = prepare_baseline(town, base_year)
            baseline.to_csv(self.p_baseline, index=False)
            print(f"[L5] 基準年データを保存: {self.p_baseline}")
            
            # Step 3: 将来特徴の構築
            print("\n[L5] Step 3: 将来特徴を構築中...")
            future_features = build_future_features(baseline, future_events, scenario)
            future_features.to_csv(self.p_future_features, index=False)
            print(f"[L5] 将来特徴を保存: {self.p_future_features}")
            
            # Step 4: 人口予測の実行
            print("\n[L5] Step 4: 人口予測を実行中...")
            base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
            if pd.isna(base_population):
                print("[WARN] ベース人口が不明のため、0を使用します")
                base_population = 0.0
            
            result = forecast_population(town, base_year, horizons, base_population)
            
            # Step 5: 結果の保存
            if output_path is None:
                scenario_name = scenario.get("name", "scenario")
                output_path = os.path.join(self.output_dir, f"l5_forecast_{scenario_name}.json")
            
            # 出力ディレクトリの作成
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"[L5] 予測結果を保存: {output_path}")
            
            # 結果のサマリー表示
            self._print_result_summary(result)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] シナリオ実行中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """結果のサマリーを表示"""
        print("\n[L5] 予測結果サマリー:")
        print(f"  町丁: {result['town']}")
        print(f"  基準年: {result['base_year']}")
        print(f"  予測期間: {result['horizons']}")
        print("  予測結果:")
        
        for path_entry in result["path"]:
            year = path_entry["year"]
            delta = path_entry["delta_hat"]
            pop = path_entry["pop_hat"]
            pi_delta = path_entry["pi95_delta"]
            pi_pop = path_entry["pi95_pop"]
            contrib = path_entry["contrib"]
            
            print(f"    {year}年: Δ={delta:+.1f}人, 人口={pop:.1f}人")
            print(f"      PI95(Δ): [{pi_delta[0]:.1f}, {pi_delta[1]:.1f}], PI95(人口): [{pi_pop[0]:.1f}, {pi_pop[1]:.1f}]")
            print(f"      寄与: exp={contrib['exp']:+.1f}, macro={contrib['macro']:+.1f}, inertia={contrib['inertia']:+.1f}, other={contrib['other']:+.1f}")

# 使用例のシナリオ
EXAMPLE_SCENARIOS = {
    "housing_boost": {
        "name": "housing_boost",
        "town": "九品寺5丁目",
        "base_year": 2025,
        "horizons": [1, 2, 3],
        "base_population": 7000.0,
        "events": [
            {
                "type": "housing",
                "direction": "increase",
                "year_offset": 1,
                "confidence": 0.8,
                "intensity": 0.5,
                "lag_t": 0.3,
                "lag_t1": 0.7
            }
        ],
        "macros": {
            "foreign_pop_growth_rate": 0.02
        },
        "manual_delta": {
            "h1": 0.0,
            "h2": 0.0,
            "h3": 0.0
        }
    },
    
    "commercial_development": {
        "name": "commercial_development",
        "town": "九品寺5丁目",
        "base_year": 2025,
        "horizons": [1, 2, 3],
        "base_population": 7000.0,
        "events": [
            {
                "type": "commercial",
                "direction": "increase",
                "year_offset": 2,
                "confidence": 0.9,
                "intensity": 0.6,
                "lag_t": 0.4,
                "lag_t1": 0.6
            }
        ],
        "macros": {
            "foreign_pop_growth_rate": 0.015
        },
        "manual_delta": {
            "h1": 0.0,
            "h2": 0.0,
            "h3": 0.0
        }
    }
}

def main():
    """メイン実行関数"""
    print("=== レイヤー5 Colab実行スクリプト ===")
    
    # ランナーの初期化
    runner = Layer5ColabRunner()
    
    # 例1: 住宅開発シナリオ
    print("\n=== 例1: 住宅開発シナリオ ===")
    scenario1 = EXAMPLE_SCENARIOS["housing_boost"]
    result1 = runner.run_scenario(scenario1)
    
    # 例2: 商業開発シナリオ
    print("\n=== 例2: 商業開発シナリオ ===")
    scenario2 = EXAMPLE_SCENARIOS["commercial_development"]
    result2 = runner.run_scenario(scenario2)
    
    print("\n=== 実行完了 ===")
    return result1, result2

# カスタムシナリオ実行用の関数
def run_custom_scenario(scenario_dict: Dict[str, Any], project_root: str = "/content/インターンシップ本課題_地域科学研究所"):
    """カスタムシナリオを実行"""
    runner = Layer5ColabRunner(project_root)
    return runner.run_scenario(scenario_dict)

if __name__ == "__main__":
    main()
