# -*- coding: utf-8 -*-
# src/layer5/cli_run_scenario.py
"""
CLI エントリポイント：シナリオJSONから人口予測を実行
使用方法: python cli_run_scenario.py <scenario.json> [output.json]
"""
import json
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# 各モジュールのインポート
from scenario_to_events import scenario_to_events
from prepare_baseline import prepare_baseline
from build_future_features import build_future_features
from forecast_service import forecast_population

# パス設定
P_OUTPUT_DIR = "../../data/processed"
P_BASELINE = f"{P_OUTPUT_DIR}/l5_baseline.csv"
P_FUTURE_EVENTS = f"{P_OUTPUT_DIR}/l5_future_events.csv"
P_FUTURE_FEATURES = f"{P_OUTPUT_DIR}/l5_future_features.csv"

def run_scenario(scenario_path: str, output_path: str = None) -> Dict[str, Any]:
    """シナリオを実行して予測結果を返す"""
    print(f"[L5] シナリオを実行中: {scenario_path}")
    
    # シナリオJSONの読み込み
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)
    
    town = scenario["town"]
    base_year = scenario["base_year"]
    horizons = scenario["horizons"]
    
    print(f"[L5] 町丁: {town}, 基準年: {base_year}, 予測期間: {horizons}")
    
    # Step 1: 将来イベント行列の生成
    print("\n[L5] Step 1: 将来イベント行列を生成中...")
    future_events = scenario_to_events(scenario)
    future_events.to_csv(P_FUTURE_EVENTS, index=False)
    print(f"[L5] 将来イベント行列を保存: {P_FUTURE_EVENTS}")
    
    # Step 2: 基準年データの準備
    print("\n[L5] Step 2: 基準年データを準備中...")
    baseline = prepare_baseline(town, base_year)
    baseline.to_csv(P_BASELINE, index=False)
    print(f"[L5] 基準年データを保存: {P_BASELINE}")
    
    # Step 3: 将来特徴の構築
    print("\n[L5] Step 3: 将来特徴を構築中...")
    future_features = build_future_features(baseline, future_events, scenario)
    future_features.to_csv(P_FUTURE_FEATURES, index=False)
    print(f"[L5] 将来特徴を保存: {P_FUTURE_FEATURES}")
    
    # Step 4: 人口予測の実行
    print("\n[L5] Step 4: 人口予測を実行中...")
    base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
    if pd.isna(base_population):
        print("[WARN] ベース人口が不明のため、0を使用します")
        base_population = 0.0
    
    result = forecast_population(town, base_year, horizons, base_population)
    
    # Step 5: 結果の保存
    if output_path is None:
        scenario_name = Path(scenario_path).stem
        output_path = f"{P_OUTPUT_DIR}/l5_forecast_{scenario_name}.json"
    
    # 出力ディレクトリの作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"[L5] 予測結果を保存: {output_path}")
    
    # 結果のサマリー表示
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
    
    return result

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("使用方法: python cli_run_scenario.py <scenario.json> [output.json]")
        print("例: python cli_run_scenario.py scenario_examples/housing_boost.json")
        sys.exit(1)
    
    scenario_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # シナリオファイルの存在確認
    if not Path(scenario_path).exists():
        print(f"エラー: シナリオファイルが見つかりません: {scenario_path}")
        sys.exit(1)
    
    try:
        # シナリオの実行
        result = run_scenario(scenario_path, output_path)
        
        print("\n[L5] シナリオ実行が完了しました！")
        
    except Exception as e:
        print(f"\n[ERROR] シナリオ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
