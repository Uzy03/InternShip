#!/usr/bin/env python3
"""
品質担保のためのスモークテストと回帰テスト
3つのシナリオで全町丁予測を実行し、結果を検証する
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from typing import Dict, List, Tuple

# Add layer5 to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))

def create_test_scenarios() -> Dict[str, Dict]:
    """テスト用の3つのシナリオを作成"""
    
    # シナリオ1: イベントなし
    no_event_scenario = {
        "town": "test_town",  # CLIで置き換えられる
        "base_year": 2025,
        "horizons": [1, 2, 3],
        "events": [],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    # シナリオ2: employment_inc のみ
    employment_only_scenario = {
        "town": "test_town",  # CLIで置き換えられる
        "base_year": 2025,
        "horizons": [1, 2, 3],
        "events": [
            {
                "year_offset": 1,
                "event_type": "employment",
                "effect_direction": "increase",
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1,
                "lag_t1": 1,
                "note": "employment (increase)"
            }
        ],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    # シナリオ3: commercial_inc + employment_inc
    commercial_employment_scenario = {
        "town": "test_town",  # CLIで置き換えられる
        "base_year": 2025,
        "horizons": [1, 2, 3],
        "events": [
            {
                "year_offset": 1,
                "event_type": "commercial",
                "effect_direction": "increase",
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1,
                "lag_t1": 1,
                "note": "commercial (increase)"
            },
            {
                "year_offset": 1,
                "event_type": "employment",
                "effect_direction": "increase",
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1,
                "lag_t1": 1,
                "note": "employment (increase)"
            }
        ],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    return {
        "no_event": no_event_scenario,
        "employment_only": employment_only_scenario,
        "commercial_employment": commercial_employment_scenario
    }

def save_scenario_to_file(scenario: Dict, filepath: str) -> None:
    """シナリオをJSONファイルに保存"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

def validate_delta_consistency(df: pd.DataFrame, tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
    """Δの整合性チェック: delta = exp + macro + inertia + other"""
    errors = []
    
    for idx, row in df.iterrows():
        delta = row['delta']
        exp = row['exp']
        macro = row['macro']
        inertia = row['inertia']
        other = row['other']
        
        # NaNチェック
        if pd.isna(delta) or pd.isna(exp) or pd.isna(macro) or pd.isna(inertia) or pd.isna(other):
            errors.append(f"行{idx}: NaN値が含まれています (town={row['town']}, year={row['year']})")
            continue
        
        # 整合性チェック
        delta_sum = exp + macro + inertia + other
        diff = abs(delta - delta_sum)
        
        if diff > tolerance:
            errors.append(f"行{idx}: Δの整合性エラー (town={row['town']}, year={row['year']}, "
                         f"delta={delta:.6f}, sum={delta_sum:.6f}, diff={diff:.6f})")
    
    return len(errors) == 0, errors

def validate_population_monotonicity(df: pd.DataFrame, town: str) -> Tuple[bool, List[str]]:
    """人口パスの単調性チェック: pop[t] = pop[t-1] + delta[t]"""
    errors = []
    
    town_data = df[df['town'] == town].sort_values('year')
    
    if len(town_data) < 2:
        return True, []  # データが不足している場合はスキップ
    
    for i in range(1, len(town_data)):
        prev_pop = town_data.iloc[i-1]['pop']
        curr_pop = town_data.iloc[i]['pop']
        curr_delta = town_data.iloc[i]['delta']
        
        if pd.isna(prev_pop) or pd.isna(curr_pop) or pd.isna(curr_delta):
            continue  # NaNの場合はスキップ
        
        expected_pop = prev_pop + curr_delta
        diff = abs(curr_pop - expected_pop)
        
        if diff > 1e-6:
            errors.append(f"人口パス単調性エラー (town={town}, year={town_data.iloc[i]['year']}, "
                         f"expected={expected_pop:.6f}, actual={curr_pop:.6f}, diff={diff:.6f})")
    
    return len(errors) == 0, errors

def analyze_distribution_changes(results_dir: Path) -> None:
    """年ごとの分布変化を分析"""
    print("\n=== 分布変化分析 ===")
    
    for scenario_name in ["no_event", "employment_only", "commercial_employment"]:
        csv_path = results_dir / f"forecast_all_rows_{scenario_name}.csv"
        
        if not csv_path.exists():
            print(f"[WARN] {scenario_name}の結果ファイルが見つかりません: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        print(f"\n--- {scenario_name} ---")
        print(f"総行数: {len(df)}")
        
        # 年ごとの統計
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            print(f"  {year}年: 町丁数={len(year_data)}, "
                  f"平均Δ={year_data['delta'].mean():.2f}, "
                  f"最大Δ={year_data['delta'].max():.2f}, "
                  f"最小Δ={year_data['delta'].min():.2f}")

def run_quality_tests(output_dir: str = "../../output") -> None:
    """品質テストを実行"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # テストシナリオを作成
    scenarios = create_test_scenarios()
    
    print("=== 品質担保テスト開始 ===")
    
    # 各シナリオでテスト実行
    for scenario_name, scenario in scenarios.items():
        print(f"\n--- {scenario_name} シナリオ ---")
        
        # シナリオファイルを保存
        scenario_file = output_path / f"test_scenario_{scenario_name}.json"
        save_scenario_to_file(scenario, str(scenario_file))
        
        # CLI実行（subprocessを使用）
        print(f"CLI実行中: {scenario_file}")
        try:
            import subprocess
            result = subprocess.run([
                "python", "cli_run_all.py", 
                "--scenario", str(scenario_file),
                "--output-dir", str(output_path / f"results_{scenario_name}")
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode != 0:
                print(f"[ERROR] {scenario_name}の実行に失敗: {result.stderr}")
                continue
            
            # 結果ファイルをリネーム
            results_csv = output_path / f"results_{scenario_name}" / "forecast_all_rows.csv"
            if results_csv.exists():
                final_csv = output_path / f"forecast_all_rows_{scenario_name}.csv"
                results_csv.rename(final_csv)
                print(f"結果を保存: {final_csv}")
            
        except Exception as e:
            print(f"[ERROR] {scenario_name}の実行に失敗: {e}")
            continue
        
        # 結果の検証
        results_csv = output_path / f"forecast_all_rows_{scenario_name}.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            
            # Δの整合性チェック
            print("Δの整合性チェック中...")
            is_consistent, errors = validate_delta_consistency(df)
            
            if is_consistent:
                print("✅ Δの整合性チェック: OK")
            else:
                print(f"❌ Δの整合性チェック: NG ({len(errors)}件のエラー)")
                for error in errors[:5]:  # 最初の5件のみ表示
                    print(f"  - {error}")
                if len(errors) > 5:
                    print(f"  ... 他{len(errors)-5}件")
            
            # 人口パスの単調性チェック（ランダム5町丁）
            print("人口パス単調性チェック中...")
            towns = df['town'].unique()
            if len(towns) >= 5:
                test_towns = np.random.choice(towns, 5, replace=False)
            else:
                test_towns = towns
            
            monotonicity_errors = []
            for town in test_towns:
                is_monotonic, errors = validate_population_monotonicity(df, town)
                if not is_monotonic:
                    monotonicity_errors.extend(errors)
            
            if len(monotonicity_errors) == 0:
                print("✅ 人口パス単調性チェック: OK")
            else:
                print(f"❌ 人口パス単調性チェック: NG ({len(monotonicity_errors)}件のエラー)")
                for error in monotonicity_errors[:3]:  # 最初の3件のみ表示
                    print(f"  - {error}")
                if len(monotonicity_errors) > 3:
                    print(f"  ... 他{len(monotonicity_errors)-3}件")
            
            # NaN/欠損値チェック
            print("NaN/欠損値チェック中...")
            nan_counts = df.isnull().sum()
            nan_columns = nan_counts[nan_counts > 0]
            
            if len(nan_columns) == 0:
                print("✅ NaN/欠損値チェック: OK")
            else:
                print(f"⚠️ NaN/欠損値チェック: {len(nan_columns)}列に欠損値")
                for col, count in nan_columns.items():
                    print(f"  - {col}: {count}件")
    
    # 分布変化分析
    analyze_distribution_changes(output_path)
    
    print("\n=== 品質担保テスト完了 ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="品質担保テストを実行")
    parser.add_argument("--output_dir", type=str, default="../../output",
                        help="結果を保存するディレクトリのパス")
    
    args = parser.parse_args()
    
    try:
        run_quality_tests(args.output_dir)
    except Exception as e:
        print(f"[CRITICAL ERROR] 品質テスト実行中にエラーが発生しました: {e}")
        sys.exit(1)
