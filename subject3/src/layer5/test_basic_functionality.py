#!/usr/bin/env python3
"""
基本的な機能テスト
単一町丁での予測と全町丁予測の動作確認
"""

import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add layer5 to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))
from forecast_service import run_scenario

def test_single_town_prediction():
    """単一町丁予測のテスト"""
    print("=== 単一町丁予測テスト ===")
    
    # テストシナリオ
    test_scenario = {
        "town": "万楽寺町",
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
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0},
        "base_population": 1000.0
    }
    
    try:
        # 予測実行
        result = run_scenario(test_scenario, out_path="../../output/test_single")
        
        # 結果の検証
        print(f"町丁: {result['town']}")
        print(f"基準年: {result['baseline_year']}")
        print(f"予測期間: {result['horizons']}")
        print(f"結果数: {len(result['results'])}")
        
        # 各年の結果を表示
        for entry in result['results']:
            year = entry['year']
            delta = entry['delta']
            pop = entry['pop']
            contrib = entry['contrib']
            
            print(f"  {year}年: Δ={delta:.1f}, 人口={pop:.1f}")
            print(f"    寄与: exp={contrib['exp']:.1f}, macro={contrib['macro']:.1f}, "
                  f"inertia={contrib['inertia']:.1f}, other={contrib['other']:.1f}")
            
            # Δの整合性チェック
            delta_sum = contrib['exp'] + contrib['macro'] + contrib['inertia'] + contrib['other']
            diff = abs(delta - delta_sum)
            if diff < 1e-6:
                print(f"    ✅ Δ整合性OK (diff={diff:.2e})")
            else:
                print(f"    ❌ Δ整合性NG (diff={diff:.2e})")
        
        print("✅ 単一町丁予測テスト: 成功")
        return True
        
    except Exception as e:
        print(f"❌ 単一町丁予測テスト: 失敗 - {e}")
        return False

def test_all_towns_prediction():
    """全町丁予測のテスト（小規模サンプル）"""
    print("\n=== 全町丁予測テスト ===")
    
    # テスト用シナリオ（イベントなし）
    test_scenario = {
        "town": "test_town",  # CLIで置き換えられる
        "base_year": 2025,
        "horizons": [1, 2, 3],
        "events": [],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    # シナリオファイルを保存
    scenario_file = Path("../../output/test_scenario_basic.json")
    scenario_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(scenario_file, 'w', encoding='utf-8') as f:
        json.dump(test_scenario, f, ensure_ascii=False, indent=2)
    
    try:
        # CLI実行（subprocessを使用）
        import subprocess
        print("全町丁予測を実行中...")
        result = subprocess.run([
            "python", "cli_run_all.py", 
            "--scenario", str(scenario_file),
            "--output-dir", "../../output/test_all"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"❌ CLI実行エラー: {result.stderr}")
            return False
        
        # 結果の確認
        results_csv = Path("../../output/test_all/forecast_all_rows.csv")
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            print(f"✅ 全町丁予測テスト: 成功")
            print(f"  結果行数: {len(df)}")
            print(f"  町丁数: {df['town'].nunique()}")
            print(f"  年数: {df['year'].nunique()}")
            
            # サンプル結果を表示
            print("  サンプル結果（最初の5行）:")
            sample_cols = ['town', 'year', 'delta', 'pop', 'exp', 'macro', 'inertia', 'other']
            print(df[sample_cols].head().to_string(index=False))
            
            return True
        else:
            print("❌ 全町丁予測テスト: 結果ファイルが見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ 全町丁予測テスト: 失敗 - {e}")
        return False

def test_dashboard_data_loading():
    """ダッシュボードのデータ読み込みテスト"""
    print("\n=== ダッシュボードデータ読み込みテスト ===")
    
    # 全町丁予測の結果ファイルを確認
    results_csv = Path("../../output/test_all/forecast_all_rows.csv")
    
    if not results_csv.exists():
        print("❌ ダッシュボードデータ読み込みテスト: 結果ファイルが見つかりません")
        print("   先に全町丁予測を実行してください")
        return False
    
    try:
        # データ読み込み
        df = pd.read_csv(results_csv)
        
        # 必要な列の存在確認
        required_columns = ['town', 'baseline_year', 'year', 'h', 'delta', 'pop', 
                           'exp', 'macro', 'inertia', 'other', 
                           'pi_delta_low', 'pi_delta_high', 'pi_pop_low', 'pi_pop_high']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ ダッシュボードデータ読み込みテスト: 必要な列が不足 - {missing_columns}")
            return False
        
        # データの基本統計
        print(f"✅ ダッシュボードデータ読み込みテスト: 成功")
        print(f"  総行数: {len(df)}")
        print(f"  町丁数: {df['town'].nunique()}")
        print(f"  年範囲: {df['year'].min()}-{df['year'].max()}")
        
        # 年ごとの統計
        year_stats = df.groupby('year').agg({
            'delta': ['count', 'mean', 'std'],
            'pop': ['mean', 'min', 'max']
        }).round(2)
        
        print("  年ごとの統計:")
        print(year_stats.to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ ダッシュボードデータ読み込みテスト: 失敗 - {e}")
        return False

def main():
    """メインテスト実行"""
    print("=== 基本機能テスト開始 ===")
    
    # テスト実行
    test1 = test_single_town_prediction()
    test2 = test_all_towns_prediction()
    test3 = test_dashboard_data_loading()
    
    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    print(f"単一町丁予測: {'✅ 成功' if test1 else '❌ 失敗'}")
    print(f"全町丁予測: {'✅ 成功' if test2 else '❌ 失敗'}")
    print(f"ダッシュボードデータ読み込み: {'✅ 成功' if test3 else '❌ 失敗'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 すべてのテストが成功しました！")
        return 0
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
