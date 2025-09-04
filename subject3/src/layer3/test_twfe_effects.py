#!/usr/bin/env python3
"""
Layer3 TWFE Effects 実装のテストスクリプト

このスクリプトは、実装されたTWFE推定器の動作を確認します。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from twfe_effects import TWFEEstimator

def test_basic_functionality():
    """基本機能のテスト"""
    print("=== Layer3 TWFE Effects 基本機能テスト ===")
    
    # 推定器を初期化
    estimator = TWFEEstimator()
    
    # データ読み込みテスト
    print("\n1. データ読み込みテスト")
    df = estimator.load_data(use_year_min=1998, use_year_max=2025)
    print(f"   ✓ データ読み込み成功: {len(df)} 行")
    print(f"   ✓ 町丁数: {df['town'].nunique()}")
    print(f"   ✓ 年範囲: {df['year'].min()}-{df['year'].max()}")
    
    # コントロール変数追加テスト
    print("\n2. コントロール変数追加テスト")
    estimator.add_controls()
    control_cols = [col for col in df.columns if 'dummy' in col or 'growth_adj' in col]
    print(f"   ✓ コントロール変数: {control_cols}")
    
    return estimator

def test_standard_estimation():
    """標準推定のテスト"""
    print("\n=== 標準TWFE推定テスト ===")
    
    estimator = TWFEEstimator()
    
    # 標準推定を実行
    results = estimator.run_analysis(
        use_year_min=1998,
        use_year_max=2025,
        separate_t_and_t1=False,  # 多重共線性回避のため合算
        use_signed_from_labeled=False,
        weight_mode="sqrt_prev",
        ridge_alpha=0.0,
        include_growth_adj=True,
        include_policy_boundary=True,
        include_disaster_2016=True
    )
    
    print(f"\n推定結果: {len(results)} 個のイベントタイプ")
    print("\n主要係数:")
    for _, row in results.head().iterrows():
        print(f"  {row['event_var']:15s}: β = {row['beta']:8.3f} "
              f"(γ₀ = {row['gamma0']:.3f}, γ₁ = {row['gamma1']:.3f})")
    
    # 受け入れ基準の確認
    print("\n=== 受け入れ基準確認 ===")
    criteria = [
        ("effects_coefficients.csv 生成", 
         "effects_coefficients.csv" in [f.name for f in estimator.output_dir.glob("*.csv")]),
        ("主要カテゴリのβがNaNなし", not results["beta"].isna().any()),
        ("gamma0 + gamma1 ≈ 1", abs(results["gamma0"] + results["gamma1"] - 1).max() < 1e-6),
        ("観測数 > 0", results["n_obs"].iloc[0] > 0 if len(results) > 0 else False),
        ("R² > 0", results["r2_within"].iloc[0] > 0 if len(results) > 0 else False)
    ]
    
    for criterion, passed in criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
    
    return results

def test_signed_estimation():
    """有向イベント推定のテスト"""
    print("\n=== 有向イベントTWFE推定テスト ===")
    
    estimator = TWFEEstimator()
    
    # 有向イベント推定を実行
    results = estimator.run_analysis(
        use_year_min=1998,
        use_year_max=2025,
        separate_t_and_t1=False,  # 多重共線性回避のため合算
        use_signed_from_labeled=True,  # 有向イベントを使用
        weight_mode="sqrt_prev",
        ridge_alpha=0.0,
        include_growth_adj=True,
        include_policy_boundary=True,
        include_disaster_2016=True
    )
    
    print(f"\n推定結果: {len(results)} 個のイベントタイプ")
    
    # 有向イベントの結果を表示
    signed_results = results[results['event_var'].str.contains('_inc|_dec')]
    if len(signed_results) > 0:
        print("\n有向イベント係数:")
        for _, row in signed_results.head(10).iterrows():
            print(f"  {row['event_var']:25s}: β = {row['beta']:8.3f}")
    
    return results

def main():
    """メイン実行関数"""
    try:
        # 基本機能テスト
        estimator = test_basic_functionality()
        
        # 標準推定テスト
        standard_results = test_standard_estimation()
        
        # 有向イベント推定テスト
        signed_results = test_signed_estimation()
        
        print("\n=== テスト完了 ===")
        print("✓ すべてのテストが正常に完了しました")
        print(f"✓ 標準推定: {len(standard_results)} 個のイベントタイプ")
        print(f"✓ 有向推定: {len(signed_results)} 個のイベントタイプ")
        
        # 出力ファイルの確認
        output_dir = Path("/Users/ujihara/インターンシップ本課題_地域科学研究所/subject3/output")
        if output_dir.exists():
            csv_files = list(output_dir.glob("*.csv"))
            print(f"✓ 出力ファイル: {[f.name for f in csv_files]}")
        
    except Exception as e:
        print(f"\n✗ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
