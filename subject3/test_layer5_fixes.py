#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 5修正後の動作確認テスト
"""
import sys
import os
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent / "src" / "dashboard"))
sys.path.append(str(Path(__file__).parent / "src" / "layer5"))

from schema import Scenario, ScenarioEvent

def test_factory_scenario():
    """Factory（四方寄町）シナリオのテスト"""
    print("=== Factory（四方寄町）シナリオテスト ===")
    
    # シナリオ作成
    scn = Scenario(
        town="四方寄町",
        base_year=2025,
        horizons=[1, 2, 3],
        events=[
            ScenarioEvent(
                year_offset=0,
                event_type="employment",
                effect_direction="increase",
                confidence=1.0,
                intensity=1.0,
                lag_t=1,
                lag_t1=1,
                note="工場建設"
            )
        ],
        macros={"foreign_population_growth_pct": {"h1": 0.05, "h2": 0.03, "h3": 0.02}},
        manual_delta={"h1": 100, "h2": 60, "h3": 30}
    )
    
    print(f"シナリオ: {scn.town}, {scn.base_year}")
    print(f"イベント数: {len(scn.events)}")
    print(f"manual_delta: {scn.manual_delta}")
    
    # 衝突チェック
    warnings = scn.validate_conflicts()
    if warnings:
        print("警告:", warnings)
    else:
        print("衝突なし")
    
    return scn

def test_housing_scenario():
    """Housing（九品寺5丁目）シナリオのテスト"""
    print("\n=== Housing（九品寺5丁目）シナリオテスト ===")
    
    # シナリオ作成
    scn = Scenario(
        town="九品寺5丁目",
        base_year=2025,
        horizons=[1, 2, 3],
        events=[
            ScenarioEvent(
                year_offset=0,
                event_type="housing",
                effect_direction="increase",
                confidence=1.0,
                intensity=1.0,
                lag_t=1,
                lag_t1=1,
                note="住宅開発"
            )
        ],
        macros={},
        manual_delta={}
    )
    
    print(f"シナリオ: {scn.town}, {scn.base_year}")
    print(f"イベント数: {len(scn.events)}")
    print(f"manual_delta: {scn.manual_delta}")
    
    # 衝突チェック
    warnings = scn.validate_conflicts()
    if warnings:
        print("警告:", warnings)
    else:
        print("衝突なし")
    
    return scn

def test_scenario_to_events():
    """scenario_to_events のテスト"""
    print("\n=== scenario_to_events テスト ===")
    
    from scenario_to_events import from_scenario
    
    # Factoryシナリオ
    scn_factory = test_factory_scenario()
    events_df = from_scenario(scn_factory)
    print(f"Factory イベント行列: {len(events_df)} 行")
    print("列:", list(events_df.columns))
    
    # 非ゼロ列を確認
    non_zero_cols = []
    for col in events_df.columns:
        if col.startswith("event_") and (events_df[col] != 0).any():
            non_zero_cols.append(col)
    
    print("非ゼロ列:", non_zero_cols)
    for col in non_zero_cols:
        print(f"  {col}: {events_df[col].values}")

def test_build_future_features():
    """build_future_features のテスト"""
    print("\n=== build_future_features テスト ===")
    
    # データファイルの存在確認
    features_panel_path = Path("data/processed/features_panel.csv")
    if not features_panel_path.exists():
        print(f"❌ {features_panel_path} が見つかりません")
        print("データファイルを確認してください")
        return
    
    # パスの確認
    print(f"現在のディレクトリ: {Path.cwd()}")
    print(f"features_panel.csv のパス: {features_panel_path.absolute()}")
    print(f"ファイル存在確認: {features_panel_path.exists()}")
    
    from scenario_to_events import from_scenario
    from prepare_baseline import get_baseline
    from build_future_features import build
    
    # Factoryシナリオ
    scn = test_factory_scenario()
    
    # 将来イベント行列
    fut_events_df = from_scenario(scn)
    print(f"将来イベント行列: {len(fut_events_df)} 行")
    
    # ベースライン
    baseline_df = get_baseline(scn.town, scn.base_year)
    print(f"ベースライン: {len(baseline_df)} 行")
    print("ベースライン列:", list(baseline_df.columns))
    
    # 将来特徴
    fut_features_df = build(scn, baseline_df, fut_events_df)
    print(f"将来特徴: {len(fut_features_df)} 行")
    print("将来特徴列:", list(fut_features_df.columns))
    
    # exp_all_h* の値を確認
    for h in [1, 2, 3]:
        exp_col = f"exp_all_h{h}"
        if exp_col in fut_features_df.columns:
            values = fut_features_df[exp_col].values
            print(f"{exp_col}: {values}")

if __name__ == "__main__":
    print("Layer 5修正後の動作確認テスト開始")
    
    try:
        # シナリオ作成テスト
        test_factory_scenario()
        test_housing_scenario()
        
        # 各モジュールのテスト
        test_scenario_to_events()
        test_build_future_features()
        
        print("\n✅ すべてのテストが完了しました")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
