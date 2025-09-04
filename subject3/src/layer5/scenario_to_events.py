# -*- coding: utf-8 -*-
# src/layer5/scenario_to_events.py
"""
シナリオJSON → 将来イベント行列（L2互換の形）への変換
出力: data/processed/l5_future_events.csv

設計:
- 入力: シナリオJSON
- 出力: DataFrame（列：town, year, event_<type>_t, event_<type>_t1）
- 年 = base_year + year_offset
- スコア s = clip(confidence * intensity, 0,1)
- 符号 sign = +1(increase) / -1(decrease)
- event_<type>_t = sign * s * lag_t、event_<type>_t1 = sign * s * lag_t1
- 同じ (town,year,type) に複数定義が来たら 和を [-1,1] にクリップ
- 衝突ルール：policy_boundary 優先で transit 無効化
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# パス設定
P_FEATURES_PANEL = "../../data/processed/features_panel.csv"
P_OUTPUT = "../../data/processed/l5_future_events.csv"

# イベントタイプの定義
EVENT_TYPES = {
    "housing", "commercial", "transit", "policy_boundary", 
    "public_edu_medical", "employment", "disaster"
}

def validate_scenario(scenario: Dict[str, Any]) -> None:
    """シナリオJSONのバリデーション"""
    # 基本キーの存在チェック
    required_keys = ["town", "base_year", "horizons", "events"]
    for key in required_keys:
        if key not in scenario:
            raise ValueError(f"必須キー '{key}' が不足しています")
    
    # townの存在チェック
    panel = pd.read_csv(P_FEATURES_PANEL)
    if scenario["town"] not in panel["town"].unique():
        raise ValueError(f"町丁 '{scenario['town']}' がfeatures_panel.csvに存在しません")
    
    # base_yearの範囲チェック
    year_range = (panel["year"].min(), panel["year"].max())
    if not (year_range[0] <= scenario["base_year"] <= year_range[1]):
        raise ValueError(f"base_year {scenario['base_year']} が範囲 {year_range} 外です")
    
    # horizonsのチェック
    valid_horizons = {1, 2, 3}
    if not set(scenario["horizons"]).issubset(valid_horizons):
        raise ValueError(f"horizons {scenario['horizons']} が有効範囲 {valid_horizons} 外です")
    
    # イベントのバリデーション
    for i, event in enumerate(scenario["events"]):
        # 必須キー
        event_required = ["year_offset", "event_type", "effect_direction", 
                         "confidence", "intensity", "lag_t", "lag_t1"]
        for key in event_required:
            if key not in event:
                raise ValueError(f"イベント {i}: 必須キー '{key}' が不足しています")
        
        # event_typeのチェック
        if event["event_type"] not in EVENT_TYPES:
            raise ValueError(f"イベント {i}: event_type '{event['event_type']}' が無効です")
        
        # effect_directionのチェック
        if event["effect_direction"] not in {"increase", "decrease"}:
            raise ValueError(f"イベント {i}: effect_direction '{event['effect_direction']}' が無効です")
        
        # 数値範囲のチェック
        if not (0 <= event["confidence"] <= 1):
            raise ValueError(f"イベント {i}: confidence {event['confidence']} が範囲 [0,1] 外です")
        
        if not (0 <= event["intensity"] <= 1):
            raise ValueError(f"イベント {i}: intensity {event['intensity']} が範囲 [0,1] 外です")
        
        if event["lag_t"] not in {0, 1}:
            raise ValueError(f"イベント {i}: lag_t {event['lag_t']} が {0,1} 外です")
        
        if event["lag_t1"] not in {0, 1}:
            raise ValueError(f"イベント {i}: lag_t1 {event['lag_t1']} が {0,1} 外です")

def apply_conflict_rules(events_df: pd.DataFrame) -> pd.DataFrame:
    """衝突ルールの適用：policy_boundary 優先で transit 無効化"""
    # 同じ (town, year) で policy_boundary と transit が併存する場合
    conflict_mask = (
        (events_df["event_policy_boundary_t"] != 0) | 
        (events_df["event_policy_boundary_t1"] != 0)
    ) & (
        (events_df["event_transit_t"] != 0) | 
        (events_df["event_transit_t1"] != 0)
    )
    
    if conflict_mask.any():
        print(f"[WARN] {conflict_mask.sum()} 行で policy_boundary と transit の衝突を検出。transit を無効化します。")
        events_df.loc[conflict_mask, "event_transit_t"] = 0
        events_df.loc[conflict_mask, "event_transit_t1"] = 0
    
    return events_df

def scenario_to_events(scenario: Dict[str, Any]) -> pd.DataFrame:
    """シナリオJSONを将来イベント行列に変換"""
    # バリデーション
    validate_scenario(scenario)
    
    town = scenario["town"]
    base_year = scenario["base_year"]
    events = scenario["events"]
    
    # 将来年の範囲を計算
    max_horizon = max(scenario["horizons"])
    years = list(range(base_year, base_year + max_horizon + 1))
    
    # イベント行列の初期化
    event_cols = []
    for event_type in EVENT_TYPES:
        event_cols.extend([f"event_{event_type}_t", f"event_{event_type}_t1"])
    
    events_df = pd.DataFrame({
        "town": [town] * len(years),
        "year": years
    })
    
    for col in event_cols:
        events_df[col] = 0.0
    
    # イベントの処理
    for event in events:
        year_offset = event["year_offset"]
        event_type = event["event_type"]
        effect_direction = event["effect_direction"]
        confidence = event["confidence"]
        intensity = event["intensity"]
        lag_t = event["lag_t"]
        lag_t1 = event["lag_t1"]
        
        # スコアと符号の計算
        s = np.clip(confidence * intensity, 0, 1)
        sign = 1 if effect_direction == "increase" else -1
        
        # 対象年
        target_year = base_year + year_offset
        
        if target_year in years:
            year_idx = years.index(target_year)
            
            # 当年効果
            if lag_t == 1:
                col_t = f"event_{event_type}_t"
                events_df.loc[year_idx, col_t] += sign * s
            
            # 翌年効果
            if lag_t1 == 1 and target_year + 1 in years:
                next_year_idx = years.index(target_year + 1)
                col_t1 = f"event_{event_type}_t1"
                events_df.loc[next_year_idx, col_t1] += sign * s
    
    # 同じ (town,year,type) の重複をクリップ
    for event_type in EVENT_TYPES:
        for lag in ["t", "t1"]:
            col = f"event_{event_type}_{lag}"
            events_df[col] = np.clip(events_df[col], -1, 1)
    
    # 衝突ルールの適用
    events_df = apply_conflict_rules(events_df)
    
    return events_df

def main(scenario_path: str) -> None:
    """メイン処理"""
    # シナリオJSONの読み込み
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario = json.load(f)
    
    # 将来イベント行列の生成
    events_df = scenario_to_events(scenario)
    
    # 出力ディレクトリの作成
    Path(P_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    
    # 健全性チェック
    validate_event_matrix(events_df)
    
    # 保存
    events_df.to_csv(P_OUTPUT, index=False)
    print(f"[L5] 将来イベント行列を保存しました: {P_OUTPUT}")
    print(f"[L5] 行数: {len(events_df)}, 列数: {len(events_df.columns)}")
    
    # デバッグ情報
    non_zero_cols = []
    for col in events_df.columns:
        if col.startswith("event_") and (events_df[col] != 0).any():
            non_zero_cols.append(col)
    
    if non_zero_cols:
        print(f"[L5] 非ゼロ列: {non_zero_cols}")
        for col in non_zero_cols:
            non_zero_rows = events_df[events_df[col] != 0]
            print(f"  {col}: {len(non_zero_rows)} 行")
    else:
        print("[L5] 非ゼロのイベント列はありません")

def validate_event_matrix(events_df: pd.DataFrame) -> None:
    """イベント行列の健全性チェック"""
    print("[L5] イベント行列の健全性チェック中...")
    
    # 各年のイベント列の合計をチェック
    event_cols = [col for col in events_df.columns if col.startswith("event_")]
    
    for year in events_df["year"].unique():
        year_data = events_df[events_df["year"] == year]
        if len(year_data) == 0:
            continue
            
        row = year_data.iloc[0]
        total_events = 0
        non_zero_events = []
        
        for col in event_cols:
            if col in row and not pd.isna(row[col]) and row[col] != 0:
                total_events += abs(row[col])
                non_zero_events.append(f"{col}={row[col]}")
        
        if total_events > 0:
            print(f"  年 {year}: イベント合計={total_events:.2f}, 非ゼロ={non_zero_events}")
        else:
            print(f"  年 {year}: イベントなし")
        
        # 各年の sum(|event_*_t| + |event_*_t1|) をチェック
        t_events = 0
        t1_events = 0
        for col in event_cols:
            if col in row and not pd.isna(row[col]):
                if col.endswith("_t"):
                    t_events += abs(row[col])
                elif col.endswith("_t1"):
                    t1_events += abs(row[col])
        
        print(f"    年 {year}: |event_*_t|合計={t_events:.2f}, |event_*_t1|合計={t1_events:.2f}")
    
    # policy_boundary と transit の衝突チェック
    conflict_years = []
    for year in events_df["year"].unique():
        year_data = events_df[events_df["year"] == year]
        if len(year_data) == 0:
            continue
            
        row = year_data.iloc[0]
        has_policy = False
        has_transit = False
        
        for col in event_cols:
            if "policy_boundary" in col and not pd.isna(row[col]) and row[col] != 0:
                has_policy = True
            if "transit" in col and not pd.isna(row[col]) and row[col] != 0:
                has_transit = True
        
        if has_policy and has_transit:
            conflict_years.append(year)
    
    if conflict_years:
        print(f"[WARN] policy_boundary と transit の衝突を検出: 年 {conflict_years}")
    else:
        print("[L5] 衝突なし")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python scenario_to_events.py <scenario.json>")
        sys.exit(1)
    
    main(sys.argv[1])
