"""
L5一括オーケストレーター
既存L5の各モジュールを1関数に束ねる
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import os

# L5モジュールのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))

from schema import Scenario
from scenario_to_events import scenario_to_events
from prepare_baseline import prepare_baseline
from build_future_features import build_future_features
from forecast_service import forecast_population
from intervals import pi95_delta, pi95_pop

DATA_DIR = Path("../../data/processed")


def run_scenario(scn: Scenario, debug: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    シナリオを実行して予測結果を返す
    
    Args:
        scn: シナリオオブジェクト
        debug: デバッグ出力を有効にするか
        
    Returns:
        (result, debug_info): 予測結果とデバッグ情報
    """
    print(f"[Dashboard] シナリオ実行開始: {scn.town}, {scn.base_year}")
    
    # 1) 検証
    df_panel = pd.read_csv(DATA_DIR / "features_panel.csv")  # 全データを読み込み
    print(f"[Dashboard] 検証中: 町丁='{scn.town}', データ行数={len(df_panel)}")
    print(f"[Dashboard] 利用可能な町丁（最初の10個）: {sorted(df_panel['town'].unique())[:10]}")
    print(f"[Dashboard] 新屋敷1丁目が存在するか: {'新屋敷1丁目' in df_panel['town'].values}")
    
    if not (df_panel["town"] == scn.town).any():
        available_towns = sorted(df_panel["town"].unique())
        print(f"[Dashboard] 利用可能な町丁: {available_towns[:20]}...")
        raise ValueError(f"町丁が見つかりません: {scn.town}")
    
    # 衝突チェック
    warnings = scn.validate_conflicts()
    if warnings:
        print("[Dashboard] 警告:")
        for warning in warnings:
            print(f"  {warning}")
    
    # 2) 将来イベント行列
    print("[Dashboard] 将来イベント行列を生成中...")
    fut_events = scenario_to_events(scn.dict())  # DataFrame
    fut_events.to_csv(DATA_DIR / "l5_future_events.csv", index=False)
    
    # 3) ベース行
    print("[Dashboard] 基準年データを準備中...")
    baseline = prepare_baseline(scn.town, scn.base_year)
    baseline.to_csv(DATA_DIR / "l5_baseline.csv", index=False)
    
    # 4) 将来特徴
    print("[Dashboard] 将来特徴を構築中...")
    fut_features = build_future_features(baseline, fut_events, scn.dict())
    fut_features.to_csv(DATA_DIR / "l5_future_features.csv", index=False)
    
    # 5) 予測
    print("[Dashboard] 人口予測を実行中...")
    base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
    result = forecast_population(scn.town, scn.base_year, scn.horizons, base_population)
    
    # デバッグ情報の保存
    debug_info = {}
    if debug:
        # デバッグファイルが存在する場合は読み込む
        debug_features_path = DATA_DIR / f"l5_debug_features_{scn.town}.csv"
        debug_contrib_path = DATA_DIR / f"l5_debug_contrib_{scn.town}.csv"
        
        if debug_features_path.exists():
            debug_info["debug_features"] = pd.read_csv(debug_features_path)
        if debug_contrib_path.exists():
            debug_info["debug_contrib"] = pd.read_csv(debug_contrib_path)
    
    print(f"[Dashboard] シナリオ実行完了: {scn.town}")
    return result, debug_info


def load_metadata() -> Tuple[list, list]:
    """
    メタデータを読み込む
    
    Returns:
        (towns, years): 町丁リストと年リスト
    """
    try:
        df = pd.read_csv(DATA_DIR / "features_panel.csv", usecols=["town", "year"]).drop_duplicates()
        towns = sorted(df["town"].unique().tolist())
        years = sorted(df["year"].unique().tolist())
        return towns, years
    except FileNotFoundError:
        print(f"[Dashboard] エラー: {DATA_DIR / 'features_panel.csv'} が見つかりません")
        return [], []
    except Exception as e:
        print(f"[Dashboard] エラー: メタデータの読み込みに失敗しました: {e}")
        return [], []


def check_dependencies() -> bool:
    """
    依存ファイルの存在をチェック
    
    Returns:
        bool: すべての依存ファイルが存在するか
    """
    required_files = [
        ("features_panel.csv", DATA_DIR),
        ("effects_coefficients.csv", Path("../../output")), 
        ("l4_model.joblib", Path("../../models")),
        ("l4_cv_metrics.json", DATA_DIR)
    ]
    
    missing_files = []
    for file, dir_path in required_files:
        if not (dir_path / file).exists():
            missing_files.append(f"{dir_path / file}")
    
    if missing_files:
        print(f"[Dashboard] エラー: 以下のファイルが見つかりません:")
        for file in missing_files:
            print(f"  {file}")
        return False
    
    return True
