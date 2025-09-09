# -*- coding: utf-8 -*-
"""
全町丁一括実行CLI
指定されたシナリオを全町丁に適用して予測を実行し、結果をCSVとJSONで出力
"""
import pandas as pd
import json
import argparse
from pathlib import Path
import sys
import os
from typing import Dict, List, Any
import logging

# パス設定
sys.path.append(os.path.dirname(__file__))

# Layer5モジュールのインポート
from scenario_to_events import scenario_to_events
from prepare_baseline import prepare_baseline
from build_future_features import build_future_features
from forecast_service import run_scenario

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_town_list(baseline_path: str) -> List[str]:
    """町丁一覧を読み込み"""
    try:
        baseline_df = pd.read_csv(baseline_path)
        towns = sorted(baseline_df["town"].unique().tolist())
        logger.info(f"[ALL] 町丁数: {len(towns)}")
        return towns
    except Exception as e:
        logger.error(f"[ALL] 町丁一覧の読み込みに失敗: {e}")
        raise

def load_scenario_template(template_path: str) -> Dict[str, Any]:
    """シナリオテンプレートを読み込み"""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        logger.info(f"[ALL] シナリオテンプレートを読み込み: {template_path}")
        return scenario
    except Exception as e:
        logger.error(f"[ALL] シナリオテンプレートの読み込みに失敗: {e}")
        raise

def create_town_scenario(template: Dict[str, Any], town: str) -> Dict[str, Any]:
    """町丁用のシナリオを作成"""
    scenario = template.copy()
    scenario["town"] = town
    return scenario

def run_single_town_scenario(town: str, scenario: Dict[str, Any], output_dir: Path, 
                            i: int, total: int) -> List[Dict[str, Any]]:
    """単一町丁のシナリオを実行"""
    logger.info(f"[ALL] ({i}/{total}) town={town}")
    
    try:
        # 町丁用シナリオを作成
        town_scenario = create_town_scenario(scenario, town)
        
        # 出力ディレクトリの設定
        town_output_dir = output_dir / "towns" / town.replace(" ", "_")
        town_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: 将来イベント行列の生成
        future_events = scenario_to_events(town_scenario)
        future_events.to_csv(town_output_dir / "l5_future_events.csv", index=False)
        
        # Step 2: 基準年データの準備
        baseline = prepare_baseline(town, town_scenario["base_year"])
        baseline.to_csv(town_output_dir / "l5_baseline.csv", index=False)
        
        # Step 3: 将来特徴の構築
        future_features = build_future_features(baseline, future_events, town_scenario)
        future_features.to_csv(town_output_dir / "l5_future_features.csv", index=False)
        
        # ベース人口を取得
        base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
        if pd.isna(base_population):
            base_population = 0.0
        
        # シナリオにベース人口を設定
        town_scenario["base_population"] = base_population
        
        # Step 4: 人口予測の実行
        result = run_scenario(town_scenario, str(town_output_dir))
        
        # 結果を行データに変換
        rows = []
        for year_result in result["results"]:
            row = {
                "town": result["town"],
                "baseline_year": result["baseline_year"],
                "year": year_result["year"],
                "h": year_result["year"] - result["baseline_year"],
                "delta": year_result["delta"],
                "pop": year_result["pop"],
                "exp": year_result["contrib"]["exp"],
                "macro": year_result["contrib"]["macro"],
                "inertia": year_result["contrib"]["inertia"],
                "other": year_result["contrib"]["other"],
                "pi_delta_low": year_result["pi"]["delta_low"],
                "pi_delta_high": year_result["pi"]["delta_high"],
                "pi_pop_low": year_result["pi"]["pop_low"],
                "pi_pop_high": year_result["pi"]["pop_high"]
            }
            rows.append(row)
        
        return rows
        
    except Exception as e:
        logger.error(f"[ALL] 町丁 {town} の処理に失敗: {e}")
        # エラー時は空の結果を返す
        return []

def load_centroids(centroids_path: str) -> pd.DataFrame:
    """重心データを読み込み（オプション）"""
    try:
        if Path(centroids_path).exists():
            centroids_df = pd.read_csv(centroids_path)
            logger.info(f"[ALL] 重心データを読み込み: {len(centroids_df)}件")
            return centroids_df
        else:
            logger.info(f"[ALL] 重心データが見つかりません: {centroids_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"[ALL] 重心データの読み込みに失敗: {e}")
        return pd.DataFrame()

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="全町丁一括実行CLI")
    parser.add_argument("--scenario", required=True, help="シナリオテンプレートJSONファイル")
    parser.add_argument("--baseline", default="../../data/processed/l5_baseline.csv", 
                       help="ベースラインデータCSVファイル")
    parser.add_argument("--centroids", default="../../data/interim/centroids.csv",
                       help="重心データCSVファイル（オプション）")
    parser.add_argument("--output-dir", default="../../output", help="出力ディレクトリ")
    parser.add_argument("--max-towns", type=int, help="処理する最大町丁数（テスト用）")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 町丁一覧の読み込み
    towns = load_town_list(args.baseline)
    
    # 最大町丁数の制限（テスト用）
    if args.max_towns:
        towns = towns[:args.max_towns]
        logger.info(f"[ALL] テストモード: 最大町丁数を{args.max_towns}に制限")
    
    # シナリオテンプレートの読み込み
    scenario_template = load_scenario_template(args.scenario)
    
    # 重心データの読み込み（オプション）
    centroids_df = load_centroids(args.centroids)
    
    # 全町丁の結果を格納するリスト
    all_rows = []
    
    # 各町丁を順次処理
    for i, town in enumerate(towns, 1):
        rows = run_single_town_scenario(town, scenario_template, output_dir, i, len(towns))
        all_rows.extend(rows)
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(all_rows)
    
    if results_df.empty:
        logger.error("[ALL] 結果が空です")
        return
    
    # 重心データとの結合（オプション）
    if not centroids_df.empty and "town" in centroids_df.columns:
        results_df = results_df.merge(centroids_df[["town", "lat", "lon"]], on="town", how="left")
        logger.info("[ALL] 重心データを結合しました")
    
    # CSV出力
    csv_path = output_dir / "forecast_all_rows.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"[ALL] CSV出力完了: {csv_path} ({len(results_df)}行)")
    
    # JSON出力（検証用）
    json_path = output_dir / "forecast_all.json"
    json_data = {
        "metadata": {
            "total_towns": len(towns),
            "total_rows": len(results_df),
            "scenario_template": scenario_template,
            "output_timestamp": pd.Timestamp.now().isoformat()
        },
        "data": results_df.to_dict('records')
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"[ALL] JSON出力完了: {json_path}")
    
    # サマリー統計
    logger.info(f"[ALL] 処理完了:")
    logger.info(f"  - 処理町丁数: {len(towns)}")
    logger.info(f"  - 出力行数: {len(results_df)}")
    logger.info(f"  - 年数: {results_df['year'].nunique()}")
    logger.info(f"  - 平均Δ人口: {results_df['delta'].mean():.2f}")
    logger.info(f"  - 最大Δ人口: {results_df['delta'].max():.2f}")
    logger.info(f"  - 最小Δ人口: {results_df['delta'].min():.2f}")
    
    # Δの整合性チェック
    delta_consistency = results_df.apply(
        lambda row: abs(row['delta'] - (row['exp'] + row['macro'] + row['inertia'] + row['other'])), 
        axis=1
    )
    max_inconsistency = delta_consistency.max()
    if max_inconsistency > 1e-6:
        logger.warning(f"[ALL] Δの整合性チェック警告: 最大誤差={max_inconsistency:.2e}")
    else:
        logger.info(f"[ALL] Δの整合性チェックOK: 最大誤差={max_inconsistency:.2e}")

if __name__ == "__main__":
    main()
