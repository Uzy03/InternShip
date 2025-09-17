# -*- coding: utf-8 -*-
# src/evaluation/combined_model_evaluation.py
"""
TWFE+LightGBM統合モデルの評価システム

このモジュールは、layer3のTWFEとlayer4のLightGBMを組み合わせた
最終的なモデルの性能を評価します。

評価方法はlayer4と同じ時系列クロスバリデーションを使用し、
TWFEの係数とLightGBMの予測を組み合わせて統合予測を行います。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.feature_gate import select_feature_columns

# ファイルパス
P_FEAT = "data/processed/features_l4.csv"
P_TWFE_COEFFS = "output/effects_coefficients_rate.csv"
P_LGBM_MODEL = "models/l4_model.joblib"
P_LGBM_METRICS = "data/processed/l4_cv_metrics.json"
P_LGBM_PREDICTIONS = "data/processed/l4_predictions.csv"
P_OUTPUT = "src/evaluation/combined_model_evaluation.json"

TARGET = "delta_people"
ID_KEYS = ["town", "year"]

class CombinedModelEvaluator:
    """TWFE+LightGBM統合モデルの評価クラス"""
    
    def __init__(self, features_path: str = P_FEAT, 
                 twfe_coeffs_path: str = P_TWFE_COEFFS,
                 lgbm_model_path: str = P_LGBM_MODEL,
                 lgbm_metrics_path: str = P_LGBM_METRICS,
                 lgbm_predictions_path: str = P_LGBM_PREDICTIONS):
        """
        初期化
        
        Parameters:
        -----------
        features_path : str
            特徴量データのパス
        twfe_coeffs_path : str
            TWFE係数のパス
        lgbm_model_path : str
            LightGBMモデルのパス
        lgbm_metrics_path : str
            LightGBMメトリクスのパス
        """
        self.features_path = features_path
        self.twfe_coeffs_path = twfe_coeffs_path
        self.lgbm_model_path = lgbm_model_path
        self.lgbm_metrics_path = lgbm_metrics_path
        self.lgbm_predictions_path = lgbm_predictions_path
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # データ読み込み
        self.df = None
        self.twfe_coeffs = None
        self.lgbm_model = None
        self.lgbm_metrics = None
        self.lgbm_predictions = None
        
        self._load_data()
    
    def _load_data(self):
        """必要なデータを読み込み"""
        self.logger.info("データを読み込み中...")
        
        # 特徴量データ
        self.df = pd.read_csv(self.features_path).sort_values(ID_KEYS)
        self.logger.info(f"特徴量データ: {len(self.df)} 行")
        
        # TWFE係数
        self.twfe_coeffs = pd.read_csv(self.twfe_coeffs_path)
        self.logger.info(f"TWFE係数: {len(self.twfe_coeffs)} 個")
        
        # LightGBMモデル
        try:
            import joblib
            self.lgbm_model = joblib.load(self.lgbm_model_path)
            self.logger.info("LightGBMモデルを読み込みました")
        except Exception as e:
            self.logger.error(f"LightGBMモデルの読み込みに失敗: {e}")
            self.lgbm_model = None
        
        # LightGBMメトリクス
        with open(self.lgbm_metrics_path, 'r', encoding='utf-8') as f:
            self.lgbm_metrics = json.load(f)
        self.logger.info("LightGBMメトリクスを読み込みました")
        
        # LightGBM予測結果
        self.lgbm_predictions = pd.read_csv(self.lgbm_predictions_path)
        self.logger.info(f"LightGBM予測結果: {len(self.lgbm_predictions)} 行")
        
        # 目的変数の整備
        if TARGET not in self.df.columns:
            if "pop_total" not in self.df.columns:
                raise ValueError("features_l4.csv に delta_people も pop_total もありません。")
            self.df[TARGET] = self.df.groupby("town")["pop_total"].diff()
        
        # NaN/∞ を除去
        self.df[TARGET] = pd.to_numeric(self.df[TARGET], errors="coerce")
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df[~self.df[TARGET].isna()].copy()
        
        self.logger.info(f"データ準備完了: {len(self.df)} 行")
    
    def time_series_folds(self, years: List[int], min_train_years: int = 20, 
                         test_window: int = 1, last_n_tests: Optional[int] = None) -> List[Tuple[set, set]]:
        """
        時系列クロスバリデーションのフォールドを作成
        
        Parameters:
        -----------
        years : List[int]
            年リスト
        min_train_years : int
            最小訓練年数
        test_window : int
            テストウィンドウサイズ
        last_n_tests : Optional[int]
            最後のN回のテストのみ実行
        
        Returns:
        --------
        List[Tuple[set, set]]
            フォールドのリスト（訓練年, テスト年）
        """
        ys = sorted(years)
        folds = []
        
        # 全期間の年リストを取得（1998年から）
        all_years = sorted(self.df["year"].unique().tolist())
        
        for i, test_year in enumerate(ys):
            # テスト年より前の年を訓練年とする
            train_years = [y for y in all_years if y < test_year]
            
            # 最小訓練年数の条件をチェック
            if len(train_years) >= min_train_years:
                folds.append((set(train_years), {test_year}))
                
                if last_n_tests is not None and len(folds) >= last_n_tests:
                    break
        
        return folds
    
    def calculate_twfe_prediction(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        TWFEによる予測を計算
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            テストデータ
        
        Returns:
        --------
        np.ndarray
            TWFE予測値
        """
        predictions = np.zeros(len(test_data))
        
        # イベント変数の列を特定
        event_cols = [col for col in test_data.columns if col.startswith('event_')]
        
        for i, (idx, row) in enumerate(test_data.iterrows()):
            twfe_pred = 0.0
            
            # 各イベント変数について係数を適用
            for _, coeff_row in self.twfe_coeffs.iterrows():
                event_var = coeff_row['event_var']
                beta = coeff_row['beta']
                
                # 対応する列を探す
                matching_cols = [col for col in event_cols if event_var in col]
                
                for col in matching_cols:
                    if col in test_data.columns:
                        twfe_pred += beta * row[col]
            
            predictions[i] = twfe_pred
        
        return predictions
    
    def calculate_lgbm_prediction(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        LightGBMによる予測を計算（layer4の結果を使用）
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            テストデータ
        
        Returns:
        --------
        np.ndarray
            LightGBM予測値
        """
        # layer4の予測結果から対応する予測値を取得
        predictions = np.zeros(len(test_data))
        
        for i, (idx, row) in enumerate(test_data.iterrows()):
            town = row['town']
            year = row['year']
            
            # layer4の予測結果から対応する予測値を取得
            lgbm_pred = self.lgbm_predictions[
                (self.lgbm_predictions['town'] == town) & 
                (self.lgbm_predictions['year'] == year)
            ]
            
            if len(lgbm_pred) > 0:
                predictions[i] = lgbm_pred['y_pred'].iloc[0]
            else:
                # 該当する予測値がない場合は0とする
                predictions[i] = 0.0
        
        return predictions
    
    def calculate_combined_prediction(self, test_data: pd.DataFrame, 
                                    twfe_weight: float = 0.5) -> np.ndarray:
        """
        TWFE+LightGBMの統合予測を計算
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            テストデータ
        twfe_weight : float
            TWFEの重み（0-1）
        
        Returns:
        --------
        np.ndarray
            統合予測値
        """
        twfe_pred = self.calculate_twfe_prediction(test_data)
        lgbm_pred = self.calculate_lgbm_prediction(test_data)
        
        # 重み付き平均
        combined_pred = twfe_weight * twfe_pred + (1 - twfe_weight) * lgbm_pred
        
        return combined_pred
    
    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """評価指標を計算"""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, np.abs(y_true)))))
        r2 = float(r2_score(y_true, y_pred))
        return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)
    
    def evaluate_combined_model(self, twfe_weight: float = 0.5) -> Dict:
        """
        統合モデルを評価
        
        Parameters:
        -----------
        twfe_weight : float
            TWFEの重み（0-1）
        
        Returns:
        --------
        Dict
            評価結果
        """
        self.logger.info(f"統合モデル評価開始 (TWFE重み: {twfe_weight})")
        
        # 年データの準備
        years = sorted(self.df["year"].unique().tolist())
        self.logger.info(f"年範囲: {years[0]}..{years[-1]} (#years={len(years)})")
        
        if len(years) < 21:
            raise RuntimeError("年数が20未満のため、要件（20年以上）を満たせません。")
        
        # クロスバリデーションの設定（layer4と同じ：2019-2025年）
        # 2018年を除外して、2019年から開始
        years_filtered = [y for y in years if y >= 2019]
        self.logger.info(f"フィルタ後の年範囲: {years_filtered[0]}..{years_filtered[-1]} (#years={len(years_filtered)})")
        
        folds = self.time_series_folds(years_filtered, min_train_years=20, test_window=1, last_n_tests=None)
        self.logger.info(f"#folds={len(folds)}")
        
        all_preds = []
        fold_metrics = []
        
        for fi, (train_years, test_years) in enumerate(folds, 1):
            self.logger.info(f"フォールド {fi}/{len(folds)} 実行中...")
            
            # テストデータを取得
            test_data = self.df[self.df["year"].isin(test_years)].copy()
            
            # 統合予測を計算
            predictions = self.calculate_combined_prediction(test_data, twfe_weight)
            
            test_data['y_pred'] = predictions
            all_preds.append(test_data[ID_KEYS + [TARGET] + ['y_pred']].copy())
            
            # メトリクス計算
            m = self.metrics(test_data[TARGET].values, predictions)
            m["fold"] = fi
            m["train_years"] = {
                "len": len(train_years),
                "first": int(min(train_years)),
                "last": int(max(train_years))
            }
            m["test_years"] = sorted(list(test_years))
            fold_metrics.append(m)
        
        # 全予測結果を結合
        preds = pd.concat(all_preds, axis=0, ignore_index=True)
        
        # 集計メトリクス
        agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        
        return {
            "folds": fold_metrics,
            "aggregate": agg,
            "predictions": preds,
            "twfe_weight": twfe_weight
        }
    
    def compare_models(self) -> Dict:
        """
        各モデル（TWFE単体、LightGBM単体、統合）を比較
        
        Returns:
        --------
        Dict
            比較結果
        """
        self.logger.info("モデル比較を開始...")
        
        # 年データの準備（layer4と同じ：2019-2025年）
        years = sorted(self.df["year"].unique().tolist())
        years_filtered = [y for y in years if y >= 2019]
        self.logger.info(f"フィルタ後の年範囲: {years_filtered[0]}..{years_filtered[-1]} (#years={len(years_filtered)})")
        folds = self.time_series_folds(years_filtered, min_train_years=20, test_window=1, last_n_tests=None)
        self.logger.info(f"#folds={len(folds)}")
        
        if len(folds) == 0:
            raise RuntimeError("フォールドが生成されませんでした。年データを確認してください。")
        
        results = {}
        
        # 1. TWFE単体
        self.logger.info("TWFE単体モデルを評価中...")
        twfe_results = self._evaluate_single_model("twfe", folds)
        results["TWFE_only"] = twfe_results
        
        # 2. LightGBM単体（layer4の結果を使用）
        self.logger.info("LightGBM単体モデルを評価中...")
        # layer4の結果を直接使用
        lgbm_results = {
            "folds": self.lgbm_metrics["folds"],
            "aggregate": self.lgbm_metrics["aggregate"],
            "predictions": []  # 予測結果は空にする
        }
        results["LightGBM_only"] = lgbm_results
        
        # 3. 統合モデル（重み0.5）
        self.logger.info("統合モデル（重み0.5）を評価中...")
        combined_results = self.evaluate_combined_model(twfe_weight=0.5)
        results["Combined_0.5"] = combined_results
        
        # 4. 統合モデル（重み0.3）
        self.logger.info("統合モデル（重み0.3）を評価中...")
        combined_results_03 = self.evaluate_combined_model(twfe_weight=0.3)
        results["Combined_0.3"] = combined_results_03
        
        # 5. 統合モデル（重み0.7）
        self.logger.info("統合モデル（重み0.7）を評価中...")
        combined_results_07 = self.evaluate_combined_model(twfe_weight=0.7)
        results["Combined_0.7"] = combined_results_07
        
        return results
    
    def _evaluate_single_model(self, model_type: str, folds: List[Tuple[set, set]]) -> Dict:
        """
        単体モデルを評価
        
        Parameters:
        -----------
        model_type : str
            モデルタイプ ("twfe" or "lgbm")
        folds : List[Tuple[set, set]]
            フォールドのリスト
        
        Returns:
        --------
        Dict
            評価結果
        """
        all_preds = []
        fold_metrics = []
        
        self.logger.info(f"フォールド数: {len(folds)}")
        
        for fi, (train_years, test_years) in enumerate(folds, 1):
            self.logger.info(f"フォールド {fi}/{len(folds)} 実行中...")
            
            # テストデータを取得
            test_data = self.df[self.df["year"].isin(test_years)].copy()
            self.logger.info(f"テストデータ: {len(test_data)} 行, 年: {sorted(list(test_years))}")
            
            if len(test_data) == 0:
                self.logger.warning(f"フォールド {fi}: テストデータが空です。スキップします。")
                continue
            
            # 予測を計算
            if model_type == "twfe":
                predictions = self.calculate_twfe_prediction(test_data)
            elif model_type == "lgbm":
                predictions = self.calculate_lgbm_prediction(test_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            test_data['y_pred'] = predictions
            all_preds.append(test_data[ID_KEYS + [TARGET] + ['y_pred']].copy())
            
            # メトリクス計算
            m = self.metrics(test_data[TARGET].values, predictions)
            m["fold"] = fi
            m["train_years"] = {
                "len": len(train_years),
                "first": int(min(train_years)),
                "last": int(max(train_years))
            }
            m["test_years"] = sorted(list(test_years))
            fold_metrics.append(m)
            
            self.logger.info(f"フォールド {fi} 完了: MAE={m['MAE']:.2f}, R²={m['R2']:.2f}")
        
        # 全予測結果を結合
        preds = pd.concat(all_preds, axis=0, ignore_index=True)
        
        # 集計メトリクス
        agg = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        
        return {
            "folds": fold_metrics,
            "aggregate": agg,
            "predictions": preds
        }
    
    def save_results(self, results: Dict, output_path: str = P_OUTPUT):
        """結果を保存（各モデル別に分けて保存）"""
        self.logger.info(f"結果を保存中: {output_path}")
        
        # ディレクトリ作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrameを辞書に変換
        serializable_results = self._make_serializable(results)
        
        # 各モデル別に保存
        for model_name, result in serializable_results.items():
            model_output_path = output_dir / f"{model_name}_evaluation.json"
            
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"  {model_name}: {model_output_path}")
        
        # 全体の結果も保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info("保存完了")
    
    def _make_serializable(self, obj):
        """オブジェクトをJSONシリアライズ可能な形式に変換"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def print_comparison_summary(self, results: Dict):
        """比較結果のサマリーを表示"""
        print("\n" + "="*80)
        print("TWFE+LightGBM統合モデル評価結果")
        print("="*80)
        
        for model_name, result in results.items():
            print(f"\n[{model_name}]")
            agg = result["aggregate"]
            print(f"  R²:  {agg['R2']:.4f}")
            print(f"  MAE: {agg['MAE']:.2f}")
            print(f"  RMSE: {agg['RMSE']:.2f}")
            print(f"  MAPE: {agg['MAPE']:.4f}")
        
        # 最良のモデルを特定
        best_model = min(results.items(), key=lambda x: x[1]["aggregate"]["MAE"])
        print(f"\n最良のモデル（MAE基準）: {best_model[0]} (MAE: {best_model[1]['aggregate']['MAE']:.2f})")


def main():
    """メイン実行関数"""
    print("=== TWFE+LightGBM統合モデル評価を開始 ===")
    
    # 評価器を初期化
    evaluator = CombinedModelEvaluator()
    
    # モデル比較を実行
    results = evaluator.compare_models()
    
    # 結果を保存
    evaluator.save_results(results)
    
    # サマリーを表示
    evaluator.print_comparison_summary(results)
    
    print(f"\n結果を保存しました: {P_OUTPUT}")


if __name__ == "__main__":
    main()
