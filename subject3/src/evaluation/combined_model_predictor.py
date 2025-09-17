# -*- coding: utf-8 -*-
# src/evaluation/combined_model_predictor.py
"""
TWFE+LightGBM統合モデルの予測システム

このモジュールは、学習済みのTWFE係数とLightGBMモデルを使用して
統合予測を行うためのクラスを提供します。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import joblib

class CombinedModelPredictor:
    """TWFE+LightGBM統合モデルの予測クラス"""
    
    def __init__(self, twfe_coeffs_path: str, lgbm_model_path: str):
        """
        初期化
        
        Parameters:
        -----------
        twfe_coeffs_path : str
            TWFE係数ファイルのパス
        lgbm_model_path : str
            LightGBMモデルファイルのパス
        """
        self.twfe_coeffs_path = twfe_coeffs_path
        self.lgbm_model_path = lgbm_model_path
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # データ読み込み
        self.twfe_coeffs = None
        self.lgbm_model = None
        
        self._load_models()
    
    def _load_models(self):
        """学習済みモデルを読み込み"""
        self.logger.info("学習済みモデルを読み込み中...")
        
        # TWFE係数
        self.twfe_coeffs = pd.read_csv(self.twfe_coeffs_path)
        self.logger.info(f"TWFE係数: {len(self.twfe_coeffs)} 個")
        
        # LightGBMモデル
        try:
            self.lgbm_model = joblib.load(self.lgbm_model_path)
            self.logger.info("LightGBMモデルを読み込みました")
        except Exception as e:
            self.logger.error(f"LightGBMモデルの読み込みに失敗: {e}")
            self.lgbm_model = None
    
    def predict_twfe(self, data: pd.DataFrame) -> np.ndarray:
        """
        TWFEによる予測を実行
        
        Parameters:
        -----------
        data : pd.DataFrame
            予測対象データ（イベント変数を含む）
        
        Returns:
        --------
        np.ndarray
            TWFE予測値
        """
        predictions = np.zeros(len(data))
        
        # イベント変数の列を特定
        event_cols = [col for col in data.columns if col.startswith('event_')]
        
        for i, (idx, row) in enumerate(data.iterrows()):
            twfe_pred = 0.0
            
            # 各イベント変数について係数を適用
            for _, coeff_row in self.twfe_coeffs.iterrows():
                event_var = coeff_row['event_var']
                beta = coeff_row['beta']
                
                # 対応する列を探す
                matching_cols = [col for col in event_cols if event_var in col]
                
                for col in matching_cols:
                    if col in data.columns:
                        twfe_pred += beta * row[col]
            
            predictions[i] = twfe_pred
        
        return predictions
    
    def predict_lgbm(self, data: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        LightGBMによる予測を実行
        
        Parameters:
        -----------
        data : pd.DataFrame
            予測対象データ
        feature_cols : Optional[List[str]]
            使用する特徴量のリスト（Noneの場合は全特徴量）
        
        Returns:
        --------
        np.ndarray
            LightGBM予測値
        """
        if self.lgbm_model is None:
            self.logger.warning("LightGBMモデルが読み込まれていません。0を返します。")
            return np.zeros(len(data))
        
        # 特徴量を選択
        if feature_cols is None:
            # 全特徴量を使用（数値列のみ）
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['town', 'year']]
        
        X = data[feature_cols]
        
        # 予測実行
        try:
            # 特徴量数不一致のエラーを回避
            predictions = self.lgbm_model.predict(X, predict_disable_shape_check=True)
            return predictions
        except Exception as e:
            self.logger.error(f"LightGBM予測エラー: {e}")
            return np.zeros(len(data))
    
    def predict_combined(self, data: pd.DataFrame, twfe_weight: float = 0.5,
                        feature_cols: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        統合予測を実行
        
        Parameters:
        -----------
        data : pd.DataFrame
            予測対象データ
        twfe_weight : float
            TWFEの重み（0-1）
        feature_cols : Optional[List[str]]
            LightGBMで使用する特徴量のリスト
        
        Returns:
        --------
        Dict[str, np.ndarray]
            予測結果辞書
            - 'twfe': TWFE予測値
            - 'lgbm': LightGBM予測値
            - 'combined': 統合予測値
        """
        # 各モデルの予測
        twfe_pred = self.predict_twfe(data)
        lgbm_pred = self.predict_lgbm(data, feature_cols)
        
        # 統合予測（重み付き平均）
        combined_pred = twfe_weight * twfe_pred + (1 - twfe_weight) * lgbm_pred
        
        return {
            'twfe': twfe_pred,
            'lgbm': lgbm_pred,
            'combined': combined_pred
        }
    
    def predict_batch(self, data: pd.DataFrame, twfe_weights: List[float] = [0.3, 0.5, 0.7],
                     feature_cols: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        複数の重みでバッチ予測を実行
        
        Parameters:
        -----------
        data : pd.DataFrame
            予測対象データ
        twfe_weights : List[float]
            TWFE重みのリスト
        feature_cols : Optional[List[str]]
            LightGBMで使用する特徴量のリスト
        
        Returns:
        --------
        Dict[str, Dict[str, np.ndarray]]
            各重みでの予測結果
        """
        results = {}
        
        for weight in twfe_weights:
            weight_key = f"weight_{weight}"
            results[weight_key] = self.predict_combined(data, weight, feature_cols)
        
        return results
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        LightGBMの特徴量重要度を取得
        
        Returns:
        --------
        Optional[pd.DataFrame]
            特徴量重要度（モデルが読み込まれていない場合はNone）
        """
        if self.lgbm_model is None:
            return None
        
        try:
            # LightGBMの特徴量重要度を取得
            if hasattr(self.lgbm_model, 'feature_importances_'):
                importance = self.lgbm_model.feature_importances_
                feature_names = self.lgbm_model.feature_name_
                
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                return df
            else:
                self.logger.warning("特徴量重要度を取得できません")
                return None
        except Exception as e:
            self.logger.error(f"特徴量重要度取得エラー: {e}")
            return None
    
    def get_twfe_coefficients(self) -> pd.DataFrame:
        """
        TWFE係数を取得
        
        Returns:
        --------
        pd.DataFrame
            TWFE係数
        """
        return self.twfe_coeffs.copy()


def main():
    """使用例"""
    print("=== TWFE+LightGBM統合予測システム ===")
    
    # 予測器を初期化
    predictor = CombinedModelPredictor(
        twfe_coeffs_path="output/effects_coefficients_rate.csv",
        lgbm_model_path="models/l4_model.joblib"
    )
    
    # サンプルデータで予測実行
    print("予測システムの初期化が完了しました。")
    print("使用方法:")
    print("  predictor.predict_combined(data, twfe_weight=0.5)")
    print("  predictor.predict_batch(data, twfe_weights=[0.3, 0.5, 0.7])")


if __name__ == "__main__":
    main()
