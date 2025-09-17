# -*- coding: utf-8 -*-
# src/layer4/lgbm_model.py
"""
LightGBMモデルのクラス定義
- 学習・予測・パラメータ管理を統合
- 後方互換性を保つ
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path

# LightGBM が無ければ HistGradientBoosting にフォールバック
try:
    import lightgbm as lgb
    USE_LGBM = True
except Exception:
    USE_LGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor


class LightGBMModel:
    """LightGBMモデルのラッパークラス"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params: LightGBMのパラメータ辞書
        """
        self.params = params or self.get_default_params()
        self.model = None
        self.best_iteration_ = None
        self.feature_importance_ = None
        
    def get_default_params(self) -> Dict[str, Any]:
        """デフォルトパラメータを取得"""
        if USE_LGBM:
            return {
                'objective': 'huber',
                'alpha': 0.9,
                'n_estimators': 25000,
                'learning_rate': 0.008,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.15,
                'reg_lambda': 0.4,
                'num_leaves': 50,
                'min_child_samples': 25,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            return {
                'max_depth': None,
                'learning_rate': 0.05,
                'max_leaf_nodes': 63,
                'l2_regularization': 0.0,
                'random_state': 42
            }
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple] = None,
            early_stopping_rounds: int = 800,
            log_evaluation: int = 200,
            verbose: int = -1) -> 'LightGBMModel':
        """
        モデルを学習
        
        Args:
            X: 特徴量
            y: 目的変数
            sample_weight: サンプル重み
            eval_set: 評価用データセット
            early_stopping_rounds: 早期停止ラウンド数
            log_evaluation: ログ出力間隔
        """
        if USE_LGBM:
            # 警告を抑制するためにverboseを設定
            params_with_verbose = self.params.copy()
            params_with_verbose['verbose'] = verbose
            
            self.model = lgb.LGBMRegressor(**params_with_verbose)
            
            callbacks = []
            if early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
            if log_evaluation > 0:
                callbacks.append(lgb.log_evaluation(log_evaluation))
            
            self.model.fit(
                X, y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                callbacks=callbacks
            )
            self.best_iteration_ = self.model.best_iteration_
            
            # 特徴量重要度を取得
            if hasattr(self.model, 'booster_'):
                self.feature_importance_ = self.model.booster_.feature_importance(importance_type="gain")
        else:
            self.model = HistGradientBoostingRegressor(**self.params)
            self.model.fit(X, y)
            self.best_iteration_ = None
            self.feature_importance_ = None
            
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行
        
        Args:
            X: 特徴量
            
        Returns:
            予測値
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")
        
        if USE_LGBM and self.best_iteration_ is not None:
            return self.model.predict(X, num_iteration=self.best_iteration_)
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        特徴量重要度を取得
        
        Args:
            feature_names: 特徴量名のリスト
            
        Returns:
            特徴量重要度のDataFrame
        """
        if self.feature_importance_ is None:
            return pd.DataFrame({"feature": feature_names, "importance": 0})
        
        return pd.DataFrame({
            "feature": feature_names,
            "importance": self.feature_importance_
        }).sort_values("importance", ascending=False)
    
    def save(self, filepath: str) -> None:
        """
        モデルを保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LightGBMModel':
        """
        モデルを読み込み
        
        Args:
            filepath: ファイルパス
            
        Returns:
            読み込まれたモデル
        """
        return joblib.load(filepath)
    
    def get_lgbm_model(self):
        """
        内部のLightGBMモデルを取得（後方互換性用）
        
        Returns:
            LightGBMモデルまたはNone
        """
        return self.model if USE_LGBM else None
