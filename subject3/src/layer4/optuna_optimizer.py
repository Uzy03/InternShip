# -*- coding: utf-8 -*-
# src/layer4/optuna_optimizer.py
"""
Optuna最適化クラス
- 時系列CVを使用したパラメータ最適化
- 後方互換性を保つ
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import optuna
from sklearn.metrics import r2_score
import json
from pathlib import Path

from lgbm_model import LightGBMModel


class OptunaOptimizer:
    """Optunaを使用したLightGBMパラメータ最適化クラス"""
    
    def __init__(self, cv_folds: List[Tuple], n_trials: int = 50, 
                 timeout: Optional[int] = None, study_name: str = "l4_optuna"):
        """
        Args:
            cv_folds: 時系列CVのfoldリスト
            n_trials: 最適化試行回数
            timeout: タイムアウト時間（秒）
            study_name: スタディ名
        """
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.study = None
        self.best_params = None
        self.best_score = None
        
    def objective(self, trial, df: pd.DataFrame, Xcols: List[str], 
                  target: str, make_weights_func) -> float:
        """
        Optunaの目的関数
        
        Args:
            trial: Optunaのtrialオブジェクト
            df: データフレーム
            Xcols: 特徴量列名
            target: 目的変数名
            make_weights_func: サンプル重み生成関数
            
        Returns:
            CVスコア（R²）
        """
        # パラメータの提案
        params = self._suggest_params(trial)
        
        # 時系列CVで評価
        cv_scores = []
        
        for train_years, test_years in self.cv_folds:
            # データ分割
            tr = df[df["year"].isin(train_years)].copy()
            te = df[df["year"].isin(test_years)].copy()
            
            if len(tr) == 0 or len(te) == 0:
                continue
                
            # 特徴量準備
            tr_X = tr[Xcols].replace([np.inf, -np.inf], np.nan)
            te_X = te[Xcols].replace([np.inf, -np.inf], np.nan)
            
            # サンプル重み
            sw = make_weights_func(tr)
            
            # 学習用yのウィンズライズ
            y_tr = tr[target].values
            ql, qh = np.quantile(y_tr, [0.005, 0.995])
            y_tr = np.clip(y_tr, ql, qh)
            
            # モデル学習
            model = LightGBMModel(params=params)
            model.fit(
                tr_X, y_tr,
                sample_weight=sw,
                eval_set=(te_X, te[target].values),
                early_stopping_rounds=100,  # 最適化時はさらに短縮（200→100）
                log_evaluation=0  # ログを抑制
            )
            
            # 予測・評価
            pred = model.predict(te_X)
            score = r2_score(te[target].values, pred)
            cv_scores.append(score)
        
        # 平均スコアを返す
        return np.mean(cv_scores) if cv_scores else 0.0
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """パラメータを提案（高速化版）"""
        return {
            'objective': 'huber',
            'alpha': 0.9,
            'n_estimators': 10000,  # 最適化時は短縮（20000→10000）
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),  # 範囲を狭める
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),  # 範囲を狭める
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),  # 範囲を狭める
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),  # 範囲を狭める
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),  # 範囲を狭める
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # 範囲を狭める
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # 範囲を狭める
            'random_state': 42,
            'n_jobs': -1
        }
    
    def optimize(self, df: pd.DataFrame, Xcols: List[str], 
                 target: str, make_weights_func) -> Dict[str, Any]:
        """
        最適化を実行
        
        Args:
            df: データフレーム
            Xcols: 特徴量列名
            target: 目的変数名
            make_weights_func: サンプル重み生成関数
            
        Returns:
            最適パラメータ
        """
        print(f"[Optuna] 最適化開始: {self.n_trials}回の試行")
        
        # スタディを作成
        self.study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            storage=f"sqlite:///data/processed/{self.study_name}.db",
            load_if_exists=True
        )
        
        # 最適化実行（並列化を追加）
        self.study.optimize(
            lambda trial: self.objective(trial, df, Xcols, target, make_weights_func),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=2,  # CPU並列化（Colab無料環境では2-3が適切）
            show_progress_bar=True
        )
        
        # 結果を保存
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"[Optuna] 最適化完了: 最良スコア={self.best_score:.4f}")
        print(f"[Optuna] 最適パラメータ: {self.best_params}")
        
        return self.best_params
    
    def save_results(self, output_dir: str = "data/processed") -> None:
        """最適化結果を保存"""
        if self.study is None:
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 最適化結果をJSONで保存
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params
                }
                for trial in self.study.trials
            ]
        }
        
        results_file = output_path / f"{self.study_name}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"[Optuna] 結果を保存: {results_file}")
    
    def get_study_summary(self) -> Dict[str, Any]:
        """スタディの要約を取得"""
        if self.study is None:
            return {}
            
        return {
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial": self.study.best_trial.number if self.study.best_trial else None
        }


class FastOptunaOptimizer(OptunaOptimizer):
    """高速化版Optuna最適化クラス（Colab無料環境用）"""
    
    def __init__(self, cv_folds: List[Tuple], n_trials: int = 30, 
                 timeout: Optional[int] = 1800, study_name: str = "l4_fast_optuna"):
        """
        Args:
            cv_folds: 時系列CVのfoldリスト
            n_trials: 最適化試行回数（デフォルト30に削減）
            timeout: タイムアウト時間（秒、デフォルト30分）
            study_name: スタディ名
        """
        super().__init__(cv_folds, n_trials, timeout, study_name)
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """パラメータを提案（超高速化版）"""
        return {
            'objective': 'huber',
            'alpha': 0.9,
            'n_estimators': 5000,  # さらに短縮
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, log=True),  # さらに狭める
            'subsample': trial.suggest_float('subsample', 0.8, 0.9),  # さらに狭める
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.9),  # さらに狭める
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.2),  # さらに狭める
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.2),  # さらに狭める
            'num_leaves': trial.suggest_int('num_leaves', 30, 70),  # さらに狭める
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 35),  # さらに狭める
            'random_state': 42,
            'n_jobs': -1
        }
    
    def objective(self, trial, df: pd.DataFrame, Xcols: List[str], 
                  target: str, make_weights_func) -> float:
        """高速化版の目的関数（fold数を制限）"""
        params = self._suggest_params(trial)
        cv_scores = []
        
        # 最初の3foldのみで評価（高速化）
        limited_folds = self.cv_folds[:3]
        
        for train_years, test_years in limited_folds:
            tr = df[df["year"].isin(train_years)].copy()
            te = df[df["year"].isin(test_years)].copy()
            
            if len(tr) == 0 or len(te) == 0:
                continue
                
            tr_X = tr[Xcols].replace([np.inf, -np.inf], np.nan)
            te_X = te[Xcols].replace([np.inf, -np.inf], np.nan)
            sw = make_weights_func(tr)
            
            y_tr = tr[target].values
            ql, qh = np.quantile(y_tr, [0.005, 0.995])
            y_tr = np.clip(y_tr, ql, qh)
            
            model = LightGBMModel(params=params)
            model.fit(
                tr_X, y_tr,
                sample_weight=sw,
                eval_set=(te_X, te[target].values),
                early_stopping_rounds=50,  # さらに短縮
                log_evaluation=0
            )
            
            pred = model.predict(te_X)
            score = r2_score(te[target].values, pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores) if cv_scores else 0.0
