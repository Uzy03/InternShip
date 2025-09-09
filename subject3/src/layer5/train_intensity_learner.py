# -*- coding: utf-8 -*-
# src/layer5/train_intensity_learner.py
"""
強度学習システム: イベント強度とlag効果を機械学習で最適化
- 入力: 過去のイベントデータと実際の人口変化
- 出力: 最適化された強度パラメータ
- 用途: シナリオ生成時の自動強度設定

設計:
- イベントタイプ別の強度学習
- 地域特性を考慮した強度調整
- 時系列での強度パターン学習
- 予測精度向上のための強度最適化
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys
import os

# パス設定
P_FEATURES_PANEL = "../../data/processed/features_panel.csv"
P_FUTURE_EVENTS = "../../data/processed/l5_future_events.csv"
P_FUTURE_FEATURES = "../../data/processed/l5_future_features.csv"
P_OUTPUT_MODEL = "../../models/intensity_learner.joblib"
P_OUTPUT_PARAMS = "../../output/learned_intensity_params.json"

class IntensityLearner:
    """強度学習システム"""
    
    def __init__(self):
        self.intensity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lag_t_model = Ridge(alpha=1.0)
        self.lag_t1_model = Ridge(alpha=1.0)
        self.is_fitted = False
        
    def prepare_training_data(self, features_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """学習データの準備"""
        print("[強度学習] 学習データを準備中...")
        
        # イベント効果の特徴量を抽出
        event_cols = [col for col in features_panel.columns if col.startswith('exp_') and col.endswith('_h1')]
        
        # 基本特徴量
        base_features = ['town', 'year', 'pop_total', 'male', 'female']
        available_base = [col for col in base_features if col in features_panel.columns]
        
        # 学習用データフレーム
        train_data = features_panel[available_base + event_cols].copy()
        
        # イベントタイプ別の特徴量を作成
        event_types = ['housing', 'commercial', 'employment', 'transit', 'disaster', 'policy_boundary', 'public_edu_medical']
        for event_type in event_types:
            inc_col = f'exp_{event_type}_inc_h1'
            dec_col = f'exp_{event_type}_dec_h1'
            
            if inc_col in train_data.columns:
                train_data[f'{event_type}_event'] = (train_data[inc_col] != 0).astype(int)
                train_data[f'{event_type}_intensity'] = train_data[inc_col].abs()
            else:
                train_data[f'{event_type}_event'] = 0
                train_data[f'{event_type}_intensity'] = 0.0
        
        # 地域特性特徴量
        if 'pop_total' in train_data.columns:
            train_data['log_pop'] = np.log1p(train_data['pop_total'])
            train_data['pop_density'] = train_data['pop_total'] / 1000  # 簡易的な密度
        
        # 年次特徴量
        train_data['year_normalized'] = (train_data['year'] - train_data['year'].min()) / (train_data['year'].max() - train_data['year'].min())
        
        # ターゲット変数（人口変化）
        train_data['delta_people'] = train_data.groupby('town')['pop_total'].diff()
        
        # 欠損値を除去
        train_data = train_data.dropna()
        
        print(f"[強度学習] 学習データ準備完了: {len(train_data)}行")
        return train_data
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """特徴量抽出"""
        feature_cols = []
        
        # イベントタイプ特徴量
        event_types = ['housing', 'commercial', 'employment', 'transit', 'disaster', 'policy_boundary', 'public_edu_medical']
        for event_type in event_types:
            if f'{event_type}_event' in df.columns:
                feature_cols.append(f'{event_type}_event')
            if f'{event_type}_intensity' in df.columns:
                feature_cols.append(f'{event_type}_intensity')
        
        # 地域特性特徴量
        for col in ['log_pop', 'pop_density', 'year_normalized']:
            if col in df.columns:
                feature_cols.append(col)
        
        # 特徴量行列を作成
        X = df[feature_cols].values
        return X, feature_cols
    
    def fit(self, train_data: pd.DataFrame):
        """強度学習モデルの学習"""
        print("[強度学習] モデル学習を開始...")
        
        # 特徴量抽出
        X, feature_names = self.extract_features(train_data)
        
        # 各イベントタイプごとに強度を学習
        event_types = ['housing', 'commercial', 'employment', 'transit', 'disaster', 'policy_boundary', 'public_edu_medical']
        
        # 学習データを準備
        X_list = []
        y_intensity_list = []
        y_lag_t_list = []
        y_lag_t1_list = []
        
        for event_type in event_types:
            event_mask = train_data[f'{event_type}_event'] == 1
            if event_mask.sum() < 10:  # データが少なすぎる場合はスキップ
                continue
                
            event_data = train_data[event_mask]
            event_X = X[event_mask]
            
            # 実際の強度を推定（人口変化から逆算）
            actual_intensity = event_data['delta_people'].abs() / event_data['pop_total'].clip(lower=1)
            actual_intensity = np.clip(actual_intensity, 0, 1)  # 0-1の範囲にクリップ
            
            # lag効果を推定（簡易的に強度の半分程度）
            actual_lag_t = actual_intensity * 0.8
            actual_lag_t1 = actual_intensity * 0.4
            
            X_list.append(event_X)
            y_intensity_list.extend(actual_intensity)
            y_lag_t_list.extend(actual_lag_t)
            y_lag_t1_list.extend(actual_lag_t1)
        
        if len(X_list) == 0:
            print("[強度学習] 学習データが不足しています。デフォルト値を設定します。")
            self._set_default_params()
            return
        
        # 全データを結合
        X_all = np.vstack(X_list)
        y_intensity = np.array(y_intensity_list)
        y_lag_t = np.array(y_lag_t_list)
        y_lag_t1 = np.array(y_lag_t1_list)
        
        # モデル学習
        self.intensity_model.fit(X_all, y_intensity)
        self.lag_t_model.fit(X_all, y_lag_t)
        self.lag_t1_model.fit(X_all, y_lag_t1)
        
        self.is_fitted = True
        print("[強度学習] モデル学習完了")
        
        # 学習結果を評価
        self._evaluate_models(X_all, y_intensity, y_lag_t, y_lag_t1)
    
    def _calculate_year_factor(self, year_offset: int, event_type: str) -> float:
        """年次減衰効果を計算"""
        # イベントタイプ別の減衰パターン
        decay_patterns = {
            'housing': [1.0, 0.3, 0.1],      # 1年目: 100%, 2年目: 30%, 3年目: 10%
            'commercial': [1.0, 0.5, 0.2],   # 1年目: 100%, 2年目: 50%, 3年目: 20%
            'employment': [1.0, 0.7, 0.4],   # 1年目: 100%, 2年目: 70%, 3年目: 40%
            'transit': [1.0, 0.8, 0.6],      # 1年目: 100%, 2年目: 80%, 3年目: 60%
            'disaster': [1.0, 0.2, 0.05],    # 1年目: 100%, 2年目: 20%, 3年目: 5%
            'policy_boundary': [1.0, 0.9, 0.8], # 1年目: 100%, 2年目: 90%, 3年目: 80%
            'public_edu_medical': [1.0, 0.6, 0.3] # 1年目: 100%, 2年目: 60%, 3年目: 30%
        }
        
        pattern = decay_patterns.get(event_type, [1.0, 0.5, 0.2])
        year_idx = min(year_offset, len(pattern) - 1)
        return pattern[year_idx]
    
    def _evaluate_models(self, X: np.ndarray, y_intensity: np.ndarray, y_lag_t: np.ndarray, y_lag_t1: np.ndarray):
        """モデル性能の評価"""
        print("[強度学習] モデル性能評価:")
        
        # 強度モデル
        y_pred_intensity = self.intensity_model.predict(X)
        mae_intensity = mean_absolute_error(y_intensity, y_pred_intensity)
        r2_intensity = r2_score(y_intensity, y_pred_intensity)
        print(f"  強度モデル - MAE: {mae_intensity:.4f}, R²: {r2_intensity:.4f}")
        
        # lag_tモデル
        y_pred_lag_t = self.lag_t_model.predict(X)
        mae_lag_t = mean_absolute_error(y_lag_t, y_pred_lag_t)
        r2_lag_t = r2_score(y_lag_t, y_pred_lag_t)
        print(f"  lag_tモデル - MAE: {mae_lag_t:.4f}, R²: {r2_lag_t:.4f}")
        
        # lag_t1モデル
        y_pred_lag_t1 = self.lag_t1_model.predict(X)
        mae_lag_t1 = mean_absolute_error(y_lag_t1, y_pred_lag_t1)
        r2_lag_t1 = r2_score(y_lag_t1, y_pred_lag_t1)
        print(f"  lag_t1モデル - MAE: {mae_lag_t1:.4f}, R²: {r2_lag_t1:.4f}")
    
    def _set_default_params(self):
        """デフォルトパラメータの設定"""
        self.default_params = {
            'housing': {'intensity': 0.8, 'lag_t': 0.7, 'lag_t1': 0.3},
            'commercial': {'intensity': 0.9, 'lag_t': 0.8, 'lag_t1': 0.4},
            'employment': {'intensity': 0.85, 'lag_t': 0.6, 'lag_t1': 0.2},
            'transit': {'intensity': 0.7, 'lag_t': 0.5, 'lag_t1': 0.1},
            'disaster': {'intensity': 0.6, 'lag_t': 0.4, 'lag_t1': 0.1},
            'policy_boundary': {'intensity': 0.9, 'lag_t': 0.8, 'lag_t1': 0.5},
            'public_edu_medical': {'intensity': 0.75, 'lag_t': 0.6, 'lag_t1': 0.3}
        }
        self.is_fitted = False
    
    def predict_intensity(self, event_type: str, town_features: Dict[str, float], year_offset: int = 0) -> Dict[str, float]:
        """最適な強度パラメータを予測（年次別対応）"""
        if not self.is_fitted:
            if hasattr(self, 'default_params'):
                base_params = self.default_params.get(event_type, {'intensity': 1.0, 'lag_t': 1.0, 'lag_t1': 1.0})
            else:
                base_params = {'intensity': 1.0, 'lag_t': 1.0, 'lag_t1': 1.0}
            
            # 年次減衰効果を適用
            year_factor = self._calculate_year_factor(year_offset, event_type)
            return {
                'intensity': base_params['intensity'] * year_factor,
                'lag_t': base_params['lag_t'] * year_factor,
                'lag_t1': base_params['lag_t1'] * year_factor
            }
        
        # 特徴量を準備
        feature_vector = np.array([[
            town_features.get('log_pop', 0),
            town_features.get('pop_density', 0),
            town_features.get('year_normalized', 0.5)
        ]])
        
        # ベース予測
        base_intensity = self.intensity_model.predict(feature_vector)[0]
        base_lag_t = self.lag_t_model.predict(feature_vector)[0]
        base_lag_t1 = self.lag_t1_model.predict(feature_vector)[0]
        
        # 年次減衰効果を適用
        year_factor = self._calculate_year_factor(year_offset, event_type)
        
        # 年次調整後の強度
        intensity = base_intensity * year_factor
        lag_t = base_lag_t * year_factor
        lag_t1 = base_lag_t1 * year_factor
        
        # 0-1の範囲にクリップ
        intensity = np.clip(intensity, 0, 1)
        lag_t = np.clip(lag_t, 0, 1)
        lag_t1 = np.clip(lag_t1, 0, 1)
        
        return {
            'intensity': float(intensity),
            'lag_t': float(lag_t),
            'lag_t1': float(lag_t1)
        }
    
    def save_models(self, model_path: str, params_path: str):
        """学習済みモデルとパラメータを保存"""
        # モデル保存
        model_data = {
            'intensity_model': self.intensity_model,
            'lag_t_model': self.lag_t_model,
            'lag_t1_model': self.lag_t1_model,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, model_path)
        
        # デフォルトパラメータ保存
        if hasattr(self, 'default_params'):
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(self.default_params, f, ensure_ascii=False, indent=2)
        
        print(f"[強度学習] モデルを保存: {model_path}")
        print(f"[強度学習] パラメータを保存: {params_path}")
    
    def load_models(self, model_path: str):
        """学習済みモデルを読み込み"""
        if not Path(model_path).exists():
            print(f"[強度学習] モデルファイルが見つかりません: {model_path}")
            self._set_default_params()
            return
        
        model_data = joblib.load(model_path)
        self.intensity_model = model_data['intensity_model']
        self.lag_t_model = model_data['lag_t_model']
        self.lag_t1_model = model_data['lag_t1_model']
        self.is_fitted = model_data['is_fitted']
        
        print(f"[強度学習] モデルを読み込み: {model_path}")

def main():
    """メイン処理"""
    print("=== 強度学習システム ===")
    
    # データ読み込み
    features_panel = pd.read_csv(P_FEATURES_PANEL)
    print(f"[強度学習] データ読み込み完了: {len(features_panel)}行")
    
    # 強度学習システム初期化
    learner = IntensityLearner()
    
    # 学習データ準備
    train_data = learner.prepare_training_data(features_panel)
    
    # モデル学習
    learner.fit(train_data)
    
    # モデル保存
    learner.save_models(P_OUTPUT_MODEL, P_OUTPUT_PARAMS)
    
    # テスト予測
    print("\n=== テスト予測 ===")
    test_features = {
        'log_pop': 8.0,
        'pop_density': 3.0,
        'year_normalized': 0.5
    }
    
    for event_type in ['housing', 'employment', 'commercial']:
        params = learner.predict_intensity(event_type, test_features)
        print(f"{event_type}: intensity={params['intensity']:.3f}, lag_t={params['lag_t']:.3f}, lag_t1={params['lag_t1']:.3f}")

if __name__ == "__main__":
    main()
