# -*- coding: utf-8 -*-
# src/layer5/scenario_with_learned_intensity.py
"""
学習された強度を使用したシナリオ生成
- 強度学習システムから最適な強度パラメータを取得
- 自動的に最適化されたシナリオを生成
- ダッシュボードやCLIで使用可能
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from train_intensity_learner import IntensityLearner

class LearnedScenarioGenerator:
    """学習された強度を使用したシナリオ生成器"""
    
    def __init__(self, model_path: str = "../../models/intensity_learner.joblib"):
        self.learner = IntensityLearner()
        self.learner.load_models(model_path)
        
        # 町丁の基本情報を読み込み
        self.town_features = self._load_town_features()
    
    def _load_town_features(self) -> Dict[str, Dict[str, float]]:
        """町丁の基本特徴量を読み込み"""
        features_path = "../../data/processed/features_panel.csv"
        if not Path(features_path).exists():
            print(f"[警告] 特徴量ファイルが見つかりません: {features_path}")
            return {}
        
        df = pd.read_csv(features_path)
        
        # 最新年のデータを取得
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        town_features = {}
        for _, row in latest_data.iterrows():
            town = row['town']
            town_features[town] = {
                'log_pop': np.log1p(row.get('pop_total', 0)),
                'pop_density': row.get('pop_total', 0) / 1000,
                'year_normalized': (latest_year - df['year'].min()) / (df['year'].max() - df['year'].min())
            }
        
        print(f"[強度学習] 町丁特徴量を読み込み: {len(town_features)}町丁")
        return town_features
    
    def create_learned_scenario(self, 
                               town: str, 
                               event_type: str, 
                               effect_direction: str = "increase",
                               base_year: int = 2025,
                               year_offset: int = 1,
                               confidence: float = 1.0) -> Dict[str, Any]:
        """学習された強度でシナリオを作成"""
        
        # 町丁の特徴量を取得
        if town in self.town_features:
            features = self.town_features[town]
        else:
            # デフォルト特徴量
            features = {
                'log_pop': 8.0,
                'pop_density': 3.0,
                'year_normalized': 0.5
            }
            print(f"[警告] 町丁 '{town}' の特徴量が見つかりません。デフォルト値を使用します。")
        
        # 学習された強度パラメータを取得（年次別対応）
        learned_params = self.learner.predict_intensity(event_type, features, year_offset)
        
        # 年次別強度の表示
        print(f"[年次別強度] {event_type} (年オフセット={year_offset}):")
        print(f"  intensity={learned_params['intensity']:.3f}, lag_t={learned_params['lag_t']:.3f}, lag_t1={learned_params['lag_t1']:.3f}")
        
        # シナリオ作成
        scenario = {
            "town": town,
            "base_year": base_year,
            "horizons": [1, 2, 3],
            "events": [{
                "year_offset": year_offset,
                "event_type": event_type,
                "effect_direction": effect_direction,
                "confidence": confidence,
                "intensity": learned_params['intensity'],
                "lag_t": learned_params['lag_t'],
                "lag_t1": learned_params['lag_t1'],
                "note": f"{event_type} ({effect_direction}) - 学習された強度"
            }],
            "macros": {},
            "manual_delta": {
                "h1": 0.0,
                "h2": 0.0,
                "h3": 0.0
            },
            "manual_delta_rate": {
                "h1": 0.0,
                "h2": 0.0,
                "h3": 0.0
            }
        }
        
        return scenario
    
    def create_learned_scenario_with_yearly_intensity(self, town: str, event_type: str, effect_direction: str = "increase") -> Dict[str, Any]:
        """年次別強度を適用したシナリオを作成"""
        
        # 町丁の特徴量を取得
        if town in self.town_features:
            features = self.town_features[town]
        else:
            # デフォルト特徴量
            features = {
                'log_pop': 8.0,
                'pop_density': 3.0,
                'year_normalized': 0.5
            }
            print(f"[警告] 町丁 '{town}' の特徴量が見つかりません。デフォルト値を使用します。")
        
        # 年次別の強度パラメータを予測
        events = []
        for year_offset in [0, 1, 2]:  # 1年目、2年目、3年目
            learned_params = self.learner.predict_intensity(event_type, features, year_offset)
            
            print(f"[年次別強度] {event_type} (年オフセット={year_offset}):")
            print(f"  intensity={learned_params['intensity']:.3f}, lag_t={learned_params['lag_t']:.3f}, lag_t1={learned_params['lag_t1']:.3f}")
            
            # 各年でイベントを作成
            if learned_params['intensity'] > 0.01:  # 強度が十分大きい場合のみ
                events.append({
                    "year_offset": year_offset + 1,  # 1年目、2年目、3年目
                    "event_type": event_type,
                    "effect_direction": effect_direction,
                    "confidence": 1.0,
                    "intensity": learned_params['intensity'],
                    "lag_t": learned_params['lag_t'],
                    "lag_t1": learned_params['lag_t1'],
                    "note": f"{event_type} ({effect_direction}) - 年次別強度 (年{year_offset+1})"
                })
        
        # シナリオ作成
        scenario = {
            "town": town,
            "base_year": 2025,
            "horizons": [1, 2, 3],
            "events": events,
            "macros": {},
            "manual_delta": {
                "h1": 0.0,
                "h2": 0.0,
                "h3": 0.0
            },
            "manual_delta_rate": {
                "h1": 0.0,
                "h2": 0.0,
                "h3": 0.0
            }
        }
        return scenario
    
    def create_learned_scenario_with_year_breakdown(self, town: str, event_type: str, effect_direction: str = "increase") -> Dict[str, Any]:
        """学習された強度でシナリオを作成（年次別内訳表示付き）"""
        
        # 町丁の特徴量を取得
        if town in self.town_features:
            features = self.town_features[town]
        else:
            # デフォルト特徴量
            features = {
                'log_pop': 8.0,
                'pop_density': 3.0,
                'year_normalized': 0.5
            }
            print(f"[警告] 町丁 '{town}' の特徴量が見つかりません。デフォルト値を使用します。")
        
        # 年次別の強度パラメータを予測
        learned_params_by_year = {}
        for year_offset in [0, 1, 2]:  # 1年目、2年目、3年目
            learned_params = self.learner.predict_intensity(event_type, features, year_offset)
            learned_params_by_year[year_offset] = learned_params

        print(f"\n町丁: {town}")
        print("==================================================")
        print(f"\n{event_type.upper()} イベント（年次別強度）:")
        
        for year_offset in [0, 1, 2]:
            year_name = ["1年目", "2年目", "3年目"][year_offset]
            learned = learned_params_by_year[year_offset]
            print(f"  {year_name}:")
            print(f"    学習された強度: intensity={learned['intensity']:.3f}, lag_t={learned['lag_t']:.3f}, lag_t1={learned['lag_t1']:.3f}")
            
            # 年次減衰効果の表示
            year_factor = self.learner._calculate_year_factor(year_offset, event_type)
            print(f"    年次減衰係数: {year_factor:.3f}")

        # シナリオJSONを構築（1年目の強度を使用）
        first_year_params = learned_params_by_year[0]
        scenario = {
            "town": town,
            "base_year": 2025,
            "horizons": [1, 2, 3],
            "events": [
                {
                    "year_offset": 1,
                    "event_type": event_type,
                    "effect_direction": effect_direction,
                    "confidence": 1.0,
                    "intensity": first_year_params.get('intensity', 1.0),
                    "lag_t": first_year_params.get('lag_t', 1.0),
                    "lag_t1": first_year_params.get('lag_t1', 1.0),
                    "note": f"{event_type} (学習された強度 - 年次減衰適用)"
                }
            ],
            "macros": {},
            "manual_delta": {
                "h1": 0.0,
                "h2": 0.0,
                "h3": 0.0
            },
            "manual_delta_rate": {
                "h1": 0.0,
                "h2": 0.0,
                "h3": 0.0
            }
        }
        return scenario
    
    def create_comparison_scenarios(self, 
                                   town: str, 
                                   event_type: str,
                                   base_year: int = 2025) -> Dict[str, Dict[str, Any]]:
        """学習された強度とデフォルト強度の比較シナリオを作成"""
        
        # 学習された強度のシナリオ
        learned_inc = self.create_learned_scenario(town, event_type, "increase", base_year)
        learned_dec = self.create_learned_scenario(town, event_type, "decrease", base_year)
        
        # デフォルト強度のシナリオ
        default_inc = self._create_default_scenario(town, event_type, "increase", base_year)
        default_dec = self._create_default_scenario(town, event_type, "decrease", base_year)
        
        return {
            "learned_increase": learned_inc,
            "learned_decrease": learned_dec,
            "default_increase": default_inc,
            "default_decrease": default_dec
        }
    
    def _create_default_scenario(self, 
                                town: str, 
                                event_type: str, 
                                effect_direction: str,
                                base_year: int = 2025) -> Dict[str, Any]:
        """デフォルト強度（1.0）のシナリオを作成"""
        return {
            "town": town,
            "base_year": base_year,
            "horizons": [1, 2, 3],
            "events": [{
                "year_offset": 1,
                "event_type": event_type,
                "effect_direction": effect_direction,
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1.0,
                "lag_t1": 1.0,
                "note": f"{event_type} ({effect_direction}) - デフォルト強度"
            }],
            "macros": {},
            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0},
            "manual_delta_rate": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
        }

def main():
    """メイン処理 - 学習された強度でのシナリオ生成テスト"""
    print("=== 学習された強度でのシナリオ生成 ===")
    
    # シナリオ生成器初期化
    generator = LearnedScenarioGenerator()
    
    # テスト用のシナリオ生成
    test_town = "四方寄町"
    test_event_types = ["housing", "employment", "commercial"]
    
    print(f"\n町丁: {test_town}")
    print("=" * 50)
    
    for event_type in test_event_types:
        print(f"\n{event_type.upper()} イベント:")
        
        # 学習された強度のシナリオ
        learned_scenario = generator.create_learned_scenario(test_town, event_type, "increase")
        event = learned_scenario["events"][0]
        
        print(f"  学習された強度:")
        print(f"    intensity: {event['intensity']:.3f}")
        print(f"    lag_t: {event['lag_t']:.3f}")
        print(f"    lag_t1: {event['lag_t1']:.3f}")
        
        # デフォルト強度との比較
        default_scenario = generator._create_default_scenario(test_town, event_type, "increase")
        default_event = default_scenario["events"][0]
        
        print(f"  デフォルト強度:")
        print(f"    intensity: {default_event['intensity']:.3f}")
        print(f"    lag_t: {default_event['lag_t']:.3f}")
        print(f"    lag_t1: {default_event['lag_t1']:.3f}")
        
        # 強度の違いを計算
        intensity_diff = event['intensity'] - default_event['intensity']
        lag_t_diff = event['lag_t'] - default_event['lag_t']
        lag_t1_diff = event['lag_t1'] - default_event['lag_t1']
        
        print(f"  差 (学習 - デフォルト):")
        print(f"    intensity: {intensity_diff:+.3f}")
        print(f"    lag_t: {lag_t_diff:+.3f}")
        print(f"    lag_t1: {lag_t1_diff:+.3f}")

if __name__ == "__main__":
    main()
