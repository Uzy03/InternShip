# -*- coding: utf-8 -*-
"""
空間的影響予測モジュール (Panel版)
A町丁でイベント発生時の周辺町丁への影響を予測
"""
import panel as pn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
from typing import Dict, List, Any
import param

# パス設定（Colab用）
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))

# Layer5モジュールのインポート
try:
    from scenario_to_events import scenario_to_events
    from prepare_baseline import prepare_baseline
    from build_future_features import build_future_features
    from forecast_service import forecast_population
except ImportError as e:
    print(f"Layer5モジュールのインポートに失敗しました: {e}")

def calculate_decay_factor(event_type, distance):
    """
    距離減衰関数（イベントタイプに応じて調整）
    
    Args:
        event_type: イベントタイプ
        distance: 距離
    
    Returns:
        float: 減衰係数
    """
    # 基本減衰係数
    base_decay = 1.0 / (1.0 + distance * 0.1)
    
    # イベントタイプ別の調整
    if event_type == "housing":
        # 住宅供給：主に発生地のみ（周辺への影響小）
        return base_decay * 0.3
    elif event_type == "disaster":
        # 災害：広範囲に影響（周辺への影響大）
        return base_decay * 1.5
    elif event_type == "commercial":
        # 商業施設：中程度の周辺影響
        return base_decay * 0.8
    elif event_type == "transit":
        # 交通：沿線への影響
        return base_decay * 1.2
    elif event_type == "employment":
        # 雇用：中程度の周辺影響
        return base_decay * 0.7
    else:
        # その他：標準的な減衰
        return base_decay

def batch_spatial_prediction(event_town, event_type, effect_direction, base_year, towns, centroids_df, debug_mode=False):
    """
    バッチ処理による空間的影響予測
    
    Args:
        event_town: イベント発生町丁
        event_type: イベントタイプ
        effect_direction: 効果方向
        base_year: 基準年
        towns: 全町丁リスト
        centroids_df: 重心データ
    
    Returns:
        dict: 空間的影響の結果
    """
    # シナリオ設定（scenario_to_eventsの形式に合わせる）
    scenario = {
        "town": event_town,
        "base_year": base_year,
        "horizons": [1, 2, 3],
        "events": [{
            "year_offset": 0,  # base_yearからのオフセット
            "event_type": event_type,
            "effect_direction": effect_direction,
            "confidence": 1.0,  # 確信度
            "intensity": 1.0,   # 強度
            "lag_t": 0.8,       # 当年効果の割合
            "lag_t1": 0.2,      # 翌年効果の割合
            "note": f"{event_type} ({effect_direction})"
        }],
        "macros": {},
        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
    }
    
    # イベント発生町丁の予測（直接効果）
    event_baseline = prepare_baseline(event_town, base_year)
    event_future_events = scenario_to_events(scenario)
    event_future_features = build_future_features(event_baseline, event_future_events, scenario)
    
    # 将来特徴を保存
    features_path = Path("output/l5_future_features.csv")
    event_future_features.to_csv(features_path, index=False)
    
    # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
    event_future_features.to_csv(Path("../../data/processed/l5_future_features.csv"), index=False)
    
    # イベント発生町丁の予測実行
    event_result = forecast_population(
        town=event_town,
        base_year=base_year,
        horizons=[1, 2, 3],
        base_population=float(event_baseline["pop_total"].iloc[0]),
        debug_output_dir=None,
        manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
        apply_event_to_prediction=True
    )
    
    # バッチ処理：全町丁の将来特徴を一括構築
    print("🔄 全町丁の将来特徴を一括構築中...")
    
    # 全町丁のベースラインデータを一括取得
    all_baselines = {}
    for town in towns:
        if town != event_town:  # イベント発生町丁は除外
            all_baselines[town] = prepare_baseline(town, base_year)
    
    # 全町丁の将来特徴を一括構築
    all_future_features_list = []
    for town, baseline in all_baselines.items():
        town_scenario = {
            "town": town,
            "base_year": base_year,
            "horizons": [1, 2, 3],
            "events": [],  # イベントなし
            "macros": {},
            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
        }
        town_future_events = scenario_to_events(town_scenario)
        town_future_features = build_future_features(baseline, town_future_events, town_scenario)
        all_future_features_list.append(town_future_features)
    
    # 全町丁の将来特徴を結合して保存
    if all_future_features_list:
        combined_future_features = pd.concat(all_future_features_list, ignore_index=True)
        combined_future_features.to_csv(features_path, index=False)
        
        # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
        combined_future_features.to_csv(Path("../../data/processed/l5_future_features.csv"), index=False)
    
    # 周辺町丁の影響を計算
    spatial_impacts = []
    error_count = 0
    
    # 重心データがない場合は簡易的な処理
    if centroids_df.empty or "lat" not in centroids_df.columns or "lon" not in centroids_df.columns:
        print("重心データが利用できないため、簡易的な空間的影響予測を実行します")
        
        # 全町丁に対して簡易的な影響を計算
        for town in towns:
            if town == event_town:
                continue
                
            # 簡易的な距離（町丁名の類似度ベース）
            distance = 1.0  # 固定値
            decay_factor = calculate_decay_factor(event_type, distance)
            
            # 周辺町丁の予測（イベントなし）
            try:
                town_baseline = prepare_baseline(town, base_year)
                town_scenario = {
                    "town": town,
                    "base_year": base_year,
                    "horizons": [1, 2, 3],
                    "events": [],
                    "macros": {},
                    "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                }
                town_future_events = scenario_to_events(town_scenario)
                town_future_features = build_future_features(town_baseline, town_future_events, town_scenario)
                
                # 将来特徴を保存
                town_future_features.to_csv(features_path, index=False)
                
                # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
                town_future_features.to_csv(Path("../../data/processed/l5_future_features.csv"), index=False)
                
                # 周辺町丁の予測実行
                town_result = forecast_population(
                    town=town,
                    base_year=base_year,
                    horizons=[1, 2, 3],
                    base_population=float(town_baseline["pop_total"].iloc[0]),
                    debug_output_dir=None,
                    manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                    apply_event_to_prediction=False
                )
                
                # 空間的影響を計算
                if town_result and "path" in town_result:
                    for entry in town_result["path"]:
                        spatial_impacts.append({
                            "town": town,
                            "year": entry["year"],
                            "h": entry["year"] - base_year,
                            "delta": entry["delta_hat"],
                            "pop": entry["pop_hat"],
                            "exp": entry["contrib"]["exp"],
                            "macro": entry["contrib"]["macro"],
                            "inertia": entry["contrib"]["inertia"],
                            "other": entry["contrib"]["other"],
                            "distance": distance,
                            "decay_factor": decay_factor,
                            "spatial_impact": entry["delta_hat"] * decay_factor,
                            "lat": 0.0,  # デフォルト値
                            "lon": 0.0   # デフォルト値
                        })
            except Exception as e:
                error_count += 1
                if not debug_mode:
                    print(f"町丁 '{town}' の予測に失敗: {e}")
                continue
    else:
        # イベント発生町丁からの距離を計算
        event_coords = centroids_df[centroids_df["town"] == event_town]
        if not event_coords.empty:
            event_lat = event_coords["lat"].iloc[0]
            event_lon = event_coords["lon"].iloc[0]
            
            # 各町丁への距離を計算
            for _, town_row in centroids_df.iterrows():
                town = town_row["town"]
                if town == event_town:
                    continue
                
                # 簡易的な距離計算（実際の距離計算に置き換え可能）
                lat_diff = town_row["lat"] - event_lat
                lon_diff = town_row["lon"] - event_lon
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                
                # 距離減衰関数（イベントタイプに応じて調整）
                decay_factor = calculate_decay_factor(event_type, distance)
                
                # 周辺町丁の予測（イベントなし）
                try:
                    town_baseline = prepare_baseline(town, base_year)
                    town_scenario = {
                        "town": town,
                        "base_year": base_year,
                        "horizons": [1, 2, 3],
                        "events": [],  # イベントなし
                        "macros": {},
                        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                    }
                    town_future_events = scenario_to_events(town_scenario)
                    town_future_features = build_future_features(town_baseline, town_future_events, town_scenario)
                    
                    # 将来特徴を保存
                    town_future_features.to_csv(features_path, index=False)
                    
                    # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
                    town_future_features.to_csv(Path("../../data/processed/l5_future_features.csv"), index=False)
                    
                    # 周辺町丁の予測実行
                    town_result = forecast_population(
                        town=town,
                        base_year=base_year,
                        horizons=[1, 2, 3],
                        base_population=float(town_baseline["pop_total"].iloc[0]),
                        debug_output_dir=None,
                        manual_add_by_h={1: 0.0, 2: 0.0, 3: 0.0},
                        apply_event_to_prediction=False
                    )
                    
                    # 空間的影響を計算
                    if town_result and "path" in town_result:
                        for entry in town_result["path"]:
                            spatial_impacts.append({
                                "town": town,
                                "year": entry["year"],
                                "h": entry["year"] - base_year,
                                "delta": entry["delta_hat"],
                                "pop": entry["pop_hat"],
                                "exp": entry["contrib"]["exp"],
                                "macro": entry["contrib"]["macro"],
                                "inertia": entry["contrib"]["inertia"],
                                "other": entry["contrib"]["other"],
                                "distance": distance,
                                "decay_factor": decay_factor,
                                "spatial_impact": entry["delta_hat"] * decay_factor,
                                "lat": town_row["lat"],
                                "lon": town_row["lon"]
                            })
                except Exception as e:
                    error_count += 1
                    if not debug_mode:
                        print(f"町丁 '{town}' の予測に失敗: {e}")
                    continue
    
    # エラー統計を表示
    if debug_mode and error_count > 0:
        print(f"🔧 デバッグモード: {error_count}町丁で予測に失敗しました（エラーログは非表示）")
    
    return {
        "event_town": event_town,
        "event_result": event_result,
        "spatial_impacts": spatial_impacts
    }

class SpatialImpactPrediction(param.Parameterized):
    """空間的影響予測のPanelクラス"""
    
    # パラメータ
    event_town = param.Selector(default="", objects=[], doc="イベント発生町丁")
    event_type_display = param.Selector(default="", objects=[], doc="イベントタイプ")
    base_year = param.Integer(default=2025, bounds=(2020, 2030), doc="基準年")
    selected_year = param.Selector(default=2026, objects=[2026, 2027, 2028], doc="表示年")
    debug_mode = param.Boolean(default=True, doc="デバッグモード")
    use_batch = param.Boolean(default=True, doc="バッチ処理を使用")
    
    # 内部状態
    result = param.Dict(default={}, doc="予測結果")
    loading = param.Boolean(default=False, doc="読み込み中")
    error_message = param.String(default="", doc="エラーメッセージ")
    centroids_df = param.DataFrame(default=pd.DataFrame(), doc="重心データ")
    
    def __init__(self, towns, **params):
        super().__init__(**params)
        
        # イベントタイプのマッピング
        self.EVENT_TYPE_MAPPING = {
            "housing_inc": "住宅供給の増加（竣工）",
            "housing_dec": "住宅の減少・喪失",
            "commercial_inc": "商業施設の増加", 
            "transit_inc": "交通利便の向上",
            "transit_dec": "交通利便の低下",
            "public_edu_medical_inc": "公共・教育・医療の増加",
            "public_edu_medical_dec": "公共・教育・医療の減少",
            "employment_inc": "雇用機会の増加",
            "employment_dec": "雇用機会の減少",
            "disaster_inc": "災害被害・リスクの増加",
            "disaster_dec": "災害リスクの低下（防災整備）"
        }
        
        # パラメータの初期化
        self.param.event_town.objects = towns
        self.param.event_town.default = towns[0] if towns else ""
        self.param.event_type_display.objects = list(self.EVENT_TYPE_MAPPING.values())
        self.param.event_type_display.default = list(self.EVENT_TYPE_MAPPING.values())[0] if self.EVENT_TYPE_MAPPING else ""
        
        # 重心データの読み込み
        self.load_centroids()
    
    def load_centroids(self):
        """重心データを読み込み"""
        centroids_path = Path("../../data/processed/town_centroids.csv")
        if not centroids_path.exists():
            centroids_path = Path("../data/processed/town_centroids.csv")
        
        if centroids_path.exists():
            try:
                self.centroids_df = pd.read_csv(centroids_path, usecols=["town", "lat", "lon"])
                print(f"重心データを読み込みました: {len(self.centroids_df)}町丁")
            except Exception as e:
                print(f"重心データの読み込みに失敗: {e}")
                self.centroids_df = pd.DataFrame()
        else:
            print("重心データファイルが見つかりません")
            self.centroids_df = pd.DataFrame()
    
    def get_event_type_and_direction(self):
        """表示名から内部キーに変換"""
        event_type_full = [k for k, v in self.EVENT_TYPE_MAPPING.items() if v == self.event_type_display][0]
        
        if event_type_full.endswith("_inc"):
            event_type = event_type_full[:-4]
            effect_direction = "increase"
        elif event_type_full.endswith("_dec"):
            event_type = event_type_full[:-4]
            effect_direction = "decrease"
        else:
            event_type = event_type_full
            effect_direction = "increase"
        
        return event_type, effect_direction
    
    def run_prediction(self):
        """予測を実行"""
        if not self.event_town or not self.event_type_display:
            self.error_message = "イベント発生町丁とイベントタイプを選択してください。"
            return
        
        self.loading = True
        self.error_message = ""
        
        try:
            event_type, effect_direction = self.get_event_type_and_direction()
            
            # 処理対象の町丁を決定
            if self.debug_mode:
                towns_to_process = self.param.event_town.objects[:10]
                print(f"🔧 デバッグモード: {len(towns_to_process)}町丁で処理します")
            else:
                towns_to_process = self.param.event_town.objects
                print(f"📊 本格モード: {len(towns_to_process)}町丁で処理します")
            
            # 空間的影響を計算
            import time
            start_time = time.time()
            
            if self.use_batch:
                print("⚡ バッチ処理を使用します")
                self.result = batch_spatial_prediction(
                    event_town=self.event_town,
                    event_type=event_type,
                    effect_direction=effect_direction,
                    base_year=self.base_year,
                    towns=towns_to_process,
                    centroids_df=self.centroids_df,
                    debug_mode=self.debug_mode
                )
            else:
                print("🔄 個別処理を使用します")
                # 個別処理の実装は省略（必要に応じて追加）
                self.result = batch_spatial_prediction(
                    event_town=self.event_town,
                    event_type=event_type,
                    effect_direction=effect_direction,
                    base_year=self.base_year,
                    towns=towns_to_process,
                    centroids_df=self.centroids_df,
                    debug_mode=self.debug_mode
                )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if self.result and self.result.get("spatial_impacts"):
                print(f"✅ 空間的影響予測が完了しました！（処理時間: {processing_time:.1f}秒）")
                
                # 利用可能な年を更新
                years = sorted(list(set([impact["year"] for impact in self.result["spatial_impacts"]])))
                self.param.selected_year.objects = years
                if years:
                    self.selected_year = years[-1]  # 最新年を選択
            else:
                self.error_message = "空間的影響の計算に失敗しました。"
                
        except Exception as e:
            self.error_message = f"エラーが発生しました: {str(e)}"
            print(f"予測実行エラー: {e}")
        
        finally:
            self.loading = False
    
    @param.depends('result', 'selected_year')
    def event_town_details(self):
        """イベント発生町丁の直接効果"""
        if not self.result or not self.result.get("event_result"):
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        event_data = []
        for entry in self.result["event_result"]["path"]:
            event_data.append({
                "年": entry["year"],
                "Δ人口": f"{entry['delta_hat']:.1f}人",
                "人口": f"{entry['pop_hat']:.1f}人",
                "期待効果": f"{entry['contrib']['exp']:.1f}人",
                "マクロ": f"{entry['contrib']['macro']:.1f}人",
                "慣性": f"{entry['contrib']['inertia']:.1f}人",
                "その他": f"{entry['contrib']['other']:.1f}人"
            })
        
        event_df = pd.DataFrame(event_data)
        
        return pn.widgets.Tabulator(
            event_df,
            pagination='remote',
            page_size=10,
            sizing_mode="stretch_width"
        )
    
    @param.depends('result', 'selected_year')
    def spatial_impact_map(self):
        """空間的影響の地図"""
        if not self.result or not self.result.get("spatial_impacts"):
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        # 選択年のデータをフィルタ
        year_data = [impact for impact in self.result["spatial_impacts"] if impact["year"] == self.selected_year]
        
        if not year_data:
            return pn.pane.HTML("<p>選択された年のデータがありません。</p>")
        
        impacts_df = pd.DataFrame(year_data)
        
        # 重心データがない場合はテーブル表示のみ
        if self.centroids_df.empty or "lat" not in self.centroids_df.columns or "lon" not in self.centroids_df.columns:
            return pn.pane.HTML("<p>重心データが利用できないため、マップ表示はスキップします</p>")
        
        # マップ表示
        fig = go.Figure()
        
        # イベント発生町丁（赤色で表示）
        event_coords = self.centroids_df[self.centroids_df["town"] == self.event_town]
        if not event_coords.empty:
            fig.add_trace(go.Scattermapbox(
                lat=[event_coords["lat"].iloc[0]],
                lon=[event_coords["lon"].iloc[0]],
                mode='markers',
                marker=dict(
                    size=30,
                    color='red',
                    opacity=0.9
                ),
                text=[f"{self.event_town}<br>【イベント発生地】"],
                hovertemplate='%{text}<extra></extra>',
                name='イベント発生地'
            ))
        
        # 周辺町丁（空間的影響に応じて色とサイズを調整）
        fig.add_trace(go.Scattermapbox(
            lat=impacts_df['lat'],
            lon=impacts_df['lon'],
            mode='markers',
            marker=dict(
                size=15,
                color=impacts_df['spatial_impact'],  # 数値で色を指定
                colorscale='RdBu',  # 赤-青のカラースケール
                cmin=impacts_df['spatial_impact'].min(),
                cmax=impacts_df['spatial_impact'].max(),
                colorbar=dict(
                    title="空間的影響（人）",
                    tickmode="auto",
                    nticks=5
                ),
                opacity=0.8
            ),
            text=impacts_df['town'] + '<br>空間的影響: ' + impacts_df['spatial_impact'].round(3).astype(str) + '人<br>距離: ' + impacts_df['distance'].round(2).astype(str),
            hovertemplate='%{text}<extra></extra>',
            name='周辺町丁'
        ))
        
        # 地図の中心をイベント発生町丁に設定
        if not event_coords.empty:
            center_lat = event_coords["lat"].iloc[0]
            center_lon = event_coords["lon"].iloc[0]
        else:
            center_lat = impacts_df['lat'].mean()
            center_lon = impacts_df['lon'].mean()
        
        # 影響度の範囲を計算
        min_impact = impacts_df['spatial_impact'].min()
        max_impact = impacts_df['spatial_impact'].max()
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            height=600,
            title=f"{self.selected_year}年の空間的影響分布（赤: イベント発生地、赤-青: 影響度）<br>影響度範囲: {min_impact:.3f} ～ {max_impact:.3f}人"
        )
        
        return pn.pane.Plotly(fig)
    
    @param.depends('result', 'selected_year')
    def spatial_impact_table(self):
        """空間的影響のテーブル"""
        if not self.result or not self.result.get("spatial_impacts"):
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        # 選択年のデータをフィルタ
        year_data = [impact for impact in self.result["spatial_impacts"] if impact["year"] == self.selected_year]
        
        if not year_data:
            return pn.pane.HTML("<p>選択された年のデータがありません。</p>")
        
        impacts_df = pd.DataFrame(year_data)
        impacts_df = impacts_df.sort_values('spatial_impact', ascending=False)
        
        # 表示する列を選択（重心データの有無に応じて調整）
        if not self.centroids_df.empty and "lat" in self.centroids_df.columns and "lon" in self.centroids_df.columns:
            display_columns = ["town", "spatial_impact", "delta", "distance", "decay_factor", "lat", "lon"]
        else:
            display_columns = ["town", "spatial_impact", "delta", "distance", "decay_factor"]
        
        display_df = impacts_df[display_columns].copy()
        
        # 列名を日本語に変更
        column_mapping = {
            "town": "町丁",
            "spatial_impact": "空間的影響",
            "delta": "Δ人口",
            "distance": "距離",
            "decay_factor": "減衰係数",
            "lat": "緯度",
            "lon": "経度"
        }
        
        display_df = display_df.rename(columns=column_mapping)
        
        # 数値列を丸める
        numeric_columns = ["空間的影響", "Δ人口", "距離", "減衰係数"]
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        return pn.widgets.Tabulator(
            display_df,
            pagination='remote',
            page_size=20,
            sizing_mode="stretch_width"
        )
    
    @param.depends('result', 'selected_year')
    def statistics(self):
        """統計情報"""
        if not self.result or not self.result.get("spatial_impacts"):
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        # 選択年のデータをフィルタ
        year_data = [impact for impact in self.result["spatial_impacts"] if impact["year"] == self.selected_year]
        
        if not year_data:
            return pn.pane.HTML("<p>選択された年のデータがありません。</p>")
        
        impacts_df = pd.DataFrame(year_data)
        
        # 統計計算
        stats = {
            "影響を受ける町丁数": len(impacts_df),
            "平均空間的影響": f"{impacts_df['spatial_impact'].mean():.2f}人",
            "最大空間的影響": f"{impacts_df['spatial_impact'].max():.2f}人",
            "最小空間的影響": f"{impacts_df['spatial_impact'].min():.2f}人"
        }
        
        html = f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3>📈 統計情報</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
        """
        
        for key, value in stats.items():
            html += f'<div><strong>{key}:</strong> {value}</div>'
        
        html += "</div></div>"
        
        return pn.pane.HTML(html)
    
    def view(self):
        """メインビュー"""
        # コントロール
        controls = pn.Column(
            pn.pane.HTML("<h2>🌍 空間的影響予測</h2>"),
            pn.pane.HTML("<p>A町丁でイベント発生時の周辺町丁への影響を予測します</p>"),
            pn.pane.HTML("<h3>🎯 イベント設定</h3>"),
            pn.Param(self, parameters=['event_town', 'event_type_display', 'base_year']),
            pn.pane.HTML("<h3>⚙️ 処理設定</h3>"),
            pn.Param(self, parameters=['debug_mode', 'use_batch']),
            pn.widgets.Button(name="🚀 空間的影響予測を実行", button_type="primary"),
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>🔍 表示設定</h3>"),
            pn.Param(self, parameters=['selected_year']),
            width=300
        )
        
        # メインコンテンツ
        main_content = pn.Column(
            self.event_town_details,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>🗺️ 空間的影響の可視化</h3>"),
            self.spatial_impact_map,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>📊 空間的影響の詳細</h3>"),
            self.spatial_impact_table,
            pn.pane.HTML("<hr>"),
            self.statistics,
            width=1000
        )
        
        # エラーメッセージ
        if self.error_message:
            error_pane = pn.pane.Alert(self.error_message, alert_type="danger")
            main_content.insert(0, error_pane)
        
        # ローディング表示
        if self.loading:
            loading_pane = pn.pane.HTML("<div style='text-align: center; padding: 20px;'><h3>🔄 空間的影響予測を実行中...</h3></div>")
            main_content.insert(0, loading_pane)
        
        # ボタンイベント
        def on_button_click(event):
            self.run_prediction()
        
        controls[6].on_click(on_button_click)
        
        return pn.Row(controls, main_content, sizing_mode="stretch_width")

def create_spatial_impact_prediction(towns):
    """空間的影響予測コンポーネントを作成"""
    return SpatialImpactPrediction(towns)
