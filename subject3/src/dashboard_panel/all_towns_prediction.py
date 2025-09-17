# -*- coding: utf-8 -*-
"""
全地域予測機能 (Panel版)
イベント発生町丁を指定して全町丁の人口予測を実行
"""
import panel as pn
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
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
    from forecast_service import forecast_population, run_scenario
    from scenario_with_learned_intensity import LearnedScenarioGenerator
except ImportError as e:
    print(f"Layer5モジュールのインポートに失敗しました: {e}")

def run_all_towns_realtime_prediction(event_town: str, event_type: str, effect_direction: str, base_year: int = 2025, horizons: list = [1, 2, 3]):
    """指定されたイベントで全町丁のリアルタイム予測を実行"""
    try:
        # 利用可能な町丁リストを取得
        features_path = Path("../../data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../data/processed/features_panel.csv")
        
        if not features_path.exists():
            print(f"features_panel.csv が見つかりません: {features_path}")
            return None
        
        df = pd.read_csv(features_path, usecols=["town"]).drop_duplicates()
        all_towns = sorted(df["town"].unique().tolist())
        
        # デバッグ用：最初の10町丁のみでテスト
        debug_towns = all_towns[:11]  # イベント発生町丁 + 10町丁
        if len(debug_towns) < len(all_towns):
            print(f"デバッグモード: 最初の{len(debug_towns)}町丁のみでテストします（全{len(all_towns)}町丁中）")
        
        # 将来特徴が見つからない町丁のカウンター
        missing_features_count = 0
        max_missing_features = 5  # 最大5町丁まで警告を表示
        
        # ベースラインデータを取得
        baseline_path = Path("../../data/processed/l5_baseline.csv")
        if not baseline_path.exists():
            baseline_path = Path("../data/processed/l5_baseline.csv")
        
        if not baseline_path.exists():
            print(f"l5_baseline.csv が見つかりません: {baseline_path}")
            return None
        
        baseline_df = pd.read_csv(baseline_path)
        
        # シナリオを作成
        scenario = {
            "town": event_town,  # イベント発生町丁
            "base_year": base_year,
            "horizons": horizons,
            "events": [{
                "year_offset": 1,
                "event_type": event_type,
                "effect_direction": effect_direction,
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1,
                "lag_t1": 1,
                "note": f"{event_type} ({effect_direction})"
            }],
            "macros": {},
            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
        }
        
        # 全町丁の予測を実行
        print("全町丁の予測を実行中...")
        
        all_results = []
        
        # 全町丁の将来特徴を一括構築（空間ラグ効果を計算するため）
        print("全町丁の将来特徴を一括構築中...")
        all_future_features = []
        
        try:
            # イベント発生町丁の将来特徴を構築
            event_baseline = prepare_baseline(event_town, base_year)
            event_future_events = scenario_to_events(scenario)
            event_future_features = build_future_features(event_baseline, event_future_events, scenario)
            all_future_features.append(event_future_features)
            
            # 他の町丁の将来特徴を構築（イベントなし）
            for town in debug_towns:
                if town != event_town:
                    town_baseline = prepare_baseline(town, base_year)
                    town_scenario = {
                        "town": town,
                        "base_year": base_year,
                        "horizons": horizons,
                        "events": [],  # イベントなし
                        "macros": {},
                        "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                    }
                    town_future_events = scenario_to_events(town_scenario)
                    town_future_features = build_future_features(town_baseline, town_future_events, town_scenario)
                    all_future_features.append(town_future_features)
            
            # 全町丁の将来特徴を結合して保存
            combined_future_features = pd.concat(all_future_features, ignore_index=True)
            features_output_path = Path("output/l5_future_features.csv")
            combined_future_features.to_csv(features_output_path, index=False)
            
            # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
            combined_future_features.to_csv(Path("../../data/processed/l5_future_features.csv"), index=False)
            
        except Exception as e:
            print(f"将来特徴の一括構築に失敗: {e}")
            # フォールバック: 個別に処理
            pass
        
        # 各町丁の予測を実行
        for i, town in enumerate(debug_towns):
            try:
                # ベースラインデータを取得
                baseline = prepare_baseline(town, base_year)
                
                # 将来特徴を取得（一括構築したものから）
                if i < len(all_future_features):
                    future_features = all_future_features[i]
                else:
                    # フォールバック: 個別に構築
                    if town == event_town:
                        future_events = scenario_to_events(scenario)
                    else:
                        town_scenario = {
                            "town": town,
                            "base_year": base_year,
                            "horizons": horizons,
                            "events": [],
                            "macros": {},
                            "manual_delta": {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                        }
                        future_events = scenario_to_events(town_scenario)
                    
                    future_features = build_future_features(baseline, future_events, scenario if town == event_town else town_scenario)
                
                # 将来特徴を保存
                features_path = Path("../../data/processed/l5_future_features.csv")
                future_features.to_csv(features_path, index=False)
                
                # 人口予測の実行
                base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
                if pd.isna(base_population):
                    base_population = 0.0
                
                manual_add = {1: 0.0, 2: 0.0, 3: 0.0}
                result = forecast_population(town, base_year, horizons, base_population, None, manual_add)
                
                # 結果をフラット化
                for entry in result["path"]:
                    rate_total = entry["contrib"]["exp"] + entry["contrib"]["macro"] + entry["contrib"]["inertia"] + entry["contrib"]["other"]
                    
                    row = {
                        "town": result["town"],
                        "baseline_year": result["base_year"],
                        "year": entry["year"],
                        "h": entry["year"] - result["base_year"],
                        "delta": entry["delta_hat"],
                        "pop": entry["pop_hat"],
                        "exp": entry["contrib"]["exp"],
                        "macro": entry["contrib"]["macro"],
                        "inertia": entry["contrib"]["inertia"],
                        "other": entry["contrib"]["other"],
                        "rate_total": rate_total,
                        "pi_delta_low": entry["pi95_delta"][0] if isinstance(entry["pi95_delta"], list) else entry["pi95_delta"],
                        "pi_delta_high": entry["pi95_delta"][1] if isinstance(entry["pi95_delta"], list) else entry["pi95_delta"],
                        "pi_pop_low": entry["pi95_pop"][0] if isinstance(entry["pi95_pop"], list) else entry["pi95_pop"],
                        "pi_pop_high": entry["pi95_pop"][1] if isinstance(entry["pi95_pop"], list) else entry["pi95_pop"],
                        "is_event_town": (town == event_town),
                    }
                    all_results.append(row)
                
            except Exception as e:
                print(f"町丁 '{town}' の予測でエラー: {e}")
                
                # エラーが発生した町丁も結果に含める（NaNなどで）
                for h_val in horizons:
                    all_results.append({
                        "town": town,
                        "baseline_year": base_year,
                        "year": base_year + h_val,
                        "h": h_val,
                        "delta": float('nan'),
                        "pop": float('nan'),
                        "exp": float('nan'),
                        "macro": float('nan'),
                        "inertia": float('nan'),
                        "other": float('nan'),
                        "pi_delta_low": float('nan'),
                        "pi_delta_high": float('nan'),
                        "pi_pop_low": float('nan'),
                        "pi_pop_high": float('nan'),
                        "is_event_town": (town == event_town),
                    })
        
        print(f"処理完了: {len(all_results)}件の結果を取得")
        
        if not all_results:
            print("予測結果がありません")
            return None
        
        # DataFrameに変換
        result_df = pd.DataFrame(all_results)
        
        # 重心データがあれば結合
        centroids_path = Path("../../data/interim/centroids.csv")
        if not centroids_path.exists():
            centroids_path = Path("../data/interim/centroids.csv")
        
        if centroids_path.exists():
            try:
                centroids_df = pd.read_csv(centroids_path, usecols=["town", "lat", "lon"])
                result_df = pd.merge(result_df, centroids_df, on="town", how="left")
            except Exception as e:
                print(f"重心データの結合に失敗: {e}")
        
        return result_df
        
    except Exception as e:
        print(f"全地域予測の実行に失敗しました: {e}")
        return None

class AllTownsPrediction(param.Parameterized):
    """全地域予測のPanelクラス"""
    
    # パラメータ
    event_town = param.Selector(default="", objects=[], doc="イベント発生町丁")
    event_type_display = param.Selector(default="", objects=[], doc="イベントタイプ")
    selected_year = param.Selector(default=2026, objects=[2026, 2027, 2028], doc="年")
    selected_metric = param.Selector(default="delta", objects=["delta", "pop", "exp", "macro", "inertia", "other"], doc="指標")
    search_term = param.String(default="", doc="町丁名検索")
    sort_option = param.Selector(default="delta_desc", objects=["delta_desc", "delta_asc", "pop_desc", "pop_asc", "town_asc"], doc="ソート順")
    
    # 内部状態
    forecast_df = param.DataFrame(default=pd.DataFrame(), doc="予測結果")
    loading = param.Boolean(default=False, doc="読み込み中")
    error_message = param.String(default="", doc="エラーメッセージ")
    
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
        
        # 指標オプション
        self.METRIC_OPTIONS = {
            "delta": "Δ人口",
            "pop": "人口",
            "exp": "期待効果",
            "macro": "マクロ",
            "inertia": "慣性",
            "other": "その他"
        }
        
        # ソートオプション
        self.SORT_OPTIONS = {
            "delta_desc": "Δ人口（降順）",
            "delta_asc": "Δ人口（昇順）",
            "pop_desc": "人口（降順）",
            "pop_asc": "人口（昇順）",
            "town_asc": "町丁名（昇順）"
        }
        
        # パラメータの初期化
        self.param.event_town.objects = towns
        self.param.event_town.default = towns[0] if towns else ""
        self.param.event_type_display.objects = list(self.EVENT_TYPE_MAPPING.values())
        self.param.event_type_display.default = list(self.EVENT_TYPE_MAPPING.values())[0] if self.EVENT_TYPE_MAPPING else ""
    
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
            
            self.forecast_df = run_all_towns_realtime_prediction(
                event_town=self.event_town,
                event_type=event_type,
                effect_direction=effect_direction
            )
            
            if self.forecast_df is None or self.forecast_df.empty:
                self.error_message = "予測結果が取得できませんでした。"
                return
            
            # 利用可能な年を更新
            available_years = sorted(self.forecast_df['year'].unique())
            self.param.selected_year.objects = available_years
            if available_years:
                self.selected_year = available_years[-1]  # 最新年を選択
            
        except Exception as e:
            self.error_message = f"エラーが発生しました: {str(e)}"
            print(f"予測実行エラー: {e}")
        
        finally:
            self.loading = False
    
    @param.depends('forecast_df', 'selected_year', 'search_term', 'sort_option')
    def filtered_data(self):
        """フィルタリングされたデータ"""
        if self.forecast_df.empty:
            return pd.DataFrame()
        
        # データフィルタリング
        filtered_df = self.forecast_df[self.forecast_df['year'] == self.selected_year].copy()
        
        if self.search_term:
            filtered_df = filtered_df[filtered_df['town'].str.contains(self.search_term, case=False, na=False)]
        
        # ソート
        if self.sort_option == "delta_desc":
            filtered_df = filtered_df.sort_values('delta', ascending=False)
        elif self.sort_option == "delta_asc":
            filtered_df = filtered_df.sort_values('delta', ascending=True)
        elif self.sort_option == "pop_desc":
            filtered_df = filtered_df.sort_values('pop', ascending=False)
        elif self.sort_option == "pop_asc":
            filtered_df = filtered_df.sort_values('pop', ascending=True)
        elif self.sort_option == "town_asc":
            filtered_df = filtered_df.sort_values('town', ascending=True)
        
        return filtered_df
    
    @param.depends('forecast_df', 'selected_year')
    def statistics(self):
        """統計情報"""
        if self.forecast_df.empty:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        filtered_df = self.filtered_data()
        
        if filtered_df.empty:
            return pn.pane.HTML("<p>フィルタリング結果がありません。</p>")
        
        # 統計計算
        stats = {
            "表示町丁数": len(filtered_df),
            "平均Δ人口": f"{filtered_df['delta'].mean():.1f}人",
            "最大Δ人口": f"{filtered_df['delta'].max():.1f}人",
            "最小Δ人口": f"{filtered_df['delta'].min():.1f}人"
        }
        
        html = f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3>📊 統計情報</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
        """
        
        for key, value in stats.items():
            html += f'<div><strong>{key}:</strong> {value}</div>'
        
        html += "</div></div>"
        
        return pn.pane.HTML(html)
    
    @param.depends('forecast_df', 'selected_year')
    def event_town_details(self):
        """イベント発生町丁の詳細"""
        if self.forecast_df.empty:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        event_town_data = self.forecast_df[self.forecast_df['is_event_town'] == True]
        if event_town_data.empty:
            return pn.pane.HTML("<p>イベント発生町丁のデータがありません。</p>")
        
        event_year_data = event_town_data[event_town_data['year'] == self.selected_year]
        if event_year_data.empty:
            return pn.pane.HTML("<p>選択された年のデータがありません。</p>")
        
        row = event_year_data.iloc[0]
        
        html = f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #fff3cd;">
            <h3>🎯 イベント発生町丁の詳細</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; gap: 10px;">
                <div><strong>人口:</strong> {row['pop']:.1f}人</div>
                <div><strong>Δ人口:</strong> {row['delta']:.1f}人</div>
                <div><strong>期待効果:</strong> {row['exp']:.1f}人</div>
                <div><strong>その他:</strong> {row['other']:.1f}人</div>
                <div><strong>率由来合計:</strong> {row['rate_total']:.1f}人</div>
            </div>
        </div>
        """
        
        return pn.pane.HTML(html)
    
    @param.depends('forecast_df', 'selected_year', 'selected_metric')
    def map_chart(self):
        """空間的影響の地図"""
        if self.forecast_df.empty:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        filtered_df = self.filtered_data()
        
        if filtered_df.empty or "lat" not in filtered_df.columns or "lon" not in filtered_df.columns:
            return pn.pane.HTML("<p>地図データがありません。</p>")
        
        # イベント発生町丁とその他の町丁を分ける
        event_town_data = filtered_df[filtered_df['is_event_town'] == True]
        other_towns_data = filtered_df[filtered_df['is_event_town'] == False]
        
        fig = go.Figure()
        
        # イベント発生町丁（大きく、赤色で表示）
        if not event_town_data.empty:
            fig.add_trace(go.Scattermapbox(
                lat=event_town_data['lat'],
                lon=event_town_data['lon'],
                mode='markers',
                marker=dict(
                    size=30,
                    color='red',
                    opacity=0.9,
                    line=dict(width=3, color='darkred')
                ),
                text=event_town_data['town'] + '<br>Δ人口: ' + event_town_data['delta'].astype(str) + '<br>【イベント発生地】',
                hovertemplate='%{text}<extra></extra>',
                name='イベント発生地'
            ))
        
        # その他の町丁（Δの値に応じて色とサイズを調整）
        if not other_towns_data.empty:
            colors = ['orange' if x > 0 else 'blue' for x in other_towns_data['delta']]
            sizes = [max(5, min(20, abs(x) / 10)) for x in other_towns_data['delta']]
            
            fig.add_trace(go.Scattermapbox(
                lat=other_towns_data['lat'],
                lon=other_towns_data['lon'],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=other_towns_data['town'] + '<br>Δ人口: ' + other_towns_data['delta'].astype(str),
                hovertemplate='%{text}<extra></extra>',
                name='その他の町丁'
            ))
        
        # 地図の中心をイベント発生町丁に設定
        if not event_town_data.empty:
            center_lat = event_town_data['lat'].iloc[0]
            center_lon = event_town_data['lon'].iloc[0]
        else:
            center_lat = filtered_df['lat'].mean()
            center_lon = filtered_df['lon'].mean()
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            height=600,
            title=f"{self.selected_year}年の空間的影響分布（赤: イベント発生地、オレンジ: 正の影響、青: 負の影響）"
        )
        
        return pn.pane.Plotly(fig)
    
    @param.depends('forecast_df', 'selected_year', 'selected_metric')
    def histogram_chart(self):
        """分布ヒストグラム"""
        if self.forecast_df.empty:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        filtered_df = self.filtered_data()
        
        if filtered_df.empty:
            return pn.pane.HTML("<p>フィルタリング結果がありません。</p>")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered_df[self.selected_metric],
            nbinsx=30,
            name=self.METRIC_OPTIONS[self.selected_metric],
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{self.selected_year}年の{self.METRIC_OPTIONS[self.selected_metric]}分布",
            xaxis_title=self.METRIC_OPTIONS[self.selected_metric],
            yaxis_title="町丁数",
            height=400
        )
        
        return pn.pane.Plotly(fig)
    
    @param.depends('forecast_df', 'selected_year', 'search_term', 'sort_option')
    def data_table(self):
        """データテーブル"""
        if self.forecast_df.empty:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        filtered_df = self.filtered_data()
        
        if filtered_df.empty:
            return pn.pane.HTML("<p>フィルタリング結果がありません。</p>")
        
        # 表示する列を選択
        display_columns = ["town", "pop", "delta", "exp", "macro", "inertia", "other", "rate_total", "is_event_town"]
        
        # 重心データがある場合は追加
        if "lat" in filtered_df.columns and "lon" in filtered_df.columns:
            display_columns.extend(["lat", "lon"])
        
        # 列名を日本語に変更
        column_mapping = {
            "town": "町丁",
            "pop": "人口",
            "delta": "Δ人口",
            "exp": "期待効果",
            "macro": "マクロ",
            "inertia": "慣性",
            "other": "その他",
            "rate_total": "率由来合計",
            "is_event_town": "イベント発生地",
            "lat": "緯度",
            "lon": "経度"
        }
        
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.rename(columns=column_mapping)
        
        # 数値列を丸める
        numeric_columns = ["人口", "Δ人口", "期待効果", "マクロ", "慣性", "その他", "率由来合計"]
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(1)
        
        return pn.widgets.Tabulator(
            display_df,
            pagination='remote',
            page_size=20,
            sizing_mode="stretch_width"
        )
    
    def view(self):
        """メインビュー"""
        # コントロール
        controls = pn.Column(
            pn.pane.HTML("<h2>🌍 全地域表示 - リアルタイム予測</h2>"),
            pn.pane.HTML("<h3>🎯 イベント設定</h3>"),
            pn.Param(self, parameters=['event_town', 'event_type_display']),
            pn.widgets.Button(name="🚀 全地域予測実行", button_type="primary"),
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>🔍 フィルタリング・表示設定</h3>"),
            pn.Param(self, parameters=['selected_year', 'selected_metric', 'search_term', 'sort_option']),
            width=300
        )
        
        # メインコンテンツ
        main_content = pn.Column(
            self.statistics,
            pn.pane.HTML("<hr>"),
            self.event_town_details,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML(f"<h3>📊 {self.selected_year}年の{self.METRIC_OPTIONS[self.selected_metric]}ランキング</h3>"),
            self.data_table,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>🗺️ 空間的影響の可視化</h3>"),
            self.map_chart,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>📈 分布ヒストグラム</h3>"),
            self.histogram_chart,
            width=1000
        )
        
        # エラーメッセージ
        if self.error_message:
            error_pane = pn.pane.Alert(self.error_message, alert_type="danger")
            main_content.insert(0, error_pane)
        
        # ローディング表示
        if self.loading:
            loading_pane = pn.pane.HTML("<div style='text-align: center; padding: 20px;'><h3>🔄 全地域予測を実行中...</h3></div>")
            main_content.insert(0, loading_pane)
        
        # ボタンイベント
        def on_button_click(event):
            self.run_prediction()
        
        controls[3].on_click(on_button_click)
        
        return pn.Row(controls, main_content, sizing_mode="stretch_width")

def create_all_towns_prediction(towns):
    """全地域予測コンポーネントを作成"""
    return AllTownsPrediction(towns)
