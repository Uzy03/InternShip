# -*- coding: utf-8 -*-
"""
単一町丁予測機能 (Panel版)
町丁、イベントタイプ、効果方向を選択して単一町丁の人口予測を実行
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

def run_baseline_prediction(scenario, town):
    """ベースライン予測（イベントなし）を実行"""
    # 出力ディレクトリの設定
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 将来イベント行列の生成（空のイベント）
    future_events = scenario_to_events(scenario)
    print(f"🔍 ベースライン future_events shape: {future_events.shape}")
    print(f"🔍 ベースライン future_events head:\n{future_events.head()}")
    future_events.to_csv(output_dir / "l5_future_events_baseline.csv", index=False)
    
    # Step 2: 基準年データの準備
    baseline = prepare_baseline(town, 2025)
    print(f"🔍 ベースライン baseline shape: {baseline.shape}")
    print(f"🔍 ベースライン baseline head:\n{baseline.head()}")
    baseline.to_csv(output_dir / "l5_baseline_baseline.csv", index=False)
    
    # Step 3: 将来特徴の構築
    future_features = build_future_features(baseline, future_events, scenario)
    print(f"🔍 ベースライン future_features shape: {future_features.shape}")
    print(f"🔍 ベースライン future_features head:\n{future_features.head()}")
    future_features.to_csv(output_dir / "l5_future_features_baseline.csv", index=False)
    
    # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
    future_events.to_csv(Path("../../data/processed/l5_future_events_baseline.csv"), index=False)
    baseline.to_csv(Path("../../data/processed/l5_baseline_baseline.csv"), index=False)
    future_features.to_csv(Path("../../data/processed/l5_future_features_baseline.csv"), index=False)
    
    # Step 4: 人口予測の実行
    base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
    if pd.isna(base_population):
        base_population = 0.0
    
    # 手動加算パラメータ（ベースラインは0）
    manual_add = {1: 0.0, 2: 0.0, 3: 0.0}
    
    # ベースライン予測では専用のファイルを使用
    # ベースライン用のファイルを標準名にコピー（メイン予測で上書きされる前に）
    baseline_features_path = output_dir / "l5_future_features_baseline.csv"
    standard_features_path = output_dir / "l5_future_features.csv"
    if baseline_features_path.exists():
        import shutil
        shutil.copy2(baseline_features_path, standard_features_path)
        # Layer5の標準パスにもコピー
        shutil.copy2(baseline_features_path, Path("../../data/processed/l5_future_features.csv"))
    
    result = forecast_population(town, 2025, [1, 2, 3], base_population, str(output_dir), manual_add)
    print(f"🔍 ベースライン予測のforecast_population結果: {result}")
    print(f"🔍 ベースライン予測の人口値: h1={result.get('h1', 'N/A')}, h2={result.get('h2', 'N/A')}, h3={result.get('h3', 'N/A')}")
    
    return result

class SingleTownPrediction(param.Parameterized):
    """単一町丁予測のPanelクラス"""
    
    # パラメータ
    town = param.Selector(default="", objects=[], doc="町丁")
    event_type_display = param.Selector(default="", objects=[], doc="イベントタイプ")
    
    # 内部状態
    result = param.Dict(default={}, doc="予測結果")
    baseline_result = param.Dict(default={}, doc="ベースライン予測結果")
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
        
        # パラメータの初期化
        self.param.town.objects = towns
        self.param.town.default = towns[0] if towns else ""
        self.param.event_type_display.objects = list(self.EVENT_TYPE_MAPPING.values())
        self.param.event_type_display.default = list(self.EVENT_TYPE_MAPPING.values())[0] if self.EVENT_TYPE_MAPPING else ""
        
        # イベントの詳細説明
        self.EVENT_DESCRIPTIONS = {
            "housing": {
                "increase": "新規のマンション・アパート・戸建てが供給される（分譲マンション竣工、団地入居、宅地造成後の入居）",
                "decrease": "住宅の解体・用途転用・空き家化などで実質的な供給が減る（一斉解体、老朽化で未利用化、住宅→駐車場転用）"
            },
            "commercial": {
                "increase": "店舗・モールなど商業集積が拡大（大型商業施設開業、スーパー新設、商店集積）",
                "decrease": "商業施設の撤退・閉鎖で集積が減少（店舗閉鎖、モール撤退）"
            },
            "transit": {
                "increase": "新駅・増便・道路整備などでアクセスが改善（新駅開業、バス増便、IC供用）",
                "decrease": "路線撤退・減便等でアクセスが悪化（バス減便、路線廃止）"
            },
            "policy_boundary": {
                "increase": "政策境界の変更により区域が拡大",
                "decrease": "政策境界の変更により区域が縮小"
            },
            "public_edu_medical": {
                "increase": "学校・病院など公共系施設が増える（小中学校新設、病院開設、大学キャンパス誘致）",
                "decrease": "統廃合・閉鎖で公共系施設が減る（学校統廃合、病院閉鎖）"
            },
            "employment": {
                "increase": "新規雇用創出・大規模採用（工場稼働、物流拠点開設、事業拡張）",
                "decrease": "事業所撤退・解雇で雇用が減る（事業所閉鎖、工場撤退）"
            },
            "disaster": {
                "increase": "災害発生や被害拡大により魅力が低下（洪水・地震被害、土砂災害）",
                "decrease": "復旧・治水・耐震化等で被害リスクが下がる（堤防整備、河川改修、耐震化）"
            }
        }
        
        # 効果の強さ
        self.EFFECT_STRENGTH = {
            "housing": {"increase": "弱", "decrease": "強"},
            "commercial": {"increase": "強", "decrease": "中"},
            "transit": {"increase": "弱", "decrease": "中"},
            "policy_boundary": {"increase": "中", "decrease": "中"},
            "public_edu_medical": {"increase": "なし", "decrease": "なし"},
            "employment": {"increase": "中", "decrease": "中"},
            "disaster": {"increase": "中", "decrease": "中"}
        }
    
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
    
    @param.depends('town', 'event_type_display')
    def scenario_info(self):
        """シナリオ情報の表示"""
        if not self.town or not self.event_type_display:
            return pn.pane.HTML("<p>町丁とイベントタイプを選択してください。</p>")
        
        event_type, effect_direction = self.get_event_type_and_direction()
        
        # シナリオ詳細
        scenario_details = {
            "町丁": self.town,
            "基準年": 2025,
            "予測期間": "1-3年先",
            "イベントタイプ": self.event_type_display,
            "年オフセット": "1年（翌年）",
            "信頼度": "1.0",
            "強度": "1.0",
            "手動加算": "h1=0人, h2=0人, h3=0人（固定値）",
            "強度設定": "学習された強度（自動最適化）"
        }
        
        # イベントの詳細説明
        event_description = self.EVENT_DESCRIPTIONS[event_type][effect_direction]
        effect_strength = self.EFFECT_STRENGTH[event_type][effect_direction]
        
        html = f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3>📋 現在のシナリオ</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px;">
                <div><strong>選択町丁:</strong> {self.town}</div>
                <div><strong>イベントタイプ:</strong> {self.event_type_display}</div>
            </div>
            <h4>📝 シナリオ詳細</h4>
            <ul>
        """
        
        for key, value in scenario_details.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        
        html += f"""
            </ul>
            <h4>📋 選択されたイベントの詳細</h4>
            <p><strong>説明:</strong> {event_description}</p>
            <p><strong>推定効果:</strong> {effect_strength}</p>
        </div>
        """
        
        return pn.pane.HTML(html)
    
    def run_prediction(self):
        """予測を実行"""
        if not self.town or not self.event_type_display:
            self.error_message = "町丁とイベントタイプを選択してください。"
            return
        
        self.loading = True
        self.error_message = ""
        
        try:
            event_type, effect_direction = self.get_event_type_and_direction()
            
            # ベースライン予測（イベントなし）の実行
            # デバッグのため毎回実行（キャッシュ機能を一時的に無効化）
            try:
                baseline_scenario = {
                    "town": self.town,
                    "base_year": 2025,
                    "horizons": [1, 2, 3],
                    "events": [],
                    "macros": {},
                    "manual_delta": {"h1": 0, "h2": 0, "h3": 0}
                }
                
                print(f"🔍 ベースラインシナリオ: {baseline_scenario}")
                self.baseline_result = run_baseline_prediction(baseline_scenario, self.town)
                print(f"🔍 ベースライン予測結果: {self.baseline_result}")
                
            except Exception as e:
                print(f"ベースライン予測に失敗しました: {e}")
                self.baseline_result = {}
            
            # イベントありのシナリオ作成
            try:
                generator = LearnedScenarioGenerator()
                scenario = generator.create_learned_scenario_with_yearly_intensity(self.town, event_type, effect_direction)
                scenario["manual_delta"] = {"h1": 0.0, "h2": 0.0, "h3": 0.0}
                
            except Exception as e:
                print(f"年次別強度の取得に失敗しました: {e}。デフォルト強度を使用します。")
                scenario = {
                    "town": self.town,
                    "base_year": 2025,
                    "horizons": [1, 2, 3],
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
            
            # 予測実行
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: 将来イベント行列の生成
            future_events = scenario_to_events(scenario)
            future_events.to_csv(output_dir / "l5_future_events.csv", index=False)
            
            # Layer5の標準パスにも保存
            future_events.to_csv(Path("../../data/processed/l5_future_events.csv"), index=False)
            
            # Step 2: 基準年データの準備
            baseline = prepare_baseline(self.town, 2025)
            baseline.to_csv(output_dir / "l5_baseline.csv", index=False)
            
            # Layer5の標準パスにも保存
            baseline.to_csv(Path("../../data/processed/l5_baseline.csv"), index=False)
            
            # Step 3: 将来特徴の構築
            future_features = build_future_features(baseline, future_events, scenario)
            future_features.to_csv(output_dir / "l5_future_features.csv", index=False)
            
            # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
            future_features.to_csv(Path("../../data/processed/l5_future_features.csv"), index=False)
            
            # Step 4: 人口予測の実行
            base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
            if pd.isna(base_population):
                base_population = 0.0
            
            manual_add = {1: 0.0, 2: 0.0, 3: 0.0}
            self.result = forecast_population(self.town, 2025, [1, 2, 3], base_population, str(output_dir), manual_add)
            print(f"🔍 メイン予測のforecast_population結果: {self.result}")
            
            # ベースライン予測の結果を復元（メイン予測で上書きされた可能性があるため）
            if hasattr(self, 'baseline_result') and self.baseline_result:
                print(f"🔍 ベースライン予測結果を復元: {self.baseline_result}")
                # ベースライン用のファイルを再度コピーして復元
                baseline_features_path = output_dir / "l5_future_features_baseline.csv"
                if baseline_features_path.exists():
                    import shutil
                    shutil.copy2(baseline_features_path, standard_features_path)
                    shutil.copy2(baseline_features_path, Path("../../data/processed/l5_future_features.csv"))
            
        except Exception as e:
            self.error_message = f"エラーが発生しました: {str(e)}"
            print(f"予測実行エラー: {e}")
        
        finally:
            self.loading = False
    
    @param.depends('result', 'baseline_result')
    def population_chart(self):
        """人口予測の折れ線グラフ"""
        if not self.result:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        path_df = pd.DataFrame(self.result["path"])
        
        fig = go.Figure()
        
        if self.baseline_result:
            # ベースライン（イベントなし）の人口パス
            baseline_path_df = pd.DataFrame(self.baseline_result["path"])
            
            fig.add_trace(go.Scatter(
                x=baseline_path_df["year"],
                y=baseline_path_df["pop_hat"],
                mode='lines+markers',
                name='イベントなし（ベースライン）',
                line=dict(color='#2E8B57', width=3, dash='dash'),
                marker=dict(size=10, color='#2E8B57')
            ))
            
            # ベースラインの信頼区間
            if "pi95_pop" in baseline_path_df.columns:
                lower_baseline = [p[0] if isinstance(p, list) else p for p in baseline_path_df["pi95_pop"]]
                upper_baseline = [p[1] if isinstance(p, list) else p for p in baseline_path_df["pi95_pop"]]
                
                fig.add_trace(go.Scatter(
                    x=baseline_path_df["year"].tolist() + baseline_path_df["year"].tolist()[::-1],
                    y=upper_baseline + lower_baseline[::-1],
                    fill='tonexty',
                    fillcolor='rgba(46, 139, 87, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='ベースライン95%信頼区間',
                    showlegend=True
                ))
        
        # イベントありの人口パス
        fig.add_trace(go.Scatter(
            x=path_df["year"],
            y=path_df["pop_hat"],
            mode='lines+markers',
            name='イベントあり',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10, color='#1f77b4')
        ))
        
        # イベントありの信頼区間
        if "pi95_pop" in path_df.columns:
            lower = [p[0] if isinstance(p, list) else p for p in path_df["pi95_pop"]]
            upper = [p[1] if isinstance(p, list) else p for p in path_df["pi95_pop"]]
            
            fig.add_trace(go.Scatter(
                x=path_df["year"].tolist() + path_df["year"].tolist()[::-1],
                y=upper + lower[::-1],
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='イベントあり95%信頼区間',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"人口予測パス: {self.result['town']} (基準年: {self.result['base_year']})",
            xaxis_title="年",
            yaxis_title="人口（人）",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        return pn.pane.Plotly(fig)
    
    @param.depends('result', 'baseline_result')
    def delta_chart(self):
        """人口変化量のグラフ"""
        if not self.result:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        path_df = pd.DataFrame(self.result["path"])
        
        fig = go.Figure()
        
        if self.baseline_result:
            # ベースライン（イベントなし）のΔ人口
            baseline_path_df = pd.DataFrame(self.baseline_result["path"])
            
            fig.add_trace(go.Bar(
                x=baseline_path_df["year"],
                y=baseline_path_df["delta_hat"],
                name='イベントなしΔ人口',
                marker_color='#2E8B57',
                opacity=0.7,
                text=[f"{x:+.1f}" for x in baseline_path_df["delta_hat"]],
                textposition='auto'
            ))
        
        # イベントありのΔ人口
        fig.add_trace(go.Bar(
            x=path_df["year"],
            y=path_df["delta_hat"],
            name='イベントありΔ人口',
            marker_color=['#ff7f0e' if x > 0 else '#d62728' for x in path_df["delta_hat"]],
            opacity=0.7,
            text=[f"{x:+.1f}" for x in path_df["delta_hat"]],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="年別人口変化量",
            xaxis_title="年",
            yaxis_title="人口変化量（人）",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        return pn.pane.Plotly(fig)
    
    @param.depends('result')
    def contribution_chart(self):
        """寄与分解のグラフ"""
        if not self.result:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        path_df = pd.DataFrame(self.result["path"])
        
        # 寄与データを準備
        contrib_data = []
        for _, row in path_df.iterrows():
            contrib = row["contrib"]
            contrib_data.append({
                "year": row["year"],
                "exp": contrib.get("exp", 0),
                "macro": contrib.get("macro", 0),
                "inertia": contrib.get("inertia", 0),
                "other": contrib.get("other", 0),
                "delta_hat": row["delta_hat"]
            })
        
        contrib_df = pd.DataFrame(contrib_data)
        
        # 寄与分解の積み上げバー
        fig = go.Figure()
        
        colors = {
            "exp": "#FF6B6B",      # 赤（期待効果）
            "macro": "#4ECDC4",    # 青緑（マクロ）
            "inertia": "#45B7D1",  # 青（慣性）
            "other": "#96CEB4"     # 緑（その他）
        }
        
        for col in ["exp", "macro", "inertia", "other"]:
            fig.add_trace(go.Bar(
                x=contrib_df["year"],
                y=contrib_df[col],
                name=col,
                marker_color=colors[col],
                opacity=0.8
            ))
        
        fig.update_layout(
            title="寄与分解（積み上げバー）",
            xaxis_title="年",
            yaxis_title="寄与（人）",
            barmode='relative',
            template="plotly_white",
            height=500
        )
        
        return pn.pane.Plotly(fig)
    
    @param.depends('result')
    def data_table(self):
        """詳細データテーブル"""
        if not self.result:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        path_df = pd.DataFrame(self.result["path"])
        
        # 表示用データフレームの準備
        display_df = path_df.copy()
        display_df["人口"] = display_df["pop_hat"].round(1)
        display_df["Δ人口"] = display_df["delta_hat"].round(1)
        display_df["期待効果"] = display_df["contrib"].apply(lambda x: x.get("exp", 0)).round(1)
        display_df["マクロ"] = display_df["contrib"].apply(lambda x: x.get("macro", 0)).round(1)
        display_df["慣性"] = display_df["contrib"].apply(lambda x: x.get("inertia", 0)).round(1)
        display_df["その他"] = display_df["contrib"].apply(lambda x: x.get("other", 0)).round(1)
        
        # 列名を日本語に変更
        display_df = display_df.rename(columns={
            "year": "年",
            "人口": "人口",
            "Δ人口": "Δ人口",
            "期待効果": "期待効果",
            "マクロ": "マクロ",
            "慣性": "慣性",
            "その他": "その他"
        })
        
        display_columns = ["年", "人口", "Δ人口", "期待効果", "マクロ", "慣性", "その他"]
        
        return pn.widgets.Tabulator(
            display_df[display_columns],
            pagination='remote',
            page_size=10,
            sizing_mode="stretch_width"
        )
    
    @param.depends('result', 'baseline_result')
    def summary_metrics(self):
        """サマリー統計"""
        if not self.result:
            return pn.pane.HTML("<p>予測を実行してください。</p>")
        
        path_df = pd.DataFrame(self.result["path"])
        
        # 基本統計
        final_pop = path_df["pop_hat"].iloc[-1]
        initial_pop = path_df["pop_hat"].iloc[0]
        total_change = final_pop - initial_pop
        avg_delta = path_df["delta_hat"].mean()
        max_exp = path_df["contrib"].apply(lambda x: x.get("exp", 0)).max()
        total_exp = path_df["contrib"].apply(lambda x: x.get("exp", 0)).sum()
        
        # ベースライン比較
        comparison_html = ""
        if self.baseline_result:
            baseline_path_df = pd.DataFrame(self.baseline_result["path"])
            final_diff = path_df["pop_hat"].iloc[-1] - baseline_path_df["pop_hat"].iloc[-1]
            max_diff = (path_df["pop_hat"] - baseline_path_df["pop_hat"]).max()
            avg_diff = (path_df["pop_hat"] - baseline_path_df["pop_hat"]).mean()
            final_rate = ((path_df["pop_hat"].iloc[-1] - baseline_path_df["pop_hat"].iloc[-1]) / baseline_path_df["pop_hat"].iloc[-1] * 100)
            
            comparison_html = f"""
            <h4>📊 ベースライン比較</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px;">
                <div><strong>最終年人口差:</strong> {final_diff:.1f}人</div>
                <div><strong>最大人口差:</strong> {max_diff:.1f}人</div>
                <div><strong>平均人口差:</strong> {avg_diff:.1f}人</div>
                <div><strong>最終年効果率:</strong> {final_rate:.2f}%</div>
            </div>
            """
        
        html = f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3>📊 サマリー統計</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px;">
                <div><strong>総人口変化:</strong> {total_change:.1f}人<br><small>{initial_pop:.1f} → {final_pop:.1f}</small></div>
                <div><strong>平均年次変化:</strong> {avg_delta:.1f}人/年</div>
                <div><strong>最大期待効果:</strong> {max_exp:.1f}人</div>
                <div><strong>期待効果合計:</strong> {total_exp:.1f}人</div>
            </div>
            {comparison_html}
        </div>
        """
        
        return pn.pane.HTML(html)
    
    def view(self):
        """メインビュー"""
        # コントロール
        controls = pn.Column(
            pn.pane.HTML("<h2>🏘️ 単一町丁予測</h2>"),
            pn.pane.HTML("<h3>🎯 シナリオ設定</h3>"),
            pn.pane.HTML("<h4>基本設定</h4>"),
            pn.Param(self, parameters=['town', 'event_type_display']),
            pn.pane.HTML("<h4>固定パラメータ</h4>"),
            pn.pane.HTML("""
            <div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>固定設定:</strong><br>
                • 基準年: 2025<br>
                • 予測期間: [1, 2, 3]年先<br>
                • 年オフセット: 1年（翌年）<br>
                • 信頼度: 1.0<br>
                • 強度: 1.0<br>
                • ラグ効果: 当年・翌年両方
            </div>
            """),
            pn.widgets.Button(name="🚀 予測実行", button_type="primary"),
            pn.pane.HTML("<hr>"),
            width=300
        )
        
        # メインコンテンツ
        main_content = pn.Column(
            self.scenario_info,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>📊 予測結果</h3>"),
            self.population_chart,
            pn.pane.HTML("<hr>"),
            self.delta_chart,
            pn.pane.HTML("<hr>"),
            self.contribution_chart,
            pn.pane.HTML("<hr>"),
            pn.pane.HTML("<h3>📋 詳細データ</h3>"),
            self.data_table,
            pn.pane.HTML("<hr>"),
            self.summary_metrics,
            width=800
        )
        
        # エラーメッセージ
        if self.error_message:
            error_pane = pn.pane.Alert(self.error_message, alert_type="danger")
            main_content.insert(0, error_pane)
        
        # ローディング表示
        if self.loading:
            loading_pane = pn.pane.HTML("<div style='text-align: center; padding: 20px;'><h3>🔄 予測を実行中...</h3></div>")
            main_content.insert(0, loading_pane)
        
        # ボタンイベント
        def on_button_click(event):
            self.run_prediction()
        
        controls[6].on_click(on_button_click)
        
        return pn.Row(controls, main_content, sizing_mode="stretch_width")

def create_single_town_prediction(towns):
    """単一町丁予測コンポーネントを作成"""
    return SingleTownPrediction(towns)
