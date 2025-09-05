# -*- coding: utf-8 -*-
"""
シンプル人口予測ダッシュボード
町丁、イベントタイプ、効果方向のみを選択して予測実行
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import os
from typing import Dict, List, Any

# パス設定
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))

# Layer5モジュールのインポート
try:
    from scenario_to_events import scenario_to_events
    from prepare_baseline import prepare_baseline
    from build_future_features import build_future_features
    from forecast_service import forecast_population
except ImportError as e:
    st.error(f"Layer5モジュールのインポートに失敗しました: {e}")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="シンプル人口予測ダッシュボード", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏘️ シンプル人口予測ダッシュボード")
st.markdown("町丁、イベントタイプ、効果方向を選択して人口予測を実行")
st.markdown("---")

# メタデータの読み込み
@st.cache_data
def load_metadata():
    """利用可能な町丁のリストを取得"""
    try:
        features_path = Path("../../data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../data/processed/features_panel.csv")
        
        if not features_path.exists():
            st.error(f"features_panel.csv が見つかりません: {features_path}")
            return []
        
        df = pd.read_csv(features_path, usecols=["town"]).drop_duplicates()
        towns = sorted(df["town"].unique().tolist())
        
        return towns
    except Exception as e:
        st.error(f"メタデータの読み込みに失敗しました: {e}")
        return []

towns = load_metadata()

if not towns:
    st.error("メタデータの読み込みに失敗しました。")
    st.stop()

# サイドバー: シナリオ設定
st.sidebar.header("🎯 シナリオ設定")

# 基本設定
st.sidebar.subheader("基本設定")
town = st.sidebar.selectbox("町丁", towns, index=0, help="予測対象の町丁を選択")

# イベント設定
st.sidebar.subheader("イベント設定")
event_type = st.sidebar.selectbox(
    "イベントタイプ", 
    ["housing", "commercial", "transit", "policy_boundary", 
     "public_edu_medical", "employment", "disaster"],
    help="イベントの種類"
)

effect_direction = st.sidebar.selectbox(
    "効果方向", 
    ["increase", "decrease"], 
    help="人口への影響方向"
)

# 固定パラメータ
st.sidebar.subheader("固定パラメータ")
st.sidebar.info("""
**固定設定:**
- 基準年: 2025
- 予測期間: [1, 2, 3]年先
- 年オフセット: 0年（当年）
- 信頼度: 1.0
- 強度: 1.0
- ラグ効果: 当年・翌年両方
- 手動加算: h1=50人, h2=30人, h3=20人
""")

# メインエリア: 現在のシナリオ状況
st.header("📋 現在のシナリオ")

# シナリオ概要を表示
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("選択町丁", town)
with col2:
    st.metric("イベントタイプ", event_type)
with col3:
    st.metric("効果方向", effect_direction)

# シナリオ詳細
st.subheader("📝 シナリオ詳細")
scenario_details = {
    "町丁": town,
    "基準年": 2025,
    "予測期間": "1-3年先",
    "イベントタイプ": event_type,
    "効果方向": effect_direction,
    "年オフセット": "0年（当年）",
    "信頼度": "1.0",
    "強度": "1.0",
    "手動加算": "h1=50人, h2=30人, h3=20人"
}

for key, value in scenario_details.items():
    st.write(f"**{key}**: {value}")

st.markdown("---")

# 予測実行セクション
st.header("📊 予測実行")

# 実行ボタン
if st.button("🚀 予測実行", type="primary", use_container_width=True):
    try:
        # シナリオ作成（固定パラメータ使用）
        scenario = {
            "town": town,
            "base_year": 2025,
            "horizons": [1, 2, 3],
            "events": [{
                "year_offset": 0,
                "event_type": event_type,
                "effect_direction": effect_direction,
                "confidence": 1.0,
                "intensity": 1.0,
                "lag_t": 1,
                "lag_t1": 1,
                "note": f"{event_type} ({effect_direction})"
            }],
            "macros": {},
            "manual_delta": {"h1": 50, "h2": 30, "h3": 20}
        }
        
        # 予測実行（CLIと同じフロー）
        with st.spinner("予測を実行中..."):
            # 出力ディレクトリの設定（ダッシュボード専用）
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: 将来イベント行列の生成
            st.info("Step 1: 将来イベント行列を生成中...")
            future_events = scenario_to_events(scenario)
            future_events.to_csv(output_dir / "l5_future_events.csv", index=False)
            
            # Layer5の標準パスにも保存
            layer5_events_path = Path("../../data/processed/l5_future_events.csv")
            future_events.to_csv(layer5_events_path, index=False)
            
            # Step 2: 基準年データの準備
            st.info("Step 2: 基準年データを準備中...")
            baseline = prepare_baseline(town, 2025)
            baseline.to_csv(output_dir / "l5_baseline.csv", index=False)
            
            # Layer5の標準パスにも保存
            layer5_baseline_path = Path("../../data/processed/l5_baseline.csv")
            baseline.to_csv(layer5_baseline_path, index=False)
            
            # Step 3: 将来特徴の構築
            st.info("Step 3: 将来特徴を構築中...")
            future_features = build_future_features(baseline, future_events, scenario)
            future_features.to_csv(output_dir / "l5_future_features.csv", index=False)
            
            # Layer5の標準パスにも保存（forecast_service.pyが読み込むため）
            layer5_features_path = Path("../../data/processed/l5_future_features.csv")
            future_features.to_csv(layer5_features_path, index=False)
            
            # Step 4: 人口予測の実行
            st.info("Step 4: 人口予測を実行中...")
            base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
            if pd.isna(base_population):
                st.warning("ベース人口が不明のため、0を使用します")
                base_population = 0.0
            
            result = forecast_population(town, 2025, [1, 2, 3], base_population, str(output_dir))
        
        # 結果表示
        st.success("✅ 予測が完了しました！")
        
        # 基本情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("町丁", result["town"])
        with col2:
            st.metric("基準年", result["base_year"])
        with col3:
            st.metric("予測期間", f"{min(result['horizons'])}-{max(result['horizons'])}年先")

        st.markdown("---")

        # データフレーム作成
        path_df = pd.DataFrame(result["path"])

        # 人口予測の折れ線グラフ
        st.subheader("📈 人口予測パス")

        fig_pop = go.Figure()

        # 人口パス（線）
        fig_pop.add_trace(go.Scatter(
            x=path_df["year"],
            y=path_df["pop_hat"],
            mode='lines+markers',
            name='予測人口',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10, color='#1f77b4')
        ))

        # 信頼区間（帯）
        if "pi95_pop" in path_df.columns:
            lower = [p[0] if isinstance(p, list) else p for p in path_df["pi95_pop"]]
            upper = [p[1] if isinstance(p, list) else p for p in path_df["pi95_pop"]]
            
            fig_pop.add_trace(go.Scatter(
                x=path_df["year"].tolist() + path_df["year"].tolist()[::-1],
                y=upper + lower[::-1],
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95%信頼区間',
                showlegend=True
            ))

        fig_pop.update_layout(
            title=f"人口予測パス: {result['town']} (基準年: {result['base_year']})",
            xaxis_title="年",
            yaxis_title="人口（人）",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_pop, use_container_width=True)

        # 人口変化量のグラフ
        st.subheader("📊 人口変化量（Δ人口）")

        fig_delta = go.Figure()

        # Δ人口のバー
        fig_delta.add_trace(go.Bar(
            x=path_df["year"],
            y=path_df["delta_hat"],
            name='Δ人口',
            marker_color=['#ff7f0e' if x > 0 else '#d62728' for x in path_df["delta_hat"]],
            text=[f"{x:+.1f}" for x in path_df["delta_hat"]],
            textposition='auto'
        ))

        # 信頼区間
        if "pi95_delta" in path_df.columns:
            lower_delta = [p[0] if isinstance(p, list) else p for p in path_df["pi95_delta"]]
            upper_delta = [p[1] if isinstance(p, list) else p for p in path_df["pi95_delta"]]
            
            fig_delta.add_trace(go.Scatter(
                x=path_df["year"],
                y=upper_delta,
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-up'),
                name='95%信頼区間上限',
                showlegend=True
            ))
            
            fig_delta.add_trace(go.Scatter(
                x=path_df["year"],
                y=lower_delta,
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-down'),
                name='95%信頼区間下限',
                showlegend=True
            ))

        fig_delta.update_layout(
            title="年別人口変化量",
            xaxis_title="年",
            yaxis_title="人口変化量（人）",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_delta, use_container_width=True)

        # 寄与分解のグラフ
        st.subheader("🥧 寄与分解")

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
        fig_contrib = go.Figure()

        colors = {
            "exp": "#FF6B6B",      # 赤（期待効果）
            "macro": "#4ECDC4",    # 青緑（マクロ）
            "inertia": "#45B7D1",  # 青（慣性）
            "other": "#96CEB4"     # 緑（その他）
        }

        for col in ["exp", "macro", "inertia", "other"]:
            fig_contrib.add_trace(go.Bar(
                x=contrib_df["year"],
                y=contrib_df[col],
                name=col,
                marker_color=colors[col],
                opacity=0.8
            ))

        fig_contrib.update_layout(
            title="寄与分解（積み上げバー）",
            xaxis_title="年",
            yaxis_title="寄与（人）",
            barmode='relative',
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_contrib, use_container_width=True)

        # 年別寄与分解の円グラフ
        st.subheader("🥧 年別寄与分解（円グラフ）")

        selected_year = st.selectbox("年を選択", path_df["year"].tolist())

        year_data = path_df[path_df["year"] == selected_year].iloc[0]
        contrib = year_data["contrib"]

        # 円グラフ用データ
        labels = []
        values = []
        colors_pie = []

        for key, value in contrib.items():
            if abs(value) > 0.1:  # 0に近い値は除外
                labels.append(key)
                values.append(abs(value))
                colors_pie.append(colors.get(key, "#CCCCCC"))

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors_pie,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>(%{value:.1f}人)'
        )])

        fig_pie.update_layout(
            title=f"寄与分解: {selected_year}年",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # 詳細データテーブル
        st.subheader("📋 詳細データ")

        # 表示用データフレームの準備
        display_df = path_df.copy()
        display_df["人口"] = display_df["pop_hat"].round(1)
        display_df["Δ人口"] = display_df["delta_hat"].round(1)
        display_df["期待効果"] = display_df["contrib"].apply(lambda x: x.get("exp", 0)).round(1)
        display_df["マクロ"] = display_df["contrib"].apply(lambda x: x.get("macro", 0)).round(1)
        display_df["慣性"] = display_df["contrib"].apply(lambda x: x.get("inertia", 0)).round(1)
        display_df["その他"] = display_df["contrib"].apply(lambda x: x.get("other", 0)).round(1)
        
        # デバッグファイルの読み込み（内訳透明化）
        try:
            debug_detail_path = Path(f"output/l5_debug_detail_{town.replace(' ', '_')}.csv")
            if debug_detail_path.exists():
                debug_detail_df = pd.read_csv(debug_detail_path)
                
                # 内訳情報を追加
                if not debug_detail_df.empty:
                    # 年でマージ
                    merged_df = display_df.merge(
                        debug_detail_df[["year", "exp_people_from_rate", "exp_people_manual", "exp_people_total"]], 
                        on="year", 
                        how="left"
                    )
                    
                    # 内訳列を追加
                    merged_df["期待効果(率由来)"] = merged_df["exp_people_from_rate"].round(1)
                    merged_df["期待効果(手動)"] = merged_df["exp_people_manual"].round(1)
                    merged_df["期待効果(合計)"] = merged_df["exp_people_total"].round(1)
                    
                    display_df = merged_df
                    
                    # 内訳の表示
                    st.subheader("🔍 期待効果の内訳")
                    st.info("期待効果を「率由来」と「手動人数」に分けて表示")
                    
                    # 内訳のグラフ
                    fig_breakdown = go.Figure()
                    
                    # 率由来の期待効果
                    fig_breakdown.add_trace(go.Bar(
                        x=merged_df["year"],
                        y=merged_df["exp_people_from_rate"],
                        name='期待効果(率由来)',
                        marker_color='#FF6B6B',
                        opacity=0.8
                    ))
                    
                    # 手動の期待効果
                    fig_breakdown.add_trace(go.Bar(
                        x=merged_df["year"],
                        y=merged_df["exp_people_manual"],
                        name='期待効果(手動)',
                        marker_color='#4ECDC4',
                        opacity=0.8
                    ))
                    
                    fig_breakdown.update_layout(
                        title="期待効果の内訳",
                        xaxis_title="年",
                        yaxis_title="寄与（人）",
                        barmode='stack',
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_breakdown, use_container_width=True)
                    
        except Exception as e:
            st.warning(f"デバッグ詳細ファイルの読み込みに失敗しました: {e}")

        # 信頼区間の表示
        if "pi95_pop" in display_df.columns:
            display_df["人口95%CI"] = display_df["pi95_pop"].apply(
                lambda x: f"[{x[0]:.1f}, {x[1]:.1f}]" if isinstance(x, list) else f"[{x:.1f}, {x:.1f}]"
            )

        if "pi95_delta" in display_df.columns:
            display_df["Δ人口95%CI"] = display_df["pi95_delta"].apply(
                lambda x: f"[{x[0]:.1f}, {x[1]:.1f}]" if isinstance(x, list) else f"[{x:.1f}, {x:.1f}]"
            )

        # 表示する列を選択
        display_columns = ["年", "人口", "Δ人口", "期待効果", "マクロ", "慣性", "その他"]
        
        # 内訳列があれば追加
        if "期待効果(率由来)" in display_df.columns:
            display_columns.extend(["期待効果(率由来)", "期待効果(手動)", "期待効果(合計)"])
        
        if "人口95%CI" in display_df.columns:
            display_columns.append("人口95%CI")
        if "Δ人口95%CI" in display_df.columns:
            display_columns.append("Δ人口95%CI")

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

        st.dataframe(display_df[display_columns], use_container_width=True)

        # サマリー統計
        st.subheader("📊 サマリー統計")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            final_pop = path_df["pop_hat"].iloc[-1]
            initial_pop = path_df["pop_hat"].iloc[0]
            total_change = final_pop - initial_pop
            st.metric(
                "総人口変化",
                f"{total_change:+.1f}人",
                f"{initial_pop:.1f} → {final_pop:.1f}"
            )

        with col2:
            avg_delta = path_df["delta_hat"].mean()
            st.metric(
                "平均年次変化",
                f"{avg_delta:+.1f}人/年"
            )

        with col3:
            max_exp = path_df["contrib"].apply(lambda x: x.get("exp", 0)).max()
            st.metric(
                "最大期待効果",
                f"{max_exp:+.1f}人"
            )

        with col4:
            total_exp = path_df["contrib"].apply(lambda x: x.get("exp", 0)).sum()
            st.metric(
                "期待効果合計",
                f"{total_exp:+.1f}人"
            )
        
        # 内訳サマリー（デバッグファイルがある場合）
        try:
            debug_detail_path = Path(f"output/l5_debug_detail_{town.replace(' ', '_')}.csv")
            if debug_detail_path.exists():
                debug_detail_df = pd.read_csv(debug_detail_path)
                if not debug_detail_df.empty:
                    st.subheader("🔍 期待効果内訳サマリー")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_rate = debug_detail_df["exp_people_from_rate"].sum()
                        st.metric(
                            "率由来合計",
                            f"{total_rate:+.1f}人"
                        )
                    
                    with col2:
                        total_manual = debug_detail_df["exp_people_manual"].sum()
                        st.metric(
                            "手動合計",
                            f"{total_manual:+.1f}人"
                        )
                    
                    with col3:
                        total_combined = debug_detail_df["exp_people_total"].sum()
                        st.metric(
                            "合計",
                            f"{total_combined:+.1f}人"
                        )
                        
        except Exception as e:
            pass  # デバッグファイルがない場合は無視

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        st.exception(e)

# ヘルプ
with st.expander("❓ ヘルプ"):
    st.markdown("""
    ### 使用方法
    
    1. **町丁選択**: サイドバーで予測対象の町丁を選択
    2. **イベント設定**: イベントタイプと効果方向を選択
    3. **予測実行**: 「予測実行」ボタンをクリック
    
    ### イベントタイプの説明
    
    - **housing**: 住宅開発
    - **commercial**: 商業施設
    - **transit**: 交通インフラ
    - **policy_boundary**: 政策境界変更
    - **public_edu_medical**: 公共・教育・医療施設
    - **employment**: 雇用創出
    - **disaster**: 災害
    
    ### 固定パラメータ
    
    - 基準年: 2025年
    - 予測期間: 1-3年先
    - 年オフセット: 0年（当年）
    - 信頼度: 1.0
    - 強度: 1.0
    - 手動加算: h1=50人, h2=30人, h3=20人
    """)

# フッター
st.markdown("---")
st.markdown("**熊本町丁人口予測システム** - シンプルダッシュボード")
