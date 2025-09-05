# -*- coding: utf-8 -*-
"""
インタラクティブ人口予測ダッシュボード
町丁とイベントをその場で選択・設定してリアルタイムで結果を表示
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
    page_title="インタラクティブ人口予測ダッシュボード", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏘️ インタラクティブ人口予測ダッシュボード")
st.markdown("町丁とイベントを選択して、リアルタイムで人口予測を実行できます")
st.markdown("---")

# メタデータの読み込み
@st.cache_data
def load_metadata():
    """利用可能な町丁と年のリストを取得"""
    try:
        features_path = Path("../../data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../data/processed/features_panel.csv")
        
        if not features_path.exists():
            st.error(f"features_panel.csv が見つかりません: {features_path}")
            return [], []
        
        df = pd.read_csv(features_path, usecols=["town", "year"]).drop_duplicates()
        towns = sorted(df["town"].unique().tolist())
        years = sorted(df["year"].unique().tolist())
        
        return towns, years
    except Exception as e:
        st.error(f"メタデータの読み込みに失敗しました: {e}")
        return [], []

towns, years = load_metadata()

if not towns or not years:
    st.error("メタデータの読み込みに失敗しました。")
    st.stop()

# セッション状態の初期化
if "events" not in st.session_state:
    st.session_state.events = []

# デフォルトのサンプルイベントを追加（初回のみ）
if "sample_added" not in st.session_state:
    st.session_state.sample_added = True
    # サンプルイベントを追加
    sample_event = {
        "year_offset": 0,
        "event_type": "housing",
        "effect_direction": "increase",
        "confidence": 0.8,
        "intensity": 0.6,
        "lag_t": 1,
        "lag_t1": 1,
        "note": "サンプル住宅開発イベント"
    }
    st.session_state.events.append(sample_event)

# サイドバー: シナリオ設定
st.sidebar.header("🎯 シナリオ設定")

# クイックスタート
st.sidebar.subheader("🚀 クイックスタート")
if st.sidebar.button("📋 サンプルシナリオを読み込み", type="secondary"):
    # サンプルシナリオをクリアして追加
    st.session_state.events = []
    sample_events = [
        {
            "year_offset": 0,
            "event_type": "housing",
            "effect_direction": "increase",
            "confidence": 0.8,
            "intensity": 0.6,
            "lag_t": 1,
            "lag_t1": 1,
            "note": "住宅開発プロジェクト"
        },
        {
            "year_offset": 1,
            "event_type": "commercial",
            "effect_direction": "increase",
            "confidence": 0.7,
            "intensity": 0.5,
            "lag_t": 1,
            "lag_t1": 1,
            "note": "商業施設開業"
        }
    ]
    st.session_state.events.extend(sample_events)
    st.sidebar.success("サンプルシナリオを読み込みました！")
    st.rerun()

if st.sidebar.button("🧹 シナリオをクリア", type="secondary"):
    st.session_state.events = []
    st.sidebar.success("シナリオをクリアしました！")
    st.rerun()

# 基本設定
st.sidebar.subheader("基本設定")
town = st.sidebar.selectbox("町丁", towns, index=0, help="予測対象の町丁を選択")
base_year = st.sidebar.selectbox("基準年", years, index=len(years)-1, help="予測の基準となる年")
horizons = st.sidebar.multiselect("予測期間", [1, 2, 3], default=[1, 2, 3], help="何年先まで予測するか")

# イベント設定
st.sidebar.subheader("📅 イベント設定")
st.sidebar.markdown("**イベントを追加**")

col1, col2 = st.sidebar.columns(2)
with col1:
    etype = st.selectbox("タイプ", 
        ["housing", "commercial", "transit", "policy_boundary", 
         "public_edu_medical", "employment", "disaster"],
        help="イベントの種類"
    )
    edir = st.selectbox("方向", ["increase", "decrease"], help="人口への影響方向")
    yoff = st.slider("年オフセット", 0, 3, 0, help="基準年から何年後に発生するか")

with col2:
    conf = st.slider("信頼度", 0.0, 1.0, 1.0, 0.1, help="イベント発生の確実性")
    inten = st.slider("強度", 0.0, 1.0, 1.0, 0.1, help="イベントの影響の強さ")
    lag_t = st.checkbox("lag_t (当年効果)", value=True, help="当年に効果が現れるか")
    lag_t1 = st.checkbox("lag_t1 (翌年効果)", value=True, help="翌年に効果が現れるか")

note = st.sidebar.text_input("備考", placeholder="イベントの詳細説明（任意）")

if st.sidebar.button("➕ イベント追加", type="primary"):
    new_event = {
        "year_offset": yoff,
        "event_type": etype,
        "effect_direction": edir,
        "confidence": conf,
        "intensity": inten,
        "lag_t": int(lag_t),
        "lag_t1": int(lag_t1),
        "note": note
    }
    st.session_state.events.append(new_event)
    st.sidebar.success(f"イベントを追加しました: {etype} ({edir})")
    st.rerun()

# イベント一覧
events = st.session_state.get("events", [])
if events:
    st.sidebar.subheader("📝 追加済みイベント")
    for i, event in enumerate(events):
        with st.sidebar.container():
            st.write(f"**{i+1}.** {event['event_type']} ({event['effect_direction']})")
            st.write(f"   年: +{event['year_offset']}, 信頼度: {event['confidence']:.1f}, 強度: {event['intensity']:.1f}")
            if event['note']:
                st.write(f"   備考: {event['note']}")
            
            if st.button(f"🗑️ 削除", key=f"del_{i}"):
                events.pop(i)
                st.session_state.events = events
                st.rerun()
    
    # クリアボタン
    if st.sidebar.button("🧹 Clear events"):
        st.session_state.events = []
        st.rerun()

# マクロ設定
st.sidebar.subheader("🌍 マクロ変数")
st.sidebar.markdown("**外国人人口成長率 (%)**")
st.sidebar.markdown("*未入力の場合はNaNのまま（木モデル任せ）*")

f_h1 = st.sidebar.number_input("h1 growth %", value=0.00, step=0.01, format="%.2f")
f_h2 = st.sidebar.number_input("h2 growth %", value=0.00, step=0.01, format="%.2f")
f_h3 = st.sidebar.number_input("h3 growth %", value=0.00, step=0.01, format="%.2f")

# 手動加算
st.sidebar.subheader("🏭 手動加算")
st.sidebar.markdown("**直接的な人口変化 (人)**")
st.sidebar.markdown("*「工場で+100人」などの直接効果*")

m_h1 = st.sidebar.number_input("h1 +人", value=0, step=10)
m_h2 = st.sidebar.number_input("h2 +人", value=0, step=10)
m_h3 = st.sidebar.number_input("h3 +人", value=0, step=10)

# シナリオクリア
if st.sidebar.button("🗑️ シナリオクリア", type="secondary"):
    st.session_state.events = []
    st.rerun()

# メインエリア: 現在のシナリオ状況
st.header("📋 現在のシナリオ状況")

# シナリオ概要を表示
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("選択町丁", town)
with col2:
    st.metric("基準年", base_year)
with col3:
    st.metric("予測期間", f"{min(horizons)}-{max(horizons)}年先")

# イベント数と手動加算の表示
col1, col2 = st.columns(2)
with col1:
    st.metric("設定済みイベント数", len(events))
with col2:
    manual_total = sum([m_h1, m_h2, m_h3])
    st.metric("手動加算合計", f"{manual_total:+d}人")

# 現在のイベント一覧
if events:
    st.subheader("📝 設定済みイベント")
    for i, event in enumerate(events):
        with st.expander(f"イベント {i+1}: {event['event_type']} ({event['effect_direction']})"):
            st.write(f"**年オフセット**: +{event['year_offset']}年")
            st.write(f"**信頼度**: {event['confidence']:.1f}")
            st.write(f"**強度**: {event['intensity']:.1f}")
            st.write(f"**ラグ効果**: 当年={bool(event['lag_t'])}, 翌年={bool(event['lag_t1'])}")
            if event['note']:
                st.write(f"**備考**: {event['note']}")
else:
    st.info("ℹ️ イベントが設定されていません。サイドバーでイベントを追加するか、クイックスタートボタンを使用してください。")

# 手動加算の表示
if any([m_h1, m_h2, m_h3]):
    st.subheader("🏭 手動加算設定")
    manual_data = {
        "年": [f"h{h}" for h in [1, 2, 3]],
        "加算値": [m_h1, m_h2, m_h3]
    }
    st.dataframe(pd.DataFrame(manual_data), use_container_width=True)

st.markdown("---")

# 予測実行セクション
st.header("📊 予測実行")

# 実行ボタン
if st.button("🚀 予測実行", type="primary", use_container_width=True):
    try:
        # シナリオ作成
        scenario = {
            "town": town,
            "base_year": base_year,
            "horizons": horizons,
            "events": events,
            "macros": {"foreign_population_growth_pct": {"h1": f_h1/100.0, "h2": f_h2/100.0, "h3": f_h3/100.0}},
            "manual_delta": {"h1": m_h1, "h2": m_h2, "h3": m_h3}
        }
        
        # バリデーション
        if len(scenario["events"]) == 0 and all(v == 0 for v in scenario["manual_delta"].values()):
            st.warning("⚠️ イベントも手動加算も設定されていません。")
            st.info("💡 **ヒント**: 以下のいずれかを設定してください：")
            st.info("1. **イベントを追加**: サイドバーでイベントタイプ、方向、年などを設定して「➕ イベント追加」をクリック")
            st.info("2. **手動加算**: サイドバーで「手動加算」セクションで直接的な人口変化を設定")
            st.info("3. **サンプルイベント**: 既にサンプルの住宅開発イベントが追加されています")
            
            # サンプルイベントを追加するボタン
            if st.button("🎯 サンプルイベントを追加", type="primary"):
                sample_events = [
                    {
                        "year_offset": 0,
                        "event_type": "housing",
                        "effect_direction": "increase",
                        "confidence": 0.8,
                        "intensity": 0.6,
                        "lag_t": 1,
                        "lag_t1": 1,
                        "note": "住宅開発プロジェクト"
                    },
                    {
                        "year_offset": 1,
                        "event_type": "commercial",
                        "effect_direction": "increase",
                        "confidence": 0.7,
                        "intensity": 0.5,
                        "lag_t": 1,
                        "lag_t1": 1,
                        "note": "商業施設開業"
                    }
                ]
                st.session_state.events.extend(sample_events)
                st.success("サンプルイベントを追加しました！")
                st.rerun()
            
            st.stop()
        
        # 予測実行（CLIと同じフロー）
        with st.spinner("予測を実行中..."):
            # 出力ディレクトリの設定（CLIと同じパス）
            output_dir = Path("../../data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # デバッグ用：パスを表示
            st.info(f"出力ディレクトリ: {output_dir.absolute()}")
            
            # Step 1: 将来イベント行列の生成
            st.info("Step 1: 将来イベント行列を生成中...")
            future_events = scenario_to_events(scenario)
            future_events.to_csv(output_dir / "l5_future_events.csv", index=False)
            
            # Step 2: 基準年データの準備
            st.info("Step 2: 基準年データを準備中...")
            baseline = prepare_baseline(town, base_year)
            baseline.to_csv(output_dir / "l5_baseline.csv", index=False)
            
            # Step 3: 将来特徴の構築
            st.info("Step 3: 将来特徴を構築中...")
            future_features = build_future_features(baseline, future_events, scenario)
            future_features.to_csv(output_dir / "l5_future_features.csv", index=False)
            
            # Step 4: 人口予測の実行
            st.info("Step 4: 人口予測を実行中...")
            base_population = baseline["pop_total"].iloc[0] if "pop_total" in baseline.columns else 0.0
            if pd.isna(base_population):
                st.warning("ベース人口が不明のため、0を使用します")
                base_population = 0.0
            
            result = forecast_population(town, base_year, horizons, base_population)
        
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

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        st.exception(e)

# ヘルプ
with st.expander("❓ ヘルプ"):
    st.markdown("""
    ### 使用方法
    
    1. **基本設定**: 町丁、基準年、予測期間を選択
    2. **イベント設定**: 人口に影響するイベントを追加
    3. **マクロ変数**: 外国人人口の成長率を設定（任意）
    4. **手動加算**: 直接的な人口変化を設定（任意）
    5. **予測実行**: 「予測実行」ボタンをクリック
    
    ### イベントの説明
    
    - **housing**: 住宅開発
    - **commercial**: 商業施設
    - **transit**: 交通インフラ
    - **policy_boundary**: 政策境界変更
    - **public_edu_medical**: 公共・教育・医療施設
    - **employment**: 雇用創出
    - **disaster**: 災害
    
    ### 注意事項
    
    - policy_boundary と transit を同年同町に入れた場合、transit が無効化されます
    - 未入力のマクロ変数は NaN のまま（木モデル任せ）
    - 手動加算は期待効果（exp）に直接加算されます
    """)

# フッター
st.markdown("---")
st.markdown("**熊本町丁人口予測システム** - インタラクティブダッシュボード")
