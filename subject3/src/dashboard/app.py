"""
Streamlit ダッシュボードアプリ
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os

# パスを追加
sys.path.append(os.path.dirname(__file__))

from schema import Scenario, ScenarioEvent
from service import run_scenario, load_metadata, check_dependencies
from components import (
    plot_population_path, 
    plot_contrib_bars, 
    plot_contribution_pie,
    create_summary_cards,
    display_summary_cards
)

# ページ設定
st.set_page_config(
    page_title="Kumamoto Town Forecast", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏘️ 熊本町丁人口予測ダッシュボード")
st.markdown("---")

# 依存ファイルのチェック
if not check_dependencies():
    st.error("必要なファイルが見つかりません。data/processed/ ディレクトリを確認してください。")
    st.stop()

# メタデータの読み込み
towns, years = load_metadata()

if not towns or not years:
    st.error("メタデータの読み込みに失敗しました。")
    st.stop()

# デバッグ情報
st.sidebar.write(f"利用可能な町丁数: {len(towns)}")
st.sidebar.write(f"利用可能な年数: {len(years)}")

# セッション状態の初期化
if "events" not in st.session_state:
    st.session_state.events = []
if "warnings" not in st.session_state:
    st.session_state.warnings = []

# --- サイドバー: 入力 ---
st.sidebar.header("📋 シナリオ設定")

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
if st.session_state.events:
    st.sidebar.subheader("📝 追加済みイベント")
    for i, event in enumerate(st.session_state.events):
        with st.sidebar.container():
            st.write(f"**{i+1}.** {event['event_type']} ({event['effect_direction']})")
            st.write(f"   年: +{event['year_offset']}, 信頼度: {event['confidence']:.1f}, 強度: {event['intensity']:.1f}")
            if event['note']:
                st.write(f"   備考: {event['note']}")
            
            if st.button(f"🗑️ 削除", key=f"del_{i}"):
                st.session_state.events.pop(i)
                st.rerun()

# マクロ設定
st.sidebar.subheader("🌍 マクロ変数")
st.sidebar.markdown("**外国人人口成長率 (%)**")
st.sidebar.markdown("*未入力の場合はNaNのまま（木モデル任せ）*")

f_h1 = st.sidebar.number_input("h1 成長率 %", value=0.00, step=0.01, format="%.2f")
f_h2 = st.sidebar.number_input("h2 成長率 %", value=0.00, step=0.01, format="%.2f")
f_h3 = st.sidebar.number_input("h3 成長率 %", value=0.00, step=0.01, format="%.2f")

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

# --- メインエリア ---
st.header("📊 予測結果")

# 実行ボタン
if st.button("🚀 予測実行", type="primary", use_container_width=True):
    try:
        # シナリオ作成
        scn = Scenario(
            town=town,
            base_year=base_year,
            horizons=horizons,
            events=[ScenarioEvent(**e) for e in st.session_state.events],
            macros={"foreign_population_growth_pct": {"h1": f_h1, "h2": f_h2, "h3": f_h3}} if any([f_h1, f_h2, f_h3]) else {},
            manual_delta={"h1": m_h1, "h2": m_h2, "h3": m_h3} if any([m_h1, m_h2, m_h3]) else {}
        )
        
        # 衝突チェック
        warnings = scn.validate_conflicts()
        if warnings:
            st.warning("⚠️ 衝突が検出されました:")
            for warning in warnings:
                st.warning(f"  {warning}")
        
        # 予測実行
        with st.spinner("予測を実行中..."):
            result, debug_info = run_scenario(scn, debug=True)
        
        # 結果表示
        st.success("✅ 予測が完了しました！")
        
        # サマリーカード
        cards = create_summary_cards(result)
        display_summary_cards(cards)
        
        # グラフ表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 人口予測パス")
            fig_pop = plot_population_path(result)
            st.plotly_chart(fig_pop, use_container_width=True)
        
        with col2:
            st.subheader("📊 寄与分解")
            fig_contrib = plot_contrib_bars(result)
            st.plotly_chart(fig_contrib, use_container_width=True)
        
        # 詳細結果
        with st.expander("📋 詳細結果 (JSON)"):
            st.json(result)
        
        # 寄与分解の円グラフ
        st.subheader("🥧 年別寄与分解")
        path = result["path"]
        years_available = [p["year"] for p in path]
        
        if years_available:
            selected_year = st.selectbox("年を選択", years_available)
            fig_pie = plot_contribution_pie(result, selected_year)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # デバッグ情報
        if debug_info:
            with st.expander("🔍 デバッグ情報"):
                if "debug_features" in debug_info:
                    st.subheader("将来特徴")
                    st.dataframe(debug_info["debug_features"])
                
                if "debug_contrib" in debug_info:
                    st.subheader("寄与分解詳細")
                    st.dataframe(debug_info["debug_contrib"])
        
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
st.markdown("**熊本町丁人口予測システム** - Layer 5 ダッシュボード")
