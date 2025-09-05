# -*- coding: utf-8 -*-
"""
人口予測ダッシュボード
シナリオファイルを選択して実行し、結果を可視化するダッシュボード
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
import subprocess
import tempfile

# パス設定
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'layer5'))

# ページ設定
st.set_page_config(
    page_title="人口予測ダッシュボード", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏘️ 熊本町丁人口予測ダッシュボード")
st.markdown("---")

# サイドバー: シナリオ選択と実行
st.sidebar.header("🎯 シナリオ選択と実行")

# シナリオファイルの検索
scenario_dir = Path("../../src/layer5/scenario_examples")
if not scenario_dir.exists():
    scenario_dir = Path("../layer5/scenario_examples")

json_files = list(scenario_dir.glob("*.json"))
if not json_files:
    st.error(f"シナリオファイルが見つかりません。ディレクトリ: {scenario_dir}")
    st.stop()

# シナリオファイル選択
selected_scenario = st.sidebar.selectbox(
    "シナリオファイルを選択",
    json_files,
    format_func=lambda x: x.name
)

# シナリオの内容を表示
st.sidebar.subheader("📋 選択されたシナリオ")
try:
    with open(selected_scenario, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)
    
    st.sidebar.json(scenario_data)
except Exception as e:
    st.sidebar.error(f"シナリオファイルの読み込みに失敗: {e}")

# 実行ボタン
if st.sidebar.button("🚀 シナリオ実行", type="primary", use_container_width=True):
    with st.spinner("シナリオを実行中..."):
        try:
            # CLIを実行
            cli_path = Path("../../src/layer5/cli_run_scenario.py")
            if not cli_path.exists():
                cli_path = Path("../layer5/cli_run_scenario.py")
            
            # 出力ファイル名を生成
            output_file = f"output/forecast_result_{selected_scenario.stem}.json"
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True)
            
            # CLI実行
            result = subprocess.run([
                "python", str(cli_path), 
                str(selected_scenario), 
                str(output_path)
            ], capture_output=True, text=True, cwd=Path("../../src/layer5"))
            
            if result.returncode == 0:
                st.success("✅ シナリオ実行が完了しました！")
                st.session_state.last_result_file = str(output_path)
                st.rerun()
            else:
                st.error(f"❌ シナリオ実行に失敗しました:\n{result.stderr}")
                
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")

# 結果ファイルの選択
st.sidebar.subheader("📁 予測結果ファイル選択")

# 結果ファイルの検索
result_dir = Path("../../src/layer5/output")
if not result_dir.exists():
    result_dir = Path("../layer5/output")

result_files = list(result_dir.glob("*.json"))
if not result_files:
    st.warning("予測結果ファイルが見つかりません。シナリオを実行してください。")
    st.stop()

# 最後に実行した結果をデフォルト選択
default_index = 0
if hasattr(st.session_state, 'last_result_file'):
    for i, file in enumerate(result_files):
        if str(file) == st.session_state.last_result_file:
            default_index = i
            break

# ファイル選択
selected_file = st.sidebar.selectbox(
    "予測結果ファイルを選択",
    result_files,
    index=default_index,
    format_func=lambda x: x.name
)

# 結果の読み込み
@st.cache_data
def load_forecast_result(file_path):
    """予測結果を読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
        return None

result = load_forecast_result(selected_file)
if result is None:
    st.stop()

# メイン表示エリア
st.header("📊 予測結果")

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
    # pi95_popがリスト形式の場合
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

# フッター
st.markdown("---")
st.markdown("**熊本町丁人口予測システム** - 結果表示ダッシュボード")
