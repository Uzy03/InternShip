"""
可視化コンポーネント（Plotly）
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List


def plot_population_path(result_json: Dict[str, Any]) -> go.Figure:
    """
    人口パス（PI帯付き）をプロット
    
    Args:
        result_json: 予測結果のJSON
        
    Returns:
        plotly.graph_objects.Figure: 人口パスグラフ
    """
    path = result_json["path"]
    df = pd.DataFrame(path)
    
    fig = go.Figure()
    
    # 人口パス（線）
    fig.add_trace(go.Scatter(
        x=df["year"],
        y=df["pop_hat"],
        mode='lines+markers',
        name='予測人口',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # 信頼区間（帯）
    if "pi95_pop" in df.columns and not df["pi95_pop"].isna().all():
        # pi95_popがリスト形式の場合
        if df["pi95_pop"].iloc[0] is not None and isinstance(df["pi95_pop"].iloc[0], list):
            lower = [p[0] if p is not None else None for p in df["pi95_pop"]]
            upper = [p[1] if p is not None else None for p in df["pi95_pop"]]
        else:
            # 数値形式の場合（簡易計算）
            margin = 1.96 * 50  # 簡易的な信頼区間
            lower = df["pop_hat"] - margin
            upper = df["pop_hat"] + margin
        
        fig.add_trace(go.Scatter(
            x=df["year"].tolist() + df["year"].tolist()[::-1],
            y=upper + lower[::-1],
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95%信頼区間',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"人口予測パス: {result_json['town']} (基準年: {result_json['base_year']})",
        xaxis_title="年",
        yaxis_title="人口（人）",
        hovermode='x unified',
        template="plotly_white"
    )
    
    return fig


def plot_contrib_bars(result_json: Dict[str, Any]) -> go.Figure:
    """
    寄与分解の積み上げバーをプロット
    
    Args:
        result_json: 予測結果のJSON
        
    Returns:
        plotly.graph_objects.Figure: 寄与分解グラフ
    """
    path = result_json["path"]
    df = pd.DataFrame(path)
    
    # 寄与データを準備
    contrib_data = []
    for _, row in df.iterrows():
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
    
    # サブプロット作成（バーと折れ線）
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 寄与の積み上げバー
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
        ), secondary_y=False)
    
    # Δ人口の折れ線
    fig.add_trace(go.Scatter(
        x=contrib_df["year"],
        y=contrib_df["delta_hat"],
        mode='lines+markers',
        name='Δ人口（予測）',
        line=dict(color='black', width=3),
        marker=dict(size=8)
    ), secondary_y=True)
    
    # レイアウト設定
    fig.update_layout(
        title=f"寄与分解: {result_json['town']} (基準年: {result_json['base_year']})",
        xaxis_title="年",
        barmode='relative',
        template="plotly_white",
        height=500
    )
    
    # Y軸設定
    fig.update_yaxes(title_text="寄与（人）", secondary_y=False)
    fig.update_yaxes(title_text="Δ人口（人）", secondary_y=True)
    
    return fig


def plot_contribution_pie(result_json: Dict[str, Any], year: int) -> go.Figure:
    """
    特定年の寄与分解を円グラフで表示
    
    Args:
        result_json: 予測結果のJSON
        year: 対象年
        
    Returns:
        plotly.graph_objects.Figure: 寄与分解円グラフ
    """
    path = result_json["path"]
    year_data = next((p for p in path if p["year"] == year), None)
    
    if not year_data:
        return go.Figure()
    
    contrib = year_data["contrib"]
    labels = []
    values = []
    colors = []
    
    color_map = {
        "exp": "#FF6B6B",
        "macro": "#4ECDC4", 
        "inertia": "#45B7D1",
        "other": "#96CEB4"
    }
    
    for key, value in contrib.items():
        if abs(value) > 0.1:  # 0に近い値は除外
            labels.append(key)
            values.append(abs(value))
            colors.append(color_map.get(key, "#CCCCCC"))
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{percent}<br>(%{value:.1f}人)'
    )])
    
    fig.update_layout(
        title=f"寄与分解: {year}年",
        template="plotly_white"
    )
    
    return fig


def create_summary_cards(result_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    サマリーカードのデータを作成
    
    Args:
        result_json: 予測結果のJSON
        
    Returns:
        List[Dict]: カードデータのリスト
    """
    path = result_json["path"]
    df = pd.DataFrame(path)
    
    cards = []
    
    # 最終年の人口
    final_year = df["year"].max()
    final_pop = df[df["year"] == final_year]["pop_hat"].iloc[0]
    cards.append({
        "title": f"{final_year}年予測人口",
        "value": f"{final_pop:,.0f}人",
        "delta": f"{final_pop - df['pop_hat'].iloc[0]:+,.0f}人",
        "color": "normal"
    })
    
    # 平均Δ人口
    avg_delta = df["delta_hat"].mean()
    cards.append({
        "title": "平均Δ人口",
        "value": f"{avg_delta:+.1f}人/年",
        "delta": None,
        "color": "normal" if avg_delta > 0 else "inverse"
    })
    
    # 最大期待効果
    max_exp = max([p["contrib"].get("exp", 0) for p in path])
    cards.append({
        "title": "最大期待効果",
        "value": f"{max_exp:+.1f}人",
        "delta": None,
        "color": "normal" if max_exp > 0 else "inverse"
    })
    
    return cards


def display_summary_cards(cards: List[Dict[str, Any]]):
    """
    サマリーカードを表示
    
    Args:
        cards: カードデータのリスト
    """
    cols = st.columns(len(cards))
    
    for i, card in enumerate(cards):
        with cols[i]:
            if card["color"] == "inverse":
                st.metric(
                    label=card["title"],
                    value=card["value"],
                    delta=card["delta"]
                )
            else:
                st.metric(
                    label=card["title"],
                    value=card["value"],
                    delta=card["delta"]
                )
