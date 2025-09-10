# -*- coding: utf-8 -*-
"""
メイン人口予測ダッシュボード
単一町丁予測と全地域予測を統合したメインアプリケーション
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# パス設定
sys.path.append(os.path.dirname(__file__))

# 各機能モジュールのインポート
from single_town_prediction import render_single_town_prediction
from all_towns_prediction import render_all_towns_prediction
from spatial_impact_prediction import render_spatial_impact_prediction

# ページ設定
st.set_page_config(
    page_title="人口予測ダッシュボード", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏘️ 人口予測ダッシュボード")
st.markdown("町丁、イベントタイプ、効果方向を選択して人口予測を実行")

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

# メタデータを読み込み
towns = load_metadata()

if not towns:
    st.error("メタデータの読み込みに失敗しました。")
    st.stop()

# ビューモード選択
view_mode = st.radio(
    "表示モードを選択",
    ["単一町丁予測", "全地域表示（空間分析）", "空間的影響予測"],
    horizontal=True
)

st.markdown("---")

# 選択されたモードに応じて対応する機能を表示
if view_mode == "単一町丁予測":
    render_single_town_prediction(towns)
elif view_mode == "全地域表示（空間分析）":
    render_all_towns_prediction(towns)
elif view_mode == "空間的影響予測":
    render_spatial_impact_prediction(towns)
