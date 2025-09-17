# -*- coding: utf-8 -*-
"""
メイン人口予測ダッシュボード (Panel版)
単一町丁予測と全地域予測を統合したメインアプリケーション
"""
import panel as pn
import pandas as pd
from pathlib import Path
import sys
import os

# パス設定
sys.path.append(os.path.dirname(__file__))

# 各機能モジュールのインポート
from single_town_prediction import create_single_town_prediction
from all_towns_prediction import create_all_towns_prediction
from spatial_impact_prediction import create_spatial_impact_prediction

# Panelの設定（Colab用）
pn.extension('plotly', 'tabulator')
pn.config.sizing_mode = "stretch_width"

def load_metadata():
    """利用可能な町丁のリストを取得"""
    try:
        # Colab環境用のパス修正
        features_path = Path("subject3-2/data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("subject3-1/data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../../data/processed/features_panel.csv")
        if not features_path.exists():
            features_path = Path("../data/processed/features_panel.csv")
        
        if not features_path.exists():
            print(f"features_panel.csv が見つかりません: {features_path}")
            return []
        
        df = pd.read_csv(features_path, usecols=["town"]).drop_duplicates()
        towns = sorted(df["town"].unique().tolist())
        
        return towns
    except Exception as e:
        print(f"メタデータの読み込みに失敗しました: {e}")
        return []

def create_dashboard():
    """ダッシュボードを作成して返す（Colab用）"""
    towns = load_metadata()
    
    if not towns:
        return pn.pane.Alert("features_panel.csv が見つかりません。パスを確認してください。", alert_type="danger")
    
    # 各機能コンポーネントを作成
    single_town_component = create_single_town_prediction(towns)
    all_towns_component = create_all_towns_prediction(towns)
    spatial_impact_component = create_spatial_impact_prediction(towns)
    
    # タブの作成
    tabs = pn.Tabs(
        ("単一町丁予測", single_town_component.view()),
        ("全地域表示（空間分析）", all_towns_component.view()),
        ("空間的影響予測", spatial_impact_component.view()),
        tabs_location="above",
        sizing_mode="stretch_both"
    )
    
    # ヘッダー
    header = pn.pane.Markdown(
        "# 🏘️ 人口予測ダッシュボード\n町丁、イベントタイプ、効果方向を選択して人口予測を実行",
        styles={"background":"#f0f0f0","padding":"12px","border-radius":"8px"}
    )
    
    return pn.Column(header, tabs, sizing_mode="stretch_both")

# Colab用：セル内表示のため、ただ返すだけ
dashboard = create_dashboard()
