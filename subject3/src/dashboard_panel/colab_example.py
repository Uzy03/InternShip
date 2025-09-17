# -*- coding: utf-8 -*-
"""
Colab用の人口予測ダッシュボード実行例
"""
# まずインストール＆拡張読み込み（別セルで実行）
# !pip install -q panel plotly tabulator

import panel as pn
import pandas as pd
from pathlib import Path
import sys
import os

# Panelの設定（Colab用）
pn.extension('plotly','tabulator')   # ← これ大事
pn.config.sizing_mode = "stretch_width"

# パス設定（Colab環境用）
current_dir = Path.cwd()
if 'content' in str(current_dir):
    # Colab環境の場合
    sys.path.append('/content/インターンシップ本課題_地域科学研究所/subject3/src/dashboard_panel')
else:
    # ローカル環境の場合
    sys.path.append(os.path.dirname(__file__))

# ダッシュボードのインポート
from app_main import create_dashboard

# Colab用：セル内表示のため、ただ返すだけ
dashboard = create_dashboard()
