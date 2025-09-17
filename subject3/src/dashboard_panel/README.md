# 人口予測ダッシュボード (Panel版)

CLIで実行した人口予測結果を可視化するPanelベースのダッシュボードです。
Colab環境での実行に最適化されています。

## 機能

- 📈 人口予測パスの折れ線グラフ（信頼区間付き）
- 📊 人口変化量のバーグラフ
- 🥧 寄与分解の積み上げバーグラフ
- 🥧 年別寄与分解の円グラフ
- 📋 詳細データテーブル
- 📊 サマリー統計
- 🗺️ 空間分析（地図表示）
- 🔄 リアルタイム予測実行

## 使用方法

### Colab環境での実行

1. 依存関係のインストール
```python
!pip install panel pandas plotly numpy geopandas pyproj shapely fiona scipy bokeh holoviews param
```

2. ダッシュボードの起動
```python
import panel as pn
from app_main import create_dashboard

# ダッシュボードを作成
dashboard = create_dashboard()

# Colabで表示
dashboard.show()
```

### ローカル環境での実行

1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

2. ダッシュボードの起動
```bash
python app_main.py
```

## ファイル構成

- `app_main.py`: メインのダッシュボードアプリケーション
- `single_town_prediction.py`: 単一町丁予測機能
- `all_towns_prediction.py`: 全地域表示（空間分析）機能
- `spatial_impact_prediction.py`: 空間的影響予測機能
- `requirements.txt`: 必要なPythonパッケージ
- `README.md`: このファイル

## 前提条件

- CLIで予測結果が `../../data/processed/` ディレクトリに保存されていること
- 予測結果はJSON形式で、以下の構造を持つこと：
  ```json
  {
    "town": "町丁名",
    "base_year": 基準年,
    "horizons": [予測期間],
    "path": [
      {
        "year": 年,
        "delta_hat": 人口変化量,
        "pop_hat": 予測人口,
        "pi95_delta": [信頼区間下限, 信頼区間上限],
        "pi95_pop": [信頼区間下限, 信頼区間上限],
        "contrib": {
          "exp": 期待効果,
          "macro": マクロ効果,
          "inertia": 慣性効果,
          "other": その他
        }
      }
    ]
  }
  ```

## 注意事項

- 予測結果ファイルは自動的に検索されます
- 複数の結果ファイルがある場合は、サイドバーから選択できます
- 結果はキャッシュされるため、同じファイルを再読み込みする際は高速です
- Colab環境では、Panelの表示に時間がかかる場合があります
