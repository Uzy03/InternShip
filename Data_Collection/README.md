# 熊本市統計データ収集ツール（Google Colab版）

このツールは、Google Colab環境で熊本市の統計サイトから人口統計データを自動収集するためのPythonスクリプトです。

## 収集対象データ

- **期間**: 平成10年1月〜令和7年12月（1998年1月〜2035年12月）
- **統計**: 年齢5歳刻み
- **区分**: 町丁別
- **地区**: すべて

## ファイル構成

```
Data_Collection/
├── README.md                                    # このファイル
├── requirements.txt                             # Python依存関係
├── detailed_site_inspector_fixed.py            # サイト構造調査用（修正版）
└── kumamoto_data_collector_colab_fixed.py      # メインデータ収集用（修正版）
```

## 使用方法

### ステップ1: パッケージのインストール

```python
!pip install selenium webdriver-manager beautifulsoup4 pandas openpyxl matplotlib numpy
```

### ステップ2: サイト構造の調査

```python
# detailed_site_inspector_fixed.pyの内容をコピペして実行
```

### ステップ3: データ収集の実行

```python
# kumamoto_data_collector_colab_fixed.pyの内容をコピペして実行
```

## 主な特徴

- **Google Colab対応**: ブラウザベースの環境で動作
- **Chromium自動インストール**: 必要なブラウザを自動セットアップ
- **ユーザーデータディレクトリ競合解決**: セッションエラーを回避
- **詳細なサイト構造調査**: 要素の特定と分析
- **自動データ収集**: 指定期間の全データを自動取得

## 注意事項

- 各スクリプトは上から順番に実行してください
- エラーが発生した場合は、前のセルを再実行してください
- ランタイムタイプは「GPU」または「TPU」に設定することをお勧めします

## トラブルシューティング

問題が発生した場合は、以下を確認してください：

1. パッケージのインストールが完了しているか
2. ランタイムが正常に動作しているか
3. エラーログの詳細内容

## ライセンス

このツールは教育・研究目的で作成されています。
