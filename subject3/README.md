# Subject3: パネルデータ構築フレームワーク

## 概要
このプロジェクトは、人口時系列データを正規化してパネルデータ形式に変換するフレームワークです。

## ディレクトリ構造
```
subject3/
├── requirements.txt          # 依存関係
├── data/
│   ├── input/               # 入力データ（自動生成）
│   └── processed/           # 処理済みデータ（自動生成）
├── logs/                    # ログファイル（自動生成）
├── src/
│   ├── common/              # 共通ユーティリティ
│   │   ├── __init__.py
│   │   ├── io.py            # 入出力ユーティリティ
│   │   └── utils.py         # 町丁名正規化ユーティリティ
│   └── layer0_data/         # レイヤ0: データ構築
│       ├── __init__.py
│       └── build_panel.py   # パネルデータ構築スクリプト
└── README.md
```

## セットアップ

### 依存関係のインストール
```bash
pip install -r requirements.txt
```

## 使用方法

### パネルデータ構築の実行
**ルートパス（インターンシップ本課題_地域科学研究所）から実行してください**

```bash
python subject3/src/layer0_data/build_panel.py
```

## 機能

### 共通ユーティリティ (src/common/)

#### io.py
- `resolve_path(rel: str) -> Path`: プロジェクトルートからの相対パス解決
- `ensure_parent(path: str|Path) -> None`: 親ディレクトリ作成
- `get_logger(run_id: str) -> logging.Logger`: ログ設定
- `set_seed(seed: int) -> None`: 乱数シード設定

#### utils.py
- `normalize_town(name: str) -> str`: 町丁名正規化

### レイヤ0 (src/layer0_data/)

#### build_panel.py
- 入力CSVの必須列チェック
- 町丁名の正規化
- year列の妥当性チェック
- (town,year)の重複検出
- 合併3町（城南/富合/植木）の2009年以前データの保持
- データの並び替え（town ASC, year ASC）
- パネルデータの出力

## 入力データ仕様
- 必須列: `town` (str), `year` (int), `pop_total` (int)
- 任意列: `hh_total` (int), 5歳階級列（列名はそのまま保持）
- 年は西暦、1年刻み（4月時点）
- 入力ファイル: `subject2/processed_data/population_time_series.csv`

## 出力データ仕様
- 列: `town`, `year`, `pop_total`, (`hh_total`), (age bins...)
- 町丁名は正規化済み
- (town,year)が一意
- 並び順: town ASC, year ASC
- エンコーディング: UTF-8
- インデックス: なし
- 出力ファイル: `subject3/data/processed/panel_raw.csv`

## ログ
処理ログは`logs/{run_id}/run.log`に出力されます。run_idは実行時のタイムスタンプで自動生成されます。

## エラーハンドリング
- ファイル未発見・書込不可は例外送出
- 必須列不足、重複データ、不正な年データは適切なエラーメッセージとともに例外送出
- 合併3町の2009年以前データはNaNのまま保持（無理に埋めない）
