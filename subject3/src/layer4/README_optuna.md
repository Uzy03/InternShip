# LightGBM + Optuna最適化

## 概要

LightGBMモデルのパラメータをOptunaで自動最適化する機能を追加しました。

## ファイル構成

- `lgbm_model.py` - LightGBMモデルのクラス定義
- `optuna_optimizer.py` - Optuna最適化クラス
- `train_lgbm.py` - メインの学習スクリプト（拡張）

## 使用方法

### 1. 通常の学習（固定パラメータ）

```bash
python train_lgbm.py
```

### 2. Optuna最適化 + 学習

```bash
# デフォルト（50回の試行）
python train_lgbm.py --optimize

# 試行回数を指定
python train_lgbm.py --optimize --n_trials 100
```

## 出力ファイル

### 既存ファイル（後方互換性維持）
- `l4_cv_metrics.json` - CV結果（拡張）
- `l4_predictions.csv` - 予測結果
- `l4_model.joblib` - 学習済みモデル（既存形式）

### 新規ファイル
- `l4_model_full.joblib` - 完全なモデルクラス
- `l4_optuna_study.db` - Optuna最適化履歴
- `l4_optuna_results.json` - 最適化結果の詳細

## 最適化パラメータ

以下のパラメータが最適化対象です：

- `learning_rate`: 0.001 - 0.1
- `subsample`: 0.6 - 1.0
- `colsample_bytree`: 0.6 - 1.0
- `reg_alpha`: 0.0 - 1.0
- `reg_lambda`: 0.0 - 1.0
- `num_leaves`: 10 - 300
- `min_child_samples`: 5 - 100

## 実行時間の目安

- **50回の試行**: 1-2時間（Colab無料プラン）
- **100回の試行**: 2-3時間（Colab無料プラン）

## 後方互換性

- layer5とdashboardは変更不要
- 既存の`joblib.load()`はそのまま動作
- 既存の学習済みモデルも使用可能

## 注意事項

- Optunaがインストールされていない場合は警告が出て、通常モードで実行されます
- 最適化中はログが大量に出力されるため、Colabでは適宜抑制してください
- 最適化結果はSQLiteデータベースに保存されるため、途中で中断しても再開可能です
