# 削除特徴量モデルの学習と比較

このディレクトリには、特徴量削減分析で特定された削除対象特徴量を除外してモデルを再学習し、フルモデルと性能を比較するスクリプトが含まれています。

## ファイル構成

### `train_reduced_model.py`
削除対象特徴量を除外してモデルを再学習するスクリプト

**機能:**
- 特徴量削減分析結果を読み込み
- 削除対象特徴量を除外
- フルモデルと同じ条件で学習
- 評価指標とモデルを保存

**出力:**
- `data/processed/feature_reduced_training/l4_predictions_reduced.csv`
- `data/processed/feature_reduced_training/l4_cv_metrics_reduced.json`
- `data/processed/feature_reduced_training/l4_feature_importance_reduced.csv`
- `models/l4_model_reduced.joblib`
- `src/layer4/feature_reduced_training/feature_list_reduced.json`

### `compare_models.py`
フルモデルと削除特徴量モデルの性能を比較するスクリプト

**機能:**
- 両モデルの評価指標を比較
- 改善率を計算
- 削減効果を評価
- 比較結果をプロット（オプション）

**出力:**
- `data/processed/feature_reduced_training/model_comparison.json`
- `data/processed/feature_reduced_training/model_comparison_plot.png`

## 使用方法

### 1. 削除特徴量モデルの学習
```bash
cd /Users/ujihara/インターンシップ本課題_地域科学研究所/subject3
python src/layer4/feature_reduced_training/train_reduced_model.py
```

### 2. モデル比較
```bash
python src/layer4/feature_reduced_training/compare_models.py
```

## 学習パラメータ

フルモデルと完全に同じパラメータを使用:
- **N_ESTIMATORS**: 25000
- **LEARNING_RATE**: 0.008
- **EARLY_STOPPING_ROUNDS**: 800
- **USE_HUBER**: True
- **HUBER_ALPHA**: 0.9
- **TIME_DECAY**: 0.05
- **ANOMALY_YEARS**: [2022, 2023]
- **ANOMALY_WEIGHT**: 0.3

## 評価指標

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## 期待される結果

特徴量削減により以下の効果が期待されます:

1. **モデルの簡素化**: 無効な特徴量の除去
2. **過学習の防止**: ノイズの削減
3. **計算効率の向上**: 特徴量数の削減
4. **性能の維持/向上**: 重要な特徴量のみに焦点

## 注意事項

- 削除対象特徴量は `src/layer4/ablation/feature_reduction_analysis.json` から自動的に読み込まれます
- フルモデルの学習結果は `data/processed/l4_cv_metrics_feature_engineered.json` から読み込まれます
- 学習には時間がかかる場合があります（フルモデルと同様の設定のため）
