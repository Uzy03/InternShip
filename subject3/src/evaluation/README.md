# TWFE+LightGBM統合モデル評価システム

このディレクトリには、layer3のTWFEとlayer4のLightGBMを組み合わせた最終的なモデルの評価と予測を行うシステムが含まれています。

## ファイル構成

- `combined_model_evaluation.py`: 統合モデルの評価システム
- `combined_model_predictor.py`: 統合モデルの予測システム
- `README.md`: このファイル

## 概要

従来の評価では、LightGBMのみを評価していましたが、最終的なモデルはTWFE+LightGBMの組み合わせであるべきです。このシステムでは：

1. **TWFE単体**の評価
2. **LightGBM単体**の評価  
3. **TWFE+LightGBM統合**の評価（複数の重みで）

を実行し、最適な組み合わせを特定します。

## 使用方法

### 1. 統合モデルの評価

```bash
cd /Users/ujihara/インターンシップ本課題_地域科学研究所/subject3
python src/evaluation/combined_model_evaluation.py
```

このスクリプトは以下を実行します：
- 時系列クロスバリデーション（layer4と同じ方法）
- 各モデルの性能評価（MAE, RMSE, MAPE, R2）
- 結果のJSONファイル保存
- 比較サマリーの表示

### 2. 統合モデルでの予測

```python
from src.evaluation.combined_model_predictor import CombinedModelPredictor

# 予測器を初期化
predictor = CombinedModelPredictor(
    twfe_coeffs_path="output/effects_coefficients_rate.csv",
    lgbm_model_path="models/l4_model.joblib"
)

# 統合予測を実行
predictions = predictor.predict_combined(data, twfe_weight=0.5)

# 複数の重みで予測
batch_predictions = predictor.predict_batch(data, twfe_weights=[0.3, 0.5, 0.7])
```

## 評価方法

### 評価指標
- **MAE**: 平均絶対誤差
- **RMSE**: 二乗平均平方根誤差
- **MAPE**: 平均絶対パーセント誤差
- **R²**: 決定係数

### クロスバリデーション
- **方法**: 時系列クロスバリデーション（expanding window）
- **最小訓練年数**: 20年
- **テストウィンドウ**: 1年
- **フォールド数**: データに応じて自動決定

### 統合方法
- **重み付き平均**: `twfe_weight * twfe_pred + (1 - twfe_weight) * lgbm_pred`
- **重みの候補**: 0.3, 0.5, 0.7

## 出力ファイル

### `src/evaluation/combined_model_evaluation.json`
各モデルの評価結果が含まれます：

```json
{
  "TWFE_only": {
    "folds": [...],
    "aggregate": {"MAE": ..., "RMSE": ..., "MAPE": ..., "R2": ...},
    "predictions": [...]
  },
  "LightGBM_only": {...},
  "Combined_0.3": {...},
  "Combined_0.5": {...},
  "Combined_0.7": {...}
}
```

## 前提条件

以下のファイルが存在する必要があります：
- `data/processed/features_l4.csv`: 特徴量データ
- `output/effects_coefficients_rate.csv`: TWFE係数
- `models/l4_model.joblib`: LightGBMモデル
- `data/processed/l4_cv_metrics.json`: LightGBMメトリクス

## 注意事項

1. **メモリ使用量**: 大量のデータを扱う場合、メモリ使用量に注意してください
2. **実行時間**: クロスバリデーションの実行には時間がかかる場合があります
3. **エラーハンドリング**: モデルファイルが存在しない場合は適切なエラーメッセージが表示されます

## トラブルシューティング

### よくある問題

1. **ファイルが見つからない**
   - パスを確認してください
   - 必要なファイルが生成されているか確認してください

2. **メモリエラー**
   - データサイズを確認してください
   - 必要に応じてデータをサンプリングしてください

3. **予測エラー**
   - 特徴量の形式を確認してください
   - モデルの互換性を確認してください

## 更新履歴

- 2024-01-XX: 初回実装
  - TWFE+LightGBM統合評価システム
  - 時系列クロスバリデーション対応
  - 複数重みでの比較機能
