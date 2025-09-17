# 因果関係学習システム

## 概要

このディレクトリには、従来の期待効果ベースモデルを改善し、**真の因果関係を学習**するためのシステムが含まれています。

## 問題点

従来のシステムでは：
- Layer3で効果係数（`beta_t`, `beta_t1`）を事前計算
- Layer4でその係数を使って期待効果特徴量（`exp_*_h*`）を事前計算
- LightGBMは既に計算済みの期待効果特徴量を入力として使用

**結果**: LightGBMは生のイベントデータを見ておらず、真の因果関係を学習していない

## 解決策

### 1. 生のイベントデータを直接使用
- `event_housing_inc_t`, `event_housing_inc_t1` などの生のイベントデータ
- 事前計算された期待効果ではなく、生のデータから因果関係を学習

### 2. 時空間的相互作用特徴量を追加
- **時系列ラグ特徴量**: 過去のイベント影響（`_lag1`, `_lag2`, `_ma2`, `_ma3`など）
- **空間ラグ特徴量**: 近隣町丁のイベント影響（`ring1_*`, `ring2_*`など）
- **相互作用特徴量**: イベント間の相互作用（`housing_x_commercial_interaction`など）
- **強度特徴量**: イベントの強度と密度（`total_events_t`, `housing_intensity`など）

### 3. レジーム依存の学習
- 時代区分による効果の違いを学習
- コロナ期間、政令市化後などのレジーム変化を考慮

## ファイル構成

```
improved/
├── build_causal_features.py          # 因果関係学習用特徴量構築
├── train_causal_lgbm.py             # 因果関係学習モデル訓練
├── compare_causal_vs_traditional.py # 従来モデルとの性能比較
├── run_causal_learning.py           # 一括実行スクリプト
└── README.md                        # このファイル
```

## 使用方法

### 1. 一括実行（推奨）
```bash
cd subject3/src/layer4/improved/
python run_causal_learning.py
```

### 2. 個別実行
```bash
# 1. 特徴量構築
python build_causal_features.py

# 2. モデル訓練
python train_causal_lgbm.py

# 3. 性能比較
python compare_causal_vs_traditional.py
```

## 特徴量の種類

### 生のイベント特徴量
- `event_housing_inc_t`, `event_housing_inc_t1`
- `event_commercial_inc_t`, `event_commercial_inc_t1`
- `event_disaster_inc_t`, `event_disaster_dec_t`
- など

### 時系列ラグ特徴量
- `event_housing_inc_t_lag1`, `event_housing_inc_t_lag2`
- `event_housing_inc_t_ma2`, `event_housing_inc_t_ma3`
- `event_housing_inc_t_cumsum`, `event_housing_inc_t_pct_change`
- など

### 空間ラグ特徴量
- `ring1_event_housing_inc_t`, `ring2_event_housing_inc_t`
- 近隣町丁のイベント影響を定量化

### 相互作用特徴量
- `event_housing_inc_t_x_event_commercial_inc_t`
- `housing_x_commercial_interaction`
- イベント間の相互作用をモデル化

### 強度特徴量
- `total_events_t`, `total_events_t1`
- `housing_intensity`, `commercial_intensity`
- `positive_events_t`, `negative_events_t`
- イベントの強度と密度を定量化

### レジーム特徴量
- `era_pre2010`, `era_2010_2015`, `era_2015_2020`, `era_post2020`
- `era_covid`
- 時代区分による効果の違いを考慮

## 出力ファイル

### データファイル
- `../../../data/processed/features_causal.csv` - 因果関係学習用特徴量
- `../../../data/processed/l4_causal_predictions.csv` - 予測結果
- `../../../data/processed/l4_causal_metrics.json` - メトリクス
- `../../../data/processed/l4_causal_feature_importance.csv` - 特徴量重要度

### モデルファイル
- `../../../models/l4_causal_model.joblib` - 因果関係学習モデル

### 分析ファイル
- `causal_vs_traditional_comparison.json` - 比較結果
- `causal_vs_traditional_comparison.png` - 比較プロット
- `causal_feature_list.json` - 特徴量リスト

## 期待される効果

### 1. 真の因果関係の学習
- 生のイベントデータから直接因果関係を学習
- 複雑な非線形相互作用を捉える

### 2. 時空間的伝播の学習
- 時間的減衰と空間的拡散を自動学習
- 固定された係数に依存しない柔軟なモデル

### 3. レジーム依存の学習
- 時代や地域特性に応じた効果の違いを学習
- 新しい状況への適応能力の向上

### 4. 解釈可能性の向上
- 特徴量重要度から重要な因果関係を特定
- イベント間の相互作用を可視化

## 注意事項

1. **データの準備**: 必要なファイルが存在することを確認
   - `features_panel.csv`
   - `events_matrix_signed.csv`
   - `town_centroids.csv`

2. **計算リソース**: 大量の特徴量生成のため、メモリ使用量が増加する可能性

3. **過学習のリスク**: 複雑な特徴量のため、適切な正則化が重要

## 今後の改善案

1. **特徴量選択の最適化**: 重要度に基づく特徴量選択
2. **アンサンブル学習**: 複数のモデルの組み合わせ
3. **深層学習の導入**: より複雑な相互作用の学習
4. **因果推論手法の導入**: より厳密な因果関係の特定
