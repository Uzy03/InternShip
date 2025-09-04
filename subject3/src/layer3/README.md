# Layer3: TWFE Effects Analysis

Two-Way Fixed Effects (TWFE) 推定によるイベント影響分析の実装です。

## 概要

このモジュールは、様々なイベント（住宅、商業、交通、雇用など）が人口変化に与える因果効果を、パネルデータを用いたTWFE推定により分析します。

## ファイル構成

```
src/layer3_effects/
├── __init__.py          # パッケージ初期化
├── twfe_effects.py      # メイン実装
└── README.md           # このファイル
```

## 主要機能

### 1. データ読み込み・マージ
- `features_panel.csv`: 人口データと特徴量
- `events_matrix.csv`: イベント強度データ（t, t+1）
- `events_labeled.csv`: 有向イベントデータ（オプション）

### 2. TWFE推定
- **目的変数**: 年差人口 `ΔPop_{i,t} = Pop_{i,t} - Pop_{i,t-1}`
- **説明変数**: イベント強度（カテゴリ別、当年t と翌年t+1）
- **固定効果**: 町丁FE + 年FE（two-way FE）
- **コントロール**: growth_adj_log、ポリシー境界ダミー、2016震災ダミー
- **標準誤差**: town クラスタでロバスト

### 3. 出力
- `subject3/output/effects_coefficients.csv`: 係数結果
- `subject3/logs/{run_id}/twfe_summary.txt`: 分析サマリ

## 使用方法

### 基本的な使用例

```python
# プロジェクトルートから実行する場合
import sys
sys.path.append('subject3/src/layer3')
from twfe_effects import TWFEEstimator

# 推定器を初期化
estimator = TWFEEstimator()

# 分析実行
results = estimator.run_analysis(
    use_year_min=1998,
    use_year_max=2025,
    separate_t_and_t1=True,
    use_signed_from_labeled=False,
    weight_mode="sqrt_prev",
    ridge_alpha=0.0,
    include_growth_adj=True,
    include_policy_boundary=True,
    include_disaster_2016=True
)

print(f"推定完了: {len(results)} 個のイベントタイプ")
print(results[['event_var', 'beta', 'ci_low', 'ci_high']])
```

### 実行方法

```bash
# プロジェクトルートから直接実行
python subject3/src/layer3/twfe_effects.py

# または、layer3ディレクトリに移動して実行
cd subject3/src/layer3
python twfe_effects.py
```

### 有向イベント分析

```python
# 有向イベント版（increase/decrease を分離）
results = estimator.run_analysis(
    use_signed_from_labeled=True,  # 有向イベントを使用
    separate_t_and_t1=True
)
```

## パラメータ

### データフィルタ
- `use_year_min`: 使用する最小年（デフォルト: 1998）
- `use_year_max`: 使用する最大年（デフォルト: 2025）

### 推定設定
- `separate_t_and_t1`: t と t+1 を別の変数として扱うか（デフォルト: True）
- `use_signed_from_labeled`: 有向イベントを使用するか（デフォルト: False）
- `weight_mode`: 重み付けモード（"unit", "sqrt_prev"）
- `ridge_alpha`: Ridge正則化パラメータ（デフォルト: 0.0）

### コントロール変数
- `include_growth_adj`: growth_adj_log を含めるか
- `include_policy_boundary`: ポリシー境界ダミーを含めるか
- `include_disaster_2016`: 2016年震災ダミーを含めるか

## 出力形式

### effects_coefficients.csv
```csv
event_var,beta,se,pval,ci_low,ci_high,beta_t,beta_t1,gamma0,gamma1,n_obs,r2_within
housing,1.693,0.123,0.001,1.452,1.934,1.693,0.000,1.000,0.000,17381,0.9887
commercial,1.507,0.089,0.001,1.332,1.682,0.000,1.507,0.000,1.000,17381,0.9887
```

### 係数の解釈
- `beta`: 総合効果（1単位強度あたりの人口変化）
- `beta_t`: 当年効果
- `beta_t1`: 翌年効果
- `gamma0`, `gamma1`: 分配比（t, t+1。和=1）
- `ci_low`, `ci_high`: 95%信頼区間

## 数式

### 基本モデル
```
ΔPop_{i,t} = β_h,0 * Housing_t(i,t)  +  β_h,1 * Housing_t1(i,t)
           + β_c,0 * Commercial_t(i,t) + β_c,1 * Commercial_t1(i,t)
           + ... + Controls_{i,t} * θ
           + α_i + τ_t + ε_{i,t}
```

### 分配比の計算
```
gamma0 = max(0, β_0) / max(ε, β_0 + β_1)
gamma1 = 1 - gamma0
```

## 受け入れ基準

1. ✅ `effects_coefficients.csv` が生成される
2. ✅ 主要カテゴリ（housing, commercial, transit, ...）について NaNなしの β とCIが埋まっている
3. ✅ `gamma0 + gamma1 ≈ 1`（±1e-6 以内、負値は0にクリップ）
4. ✅ 回帰はエラーなく終了し、`r2_within` と観測数がログに出る
5. ✅ マージ後の学習データ (town,year) が 0件ではない

## 実装の注意点

### マルチコリニアリティ
- `event_*_t` と `event_*_t1` が強相関になる場合がある
- 分散が小さすぎる場合は `separate_t_and_t1=false` で合成変数に変更
- または `ridge_alpha>0` でRidge正則化を有効化

### スケール差
- カテゴリ別で単位（戸/人/m²/1）が混在
- 係数は「1単位強度あたりの人数」
- UI表示時には ×100戸 のように変換倍率も併記推奨

### 方向性
- 減少イベントを明確に分けたい場合は `use_signed_from_labeled=true`
- `*_inc/*_dec` を別回帰変数に切り出し

## テスト

```bash
# プロジェクトルートから実行
python subject3/src/layer3/test_twfe_effects.py

# または、layer3ディレクトリに移動して実行
cd subject3/src/layer3
python test_twfe_effects.py
```

## 依存関係

- pandas
- numpy
- statsmodels
- scikit-learn (Ridge正則化用)
- pathlib
- logging
- uuid
- datetime

## ログ

実行ログは `subject3/logs/{run_id}/` ディレクトリに保存されます：
- `twfe_analysis.log`: 詳細ログ
- `twfe_summary.txt`: 分析サマリ
