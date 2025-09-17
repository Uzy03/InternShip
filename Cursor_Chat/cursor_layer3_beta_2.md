# Layer3でのイベントのbeta算出
_Exported on 2025/9/17 at 10:50:17 JST from Cursor (1.5.9)_

---

**User**

layer3でイベントのbetaとかを算出してるけど，これ何？

---

**Cursor**

layer3でイベントのbetaを算出している部分について調べてみますね。まず、layer3のファイルを確認して、betaがどのように計算されているかを見てみましょう。


layer3で算出されているイベントのbetaについて説明しますね。

## イベントのbetaとは

layer3の`twfe_effects.py`で算出されている**beta（β）**は、**各イベントが人口変化に与える因果効果の係数**です。

### 具体的には：

1. **目的変数**: 人口変化（`ΔPop_{i,t} = Pop_{i,t} - Pop_{i,t-1}`）
2. **説明変数**: 各イベントの強度（住宅、商業、交通、雇用、災害など）
3. **beta**: イベント強度1単位あたりの人口変化人数

### 数式で表すと：

```
ΔPop_{i,t} = β_housing * Housing_t(i,t) + β_commercial * Commercial_t(i,t) 
           + β_transit * Transit_t(i,t) + ... + Controls + α_i + τ_t + ε_{i,t}
```

ここで：
- `β_housing`: 住宅イベント1単位あたりの人口変化人数
- `β_commercial`: 商業イベント1単位あたりの人口変化人数
- `β_transit`: 交通イベント1単位あたりの人口変化人数

### 時間的な分解：

コードでは`separate_t_and_t1=True`の設定で、イベントの効果を**当年（t）**と**翌年（t+1）**に分けて推定しています：

- `beta_t`: 当年の効果
- `beta_t1`: 翌年の効果  
- `beta`: 総合効果（`beta_t + beta_t1`）

### 実際の出力例：

```csv
event_var,beta,se,pval,ci_low,ci_high,beta_t,beta_t1,gamma0,gamma1
housing_inc,1.693,0.123,0.001,1.452,1.934,1.693,0.000,1.000,0.000
commercial_inc,1.507,0.089,0.001,1.332,1.682,0.000,1.507,0.000,1.000
```

### 解釈：

- **housing_inc**: 住宅増加イベント1単位あたり、人口が1.693人増加
- **commercial_inc**: 商業増加イベント1単位あたり、人口が1.507人増加

これらのbeta値は、layer4の機械学習モデルで**期待効果**として使用され、layer5の予測サービスで**シナリオ分析**に活用されます。

```plaintext
ΔPop_{i,t} = β_housing * Housing_t(i,t) + β_commercial * Commercial_t(i,t) 
           + β_transit * Transit_t(i,t) + ... + Controls + α_i + τ_t + ε_{i,t}
```

```csv
event_var,beta,se,pval,ci_low,ci_high,beta_t,beta_t1,gamma0,gamma1
housing_inc,1.693,0.123,0.001,1.452,1.934,1.693,0.000,1.000,0.000
commercial_inc,1.507,0.089,0.001,1.332,1.682,0.000,1.507,0.000,1.000
```

---

**User**

twfeって何？

---

**Cursor**

TWFEについて詳しく説明しますね。

## TWFEとは

**TWFE = Two-Way Fixed Effects（二方向固定効果）**の略で、パネルデータ分析でよく使われる計量経済学の手法です。

### 基本的な考え方

パネルデータ（複数の個体×複数の時点）において、**個体固有の効果**と**時点固有の効果**を同時に制御する手法です。

### 数式で表すと：

```
Y_{i,t} = α_i + τ_t + βX_{i,t} + ε_{i,t}
```

ここで：
- `Y_{i,t}`: 個体i、時点tの被説明変数（人口変化など）
- `α_i`: 個体i固有の固定効果（町丁固有の特性）
- `τ_t`: 時点t固有の固定効果（年固有の効果）
- `X_{i,t}`: 説明変数（イベント強度など）
- `β`: 推定したい係数（イベントの効果）
- `ε_{i,t}`: 誤差項

### このプロジェクトでの適用

layer3の`twfe_effects.py`では：

1. **個体固定効果（α_i）**: 各町丁の固有特性
   - 地理的条件、歴史的経緯、行政区分など

2. **時点固定効果（τ_t）**: 各年の共通要因
   - 景気変動、全国的な政策、災害など

3. **説明変数（X_{i,t）**: イベント強度
   - 住宅、商業、交通、雇用、災害などのイベント

### なぜTWFEを使うのか？

1. **因果推論の精度向上**
   - 町丁固有の特性や年固有の要因を制御
   - イベントの純粋な効果を抽出

2. **内生性問題の解決**
   - 観察されない要因によるバイアスを軽減
   - より信頼性の高い係数推定

3. **パネルデータの特性を活用**
   - 時系列とクロスセクションの両方の情報を利用

### 実際の推定手順

1. **データ準備**: 町丁×年のパネルデータ
2. **固定効果の除去**: 個体平均・時点平均を差し引く
3. **回帰分析**: 除去後のデータで係数推定
4. **標準誤差**: クラスタロバスト標準誤差で調整

### 出力される係数の意味

- `β_housing`: 住宅イベント1単位あたりの人口変化
- `β_commercial`: 商業イベント1単位あたりの人口変化
- など

これらの係数は、**他の要因を制御した上での純粋なイベント効果**を表します。

### メリット・デメリット

**メリット**:
- 因果推論に適している
- 観察されない要因を制御
- パネルデータの特性を活用

**デメリット**:
- 計算が複雑
- データの要件が厳しい
- 解釈に注意が必要

このプロジェクトでは、TWFEにより各イベントが人口変化に与える**因果効果**を定量的に測定し、それをlayer4の機械学習モデルやlayer5の予測サービスで活用しています。

```plaintext
Y_{i,t} = α_i + τ_t + βX_{i,t} + ε_{i,t}
```

---

**User**

推論時の計算式は？βをどう使うの？

---

**Cursor**

推論時の計算式とβの使い方について、layer4とlayer5のコードを確認して詳しく説明しますね。


推論時の計算式とβの使い方について詳しく説明しますね。

## 推論時の計算式

### 1. 期待効果の計算（Layer4）

layer3で算出されたβ係数を使って、**期待効果**を計算します：

```python
# 基本計算式（build_expected_effects.py より）
eff_h1 = β_t * event_*_t + β_t1 * event_*_t1
eff_h2 = DECAY_H2 * eff_h1  # 減衰率0.5
eff_h3 = DECAY_H3 * eff_h1  # 減衰率0.25
```

**具体例**：
- `exp_housing_inc_h1 = β_housing_inc_t * event_housing_inc_t + β_housing_inc_t1 * event_housing_inc_t1`
- `exp_housing_inc_h2 = 0.5 * exp_housing_inc_h1`（1年後は50%減衰）
- `exp_housing_inc_h3 = 0.25 * exp_housing_inc_h1`（2年後は25%減衰）

### 2. 人口予測の計算（Layer5）

layer4の機械学習モデルを使って、将来の人口変化を予測：

```python
# 人口パスの計算（forecast_service.py より）
pop_{t+1} = pop_t + delta_hat_{h1}
pop_{t+2} = pop_{t+1} + delta_hat_{h2}
pop_{t+3} = pop_{t+2} + delta_hat_{h3}
```

ここで：
- `delta_hat_{h1}`: 1年後の人口変化予測
- `delta_hat_{h2}`: 2年後の人口変化予測
- `delta_hat_{h3}`: 3年後の人口変化予測

### 3. 寄与分解

予測結果を以下の要素に分解：

```python
# 寄与分解（forecast_service.py より）
contrib_exp = exp_rate_all_h1 + exp_rate_all_h2 + exp_rate_all_h3  # イベント効果
contrib_macro = macro_shock + foreign_change  # マクロ要因
contrib_inertia = lag_d1 + lag_d2  # 慣性（過去の変化）
contrib_other = delta_hat - contrib_exp - contrib_macro - contrib_inertia  # その他
```

## βの使い方の流れ

### 1. **Layer3（TWFE推定）**
```
イベント強度 → β係数（因果効果）
```

### 2. **Layer4（期待効果計算）**
```
β係数 × イベント強度 → 期待効果特徴量
```

### 3. **Layer5（人口予測）**
```
期待効果特徴量 → 機械学習モデル → 人口変化予測
```

## 実際の計算例

### 住宅イベントの場合：

1. **Layer3**: `β_housing_inc = 1.693`（住宅1単位あたり1.693人増加）

2. **Layer4**: 
   - 住宅イベント強度が100戸の場合
   - `exp_housing_inc_h1 = 1.693 × 100 = 169.3人`
   - `exp_housing_inc_h2 = 0.5 × 169.3 = 84.65人`
   - `exp_housing_inc_h3 = 0.25 × 169.3 = 42.33人`

3. **Layer5**: 
   - 機械学習モデルが期待効果を入力として人口変化を予測
   - 予測結果を各要因に分解して表示

## 重要なポイント

1. **時間的分解**: βは当年（t）と翌年（t+1）に分けて推定
2. **減衰効果**: イベント効果は時間とともに減衰（h2: 50%, h3: 25%）
3. **寄与分解**: 予測結果をイベント効果、マクロ要因、慣性、その他に分解
4. **率ベース**: 最終的には人口変化率（%）で表示

このように、β係数は**イベントの因果効果を定量的に表現**し、それを基に**将来の人口変化を予測**するための重要なパラメータとして機能しています。

```python
# 基本計算式（build_expected_effects.py より）
eff_h1 = β_t * event_*_t + β_t1 * event_*_t1
eff_h2 = DECAY_H2 * eff_h1  # 減衰率0.5
eff_h3 = DECAY_H3 * eff_h1  # 減衰率0.25
```

```python
# 人口パスの計算（forecast_service.py より）
pop_{t+1} = pop_t + delta_hat_{h1}
pop_{t+2} = pop_{t+1} + delta_hat_{h2}
pop_{t+3} = pop_{t+2} + delta_hat_{h3}
```

```python
# 寄与分解（forecast_service.py より）
contrib_exp = exp_rate_all_h1 + exp_rate_all_h2 + exp_rate_all_h3  # イベント効果
contrib_macro = macro_shock + foreign_change  # マクロ要因
contrib_inertia = lag_d1 + lag_d2  # 慣性（過去の変化）
contrib_other = delta_hat - contrib_exp - contrib_macro - contrib_inertia  # その他
```

```plaintext
イベント強度 → β係数（因果効果）
```

```plaintext
β係数 × イベント強度 → 期待効果特徴量
```

```plaintext
期待効果特徴量 → 機械学習モデル → 人口変化予測
```

---

**User**

イベント強度って何？

---

**Cursor**

イベント強度について、layer3のコードを確認して詳しく説明しますね。


## イベント強度とは

**イベント強度**は、各町丁・各年において発生したイベントの**規模や重要度を数値化した指標**です。

### データの構造

#### 1. **events_labeled.csv**（元データ）
```csv
town,year,event_type,confidence,intensity,lag_t,lag_t1,effect_direction
八幡11丁目,2012,housing,1.0,1.0,1,1,increase
九品寺5丁目,2019,housing,1.0,1.0,1,1,increase
```

- `intensity`: イベントの強度（通常は1.0）
- `event_type`: イベントの種類（housing, commercial, transit等）
- `effect_direction`: 効果の方向（increase, decrease）

#### 2. **events_matrix_signed.csv**（変換後）
```csv
town,year,event_housing_inc_t,event_housing_inc_t1,event_commercial_inc_t,...
上南部町,2001,0.0,0.0,0.0,...
上熊本1丁目,2021,1.0,1.0,1.0,...
```

- `event_*_t`: 当年のイベント強度
- `event_*_t1`: 翌年のイベント強度

### イベント強度の意味

#### 1. **基本的な値**
- **1.0**: イベントが発生（通常の強度）
- **0.0**: イベントが発生していない
- **-1.0**: 減少イベント（例：施設閉鎖）

#### 2. **カテゴリ別の単位**
- **housing**: 戸数（住宅開発件数）
- **commercial**: 店舗数（商業施設数）
- **transit**: 駅・バス停数（交通インフラ）
- **employment**: 雇用者数（就業機会）
- **disaster**: 災害規模（1=発生、0=未発生）
- **public_edu_medical**: 施設数（公共・教育・医療施設）

### 時間的な分解

#### 1. **lag_t（当年効果）**
- イベントが発生した年の効果
- 例：2020年に住宅開発 → `event_housing_inc_t = 1.0`

#### 2. **lag_t1（翌年効果）**
- イベント発生の翌年の効果
- 例：2020年に住宅開発 → `event_housing_inc_t1 = 1.0`

### 実際の使用例

#### 1. **TWFE推定での使用**
```python
# 回帰式
ΔPop_{i,t} = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1 + ...
```

#### 2. **期待効果の計算**
```python
# 期待効果
exp_housing_inc_h1 = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1
```

### データの特徴

#### 1. **スパース性**
- ほとんどの町丁・年でイベント強度は0
- イベントが発生した場合のみ1.0

#### 2. **方向性**
- `_inc`: 増加イベント（住宅開発、商業施設開設等）
- `_dec`: 減少イベント（施設閉鎖、災害等）

#### 3. **時間的持続性**
- イベントは当年（t）と翌年（t+1）の両方に影響
- 例：住宅開発 → 建設中（t年）と完成後（t+1年）

### 解釈のポイント

1. **係数との関係**: β係数は「イベント強度1単位あたりの人口変化人数」
2. **スケール**: カテゴリごとに単位が異なる（戸数、店舗数等）
3. **時間効果**: 当年と翌年で異なる効果を持つ
4. **方向性**: 増加・減少を明確に区別

このように、イベント強度は**各イベントの発生規模を数値化した指標**であり、TWFE推定や人口予測の重要な入力データとして機能しています。

```csv
town,year,event_type,confidence,intensity,lag_t,lag_t1,effect_direction
八幡11丁目,2012,housing,1.0,1.0,1,1,increase
九品寺5丁目,2019,housing,1.0,1.0,1,1,increase
```

```csv
town,year,event_housing_inc_t,event_housing_inc_t1,event_commercial_inc_t,...
上南部町,2001,0.0,0.0,0.0,...
上熊本1丁目,2021,1.0,1.0,1.0,...
```

```python
# 回帰式
ΔPop_{i,t} = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1 + ...
```

```python
# 期待効果
exp_housing_inc_h1 = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1
```

---

**User**

つまりどういうこと？あなたがいうようにイベント強度が100でexp_housing_inc_h1 = 1.693 × 100 = 169.3人　こんなようになる例はないってこと？

---

**Cursor**

はい、その通りです！私の説明が不正確でした。

## 実際のイベント強度の値

このプロジェクトでは、イベント強度は**0.0か1.0の二値**です：

- **0.0**: イベントが発生していない
- **1.0**: イベントが発生している

## 正しい計算例

### 実際のケース：
```python
# イベント強度（実際の値）
event_housing_inc_t = 1.0    # 住宅イベント発生
event_housing_inc_t1 = 1.0   # 翌年も効果あり

# β係数
β_housing_inc_t = 1.693
β_housing_inc_t1 = 0.0

# 期待効果の計算
exp_housing_inc_h1 = 1.693 × 1.0 + 0.0 × 1.0 = 1.693人
```

### 私の間違った例：
```python
# これは間違い（100戸のような値は存在しない）
exp_housing_inc_h1 = 1.693 × 100 = 169.3人  # ❌
```

## 実際のデータ構造

### events_matrix_signed.csv の例：
```csv
town,year,event_housing_inc_t,event_housing_inc_t1
上熊本1丁目,2021,1.0,1.0  # 住宅イベント発生
上南部町,2001,0.0,0.0     # イベントなし
```

## 正しい解釈

1. **β係数の意味**: 「住宅イベントが発生した場合の人口変化人数」
2. **期待効果**: イベント発生時のみ1.693人増加
3. **スケール**: イベント強度は0/1の二値、β係数は人数

## なぜ二値なのか？

1. **データの性質**: イベントの発生/未発生を記録
2. **TWFE推定**: 二値変数として因果効果を測定
3. **解釈の簡潔性**: イベント発生時の効果を直接表現

つまり、**イベント強度100のような連続値は存在せず、常に0か1の二値**です。私の最初の説明は誤解を招くものでした。

```python
# イベント強度（実際の値）
event_housing_inc_t = 1.0    # 住宅イベント発生
event_housing_inc_t1 = 1.0   # 翌年も効果あり

# β係数
β_housing_inc_t = 1.693
β_housing_inc_t1 = 0.0

# 期待効果の計算
exp_housing_inc_h1 = 1.693 × 1.0 + 0.0 × 1.0 = 1.693人
```

```python
# これは間違い（100戸のような値は存在しない）
exp_housing_inc_h1 = 1.693 × 100 = 169.3人  # ❌
```

```csv
town,year,event_housing_inc_t,event_housing_inc_t1
上熊本1丁目,2021,1.0,1.0  # 住宅イベント発生
上南部町,2001,0.0,0.0     # イベントなし
```

---

**User**

それで2.Layer4（期待効果計算）は何をしてるの？Layer4はLightGBMの学習をしているところだと思うけど，何かイベントとの因果関係を学習しているの？

---

**Cursor**

良い質問ですね！Layer4の役割について詳しく説明します。


## Layer4の役割について

Layer4は**2つの異なるアプローチ**を実装しています：

### 1. **従来の期待効果ベースアプローチ**（現在のメイン）

#### 何をしているか：
```python
# build_expected_effects.py で事前計算
exp_housing_inc_h1 = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1
exp_housing_inc_h2 = 0.5 * exp_housing_inc_h1  # 減衰
exp_housing_inc_h3 = 0.25 * exp_housing_inc_h1  # 減衰
```

#### LightGBMの学習：
- **入力**: 事前計算された期待効果特徴量（`exp_*_h*`）
- **目的変数**: `delta_people`（人口変化）
- **学習内容**: 期待効果から人口変化への**非線形な関係**を学習

#### 因果関係の学習：
- **❌ 直接的な因果関係は学習していない**
- **✅ 期待効果の重み付けや組み合わせを学習**
- **✅ 他の要因（マクロ、空間、時間）との相互作用を学習**

### 2. **新しい因果関係学習アプローチ**（improved/ディレクトリ）

#### 何をしているか：
```python
# 生のイベントデータを直接使用
features = [
    "event_housing_inc_t", "event_housing_inc_t1",
    "event_commercial_inc_t", "event_commercial_inc_t1",
    # ... 他の生のイベントデータ
]
```

#### LightGBMの学習：
- **入力**: 生のイベントデータ（`event_*_*_t`）
- **目的変数**: `delta_people`
- **学習内容**: **真の因果関係**を学習

## 従来アプローチの問題点

### 1. **因果関係の学習不足**
```
Layer3: イベント → β係数（因果効果）
Layer4: β係数 → 期待効果（事前計算）
LightGBM: 期待効果 → 人口変化（非線形関係のみ）
```

### 2. **生データの情報損失**
- イベントの詳細な情報が失われる
- 時空間的な相互作用を学習できない
- 真の因果メカニズムを理解できない

## 新しいアプローチの利点

### 1. **真の因果関係学習**
```
LightGBM: 生のイベントデータ → 人口変化（直接的な因果関係）
```

### 2. **豊富な特徴量**
- 生のイベントデータ
- 時空間的相互作用特徴量
- マクロ要因との相互作用

## まとめ

**Layer4は因果関係を学習しているが、アプローチによって異なる**：

1. **従来**: 事前計算された期待効果から非線形関係を学習
2. **新方式**: 生のイベントデータから真の因果関係を学習

現在のメインシステムは従来方式ですが、`improved/`ディレクトリで真の因果関係学習を実装しています。

```python
# build_expected_effects.py で事前計算
exp_housing_inc_h1 = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1
exp_housing_inc_h2 = 0.5 * exp_housing_inc_h1  # 減衰
exp_housing_inc_h3 = 0.25 * exp_housing_inc_h1  # 減衰
```

```python
# 生のイベントデータを直接使用
features = [
    "event_housing_inc_t", "event_housing_inc_t1",
    "event_commercial_inc_t", "event_commercial_inc_t1",
    # ... 他の生のイベントデータ
]
```

```plaintext
Layer3: イベント → β係数（因果効果）
Layer4: β係数 → 期待効果（事前計算）
LightGBM: 期待効果 → 人口変化（非線形関係のみ）
```

```plaintext
LightGBM: 生のイベントデータ → 人口変化（直接的な因果関係）
```

---

**User**

いや，/ablation とか/improvedとかの更にディレクトリのなかにあるようなファイルは無視していいよ

---

**Cursor**

了解しました！`/ablation`や`/improved`などのサブディレクトリは無視して、メインのLayer4について説明しますね。

## Layer4の実際の役割

### 1. **期待効果の事前計算**（`build_expected_effects.py`）

```python
# Layer3のβ係数を使って期待効果を計算
exp_housing_inc_h1 = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1
exp_housing_inc_h2 = 0.5 * exp_housing_inc_h1  # 減衰
exp_housing_inc_h3 = 0.25 * exp_housing_inc_h1  # 減衰
```

### 2. **LightGBMでの学習**（`train_lgbm.py`）

#### 入力特徴量：
- **期待効果特徴量**: `exp_*_h1`, `exp_*_h2`, `exp_*_h3`
- **空間ラグ特徴量**: `ring1_exp_*_h*`（近隣町丁の期待効果）
- **マクロ要因**: `foreign_population`, `macro_shock`等
- **時間特徴量**: `lag_d1`, `lag_d2`, `era_*`等

#### 目的変数：
- `delta_people`（人口変化）

#### 学習内容：
- **期待効果から人口変化への非線形な関係**
- **マクロ要因との相互作用**
- **空間的な波及効果**
- **時間的な慣性効果**

## 因果関係の学習について

### **Layer4は因果関係を学習している**

ただし、**2段階の因果関係**：

1. **Layer3**: イベント → 人口変化（TWFE推定）
2. **Layer4**: 期待効果 + 他の要因 → 人口変化（機械学習）

### 具体例：

```python
# Layer3で推定された因果効果
β_housing_inc = 1.693  # 住宅イベント1単位あたり1.693人増加

# Layer4で学習される関係
if exp_housing_inc_h1 > 0 and era_covid == 1:
    # COVID期の住宅効果は通常の0.8倍
    delta_people = 0.8 * exp_housing_inc_h1 + other_factors
elif ring1_exp_housing_inc_h1 > 0:
    # 近隣の住宅効果は0.3倍で波及
    delta_people = 0.3 * ring1_exp_housing_inc_h1 + other_factors
```

## まとめ

**Layer4は因果関係を学習している**：

- **直接的な因果関係**: Layer3で既に推定済み
- **間接的な因果関係**: マクロ要因、空間効果、時間効果との相互作用
- **非線形な因果関係**: 期待効果の重み付けや組み合わせ

つまり、Layer4は**Layer3の因果効果を基盤として、より複雑な因果メカニズムを学習**しているということです。

```python
# Layer3のβ係数を使って期待効果を計算
exp_housing_inc_h1 = β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1
exp_housing_inc_h2 = 0.5 * exp_housing_inc_h1  # 減衰
exp_housing_inc_h3 = 0.25 * exp_housing_inc_h1  # 減衰
```

```python
# Layer3で推定された因果効果
β_housing_inc = 1.693  # 住宅イベント1単位あたり1.693人増加

# Layer4で学習される関係
if exp_housing_inc_h1 > 0 and era_covid == 1:
    # COVID期の住宅効果は通常の0.8倍
    delta_people = 0.8 * exp_housing_inc_h1 + other_factors
elif ring1_exp_housing_inc_h1 > 0:
    # 近隣の住宅効果は0.3倍で波及
    delta_people = 0.3 * ring1_exp_housing_inc_h1 + other_factors
```

---

**User**

つまり，イベントとの因果関係は学習してないんだよね？

---

**Cursor**

はい、その通りです！

## Layer4はイベントとの因果関係を学習していない

### 実際の流れ：

1. **Layer3（TWFE）**: イベント → 人口変化の**因果関係を推定**
   ```python
   # イベント強度からβ係数を推定
   ΔPop = β_housing * event_housing + β_commercial * event_commercial + ...
   ```

2. **Layer4（LightGBM）**: 期待効果 → 人口変化の**予測モデルを学習**
   ```python
   # 事前計算された期待効果から人口変化を予測
   ΔPop = f(exp_housing_h1, exp_housing_h2, exp_housing_h3, other_factors)
   ```

### 重要なポイント：

- **Layer3**: イベントの**因果効果**を推定（β係数）
- **Layer4**: 期待効果の**予測精度**を向上（非線形関係、相互作用）

### Layer4が学習しているのは：

1. **期待効果の重み付け**
   - どの期待効果が重要か
   - 時間的な減衰パターン

2. **他の要因との相互作用**
   - マクロ要因との相乗効果
   - 空間的な波及効果
   - 時間的な慣性効果

3. **非線形な関係**
   - 期待効果の組み合わせ
   - 閾値効果

### つまり：

**Layer4は因果関係を学習せず、Layer3で推定された因果関係を「使って」予測精度を向上させている**ということです。

イベント → 人口変化の因果関係は、Layer3のTWFE推定で既に決まっており、Layer4はその結果をより効果的に活用する方法を学習しているだけです。

```python
   # イベント強度からβ係数を推定
   ΔPop = β_housing * event_housing + β_commercial * event_commercial + ...
```

```python
   # 事前計算された期待効果から人口変化を予測
   ΔPop = f(exp_housing_h1, exp_housing_h2, exp_housing_h3, other_factors)
```

---

**User**

もう一度，元の話に戻るけど，じゃあ，layer5でイベントと町長を選択して将来を予測してるけど，その計算式は？

---

**Cursor**

layer5での将来予測の計算式について、コードを確認して詳しく説明しますね。


## Layer5での将来予測の計算式

### 1. **基本の予測フロー**

```python
# 1. 将来特徴量の準備
future_features = build_future_features(town, events, base_year)

# 2. 逐次予測の実行
for year in [base_year+1, base_year+2, base_year+3]:
    # 3. 期待効果の計算
    exp_rate_all_h1 = Σ(β_t * event_*_t + β_t1 * event_*_t1) / base_population
    exp_rate_all_h2 = 0.5 * exp_rate_all_h1
    exp_rate_all_h3 = 0.25 * exp_rate_all_h1
    
    # 4. 機械学習モデルで予測
    delta_hat = model.predict(future_features[year])
    
    # 5. 人口パスの更新
    pop_{year} = pop_{year-1} + delta_hat
```

### 2. **期待効果の計算式**

#### 率ベースの期待効果：
```python
# 各イベントカテゴリの期待効果（率）
exp_housing_inc_h1 = (β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1) / base_population
exp_commercial_inc_h1 = (β_commercial_t * event_commercial_inc_t + β_commercial_t1 * event_commercial_inc_t1) / base_population

# 合計期待効果（率）
exp_rate_all_h1 = exp_housing_inc_h1 + exp_commercial_inc_h1 + ... + exp_disaster_inc_h1

# 減衰効果
exp_rate_all_h2 = 0.5 * exp_rate_all_h1
exp_rate_all_h3 = 0.25 * exp_rate_all_h1
```

#### 人数ベースの期待効果：
```python
# 期待効果を人数に変換
exp_people_from_rate = exp_rate_all_h1 * base_population
exp_people_manual = manual_add_by_h[horizon]  # 手動追加

# 総期待効果
exp_people_total = exp_people_from_rate + exp_people_manual
```

### 3. **最終的な人口変化予測**

```python
# 機械学習モデルによる予測（期待効果を除く）
y_noexp = model.predict(features_without_exp)

# 最終的な人口変化
delta_hat = y_noexp + exp_people_total

# 人口パス更新
pop_{t+1} = pop_t + delta_hat
```

### 4. **寄与分解**

```python
# 各要因への寄与分解
contrib_exp = exp_people_total        # イベント効果
contrib_macro = macro_effects         # マクロ要因
contrib_inertia = lag_effects         # 慣性（過去の変化）
contrib_other = delta_hat - (contrib_exp + contrib_macro + contrib_inertia)
```

### 5. **実際の計算例**

#### 住宅イベントが発生した場合：
```python
# イベント強度
event_housing_inc_t = 1.0
event_housing_inc_t1 = 1.0

# β係数（Layer3で推定済み）
β_housing_t = 1.693
β_housing_t1 = 0.0

# 期待効果（率）
exp_housing_inc_h1 = (1.693 * 1.0 + 0.0 * 1.0) / 1000 = 0.001693

# 期待効果（人数）
exp_people_h1 = 0.001693 * 1000 = 1.693人

# 減衰効果
exp_people_h2 = 0.5 * 1.693 = 0.847人
exp_people_h3 = 0.25 * 1.693 = 0.423人
```

### 6. **逐次予測の特徴**

- **ラグ更新**: 前年の予測結果を次の年の特徴量として使用
- **時系列依存**: 過去の人口変化が将来に影響
- **不確実性**: 予測区間（95%信頼区間）も計算

このように、Layer5では**Layer3のβ係数を使って期待効果を計算し、Layer4の機械学習モデルで他の要因を予測し、それらを組み合わせて最終的な人口変化を予測**しています。

```python
# 1. 将来特徴量の準備
future_features = build_future_features(town, events, base_year)

# 2. 逐次予測の実行
for year in [base_year+1, base_year+2, base_year+3]:
    # 3. 期待効果の計算
    exp_rate_all_h1 = Σ(β_t * event_*_t + β_t1 * event_*_t1) / base_population
    exp_rate_all_h2 = 0.5 * exp_rate_all_h1
    exp_rate_all_h3 = 0.25 * exp_rate_all_h1
    
    # 4. 機械学習モデルで予測
    delta_hat = model.predict(future_features[year])
    
    # 5. 人口パスの更新
    pop_{year} = pop_{year-1} + delta_hat
```

```python
# 各イベントカテゴリの期待効果（率）
exp_housing_inc_h1 = (β_housing_t * event_housing_inc_t + β_housing_t1 * event_housing_inc_t1) / base_population
exp_commercial_inc_h1 = (β_commercial_t * event_commercial_inc_t + β_commercial_t1 * event_commercial_inc_t1) / base_population

# 合計期待効果（率）
exp_rate_all_h1 = exp_housing_inc_h1 + exp_commercial_inc_h1 + ... + exp_disaster_inc_h1

# 減衰効果
exp_rate_all_h2 = 0.5 * exp_rate_all_h1
exp_rate_all_h3 = 0.25 * exp_rate_all_h1
```

```python
# 期待効果を人数に変換
exp_people_from_rate = exp_rate_all_h1 * base_population
exp_people_manual = manual_add_by_h[horizon]  # 手動追加

# 総期待効果
exp_people_total = exp_people_from_rate + exp_people_manual
```

```python
# 機械学習モデルによる予測（期待効果を除く）
y_noexp = model.predict(features_without_exp)

# 最終的な人口変化
delta_hat = y_noexp + exp_people_total

# 人口パス更新
pop_{t+1} = pop_t + delta_hat
```

```python
# 各要因への寄与分解
contrib_exp = exp_people_total        # イベント効果
contrib_macro = macro_effects         # マクロ要因
contrib_inertia = lag_effects         # 慣性（過去の変化）
contrib_other = delta_hat - (contrib_exp + contrib_macro + contrib_inertia)
```

```python
# イベント強度
event_housing_inc_t = 1.0
event_housing_inc_t1 = 1.0

# β係数（Layer3で推定済み）
β_housing_t = 1.693
β_housing_t1 = 0.0

# 期待効果（率）
exp_housing_inc_h1 = (1.693 * 1.0 + 0.0 * 1.0) / 1000 = 0.001693

# 期待効果（人数）
exp_people_h1 = 0.001693 * 1000 = 1.693人

# 減衰効果
exp_people_h2 = 0.5 * 1.693 = 0.847人
exp_people_h3 = 0.25 * 1.693 = 0.423人
```

---

**User**

あともう少しで理解できそう．住宅イベントが発生した場合に期待効果を率で出してると思うけど，じゃあ，最終的に人口は何人と予測するの？具体例を説明して

---

**Cursor**

具体的な例で説明しますね！

## 住宅イベント発生時の人口予測の具体例

### 前提条件
- **町丁**: 上熊本1丁目
- **ベース年**: 2023年（人口1,000人）
- **住宅イベント**: 2024年に発生（`event_housing_inc_t = 1.0`）
- **β係数**: `β_housing_t = 1.693`, `β_housing_t1 = 0.0`

### 1. **期待効果の計算（率ベース）**

```python
# 2024年（h=1）の期待効果
exp_housing_inc_h1 = (1.693 × 1.0 + 0.0 × 1.0) / 1000 = 0.001693 (0.1693%)

# 2025年（h=2）の期待効果（減衰）
exp_housing_inc_h2 = 0.5 × 0.001693 = 0.0008465 (0.08465%)

# 2026年（h=3）の期待効果（減衰）
exp_housing_inc_h3 = 0.25 × 0.001693 = 0.00042325 (0.042325%)
```

### 2. **人数ベースの期待効果**

```python
# 2024年
exp_people_h1 = 0.001693 × 1000 = 1.693人

# 2025年
exp_people_h2 = 0.0008465 × 1000 = 0.847人

# 2026年
exp_people_h3 = 0.00042325 × 1000 = 0.423人
```

### 3. **機械学習モデルによる予測**

```python
# 期待効果を除いた他の要因の予測
y_noexp_2024 = -2.5人  # マクロ要因、慣性など
y_noexp_2025 = -1.8人
y_noexp_2026 = -1.2人
```

### 4. **最終的な人口変化予測**

```python
# 2024年
delta_hat_2024 = y_noexp_2024 + exp_people_h1 = -2.5 + 1.693 = -0.807人

# 2025年
delta_hat_2025 = y_noexp_2025 + exp_people_h2 = -1.8 + 0.847 = -0.953人

# 2026年
delta_hat_2026 = y_noexp_2026 + exp_people_h3 = -1.2 + 0.423 = -0.777人
```

### 5. **人口パスの更新**

```python
# 2023年（ベース年）
pop_2023 = 1000人

# 2024年
pop_2024 = pop_2023 + delta_hat_2024 = 1000 + (-0.807) = 999.2人

# 2025年
pop_2025 = pop_2024 + delta_hat_2025 = 999.2 + (-0.953) = 998.2人

# 2026年
pop_2026 = pop_2025 + delta_hat_2026 = 998.2 + (-0.777) = 997.4人
```

### 6. **寄与分解**

```python
# 2024年の寄与分解
contrib_exp = 1.693人      # 住宅イベント効果
contrib_macro = -3.2人     # マクロ要因（外国人減少など）
contrib_inertia = 0.7人    # 慣性（過去の変化の継続）
contrib_other = 0.0人      # その他

# 合計: 1.693 + (-3.2) + 0.7 + 0.0 = -0.807人
```

## まとめ

**住宅イベントが発生した場合の予測結果**：

- **2024年**: 999.2人（-0.8人減少）
- **2025年**: 998.2人（-1.0人減少）
- **2026年**: 997.4人（-0.8人減少）

**重要なポイント**：
1. **住宅イベントの効果**: 1.693人増加（1年目のみ）
2. **他の要因**: マクロ要因などで-2.5人減少
3. **最終結果**: 住宅イベントがあっても全体では減少
4. **時間的減衰**: 住宅効果は年々減衰（2年目50%、3年目25%）

このように、**住宅イベントの効果は正だが、他の要因（マクロ、慣性など）が負で、結果的に人口は減少**という予測になります。

```python
# 2024年（h=1）の期待効果
exp_housing_inc_h1 = (1.693 × 1.0 + 0.0 × 1.0) / 1000 = 0.001693 (0.1693%)

# 2025年（h=2）の期待効果（減衰）
exp_housing_inc_h2 = 0.5 × 0.001693 = 0.0008465 (0.08465%)

# 2026年（h=3）の期待効果（減衰）
exp_housing_inc_h3 = 0.25 × 0.001693 = 0.00042325 (0.042325%)
```

```python
# 2024年
exp_people_h1 = 0.001693 × 1000 = 1.693人

# 2025年
exp_people_h2 = 0.0008465 × 1000 = 0.847人

# 2026年
exp_people_h3 = 0.00042325 × 1000 = 0.423人
```

```python
# 期待効果を除いた他の要因の予測
y_noexp_2024 = -2.5人  # マクロ要因、慣性など
y_noexp_2025 = -1.8人
y_noexp_2026 = -1.2人
```

```python
# 2024年
delta_hat_2024 = y_noexp_2024 + exp_people_h1 = -2.5 + 1.693 = -0.807人

# 2025年
delta_hat_2025 = y_noexp_2025 + exp_people_h2 = -1.8 + 0.847 = -0.953人

# 2026年
delta_hat_2026 = y_noexp_2026 + exp_people_h3 = -1.2 + 0.423 = -0.777人
```

```python
# 2023年（ベース年）
pop_2023 = 1000人

# 2024年
pop_2024 = pop_2023 + delta_hat_2024 = 1000 + (-0.807) = 999.2人

# 2025年
pop_2025 = pop_2024 + delta_hat_2025 = 999.2 + (-0.953) = 998.2人

# 2026年
pop_2026 = pop_2025 + delta_hat_2026 = 998.2 + (-0.777) = 997.4人
```

```python
# 2024年の寄与分解
contrib_exp = 1.693人      # 住宅イベント効果
contrib_macro = -3.2人     # マクロ要因（外国人減少など）
contrib_inertia = 0.7人    # 慣性（過去の変化の継続）
contrib_other = 0.0人      # その他

# 合計: 1.693 + (-3.2) + 0.7 + 0.0 = -0.807人
```

---

**User**

これTWFEには何を入力して，何が出力されてるの？

---

**Cursor**

TWFEの入力と出力について詳しく説明しますね。


## TWFEの入力と出力

### **入力データ**

#### 1. **目的変数（ターゲット）**
```python
# 人口変化率（率ベースの場合）
y_rate = (pop_total - pop_prev) / max(pop_prev, 50)

# または人口変化数（絶対値ベースの場合）
delta = pop_total - pop_prev
```

#### 2. **説明変数（イベント強度）**
```python
# イベント強度データ（0/1の二値）
event_housing_inc_t      # 住宅増加イベント（当年）
event_housing_inc_t1     # 住宅増加イベント（翌年）
event_commercial_inc_t   # 商業増加イベント（当年）
event_commercial_inc_t1  # 商業増加イベント（翌年）
# ... 他のイベントカテゴリ
```

#### 3. **固定効果**
```python
# 町丁固定効果
C(town)  # 各町丁の固有特性

# 年固定効果  
C(year)  # 各年の共通要因
```

#### 4. **コントロール変数**
```python
# 事前トレンド制御
d1 = delta_people_{t-1}  # 1年前の人口変化
d2 = delta_people_{t-2}  # 2年前の人口変化

# その他のコントロール
growth_adj_log           # 成長調整
policy_boundary_dummy    # ポリシー境界ダミー
disaster_2016_dummy      # 2016年震災ダミー
```

### **回帰式**

```python
# 基本の回帰式
y_rate ~ 0 + C(town) + C(year) + 
         event_housing_inc_t + event_housing_inc_t1 +
         event_commercial_inc_t + event_commercial_inc_t1 +
         ... + 
         d1 + d2 + growth_adj_log + policy_boundary_dummy + disaster_2016_dummy
```

### **出力結果**

#### 1. **β係数（各イベントの因果効果）**
```python
# 住宅イベントの効果
β_housing_inc_t = 1.693   # 当年効果（1単位あたり1.693人増加）
β_housing_inc_t1 = 0.0    # 翌年効果

# 商業イベントの効果
β_commercial_inc_t = 0.0   # 当年効果
β_commercial_inc_t1 = 1.507 # 翌年効果（1単位あたり1.507人増加）
```

#### 2. **統計情報**
```python
# 各係数の統計
se_housing_inc_t = 0.123      # 標準誤差
pval_housing_inc_t = 0.001    # p値
ci_low_housing_inc_t = 1.452  # 95%信頼区間下限
ci_high_housing_inc_t = 1.934 # 95%信頼区間上限
```

#### 3. **モデル全体の統計**
```python
r2_within = 0.9887  # 決定係数（within）
nobs = 17381        # 観測数
```

### **具体例**

#### 入力データの例：
```csv
town,year,pop_total,pop_prev,y_rate,event_housing_inc_t,event_housing_inc_t1,event_commercial_inc_t,event_commercial_inc_t1
上熊本1丁目,2021,1000,995,0.005,1.0,1.0,0.0,0.0
上熊本1丁目,2022,1005,1000,0.005,0.0,1.0,1.0,1.0
```

#### 出力結果の例：
```csv
event_var,beta,se,pval,ci_low,ci_high,beta_t,beta_t1,gamma0,gamma1
housing_inc,1.693,0.123,0.001,1.452,1.934,1.693,0.000,1.000,0.000
commercial_inc,1.507,0.089,0.001,1.332,1.682,0.000,1.507,0.000,1.000
```

### **解釈**

- **β係数**: イベント強度1単位あたりの人口変化率（%）
- **β_t**: 当年の効果
- **β_t1**: 翌年の効果
- **gamma0, gamma1**: 効果の時間配分（和=1）

このように、TWFEは**イベント強度から人口変化への因果効果を定量的に推定**し、その結果をLayer4とLayer5で活用しています。

```python
# 人口変化率（率ベースの場合）
y_rate = (pop_total - pop_prev) / max(pop_prev, 50)

# または人口変化数（絶対値ベースの場合）
delta = pop_total - pop_prev
```

```python
# イベント強度データ（0/1の二値）
event_housing_inc_t      # 住宅増加イベント（当年）
event_housing_inc_t1     # 住宅増加イベント（翌年）
event_commercial_inc_t   # 商業増加イベント（当年）
event_commercial_inc_t1  # 商業増加イベント（翌年）
# ... 他のイベントカテゴリ
```

```python
# 町丁固定効果
C(town)  # 各町丁の固有特性

# 年固定効果  
C(year)  # 各年の共通要因
```

```python
# 事前トレンド制御
d1 = delta_people_{t-1}  # 1年前の人口変化
d2 = delta_people_{t-2}  # 2年前の人口変化

# その他のコントロール
growth_adj_log           # 成長調整
policy_boundary_dummy    # ポリシー境界ダミー
disaster_2016_dummy      # 2016年震災ダミー
```

```python
# 基本の回帰式
y_rate ~ 0 + C(town) + C(year) + 
         event_housing_inc_t + event_housing_inc_t1 +
         event_commercial_inc_t + event_commercial_inc_t1 +
         ... + 
         d1 + d2 + growth_adj_log + policy_boundary_dummy + disaster_2016_dummy
```

```python
# 住宅イベントの効果
β_housing_inc_t = 1.693   # 当年効果（1単位あたり1.693人増加）
β_housing_inc_t1 = 0.0    # 翌年効果

# 商業イベントの効果
β_commercial_inc_t = 0.0   # 当年効果
β_commercial_inc_t1 = 1.507 # 翌年効果（1単位あたり1.507人増加）
```

```python
# 各係数の統計
se_housing_inc_t = 0.123      # 標準誤差
pval_housing_inc_t = 0.001    # p値
ci_low_housing_inc_t = 1.452  # 95%信頼区間下限
ci_high_housing_inc_t = 1.934 # 95%信頼区間上限
```

```python
r2_within = 0.9887  # 決定係数（within）
nobs = 17381        # 観測数
```

```csv
town,year,pop_total,pop_prev,y_rate,event_housing_inc_t,event_housing_inc_t1,event_commercial_inc_t,event_commercial_inc_t1
上熊本1丁目,2021,1000,995,0.005,1.0,1.0,0.0,0.0
上熊本1丁目,2022,1005,1000,0.005,0.0,1.0,1.0,1.0
```

```csv
event_var,beta,se,pval,ci_low,ci_high,beta_t,beta_t1,gamma0,gamma1
housing_inc,1.693,0.123,0.001,1.452,1.934,1.693,0.000,1.000,0.000
commercial_inc,1.507,0.089,0.001,1.332,1.682,0.000,1.507,0.000,1.000
```

