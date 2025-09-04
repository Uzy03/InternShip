# イベントラベル付け基準書（Layer2）
**目的**：`manual_investigation_targets.csv` の各行を機械学習・効果推定で利用できる共通スキーマに正規化し、  
`events_labeled.csv`（ロング）および `events_matrix.csv`（ワイド）を生成するための判定基準を定義する。

---

## 0) スコープと原則
- **1レコード = 町丁 × 年 × イベントカテゴリ**。
- 1つの記述に複数カテゴリが含まれる場合は **行を分割**（例：商業＋交通）。
- 迷ったときは「優先順位」を参考に **1つへ丸める**か、明確でない場合は **`unknown`** とする。
- 方向（増減）は `effect_direction ∈ {increase, decrease, unknown}` で付与する。

---

## 1) イベント種別（`event_type`）と優先順位
**優先順位**：`disaster > policy_boundary > transit > employment > public_edu_medical > commercial > housing > unknown`

### 定義と包含例
- **housing**：マンション/団地/分譲・賃貸/区画整理/竣工/入居/建替え  
  例：「○○マンション竣工」「△△団地 120戸」「建替で新棟4棟」
- **commercial**：大型商業/モール/SC/百貨店/複合商業（サクラマチ等）  
  例：「サクラマチ開業」「ゆめタウン新規出店」
- **transit**：駅/バスターミナル/鉄道延伸/高架化/道路・IC/バイパス/都市計画道路  
  例：「新駅開業」「○○IC 供用」「電停移設」
- **public_edu_medical**：病院/診療所/学校/大学/庁舎/図書館/保育所・こども園  
  例：「市民病院移転開業」「小学校統合」「新庁舎供用」
- **employment**：工場/物流/倉庫/データセンター/操業/新規雇用  
  例：「××工場操業（雇用120人）」「物流センター新設」
- **disaster**：地震/豪雨/浸水/土砂/被災/罹災/倒壊/液状化  
  例：「2016地震で被災」「豪雨で浸水」
- **policy_boundary**：住居表示変更/町名変更/合併/編入/境界変更/地番変更  
  例：「住居表示変更により○丁目新設」「政令市区制施行」
- **unknown**：上記に当てはまらない、または情報が不十分

> 同一文で複数カテゴリに該当する場合は **分割**。  
> 例：「バスターミナル開業と商業棟開業」→ `transit` と `commercial` の **2行**。

---

## 2) 方向（`effect_direction`）
- **increase**：竣工、新築、完成、入居、開業、開設、開通、操業、供用開始 など
- **decrease**：解体、立退(ち退)、閉鎖、除却、撤去、取り壊し、一時休止 など
- **unknown**：いずれにも該当しない／判断できない

> **工事由来の一時的減少**（「工事のため立退き/閉鎖」等）は `decrease`。  
> 後年に `housing` や `commercial`、`transit` の `increase` へ置換されることが多い。

---

## 3) 強度（`intensity`）
- 自由記述から **数値＋単位** を抽出。見つからなければ **1.0**。
- **カテゴリ別の優先順位**：
  - housing：**戸/世帯** ＞ 棟 ＞ 区画
  - employment：**人/名（雇用）**
  - commercial / public_edu_medical：**床面積 m²/㎡**（無ければ雇用）
  - disaster：被災 **戸数/人** 等があれば採用
  - transit / policy_boundary：通常 **1.0**（固定）
- 表記ゆれ（全角/半角/「約」）は正規化。範囲表現（「50～60戸」）は **平均値**（=55）で代入可。
- **外れ値は `cap=10000` でクリップ**。

---

## 4) ラグ（`lag_t`, `lag_t1`）
- **housing**：`t=1`, `t+1=1`（竣工年＋翌年波及）
- **commercial**：`t=1`, `t+1=1`
- **public_edu_medical**：`t=0`, `t+1=1`（開設効果が翌年寄り）
- **employment**：`t=0`, `t+1=1`（操業・雇用が翌年寄り）
- **transit**：`t=0`, `t+1=1`（供用開始の翌年寄与が大きい）
- **disaster**：`t=1`, `t+1=0`（当年ショック）
- **policy_boundary**：`t=1`, `t+1=0`（見かけの変化）

**例外**：`decrease` かつ「工事/立退き/閉鎖」等を含む場合は **当年効く** → `t=1` を強制ON。

---

## 5) 優先・排他・分割のルール
- **優先衝突**：`disaster` と `policy_boundary` は最優先（単独でも必ず残す）。
- **分割**：文に `transit` と `commercial` が併記 → 2行に分割（強度が共有なら同値で可）。
- **重複（同一 `event_type` が同年に複数）**：`confidence` 最大の1行を採用、`intensity` は **最大** を採用（初期運用）。
  - 必要に応じて「合算」に切替可能（運用ポリシー次第）。

---

## 6) 信頼度（`confidence`）
- 既定：**1.0**
- `note` に「要確認/？/未確」等がある → **0.7** に下げ
- `source_name` が官公庁/自治体/公報 → **+0.1**（上限1.0）
- まとめサイト/個人ブログ等 → **-0.2**（下限0.1）
- 最終的に **[0.1, 1.0]** にクリップする。

---

## 7) 具体例（入力→出力）
1. **「エイルマンション◯◯ 120戸 竣工」**  
   → `event_type=housing`, `effect_direction=increase`, `intensity=120`, `lag_t=1, lag_t1=1`, `confidence=1.0`

2. **「サクラマチクマモト開業、バスターミナル供用」**  
   → **2行に分割**  
   - `event_type=commercial`, `increase`, `intensity=1`, `lag_t=1, lag_t1=1`  
   - `event_type=transit`, `increase`, `intensity=1`, `lag_t=0, lag_t1=1`

3. **「都市計画道路のため立退き開始」**  
   → `event_type=transit`, `effect_direction=decrease`, `intensity=1`, **`lag_t=1, lag_t1=1`（工事例外）**

4. **「住居表示変更で○丁目新設」**  
   → `event_type=policy_boundary`, `effect_direction=unknown`, `intensity=1`, `lag_t=1, lag_t1=0`

5. **「2016年地震で被災」**  
   → `event_type=disaster`, `effect_direction=decrease`, `intensity=1`, `lag_t=1, lag_t1=0`

---

## 8) 出力スキーマ
### ロング：`events_labeled.csv`
```
town, year, event_type, confidence, intensity, lag_t, lag_t1, effect_direction, source_url, source_name, publish_date
```

### ワイド：`events_matrix.csv`
- キー：`town, year`
- 列：各カテゴリの `event_{type}_t`, `event_{type}_t1`（値＝強度。該当なしは0）
- 例：`event_housing_t, event_housing_t1, event_commercial_t, ...`

> オプション：`use_signed_intensity=true` の場合、`effect_direction=decrease` を **負値**で格納。既定は非負。

---

## 9) QAチェックリスト
- `(town, year)` が `panel_raw.csv` に存在するか（不一致はログに記録）。
- `event_type ∈ {housing, employment, public_edu_medical, commercial, transit, disaster, policy_boundary, unknown}` のみ。
- `confidence ∈ [0.1, 1.0]`、`lag_t/lag_t1 ∈ {0,1}`。
- `(town, year, event_type)` の重複が **ない**。
- `disaster / policy_boundary` が他カテゴリで **上書きされていない**。

---

## 10) CSV 推奨列（入力→正規化のため）
```
town, year, raw_cause_text, event_type, effect_direction, intensity, confidence,
lag_t, lag_t1, source_url, source_name, publish_date, note
```
