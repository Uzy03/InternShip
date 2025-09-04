# Layer5 シナリオJSON仕様

## 概要
Layer5では、任意の町丁×基準年×シナリオを入力として、1〜3年先の人口予測（点推定／予測区間）と簡易寄与分解（イベント／マクロ／慣性）を返します。

## シナリオJSON仕様

### 必須キー

```json
{
  "town": "九品寺5丁目",
  "base_year": 2025,
  "horizons": [1, 2, 3],
  "events": [
    {
      "year_offset": 0,
      "event_type": "housing",
      "effect_direction": "increase",
      "confidence": 1.0,
      "intensity": 1.0,
      "lag_t": 1,
      "lag_t1": 1,
      "note": "分譲MS竣工・入居開始（例）"
    }
  ],
  "macros": {
    "foreign_population_growth_pct": { "h1": 0.05, "h2": 0.03, "h3": 0.02 }
  },
  "manual_delta": { "h1": 0, "h2": 0, "h3": 0 }
}
```

### フィールド詳細

#### 基本情報
- `town` (string): 町丁名。features_panel.csvのtown列に存在する必要があります
- `base_year` (int): 基準年。features_panel.csvのyear範囲内である必要があります
- `horizons` (array): 予測期間。{1,2,3}の部分集合（順不同可）

#### イベント設定
- `events` (array): イベントの配列
  - `year_offset` (int): 基準年からの年オフセット
  - `event_type` (string): イベントタイプ。以下のいずれか
    - `housing`: 住宅関連
    - `commercial`: 商業施設
    - `transit`: 交通インフラ
    - `policy_boundary`: 政策・境界変更
    - `public_edu_medical`: 公共・教育・医療
    - `employment`: 雇用関連
    - `disaster`: 災害
  - `effect_direction` (string): 効果の方向。`increase` または `decrease`
  - `confidence` (float): 信頼度 [0,1]
  - `intensity` (float): 強度 [0,1]
  - `lag_t` (int): 当年の効果 {0,1}
  - `lag_t1` (int): 翌年の効果 {0,1}
  - `note` (string): 備考（任意）

#### マクロ設定
- `macros` (object): マクロ経済変数
  - `foreign_population_growth_pct` (object): 外国人人口成長率
    - `h1`, `h2`, `h3` (float): 各予測期間の年次成長率

#### 手動調整
- `manual_delta` (object): 直接的な人口変化（人ベース）
  - `h1`, `h2`, `h3` (int): 各予測期間の直接加算値

### バリデーションルール

#### エラー条件
1. `town`がfeatures_panel.csvのtown列に存在しない
2. `base_year`がfeatures_panel.csvのyear範囲外
3. `horizons`が{1,2,3}の部分集合でない
4. `event_type`が許容集合外
5. `effect_direction`が{increase,decrease}でない
6. `confidence`, `intensity`が[0,1]の範囲外
7. `lag_t`, `lag_t1`が{0,1}でない

#### 衝突ルール
- 同一(town, base_year+year_offset)に`policy_boundary`と`transit`が併存した場合、`policy_boundary`を優先し`transit`を無効化

### 出力仕様

#### 予測結果JSON
```json
{
  "town": "九品寺5丁目",
  "base_year": 2025,
  "horizons": [1,2,3],
  "path": [
    {
      "year": 2026,
      "delta_hat": 45.2,
      "pop_hat": 7123.2,
      "pi95": [7109.0, 7138.0],
      "contrib": {
        "exp": 28.0,
        "macro": 6.0,
        "inertia": 10.0,
        "other": 1.2
      }
    }
  ]
}
```

#### 寄与分解
- `exp`: イベント期待効果
- `macro`: マクロ経済要因（主に外国人人口）
- `inertia`: 慣性要因（ラグ変数）
- `other`: その他（残差）

### サンプルシナリオ

#### housing_boost.json
住宅開発による人口増加シナリオ

#### factory_case.json
工場設置による雇用創出シナリオ（manual_delta使用）
