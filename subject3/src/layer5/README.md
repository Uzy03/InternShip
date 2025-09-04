# Layer5: 人口予測サービス

Layer5は、任意の町丁×基準年×シナリオを入力として、1〜3年先の人口予測（点推定／予測区間）と簡易寄与分解（イベント／マクロ／慣性）を提供します。

## ディレクトリ構成

```
subject3/src/layer5/
├── schema.md                  # シナリオJSONの仕様
├── scenario_examples/         # シナリオ例
│   ├── housing_boost.json    # 住宅開発シナリオ
│   └── factory_case.json     # 工場設置シナリオ
├── scenario_to_events.py     # シナリオ→将来イベント行列
├── prepare_baseline.py       # 基準年の土台準備
├── build_future_features.py  # 将来特徴の合成
├── forecast_service.py       # 予測本体
├── intervals.py              # 予測区間推定
├── cli_run_scenario.py       # CLIエントリポイント
└── README.md                 # このファイル
```

## 使用方法

### 基本的な使用方法

```bash
# シナリオを実行して人口予測を取得
python cli_run_scenario.py scenario_examples/housing_boost.json

# 出力ファイルを指定
python cli_run_scenario.py scenario_examples/factory_case.json output/forecast_result.json
```

### 個別モジュールの使用

```python
# 1. シナリオから将来イベント行列を生成
from scenario_to_events import scenario_to_events
events_df = scenario_to_events(scenario)

# 2. 基準年データを準備
from prepare_baseline import prepare_baseline
baseline = prepare_baseline(town, base_year)

# 3. 将来特徴を構築
from build_future_features import build_future_features
features_df = build_future_features(baseline, events_df, scenario)

# 4. 人口予測を実行
from forecast_service import forecast_population
result = forecast_population(town, base_year, horizons, base_population)
```

## シナリオJSON仕様

詳細は `schema.md` を参照してください。

### 必須キー
- `town`: 町丁名
- `base_year`: 基準年
- `horizons`: 予測期間 [1,2,3]の部分集合
- `events`: イベント配列
- `macros`: マクロ経済変数
- `manual_delta`: 直接的な人口変化

### イベントタイプ
- `housing`: 住宅関連
- `commercial`: 商業施設
- `transit`: 交通インフラ
- `policy_boundary`: 政策・境界変更
- `public_edu_medical`: 公共・教育・医療
- `employment`: 雇用関連
- `disaster`: 災害

## 出力形式

### 予測結果JSON
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

### 寄与分解
- `exp`: イベント期待効果
- `macro`: マクロ経済要因（主に外国人人口）
- `inertia`: 慣性要因（ラグ変数）
- `other`: その他（残差）

## 中間成果物

- `data/processed/l5_future_events.csv`: 将来イベント行列
- `data/processed/l5_future_features.csv`: 将来特徴
- `data/processed/l5_forecast_<scenario>.json`: 予測結果
- `data/processed/l5_debug_*.csv`: デバッグ用ファイル

## 依存関係

- Layer4の学習済みモデル (`models/l4_model.joblib`)
- Layer3の効果係数 (`output/effects_coefficients.csv`)
- パネルデータ (`data/processed/features_panel.csv`)
- 年別メトリクス (`data/processed/l4_per_year_metrics.csv`)

## 注意事項

1. **NaN方針**: 埋めない（∞→NaNのみ置換）。木モデル分岐に任せる
2. **列名・型**: L4 choose_features()のルールに合致させる
3. **衝突ルール**: policy_boundary優先でtransit無効化
4. **再現性**: 乱択なし。I/Oのみで決定
5. **性能**: Colab無料版で1シナリオ数秒〜十数秒を目標

## テスト

### 受入テスト例

```bash
# 単一イベント・当年のみ
python cli_run_scenario.py scenario_examples/housing_boost.json

# 二年波及
python cli_run_scenario.py scenario_examples/factory_case.json
```

### 期待される動作
- 単一イベント: event_housing_t=+1.0が立ち、翌年のt1は0
- 二年波及: h1,h2にexp_all_h*が立つ
- manual_delta: exp_all_h*に加算される
- 衝突ルール: policy_boundary優先でtransit無効化
- PIの存在: 各年にpi95が入っており、上下幅が>0
