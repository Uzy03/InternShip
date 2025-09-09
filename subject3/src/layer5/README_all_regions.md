# 全地域表示機能 実装ガイド

## 概要

人口予測モデルに「全地域表示」機能を追加しました。これにより、単一町丁の詳細予測に加えて、全町丁の予測結果を一括で表示・分析できるようになります。

## 実装内容

### A. サービス層の返り値契約強化

**ファイル**: `forecast_service.py`

- `run_scenario()` 関数を追加
- 返り値の構造を標準化（必須キー: `town`, `baseline_year`, `horizons`, `results`）
- Δの整合性チェック（`abs(delta - (exp+macro+inertia+other)) < 1e-6`）
- 既存の `l5_debug_detail_*.csv` 生成を維持

### B. 全町丁一括実行CLI

**ファイル**: `cli_run_all.py`

- `data/processed/l5_baseline.csv` から町丁一覧を取得
- シナリオテンプレートを使用して全町丁の予測を実行
- 結果を `output/forecast_all_rows.csv` と `output/forecast_all.json` に保存
- オプションで `data/interim/centroids.csv` から緯度・経度情報を結合

### C. ダッシュボードの全地域ビュー

**ファイル**: `app_simple.py`

- 表示モード選択（「単一町丁予測」/「全地域表示」）
- 年・指標・町丁名でのフィルタリング
- ソート機能（Δ人口・人口・町丁名）
- テーブル表示・地図表示・ヒストグラム表示
- CSVダウンロード機能

### D. 品質担保テスト

**ファイル**: 
- `test_quality_checks.py` - 3シナリオでのスモークテスト
- `test_basic_functionality.py` - 基本機能テスト

## 使用方法

### 1. 全町丁予測の実行

```bash
# イベントなしシナリオ
python cli_run_all.py --scenario scenario_examples/no_event.json --output_dir output

# 雇用機会増加シナリオ
python cli_run_all.py --scenario scenario_examples/employment_inc.json --output_dir output

# 商業+雇用シナリオ
python cli_run_all.py --scenario scenario_examples/commercial_employment.json --output_dir output
```

### 2. ダッシュボードの起動

```bash
cd src/dashboard
streamlit run app_simple.py
```

ダッシュボードで「全地域表示」を選択し、生成された `output/forecast_all_rows.csv` を読み込みます。

### 3. 品質テストの実行

```bash
# 基本機能テスト
python test_basic_functionality.py

# 品質担保テスト（3シナリオ）
python test_quality_checks.py --output_dir output
```

## 出力ファイル形式

### forecast_all_rows.csv

| 列名 | 説明 | 例 |
|------|------|-----|
| town | 町丁名 | "熊本市中央区水前寺" |
| baseline_year | 基準年 | 2025 |
| year | 予測年 | 2026 |
| h | 予測期間 | 1 |
| delta | 人口変化量 | 15.2 |
| pop | 予測人口 | 1015.2 |
| exp | 期待効果 | 10.5 |
| macro | マクロ要因 | 2.1 |
| inertia | 慣性 | 1.8 |
| other | その他 | 0.8 |
| pi_delta_low | Δ信頼区間下限 | 5.1 |
| pi_delta_high | Δ信頼区間上限 | 25.3 |
| pi_pop_low | 人口信頼区間下限 | 1005.1 |
| pi_pop_high | 人口信頼区間上限 | 1025.3 |
| lat | 緯度（オプション） | 32.7891 |
| lon | 経度（オプション） | 130.7414 |

## 品質担保

### 自動チェック項目

1. **Δの整合性**: `delta = exp + macro + inertia + other` が全行で成立
2. **人口パス単調性**: `pop[t] = pop[t-1] + delta[t]` が成立
3. **NaN/欠損値**: データに欠損値がないことを確認
4. **分布変化**: シナリオ間で分布が適切に変化することを確認

### テストシナリオ

1. **イベントなし**: ベースライン予測
2. **雇用機会増加のみ**: 単一イベント効果
3. **商業+雇用増加**: 複合イベント効果

## トラブルシューティング

### よくある問題

1. **「全地域予測データが見つかりません」エラー**
   - 先に `cli_run_all.py` を実行して `forecast_all_rows.csv` を生成してください

2. **Δの整合性エラー**
   - モデルの内部計算に問題がある可能性があります
   - ログを確認して詳細を調査してください

3. **メモリ不足**
   - 町丁数が多い場合は、並列実行やバッチ処理の実装を検討してください

### ログの確認

```bash
# 予測実行時のログを確認
tail -f logs/forecast_*.log

# CLI実行時のログを確認
python cli_run_all.py --scenario scenario.json 2>&1 | tee cli_output.log
```

## 今後の拡張予定

### F. 高速化・運用改善

- 距離行列 `.npz` キャッシュ
- `build_future_features` のバッチモード最適化
- 並列実行とプログレス表示
- CI回帰テストの追加

### パフォーマンス最適化

- 大規模データセット対応
- メモリ使用量の最適化
- 実行時間の短縮

## 関連ファイル

- `forecast_service.py` - 予測サービス（返り値契約強化）
- `cli_run_all.py` - 全町丁一括実行CLI
- `app_simple.py` - ダッシュボード（全地域ビュー追加）
- `test_quality_checks.py` - 品質担保テスト
- `test_basic_functionality.py` - 基本機能テスト
- `README_all_regions.md` - このファイル
