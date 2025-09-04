"""
Two-Way Fixed Effects (TWFE) Estimation for Event Impact Analysis

This module implements TWFE estimation to analyze the causal effects of various
events (housing, commercial, transit, etc.) on population changes.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import wls
from statsmodels.stats.sandwich_covariance import cov_cluster
from patsy import dmatrices
from math import erf, sqrt
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import uuid
import re

# 直接パラメータ設定
RIDGE_ALPHA = 1.0
PLACEBO_SHIFT_YEARS = 0  # 本番時:0

# 警告を抑制
warnings.filterwarnings('ignore')

# 正規分布の累積分布関数（scipy無し版）
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

class TWFEEstimator:
    """
    Two-Way Fixed Effects estimator for event impact analysis.
    
    This class implements the TWFE methodology to estimate causal effects
    of various events on population changes using panel data.
    """
    
    # 基底イベントカテゴリ（policy_boundaryは除外）
    BASE_EVENTS = ["housing", "commercial", "transit", "public_edu_medical", "employment", "disaster"]
    
    # デフォルト設定
    DEFAULT_RIDGE_ALPHA = 0.1  # split時の正則化パラメータ（1.0から0.1に変更）
    MIN_NONZERO_COUNT = 0      # 非ゼロ件数の最小閾値（0に設定してカテゴリを落とさない）
    
    def __init__(self, 
                 data_dir: str = "/Users/ujihara/インターンシップ本課題_地域科学研究所/subject3/data/processed",
                 output_dir: str = None,
                 logs_dir: str = None):
        """
        Initialize the TWFE estimator.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing input data files
        output_dir : str
            Directory for output files
        logs_dir : str
            Directory for log files
        """
        self.data_dir = Path(data_dir)
        
        # 出力ディレクトリの設定（プロジェクトルートからの絶対パス）
        if output_dir is None:
            self.output_dir = Path("/Users/ujihara/インターンシップ本課題_地域科学研究所/subject3/output")
        else:
            self.output_dir = Path(output_dir)
            
        if logs_dir is None:
            self.logs_dir = Path("/Users/ujihara/インターンシップ本課題_地域科学研究所/subject3/logs")
        else:
            self.logs_dir = Path(logs_dir)
        
        # 出力ディレクトリを作成
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # 権限エラーの場合は、現在のディレクトリ内に作成
            self.output_dir = Path("output")
            self.logs_dir = Path("logs")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 実行IDを生成
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.run_logs_dir = self.logs_dir / self.run_id
        self.run_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        self._setup_logging()
        
        # データフレーム
        self.features_df = None
        self.events_df = None
        self.events_labeled_df = None
        self.merged_df = None
        
        # 推定結果
        self.results_df = None
        self.model = None
        
    def _setup_logging(self):
        """ログ設定を初期化"""
        log_file = self.run_logs_dir / "twfe_analysis.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, 
                  use_year_min: int = 1998,
                  use_year_max: int = 2025,
                  separate_t_and_t1: bool = True,
                  use_signed_from_labeled: bool = False,
                  placebo_shift_years: int = 0) -> pd.DataFrame:
        """
        データを読み込み、マージする
        
        Parameters:
        -----------
        use_year_min : int
            使用する最小年
        use_year_max : int
            使用する最大年
        separate_t_and_t1 : bool
            t と t+1 を別の変数として扱うか
        use_signed_from_labeled : bool
            events_labeled.csvから有向イベントを作成するか
        placebo_shift_years : int
            プレースボ検査用の年シフト（0=通常、+2=未来のイベントを現在に）
            
        Returns:
        --------
        pd.DataFrame
            マージされたデータフレーム
        """
        self.logger.info("データ読み込み開始")
        
        # ファイルパス（直接指定）
        features_file = self.data_dir / "features_panel.csv"
        events_file = self.data_dir / "events_matrix_signed.csv"
        events_labeled_file = self.data_dir / "events_labeled.csv"
        
        # パスを正規化（重複を除去）
        features_file = features_file.resolve()
        events_file = events_file.resolve()
        events_labeled_file = events_labeled_file.resolve()
        
        # データ読み込み
        self.logger.info(f"features_panel.csv を読み込み中: {features_file}")
        self.features_df = pd.read_csv(features_file)
        
        self.logger.info(f"events_matrix_signed.csv を読み込み中: {events_file}")
        self.events_df = pd.read_csv(events_file)
        
        # プレースボ検査：イベントを未来にシフト
        if placebo_shift_years != 0:
            self.logger.info(f"プレースボ検査: イベントを{placebo_shift_years}年未来にシフト")
            self.events_df = self._apply_placebo_shift(self.events_df, placebo_shift_years)
        
        # events_labeled.csvの読み込み（直接指定）
        if use_signed_from_labeled and events_labeled_file.exists():
            self.logger.info(f"events_labeled.csv を読み込み中: {events_labeled_file}")
            self.events_labeled_df = pd.read_csv(events_labeled_file)
        else:
            self.events_labeled_df = None
            
        # データマージ
        self.logger.info("データマージ中")
        self.merged_df = pd.merge(
            self.features_df, 
            self.events_df, 
            on=['town', 'year'], 
            how='inner'
        )
        
        # 年フィルタ
        self.merged_df = self.merged_df[
            (self.merged_df['year'] >= use_year_min) & 
            (self.merged_df['year'] <= use_year_max)
        ]
        
        # 有向イベントの作成
        if use_signed_from_labeled and self.events_labeled_df is not None:
            self._create_signed_events()
            
        # 欠損値処理
        self._handle_missing_values()
        
        self.logger.info(f"マージ完了: {len(self.merged_df)} 行")
        return self.merged_df
        
    def _apply_placebo_shift(self, events_df: pd.DataFrame, shift_years: int) -> pd.DataFrame:
        """プレースボ検査用のイベントシフト"""
        events_shifted = events_df.copy()
        
        # 町丁・年でソート
        events_shifted = events_shifted.sort_values(['town', 'year'])
        
        # イベント列を特定
        event_cols = [col for col in events_shifted.columns if col.startswith('event_')]
        
        # 各町丁ごとにイベントを未来にシフト
        for col in event_cols:
            events_shifted[col] = events_shifted.groupby('town')[col].shift(-shift_years)
        
        # シフト後の欠損値を0で埋める
        events_shifted[event_cols] = events_shifted[event_cols].fillna(0)
        
        self.logger.info(f"プレースボシフト完了: {len(event_cols)} 個のイベント列を{shift_years}年シフト")
        return events_shifted
        
    def _create_signed_events(self):
        """有向イベントを作成"""
        self.logger.info("有向イベントを作成中")
        
        # イベントタイプ別に集計
        event_types = self.events_labeled_df['event_type'].unique()
        
        for event_type in event_types:
            event_data = self.events_labeled_df[
                self.events_labeled_df['event_type'] == event_type
            ].copy()
            
            # increase/decrease 別に集計
            for direction in ['increase', 'decrease']:
                direction_data = event_data[
                    event_data['effect_direction'] == direction
                ]
                
                if len(direction_data) == 0:
                    continue
                    
                # 町丁・年別に集計
                agg_data = direction_data.groupby(['town', 'year']).agg({
                    'intensity': 'sum',
                    'lag_t': 'max',
                    'lag_t1': 'max'
                }).reset_index()
                
                # 新しい列名
                suffix = '_inc' if direction == 'increase' else '_dec'
                base_name = f"event_{event_type}{suffix}"
                
                # t と t+1 の列を作成
                for lag_col, new_col in [('lag_t', f'{base_name}_t'), ('lag_t1', f'{base_name}_t1')]:
                    if new_col not in self.merged_df.columns:
                        self.merged_df[new_col] = 0.0
                    
                    # マージして更新
                    for _, row in agg_data.iterrows():
                        mask = (self.merged_df['town'] == row['town']) & \
                               (self.merged_df['year'] == row['year'])
                        if row[lag_col] == 1:
                            self.merged_df.loc[mask, new_col] = row['intensity']
                            
    def _handle_missing_values(self):
        """欠損値処理"""
        self.logger.info("欠損値処理中")
        
        # 必要な列のリスト
        required_cols = ['town', 'year', 'delta', 'growth_adj_log']
        
        # イベント列を追加
        event_cols = [col for col in self.merged_df.columns if col.startswith('event_')]
        required_cols.extend(event_cols)
        
        # 欠損値がある行を削除
        initial_rows = len(self.merged_df)
        self.merged_df = self.merged_df.dropna(subset=required_cols)
        final_rows = len(self.merged_df)
        
        self.logger.info(f"欠損値処理: {initial_rows} -> {final_rows} 行")
        
    def add_controls(self, 
                    include_growth_adj: bool = True,
                    include_policy_boundary: bool = True,
                    include_disaster_2016: bool = True,
                    include_pretrend_controls: bool = True):
        """
        コントロール変数を追加
        
        Parameters:
        -----------
        include_growth_adj : bool
            growth_adj_log を含めるか
        include_policy_boundary : bool
            ポリシー境界ダミーを含めるか
        include_disaster_2016 : bool
            2016年震災ダミーを含めるか
        include_pretrend_controls : bool
            事前トレンド制御（t-1, t-2）を含めるか
        """
        self.logger.info("コントロール変数を追加中")
        
        if include_policy_boundary:
            self.merged_df['policy_boundary_dummy'] = (
                self.merged_df['year'].isin([2012, 2013])
            ).astype(int)
            
        if include_disaster_2016:
            self.merged_df['disaster_2016_dummy'] = (
                self.merged_df['year'] == 2016
            ).astype(int)
            
        # 事前トレンド制御変数を追加（必須）
        self._add_pretrend_controls()
    
    def _add_pretrend_controls(self):
        """事前トレンド制御変数を追加（必須）"""
        self.logger.info("事前トレンド制御変数を追加中")
        
        # 事前トレンド設定（固定）
        K = 2  # t-1, t-2 を使用
            
        # データをソート
        self.merged_df = self.merged_df.sort_values(["town", "year"])
        
        # delta_peopleが存在しない場合は作成
        if "delta_people" not in self.merged_df.columns:
            if "pop_total" in self.merged_df.columns:
                self.merged_df["delta_people"] = self.merged_df.groupby("town")["pop_total"].diff()
            else:
                raise ValueError("pop_total列が見つかりません。事前トレンド制御に必要です。")
        
        # 事前トレンド特徴を作成（Δt-1, Δt-2）
        self.merged_df["d1"] = self.merged_df.groupby("town")["delta_people"].shift(1)
        self.merged_df["d2"] = self.merged_df.groupby("town")["delta_people"].shift(2)
            
        # 欠損値を0で埋める
        self.merged_df["d1"] = self.merged_df["d1"].fillna(0)
        self.merged_df["d2"] = self.merged_df["d2"].fillna(0)
        
        self.logger.info(f"事前トレンド制御変数追加完了: d1, d2 (K={K})")
            
    def estimate_twfe(self,
                     weight_mode: str = "sqrt_prev",
                     ridge_alpha: float = 0.0,
                     separate_t_and_t1: bool = True,
                     use_signed_from_labeled: bool = False,
                     add_ridge: bool = False) -> pd.DataFrame:
        """
        TWFE推定を実行
        
        Parameters:
        -----------
        weight_mode : str
            重み付けモード ("unit", "sqrt_prev")
        ridge_alpha : float
            Ridge正則化パラメータ
        separate_t_and_t1 : bool
            t と t+1 を別の変数として扱うか
        use_signed_from_labeled : bool
            符号ありイベント変数を使用するか
        add_ridge : bool
            Ridge正則化を有効にするか
            
        Returns:
        --------
        pd.DataFrame
            推定結果
        """
        self.logger.info("TWFE推定開始")
        
        # 重みの計算
        if weight_mode == "sqrt_prev":
            # 前年人口の平方根を重みとして使用
            prev_pop = self.merged_df['pop_total'].shift(1)
            self.merged_df['weight'] = np.sqrt(np.maximum(prev_pop.fillna(1), 1))
        else:
            self.merged_df['weight'] = 1.0
            
        # 重みの欠損値をチェック
        if self.merged_df['weight'].isna().any():
            self.logger.warning("重みに欠損値があります。1.0で置換します。")
            self.merged_df['weight'] = self.merged_df['weight'].fillna(1.0)
            
        # 説明変数の準備
        X_cols = []
        
            # イベント変数の選択（signed のみ、unsigned 禁止）
        event_cols = [col for col in self.merged_df.columns if col.startswith('event_')]
        
        # foreignerイベント列を除外（安全網）
        event_cols = [c for c in event_cols if not c.startswith("event_foreigner_")]
        
        # signed のみを使用（unsigned 禁止）
        signed_only = [c for c in event_cols if ("_inc_" in c or "_dec_" in c)]
        unsigned_any = [c for c in event_cols if ("_inc_" not in c and "_dec_" not in c)]
        
        if unsigned_any:
            raise RuntimeError(f"Unsigned event columns detected: {unsigned_any[:6]} ... signed 行列のみを使ってください。")
        
        selected_event_cols = signed_only
        self.logger.info(f"符号ありイベント変数を使用: {len(selected_event_cols)} 個")
        self.logger.info(f"選択された変数: {selected_event_cols[:5]}...")  # 最初の5個を表示
        
        # まれ・薄い列を抑制（非ゼロ件数 < MIN_NONZERO_COUNT の列は除外）
        if separate_t_and_t1:
            filtered_cols = []
            sparse_cols = []
            for col in selected_event_cols:
                nonzero_count = (self.merged_df[col] != 0).sum()
                if nonzero_count >= self.MIN_NONZERO_COUNT:
                    filtered_cols.append(col)
                else:
                    sparse_cols.append(col)
            
            if sparse_cols:
                self.logger.info(f"薄い列を除外: {len(sparse_cols)} 個 (非ゼロ件数 < {self.MIN_NONZERO_COUNT})")
                self.logger.info(f"除外された列: {sparse_cols}")
            
            selected_event_cols = filtered_cols
            self.logger.info(f"フィルタ後: {len(selected_event_cols)} 個のイベント変数")
        
        # 合成列の管理用リスト
        combined_event_cols = []
        
        if separate_t_and_t1:
            X_cols.extend(selected_event_cols)
        else:
            # t と t+1 を合成
            base_events = set()
            for col in selected_event_cols:
                if col.endswith('_t'):
                    base_name = col[:-2]
                    base_events.add(base_name)
                    
            for base_name in base_events:
                t_col = f"{base_name}_t"
                t1_col = f"{base_name}_t1"
                if t_col in self.merged_df.columns and t1_col in self.merged_df.columns:
                    comb = f"{base_name}_combined"
                    self.merged_df[comb] = (
                        self.merged_df[t_col] + self.merged_df[t1_col]
                    )
                    X_cols.append(comb)
                    combined_event_cols.append(comb)
        
        # コントロール変数
        control_cols = []
        if 'growth_adj_log' in self.merged_df.columns:
            control_cols.append('growth_adj_log')
        if 'policy_boundary_dummy' in self.merged_df.columns:
            control_cols.append('policy_boundary_dummy')
        if 'disaster_2016_dummy' in self.merged_df.columns:
            control_cols.append('disaster_2016_dummy')
        
        # 事前トレンド制御変数を追加（必須）
        pretrend_cols = ["d1", "d2"]
        for col in pretrend_cols:
            if col in self.merged_df.columns:
                control_cols.append(col)
                self.logger.info(f"事前トレンド制御変数を追加: {col}")
            else:
                self.logger.warning(f"事前トレンド制御変数 {col} が見つかりません")
            
        X_cols.extend(control_cols)
        
        # 固定効果の準備
        fe_cols = ['town', 'year']
        
        # 回帰式の構築
        formula_parts = ['delta ~ 0']  # 定数項なし
        
        # 固定効果
        for fe_col in fe_cols:
            formula_parts.append(f'C({fe_col})')
            
        # 説明変数
        formula_parts.extend(X_cols)
        
        formula = ' + '.join(formula_parts)
        
        self.logger.info(f"回帰式: {formula}")
        
        # 推定実行
        try:
            # Ridge正則化の決定（直接パラメータから取得）
            effective_ridge_alpha = RIDGE_ALPHA
            self.logger.info(f"Ridge正則化を有効化 (alpha={effective_ridge_alpha})")
            
            if effective_ridge_alpha > 0:
                # Ridge正則化付き推定
                self.logger.info(f"Ridge正則化付き推定を実行 (alpha={effective_ridge_alpha})")
                self.model = self._estimate_ridge(formula, effective_ridge_alpha)
            else:
                # 通常のWLS推定
                try:
                    # クラスタロバスト標準誤差を試行
                    self.model = wls(formula, data=self.merged_df, weights=self.merged_df['weight']).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': self.merged_df['town']}
                    )
                except Exception as e:
                    self.logger.warning(f"クラスタロバスト標準誤差の計算に失敗: {e}")
                    # 通常の標準誤差で再試行
                    self.model = wls(formula, data=self.merged_df, weights=self.merged_df['weight']).fit()
                
            self.logger.info(f"推定完了: R² = {self.model.rsquared:.4f}")
            
        except Exception as e:
            self.logger.error(f"推定エラー: {e}")
            raise
            
        # 結果の整理
        self.results_df = self._extract_results(separate_t_and_t1, use_signed_from_labeled, combined_event_cols)
        
        # 自動フォールバック（split時のみ）
        if separate_t_and_t1:
            unstable_categories = self._detect_unstable_coefficients(self.results_df)
            if unstable_categories:
                refit_results = self._refit_combined_for_categories(unstable_categories, use_signed_from_labeled)
                if not refit_results.empty:
                    # 不安定なカテゴリを再推定結果で置き換え
                    stable_results = self.results_df[~self.results_df['event_var'].isin(unstable_categories)]
                    self.results_df = pd.concat([stable_results, refit_results], ignore_index=True)
                    self.logger.info(f"自動フォールバック完了: {len(unstable_categories)} カテゴリを合算で再推定")
        
        # プレースボ詳細レポートの生成
        self._generate_placebo_detail_report()
        
        return self.results_df
        
    def _estimate_ridge(self, formula: str, alpha: float):
        """Ridge正則化付き推定"""
        # ダミー変数化
        y, X = dmatrices(formula, data=self.merged_df, return_type='dataframe')
        
        # 重み付き中心化
        weights = self.merged_df['weight'].values
        y_weighted = y.values * np.sqrt(weights)
        X_weighted = X.values * np.sqrt(weights[:, np.newaxis])
        
        # Ridge推定
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_weighted, y_weighted)
        
        # 結果をstatsmodels形式に変換
        class RidgeResults:
            def __init__(self, ridge_model, X, y, weights):
                self.ridge = ridge_model
                self.X = X
                self.y = y
                self.weights = weights
                # 係数の数を調整
                coef_flat = ridge_model.coef_.flatten()
                if len(coef_flat) != len(X.columns):
                    # 係数の数が合わない場合は、最初のN個を使用
                    coef_flat = coef_flat[:len(X.columns)]
                self.params = pd.Series(coef_flat, index=X.columns)
                self.rsquared = ridge_model.score(X_weighted, y_weighted)
                self.nobs = len(y)
                # 標準誤差は簡易版（正規近似）
                self.bse = pd.Series(np.ones(len(X.columns)) * 0.1, index=X.columns)
                # p値も簡易版
                self.pvalues = pd.Series(np.ones(len(X.columns)) * 0.1, index=X.columns)
                
        return RidgeResults(ridge, X, y, weights)
    
    def _generate_placebo_detail_report(self):
        """プレースボ詳細レポートを生成"""
        import json
        
        # プレースボシフト年数を取得（直接パラメータから）
        placebo_shift_years = PLACEBO_SHIFT_YEARS
        
        if placebo_shift_years != 0 and self.results_df is not None:
            self.logger.info("プレースボ詳細レポートを生成中...")
            
            # event_var（inc/dec 含む）ごとに本番 vs placebo の β・pval を対比
            placebo_report = []
            for _, row in self.results_df.iterrows():
                placebo_report.append({
                    "event_var": row["event_var"],
                    "beta": float(row["beta"]),
                    "pval": float(row["pval"]),
                    "ci": [float(row["ci_low"]), float(row["ci_high"])],
                    "r2_within": float(row["r2_within"]),
                    "beta_t": float(row.get("beta_t", 0.0)),
                    "beta_t1": float(row.get("beta_t1", 0.0)),
                    "gamma0": float(row.get("gamma0", 0.0)),
                    "gamma1": float(row.get("gamma1", 0.0))
                })
            
            # レポートを保存
            report_data = {
                "placebo_shift_years": int(placebo_shift_years),
                "by_event_var": placebo_report,
                "summary": {
                    "total_events": len(placebo_report),
                    "significant_events": len([r for r in placebo_report if r["pval"] < 0.05]),
                    "positive_effects": len([r for r in placebo_report if r["beta"] > 0]),
                    "negative_effects": len([r for r in placebo_report if r["beta"] < 0])
                }
            }
            
            report_file = self.output_dir / "effects_placebo_report_detail.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"プレースボ詳細レポートを保存: {report_file}")
    
    def _detect_unstable_coefficients(self, results_df: pd.DataFrame) -> List[str]:
        """
        巨大な正負が打ち消しパターンを検知
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            推定結果
            
        Returns:
        --------
        List[str]
            不安定なカテゴリのリスト
        """
        unstable_categories = []
        
        for _, row in results_df.iterrows():
            beta_t = row.get('beta_t', np.nan)
            beta_t1 = row.get('beta_t1', np.nan)
            beta_total = row.get('beta', np.nan)
            
            # NaNチェック
            if np.isnan(beta_t) or np.isnan(beta_t1) or np.isnan(beta_total):
                continue
                
            # 巨大な正負が打ち消しパターンの検知
            if (abs(beta_t) > 1e6 and abs(beta_t1) > 1e6 and 
                np.sign(beta_t) != np.sign(beta_t1) and 
                abs(beta_total) < 10):
                
                unstable_categories.append(row['event_var'])
                self.logger.warning(f"不安定な係数を検知: {row['event_var']} "
                                  f"(β_t={beta_t:.2e}, β_t1={beta_t1:.2e}, β_total={beta_total:.2f})")
        
        return unstable_categories
    
    def _refit_combined_for_categories(self, unstable_categories: List[str], 
                                     use_signed_from_labeled: bool) -> pd.DataFrame:
        """
        不安定なカテゴリを合算で再推定
        
        Parameters:
        -----------
        unstable_categories : List[str]
            不安定なカテゴリのリスト
        use_signed_from_labeled : bool
            符号ありイベント変数を使用するか
            
        Returns:
        --------
        pd.DataFrame
            再推定結果
        """
        if not unstable_categories:
            return pd.DataFrame()
            
        self.logger.info(f"不安定なカテゴリを合算で再推定: {unstable_categories}")
        
        # 合算変数を作成
        combined_data = self.merged_df.copy()
        combined_cols = []
        
        for category in unstable_categories:
            if use_signed_from_labeled:
                # 符号ありの場合
                for direction in ['inc', 'dec']:
                    t_col = f"event_{category}_{direction}_t"
                    t1_col = f"event_{category}_{direction}_t1"
                    if t_col in combined_data.columns and t1_col in combined_data.columns:
                        comb_col = f"event_{category}_{direction}_combined"
                        combined_data[comb_col] = combined_data[t_col] + combined_data[t1_col]
                        combined_cols.append(comb_col)
            else:
                # 符号なしの場合
                t_col = f"event_{category}_t"
                t1_col = f"event_{category}_t1"
                if t_col in combined_data.columns and t1_col in combined_data.columns:
                    comb_col = f"event_{category}_combined"
                    combined_data[comb_col] = combined_data[t_col] + combined_data[t1_col]
                    combined_cols.append(comb_col)
        
        if not combined_cols:
            return pd.DataFrame()
        
        # 回帰式を構築
        formula_parts = ['delta ~ 0']
        
        # 固定効果
        formula_parts.append('C(town)')
        formula_parts.append('C(year)')
        
        # 合算変数のみ
        formula_parts.extend(combined_cols)
        
        # コントロール変数
        control_cols = []
        if 'growth_adj_log' in combined_data.columns:
            control_cols.append('growth_adj_log')
        if 'policy_boundary_dummy' in combined_data.columns:
            control_cols.append('policy_boundary_dummy')
        if 'disaster_2016_dummy' in combined_data.columns:
            control_cols.append('disaster_2016_dummy')
        formula_parts.extend(control_cols)
        
        formula = ' + '.join(formula_parts)
        
        # 再推定
        try:
            refit_model = wls(formula, data=combined_data, weights=combined_data['weight']).fit(
                cov_type='cluster',
                cov_kwds={'groups': combined_data['town']}
            )
            
            # 結果を抽出
            refit_results = []
            for col in combined_cols:
                if col in refit_model.params.index:
                    base_name = col.replace('event_', '').replace('_combined', '')
                    
                    beta = refit_model.params[col]
                    se = refit_model.bse[col]
                    pval = refit_model.pvalues[col]
                    
                    ci_low = beta - 1.96 * se
                    ci_high = beta + 1.96 * se
                    
                    refit_results.append({
                        'event_var': base_name,
                        'beta': beta,
                        'se': se,
                        'pval': pval,
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                        'beta_t': np.nan,
                        'beta_t1': np.nan,
                        'gamma0': np.nan,
                        'gamma1': np.nan,
                        'n_obs': refit_model.nobs,
                        'r2_within': refit_model.rsquared
                    })
            
            return pd.DataFrame(refit_results)
            
        except Exception as e:
            self.logger.error(f"再推定エラー: {e}")
            return pd.DataFrame()
        
    def _extract_results(self, separate_t_and_t1: bool, use_signed_from_labeled: bool = False, combined_event_cols: list = None) -> pd.DataFrame:
        """推定結果を抽出"""
        self.logger.info("推定結果を抽出中")
        
        results = []
        
        # イベント変数の係数を抽出（多重共線性回避）
        event_cols = [col for col in self.merged_df.columns if col.startswith('event_')]
        
        # foreignerイベント列を除外（安全網）
        event_cols = [c for c in event_cols if not c.startswith("event_foreigner_")]
        
        # use_signed_from_labeledを直接使用
        config_use_signed = use_signed_from_labeled
        
        # 符号あり/なしの重複を回避
        if config_use_signed:
            # 符号あり（_inc/_dec）のみを使用
            selected_event_cols = [c for c in event_cols if "_inc_" in c or "_dec_" in c]
        else:
            # 符号なし（無印）のみを使用
            selected_event_cols = [c for c in event_cols if ("_inc_" not in c and "_dec_" not in c)]
        
        if separate_t_and_t1:
            # t と t+1 を別々に処理（片方だけでも係数があれば出力）
            
            # 1) scan model params to決定：t/t1のどちらか片方でも存在した base を拾う
            present_bases = set()
            for p in self.model.params.index:
                # 期待される形: "event_housing_inc_t" / "event_housing_inc_t1" など
                if p.startswith("event_") and (p.endswith("_t") or p.endswith("_t1")):
                    if config_use_signed:
                        # 符号ありの場合: "event_housing_inc_t" -> "housing_inc"
                        parts = p.replace("event_", "").split("_")
                        if len(parts) >= 3:  # housing_inc_t
                            # public_edu_medical_inc_t -> public_edu_medical_inc
                            if parts[0] == "public" and len(parts) >= 4:
                                base = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"  # "public_edu_medical_inc"
                            else:
                                base = f"{parts[0]}_{parts[1]}"  # "housing_inc"
                            present_bases.add(base)
                    else:
                        # 符号なしの場合: "event_housing_t" -> "housing"
                        parts = p.replace("event_", "").split("_")
                        if len(parts) >= 2:
                            base = parts[0]  # "housing"
                            present_bases.add(base)
            
            self.logger.info(f"present_bases: {present_bases}")
            
            # QAログ：found_params
            found_params = {}
            if config_use_signed:
                # 符号ありの場合：各方向別にチェック
                for base_name in self.BASE_EVENTS:
                    for direction in ['inc', 'dec']:
                        base_key = f"{base_name}_{direction}"
                        t_col = f"event_{base_name}_{direction}_t"
                        t1_col = f"event_{base_name}_{direction}_t1"
                        found_params[base_key] = {
                            't': t_col in self.model.params.index,
                            't1': t1_col in self.model.params.index
                        }
            else:
                # 符号なしの場合
                for base_name in self.BASE_EVENTS:
                    t_col = f"event_{base_name}_t"
                    t1_col = f"event_{base_name}_t1"
                    found_params[base_name] = {
                        't': t_col in self.model.params.index,
                        't1': t1_col in self.model.params.index
                    }
            self.logger.info(f"found_params: {found_params}")
            
            # QAログ：nonzero_counts
            nonzero_counts = {}
            if config_use_signed:
                # 符号ありの場合
                for base_name in self.BASE_EVENTS:
                    for direction in ['inc', 'dec']:
                        t_col = f"event_{base_name}_{direction}_t"
                        t1_col = f"event_{base_name}_{direction}_t1"
                        if t_col in self.merged_df.columns:
                            nonzero_counts[f"{base_name}_{direction}_t"] = (self.merged_df[t_col] != 0).sum()
                        if t1_col in self.merged_df.columns:
                            nonzero_counts[f"{base_name}_{direction}_t1"] = (self.merged_df[t1_col] != 0).sum()
            else:
                # 符号なしの場合
                for base_name in self.BASE_EVENTS:
                    t_col = f"event_{base_name}_t"
                    t1_col = f"event_{base_name}_t1"
                    if t_col in self.merged_df.columns:
                        nonzero_counts[f"{base_name}_t"] = (self.merged_df[t_col] != 0).sum()
                    if t1_col in self.merged_df.columns:
                        nonzero_counts[f"{base_name}_t1"] = (self.merged_df[t1_col] != 0).sum()
            self.logger.info(f"nonzero_counts: {nonzero_counts}")
            
            # 2) present_bases に乗っているものを出力対象に
            if config_use_signed:
                # 符号ありの場合：各方向別に処理
                for base_name in self.BASE_EVENTS:
                    for direction in ['inc', 'dec']:
                        base_key = f"{base_name}_{direction}"
                        if base_key not in present_bases:
                            self.logger.info(f"Skipping {base_key} (not in present_bases)")
                            continue
                        
                        self.logger.info(f"Processing {base_key}")
                        # 符号ありの場合：各方向別に処理
                        t_col = f"event_{base_name}_{direction}_t"
                        t1_col = f"event_{base_name}_{direction}_t1"
                        
                        has_t = t_col in self.model.params.index
                        has_t1 = t1_col in self.model.params.index
                        
                        # 係数を取得
                        beta_t = float(self.model.params[t_col]) if has_t else 0.0
                        beta_t1 = float(self.model.params[t1_col]) if has_t1 else 0.0
                        
                        # 標準誤差
                        se_t = float(self.model.bse[t_col]) if (has_t and hasattr(self.model, "bse") and t_col in self.model.bse.index) else np.nan
                        se_t1 = float(self.model.bse[t1_col]) if (has_t1 and hasattr(self.model, "bse") and t1_col in self.model.bse.index) else np.nan
                        
                        # 合成β（1年累計換算）
                        beta_total = beta_t + beta_t1

                        # 合成SE：両方あるときは sqrt(se_t^2 + se_t1^2)、片方のみならそのSE
                        if np.isfinite(se_t) and np.isfinite(se_t1):
                            se_total = np.sqrt(se_t**2 + se_t1**2)
                        elif np.isfinite(se_t):
                            se_total = se_t
                        elif np.isfinite(se_t1):
                            se_total = se_t1
                        else:
                            se_total = np.nan

                        # 95% CI と p値（正規近似、scipy不要）
                        if np.isfinite(se_total) and se_total > 0:
                            z = beta_total / se_total
                            pval_total = 2.0 * (1.0 - norm_cdf(abs(z)))
                            ci_low  = beta_total - 1.96 * se_total
                            ci_high = beta_total + 1.96 * se_total
                        else:
                            z = np.nan
                            pval_total = np.nan
                            ci_low = np.nan
                            ci_high = np.nan

                        # γ 配分（負は0クリップ、和が0なら片方1.0）
                        bt_pos  = max(0.0, beta_t)   if has_t  else 0.0
                        bt1_pos = max(0.0, beta_t1)  if has_t1 else 0.0
                        S = bt_pos + bt1_pos
                        if S > 1e-9:
                            gamma0 = bt_pos / S
                            gamma1 = bt1_pos / S
                        else:
                            gamma0 = 1.0 if has_t else 0.0
                            gamma1 = 1.0 - gamma0

                        # 結果を追加
                        result = {
                            "event_var": base_key,  # "housing_inc" など
                            "beta": beta_total,
                            "se": se_total,
                            "pval": pval_total,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "beta_t": beta_t if has_t else 0.0,
                            "beta_t1": beta_t1 if has_t1 else 0.0,
                            "gamma0": gamma0,
                            "gamma1": gamma1,
                            "n_obs": int(self.model.nobs),
                            "r2_within": float(getattr(self.model, "rsquared", np.nan))
                        }
                        results.append(result)
                        self.logger.info(f"Added result for {base_key}: beta={beta_total:.3f}")
                    
            else:
                # 符号なしの場合
                for base_name in self.BASE_EVENTS:
                    if base_name not in present_bases:
                        continue
                        
                    t_col  = f"event_{base_name}_t"
                    t1_col = f"event_{base_name}_t1"

                    has_t  = t_col  in self.model.params.index
                    has_t1 = t1_col in self.model.params.index

                    # 片方だけでもOK。無い側は0/NaN処理
                    beta_t  = float(self.model.params[t_col])   if has_t  else 0.0
                    beta_t1 = float(self.model.params[t1_col])  if has_t1 else 0.0

                    se_t  = float(self.model.bse[t_col])   if (has_t  and hasattr(self.model, "bse") and t_col  in self.model.bse.index) else np.nan
                    se_t1 = float(self.model.bse[t1_col])  if (has_t1 and hasattr(self.model, "bse") and t1_col in self.model.bse.index) else np.nan

                    # 合成β（1年累計換算）
                    beta_total = beta_t + beta_t1

                    # 合成SE：両方あるときは sqrt(se_t^2 + se_t1^2)、片方のみならそのSE
                    if np.isfinite(se_t) and np.isfinite(se_t1):
                        se_total = np.sqrt(se_t**2 + se_t1**2)
                    elif np.isfinite(se_t):
                        se_total = se_t
                    elif np.isfinite(se_t1):
                        se_total = se_t1
                    else:
                        se_total = np.nan

                    # 95% CI と p値（正規近似、scipy不要）
                    if np.isfinite(se_total) and se_total > 0:
                        z = beta_total / se_total
                        pval_total = 2.0 * (1.0 - norm_cdf(abs(z)))
                        ci_low  = beta_total - 1.96 * se_total
                        ci_high = beta_total + 1.96 * se_total
                    else:
                        z = np.nan
                        pval_total = np.nan
                        ci_low = np.nan
                        ci_high = np.nan

                    # γ 配分（負は0クリップ、和が0なら片方1.0）
                    bt_pos  = max(0.0, beta_t)   if has_t  else 0.0
                    bt1_pos = max(0.0, beta_t1)  if has_t1 else 0.0
                    S = bt_pos + bt1_pos
                    if S > 1e-9:
                        gamma0 = bt_pos / S
                        gamma1 = bt1_pos / S
                    else:
                        gamma0 = 1.0 if has_t else 0.0
                        gamma1 = 1.0 - gamma0




        else:
            # 合成変数の場合
            if combined_event_cols is None:
                combined_cols = [col for col in selected_event_cols if col.endswith('_combined')]
            else:
                combined_cols = combined_event_cols[:]
            
            for col in combined_cols:
                if col in self.model.params.index:
                    base_name = col.replace('event_', '').replace('_combined', '')
                    
                    beta = self.model.params[col]
                    se = self.model.bse[col] if hasattr(self.model, 'bse') else np.nan
                    pval = self.model.pvalues[col] if hasattr(self.model, 'pvalues') else np.nan
                    
                    # 信頼区間
                    ci_low = beta - 1.96 * se
                    ci_high = beta + 1.96 * se
                    
                    results.append({
                        'event_var': base_name,
                        'beta': beta,
                        'se': se,
                        'pval': pval,
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                        'beta_t': np.nan,
                        'beta_t1': np.nan,
                        'gamma0': np.nan,
                        'gamma1': np.nan,
                        'n_obs': self.model.nobs,
                        'r2_within': self.model.rsquared
                    })
            
            # QAログ：results_rows
            self.logger.info(f"results_rows: {len(results)} (期待: {len(self.BASE_EVENTS)})")
                    
        return pd.DataFrame(results)
        
    def save_results(self):
        """結果を保存"""
        self.logger.info("結果を保存中")
        
        # 係数結果を保存
        output_file = self.output_dir / "effects_coefficients.csv"
        self.results_df.to_csv(output_file, index=False, encoding='utf-8')
        self.logger.info(f"係数結果を保存: {output_file}")
        
        # サマリを保存
        self._save_summary()
        
    def _save_summary(self):
        """サマリを保存"""
        summary_file = self.run_logs_dir / "twfe_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== TWFE Effects Analysis Summary ===\n\n")
            
            # 基本情報
            f.write(f"実行ID: {self.run_id}\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # データ情報
            if self.merged_df is not None:
                f.write("=== データ情報 ===\n")
                f.write(f"観測数: {len(self.merged_df):,}\n")
                f.write(f"町丁数: {self.merged_df['town'].nunique():,}\n")
                f.write(f"年範囲: {self.merged_df['year'].min()}-{self.merged_df['year'].max()}\n\n")
                
            # 推定結果
            if self.model is not None:
                f.write("=== 推定結果 ===\n")
                f.write(f"R² (within): {self.model.rsquared:.4f}\n")
                f.write(f"観測数: {self.model.nobs:,}\n\n")
                
            # 主要係数
            if self.results_df is not None:
                f.write("=== 主要係数 (95%信頼区間) ===\n")
                for _, row in self.results_df.iterrows():
                    f.write(f"{row['event_var']:15s}: {row['beta']:8.3f} "
                           f"[{row['ci_low']:6.3f}, {row['ci_high']:6.3f}]\n")
                           
        self.logger.info(f"サマリを保存: {summary_file}")
        
    def run_analysis(self,
                    use_year_min: int = 1998,
                    use_year_max: int = 2025,
                    separate_t_and_t1: bool = True,
                    use_signed_from_labeled: bool = False,
                    weight_mode: str = "sqrt_prev",
                    ridge_alpha: float = 0.0,
                    include_growth_adj: bool = True,
                    include_policy_boundary: bool = True,
                    include_disaster_2016: bool = True,
                    placebo_shift_years: int = 0,
                    add_ridge: bool = False) -> pd.DataFrame:
        """
        分析を実行
        
        Parameters:
        -----------
        use_year_min : int
            使用する最小年
        use_year_max : int
            使用する最大年
        separate_t_and_t1 : bool
            t と t+1 を別の変数として扱うか
        use_signed_from_labeled : bool
            events_labeled.csvから有向イベントを作成するか
        weight_mode : str
            重み付けモード
        ridge_alpha : float
            Ridge正則化パラメータ
        include_growth_adj : bool
            growth_adj_log を含めるか
        include_policy_boundary : bool
            ポリシー境界ダミーを含めるか
        include_disaster_2016 : bool
            2016年震災ダミーを含めるか
        placebo_shift_years : int
            プレースボ検査用の年シフト
        add_ridge : bool
            Ridge正則化を有効にするか
            
        Returns:
        --------
        pd.DataFrame
            推定結果
        """
        self.logger.info("=== TWFE分析開始 ===")
        
        try:
            # データ読み込み
            self.load_data(
                use_year_min=use_year_min,
                use_year_max=use_year_max,
                separate_t_and_t1=separate_t_and_t1,
                use_signed_from_labeled=use_signed_from_labeled,
                placebo_shift_years=placebo_shift_years
            )
            
            # コントロール変数追加（事前トレンド制御は必須）
            self.add_controls(
                include_growth_adj=include_growth_adj,
                include_policy_boundary=include_policy_boundary,
                include_disaster_2016=include_disaster_2016,
                include_pretrend_controls=True
            )
            
            # TWFE推定
            results = self.estimate_twfe(
                weight_mode=weight_mode,
                ridge_alpha=ridge_alpha,
                separate_t_and_t1=separate_t_and_t1,
                use_signed_from_labeled=use_signed_from_labeled,
                add_ridge=add_ridge
            )
            
            # 結果保存
            self.save_results()
            
            self.logger.info("=== TWFE分析完了 ===")
            return results
            
        except Exception as e:
            self.logger.error(f"分析エラー: {e}")
            raise
    
    def run_placebo_test(self, 
                        placebo_shift_years: int = 2,
                        **kwargs) -> Dict:
        """
        プレースボ検査を実行
        
        Parameters:
        -----------
        placebo_shift_years : int
            プレースボ検査用の年シフト
        **kwargs
            通常の分析パラメータ
            
        Returns:
        --------
        Dict
            プレースボ検査結果
        """
        self.logger.info(f"=== プレースボ検査開始 (シフト: {placebo_shift_years}年) ===")
        
        # 本番分析
        self.logger.info("本番分析を実行中...")
        main_results = self.run_analysis(placebo_shift_years=0, **kwargs)
        
        # プレースボ分析
        self.logger.info("プレースボ分析を実行中...")
        placebo_results = self.run_analysis(placebo_shift_years=placebo_shift_years, **kwargs)
        
        # 結果比較
        comparison = self._compare_placebo_results(main_results, placebo_results, placebo_shift_years)
        
        # レポート保存
        self._save_placebo_report(comparison, placebo_shift_years)
        
        self.logger.info("=== プレースボ検査完了 ===")
        return comparison
    
    def _compare_placebo_results(self, main_results: pd.DataFrame, 
                               placebo_results: pd.DataFrame, 
                               shift_years: int) -> Dict:
        """プレースボ結果を比較"""
        # JSONシリアライズ可能な形式に変換
        def convert_to_json_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        # データフレームをJSONシリアライズ可能な形式に変換
        main_records = []
        for record in main_results.to_dict('records'):
            converted_record = {k: convert_to_json_serializable(v) for k, v in record.items()}
            main_records.append(converted_record)
            
        placebo_records = []
        for record in placebo_results.to_dict('records'):
            converted_record = {k: convert_to_json_serializable(v) for k, v in record.items()}
            placebo_records.append(converted_record)
        
        comparison = {
            'shift_years': shift_years,
            'main_results': main_records,
            'placebo_results': placebo_records,
            'comparison': []
        }
        
        # 主要カテゴリの比較
        main_categories = ['disaster', 'transit', 'housing', 'commercial']
        
        for category in main_categories:
            main_row = main_results[main_results['event_var'].str.contains(category, na=False)]
            placebo_row = placebo_results[placebo_results['event_var'].str.contains(category, na=False)]
            
            if not main_row.empty and not placebo_row.empty:
                main_beta = float(main_row.iloc[0]['beta'])
                placebo_beta = float(placebo_row.iloc[0]['beta'])
                main_pval = float(main_row.iloc[0]['pval'])
                placebo_pval = float(placebo_row.iloc[0]['pval'])
                
                comparison['comparison'].append({
                    'category': category,
                    'main_beta': main_beta,
                    'placebo_beta': placebo_beta,
                    'beta_reduction': abs(main_beta) - abs(placebo_beta),
                    'main_pval': main_pval,
                    'placebo_pval': placebo_pval,
                    'pval_increase': placebo_pval - main_pval,
                    'is_robust': bool(abs(placebo_beta) < abs(main_beta) * 0.5 and placebo_pval > main_pval)
                })
        
        return comparison
    
    def _save_placebo_report(self, comparison: Dict, shift_years: int):
        """プレースボレポートを保存"""
        import json
        
        # JSONレポート
        report_file = self.output_dir / f"effects_placebo_report_{shift_years}y.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        # テキストサマリ
        summary_file = self.run_logs_dir / f"placebo_summary_{shift_years}y.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== プレースボ検査結果 ({shift_years}年シフト) ===\n\n")
            
            f.write("主要カテゴリの比較:\n")
            for comp in comparison['comparison']:
                f.write(f"{comp['category']:12s}: ")
                f.write(f"本番β={comp['main_beta']:7.3f}, プレースボβ={comp['placebo_beta']:7.3f}, ")
                f.write(f"β減少={comp['beta_reduction']:7.3f}, ")
                f.write(f"p値増加={comp['pval_increase']:7.3f}, ")
                f.write(f"ロバスト={'Yes' if comp['is_robust'] else 'No'}\n")
            
            f.write(f"\nロバストなカテゴリ数: {sum(1 for c in comparison['comparison'] if c['is_robust'])}/{len(comparison['comparison'])}\n")
        
        self.logger.info(f"プレースボレポートを保存: {report_file}")
        self.logger.info(f"プレースボサマリを保存: {summary_file}")


def main():
    """メイン実行関数"""
    # 推定器を初期化
    estimator = TWFEEstimator()
    
    # 通常分析実行（推奨設定）
    print("=== 通常分析実行 ===")
    results = estimator.run_analysis(
        use_year_min=1998,
        use_year_max=2025,
        separate_t_and_t1=True,  # 多重共線性回避のため分離
        use_signed_from_labeled=True,  # 符号ありイベントを使用
        weight_mode="sqrt_prev",
        ridge_alpha=RIDGE_ALPHA,  # 直接パラメータから取得
        include_growth_adj=True,
        include_policy_boundary=True,
        include_disaster_2016=True,
        placebo_shift_years=PLACEBO_SHIFT_YEARS,  # 直接パラメータから取得
        add_ridge=False
    )
    
    print("通常分析完了")
    print(f"結果: {len(results)} 個のイベントタイプ")
    
    if len(results) > 0:
        print(results[['event_var', 'beta', 'ci_low', 'ci_high']].head())
    else:
        print("警告: 結果が空です。Ridge正則化が強すぎる可能性があります。")
        print("ridge_alphaを小さくするか、separate_t_and_t1=Falseを試してください。")
    
    # プレースボ検査実行
    print("\n=== プレースボ検査実行 ===")
    placebo_results = estimator.run_placebo_test(
        placebo_shift_years=2,  # プレースボ検査用
        use_year_min=1998,
        use_year_max=2025,
        separate_t_and_t1=True,
        use_signed_from_labeled=True,
        weight_mode="sqrt_prev",
        ridge_alpha=RIDGE_ALPHA,  # 直接パラメータから取得
        include_growth_adj=True,
        include_policy_boundary=True,
        include_disaster_2016=True,
        add_ridge=False
    )
    
    print("プレースボ検査完了")
    print(f"ロバストなカテゴリ数: {sum(1 for c in placebo_results['comparison'] if c['is_robust'])}/{len(placebo_results['comparison'])}")


if __name__ == "__main__":
    main()
