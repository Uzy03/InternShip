#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善版人口変動分析スクリプト
新しい基準で人口変化を分類する
prev=前年人口、Δ=人口増減、g=Δ/prevとした時の条件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ImprovedPopulationChangeAnalyzer:
    def __init__(self, data_path="subject2/processed_data/population_time_series.csv"):
        self.data_path = data_path
        self.data = None
        self.change_data = None
        self.anomaly_data = None
        
    def load_data(self):
        """時系列人口データを読み込み"""
        print("データの読み込み中...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        print(f"データ読み込み完了: {self.data.shape}")
        return self.data
    
    def calculate_population_changes(self):
        """人口変化率を計算"""
        print("人口変化率の計算中...")
        
        # 年度列を特定
        year_cols = [col for col in self.data.columns if '_総人口' in col]
        years = sorted([int(col.split('_')[0]) for col in year_cols])
        
        # 変化率データフレームを作成
        change_data = {'町丁名': self.data['町丁名']}
        
        for i in range(1, len(years)):
            current_year = years[i]
            previous_year = years[i-1]
            
            current_col = f'{current_year}_総人口'
            previous_col = f'{previous_year}_総人口'
            
            # 変化率を計算
            change_rate = ((self.data[current_col] - self.data[previous_col]) / 
                          self.data[previous_col] * 100)
            
            # 絶対変化数も計算
            change_absolute = self.data[current_col] - self.data[previous_col]
            
            change_data[f'{current_year}_変化率(%)'] = change_rate
            change_data[f'{current_year}_変化数'] = change_absolute
        
        self.change_data = pd.DataFrame(change_data)
        print(f"変化率計算完了: {self.change_data.shape}")
        return self.change_data
    
    def classify_population_changes(self):
        """人口変化を新しい基準で分類"""
        print("人口変化の分類中...")
        
        if self.change_data is None:
            print("エラー: 先に人口変化率を計算してください")
            return None
        
        # 変化率列を特定
        change_rate_cols = [col for col in self.change_data.columns if '変化率' in col]
        
        classified_changes = []
        
        for col in change_rate_cols:
            year = col.split('_')[0]
            
            # 各町丁の変化率を処理
            for idx, change_rate in self.change_data[col].items():
                if pd.isna(change_rate):
                    continue
                    
                town_name = self.change_data.loc[idx, '町丁名']
                
                # 人口増減数も取得
                change_absolute_col = col.replace('変化率(%)', '変化数')
                if change_absolute_col in self.change_data.columns:
                    change_absolute = self.change_data.loc[idx, change_absolute_col]
                else:
                    change_absolute = 0
                
                # 前年人口を計算（変化率から逆算）
                if change_rate != 0:
                    prev_population = abs(change_absolute / (change_rate / 100))
                else:
                    prev_population = 0
                
                # 変化の分類
                change_type = self._classify_change(change_rate, change_absolute, prev_population)
                
                if change_type != '変化なし':
                    classified_changes.append({
                        '町丁名': town_name,
                        '年度': year,
                        '変化率(%)': change_rate,
                        '変化タイプ': change_type,
                        '変化の大きさ': self._get_change_magnitude(change_rate),
                        '前年人口': prev_population,
                        '人口増減': change_absolute
                    })
        
        classified_df = pd.DataFrame(classified_changes)
        
        if len(classified_df) > 0:
            classified_df = classified_df.sort_values('変化率(%)', ascending=False)
        
        print(f"人口変化の分類完了: {len(classified_df)}件")
        return classified_df
    
    def _classify_change(self, change_rate, change_absolute, prev_population):
        """個別の変化率を分類（5クラス分類）"""
        if pd.isna(change_rate):
            return 'データなし'
        
        if change_rate == 0:
            return '変化なし'
        
        # 変化率を小数に変換（例：100% → 1.0）
        g = change_rate / 100
        delta = change_absolute
        
        if change_rate > 0:  # 増加
            # 激増：g ≥ +1.00 かつ Δ ≥ +100、または（小母数特例）prev < 50 かつ Δ ≥ +80、または Δ ≥ +200
            if (g >= 1.0 and delta >= 100) or (prev_population < 50 and delta >= 80) or (delta >= 200):
                return '激増'
            # 大幅増加：+0.30 ≤ g < +1.00 かつ Δ ≥ +50、または（小母数特例）prev < 50 かつ Δ ≥ +40
            elif (0.30 <= g < 1.0 and delta >= 50) or (prev_population < 50 and delta >= 40):
                return '大幅増加'
            else:
                return '自然増減'
        else:  # 減少
            # 激減：g ≤ -1.00 かつ Δ ≤ -100、または Δ ≤ -200
            if (g <= -1.0 and delta <= -100) or (delta <= -200):
                return '激減'
            # 大幅減少：-1.00 < g ≤ -0.30 かつ Δ ≤ -100
            elif -1.0 < g <= -0.30 and delta <= -100:
                return '大幅減少'
            else:
                return '自然増減'
    
    def _get_change_magnitude(self, change_rate):
        """変化の大きさを数値で表現"""
        return abs(change_rate)
    
    def create_detailed_summary(self):
        """詳細な分類サマリーを作成"""
        print("詳細分類サマリーの作成中...")
        
        if self.change_data is None:
            print("エラー: 先に人口変化率を計算してください")
            return None
        
        # 変化率列を特定
        change_rate_cols = [col for col in self.change_data.columns if '変化率' in col]
        
        summary_data = []
        
        for col in change_rate_cols:
            year = col.split('_')[0]
            change_rates = self.change_data[col].dropna()
            
            if len(change_rates) > 0:
                # 各分類の件数をカウント
                classified_counts = self._count_by_classification(change_rates)
                
                summary = {
                    '年度': year,
                    '対象町丁数': len(change_rates),
                    '平均変化率(%)': change_rates.mean(),
                    '変化率標準偏差(%)': change_rates.std(),
                    '最大増加率(%)': change_rates.max(),
                    '最大減少率(%)': change_rates.min(),
                    **classified_counts
                }
                summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        print(f"詳細分類サマリー作成完了: {len(summary_df)}年度分")
        return summary_df
    
    def _count_by_classification(self, change_rates):
        """変化率を分類別にカウント（5クラス分類）"""
        counts = {}
        
        # 5クラス分類に基づくカウント
        # 増加の分類
        counts['激増件数'] = len(change_rates[change_rates >= 100.0])  # g ≥ +1.00 (100%以上)
        counts['大幅増加件数'] = len(change_rates[(change_rates >= 30.0) & (change_rates < 100.0)])  # +0.30 ≤ g < +1.00
        
        # 減少の分類
        counts['激減件数'] = len(change_rates[change_rates <= -100.0])  # g ≤ -1.00
        counts['大幅減少件数'] = len(change_rates[(change_rates <= -30.0) & (change_rates > -100.0)])  # -1.00 < g ≤ -0.30
        
        # 変化なし
        counts['変化なし件数'] = len(change_rates[change_rates == 0])
        
        # 自然増減：激増、大幅増加、激減、大幅減少、変化なし以外すべて
        # つまり、-30% < g < +30% の範囲で、かつ変化なし以外
        counts['自然増減件数'] = len(change_rates[(change_rates > -30.0) & (change_rates < 30.0) & (change_rates != 0)])
        
        return counts
    
    def save_improved_results(self, output_dir="subject2/processed_data"):
        """改善された分析結果を保存"""
        print("改善された分析結果の保存中...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 詳細分類データ
        classified_changes = self.classify_population_changes()
        if classified_changes is not None and len(classified_changes) > 0:
            classified_path = os.path.join(output_dir, "classified_population_changes.csv")
            classified_changes.to_csv(classified_path, index=False, encoding='utf-8')
            print(f"詳細分類データ保存: {classified_path}")
        
        # 詳細サマリー
        detailed_summary = self.create_detailed_summary()
        if detailed_summary is not None:
            summary_path = os.path.join(output_dir, "detailed_population_summary.csv")
            detailed_summary.to_csv(summary_path, index=False, encoding='utf-8')
            print(f"詳細サマリー保存: {summary_path}")
        
        print("改善された分析結果の保存完了")
    
    def run_improved_analysis(self):
        """改善された全分析を実行"""
        print("=== 改善された人口変動分析開始 ===")
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. 人口変化率計算
        self.calculate_population_changes()
        
        # 3. 詳細分類
        classified_changes = self.classify_population_changes()
        
        # 4. 詳細サマリー作成
        detailed_summary = self.create_detailed_summary()
        
        # 5. 結果保存
        self.save_improved_results()
        
        print("=== 改善された人口変動分析完了 ===")
        
        return {
            'classified_changes': classified_changes,
            'detailed_summary': detailed_summary
        }

if __name__ == "__main__":
    # 改善された分析の実行
    analyzer = ImprovedPopulationChangeAnalyzer()
    results = analyzer.run_improved_analysis()
    
    print("\n=== 改善された分析結果サマリー ===")
    print(f"詳細分類データ: {len(results['classified_changes'])}件")
    print(f"対象年度数: {len(results['detailed_summary'])}年")
    
    # 分類別の件数表示
    if len(results['classified_changes']) > 0:
        change_types = results['classified_changes']['変化タイプ'].value_counts()
        print("\n変化タイプ別件数:")
        for change_type, count in change_types.items():
            print(f"  {change_type}: {count}件")
