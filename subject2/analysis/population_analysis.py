#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人口変動分析スクリプト
自然増減を除いた社会増減の特定と異常値検出を行う
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

class PopulationChangeAnalyzer:
    def __init__(self, data_path="processed_data/population_time_series.csv"):
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
    
    def detect_anomalies(self, threshold=2.0):
        """異常値検出（Z-score法）"""
        print("異常値検出中...")
        
        if self.change_data is None:
            print("エラー: 先に人口変化率を計算してください")
            return None
        
        # 変化率列を特定
        change_rate_cols = [col for col in self.change_data.columns if '変化率' in col]
        
        anomaly_data = {'町丁名': self.change_data['町丁名']}
        
        for col in change_rate_cols:
            # 欠損値を除外
            valid_data = self.change_data[col].dropna()
            
            if len(valid_data) > 0:
                # Z-scoreを計算
                z_scores = np.abs(stats.zscore(valid_data))
                
                # 異常値のインデックスを取得
                anomaly_indices = np.where(z_scores > threshold)[0]
                
                # 元のデータのインデックスに戻す
                original_indices = valid_data.index[anomaly_indices]
                
                # 異常値フラグを作成
                anomaly_flag = pd.Series(False, index=self.change_data.index)
                anomaly_flag.loc[original_indices] = True
                
                anomaly_data[f'{col}_異常値フラグ'] = anomaly_flag
                anomaly_data[f'{col}_Zスコア'] = pd.Series(
                    z_scores, index=valid_data.index
                ).reindex(self.change_data.index)
        
        self.anomaly_data = pd.DataFrame(anomaly_data)
        print(f"異常値検出完了: {self.anomaly_data.shape}")
        return self.anomaly_data
    
    def identify_major_changes(self, min_change_rate=5.0):
        """大きな人口変化を特定"""
        print("大きな人口変化の特定中...")
        
        if self.change_data is None:
            print("エラー: 先に人口変化率を計算してください")
            return None
        
        # 変化率列を特定
        change_rate_cols = [col for col in self.change_data.columns if '変化率' in col]
        
        major_changes = []
        
        for col in change_rate_cols:
            year = col.split('_')[0]
            
            # 大きな変化（増加・減少）を特定
            large_increase = self.change_data[
                self.change_data[col] > min_change_rate
            ][['町丁名', col]]
            
            large_decrease = self.change_data[
                self.change_data[col] < -min_change_rate
            ][['町丁名', col]]
            
            for _, row in large_increase.iterrows():
                major_changes.append({
                    '町丁名': row['町丁名'],
                    '年度': year,
                    '変化率(%)': row[col],
                    '変化タイプ': '大幅増加'
                })
            
            for _, row in large_decrease.iterrows():
                major_changes.append({
                    '町丁名': row['町丁名'],
                    '年度': year,
                    '変化率(%)': row[col],
                    '変化タイプ': '大幅減少'
                })
        
        major_changes_df = pd.DataFrame(major_changes)
        
        if len(major_changes_df) > 0:
            major_changes_df = major_changes_df.sort_values('変化率(%)', ascending=False)
        
        print(f"大きな人口変化の特定完了: {len(major_changes_df)}件")
        return major_changes_df
    
    def create_summary_report(self):
        """分析結果のサマリーレポートを作成"""
        print("サマリーレポートの作成中...")
        
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
                summary = {
                    '年度': year,
                    '対象町丁数': len(change_rates),
                    '平均変化率(%)': change_rates.mean(),
                    '変化率標準偏差(%)': change_rates.std(),
                    '最大増加率(%)': change_rates.max(),
                    '最大減少率(%)': change_rates.min(),
                    '増加町丁数': len(change_rates[change_rates > 0]),
                    '減少町丁数': len(change_rates[change_rates < 0]),
                    '変化なし町丁数': len(change_rates[change_rates == 0])
                }
                summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        print(f"サマリーレポート作成完了: {len(summary_df)}年度分")
        return summary_df
    
    def save_results(self, output_dir="processed_data"):
        """分析結果を保存"""
        print("分析結果の保存中...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 変化率データ
        if self.change_data is not None:
            change_path = os.path.join(output_dir, "population_changes.csv")
            self.change_data.to_csv(change_path, index=False, encoding='utf-8')
            print(f"変化率データ保存: {change_path}")
        
        # 異常値データ
        if self.anomaly_data is not None:
            anomaly_path = os.path.join(output_dir, "population_anomalies.csv")
            self.anomaly_data.to_csv(anomaly_path, index=False, encoding='utf-8')
            print(f"異常値データ保存: {anomaly_path}")
        
        # 大きな変化データ
        major_changes = self.identify_major_changes()
        if major_changes is not None and len(major_changes) > 0:
            major_path = os.path.join(output_dir, "major_population_changes.csv")
            major_changes.to_csv(major_path, index=False, encoding='utf-8')
            print(f"大きな変化データ保存: {major_path}")
        
        # サマリーレポート
        summary = self.create_summary_report()
        if summary is not None:
            summary_path = os.path.join(output_dir, "population_change_summary.csv")
            summary.to_csv(summary_path, index=False, encoding='utf-8')
            print(f"サマリーレポート保存: {summary_path}")
        
        print("分析結果の保存完了")
    
    def run_analysis(self):
        """全分析を実行"""
        print("=== 人口変動分析開始 ===")
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. 人口変化率計算
        self.calculate_population_changes()
        
        # 3. 異常値検出
        self.detect_anomalies()
        
        # 4. 大きな変化の特定
        major_changes = self.identify_major_changes()
        
        # 5. サマリーレポート作成
        summary = self.create_summary_report()
        
        # 6. 結果保存
        self.save_results()
        
        print("=== 人口変動分析完了 ===")
        
        return {
            'change_data': self.change_data,
            'anomaly_data': self.anomaly_data,
            'major_changes': major_changes,
            'summary': summary
        }

if __name__ == "__main__":
    # 分析の実行
    analyzer = PopulationChangeAnalyzer()
    results = analyzer.run_analysis()
    
    print("\n=== 分析結果サマリー ===")
    print(f"変化率データ: {results['change_data'].shape}")
    print(f"異常値データ: {results['anomaly_data'].shape}")
    print(f"大きな変化件数: {len(results['major_changes'])}件")
    print(f"対象年度数: {len(results['summary'])}年")
