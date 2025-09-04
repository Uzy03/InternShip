#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
町丁別人口データ整形スクリプト
28年度分の町丁別人口データを時系列分析用に整形する
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PopulationDataProcessor:
    def __init__(self, data_dir="100_Percent_Consistent_Data/CSV_Files"):
        self.data_dir = data_dir
        self.output_dir = "subject2/processed_data"
        self.raw_data = {}
        self.processed_data = None
        
    def load_all_data(self):
        """28年度分のCSVファイルを読み込み"""
        print("データファイルの読み込みを開始...")
        
        # CSVファイルのパスを取得
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        csv_files.sort()  # 年度順にソート
        
        print(f"発見されたCSVファイル数: {len(csv_files)}")
        
        for file_path in csv_files:
            # ファイル名から年度を抽出
            filename = os.path.basename(file_path)
            if "H" in filename:
                # 平成年度
                if "H10" in filename:
                    year = 1998
                elif "H11" in filename:
                    year = 1999
                elif "H12" in filename:
                    year = 2000
                elif "H13" in filename:
                    year = 2001
                elif "H14" in filename:
                    year = 2002
                elif "H15" in filename:
                    year = 2003
                elif "H16" in filename:
                    year = 2004
                elif "H17" in filename:
                    year = 2005
                elif "H18" in filename:
                    year = 2006
                elif "H19" in filename:
                    year = 2007
                elif "H20" in filename:
                    year = 2008
                elif "H21" in filename:
                    year = 2009
                elif "H22" in filename:
                    year = 2010
                elif "H23" in filename:
                    year = 2011
                elif "H24" in filename:
                    year = 2012
                elif "H25" in filename:
                    year = 2013
                elif "H26" in filename:
                    year = 2014
                elif "H27" in filename:
                    year = 2015
                elif "H28" in filename:
                    year = 2016
                elif "H29" in filename:
                    year = 2017
                elif "H30" in filename:
                    year = 2018
                elif "H31" in filename:
                    year = 2019
            elif "R" in filename:
                # 令和年度
                if "R02" in filename:
                    year = 2020
                elif "R03" in filename:
                    year = 2021
                elif "R04" in filename:
                    year = 2022
                elif "R05" in filename:
                    year = 2023
                elif "R06" in filename:
                    year = 2024
                elif "R07" in filename:
                    year = 2025
            
            print(f"読み込み中: {filename} -> {year}年")
            
            try:
                # CSVファイルを読み込み
                df = pd.read_csv(file_path, encoding='utf-8')
                self.raw_data[year] = df
            except Exception as e:
                print(f"エラー: {filename}の読み込みに失敗 - {e}")
        
        print(f"読み込み完了: {len(self.raw_data)}年度分のデータ")
        
    def extract_population_data(self):
        """各年度のデータから人口データを抽出（修正後のファイル形式に対応）"""
        print("人口データの抽出を開始...")
        
        population_data = {}
        
        for year, df in self.raw_data.items():
            print(f"処理中: {year}年")
            
            # 修正後のファイル形式では、既に整形されたデータが含まれている
            # 町丁名、総人口、男性、女性の列を直接取得
            if '町丁名' in df.columns and '総人口' in df.columns:
                # 新しい形式のファイル
                towns = df['町丁名'].tolist()
                total_pop = df['総人口'].tolist()
                male_pop = df['男性'].tolist()
                female_pop = df['女性'].tolist()
                
                # 有効なデータのみを抽出
                valid_data = []
                for i in range(len(towns)):
                    town_name = str(towns[i]).strip()
                    total = self._parse_population(total_pop[i])
                    male = self._parse_population(male_pop[i])
                    female = self._parse_population(female_pop[i])
                    
                    if (town_name and 
                        town_name != "nan" and 
                        town_name != "人口統計表" and
                        "【秘匿】" not in town_name and
                        total is not None and total > 0):
                        
                        valid_data.append({
                            '町丁名': town_name,
                            '総人口': total,
                            '男性': male,
                            '女性': female
                        })
                
                year_df = pd.DataFrame(valid_data)
            else:
                # 古い形式のファイル（フォールバック）
                towns = []
                total_pop = []
                male_pop = []
                female_pop = []
                
                for i in range(len(df)):
                    row = df.iloc[i]
                    
                    # 町丁名の行を特定（「総数」の前の行）
                    if i + 1 < len(df) and "総数" in str(df.iloc[i + 1, 0]):
                        town_name = str(row.iloc[0]).strip()
                        
                        # 有効な町丁名かチェック
                        if (town_name and 
                            town_name != "nan" and 
                            town_name != "人口統計表" and
                            "【秘匿】" not in town_name and
                            "χ" not in str(row.iloc[1])):
                            
                            # 次の行の「総数」から人口データを取得
                            next_row = df.iloc[i + 1]
                            
                            try:
                                total = self._parse_population(next_row.iloc[1])
                                male = self._parse_population(next_row.iloc[2])
                                female = self._parse_population(next_row.iloc[3])
                                
                                if total is not None and total > 0:
                                    towns.append(town_name)
                                    total_pop.append(total)
                                    male_pop.append(male)
                                    female_pop.append(female)
                            except:
                                continue
                
                # 年度ごとのデータフレームを作成
                year_df = pd.DataFrame({
                    '町丁名': towns,
                    '総人口': total_pop,
                    '男性': male_pop,
                    '女性': female_pop
                })
            
            population_data[year] = year_df
            print(f"  {year}年: {len(year_df)}町丁のデータを抽出")
        
        self.processed_data = population_data
        print("人口データの抽出完了")
        
    def _parse_population(self, value):
        """人口数値をパース"""
        if pd.isna(value) or value == "χ":
            return None
        
        # 文字列の場合、カンマを除去して数値に変換
        if isinstance(value, str):
            value = value.replace(",", "")
            if value == "χ" or value == "":
                return None
        
        try:
            return int(float(value))
        except:
            return None
    
    def create_time_series_data(self):
        """時系列分析用のデータフレームを作成（欠損町丁も含む）"""
        print("時系列データの作成を開始...")
        
        if not self.processed_data:
            print("エラー: 先に人口データの抽出を実行してください")
            return None
        
        # 全年度で出現するすべての町丁名を収集
        all_towns = set()
        for year, df in self.processed_data.items():
            all_towns.update(df['町丁名'].tolist())
        
        print(f"全期間で出現する町丁数: {len(all_towns)}")
        
        # 時系列データフレームを作成
        years = sorted(self.processed_data.keys())
        time_series_data = []
        
        for town in sorted(all_towns):
            town_data = {'町丁名': town}
            
            for year in years:
                year_df = self.processed_data[year]
                town_row = year_df[year_df['町丁名'] == town]
                
                if not town_row.empty:
                    town_data[f'{year}_総人口'] = town_row.iloc[0]['総人口']
                    town_data[f'{year}_男性'] = town_row.iloc[0]['男性']
                    town_data[f'{year}_女性'] = town_row.iloc[0]['女性']
                else:
                    # その年度にデータがない場合はNaNを設定
                    town_data[f'{year}_総人口'] = np.nan
                    town_data[f'{year}_男性'] = np.nan
                    town_data[f'{year}_女性'] = np.nan
            
            time_series_data.append(town_data)
        
        # データフレームに変換
        time_series_df = pd.DataFrame(time_series_data)
        
        # 出力ディレクトリを作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存
        output_path = os.path.join(self.output_dir, "population_time_series.csv")
        time_series_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"時系列データを保存: {output_path}")
        print(f"データ形状: {time_series_df.shape}")
        
        return time_series_df
    
    def create_summary_statistics(self):
        """基本統計情報を作成"""
        print("基本統計情報の作成を開始...")
        
        if not self.processed_data:
            print("エラー: 先に人口データの抽出を実行してください")
            return None
        
        summary_data = []
        
        for year, df in self.processed_data.items():
            summary = {
                '年度': year,
                '町丁数': len(df),
                '総人口': df['総人口'].sum(),
                '男性人口': df['男性'].sum(),
                '女性人口': df['女性'].sum(),
                '平均人口': df['総人口'].mean(),
                '最大人口': df['総人口'].max(),
                '最小人口': df['総人口'].min(),
                '人口標準偏差': df['総人口'].std()
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存
        output_path = os.path.join(self.output_dir, "population_summary.csv")
        summary_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"基本統計情報を保存: {output_path}")
        
        return summary_df
    
    def process_all(self):
        """全処理を実行"""
        print("=== 町丁別人口データ処理開始 ===")
        
        # 1. データ読み込み
        self.load_all_data()
        
        # 2. 人口データ抽出
        self.extract_population_data()
        
        # 3. 時系列データ作成
        time_series_df = self.create_time_series_data()
        
        # 4. 基本統計情報作成
        summary_df = self.create_summary_statistics()
        
        print("=== 処理完了 ===")
        
        return time_series_df, summary_df

if __name__ == "__main__":
    # データ処理の実行
    processor = PopulationDataProcessor()
    time_series_df, summary_df = processor.process_all()
    
    print("\n=== 処理結果サマリー ===")
    print(f"時系列データ: {time_series_df.shape[0]}町丁 × {time_series_df.shape[1]}列")
    print(f"対象期間: {summary_df['年度'].min()}年 〜 {summary_df['年度'].max()}年")
    print(f"総年度数: {len(summary_df)}年")
