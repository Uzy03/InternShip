#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_consistent.csvファイルに5歳階級の人口コラムを追加し、コラム名を元の表記に戻すスクリプト
"""

import pandas as pd
import os
from pathlib import Path
import re

def get_age_columns():
    """5歳階級のコラム名を取得"""
    age_columns = [
        '0〜4歳', '5〜9歳', '10〜14歳', '15〜19歳', '20〜24歳', '25〜29歳', '30〜34歳',
        '35〜39歳', '40〜44歳', '45〜49歳', '50〜54歳', '55〜59歳', '60〜64歳',
        '65〜69歳', '70〜74歳', '75〜79歳', '80〜84歳', '85〜89歳', '90〜94歳',
        '95〜99歳', '100歳以上'
    ]
    return age_columns

def extract_population_data(original_df, town_name):
    """元のデータから指定された町丁の人口データを抽出"""
    population_data = {}
    age_columns = get_age_columns()
    
    # 町丁名を検索
    town_found = False
    for i, row in original_df.iterrows():
        if str(row.iloc[0]).strip() == town_name:
            town_found = True
            continue
        
        if town_found:
            # 次の町丁名が見つかったら終了
            if str(row.iloc[0]).strip() and not any(age in str(row.iloc[0]) for age in age_columns):
                break
            
            # 年齢区分の行を処理
            age_group = str(row.iloc[0]).strip()
            if age_group in age_columns:
                # 総人口（男・女の合計）
                try:
                    male = int(str(row.iloc[1]).replace(',', '')) if pd.notna(row.iloc[1]) else 0
                    female = int(str(row.iloc[2]).replace(',', '')) if pd.notna(row.iloc[2]) else 0
                    total = male + female
                    population_data[age_group] = total
                except (ValueError, TypeError):
                    population_data[age_group] = 0
    
    return population_data

def add_population_columns():
    """_consistent.csvファイルに人口コラムを追加"""
    try:
        print("=== 人口コラム追加処理開始 ===")
        
        # ディレクトリ設定
        original_data_dir = Path("../Data_csv")
        consistent_data_dir = Path("../Preprocessed_Data_csv_成功")
        output_dir = Path("../Preprocessed_Data_csv_with_population")
        
        if not original_data_dir.exists():
            print(f"✗ 元データディレクトリが見つかりません: {original_data_dir}")
            return False
        
        if not consistent_data_dir.exists():
            print(f"✗ _consistent.csvディレクトリが見つかりません: {consistent_data_dir}")
            return False
        
        # 出力ディレクトリを作成
        output_dir.mkdir(exist_ok=True)
        print(f"出力先: {output_dir.absolute()}")
        
        # 5歳階級のコラム名
        age_columns = get_age_columns()
        
        # _consistent.csvファイルを検索
        consistent_files = list(consistent_data_dir.glob("*_consistent.csv"))
        if not consistent_files:
            print("_consistent.csvファイルが見つかりませんでした")
            return False
        
        print(f"処理対象ファイル数: {len(consistent_files)}")
        
        # 各ファイルを処理
        for consistent_file in sorted(consistent_files):
            print(f"\n処理中: {consistent_file.name}")
            
            # 対応する元データファイルを検索
            year_month = consistent_file.stem.replace('_consistent', '')
            original_file = original_data_dir / f"{year_month}.csv"
            
            if not original_file.exists():
                print(f"⚠️  元データファイルが見つかりません: {original_file.name}")
                continue
            
            # 元データを読み込み
            try:
                original_df = pd.read_csv(original_file, encoding='utf-8-sig')
                print(f"✓ 元データ読み込み完了: {len(original_df)}行")
            except Exception as e:
                print(f"✗ 元データ読み込み失敗: {e}")
                continue
            
            # _consistent.csvを読み込み
            try:
                consistent_df = pd.read_csv(consistent_file, encoding='utf-8-sig')
                print(f"✓ _consistent.csv読み込み完了: {len(consistent_df)}行")
            except Exception as e:
                print(f"✗ _consistent.csv読み込み失敗: {e}")
                continue
            
            # 町丁名の列を特定（最初の列）
            town_column = consistent_df.columns[0]
            print(f"町丁名列: {town_column}")
            
            # 人口データを追加
            population_data_list = []
            
            for _, row in consistent_df.iterrows():
                town_name = str(row[town_column]).strip()
                if pd.isna(town_name) or town_name == '':
                    population_data_list.append({col: 0 for col in age_columns})
                    continue
                
                # 元データから人口データを抽出
                town_population = extract_population_data(original_df, town_name)
                
                # 不足している年齢区分は0で補完
                for age_col in age_columns:
                    if age_col not in town_population:
                        town_population[age_col] = 0
                
                population_data_list.append(town_population)
            
            # 人口データをデータフレームに変換
            population_df = pd.DataFrame(population_data_list)
            
            # 元のデータフレームと結合
            result_df = pd.concat([consistent_df, population_df], axis=1)
            
            # 出力ファイル名
            output_file = output_dir / f"{year_month}_with_population.csv"
            
            # CSVファイルとして保存
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✓ 処理完了: {output_file.name}")
            print(f"  元の列数: {len(consistent_df.columns)}")
            print(f"  追加後の列数: {len(result_df.columns)}")
            print(f"  追加された人口コラム数: {len(age_columns)}")
        
        print(f"\n=== 処理完了 ===")
        print(f"出力ディレクトリ: {output_dir.absolute()}")
        
        # サンプルファイルの内容を表示
        if output_dir.exists():
            sample_files = list(output_dir.glob("*_with_population.csv"))
            if sample_files:
                sample_file = sample_files[0]
                print(f"\nサンプルファイル: {sample_file.name}")
                
                sample_df = pd.read_csv(sample_file, encoding='utf-8-sig')
                print(f"列名一覧:")
                for i, col in enumerate(sample_df.columns):
                    print(f"  {i+1:2d}. {col}")
                
                print(f"\n最初の5行:")
                print(sample_df.head())
        
        return True
        
    except Exception as e:
        print(f"✗ 処理に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=== 人口コラム追加ツール ===")
    
    print(f"\nこのツールは以下の処理を実行します:")
    print(f"1. _consistent.csvファイルを読み込み")
    print(f"2. 対応する元データファイルから5歳階級の人口データを抽出")
    print(f"3. 人口コラムを追加（コラム名は元の表記）")
    print(f"4. 新しいCSVファイルとして保存")
    
    confirm = input(f"\n処理を開始しますか？ (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        add_population_columns()
    else:
        print("処理をキャンセルしました")

if __name__ == "__main__":
    main()
