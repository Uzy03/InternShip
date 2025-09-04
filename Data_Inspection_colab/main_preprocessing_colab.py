#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熊本市統計データ前処理メインスクリプト（Google Colab版）
Data/配下のHTMLベースxlsファイルから100%一貫性のあるCSVファイルを生成
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os

def setup_colab_environment():
    """Colab環境のセットアップ"""
    print("=== Colab環境セットアップ開始 ===")
    
    # 現在のディレクトリを確認
    current_dir = os.getcwd()
    print(f"現在のディレクトリ: {current_dir}")
    
    # Dataディレクトリの存在確認
    data_dir = Path("Data")
    if not data_dir.exists():
        print("❌ Dataディレクトリが見つかりません")
        print("Data/配下にHTMLベースのxlsファイルを配置してください")
        return False
    
    # 必要なディレクトリを作成（ステップ1ではData_csvのみ）
    output_dirs = [
        "Data_csv"
    ]
    
    for dir_name in output_dirs:
        output_dir = Path(dir_name)
        output_dir.mkdir(exist_ok=True)
        print(f"✓ ディレクトリ作成/確認: {dir_name}")
    
    print("✓ Colab環境セットアップ完了")
    return True

def extract_data_from_html_files():
    """HTMLベースのxlsファイルからデータを抽出してCSVに変換"""
    print("\n=== ステップ1: HTMLファイルからのデータ抽出開始 ===")
    
    data_dir = Path("Data")
    output_dir = Path("Data_csv")
    
    # HTMLファイル（.xls拡張子）を取得
    html_files = list(data_dir.glob("*.xls"))
    html_files.sort()
    
    if not html_files:
        print("❌ HTMLファイルが見つかりません")
        return False
    
    print(f"✓ 処理対象ファイル数: {len(html_files)}")
    
    for html_file in html_files:
        print(f"\n処理中: {html_file.name}")
        
        try:
            # 複数のエンコーディングを試行
            encodings = ['shift_jis', 'cp932', 'euc_jp', 'iso2022_jp', 'utf-8', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    # HTMLファイルとして読み込み
                    df = pd.read_html(html_file, encoding=encoding)[0]
                    print(f"  ✓ エンコーディング '{encoding}' で読み込み成功")
                    break
                except Exception as e:
                    continue
            
            if df is None:
                print(f"  ✗ すべてのエンコーディングで読み込み失敗")
                continue
            
            # 年月度を抽出してファイル名を生成
            filename = extract_year_month_from_data(df)
            if filename:
                output_filename = f"{filename}.csv"
                output_path = output_dir / output_filename
                
                # CSVとして保存（UTF-8 with BOM）
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"  ✓ 保存完了: {output_filename}")
            else:
                print(f"  ⚠️  年月度を特定できませんでした")
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            continue
    
    print("✓ データ抽出処理完了")
    return True

def extract_year_month_from_data(dataframe):
    """データフレームから年月度を抽出してファイル名を生成"""
    try:
        # 最初の数行を確認して年月度を探す
        for i in range(min(10, len(dataframe))):
            row_text = ' '.join(str(cell) for cell in dataframe.iloc[i] if pd.notna(cell))
            
            # 年号と年を抽出
            era = None
            year = None
            
            if '平成' in row_text:
                era = 'H'
                year_match = re.search(r'平成(\d+)年', row_text)
                if year_match:
                    year = int(year_match.group(1))
            elif '令和' in row_text:
                era = 'R'
                year_match = re.search(r'令和(\d+)年', row_text)
                if year_match:
                    year = int(year_match.group(1))
            
            if era and year is not None:
                # 月を抽出
                month_match = re.search(r'(\d+)月', row_text)
                if month_match:
                    month = int(month_match.group(1))
                    
                    # ファイル名を生成（例：H10-04, R01-04, R07-04）
                    filename = f"{era}{year:02d}-{month:02d}"
                    print(f"  ✓ 年月度を特定: {era}{year}年{month}月 → {filename}")
                    return filename
        
        # 年月度が見つからない場合
        print("  ⚠️  年月度を特定できませんでした")
        return None
        
    except Exception as e:
        print(f"  年月度抽出中にエラー: {e}")
        return None

def main():
    """メイン処理"""
    print("=== 熊本市統計データ前処理開始（Google Colab版） ===")
    
    # ステップ1: 環境セットアップ
    if not setup_colab_environment():
        return
    
    # ステップ2: HTMLファイルからのデータ抽出
    if not extract_data_from_html_files():
        return
    
    print("\n=== ステップ1完了 ===")
    print("✓ HTMLファイルからのデータ抽出が完了しました")
    print("✓ 次のステップ: 町丁一貫性分析と前処理")
    print("\n次のスクリプトを実行してください:")
    print("python town_consistency_analysis_colab.py")

if __name__ == "__main__":
    main()
