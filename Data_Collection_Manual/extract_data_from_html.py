#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTMLファイルからデータを抽出するスクリプト
熊本市統計データのHTMLファイルからテーブルデータを抽出してCSVに保存
"""

import pandas as pd
from bs4 import BeautifulSoup
import os
from pathlib import Path
import re
from datetime import datetime

def extract_year_month_from_data(dataframe):
    """データフレームから年月度を抽出してファイル名を生成"""
    try:
        # 最初の数行を確認して年月度を探す
        for i in range(min(10, len(dataframe))):
            row_text = ' '.join(str(cell) for cell in dataframe.iloc[i] if pd.notna(cell))
            
            # 平成/令和の年号を探す
            if '平成' in row_text or '令和' in row_text:
                # 年号と年を抽出
                if '平成' in row_text:
                    era = 'H'
                    era_text = '平成'
                elif '令和' in row_text:
                    era = 'R'
                    era_text = '令和'
                else:
                    continue
                
                # 年を抽出（1-2桁の数字）
                year_match = re.search(f'{era_text}(\\d+)年', row_text)
                if year_match:
                    year = int(year_match.group(1))
                    
                    # 月を抽出
                    month_match = re.search(f'(\\d+)月', row_text)
                    if month_match:
                        month = int(month_match.group(1))
                        
                        # ファイル名を生成（例：H10-04, R01-04, R07-04）
                        filename = f"{era}{year:02d}-{month:02d}"
                        print(f"✓ 年月度を特定: {era_text}{year}年{month}月 → {filename}")
                        return filename
        
        # 年月度が見つからない場合
        print("⚠️  年月度を特定できませんでした")
        return None
        
    except Exception as e:
        print(f"年月度抽出中にエラー: {e}")
        return None

def extract_data_from_html(html_file, output_file=None):
    """HTMLファイルからデータを抽出してCSVに保存"""
    try:
        print(f"HTMLファイル処理中: {html_file}")
        
        # 複数のエンコーディングで試行
        encodings_to_try = ['shift_jis', 'cp932', 'utf-8', 'euc_jp']
        html_content = None
        successful_encoding = None
        
        for encoding in encodings_to_try:
            try:
                print(f"エンコーディング '{encoding}' を試行中...")
                with open(html_file, 'r', encoding=encoding, errors='ignore') as f:
                    html_content = f.read()
                print(f"✓ エンコーディング '{encoding}' で読み込み成功")
                successful_encoding = encoding
                break
            except Exception as e:
                print(f"✗ エンコーディング '{encoding}' で失敗: {e}")
                continue
        
        if html_content is None:
            print("すべてのエンコーディングで読み込みに失敗しました")
            return False
        
        print(f"使用エンコーディング: {successful_encoding}")
        
        # BeautifulSoupでパース
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # ページタイトルを確認
        title = soup.find('title')
        if title:
            print(f"ページタイトル: {title.text}")
        
        # テーブル要素を検索
        tables = soup.find_all('table')
        print(f"テーブル数: {len(tables)}")
        
        if not tables:
            print("テーブル要素が見つかりませんでした")
            return False
        
        # 各テーブルの内容を確認
        extracted_data = []
        
        for i, table in enumerate(tables):
            print(f"\n=== テーブル {i+1} ===")
            
            # テーブルの属性を確認
            table_attrs = table.attrs
            print(f"テーブル属性: {table_attrs}")
            
            # テーブル内の行を取得
            rows = table.find_all('tr')
            print(f"行数: {len(rows)}")
            
            if len(rows) > 0:
                # ヘッダー行を取得
                headers = []
                header_row = rows[0]
                header_cells = header_row.find_all(['th', 'td'])
                
                for cell in header_cells:
                    header_text = cell.get_text(strip=True)
                    if header_text:
                        headers.append(header_text)
                
                print(f"ヘッダー: {headers}")
                
                # データ行を処理
                table_data = []
                for row in rows[1:]:  # ヘッダー行を除く
                    cells = row.find_all(['td', 'th'])
                    row_data = []
                    
                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        row_data.append(cell_text)
                    
                    if any(cell_text for cell_text in row_data):  # 空行でない場合
                        table_data.append(row_data)
                
                print(f"データ行数: {len(table_data)}")
                
                if table_data:
                    # 列数を統一
                    max_cols = max(len(row) for row in table_data)
                    for row in table_data:
                        while len(row) < max_cols:
                            row.append('')
                    
                    # ヘッダー数も調整
                    while len(headers) < max_cols:
                        headers.append(f'Column_{len(headers)}')
                    
                    # DataFrameを作成
                    df = pd.DataFrame(table_data, columns=headers)
                    extracted_data.append({
                        'table_index': i,
                        'dataframe': df,
                        'headers': headers,
                        'row_count': len(table_data)
                    })
                    
                    print(f"テーブル {i+1} の最初の5行:")
                    print(df.head())
        
        if not extracted_data:
            print("抽出可能なデータが見つかりませんでした")
            return False
        
        # 出力ファイル名を決定
        if output_file is None:
            # Data_csvディレクトリに保存
            output_dir = Path("Data_csv")
            output_dir.mkdir(exist_ok=True)
            
            # 年月度からファイル名を生成
            main_df = extracted_data[0]['dataframe']
            year_month = extract_year_month_from_data(main_df)
            
            if year_month:
                output_file = output_dir / f"{year_month}.csv"
            else:
                # 年月度が特定できない場合は元のファイル名を使用
                output_file = output_dir / f"{Path(html_file).stem}_extracted.csv"
        
        # CSVファイルとして保存
        print(f"\nデータをCSVファイルに保存中: {output_file}")
        
        # 複数のテーブルがある場合は、最初のテーブルのみ保存（CSVは単一テーブル）
        if len(extracted_data) > 1:
            print(f"⚠️  CSVは単一テーブルのみ対応のため、最初のテーブルのみ保存します")
        
        # 最初のテーブルをCSVとして保存
        main_df = extracted_data[0]['dataframe']
        main_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSVファイル保存完了: {output_file}")
        print(f"  保存された行数: {len(main_df)}")
        print(f"  保存された列数: {len(main_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ データ抽出に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_process_data_files():
    """Data/配下の全データファイルを一括処理"""
    try:
        print("=== 一括データ処理開始 ===")
        
        # Dataディレクトリのパス（親ディレクトリから参照）
        data_dir = Path("Data/")
        if not data_dir.exists():
            print(f"✗ Dataディレクトリが見つかりません: {data_dir}")
            return False
        
        # 処理対象ファイルを検索
        data_files = list(data_dir.glob("*.xls")) + list(data_dir.glob("*.xlsx"))
        
        if not data_files:
            print("Dataディレクトリに処理可能なファイルが見つかりませんでした")
            return False
        
        print(f"処理対象ファイル数: {len(data_files)}")
        for i, file in enumerate(data_files, 1):
            print(f"  {i}. {file.name}")
        
        # 出力ディレクトリを作成（親ディレクトリから参照）
        output_dir = Path("Data_csv/")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n出力先: {output_dir.absolute()}")
        
        # 各ファイルを処理
        success_count = 0
        for i, data_file in enumerate(data_files, 1):
            print(f"\n{'='*60}")
            print(f"処理中 ({i}/{len(data_files)}): {data_file.name}")
            print(f"{'='*60}")
            
            try:
                if extract_data_from_html(data_file):
                    success_count += 1
                    print(f"✓ 処理成功: {data_file.name}")
                else:
                    print(f"✗ 処理失敗: {data_file.name}")
            except Exception as e:
                print(f"✗ 処理中にエラー: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"一括処理完了")
        print(f"成功: {success_count}/{len(data_files)}")
        print(f"出力先: {output_dir.absolute()}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"✗ 一括処理に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=== HTMLファイルデータ抽出ツール ===")
    
    print(f"\n処理方法を選択してください:")
    print(f"  1. 一括処理（Data/配下の全ファイル）")
    print(f"  2. 個別処理（現在のディレクトリのファイル）")
    print(f"  3. 特定ファイル処理")
    
    choice = input("選択 (1-3): ").strip()
    
    if choice == "1":
        # 一括処理
        batch_process_data_files()
        
    elif choice == "2":
        # 現在のディレクトリのファイルを処理
        current_dir = Path.cwd()
        html_files = list(current_dir.glob("*.html")) + list(current_dir.glob("*.htm"))
        xls_files = list(current_dir.glob("*.xls"))
        
        all_files = html_files + xls_files
        
        if all_files:
            print(f"\n処理可能なファイルを発見:")
            for i, file in enumerate(all_files, 1):
                print(f"  {i}. {file.name}")
            
            print(f"\nデータ抽出を開始します:")
            
            if len(all_files) == 1:
                extract_data_from_html(all_files[0])
            else:
                file_num = int(input(f"ファイル番号 (1-{len(all_files)}): ")) - 1
                if 0 <= file_num < len(all_files):
                    extract_data_from_html(all_files[file_num])
        else:
            print("処理可能なファイルが見つかりませんでした")
    
    elif choice == "3":
        # 特定ファイル処理
        file_path = input("ファイルパスを入力: ").strip()
        if os.path.exists(file_path):
            extract_data_from_html(file_path)
        else:
            print("ファイルが見つかりません")
    
    else:
        print("無効な選択です")

if __name__ == "__main__":
    main()
