#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文字化けしたExcelファイルを修正するスクリプト
熊本市統計データの文字化け問題を解決
"""

import pandas as pd
import os
from pathlib import Path
import chardet

def detect_encoding(file_path):
    """ファイルのエンコーディングを検出"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'], result['confidence']
    except Exception as e:
        print(f"エンコーディング検出に失敗: {e}")
        return None, 0

def fix_excel_encoding(input_file, output_file=None):
    """Excelファイルの文字化けを修正"""
    try:
        print(f"ファイル処理中: {input_file}")
        
        # 出力ファイル名を決定
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
        
        # 複数のエンコーディングで試行
        encodings_to_try = [
            'shift_jis',      # 元のエンコーディング
            'cp932',          # Windows版Shift_JIS
            'euc_jp',         # EUC-JP
            'iso2022_jp',     # ISO-2022-JP
            'utf-8',          # UTF-8
            'utf-8-sig',      # UTF-8 with BOM
        ]
        
        data = None
        successful_encoding = None
        
        for encoding in encodings_to_try:
            try:
                print(f"  エンコーディング '{encoding}' を試行中...")
                
                # Excelファイルを読み込み
                if input_file.endswith('.xls'):
                    # 古いExcel形式
                    data = pd.read_excel(input_file, encoding=encoding, engine='xlrd')
                else:
                    # 新しいExcel形式
                    data = pd.read_excel(input_file, encoding=encoding, engine='openpyxl')
                
                print(f"  ✓ エンコーディング '{encoding}' で読み込み成功")
                successful_encoding = encoding
                break
                
            except Exception as e:
                print(f"  ✗ エンコーディング '{encoding}' で失敗: {e}")
                continue
        
        if data is None:
            print("✗ すべてのエンコーディングで読み込みに失敗しました")
            return False
        
        # データの確認
        print(f"\nデータ情報:")
        print(f"  行数: {len(data)}")
        print(f"  列数: {len(data.columns)}")
        print(f"  列名: {list(data.columns)}")
        
        # 最初の数行を表示
        print(f"\n最初の5行:")
        print(data.head())
        
        # 修正されたファイルを保存
        print(f"\n修正されたファイルを保存中: {output_file}")
        
        if output_file.endswith('.xls'):
            # 古いExcel形式で保存
            data.to_excel(output_file, index=False, engine='xlwt')
        else:
            # 新しいExcel形式で保存
            data.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"✓ ファイル保存完了: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ ファイル処理に失敗: {e}")
        return False

def batch_fix_encoding(input_dir, output_dir=None):
    """ディレクトリ内のすべてのExcelファイルを一括修正"""
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / "fixed"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"一括処理開始:")
    print(f"  入力ディレクトリ: {input_path}")
    print(f"  出力ディレクトリ: {output_path}")
    
    # Excelファイルを検索
    excel_files = list(input_path.glob("*.xls")) + list(input_path.glob("*.xlsx"))
    
    if not excel_files:
        print("Excelファイルが見つかりませんでした")
        return
    
    print(f"\n処理対象ファイル数: {len(excel_files)}")
    
    success_count = 0
    for excel_file in excel_files:
        print(f"\n{'='*50}")
        output_file = output_path / f"{excel_file.stem}_fixed{excel_file.suffix}"
        
        if fix_excel_encoding(excel_file, output_file):
            success_count += 1
        
        print(f"{'='*50}")
    
    print(f"\n処理完了:")
    print(f"  成功: {success_count}/{len(excel_files)}")
    print(f"  出力先: {output_path}")

def main():
    """メイン処理"""
    print("=== Excelファイル文字化け修正ツール ===")
    
    # 現在のディレクトリのExcelファイルを確認
    current_dir = Path.cwd()
    excel_files = list(current_dir.glob("*.xls")) + list(current_dir.glob("*.xlsx"))
    
    if excel_files:
        print(f"\n現在のディレクトリでExcelファイルを発見:")
        for i, file in enumerate(excel_files, 1):
            print(f"  {i}. {file.name}")
        
        print(f"\n処理方法を選択してください:")
        print(f"  1. 個別ファイル修正")
        print(f"  2. 一括修正")
        print(f"  3. 特定ファイル修正")
        
        choice = input("選択 (1-3): ").strip()
        
        if choice == "1":
            # 個別ファイル修正
            if len(excel_files) == 1:
                fix_excel_encoding(excel_files[0])
            else:
                file_num = int(input(f"ファイル番号 (1-{len(excel_files)}): ")) - 1
                if 0 <= file_num < len(excel_files):
                    fix_excel_encoding(excel_files[file_num])
        
        elif choice == "2":
            # 一括修正
            batch_fix_encoding(current_dir)
        
        elif choice == "3":
            # 特定ファイル修正
            file_path = input("ファイルパスを入力: ").strip()
            if os.path.exists(file_path):
                fix_excel_encoding(file_path)
            else:
                print("ファイルが見つかりません")
    
    else:
        print("Excelファイルが見つかりませんでした")
        print("ファイルパスを直接入力してください:")
        file_path = input("ファイルパス: ").strip()
        if os.path.exists(file_path):
            fix_excel_encoding(file_path)
        else:
            print("ファイルが見つかりません")

if __name__ == "__main__":
    main()
