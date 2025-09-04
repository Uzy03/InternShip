#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手動調査結果更新スクリプト
causes.txtから調査結果を読み込んでCSVファイルに反映
"""

import pandas as pd
import re
import os

def parse_investigation_result(input_text):
    """調査結果のテキストをパースして辞書のリストに変換"""
    results = []
    
    # 各行を処理
    lines = input_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # パターン: 町丁名,年度,変化率,変化タイプ,変化の大きさ : 原因
        # 例: 徳堀町,2013,775.0,激増,775.0 : 分譲マンション「エイルマンション慶徳イクシオ」竣工
        # 複数原因: 徳堀町,2013,775.0,激増,775.0 : 原因1 & 原因2 & 原因3
        pattern = r'([^,]+),(\d{4}),([^,]+),([^,]+),([^,]+)\s*:\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            town, year, change_rate, change_type, change_magnitude, cause = match.groups()
            
            # 複数原因を & で分割して正規化
            causes = [c.strip() for c in cause.split('&')]
            normalized_cause = ' & '.join(causes)
            
            results.append({
                '町丁名': town.strip(),
                '年度': int(year),
                '変化率(%)': float(change_rate),
                '変化タイプ': change_type.strip(),
                '変化の大きさ': float(change_magnitude),
                '原因': normalized_cause
            })
        else:
            print(f"⚠️  パースできない行: {line}")
    
    return results

def update_csv_with_results(new_results, csv_path=None):
    """CSVファイルを新しい調査結果で更新（複数行の原因を自動結合）"""
    
    # CSVパスが指定されていない場合、絶対パスを使用
    if csv_path is None:
        csv_path = "subject2-2_v2/manual_investigation_targets.csv"
    
    # 1. 既存のCSVファイルを読み込み
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"既存のCSVファイルを読み込み: {len(df)}件")
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        return
    
    # 2. 複数行の原因を結合
    print("複数行の原因を結合中...")
    combined_results = {}
    
    for result in new_results:
        key = (result['町丁名'], result['年度'])
        if key in combined_results:
            # 既存の原因に追加（&で結合）
            existing_cause = combined_results[key]['原因']
            new_cause = result['原因']
            combined_results[key]['原因'] = f"{existing_cause} & {new_cause}"
            print(f"✓ 原因結合: {result['町丁名']} ({result['年度']}年) - {existing_cause} + {new_cause}")
        else:
            # 新規追加
            combined_results[key] = result.copy()
    
    print(f"結合後の件数: {len(combined_results)}件")
    
    # 3. 更新件数をカウント
    updated_count = 0
    
    # 4. 各調査結果を処理
    for key, result in combined_results.items():
        # 町丁名と年度でマッチング
        mask = (df['町丁名'] == result['町丁名']) & (df['年度'] == result['年度'])
        
        if mask.any():
            # 既存の行を更新
            df.loc[mask, '原因'] = result['原因']
            updated_count += 1
            print(f"✓ 更新: {result['町丁名']} ({result['年度']}年) - {result['原因']}")
        else:
            # 新規追加は禁止 - 警告を表示
            print(f"⚠️  新規追加は禁止: {result['町丁名']} ({result['年度']}年) - 元のCSVファイルに存在しません")
            print(f"    この事象は元の調査対象（115件）に含まれていません")
            print(f"    元のCSVファイルに存在する事象のみ更新可能です")
    
    # 5. 更新されたCSVファイルを保存
    output_path = "subject2-3/manual_investigation_targets.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n=== 更新完了 ===")
    print(f"更新件数: {updated_count}件")
    print(f"新規追加件数: 0件（新規追加は禁止）")
    print(f"総件数: {len(df)}件（変化なし）")
    
    # 6. 原因設定状況の確認
    known_count = df['原因'].notna().sum()
    unknown_count = df['原因'].isna().sum()
    
    print(f"\n原因設定状況:")
    print(f"  原因判明: {known_count}件")
    print(f"  原因不明: {unknown_count}件")
    
    return df

def main():
    """メイン実行関数"""
    print("手動調査結果更新ツール")
    print("=" * 30)
    
    # causes.txtファイルのパス
    causes_file = "subject2-3/causes.txt"
    
    # 1. causes.txtファイルを読み込み
    try:
        with open(causes_file, 'r', encoding='utf-8') as f:
            input_text = f.read()
        print(f"✓ causes.txtファイルを読み込みました: {len(input_text.splitlines())}行")
    except FileNotFoundError:
        print(f"エラー: causes.txtファイルが見つかりません: {causes_file}")
        return
    except Exception as e:
        print(f"エラー: ファイル読み込み中にエラーが発生しました: {e}")
        return
    
    # 2. パースして結果を取得
    results = parse_investigation_result(input_text)
    
    if not results:
        print("パースできる結果がありません")
        return
    
    # 3. 結果の確認
    print(f"\n=== パース結果 ===")
    print(f"件数: {len(results)}件")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['町丁名']} ({result['年度']}年): {result['原因']}")
    
    # 4. CSVファイルを更新
    print(f"\nCSVファイルを更新中...")
    update_csv_with_results(results)
    print("更新が完了しました！")

if __name__ == "__main__":
    main()
