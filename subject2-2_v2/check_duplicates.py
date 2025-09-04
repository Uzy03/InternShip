#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重複データチェックスクリプト
CSVファイル内の重複を確認
"""

import pandas as pd

def check_duplicates():
    """重複データをチェック"""
    
    # CSVファイルを読み込み
    csv_path = "manual_investigation_targets.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    print(f"=== 重複チェック結果 ===")
    print(f"総件数: {len(df)}件")
    
    # 1. 完全重複のチェック
    duplicates = df.duplicated()
    if duplicates.any():
        print(f"完全重複: {duplicates.sum()}件")
        print(df[duplicates])
    else:
        print("完全重複: なし")
    
    # 2. 町丁名・年度の重複チェック
    town_year_duplicates = df.duplicated(subset=['町丁名', '年度'], keep=False)
    if town_year_duplicates.any():
        print(f"\n町丁名・年度の重複: {town_year_duplicates.sum()}件")
        duplicate_rows = df[town_year_duplicates].sort_values(['町丁名', '年度'])
        print(duplicate_rows[['町丁名', '年度', '変化率(%)', '変化タイプ', '原因']])
    else:
        print("町丁名・年度の重複: なし")
    
    # 3. 原因設定状況の詳細
    print(f"\n=== 原因設定状況の詳細 ===")
    
    # 既知の原因（9件）の確認
    known_causes = {
        '慶徳堀町_2013': '分譲マンション「エイルマンション慶徳イクシオ」竣工',
        '魚屋町１丁目_2020': '「（仮称）魚屋町・紺屋町マンション新築工事」完成',
        '秋津新町_2019': '「秋津新町マンション 新築工事」完成',
        '鍛冶屋町_2014': '大型分譲「グランドオーク唐人町通り」竣工',
        '河原町_2006': '「エバーライフ熊本中央」竣工',
        '千葉城町_2013': '「アンピール熊本城」完成',
        '本山３丁目_2009': '熊本駅東側の大規模マンション群完成',
        '春日３丁目_2013': '熊本駅周辺再開発の進行',
        '近見１丁目_2001': '集合住宅「プラーノウエスト」など竣工'
    }
    
    print("既知の原因（9件）の確認:")
    for key, cause in known_causes.items():
        town, year = key.split('_')
        year = int(year)
        
        # 該当する行を検索
        mask = (df['町丁名'] == town) & (df['年度'] == year)
        if mask.any():
            row = df[mask].iloc[0]
            if pd.isna(row['原因']):
                print(f"  ❌ {town} ({year}年): 原因が設定されていない")
            else:
                print(f"  ✅ {town} ({year}年): {row['原因']}")
        else:
            print(f"  ❌ {town} ({year}年): データが見つからない")
    
    # 4. 原因不明の件数確認
    unknown_count = df['原因'].isna().sum()
    known_count = df['原因'].notna().sum()
    
    print(f"\n原因設定状況:")
    print(f"  原因判明: {known_count}件")
    print(f"  原因不明: {unknown_count}件")
    
    # 5. 原因不明の詳細（上位20件）
    unknown_data = df[df['原因'].isna()].head(20)
    print(f"\n原因不明の上位20件:")
    for idx, row in unknown_data.iterrows():
        print(f"  {row['町丁名']} ({row['年度']}年) - {row['変化タイプ']} - 変化率: {row['変化率(%)']:.1f}%")
    
    return df

if __name__ == "__main__":
    check_duplicates()
