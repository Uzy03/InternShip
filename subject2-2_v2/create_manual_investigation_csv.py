#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手動調査用CSVファイル作成スクリプト（完璧版）
激増・大幅増加・大幅減少のデータに原因コラムを追加
重複・総数行を最初から除外
"""

import pandas as pd
import numpy as np
import os

def create_manual_investigation_csv():
    """手動調査用のCSVファイルを作成（重複・総数行を除外）"""
    
    # 1. 分類済み人口変化データを読み込み
    print("データの読み込み中...")
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = "subject2/processed_data/classified_population_changes.csv"
    
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"データ読み込み完了: {df.shape}")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {data_path}")
        return
    
    # 2. データクリーニング
    print("データクリーニング中...")
    
    # 「総数」行を除外
    df_cleaned = df[df['町丁名'] != '総数']
    print(f"「総数」行除外後: {len(df_cleaned)}件")
    
    # 3. 調査対象の分類を抽出
    target_categories = ['激増', '大幅増加', '大幅減少', '激減']
    target_data = df_cleaned[df_cleaned['変化タイプ'].isin(target_categories)].copy()
    
    print(f"調査対象件数: {len(target_data)}件")
    
    # 4. 重複除去（町丁名・年度の組み合わせで）
    target_data = target_data.drop_duplicates(subset=['町丁名', '年度'], keep='first')
    print(f"重複除去後: {len(target_data)}件")
    
    # 5. 既知の原因データベース
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
    
    # 6. 原因コラムを追加
    target_data['原因'] = np.nan
    
    # 7. 既知の原因を設定
    for idx, row in target_data.iterrows():
        key = f"{row['町丁名']}_{row['年度']}"
        if key in known_causes:
            target_data.at[idx, '原因'] = known_causes[key]
            print(f"✓ 既知の原因設定: {row['町丁名']} ({row['年度']}年)")
    
    # 8. 原因が設定されている件数を確認
    known_count = target_data['原因'].notna().sum()
    unknown_count = target_data['原因'].isna().sum()
    
    print(f"\n原因設定状況:")
    print(f"  既知の原因: {known_count}件")
    print(f"  原因不明: {unknown_count}件")
    
    # 9. 手動調査用CSVとして保存
    output_path = "subject2-2_v2/manual_investigation_targets.csv"
    target_data.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n手動調査用CSVファイルを保存しました: {output_path}")
    
    # 10. サマリー表示
    print(f"\n=== 手動調査対象サマリー ===")
    print(f"総件数: {len(target_data)}件")
    
    # 変化タイプ別の件数
    type_counts = target_data['変化タイプ'].value_counts()
    print(f"\n変化タイプ別件数:")
    for change_type, count in type_counts.items():
        print(f"  {change_type}: {count}件")
    
    # 原因不明の件数（手動調査が必要）
    print(f"\n手動調査が必要な件数: {unknown_count}件")
    
    # 原因不明の詳細（上位10件）
    unknown_data = target_data[target_data['原因'].isna()].head(10)
    print(f"\n原因不明の上位10件:")
    for idx, row in unknown_data.iterrows():
        print(f"  {row['町丁名']} ({row['年度']}年) - {row['変化タイプ']} - 変化率: {row['変化率(%)']:.1f}%")
    
    # 11. データ品質チェック
    print(f"\n=== データ品質チェック ===")
    
    # 重複チェック
    duplicates = target_data.duplicated(subset=['町丁名', '年度'])
    duplicate_count = duplicates.sum()
    print(f"重複チェック: {'✓ なし' if duplicate_count == 0 else f'⚠️ {duplicate_count}件'}")
    
    # 「総数」行チェック
    total_rows = target_data[target_data['町丁名'] == '総数']
    total_count = len(total_rows)
    print(f"「総数」行チェック: {'✓ なし' if total_count == 0 else f'⚠️ {total_count}件'}")
    
    # 12. 完了メッセージ
    print(f"\n🎉 完璧なCSVファイルの作成が完了しました！")
    print(f"   これで個別の修正スクリプトは不要です")
    
    return target_data

if __name__ == "__main__":
    create_manual_investigation_csv()
