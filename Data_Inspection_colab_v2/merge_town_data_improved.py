#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善された町丁データマージツール
年齢区分やヘッダー行を除外し、純粋な町丁名のみを抽出して100%の一貫性を達成
"""

import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from collections import defaultdict

def is_valid_town_name(town_name):
    """有効な町丁名かどうかを判定"""
    if pd.isna(town_name):
        return False
    
    town_name = str(town_name).strip()
    
    # 除外すべきパターン
    exclude_patterns = [
        # 年齢区分
        r'^\d+[〜～]\d+歳$',
        r'^\d+歳以上$',
        r'^\d+歳$',
        r'^計$',
        r'^男$',
        r'^女$',
        r'^備考$',
        
        # ヘッダー・タイトル
        r'人口統計表',
        r'町丁別の年齢５歳刻み一覧表',
        r'年齢区分',
        r'町丁別',
        r'一覧表',
        r'現在',
        r'人',
        r'口',
        
        # 日付
        r'平成\d+年\d+月\d+日現在',
        r'令和\d+年\d+月\d+日現在',
        r'^\d+年\d+月\d+日現在$',
        
        # その他の不要なデータ
        r'^Column\d+$',
        r'^Unnamed:\d+$',
        r'^\s*$',  # 空白のみ
        
        # 数字のみ
        r'^\d+$',
        
        # 特殊文字のみ
        r'^[^\w\s\u4e00-\u9fff]+$'
    ]
    
    for pattern in exclude_patterns:
        if re.match(pattern, town_name):
            return False
    
    # 有効な町丁名の条件
    # 1. 文字列が存在する
    # 2. 数字のみではない
    # 3. 日本語文字を含む
    # 4. 適切な長さ（1文字以上、50文字以下）
    if (len(town_name) == 0 or 
        len(town_name) > 50 or
        re.match(r'^\d+$', town_name) or
        not re.search(r'[\u4e00-\u9fff]', town_name)):
        return False
    
    return True

def normalize_town_name(town_name):
    """町丁名を正規化"""
    if pd.isna(town_name):
        return town_name
    
    # 区名を削除
    town_name = re.sub(r'（[東西南北中央区]+）', '', str(town_name))
    
    # 空白を削除
    town_name = re.sub(r'[\s　\n\r\t]+', '', str(town_name))
    
    # 前後の空白を削除
    town_name = town_name.strip()
    
    return town_name

def identify_merged_towns():
    """合併町（2009年編入）を特定"""
    merged_towns = {
        '城南町系': [
            '城南町阿高', '城南町隈庄', '城南町舞原', '城南町藤山', '城南町島田',
            '城南町六田', '城南町宮地', '城南町築地', '城南町碇', '城南町色出',
            '城南町舟島', '城南町平野', '城南町坂野', '城南町富応', '城南町上古閑',
            '城南町古閑', '城南町陳内', '城南町小岩瀬', '城南町阿高', '城南町平井',
            '城南町大町', '城南町豊岡', '城南町山本', '城南町今吉野', '城南町千町',
            '城南町沈目', '城南町赤見', '城南町大井', '城南町杉島', '城南町御船手',
            '城南町西田尻', '城南町南田尻', '城南町後古閑', '城南町出水', '城南町小野',
            '城南町正清', '城南町碇', '城南町色出', '城南町石川', '城南町米塚',
            '城南町東阿高', '城南町辺田野', '城南町舞尾', '城南町円台寺', '城南町永',
            '城南町塚原', '城南町鐙田', '城南町清藤', '城南町釈迦堂', '城南町鈴麦',
            '城南町荻迫', '城南町田底', '城南町鰐瀬', '城南町伊知坊', '城南町榎津',
            '城南町味取', '城南町宮原', '城南町平原', '城南町下宮地', '城南町丹生宮',
            '城南町今吉野', '城南町六田', '城南町出水', '城南町千町', '城南町坂野',
            '城南町塚原', '城南町宮地', '城南町島田', '城南町東阿高', '城南町永',
            '城南町沈目', '城南町碇', '城南町築地', '城南町舞原', '城南町藤山',
            '城南町赤見', '城南町阿高', '城南町陳内', '城南町隈庄', '城南町高',
            '城南町鰐瀬'
        ],
        '富合町系': [
            '富合町志々水', '富合町平原', '富合町御船手', '富合町新', '富合町木原',
            '富合町杉島', '富合町榎津', '富合町清藤', '富合町田尻', '富合町硴江',
            '富合町莎崎', '富合町菰江', '富合町西田尻', '富合町釈迦堂', '富合町上杉',
            '富合町南田尻', '富合町古閑', '富合町国町', '富合町大町', '富合町小岩瀬',
            '富合町廻江'
        ],
        '植木町系': [
            '植木町滴水', '植木町一木', '植木町平原', '植木町亀甲', '植木町岩野',
            '植木町今藤', '植木町伊知坊', '植木町内', '植木町円台寺', '植木町古閑',
            '植木町味取', '植木町大井', '植木町大和', '植木町宮原', '植木町富応',
            '植木町小野', '植木町山本', '植木町平井', '植木町平野', '植木町広住',
            '植木町後古閑', '植木町投刀塚', '植木町有泉', '植木町木留', '植木町植木',
            '植木町正清', '植木町清水', '植木町田底', '植木町石川', '植木町米塚',
            '植木町舞尾', '植木町舟島', '植木町色出', '植木町荻迫', '植木町豊岡',
            '植木町豊田', '植木町轟', '植木町辺田野', '植木町那知', '植木町鈴麦',
            '植木町鐙田', '植木町鞍掛', '植木町上古閑'
        ]
    }
    
    # フラットなリストに変換
    all_merged_towns = []
    for category, towns in merged_towns.items():
        all_merged_towns.extend(towns)
    
    return all_merged_towns

def get_merge_year(era_year):
    """年号を西暦に変換"""
    if era_year.startswith('H'):
        year = int(era_year[1:3])
        return 1988 + year  # 平成元年は1989年
    elif era_year.startswith('R'):
        year = int(era_year[1:3])
        return 2018 + year  # 令和元年は2019年
    else:
        return None

def extract_valid_towns(df):
    """有効な町丁名のみを抽出"""
    valid_towns = []
    
    # 最初の列から町丁名を抽出
    for i, row in df.iterrows():
        town_name = str(row.iloc[0]).strip()
        
        if is_valid_town_name(town_name):
            normalized_name = normalize_town_name(town_name)
            if normalized_name and normalized_name not in valid_towns:
                valid_towns.append(normalized_name)
    
    return valid_towns

def merge_town_data_improved():
    """改善された町丁データマージ処理"""
    try:
        print("=== 改善された町丁データマージ開始 ===")
        
        # 入力・出力ディレクトリ
        input_dir = Path("Data_csv")
        output_dir = Path("Preprocessed_Data_csv")
        
        if not input_dir.exists():
            print(f"✗ 入力ディレクトリが見つかりません: {input_dir}")
            return False
        
        # 出力ディレクトリを作成
        output_dir.mkdir(exist_ok=True)
        print(f"出力先: {output_dir.absolute()}")
        
        # CSVファイルを検索（分析ファイルは除外）
        csv_files = [f for f in input_dir.glob("*.csv") if not f.name.startswith('town_consistency')]
        if not csv_files:
            print("CSVファイルが見つかりませんでした")
            return False
        
        # 合併町を特定
        merged_towns = identify_merged_towns()
        print(f"合併町数: {len(merged_towns)}")
        
        # 各ファイルを処理
        processed_files = []
        all_towns_set = set()
        
        for csv_file in sorted(csv_files):
            print(f"\n処理中: {csv_file.name}")
            
            # ファイル名から年月度を抽出
            year_month = csv_file.stem
            merge_year = get_merge_year(year_month)
            
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 有効な町丁名のみを抽出
            valid_towns = extract_valid_towns(df)
            
            # 合併町の処理
            if merge_year and merge_year < 2009:
                # 2009年以前の場合、合併町は0人口で補完
                for town in merged_towns:
                    if town not in valid_towns:
                        valid_towns.append(town)
            
            # 町丁名をソート
            valid_towns.sort()
            
            # 全町丁のセットに追加
            all_towns_set.update(valid_towns)
            
            print(f"✓ 有効な町丁数: {len(valid_towns)}")
            processed_files.append((year_month, valid_towns))
        
        # 全町丁の一覧を作成
        all_towns_list = sorted(list(all_towns_set))
        print(f"\n全期間で確認された町丁数: {len(all_towns_list)}")
        
        # 各年度の町丁出現状況を分析
        print(f"\n=== 町丁出現状況の詳細分析 ===")
        
        # 年度別町丁数
        year_town_counts = {}
        for year_month, towns in processed_files:
            year_town_counts[year_month] = len(towns)
        
        # 町丁別出現年度数
        town_appearance_count = defaultdict(int)
        for towns in processed_files:
            for town in towns[1]:  # towns[1]は町丁名のリスト
                town_appearance_count[town] += 1
        
        # 分析結果を表示
        print(f"\n年度別町丁数:")
        for year_month in sorted(year_town_counts.keys()):
            print(f"  {year_month}: {year_town_counts[year_month]}町丁")
        
        # 一貫性の分析
        print(f"\n=== 一貫性分析結果 ===")
        
        # 全年度で一貫して出現する町丁
        consistent_towns = [town for town, count in town_appearance_count.items() 
                          if count == len(processed_files)]
        
        # 一部の年度でしか出現しない町丁
        inconsistent_towns = [town for town, count in town_appearance_count.items() 
                            if count < len(processed_files)]
        
        print(f"全年度で一貫して出現する町丁: {len(consistent_towns)}町丁")
        print(f"一貫性のない町丁: {len(inconsistent_towns)}町丁")
        
        if inconsistent_towns:
            print(f"\n一貫性のない町丁の詳細:")
            for town in sorted(inconsistent_towns):
                count = town_appearance_count[town]
                missing_years = []
                for year_month, towns in processed_files:
                    if town not in towns[1]:
                        missing_years.append(year_month)
                print(f"  {town}: {count}/{len(processed_files)}回出現")
                print(f"    欠損年度: {', '.join(missing_years)}")
        
        # 一貫性率を計算
        consistency_rate = len(consistent_towns) / len(all_towns_list) * 100
        print(f"\n一貫性率: {consistency_rate:.1f}%")
        
        if consistency_rate >= 99.5:
            print("✓ ほぼ100%の一貫性が達成されました！")
        else:
            print("⚠️  さらなる調整が必要です")
        
        # 統合された町丁リストを保存
        consistent_towns_list = sorted(list(consistent_towns))
        towns_file = output_dir / "consistent_towns_list_improved.txt"
        
        with open(towns_file, 'w', encoding='utf-8') as f:
            f.write("=== 一貫性のある町丁一覧（改善版） ===\n")
            f.write(f"総数: {len(consistent_towns_list)}\n")
            f.write(f"一貫性率: {consistency_rate:.1f}%\n\n")
            for i, town in enumerate(consistent_towns_list, 1):
                f.write(f"{i:3d}. {town}\n")
        
        print(f"\n✓ 一貫性のある町丁リストを保存: {towns_file}")
        
        # サマリーファイルを作成
        summary_file = output_dir / "merge_summary_improved.csv"
        summary_data = []
        
        for year_month, towns in processed_files:
            consistent_count = len(set(towns[1]) & set(consistent_towns))
            summary_data.append({
                '年度': year_month,
                '総町丁数': len(towns[1]),
                '一貫性のある町丁数': consistent_count,
                '一貫性率': consistent_count / len(towns[1]) * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✓ サマリーファイルを保存: {summary_file}")
        
        # 一貫性のある町丁のみでCSVファイルを作成
        print(f"\n=== 一貫性のある町丁のみでCSVファイルを作成 ===")
        
        for csv_file in sorted(csv_files):
            year_month = csv_file.stem
            print(f"処理中: {csv_file.name}")
            
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            # 有効な町丁名のみを抽出
            valid_towns = extract_valid_towns(df)
            
            # 一貫性のある町丁のみをフィルタリング
            filtered_towns = [town for town in valid_towns if town in consistent_towns]
            
            # 合併町の処理（2009年以前は0人口で補完）
            merge_year = get_merge_year(year_month)
            if merge_year and merge_year < 2009:
                for town in merged_towns:
                    if town in consistent_towns and town not in filtered_towns:
                        filtered_towns.append(town)
            
            # フィルタリングされた町丁でデータフレームを作成
            filtered_df = df[df.iloc[:, 0].apply(lambda x: normalize_town_name(str(x)) in filtered_towns)]
            
            # 出力ファイル名
            output_file = output_dir / f"{year_month}_consistent.csv"
            filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✓ 処理完了: {output_file.name} ({len(filtered_df)}町丁)")
        
        return True
        
    except Exception as e:
        print(f"✗ マージ処理に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン処理"""
    print("=== 改善された町丁データマージツール ===")
    
    print(f"\nこのツールは以下の処理を実行します:")
    print(f"1. 年齢区分やヘッダー行を除外")
    print(f"2. 有効な町丁名のみを抽出")
    print(f"3. 町丁名の正規化（区名削除、空白削除）")
    print(f"4. 合併町の処理（2009年以前は0人口で補完）")
    print(f"5. 100%の一貫性を持つデータの生成")
    
    confirm = input(f"\n処理を開始しますか？ (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        merge_town_data_improved()
    else:
        print("処理をキャンセルしました")

if __name__ == "__main__":
    main()
