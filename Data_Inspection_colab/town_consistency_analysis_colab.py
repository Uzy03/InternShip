#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
町丁一貫性分析スクリプト（Google Colab版）
Data_csv配下のCSVファイルから町丁一貫性を分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv_file(file_path):
    """CSVファイルから町丁名を抽出（修正版）"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        town_names = []
        
        # 最初の列から町丁名を抽出
        for i, row in df.iterrows():
            town_name = str(row.iloc[0]) if len(row) > 0 else ""
            town_name = town_name.strip()
            
            # 町丁名の判定（調整版）
            if (town_name and 
                len(town_name) > 1 and
                # 数字のみは除外
                not town_name.isdigit() and
                # 年齢区分は除外
                not re.match(r'^\d+[〜～]\d+歳$', town_name) and
                not re.match(r'^\d+歳以上?$', town_name) and
                not re.match(r'^\d+歳$', town_name) and
                # ヘッダー・タイトルは除外
                not any(keyword in town_name for keyword in [
                    '年齢', '区分', '計', '男', '女', '備考', '人口統計表', 
                    '町丁別', '一覧表', '現在', '人', '口', '総数', '年齢区分'
                ]) and
                # 日付は除外
                not re.match(r'^平成\d+年\d+月\d+日', town_name) and
                not re.match(r'^令和\d+年\d+月\d+日', town_name) and
                # 特殊文字のみは除外
                not re.match(r'^[^\w\s\u4e00-\u9fff]+$', town_name) and
                # 空白のみは除外
                not re.match(r'^\s*$', town_name) and
                # 日本語文字が含まれている
                re.search(r'[\u4e00-\u9fff]', town_name) and
                # ファイル名のプレフィックスは除外
                not town_name.startswith('Column') and
                not town_name.startswith('Unnamed') and
                # 町丁名の特徴を確認
                # 1. 「町」「丁目」で終わる、または「町」を含む
                (town_name.endswith('町') or 
                 town_name.endswith('丁目') or 
                 '町' in town_name or
                 # 2. 特定の地域名を含む（熊本市の特徴的な地名）
                 any(area in town_name for area in [
                     '熊本', '中央', '東', '西', '南', '北', '植木', '城南', '富合'
                 ]))):
                
                town_names.append(town_name)
        
        return town_names
    except Exception as e:
        print(f"エラー: {file_path} - {e}")
        return []

def analyze_town_consistency():
    """町丁一貫性を分析"""
    print("=== ステップ2: 町丁一貫性分析開始 ===")
    
    # 必要なディレクトリを作成
    output_dir = Path("Preprocessed_Data_csv")
    output_dir.mkdir(exist_ok=True)
    print("✓ Preprocessed_Data_csvディレクトリを作成/確認しました")
    
    csv_dir = Path("Data_csv")
    
    # CSVファイルを取得
    csv_files = list(csv_dir.glob("*.csv"))
    csv_files.sort()
    
    if not csv_files:
        print("❌ CSVファイルが見つかりません")
        print("先にmain_preprocessing_colab.pyを実行してください")
        return False
    
    print(f"✓ 分析対象ファイル数: {len(csv_files)}")
    
    # 各ファイルの町丁名を取得
    all_towns = set()
    file_towns = {}
    
    for csv_file in csv_files:
        print(f"読み込み中: {csv_file.name}")
        towns = load_csv_file(csv_file)
        file_towns[csv_file.name] = set(towns)
        all_towns.update(towns)
    
    print(f"\n✓ 全町丁数: {len(all_towns)}")
    
    # 各町丁の出現回数をカウント
    town_consistency = {}
    total_files = len(csv_files)
    
    for town in all_towns:
        count = sum(1 for towns in file_towns.values() if town in towns)
        town_consistency[town] = count
    
    # 一貫性のある町丁（全ファイルに出現）を特定
    consistent_towns = [town for town, count in town_consistency.items() if count == total_files]
    inconsistent_towns = [town for town, count in town_consistency.items() if count < total_files]
    
    # 一貫性率を計算
    consistency_rate = (len(consistent_towns) / len(all_towns)) * 100
    
    print(f"\n=== 町丁一貫性分析結果 ===")
    print(f"全町丁数: {len(all_towns)}")
    print(f"一貫性のある町丁数: {len(consistent_towns)}")
    print(f"一貫性のない町丁数: {len(inconsistent_towns)}")
    print(f"町丁一貫性率: {consistency_rate:.1f}%")
    
    # 一貫性のない町丁の詳細
    if inconsistent_towns:
        print(f"\n=== 一貫性のない町丁（出現回数順） ===")
        sorted_inconsistent = sorted(inconsistent_towns, key=lambda x: town_consistency[x], reverse=True)
        
        for town in sorted_inconsistent[:20]:  # 上位20件を表示
            count = town_consistency[town]
            missing_files = [fname for fname, towns in file_towns.items() if town not in towns]
            print(f"{town}: {count}/{total_files}回出現")
            if len(missing_files) <= 5:
                print(f"  欠損年度: {', '.join(missing_files)}")
            else:
                print(f"  欠損年度: {len(missing_files)}ファイル")
    
    # 結果を保存
    output_dir = Path("Preprocessed_Data_csv")
    
    # 一貫性のある町丁リスト
    consistent_file = output_dir / "consistent_towns_list.txt"
    with open(consistent_file, 'w', encoding='utf-8') as f:
        f.write("=== 一貫性のある町丁 ===\n")
        f.write(f"一貫性率: {consistency_rate:.1f}%\n")
        f.write(f"一貫性のある町丁数: {len(consistent_towns)}\n")
        f.write(f"全町丁数: {len(all_towns)}\n\n")
        for town in sorted(consistent_towns):
            f.write(f"{town}\n")
    
    # 詳細分析結果
    analysis_file = output_dir / "town_consistency_analysis.csv"
    analysis_data = []
    for town, count in town_consistency.items():
        missing_files = [fname for fname, towns in file_towns.items() if town not in towns]
        analysis_data.append({
            '町丁名': town,
            '出現回数': count,
            '全ファイル数': total_files,
            '一貫性': count == total_files,
            '欠損ファイル数': len(missing_files),
            '欠損ファイル': ', '.join(missing_files) if len(missing_files) <= 3 else f"{len(missing_files)}ファイル"
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(analysis_file, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 分析結果保存完了 ===")
    print(f"✓ 一貫性町丁リスト: {consistent_file}")
    print(f"✓ 詳細分析結果: {analysis_file}")
    
    return True

def create_visualization():
    """町丁一貫性の可視化"""
    print("\n=== 町丁一貫性の可視化 ===")
    
    try:
        # 日本語フォント設定（Colab対応）
        import matplotlib.font_manager as fm
        
        # 日本語フォントの設定
        try:
            # Colabで利用可能な日本語フォントを探す
            font_list = [f.name for f in fm.fontManager.ttflist]
            japanese_fonts = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic']
            
            available_font = None
            for font in japanese_fonts:
                if font in font_list:
                    available_font = font
                    break
            
            if available_font:
                plt.rcParams['font.family'] = available_font
                print(f"✓ 日本語フォント設定: {available_font}")
            else:
                # フォントが見つからない場合は、日本語フォントをインストール
                try:
                    print("日本語フォントをインストール中...")
                    import subprocess
                    subprocess.run(['apt-get', 'update'], capture_output=True)
                    subprocess.run(['apt-get', 'install', '-y', 'fonts-ipafont-gothic'], capture_output=True)
                    
                    # フォントキャッシュをクリア
                    fm._rebuild()
                    
                    # 再度日本語フォントを探す
                    font_list = [f.name for f in fm.fontManager.ttflist]
                    for font in japanese_fonts:
                        if font in font_list:
                            plt.rcParams['font.family'] = font
                            print(f"✓ インストール後日本語フォント設定: {font}")
                            break
                    else:
                        plt.rcParams['font.family'] = 'sans-serif'
                        print("⚠️  日本語フォントのインストールに失敗しました。デフォルトフォントを使用します。")
                        
                except Exception as e:
                    print(f"⚠️  フォントインストール中にエラー: {e}")
                    plt.rcParams['font.family'] = 'sans-serif'
        
        except Exception as e:
            print(f"⚠️  フォント設定中にエラー: {e}")
            plt.rcParams['font.family'] = 'sans-serif'
        
        # データ読み込み
        analysis_file = Path("Preprocessed_Data_csv/town_consistency_analysis.csv")
        if analysis_file.exists():
            df = pd.read_csv(analysis_file, encoding='utf-8-sig')
            
            # 一貫性の分布
            plt.figure(figsize=(12, 8))
            
            # サブプロット1: 一貫性の分布
            plt.subplot(2, 2, 1)
            consistency_counts = df['一貫性'].value_counts()
            plt.pie(consistency_counts.values, labels=['一貫性なし', '一貫性あり'], autopct='%1.1f%%')
            plt.title('町丁一貫性の分布')
            
            # サブプロット2: 出現回数の分布
            plt.subplot(2, 2, 2)
            plt.hist(df['出現回数'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('出現回数')
            plt.ylabel('町丁数')
            plt.title('町丁出現回数の分布')
            
            # サブプロット3: 欠損ファイル数の分布
            plt.subplot(2, 2, 3)
            plt.hist(df['欠損ファイル数'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('欠損ファイル数')
            plt.ylabel('町丁数')
            plt.title('町丁欠損ファイル数の分布')
            
            # サブプロット4: 一貫性率の推移
            plt.subplot(2, 2, 4)
            sorted_df = df.sort_values('出現回数', ascending=False)
            cumulative_consistency = []
            for i in range(1, len(sorted_df) + 1):
                consistent_count = sorted_df.head(i)['一貫性'].sum()
                cumulative_consistency.append((consistent_count / i) * 100)
            
            plt.plot(range(1, len(sorted_df) + 1), cumulative_consistency)
            plt.xlabel('町丁数（出現回数順）')
            plt.ylabel('累積一貫性率 (%)')
            plt.title('町丁数による累積一貫性率の推移')
            
            plt.tight_layout()
            
            # 保存
            output_dir = Path("Preprocessed_Data_csv")
            plot_file = output_dir / "town_consistency_visualization.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ 可視化完了: {plot_file}")
            
            plt.show()
        else:
            print("⚠️  分析結果ファイルが見つかりません")
            
    except Exception as e:
        print(f"⚠️  可視化中にエラー: {e}")

def main():
    """メイン処理"""
    print("=== 町丁一貫性分析開始（Google Colab版） ===")
    
    # 町丁一貫性分析
    if not analyze_town_consistency():
        return
    
    # 可視化
    create_visualization()
    
    print("\n=== ステップ2完了 ===")
    print("✓ 町丁一貫性分析が完了しました")
    print("✓ 次のステップ: 町丁データの統合と前処理")
    print("\n次のスクリプトを実行してください:")
    print("python merge_town_data_colab.py")

if __name__ == "__main__":
    main()
