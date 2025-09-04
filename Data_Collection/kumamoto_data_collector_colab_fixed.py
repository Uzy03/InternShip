#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熊本市統計サイトから人口統計データを収集するスクリプト（Google Colab対応・修正版）
条件：
- 年月度：平成10年1月〜令和7年12月
- 統計：年齢5歳刻み
- 区分：町丁別
- 地区：すべて
"""

import time
import pandas as pd
import numpy as np
import os
import subprocess
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

class KumamotoDataCollectorColabFixed:
    def __init__(self):
        self.base_url = "https://tokei.city.kumamoto.jp/content/ASP/Jinkou/default.asp"
        self.driver = None
        self.setup_driver()
        
    def install_chromium(self):
        """Google Colab環境でChromiumをインストール"""
        try:
            print("Google Colab環境でChromiumをインストール中...")
            
            # システムの更新
            subprocess.run(["apt-get", "update"], check=True, capture_output=True)
            
            # Chromiumのインストール
            subprocess.run([
                "apt-get", "install", "-y", "chromium-browser"
            ], check=True, capture_output=True)
            
            print("✓ Chromiumのインストールが完了しました")
            return True
            
        except Exception as e:
            print(f"✗ Chromiumのインストールに失敗しました: {e}")
            return False
        
    def setup_driver(self):
        """Google Colab用Chromiumドライバーの設定"""
        try:
            print("Chromiumドライバーの設定を開始...")
            
            # Chromiumがインストールされているか確認
            chromium_path = "/usr/bin/chromium-browser"
            if not os.path.exists(chromium_path):
                print("Chromiumが見つかりません。インストールを試行します...")
                if not self.install_chromium():
                    raise Exception("Chromiumのインストールに失敗しました")
            
            chrome_options = Options()
            
            # Colab環境用の設定
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--headless")  # Colabではヘッドレスモードが推奨
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--remote-debugging-port=9222")
            
            # Chromiumのパスを明示的に設定
            chrome_options.binary_location = chromium_path
            
            # Colab環境の検出
            if 'COLAB_GPU' in os.environ:
                print("Google Colab環境を検出しました")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--no-sandbox")
            
            print("Chromeドライバーをインストール中...")
            driver_path = ChromeDriverManager().install()
            print(f"ドライバーパス: {driver_path}")
            
            service = Service(driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✓ Chromiumドライバーの初期化が完了しました")
            
        except Exception as e:
            print(f"✗ Chromiumドライバーの設定に失敗しました: {e}")
            raise
    
    def convert_japanese_date(self, japanese_date):
        """和暦を西暦に変換"""
        date_mapping = {
            '平成': 'Heisei',
            '令和': 'Reiwa'
        }
        
        for era, english in date_mapping.items():
            if era in japanese_date:
                year_part = japanese_date.replace(era, '').split('年')[0]
                if era == '平成':
                    western_year = int(year_part) + 1988
                elif era == '令和':
                    western_year = int(year_part) + 2018
                return western_year
        return None
    
    def get_date_range(self):
        """指定された期間の日付範囲を生成"""
        start_date = datetime(1998, 1, 1)  # 平成10年1月
        end_date = datetime(2035, 12, 31)  # 令和7年12月
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=32)  # 次の月へ
            current_date = current_date.replace(day=1)
        
        return dates
    
    def navigate_to_site(self):
        """統計サイトにアクセス"""
        try:
            self.driver.get(self.base_url)
            print("統計サイトにアクセスしました")
            
            # ページの読み込みを待機
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            return True
        except Exception as e:
            print(f"サイトへのアクセスに失敗しました: {e}")
            return False
    
    def inspect_site_structure(self):
        """サイトの構造を調査"""
        try:
            print("=== サイト構造の調査 ===")
            
            # ページのHTMLを取得
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # フォーム要素を調査
            forms = soup.find_all('form')
            print(f"フォーム数: {len(forms)}")
            
            # テーブル要素を調査
            tables = soup.find_all('table')
            print(f"テーブル数: {len(tables)}")
            
            # 入力要素を調査
            inputs = soup.find_all(['input', 'select', 'textarea'])
            print(f"入力要素数: {len(inputs)}")
            
            # ボタン要素を調査
            buttons = soup.find_all(['button', 'input[type="button"]', 'input[type="submit"]'])
            print(f"ボタン要素数: {len(buttons)}")
            
            # 詳細な調査結果を表示
            print("\n=== 詳細な要素情報 ===")
            for i, form in enumerate(forms[:3]):  # 最初の3つのフォームのみ表示
                print(f"\nフォーム {i+1}:")
                print(f"  アクション: {form.get('action', 'N/A')}")
                print(f"  メソッド: {form.get('method', 'N/A')}")
                
                form_inputs = form.find_all(['input', 'select', 'textarea'])
                for j, input_elem in enumerate(form_inputs[:5]):  # 最初の5つの入力要素のみ表示
                    input_type = input_elem.get('type', input_elem.name)
                    input_name = input_elem.get('name', 'N/A')
                    input_id = input_elem.get('id', 'N/A')
                    input_value = input_elem.get('value', 'N/A')
                    print(f"    入力要素 {j+1}: {input_type} - name: {input_name}, id: {input_id}, value: {input_value}")
            
            # ページのスクリーンショットを保存
            self.driver.save_screenshot("site_inspection.png")
            print("\nスクリーンショットを保存しました: site_inspection.png")
            
            return True
            
        except Exception as e:
            print(f"サイト構造の調査に失敗しました: {e}")
            return False
    
    def select_parameters(self, target_date):
        """統計パラメータを選択（実際のサイト構造に応じて調整が必要）"""
        try:
            # 年月の選択
            year = target_date.year
            month = target_date.month
            
            print(f"{year}年{month}月のパラメータを設定中...")
            
            # 実際のサイト構造に応じて以下のセレクターを調整してください
            # 例：
            # year_select = self.driver.find_element(By.NAME, "year")
            # year_select.send_keys(str(year))
            
            # month_select = self.driver.find_element(By.NAME, "month")
            # month_select.send_keys(str(month))
            
            # 年齢5歳刻みの選択
            # age_group_select = self.driver.find_element(By.NAME, "age_group")
            # age_group_select.click()
            
            # 町丁別区分の選択
            # area_type_select = self.driver.find_element(By.NAME, "area_type")
            # area_type_select.click()
            
            # すべての地区を選択
            # all_areas_checkbox = self.driver.find_element(By.NAME, "all_areas")
            # all_areas_checkbox.click()
            
            print(f"{year}年{month}月のパラメータを設定しました")
            return True
            
        except Exception as e:
            print(f"パラメータの設定に失敗しました: {e}")
            return False
    
    def extract_data(self):
        """ページからデータを抽出（実際のサイト構造に応じて調整が必要）"""
        try:
            # データテーブルの検索
            table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # データの抽出（実際のサイト構造に応じて調整が必要）
            rows = table.find_elements(By.TAG_NAME, "tr")
            data = []
            
            for row in rows[1:]:  # ヘッダー行をスキップ
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3:
                    area = cells[0].text.strip()
                    age_group = cells[1].text.strip()
                    population = cells[2].text.strip()
                    
                    data.append({
                        'area': area,
                        'age_group': age_group,
                        'population': population
                    })
            
            return data
            
        except Exception as e:
            print(f"データの抽出に失敗しました: {e}")
            return []
    
    def collect_monthly_data(self, target_date):
        """指定された月のデータを収集"""
        try:
            if not self.navigate_to_site():
                return None
            
            if not self.select_parameters(target_date):
                return None
            
            # データ抽出ボタンのクリック（実際のサイト構造に応じて調整が必要）
            # extract_button = WebDriverWait(self.driver, 10).until(
            #     EC.element_to_be_clickable((By.NAME, "extract"))
            # )
            # extract_button.click()
            
            # データの抽出
            data = self.extract_data()
            
            if data:
                # 日付情報を追加
                for item in data:
                    item['date'] = target_date.strftime('%Y-%m')
                    item['year'] = target_date.year
                    item['month'] = target_date.month
                
                print(f"{target_date.strftime('%Y年%m月')}のデータ収集が完了しました（{len(data)}件）")
                return data
            else:
                print(f"{target_date.strftime('%Y年%m月')}のデータが取得できませんでした")
                return None
                
        except Exception as e:
            print(f"{target_date.strftime('%Y年%m月')}のデータ収集に失敗しました: {e}")
            return None
    
    def collect_all_data(self):
        """指定された期間の全データを収集"""
        all_data = []
        date_range = self.get_date_range()
        
        print(f"データ収集を開始します。対象期間: {len(date_range)}ヶ月")
        
        # テスト用に最初の3ヶ月のみ収集
        test_range = date_range[:3]
        print(f"テスト用に最初の{len(test_range)}ヶ月のデータを収集します")
        
        for i, target_date in enumerate(test_range):
            print(f"進捗: {i+1}/{len(test_range)} - {target_date.strftime('%Y年%m月')}")
            
            monthly_data = self.collect_monthly_data(target_date)
            if monthly_data:
                all_data.extend(monthly_data)
            
            # サーバーへの負荷を考慮して待機
            time.sleep(2)
        
        return all_data
    
    def save_data(self, data):
        """データの保存（Colab用）"""
        try:
            df = pd.DataFrame(data)
            
            # データの整理
            df = df.sort_values(['year', 'month', 'area', 'age_group'])
            
            # CSVファイルとして保存
            csv_filename = f"kumamoto_population_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            # Excelファイルとして保存
            excel_filename = f"kumamoto_population_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            
            print(f"データを保存しました:")
            print(f"  CSV: {csv_filename}")
            print(f"  Excel: {excel_filename}")
            
            # Colabでファイルをダウンロード
            try:
                from google.colab import files
                print("\nファイルをダウンロード中...")
                files.download(csv_filename)
                files.download(excel_filename)
            except ImportError:
                print("Google Colab環境ではありません。ファイルはローカルに保存されました。")
            
            return df
            
        except Exception as e:
            print(f"データの保存に失敗しました: {e}")
            return None
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if self.driver:
            self.driver.quit()
            print("ブラウザドライバーを終了しました")

def main():
    """メイン実行関数"""
    collector = None
    try:
        print("=== 熊本市統計データ収集ツール（Google Colab版・修正版） ===")
        
        collector = KumamotoDataCollectorColabFixed()
        
        # サイト構造の調査
        if not collector.navigate_to_site():
            print("サイトへのアクセスに失敗しました")
            return
        
        # サイト構造の調査
        collector.inspect_site_structure()
        
        print("\n=== サイト構造の調査が完了しました ===")
        print("以下の手順で進めてください:")
        print("1. 上記の調査結果を確認")
        print("2. select_parameters()とextract_data()メソッドを実際のサイト構造に合わせて調整")
        print("3. 調整後、collect_all_data()を実行してデータ収集を開始")
        
        # テスト用のデータ収集（実際のサイト構造に合わせて調整後）
        # all_data = collector.collect_all_data()
        # if all_data:
        #     final_df = collector.save_data(all_data)
        #     if final_df is not None:
        #         print("\n=== データ収集完了 ===")
        #         print(f"総データ件数: {len(final_df)}")
        #         print(f"対象期間: {final_df['date'].min()} 〜 {final_df['date'].max()}")
        #         print(f"対象地区数: {final_df['area'].nunique()}")
        #         print(f"年齢区分数: {final_df['age_group'].nunique()}")
        #         
        #         # サンプルデータの表示
        #         print("\n=== サンプルデータ ===")
        #         print(final_df.head(10).to_string())
        # else:
        #     print("データの収集に失敗しました")
            
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if collector:
            collector.cleanup()

if __name__ == "__main__":
    main()
