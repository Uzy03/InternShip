#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熊本市統計サイトの詳細構造調査スクリプト（修正版）
ユーザーデータディレクトリの問題を完全に解決
"""

import time
import os
import subprocess
import tempfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def install_chromium():
    """Google Colab環境でChromiumをインストール"""
    try:
        print("Google Colab環境でChromiumをインストール中...")
        
        # 既存のChromeプロセスをクリーンアップ
        try:
            subprocess.run(["pkill", "-f", "chrome"], capture_output=True)
            subprocess.run(["pkill", "-f", "chromium"], capture_output=True)
            print("既存のChrome/Chromiumプロセスをクリーンアップしました")
        except:
            pass
        
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "chromium-browser"], check=True, capture_output=True)
        print("✓ Chromiumのインストールが完了しました")
        return True
    except Exception as e:
        print(f"✗ Chromiumのインストールに失敗しました: {e}")
        return False

def setup_driver():
    """Chromiumドライバーの設定（ユーザーデータディレクトリの問題を完全解決）"""
    try:
        print("Chromiumドライバーの設定を開始...")

        chromium_path = "/usr/bin/chromium-browser"
        if not os.path.exists(chromium_path):
            if not install_chromium():
                raise Exception("Chromiumのインストールに失敗しました")

        chrome_options = Options()

        # ユーザーデータディレクトリを完全に無効化
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        
        # ユーザーデータディレクトリ関連の設定を完全に無効化
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--disable-application-cache")
        chrome_options.add_argument("--disable-cache")
        chrome_options.add_argument("--disable-offline-load-stale-cache")
        chrome_options.add_argument("--disk-cache-size=0")
        chrome_options.add_argument("--media-cache-size=0")
        
        # より強力なユーザーデータディレクトリ無効化
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-translate")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")

        # Chromiumのパスを明示的に設定
        chrome_options.binary_location = chromium_path

        print("Chromeドライバーをインストール中...")
        driver_path = ChromeDriverManager().install()
        service = Service(driver_path)

        # より確実なドライバー初期化
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✓ Chromiumドライバーの初期化が完了しました")
        except Exception as e:
            print(f"通常の初期化に失敗、代替方法を試行: {e}")
            # 代替方法：より制限的なオプションで再試行
            chrome_options.add_argument("--single-process")
            chrome_options.add_argument("--no-zygote")
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✓ 代替方法でChromiumドライバーの初期化が完了しました")

        return driver

    except Exception as e:
        print(f"✗ ドライバーの設定に失敗しました: {e}")
        raise

def detailed_inspection():
    """詳細なサイト構造調査"""
    driver = None
    try:
        print("=== 詳細なサイト構造調査 ===")

        driver = setup_driver()
        url = "https://tokei.city.kumamoto.jp/content/ASP/Jinkou/default.asp"

        print(f"サイトにアクセス中: {url}")
        driver.get(url)
        time.sleep(10)

        print(f"ページタイトル: {driver.title}")
        print(f"現在のURL: {driver.current_url}")

        # ページソースの詳細分析
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        print(f"\nページソースの長さ: {len(page_source)} 文字")

        # 最初の2000文字を表示
        print("\n=== ページソース（最初の2000文字） ===")
        print(page_source[:2000])

        # より詳細な要素調査
        print("\n=== 詳細な要素調査 ===")

        # すべての要素を調査
        all_elements = soup.find_all()
        print(f"総要素数: {len(all_elements)}")

        # タグ別の要素数をカウント
        tag_counts = {}
        for element in all_elements:
            tag = element.name
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        print("\n=== タグ別要素数 ===")
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {tag}: {count}件")

        # フォーム要素の詳細調査
        print("\n=== フォーム要素の詳細調査 ===")
        forms = soup.find_all('form')
        print(f"フォーム数: {len(forms)}")

        for i, form in enumerate(forms):
            print(f"\nフォーム {i+1}:")
            print(f"  タグ: {form.name}")
            print(f"  アクション: {form.get('action', 'N/A')}")
            print(f"  メソッド: {form.get('method', 'N/A')}")
            print(f"  ID: {form.get('id', 'N/A')}")
            print(f"  クラス: {form.get('class', 'N/A')}")
            print(f"  その他の属性: {form.attrs}")

            # フォーム内のすべての要素を調査
            form_elements = form.find_all()
            print(f"  フォーム内要素数: {len(form_elements)}")

            for j, elem in enumerate(form_elements[:10]):  # 最初の10件
                print(f"    要素 {j+1}: {elem.name} - {elem.attrs}")

        # 入力要素の詳細調査
        print("\n=== 入力要素の詳細調査 ===")
        inputs = soup.find_all(['input', 'select', 'textarea', 'button'])
        print(f"入力関連要素数: {len(inputs)}")

        for i, input_elem in enumerate(inputs[:20]):  # 最初の20件
            print(f"\n入力要素 {i+1}:")
            print(f"  タグ: {input_elem.name}")
            print(f"  属性: {input_elem.attrs}")
            if input_elem.get_text(strip=True):
                print(f"  テキスト: {input_elem.get_text(strip=True)}")

        # テーブル要素の詳細調査
        print("\n=== テーブル要素の詳細調査 ===")
        tables = soup.find_all('table')
        print(f"テーブル数: {len(tables)}")

        for i, table in enumerate(tables):
            print(f"\nテーブル {i+1}:")
            print(f"  属性: {table.attrs}")

            rows = table.find_all('tr')
            if rows:
                print(f"  行数: {len(rows)}")
                for j, row in enumerate(rows[:3]):  # 最初の3行
                    cells = row.find_all(['td', 'th'])
                    print(f"    行 {j+1}: {len(cells)}列")
                    for k, cell in enumerate(cells[:5]):  # 最初の5列
                        cell_text = cell.get_text(strip=True)
                        if cell_text:
                            print(f"      列 {k+1}: {cell_text}")

        # リンク要素の調査
        print("\n=== リンク要素の調査 ===")
        links = soup.find_all('a')
        print(f"リンク数: {len(links)}")

        for i, link in enumerate(links[:10]):  # 最初の10件
            href = link.get('href', 'N/A')
            text = link.get_text(strip=True)
            if text or href != 'N/A':
                print(f"  リンク {i+1}: {text} -> {href}")

        # スクリプト要素の調査
        print("\n=== スクリプト要素の調査 ===")
        scripts = soup.find_all('script')
        print(f"スクリプト数: {len(scripts)}")

        for i, script in enumerate(scripts[:5]):  # 最初の5件
            src = script.get('src', 'N/A')
            script_type = script.get('type', 'N/A')
            print(f"  スクリプト {i+1}: type={script_type}, src={src}")

        # スタイル要素の調査
        print("\n=== スタイル要素の調査 ===")
        styles = soup.find_all('style')
        print(f"スタイル数: {len(styles)}")

        # ページのスクリーンショットを保存
        driver.save_screenshot("detailed_inspection.png")
        print("\n詳細調査のスクリーンショットを保存しました: detailed_inspection.png")

        return True

    except Exception as e:
        print(f"詳細調査に失敗しました: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if driver:
            driver.quit()
            print("ブラウザドライバーを終了しました")

if __name__ == "__main__":
    detailed_inspection()
