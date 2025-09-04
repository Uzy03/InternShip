#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熊本市統計サイトへの接続テストスクリプト（Google Colab対応・修正版）
基本的な動作確認用
"""

import time
import os
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def install_chrome():
    """Google Colab環境でChromeをインストール"""
    try:
        print("Google Colab環境でChromeをインストール中...")
        
        # システムの更新
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        
        # Chromeのインストール
        subprocess.run([
            "apt-get", "install", "-y", 
            "wget", "gnupg", "unzip", "curl"
        ], check=True, capture_output=True)
        
        # Google Chromeのリポジトリを追加
        subprocess.run([
            "wget", "-q", "-O", "-", 
            "https://dl.google.com/linux/linux_signing_key.pub"
        ], check=True, capture_output=True)
        
        subprocess.run([
            "wget", "-q", "-O", "-", 
            "https://dl.google.com/linux/linux_signing_key.pub"
        ], check=True, capture_output=True)
        
        # Chromeのダウンロードとインストール
        subprocess.run([
            "wget", "-q", "-O", "/tmp/chrome.deb",
            "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
        ], check=True, capture_output=True)
        
        subprocess.run([
            "apt-get", "install", "-y", "/tmp/chrome.deb"
        ], check=True, capture_output=True)
        
        print("✓ Chromeのインストールが完了しました")
        return True
        
    except Exception as e:
        print(f"✗ Chromeのインストールに失敗しました: {e}")
        return False

def setup_chrome_driver():
    """Google Colab用Chromeドライバーの設定"""
    try:
        print("Chromeドライバーの設定を開始...")
        
        # Chromeがインストールされているか確認
        chrome_path = "/usr/bin/google-chrome"
        if not os.path.exists(chrome_path):
            print("Chromeが見つかりません。インストールを試行します...")
            if not install_chrome():
                raise Exception("Chromeのインストールに失敗しました")
        
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
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Chromeのパスを明示的に設定
        chrome_options.binary_location = chrome_path
        
        # Colab環境の検出
        if 'COLAB_GPU' in os.environ:
            print("Google Colab環境を検出しました")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
        
        print("Chromeドライバーをインストール中...")
        driver_path = ChromeDriverManager().install()
        print(f"ドライバーパス: {driver_path}")
        
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("✓ Chromeドライバーの初期化が完了しました")
        return driver
        
    except Exception as e:
        print(f"✗ Chromeドライバーの設定に失敗しました: {e}")
        raise

def test_connection():
    """サイトへの接続テスト"""
    driver = None
    try:
        print("=== 熊本市統計サイト接続テスト ===")
        print("Chromeドライバーを設定中...")
        
        # Chromeドライバーの設定
        driver = setup_chrome_driver()
        print("ドライバーの設定完了")
        
        # サイトにアクセス
        url = "https://tokei.city.kumamoto.jp/content/ASP/Jinkou/default.asp"
        print(f"サイトにアクセス中: {url}")
        
        driver.get(url)
        print("ページの読み込みを待機中...")
        time.sleep(10)  # 待機時間を増やす
        
        # ページの基本情報を確認
        print(f"ページタイトル: {driver.title}")
        print(f"現在のURL: {driver.current_url}")
        
        # ページの要素を確認
        print("\n=== ページ要素の確認 ===")
        
        # フォーム要素
        forms = driver.find_elements(By.TAG_NAME, "form")
        print(f"フォーム数: {len(forms)}")
        
        # テーブル要素
        tables = driver.find_elements(By.TAG_NAME, "table")
        print(f"テーブル数: {len(tables)}")
        
        # 入力要素
        inputs = driver.find_elements(By.TAG_NAME, "input")
        print(f"入力要素数: {len(inputs)}")
        
        # セレクト要素
        selects = driver.find_elements(By.TAG_NAME, "select")
        print(f"セレクト要素数: {len(selects)}")
        
        # ボタン要素
        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"ボタン要素数: {len(buttons)}")
        
        # ページのスクリーンショットを保存
        driver.save_screenshot("connection_test.png")
        print("\nスクリーンショットを保存しました: connection_test.png")
        
        # ページソースの一部を表示
        page_source = driver.page_source
        print(f"\nページソースの長さ: {len(page_source)} 文字")
        
        # 最初の1000文字を表示
        print("\n=== ページソース（最初の1000文字） ===")
        print(page_source[:1000])
        
        print("\n=== 接続テスト完了 ===")
        print("✓ サイトへの接続は成功しました")
        
        return True
        
    except Exception as e:
        print(f"✗ 接続テストに失敗しました: {e}")
        print(f"エラーの詳細: {type(e).__name__}")
        import traceback
        print("スタックトレース:")
        traceback.print_exc()
        return False
        
    finally:
        if driver:
            print("ブラウザドライバーを終了中...")
            driver.quit()
            print("ブラウザドライバーを終了しました")

def main():
    """メイン実行関数"""
    print("接続テストを開始します...")
    result = test_connection()
    
    if result:
        print("\n=== テスト結果 ===")
        print("✓ 接続テストが成功しました")
        print("次のステップとして、サイト構造の調査を行ってください")
    else:
        print("\n=== テスト結果 ===")
        print("✗ 接続テストが失敗しました")
        print("エラーログを確認して、問題を特定してください")

if __name__ == "__main__":
    main()
