#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熊本市 統計情報室 > 人口情報
https://tokei.city.kumamoto.jp/content/ASP/Jinkou/default.asp
指定条件で「EXCEL出力」を自動ダウンロードするスクリプト（ローカル環境版）

条件:
- 年月度: 平成10年4月 〜 令和7年4月（年1回、4月時点）
- 統計  : 年齢5歳刻み
- 区分  : 町丁別
- 地区  : すべて（チェック外さず）

使用方法:
1. 必要なパッケージをインストール: pip install selenium webdriver-manager pandas openpyxl
2. Chromeブラウザがインストールされていることを確認
3. スクリプトを実行: python data_download_local.py
"""

import os
import time
import platform
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://tokei.city.kumamoto.jp/content/ASP/Jinkou/default.asp"

# ==== ダウンロード設定 ====
SAVE_DIR = Path("./downloads_kumamoto_town_age5").resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==== 収集レンジ（和暦の見出しをそのまま指定）====
# 4月は 2019年なら「平成31年」になる（令和は5月から）
JAPANESE_YEARS_APR = (
    [f"平成{n}年" for n in range(10, 31)]  # 平成10〜30
    + ["平成31年"]                          # 2019年4月
    + [f"令和{n}年" for n in range(2, 8)]   # 2020〜2025（令和2〜7）
)
TARGET_MONTH_TEXT = "4月"
STATS_TEXT = "年齢5歳刻み"
KUBUN_TEXT = "町丁別"

def setup_driver():
    """ローカル環境用のChromeドライバー設定"""
    try:
        print("Chromeドライバーの設定を開始...")
        
        # プラットフォームに応じたChromeオプション設定
        chrome_options = webdriver.ChromeOptions()
        
        # ダウンロード設定
        prefs = {
            "download.default_directory": str(SAVE_DIR),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # プラットフォーム別の設定
        if platform.system() == "Darwin":  # macOS
            print("macOS環境を検出しました")
            
            # macOS ARM64（Apple Silicon）対応
            import subprocess
            try:
                # Chromeのパスを確認
                result = subprocess.run(['which', 'google-chrome'], capture_output=True, text=True)
                if result.returncode == 0:
                    chrome_path = result.stdout.strip()
                    print(f"Chromeパス: {chrome_path}")
                    chrome_options.binary_location = chrome_path
                else:
                    # 標準的なChromeのパスを試行
                    standard_paths = [
                        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                        "/Applications/Chromium.app/Contents/MacOS/Chromium"
                    ]
                    for path in standard_paths:
                        if os.path.exists(path):
                            print(f"Chromeパス: {path}")
                            chrome_options.binary_location = path
                            break
            except Exception as e:
                print(f"Chromeパスの確認に失敗: {e}")
            
            # macOS用の設定
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
        elif platform.system() == "Linux":
            print("Linux環境を検出しました")
            # Linux用の設定
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
        else:  # Windows
            print("Windows環境を検出しました")
        
        # 共通設定
        chrome_options.add_argument("--window-size=1280,900")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        
        # ヘッドレスモード（デバッグ用に無効化）
        # chrome_options.add_argument("--headless")
        
        # デバッグ用の設定
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        print("Chromeドライバーをインストール中...")
        
        # macOS ARM64対応のドライバー管理
        if platform.system() == "Darwin":
            try:
                # アーキテクチャを確認
                result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
                if result.stdout.strip() == 'arm64':
                    print("macOS ARM64（Apple Silicon）環境を検出しました")
                    # ARM64用のドライバーを明示的に指定
                    driver_path = ChromeDriverManager(version="latest").install()
                else:
                    driver_path = ChromeDriverManager().install()
            except:
                driver_path = ChromeDriverManager().install()
        else:
            driver_path = ChromeDriverManager().install()
        
        # 正しいChromeDriverファイルを特定
        if platform.system() == "Darwin":
            try:
                # ディレクトリ内の実際のChromeDriverファイルを探す
                driver_dir = os.path.dirname(driver_path)
                print(f"ドライバーディレクトリ: {driver_dir}")
                
                # ディレクトリ内のファイルを確認
                import glob
                driver_files = glob.glob(os.path.join(driver_dir, "*"))
                print(f"ディレクトリ内のファイル: {[os.path.basename(f) for f in driver_files]}")
                
                # 実際のChromeDriverファイルを探す
                actual_driver = None
                for file_path in driver_files:
                    if os.path.basename(file_path).startswith('chromedriver') and not file_path.endswith('.txt'):
                        actual_driver = file_path
                        break
                
                if actual_driver:
                    driver_path = actual_driver
                    print(f"実際のChromeDriverファイル: {driver_path}")
                else:
                    print("警告: 実際のChromeDriverファイルが見つかりません")
                    
            except Exception as e:
                print(f"ChromeDriverファイルの特定に失敗: {e}")
        
        service = Service(driver_path)
        
        # ドライバーの実行権限を確認・修正
        if platform.system() == "Darwin":
            try:
                os.chmod(driver_path, 0o755)
                print(f"ドライバーの実行権限を設定: {driver_path}")
            except Exception as e:
                print(f"実行権限の設定に失敗: {e}")
        
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✓ Chromeドライバーの初期化が完了しました")
        except Exception as e:
            print(f"通常の初期化に失敗、代替手段を試行: {e}")
            
            # macOS ARM64環境での代替手段
            if platform.system() == "Darwin":
                try:
                    # システムのChromeを使用
                    print("システムのChromeを使用して再試行...")
                    chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    print("✓ システムChromeでの初期化が完了しました")
                except Exception as e2:
                    print(f"システムChromeでの初期化も失敗: {e2}")
                    # 最終手段：Safariドライバーを試行
                    try:
                        print("Safariドライバーでの初期化を試行...")
                        from selenium.webdriver.safari.service import Service as SafariService
                        from selenium.webdriver.safari.webdriver import SafariDriver
                        safari_service = SafariService()
                        driver = SafariDriver(service=safari_service)
                        print("✓ Safariドライバーでの初期化が完了しました")
                    except Exception as e3:
                        print(f"Safariドライバーでの初期化も失敗: {e3}")
                        # 最終手段：手動でChromeDriverをダウンロード
                        try:
                            print("手動ChromeDriverのダウンロードを試行...")
                            import urllib.request
                            import zipfile
                            
                            # ARM64用のChromeDriverを直接ダウンロード
                            arm64_url = "https://chromedriver.storage.googleapis.com/139.0.7258.154/chromedriver_mac_arm64.zip"
                            local_zip = "/tmp/chromedriver_mac_arm64.zip"
                            local_driver = "/tmp/chromedriver"
                            
                            print(f"ChromeDriverをダウンロード中: {arm64_url}")
                            urllib.request.urlretrieve(arm64_url, local_zip)
                            
                            # 解凍
                            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                                zip_ref.extractall("/tmp")
                            
                            # 実行権限を設定
                            os.chmod(local_driver, 0o755)
                            
                            # 新しいServiceでドライバーを作成
                            manual_service = Service(local_driver)
                            driver = webdriver.Chrome(service=manual_service, options=chrome_options)
                            print("✓ 手動ダウンロードしたChromeDriverでの初期化が完了しました")
                            
                        except Exception as e4:
                            print(f"手動ChromeDriverのダウンロードも失敗: {e4}")
                            raise Exception("すべてのドライバーでの初期化に失敗しました")
            else:
                raise e
        
        return driver
        
    except Exception as e:
        print(f"✗ ドライバーの設定に失敗しました: {e}")
        raise

def select_by_text(select_el, visible_text):
    """セレクトボックスでテキストを選択"""
    try:
        sel = Select(select_el)
        sel.select_by_visible_text(visible_text)
        print(f"✓ {visible_text} を選択しました")
    except Exception as e:
        print(f"✗ {visible_text} の選択に失敗: {e}")
        raise

def find_select_by_label_text(driver, label_text):
    """
    ラベルの直後/近傍にある <select> を見つけるための便利関数。
    画面の構造が多少変わっても動くよう、ゆるめに探索します。
    """
    print(f"'{label_text}' のセレクトボックスを探索中...")
    
    # ラベルの文字列を含む要素を起点に、同じ親内の select を探す
    candidates = driver.find_elements(By.XPATH, f"//*[contains(text(), '{label_text}')]")
    print(f"  候補要素数: {len(candidates)}")
    
    for i, c in enumerate(candidates):
        print(f"  候補 {i+1}: {c.tag_name} - {c.text[:50]}...")
        
        # 兄弟 or 親要素配下を探索
        for xp in ["following::select[1]", "ancestor::*//select[1]"]:
            try:
                el = c.find_element(By.XPATH, xp)
                if el.tag_name.lower() == "select":
                    print(f"  ✓ '{label_text}' のセレクトボックスを発見: {el.get_attribute('name') or 'nameなし'}")
                    return el
            except:
                continue
    
    print(f"  ✗ '{label_text}' のセレクトボックスが見つかりませんでした")
    return None

def click_excel_button(driver):
    """「EXCEL出力」ボタンをクリック"""
    print("EXCEL出力ボタンを探索中...")
    
    # ボタンは <input> or <a> の場合があるので両対応
    for i, xp in enumerate([
        "//input[contains(@value,'EXCEL') or contains(@value,'Excel') or contains(@value,'EXCEL出力')]",
        "//a[contains(text(),'EXCEL')]",
        "//button[contains(text(),'EXCEL')]",
    ]):
        try:
            print(f"  パターン {i+1} を試行中...")
            btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, xp)))
            print(f"  ✓ EXCEL出力ボタンを発見: {btn.tag_name} - {btn.text or btn.get_attribute('value')}")
            btn.click()
            print("  ✓ EXCEL出力ボタンをクリックしました")
            return True
        except Exception as e:
            print(f"  パターン {i+1} 失敗: {e}")
            continue
    
    raise RuntimeError("EXCEL出力ボタンが見つかりませんでした。画面の構造が変わっていないか確認してください。")

def wait_download_finished(before_files, timeout=60):
    """
    ダウンロード完了を簡易検出：新しい .xls/.xlsx が出現して
    .crdownload が消えるのを待つ
    """
    print("ダウンロード完了を待機中...")
    end = time.time() + timeout
    
    while time.time() < end:
        files = set(SAVE_DIR.glob("*"))
        new_files = [f for f in files - before_files if f.suffix.lower() in [".xls", ".xlsx"]]
        # .crdownload が残っていないかチェック
        tmp_left = [f for f in SAVE_DIR.glob("*.crdownload")]
        
        if new_files and not tmp_left:
            print(f"✓ ダウンロード完了: {new_files[0].name}")
            return new_files[0]
        
        time.sleep(0.5)
    
    print("✗ ダウンロード完了を検出できませんでした")
    return None

def main():
    """メイン処理"""
    print("=== 熊本市統計データ自動ダウンロード開始 ===")
    print(f"保存先: {SAVE_DIR}")
    
    driver = None
    try:
        # ドライバーの初期化
        driver = setup_driver()
        
        # サイトにアクセス
        print(f"サイトにアクセス中: {URL}")
        driver.get(URL)
        
        # ページの基本情報を確認
        print(f"ページタイトル: {driver.title}")
        print(f"現在のURL: {driver.current_url}")
        
        # 画面ロード待ち（より詳細な情報を表示）
        print("ページの読み込みを待機中...")
        
        # まず、ページの基本構造を確認
        try:
            # ページソースの長さを確認
            page_source = driver.page_source
            print(f"ページソースの長さ: {len(page_source)} 文字")
            
            # 最初の1000文字を表示して構造を確認
            print("ページソース（最初の1000文字）:")
            print(page_source[:1000])
            
            # 基本的な要素の存在確認
            body = driver.find_element(By.TAG_NAME, "body")
            print(f"body要素のテキスト長: {len(body.text)} 文字")
            
            # select要素を探す（より詳細に）
            select_elements = driver.find_elements(By.TAG_NAME, "select")
            print(f"select要素の数: {len(select_elements)}")
            
            if select_elements:
                for i, select in enumerate(select_elements):
                    print(f"select {i+1}: name={select.get_attribute('name')}, id={select.get_attribute('id')}")
            
            # フォーム要素も確認
            form_elements = driver.find_elements(By.TAG_NAME, "form")
            print(f"form要素の数: {len(form_elements)}")
            
            if form_elements:
                for i, form in enumerate(form_elements):
                    print(f"form {i+1}: action={form.get_attribute('action')}, method={form.get_attribute('method')}")
            
        except Exception as e:
            print(f"ページ構造の確認中にエラー: {e}")
        
        # より長い待機時間でselect要素を待つ
        print("select要素の出現を待機中...")
        try:
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "select")))
            print("✓ select要素が見つかりました")
        except Exception as e:
            print(f"select要素の待機に失敗: {e}")
            print("ページの構造を再確認します...")
            
            # 再度ページの状態を確認
            select_elements = driver.find_elements(By.TAG_NAME, "select")
            print(f"現在のselect要素の数: {len(select_elements)}")
            
            if not select_elements:
                print("⚠️  select要素が見つかりません。ページの読み込みに問題がある可能性があります。")
                print("手動でブラウザを確認してください。")
                input("Enterキーを押すと続行します...")
        
        print("✓ ページの読み込みが完了しました")
        
        # 各プルダウン要素を特定
        print("\n=== セレクトボックスの特定 ===")
        
        # iframe内の要素を確認
        print("iframe内の要素を探索中...")
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        print(f"iframeの数: {len(iframes)}")
        
        for i, iframe in enumerate(iframes):
            try:
                src = iframe.get_attribute("src")
                name = iframe.get_attribute("name")
                print(f"iframe {i+1}: src={src}, name={name}")
                
                # iframeに切り替え
                driver.switch_to.frame(iframe)
                print(f"iframe {i+1} に切り替えました")
                
                # iframe内の要素を確認
                iframe_selects = driver.find_elements(By.TAG_NAME, "select")
                print(f"  iframe {i+1} 内のselect要素数: {len(iframe_selects)}")
                
                if iframe_selects:
                    for j, select in enumerate(iframe_selects):
                        print(f"    select {j+1}: name={select.get_attribute('name')}, id={select.get_attribute('id')}")
                
                # メインフレームに戻る
                driver.switch_to.default_content()
                
            except Exception as e:
                print(f"iframe {i+1} の処理中にエラー: {e}")
                driver.switch_to.default_content()
        
        # iframe内から要素を取得
        print("\niframe内からセレクトボックスを取得中...")
        
        # 変数を初期化
        sel_year = None
        sel_month = None
        sel_tokei = None
        sel_kubun = None
        
        print("変数を初期化しました")
        
        # frame_left.asp（左側のiframe）に切り替え
        try:
            driver.switch_to.frame("frame_left")
            print("frame_leftに切り替えました")
            
            # 左側のiframe内で要素を直接取得
            sel_year = driver.find_element(By.ID, "cboDateYear")
            sel_month = driver.find_element(By.ID, "cboDateMonth")
            print(f"✓ 年月度の要素を取得: {sel_year.get_attribute('name')}, {sel_month.get_attribute('name')}")
            
            # メインフレームに戻る
            driver.switch_to.default_content()
            
        except Exception as e:
            print(f"frame_leftの処理に失敗: {e}")
            driver.switch_to.default_content()
        
        # frame_right.asp（右側のiframe）に切り替え
        try:
            driver.switch_to.frame("frame_right")
            print("frame_rightに切り替えました")
            
            # 右側のiframe内で要素を直接取得
            # 統計と区分の要素を探す
            all_elements = driver.find_elements(By.TAG_NAME, "*")
            print(f"frame_right内の全要素数: {len(all_elements)}")
            
            # 統計関連の要素を探す
            for elem in all_elements:
                elem_text = elem.text.strip()
                if "年齢5歳刻み" in elem_text:
                    print(f"統計要素を発見: {elem.tag_name} - {elem_text}")
                    # 親要素内のselectを探す
                    try:
                        parent = elem.find_element(By.XPATH, "..")
                        sel_tokei = parent.find_element(By.TAG_NAME, "select")
                        print(f"✓ 統計のselect要素を発見: {sel_tokei.get_attribute('name')}")
                        break
                    except:
                        continue
            
            # 区分関連の要素を探す
            for elem in all_elements:
                elem_text = elem.text.strip()
                if "町丁別" in elem_text:
                    print(f"区分要素を発見: {elem.tag_name} - {elem_text}")
                    # 親要素内のselectを探す
                    try:
                        parent = elem.find_element(By.XPATH, "..")
                        sel_kubun = parent.find_element(By.TAG_NAME, "select")
                        print(f"✓ 区分のselect要素を発見: {sel_kubun.get_attribute('name')}")
                        break
                    except:
                        continue
            
            # メインフレームに戻る
            driver.switch_to.default_content()
            
        except Exception as e:
            print(f"frame_rightの処理に失敗: {e}")
            driver.switch_to.default_content()
        
        if not (sel_year and sel_tokei and sel_kubun):
            print("⚠️  一部のセレクトボックスが見つかりません。フォールバックを試行...")
            
            # メインページで再度探索
            print("メインページで再度探索中...")
            main_selects = driver.find_elements(By.TAG_NAME, "select")
            print(f"メインページのselect要素数: {len(main_selects)}")
            
            for i, select in enumerate(main_selects):
                print(f"  select {i+1}: name={select.get_attribute('name')}, id={select.get_attribute('id')}")
                # 周辺のテキストを確認
                try:
                    parent = select.find_element(By.XPATH, "..")
                    parent_text = parent.text.strip()
                    print(f"    親要素のテキスト: {parent_text[:100]}...")
                    
                    if "年齢" in parent_text and not sel_tokei:
                        sel_tokei = select
                        print(f"✓ 統計要素として特定: {select.get_attribute('name')}")
                    elif "町丁" in parent_text and not sel_kubun:
                        sel_kubun = select
                        print(f"✓ 区分要素として特定: {select.get_attribute('name')}")
                except:
                    continue
            
            # frame_right内で再度探索
            try:
                driver.switch_to.frame("frame_right")
                print("frame_rightで再度探索中...")
                
                # より詳細な要素探索
                select_elements = driver.find_elements(By.TAG_NAME, "select")
                print(f"frame_right内のselect要素数: {len(select_elements)}")
                
                for i, select in enumerate(select_elements):
                    print(f"  select {i+1}: name={select.get_attribute('name')}, id={select.get_attribute('id')}")
                    # 周辺のテキストを確認
                    try:
                        parent = select.find_element(By.XPATH, "..")
                        parent_text = parent.text.strip()
                        print(f"    親要素のテキスト: {parent_text[:100]}...")
                        
                        if "年齢" in parent_text and not sel_tokei:
                            sel_tokei = select
                            print(f"✓ 統計要素として特定: {select.get_attribute('name')}")
                        elif "町丁" in parent_text and not sel_kubun:
                            sel_kubun = select
                            print(f"✓ 区分要素として特定: {select.get_attribute('name')}")
                    except:
                        continue
                
                driver.switch_to.default_content()
                
            except Exception as e:
                print(f"フォールバック探索に失敗: {e}")
                driver.switch_to.default_content()
        
        # 最終確認
        print(f"\n=== 要素の取得状況 ===")
        print(f"sel_year: {'✓' if sel_year else '✗'}")
        print(f"sel_month: {'✓' if sel_month else '✗'}")
        print(f"sel_tokei: {'✓' if sel_tokei else '✗'}")
        print(f"sel_kubun: {'✓' if sel_kubun else '✗'}")
        
        if not sel_tokei or not sel_kubun:
            print("\n⚠️  統計・区分の要素が見つかりません。")
            print("手動で要素を確認してください。")
            
            # 要素が見つからない場合の対処
            if not sel_tokei:
                print("統計要素が見つからないため、デフォルト値を設定します")
                # 一時的なダミー要素を作成（実際の処理はスキップ）
                sel_tokei = "DUMMY_STATS"
            
            if not sel_kubun:
                print("区分要素が見つからないため、デフォルト値を設定します")
                # 一時的なダミー要素を作成（実際の処理はスキップ）
                sel_kubun = "DUMMY_KUBUN"
            
            input("Enterキーを押すと続行します...")
        
        # 固定条件を先に設定
        print("\n=== 固定条件の設定 ===")
        
        # 要素が実際に見つかった場合のみ処理
        if sel_tokei and sel_tokei != "DUMMY_STATS":
            try:
                select_by_text(sel_tokei, STATS_TEXT)  # 年齢5歳刻み
                print("✓ 統計条件を設定しました")
            except Exception as e:
                print(f"⚠️  統計条件の設定に失敗: {e}")
        else:
            print("⚠️  統計条件の設定をスキップします（要素が見つかりませんでした）")
        
        if sel_kubun and sel_kubun != "DUMMY_KUBUN":
            try:
                select_by_text(sel_kubun, KUBUN_TEXT)  # 町丁別
                print("✓ 区分条件を設定しました")
            except Exception as e:
                print(f"⚠️  区分条件の設定に失敗: {e}")
        else:
            print("⚠️  区分条件の設定をスキップします（要素が見つかりませんでした）")
        
        # ループで年度を切り替えつつ 4月 を選び、EXCEL出力
        print(f"\n=== データ収集開始（対象期間: {len(JAPANESE_YEARS_APR)}年分） ===")
        
        for i, jp_year in enumerate(JAPANESE_YEARS_APR, 1):
            print(f"\n--- {i}/{len(JAPANESE_YEARS_APR)}: {jp_year} {TARGET_MONTH_TEXT} ---")
            
            try:
                # 年の選択
                select_by_text(sel_year, jp_year)
                
                # 月の選択（年と月が別プルダウンなら）
                try:
                    select_by_text(sel_month, TARGET_MONTH_TEXT)
                except:
                    pass  # 年月が一体のUIなら無視
                
                # ダウンロード前のファイル一覧
                before = set(SAVE_DIR.glob("*"))
                
                # EXCEL出力
                click_excel_button(driver)
                
                # ダウンロード完了待ち
                saved = wait_download_finished(before, timeout=90)
                if saved is None:
                    print(f"⚠️  {jp_year} {TARGET_MONTH_TEXT} のダウンロードを検出できませんでした。手動確認してください。")
                    continue
                
                # わかりやすいファイル名に変更
                # 和暦→西暦換算（簡易）
                # 平成: 1989+N, 令和: 2018+N
                if jp_year.startswith("平成"):
                    n = int(jp_year.replace("平成","").replace("年",""))
                    year = 1988 + n  # 平成1=1989
                elif jp_year.startswith("令和"):
                    n = int(jp_year.replace("令和","").replace("年",""))
                    year = 2018 + n  # 令和1=2019
                else:
                    year = jp_year
                
                new_name = SAVE_DIR / f"town_{year}-04{saved.suffix.lower()}"
                try:
                    saved.rename(new_name)
                    print(f"✓ 保存完了: {new_name.name}")
                except:
                    # 同名が存在する場合は連番
                    i = 1
                    while True:
                        nn = SAVE_DIR / f"town_{year}-04_{i}{saved.suffix.lower()}"
                        if not nn.exists():
                            saved.rename(nn)
                            new_name = nn
                            break
                        i += 1
                    print(f"✓ 保存完了（連番）: {new_name.name}")
                
            except Exception as e:
                print(f"✗ {jp_year} {TARGET_MONTH_TEXT} の処理に失敗: {e}")
                continue
            
            # アンチブロック用の小休止
            time.sleep(1.0)
        
        print("\n=== データ収集完了 ===")
        
    except Exception as e:
        print(f"✗ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if driver:
            driver.quit()
            print("ブラウザドライバーを終了しました")
    
    print("=== 処理完了 ===")

if __name__ == "__main__":
    main()
