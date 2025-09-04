# -*- coding: utf-8 -*-
"""
熊本市 統計情報室 > 人口情報
https://tokei.city.kumamoto.jp/content/ASP/Jinkou/default.asp
指定条件で「EXCEL出力」を自動ダウンロードするスクリプト。

条件:
- 年月度: 平成10年4月 〜 令和7年4月（年1回、4月時点）
- 統計  : 年齢5歳刻み
- 区分  : 町丁別
- 地区  : すべて（チェック外さず）
"""

import os
import time
import subprocess
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

def install_chromium():
    """Google Colab環境でChromiumをインストール"""
    try:
        print("Google Colab環境でChromiumをインストール中...")
        
        # 既存のChromeプロセスをクリーンアップ
        try:
            subprocess.run(["pkill", "-f", "chrome"], capture_output=True)
            subprocess.run(["pkill", "-f", "chromium"], capture_output=True)
            subprocess.run(["pkill", "-f", "chromedriver"], capture_output=True)
            print("既存のChrome/Chromiumプロセスをクリーンアップしました")
        except:
            pass
        
        # 一時ディレクトリのクリーンアップ
        try:
            subprocess.run(["rm", "-rf", "/tmp/.com.google.Chrome*"], capture_output=True)
            subprocess.run(["rm", "-rf", "/tmp/.org.chromium.Chromium*"], capture_output=True)
            subprocess.run(["rm", "-rf", "/tmp/chrome_*"], capture_output=True)
            print("一時ディレクトリをクリーンアップしました")
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
    """Google Colab環境対応のChromeドライバー設定"""
    try:
        # Chromiumのインストール確認
        chromium_path = "/usr/bin/chromium-browser"
        if not os.path.exists(chromium_path):
            if not install_chromium():
                raise Exception("Chromiumのインストールに失敗しました")

        # Chromeの自動ダウンロード設定
        chrome_options = webdriver.ChromeOptions()
        
        # ユーザーデータディレクトリの問題を完全に解決
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1280,900")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--disable-application-cache")
        chrome_options.add_argument("--disable-cache")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-default-apps")
        
        # より強力なユーザーデータディレクトリ無効化
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--disable-hang-monitor")
        chrome_options.add_argument("--disable-prompt-on-repost")
        chrome_options.add_argument("--disable-domain-reliability")
        chrome_options.add_argument("--disable-component-extensions-with-background-pages")
        chrome_options.add_argument("--disable-background-networking")
        
        # Chromiumのパスを明示的に設定
        chrome_options.binary_location = chromium_path
        
        prefs = {
            "download.default_directory": str(SAVE_DIR),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            # PDFをブラウザ表示せず保存（念のため）
            "plugins.always_open_pdf_externally": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        service = Service(ChromeDriverManager().install())
        
        # より確実なドライバー初期化
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✓ Chromiumドライバーの初期化が完了しました")
        except Exception as e:
            print(f"通常の初期化に失敗、代替方法を試行: {e}")
            # 代替方法：より制限的なオプションで再試行
            chrome_options.add_argument("--single-process")
            chrome_options.add_argument("--no-zygote")
            try:
                driver = webdriver.Chrome(service=service, options=chrome_options)
                print("✓ 代替方法でChromiumドライバーの初期化が完了しました")
            except Exception as e2:
                print(f"代替方法も失敗、最終手段を試行: {e2}")
                # 最終手段：完全に新しいオプションセットで再試行
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--disable-extensions")
                chrome_options.add_argument("--disable-plugins")
                chrome_options.add_argument("--incognito")
                chrome_options.add_argument("--single-process")
                chrome_options.add_argument("--no-zygote")
                chrome_options.add_argument("--disable-background-timer-throttling")
                chrome_options.add_argument("--disable-backgrounding-occluded-windows")
                chrome_options.add_argument("--disable-renderer-backgrounding")
                chrome_options.binary_location = chromium_path
                
                prefs = {
                    "download.default_directory": str(SAVE_DIR),
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "safebrowsing.enabled": True,
                    "plugins.always_open_pdf_externally": True,
                }
                chrome_options.add_experimental_option("prefs", prefs)
                
                driver = webdriver.Chrome(service=service, options=chrome_options)
                print("✓ 最終手段でChromiumドライバーの初期化が完了しました")
        
        return driver
        
    except Exception as e:
        print(f"ドライバーの設定に失敗しました: {e}")
        raise

def select_by_text(select_el, visible_text):
    sel = Select(select_el)
    sel.select_by_visible_text(visible_text)

def find_select_by_label_text(driver, label_text):
    """
    ラベルの直後/近傍にある <select> を見つけるための便利関数。
    画面の構造が多少変わっても動くよう、ゆるめに探索します。
    """
    # ラベルの文字列を含む要素を起点に、同じ親内の select を探す
    candidates = driver.find_elements(By.XPATH, f"//*[contains(text(), '{label_text}')]")
    for c in candidates:
        # 兄弟 or 親要素配下を探索
        for xp in ["following::select[1]", "ancestor::*//select[1]"]:
            try:
                el = c.find_element(By.XPATH, xp)
                if el.tag_name.lower() == "select":
                    return el
            except:
                pass
    return None

def click_excel_button(driver):
    # 「EXCEL出力」ボタン（右上）をクリック
    # ボタンは <input> or <a> の場合があるので両対応
    for xp in [
        "//input[contains(@value,'EXCEL') or contains(@value,'Excel') or contains(@value,'EXCEL出力')]",
        "//a[contains(text(),'EXCEL')]",
        "//button[contains(text(),'EXCEL')]",
    ]:
        try:
            btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, xp)))
            btn.click()
            return True
        except:
            continue
    raise RuntimeError("EXCEL出力ボタンが見つかりませんでした。画面の構造が変わっていないか確認してください。")

def wait_download_finished(before_files, timeout=60):
    """
    ダウンロード完了を簡易検出：新しい .xls/.xlsx が出現して
    .crdownload が消えるのを待つ
    """
    end = time.time() + timeout
    while time.time() < end:
        files = set(SAVE_DIR.glob("*"))
        new_files = [f for f in files - before_files if f.suffix.lower() in [".xls", ".xlsx"]]
        # .crdownload が残っていないかチェック
        tmp_left = [f for f in SAVE_DIR.glob("*.crdownload")]
        if new_files and not tmp_left:
            return new_files[0]
        time.sleep(0.5)
    return None

def main():
    driver = setup_driver()
    driver.get(URL)

    # 画面ロード待ち（主要なプルダウンが現れるまで）
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "select")))

    # 各プルダウン要素を特定（うまく見つからない場合は、下の find_select_by_label_text の代わりに
    # driver.find_element(By.NAME, "要素名") などに置き換えてください）
    sel_year = find_select_by_label_text(driver, "年月度")
    sel_month = sel_year  # 年月度のUIが年・月で別selectの場合は分けて取得
    # 年/月が別セレクトになっているなら、下のように取り直す：
    # sel_year  = find_select_by_label_text(driver, "年")
    # sel_month = find_select_by_label_text(driver, "月")

    sel_tokei = find_select_by_label_text(driver, "統計")
    sel_kubun = find_select_by_label_text(driver, "区分")

    if not (sel_year and sel_tokei and sel_kubun):
        # うまくひっかからない場合のフォールバック（name属性の推定）
        try:
            sel_tokei = sel_tokei or driver.find_element(By.NAME, "STATS")
        except: pass
        try:
            sel_kubun = sel_kubun or driver.find_element(By.NAME, "KUBUN")
        except: pass

    # 固定条件を先に設定
    select_by_text(sel_tokei, STATS_TEXT)  # 年齢5歳刻み
    select_by_text(sel_kubun, KUBUN_TEXT)  # 町丁別

    # ループで年度を切り替えつつ 4月 を選び、EXCEL出力
    for jp_year in JAPANESE_YEARS_APR:
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
            print(f"[WARN] {jp_year} {TARGET_MONTH_TEXT} のダウンロードを検出できませんでした。手動確認してください。")
        else:
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
            print(f"[OK] saved: {new_name.name}")

        # アンチブロック用の小休止
        time.sleep(1.0)

    driver.quit()
    print("DONE.")

if __name__ == "__main__":
    main()
