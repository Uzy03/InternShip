import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import glob

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.common.io import resolve_path, ensure_parent, get_logger, set_seed
from src.common.utils import normalize_town


def validate_required_columns(df: pd.DataFrame) -> None:
    """必須列の存在チェック"""
    if '町丁名' not in df.columns:
        raise ValueError("必須列'町丁名'が不足しています")
    
    # 年列の存在チェック（1998_総人口のような形式）
    year_columns = [col for col in df.columns if '_総人口' in col]
    if not year_columns:
        raise ValueError("年別総人口列が見つかりません")


def extract_year_from_column(column_name: str) -> int:
    """列名から年を抽出（例：1998_総人口 → 1998）"""
    return int(column_name.split('_')[0])


def normalize_age_data(df: pd.DataFrame) -> pd.DataFrame:
    """5歳階級データを総人口に合わせて正規化"""
    # 5歳階級列を特定
    age_columns = [col for col in df.columns if any(age in col for age in ['0〜4歳', '5〜9歳', '10〜14歳', '15〜19歳', '20〜24歳', '25〜29歳', '30〜34歳', '35〜39歳', '40〜44歳', '45〜49歳', '50〜54歳', '55〜59歳', '60〜64歳', '65〜69歳', '70〜74歳', '75〜79歳', '80〜84歳', '85〜89歳', '90〜94歳', '95〜99歳', '100歳以上'])]
    
    df_normalized = df.copy()
    
    for _, row in df.iterrows():
        # 各年のデータを処理
        for year_col in [col for col in df.columns if '_総人口' in col]:
            year = extract_year_from_column(year_col)
            total_pop = row[year_col]
            
            # その年の5歳階級列を取得
            year_age_columns = [col for col in age_columns if col.startswith(f"{year}_")]
            
            if year_age_columns and total_pop > 0:
                # 5歳階級の合計を計算
                age_sum = sum(row[col] for col in year_age_columns if pd.notna(row[col]))
                
                if age_sum > total_pop:
                    # 正規化係数を計算
                    ratio = total_pop / age_sum
                    
                    # 各年齢階級を正規化
                    for col in year_age_columns:
                        if pd.notna(row[col]):
                            df_normalized.loc[_, col] = row[col] * ratio
    
    return df_normalized


def transform_to_panel(df: pd.DataFrame) -> pd.DataFrame:
    """横長データを縦長パネルデータに変換"""
    # 町丁名列を正規化
    df['town'] = df['町丁名'].apply(normalize_town)
    
    # 年別総人口列を抽出
    pop_columns = [col for col in df.columns if '_総人口' in col]
    
    # 年別男性列を抽出
    male_columns = [col for col in df.columns if '_男性' in col]
    
    # 年別女性列を抽出
    female_columns = [col for col in df.columns if '_女性' in col]
    
    # パネルデータ用のリスト
    panel_data = []
    
    for _, row in df.iterrows():
        town = row['town']
        
        for pop_col in pop_columns:
            year = extract_year_from_column(pop_col)
            pop_total = row[pop_col]
            
            # 対応する男性・女性列を探す
            male_col = f"{year}_男性"
            female_col = f"{year}_女性"
            
            male_pop = row.get(male_col, None)
            female_pop = row.get(female_col, None)
            
            # 年齢階級列を探す（5歳階級など）
            age_columns = [col for col in df.columns if col.startswith(f"{year}_") and col not in [pop_col, male_col, female_col]]
            age_data = {}
            for age_col in age_columns:
                age_key = age_col.replace(f"{year}_", "")
                age_data[age_key] = row[age_col]
            
            # 行データを作成
            row_data = {
                'town': town,
                'year': year,
                'pop_total': pop_total,
                'male': male_pop,
                'female': female_pop
            }
            row_data.update(age_data)
            
            panel_data.append(row_data)
    
    return pd.DataFrame(panel_data)


def load_consistent_data_files(data_dir: str) -> pd.DataFrame:
    """100_Percent_Consistent_DataのCSVファイルを読み込んで統合"""
    csv_files = glob.glob(f"{data_dir}/*_nan_filled.csv")
    csv_files.sort()  # ファイル名でソート
    
    all_data = []
    
    for file_path in csv_files:
        # ファイル名から年を抽出（例：H10-04_consistent.csv → 1998）
        filename = Path(file_path).stem
        year_str = filename.split('-')[0]  # H10
        
        # 和暦を西暦に変換
        if year_str.startswith('H'):
            year = int(year_str[1:]) + 1988  # H10 → 1998
        elif year_str.startswith('R'):
            year = int(year_str[1:]) + 2018  # R02 → 2020
        else:
            continue
        
        # CSVファイルを読み込み
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 年別の列名に変換
        df_renamed = df.copy()
        for col in df.columns:
            if col != '町丁名':
                df_renamed = df_renamed.rename(columns={col: f"{year}_{col}"})
        
        all_data.append(df_renamed)
    
    # 全てのデータを町丁名で結合
    if all_data:
        merged_df = all_data[0]
        for df in all_data[1:]:
            merged_df = merged_df.merge(df, on='町丁名', how='outer')
        
        return merged_df
    else:
        raise ValueError("CSVファイルが見つかりませんでした")


def validate_year_column(df: pd.DataFrame) -> None:
    """year列の妥当性チェック"""
    # year列は既にintになっているはず
    if df['year'].isna().any():
        raise ValueError("year列に欠損値が含まれています")


def check_duplicates(df: pd.DataFrame) -> None:
    """(town,year)の重複チェック"""
    duplicates = df.duplicated(subset=['town', 'year'], keep=False)
    if duplicates.any():
        duplicate_rows = df[duplicates].sort_values(['town', 'year'])
        error_msg = f"(town,year)の重複が検出されました:\n{duplicate_rows.to_string()}"
        raise ValueError(error_msg)


def is_merged_town_before_2009(town: str, year: int) -> bool:
    """合併3町（城南/富合/植木）の2009年以前かどうか判定"""
    merged_towns = ['城南町', '富合町', '植木町']
    return town in merged_towns and year <= 2009


def build_panel() -> None:
    """パネルデータ構築のメイン処理"""
    logger = None
    try:
        # シード設定
        set_seed(42)
        
        # ロガー設定
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = get_logger(run_id)
        logger.info("パネルデータ構築を開始します")
        
        # パス設定（ルートパスからの実行を想定）
        input_dir = Path("100_Percent_Consistent_Data/CSV_Files")
        output_file = Path("subject3/data/processed/panel_raw.csv")
        
        logger.info(f"入力ディレクトリ: {input_dir}")
        logger.info(f"出力ファイル: {output_file}")
        
        # 入力ディレクトリ存在チェック
        if not input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
        
        # 100_Percent_Consistent_DataのCSVファイルを読み込み
        logger.info("100_Percent_Consistent_DataのCSVファイルを読み込んでいます...")
        df = load_consistent_data_files(str(input_dir))
        logger.info(f"読み込み完了: {len(df)}行")
        
        # 必須列チェック
        logger.info("必須列の存在をチェックしています...")
        validate_required_columns(df)
        logger.info("必須列チェック完了")
        
        # データの正規化
        logger.info("5歳階級データを正規化しています...")
        df = normalize_age_data(df)
        logger.info("正規化完了")
        
        # 横長データを縦長パネルデータに変換
        logger.info("データ形式を変換しています...")
        df_panel = transform_to_panel(df)
        logger.info(f"変換完了: {len(df_panel)}行")
        
        # year列の妥当性チェック
        logger.info("year列の妥当性をチェックしています...")
        validate_year_column(df_panel)
        logger.info("year列チェック完了")
        
        # 重複チェック
        logger.info("重複チェックを実行しています...")
        check_duplicates(df_panel)
        logger.info("重複チェック完了")
        
        # 合併3町の前処理（触らない）
        logger.info("合併3町の前処理を実行しています...")
        # この時点では何もしない（NaNのまま保持）
        logger.info("合併3町前処理完了")
        
        # 並び替え: town ASC, year ASC
        logger.info("データを並び替えています...")
        df_panel = df_panel.sort_values(['town', 'year'])
        logger.info("並び替え完了")
        
        # 出力ディレクトリ作成
        ensure_parent(output_file)
        
        # 保存
        logger.info("結果を保存しています...")
        df_panel.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"保存完了: {output_file}")
        
        # 結果サマリー
        logger.info("=== 処理結果サマリー ===")
        logger.info(f"総行数: {len(df_panel)}")
        logger.info(f"町丁数: {df_panel['town'].nunique()}")
        logger.info(f"年範囲: {df_panel['year'].min()} - {df_panel['year'].max()}")
        logger.info(f"列数: {len(df_panel.columns)}")
        logger.info("列名: " + ", ".join(df_panel.columns.tolist()))
        
        # 5歳階級列の確認
        age_columns = [col for col in df_panel.columns if any(age in col for age in ['0〜4歳', '5〜9歳', '10〜14歳', '15〜19歳', '20〜24歳', '25〜29歳', '30〜34歳', '35〜39歳', '40〜44歳', '45〜49歳', '50〜54歳', '55〜59歳', '60〜64歳', '65〜69歳', '70〜74歳', '75〜79歳', '80〜84歳', '85〜89歳', '90〜94歳', '95〜99歳', '100歳以上'])]
        logger.info(f"5歳階級列数: {len(age_columns)}")
        logger.info("5歳階級列: " + ", ".join(age_columns))
        
        # データ整合性チェック
        logger.info("データ整合性をチェックしています...")
        df_panel['age_sum'] = df_panel[age_columns].sum(axis=1)
        df_panel['difference'] = df_panel['age_sum'] - df_panel['pop_total']
        problematic = df_panel[abs(df_panel['difference']) > 0.1]  # 0.1以上の差を問題とする
        logger.info(f"整合性の問題のある行数: {len(problematic)}/{len(df_panel)} ({len(problematic)/len(df_panel)*100:.2f}%)")
        
        logger.info("パネルデータ構築が正常に完了しました")
        
    except Exception as e:
        if logger:
            logger.error(f"エラーが発生しました: {e}")
        else:
            print(f"ロガー初期化前にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    build_panel()
