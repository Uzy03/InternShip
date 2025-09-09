# -*- coding: utf-8 -*-
# src/common/spatial.py
"""
空間ユーティリティ共通モジュール
- W行列の構築（重心データから）
- 空間ラグの適用
- ラグ対象列の自動検出
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Union
import warnings

def build_W_from_centroids(centroids_df: pd.DataFrame, 
                          town_col: str = "town",
                          lon_col: str = "lon", 
                          lat_col: str = "lat",
                          k_neighbors: int = 5,
                          distance_threshold: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
    """
    重心データから空間重み行列Wを構築
    
    Parameters:
    -----------
    centroids_df : pd.DataFrame
        重心データ（town, lon, lat列を含む）
    town_col : str
        町丁名の列名
    lon_col : str
        経度の列名
    lat_col : str
        緯度の列名
    k_neighbors : int
        近傍数（デフォルト5）
    distance_threshold : float, optional
        距離閾値（指定した場合、k_neighborsより優先）
    
    Returns:
    --------
    W : np.ndarray
        空間重み行列（行和正規化済み）
    towns_order : List[str]
        町丁名の順序（Wの行・列に対応）
    """
    # 必要な列の確認
    required_cols = [town_col, lon_col, lat_col]
    missing_cols = [col for col in required_cols if col not in centroids_df.columns]
    if missing_cols:
        raise ValueError(f"重心データに列不足: {missing_cols}")
    
    # データの準備
    coords = centroids_df[[lon_col, lat_col]].values
    towns = centroids_df[town_col].tolist()
    
    # 距離行列の計算
    distances = cdist(coords, coords, metric='euclidean')
    
    # 空間重み行列の構築
    W = np.zeros_like(distances)
    
    for i in range(len(towns)):
        # 自分自身の距離は無限大に設定
        distances[i, i] = np.inf
        
        if distance_threshold is not None:
            # 距離閾値による近傍選択
            neighbors = distances[i] <= distance_threshold
        else:
            # k近傍による近傍選択
            neighbor_indices = np.argsort(distances[i])[:k_neighbors]
            neighbors = np.zeros(len(towns), dtype=bool)
            neighbors[neighbor_indices] = True
        
        # 重みの設定（距離の逆数）
        if neighbors.any():
            weights = 1.0 / (distances[i] + 1e-6)  # ゼロ除算防止
            weights[~neighbors] = 0.0
            # 行和正規化
            if weights.sum() > 0:
                W[i] = weights / weights.sum()
    
    return W, towns

def detect_cols_to_lag(df: pd.DataFrame, 
                      patterns: Optional[List[str]] = None) -> List[str]:
    """
    ラグ対象列を自動検出
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    patterns : List[str], optional
        検出パターン（デフォルトは一般的なパターン）
    
    Returns:
    --------
    cols_to_lag : List[str]
        ラグ対象列のリスト
    """
    if patterns is None:
        patterns = [
            r'pop_rate.*',  # 人口率関連
            r'event_.*',    # イベント関連
            r'.*_inc_.*',   # 増加関連
            r'.*_dec_.*',   # 減少関連
            r'.*_h[1-9]',   # ホライズン関連
            r'exp_.*',      # 期待効果関連
            r'foreign_.*',  # 外国人関連
        ]
    
    cols_to_lag = []
    for col in df.columns:
        # 数値列のみ対象
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # パターンマッチング
        for pattern in patterns:
            if pd.Series([col]).str.contains(pattern, regex=True).any():
                cols_to_lag.append(col)
                break
    
    return cols_to_lag

def apply_spatial_lags(df: pd.DataFrame, 
                      W: np.ndarray, 
                      towns_order: List[str],
                      cols_to_lag: List[str],
                      town_col: str = "town",
                      year_col: str = "year",
                      ring2: bool = False) -> pd.DataFrame:
    """
    空間ラグを適用
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    W : np.ndarray
        空間重み行列
    towns_order : List[str]
        町丁名の順序（Wの行・列に対応）
    cols_to_lag : List[str]
        ラグ対象列のリスト
    town_col : str
        町丁名の列名
    year_col : str
        年次の列名
    ring2 : bool
        ring2_*列も生成するかどうか
    
    Returns:
    --------
    df_with_lags : pd.DataFrame
        空間ラグ列が追加されたデータフレーム
    """
    result = df.copy()
    
    # 町丁名の正規化関数
    def normalize_town_name(town_name: str) -> str:
        """町丁名を正規化"""
        if pd.isna(town_name):
            return town_name
        
        town_name = str(town_name)
        
        # 漢数字を半角数字に変換
        kanji_to_num = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '十': '10'
        }
        
        for kanji, num in kanji_to_num.items():
            town_name = town_name.replace(kanji, num)
        
        # 全角数字を半角数字に変換
        town_name = town_name.replace('０', '0').replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4')
        town_name = town_name.replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9')
        
        return town_name
    
    # 町丁名のマッピング（正規化後 -> インデックス）
    town_to_idx = {}
    for i, town in enumerate(towns_order):
        normalized = normalize_town_name(town)
        town_to_idx[normalized] = i
    
    # 各年次で空間ラグを計算
    for year in sorted(df[year_col].unique()):
        year_data = df[df[year_col] == year].copy()
        
        if len(year_data) == 0:
            continue
        
        # 年次データの町丁名を正規化
        year_data['_normalized_town'] = year_data[town_col].apply(normalize_town_name)
        
        # 各町丁について空間ラグを計算
        for _, row in year_data.iterrows():
            town = row['_normalized_town']
            
            if town not in town_to_idx:
                continue
            
            town_idx = town_to_idx[town]
            
            # 各ラグ対象列について空間ラグを計算
            for col in cols_to_lag:
                if col not in year_data.columns:
                    continue
                
                # ring1_*列の計算
                ring1_col = f"ring1_{col}"
                if ring1_col not in result.columns:
                    result[ring1_col] = np.nan
                
                # 近傍の値を取得
                neighbor_values = []
                for j, neighbor_town in enumerate(towns_order):
                    if W[town_idx, j] > 0:  # 近傍関係がある場合
                        neighbor_row = year_data[year_data['_normalized_town'] == neighbor_town]
                        if len(neighbor_row) > 0 and not pd.isna(neighbor_row[col].iloc[0]):
                            neighbor_values.append(neighbor_row[col].iloc[0])
                
                if neighbor_values:
                    # 重み付き平均
                    weights = W[town_idx, [town_to_idx[t] for t in year_data['_normalized_town'].unique() if t in town_to_idx]]
                    weights = weights[weights > 0]
                    if len(weights) == len(neighbor_values):
                        ring1_value = np.average(neighbor_values, weights=weights)
                    else:
                        ring1_value = np.mean(neighbor_values)
                    
                    # 該当行に値を設定
                    mask = (result[town_col] == row[town_col]) & (result[year_col] == year)
                    result.loc[mask, ring1_col] = ring1_value
                
                # ring2_*列の計算（オプション）
                if ring2:
                    ring2_col = f"ring2_{col}"
                    if ring2_col not in result.columns:
                        result[ring2_col] = np.nan
                    
                    # より遠い近傍の値を取得（簡易実装）
                    # 実際の実装では、距離に基づいてring2を定義する必要がある
                    pass
    
    # 正規化用の一時列を削除
    if '_normalized_town' in result.columns:
        result = result.drop('_normalized_town', axis=1)
    
    return result

def calculate_spatial_lags_simple(df: pd.DataFrame, 
                                 centroids_df: pd.DataFrame,
                                 cols_to_lag: List[str],
                                 town_col: str = "town",
                                 year_col: str = "year",
                                 k_neighbors: int = 5) -> pd.DataFrame:
    """
    高速版空間ラグ計算（重心データから直接計算）
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    centroids_df : pd.DataFrame
        重心データ
    cols_to_lag : List[str]
        ラグ対象列のリスト
    town_col : str
        町丁名の列名
    year_col : str
        年次の列名
    k_neighbors : int
        近傍数
    
    Returns:
    --------
    df_with_lags : pd.DataFrame
        空間ラグ列が追加されたデータフレーム
    """
    result = df.copy()
    
    # 重心データの準備
    coords = centroids_df[['lon', 'lat']].values
    towns = centroids_df[town_col].tolist()
    
    # 距離行列の計算
    print(f"[spatial] 距離行列を計算中... ({len(towns)}x{len(towns)})")
    distances = cdist(coords, coords, metric='euclidean')
    
    # 町丁名の正規化関数
    def normalize_town_name(town_name: str) -> str:
        if pd.isna(town_name):
            return town_name
        
        town_name = str(town_name)
        kanji_to_num = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '十': '10'
        }
        
        for kanji, num in kanji_to_num.items():
            town_name = town_name.replace(kanji, num)
        
        town_name = town_name.replace('０', '0').replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4')
        town_name = town_name.replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9')
        
        return town_name
    
    # 町丁名のマッピング
    town_to_idx = {}
    for i, town in enumerate(towns):
        normalized = normalize_town_name(town)
        town_to_idx[normalized] = i
    
    # 近傍インデックスの事前計算
    print(f"[spatial] 近傍インデックスを事前計算中...")
    neighbor_indices = {}
    neighbor_weights = {}
    
    for i in range(len(towns)):
        distances_to_town = distances[i]
        # 自分を除く近傍の選択
        sorted_indices = np.argsort(distances_to_town)[1:k_neighbors+1]
        neighbor_indices[i] = sorted_indices
        
        # 重みの計算（距離の逆数）
        weights = 1.0 / (distances_to_town[sorted_indices] + 1e-6)
        weights = weights / weights.sum()
        neighbor_weights[i] = weights
    
    print(f"[spatial] 空間ラグを計算中...")
    
    # 各年次で空間ラグを計算
    years = sorted(df[year_col].unique())
    for year_idx, year in enumerate(years):
        if year_idx % 5 == 0:  # 進捗表示
            print(f"[spatial] 年次処理中: {year} ({year_idx+1}/{len(years)})")
        
        year_data = df[df[year_col] == year].copy()
        
        if len(year_data) == 0:
            continue
        
        # 年次データの町丁名を正規化
        year_data['_normalized_town'] = year_data[town_col].apply(normalize_town_name)
        
        # 年次データの町丁名とインデックスのマッピング
        year_town_to_idx = {}
        for idx, town in enumerate(year_data['_normalized_town']):
            if town in town_to_idx:
                year_town_to_idx[town] = idx
        
        # 各ラグ対象列について空間ラグを計算
        for col_idx, col in enumerate(cols_to_lag):
            if col not in year_data.columns:
                continue
            
            if col_idx % 10 == 0:  # 進捗表示
                print(f"[spatial] 列処理中: {col} ({col_idx+1}/{len(cols_to_lag)})")
            
            # ring1_*列の初期化
            ring1_col = f"ring1_{col}"
            if ring1_col not in result.columns:
                result[ring1_col] = np.nan
            
            # 各町丁について空間ラグを計算
            for _, row in year_data.iterrows():
                town = row['_normalized_town']
                
                if town not in town_to_idx:
                    continue
                
                town_idx = town_to_idx[town]
                neighbor_idx_list = neighbor_indices[town_idx]
                weights = neighbor_weights[town_idx]
                
                # 近傍の値を取得
                neighbor_values = []
                valid_weights = []
                
                for j, neighbor_idx in enumerate(neighbor_idx_list):
                    neighbor_town = towns[neighbor_idx]
                    neighbor_normalized = normalize_town_name(neighbor_town)
                    
                    if neighbor_normalized in year_town_to_idx:
                        neighbor_row_idx = year_town_to_idx[neighbor_normalized]
                        neighbor_value = year_data.iloc[neighbor_row_idx][col]
                        
                        if not pd.isna(neighbor_value):
                            neighbor_values.append(neighbor_value)
                            valid_weights.append(weights[j])
                
                if neighbor_values:
                    # 重み付き平均
                    if len(valid_weights) == len(neighbor_values):
                        ring1_value = np.average(neighbor_values, weights=valid_weights)
                    else:
                        ring1_value = np.mean(neighbor_values)
                    
                    # 該当行に値を設定
                    mask = (result[town_col] == row[town_col]) & (result[year_col] == year)
                    result.loc[mask, ring1_col] = ring1_value
    
    # 正規化用の一時列を削除
    if '_normalized_town' in result.columns:
        result = result.drop('_normalized_town', axis=1)
    
    print(f"[spatial] 空間ラグ計算完了")
    return result
