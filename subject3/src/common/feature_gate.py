# -*- coding: utf-8 -*-
# src/common/feature_gate.py
"""
特徴量ゲート機能
- 期待効果（exp_*）を完全に除外
- 空間ラグ（ring1_*）は許容列のみ残す
- 学習時と推論時で厳密に特徴量を一致させる
"""
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

# 除外パターン（期待効果と手動人数を完全除外、空間ラグは含める）
EXCLUDE_PATTERNS = [
    r'^exp_',                    # 期待効果関連（直接効果のみ除外）
    r'^manual_people_',         # 手動人数
    r'^ring1_manual_people_',   # 空間ラグの手動人数
    r'^ring2_manual_people_',   # 空間ラグの手動人数
]

# 許容する空間ラグパターン（ring1_* のうち、manual 以外）
ALLOWED_RING_PATTERNS = [
    r'^ring1_(?!manual_people_)',  # ring1_* のうち manual_people_ 以外（exp_も含む）
    r'^ring2_(?!manual_people_)',  # ring2_* のうち manual_people_ 以外（exp_も含む）
]

def drop_excluded_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    除外パターンに該当する列を削除
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    exclude_patterns : List[str], optional
        除外パターンのリスト（デフォルトは EXCLUDE_PATTERNS）
    
    Returns:
    --------
    df_kept : pd.DataFrame
        除外後のデータフレーム
    removed_cols : List[str]
        削除された列のリスト
    """
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS
    
    removed_cols = []
    kept_cols = []
    
    for col in df.columns:
        should_exclude = False
        for pattern in exclude_patterns:
            if re.match(pattern, col):
                should_exclude = True
                break
        
        if should_exclude:
            removed_cols.append(col)
        else:
            kept_cols.append(col)
    
    df_kept = df[kept_cols].copy()
    
    return df_kept, removed_cols

def select_feature_columns(df: pd.DataFrame, 
                          include_regex_list: List[str] = None,
                          exclude_patterns: List[str] = None) -> List[str]:
    """
    特徴量列を選択（許容パターンを含み、除外パターンを除外）
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    include_regex_list : List[str], optional
        含めるパターンのリスト（デフォルトは ALLOWED_RING_PATTERNS）
    exclude_patterns : List[str], optional
        除外パターンのリスト（デフォルトは EXCLUDE_PATTERNS）
    
    Returns:
    --------
    feature_cols : List[str]
        選択された特徴量列のリスト
    """
    if include_regex_list is None:
        include_regex_list = ALLOWED_RING_PATTERNS
    
    if exclude_patterns is None:
        exclude_patterns = EXCLUDE_PATTERNS
    
    feature_cols = []
    
    for col in df.columns:
        # 除外パターンに該当する場合はスキップ
        should_exclude = False
        for pattern in exclude_patterns:
            if re.match(pattern, col):
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        # 数値列のみ対象
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # 含めるパターンに該当する場合は追加
        should_include = False
        for pattern in include_regex_list:
            if re.match(pattern, col):
                should_include = True
                break
        
        # 含めるパターンに該当しない場合でも、基本的な特徴量は含める
        if not should_include:
            # 基本的な特徴量パターン（既存のL4特徴量など）
            basic_patterns = [
                r'^pop_total$',
                r'^lag_d[12]$',
                r'^ma2_delta$',
                r'^town_(ma5|std5|trend5)$',
                r'^macro_(delta|ma3|shock|excl)$',
                r'^era_(pre2013|post2009|post2013|covid|post2022)$',
                r'^foreign_(population|change|pct_change|log|ma3)',
                r'^foreign_.*_(covid|post2022)$',
            ]
            
            for pattern in basic_patterns:
                if re.match(pattern, col):
                    should_include = True
                    break
        
        if should_include:
            feature_cols.append(col)
    
    return sorted(feature_cols)  # 列順を固定

def save_feature_list(cols: List[str], path: str) -> None:
    """
    特徴量リストをJSONファイルに保存
    
    Parameters:
    -----------
    cols : List[str]
        特徴量列のリスト
    path : str
        保存先パス
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "feature_columns": cols,
        "n_features": len(cols),
        "exclude_patterns": EXCLUDE_PATTERNS,
        "allowed_ring_patterns": ALLOWED_RING_PATTERNS
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[feature_gate] 特徴量リストを保存: {path} ({len(cols)}列)")

def load_feature_list(path: str) -> List[str]:
    """
    特徴量リストをJSONファイルから読み込み
    
    Parameters:
    -----------
    path : str
        ファイルパス
    
    Returns:
    --------
    feature_cols : List[str]
        特徴量列のリスト
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"特徴量リストファイルが見つかりません: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["feature_columns"]

def align_features_for_inference(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    推論用に特徴量を整列（学習時と同じ列順・欠損0埋め）
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    feature_list : List[str]
        学習時に使用された特徴量リスト
    
    Returns:
    --------
    aligned_df : pd.DataFrame
        整列されたデータフレーム
    """
    aligned_df = pd.DataFrame(index=df.index)
    
    # 学習時の特徴量リストに合わせて列を整列
    for col in feature_list:
        if col in df.columns:
            # 存在する列はそのまま使用
            aligned_df[col] = df[col]
        else:
            # 存在しない列は0で埋める
            aligned_df[col] = 0.0
            print(f"[feature_gate] 欠損列を0で埋め: {col}")
    
    # 余分な列は捨てる（feature_listにない列）
    extra_cols = set(df.columns) - set(feature_list)
    if extra_cols:
        print(f"[feature_gate] 余分な列を除外: {sorted(extra_cols)}")
    
    # 列順を学習時と同じにする
    aligned_df = aligned_df[feature_list]
    
    # 欠損値を0で埋める
    aligned_df = aligned_df.fillna(0.0)
    
    # 無限値を0で埋める
    aligned_df = aligned_df.replace([np.inf, -np.inf], 0.0)
    
    # 元のDataFrameの非特徴量列（year, town等）を保持
    non_feature_cols = ['year', 'town', 'town_id']
    for col in non_feature_cols:
        if col in df.columns and col not in feature_list:
            aligned_df[col] = df[col]
    
    return aligned_df

def get_feature_statistics(df: pd.DataFrame, feature_list: List[str]) -> dict:
    """
    特徴量の統計情報を取得
    
    Parameters:
    -----------
    df : pd.DataFrame
        対象データフレーム
    feature_list : List[str]
        特徴量リスト
    
    Returns:
    --------
    stats : dict
        統計情報
    """
    stats = {
        "total_columns": len(df.columns),
        "feature_columns": len(feature_list),
        "missing_features": [],
        "extra_features": [],
        "feature_coverage": 0.0
    }
    
    # 欠損特徴量
    for col in feature_list:
        if col not in df.columns:
            stats["missing_features"].append(col)
    
    # 余分な特徴量
    for col in df.columns:
        if col not in feature_list:
            stats["extra_features"].append(col)
    
    # カバレッジ
    if feature_list:
        stats["feature_coverage"] = (len(feature_list) - len(stats["missing_features"])) / len(feature_list)
    
    return stats
