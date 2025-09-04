"""
Feature Engineering for Population Anomaly Detection

This module builds derived features from raw population data including:
- Basic population changes (delta, growth rates)
- Age-specific features (if available)
- Momentum, volatility, acceleration features
- Trend analysis
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_age_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """
    Detect age columns in the dataframe and return standardized names.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (age_columns, age_mapping)
    """
    age_columns = []
    age_mapping = {}
    
    # Standard age column patterns
    standard_patterns = [
        r'age_(\d{2})_(\d{2})',  # age_00_04, age_05_09, etc.
        r'age_(\d{2})p',         # age_85p, age_100p
    ]
    
    # Japanese age column patterns
    japanese_patterns = [
        r'(\d+)[〜～\-](\d+)歳',  # 0〜4歳, 0～4歳, 0-4歳
        r'(\d+)歳以上',           # 85歳以上
    ]
    
    for col in df.columns:
        # Check standard patterns
        for pattern in standard_patterns:
            match = re.match(pattern, col)
            if match:
                age_columns.append(col)
                age_mapping[col] = col  # Keep as is
                break
        
        # Check Japanese patterns
        for pattern in japanese_patterns:
            match = re.match(pattern, col)
            if match:
                age_columns.append(col)
                # Convert to standard format
                if '以上' in col:
                    age_start = match.group(1)
                    std_name = f"age_{age_start.zfill(2)}p"
                else:
                    age_start = match.group(1).zfill(2)
                    age_end = match.group(2).zfill(2)
                    std_name = f"age_{age_start}_{age_end}"
                age_mapping[col] = std_name
                break
    
    return age_columns, age_mapping


def rescale_age_columns(df: pd.DataFrame, age_columns: List[str], 
                       age_mapping: Dict[str, str], 
                       age_rescale_threshold: float = 0.1) -> pd.DataFrame:
    """
    Rescale age columns to match total population.
    
    Args:
        df: Input dataframe
        age_columns: List of age column names
        age_mapping: Mapping from original to standard names
        age_rescale_threshold: Threshold for logging mismatches
        
    Returns:
        Dataframe with rescaled age columns
    """
    df = df.copy()
    
    for idx, row in df.iterrows():
        if 'pop_total' not in row or pd.isna(row['pop_total']):
            continue
            
        # Calculate sum of age columns
        age_sum = sum(row[col] for col in age_columns if pd.notna(row[col]))
        
        if age_sum > 0:
            # Calculate rescale factor
            rescale_factor = row['pop_total'] / age_sum
            
            # Rescale age columns
            for col in age_columns:
                if pd.notna(row[col]):
                    df.at[idx, col] = row[col] * rescale_factor
            
            # Log mismatches
            mismatch = abs(1 - row['pop_total'] / age_sum)
            if mismatch > age_rescale_threshold:
                logger.warning(f"Age mismatch at {idx}: {mismatch:.3f}")
    
    return df


def create_age_features(df: pd.DataFrame, age_columns: List[str], 
                       age_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create age-specific features (shares and differences).
    
    Args:
        df: Input dataframe
        age_columns: List of age column names
        age_mapping: Mapping from original to standard names
        
    Returns:
        Dataframe with age features added
    """
    df = df.copy()
    
    # Create age share features
    for col in age_columns:
        std_name = age_mapping[col]
        share_col = f"age_share_{std_name.replace('age_', '')}"
        df[share_col] = df[col] / df['pop_total'].replace(0, 1)
    
    # Create age difference features
    df = df.sort_values(['town', 'year'])
    for col in age_columns:
        std_name = age_mapping[col]
        share_col = f"age_share_{std_name.replace('age_', '')}"
        diff_col = f"d_age_share_{std_name.replace('age_', '')}"
        df[diff_col] = df.groupby('town')[share_col].diff()
    
    # Create summary age features
    youth_cols = [col for col in age_columns if any(f"age_{i:02d}_{j:02d}" in age_mapping[col] 
                   for i, j in [(0, 4), (5, 9), (10, 14)])]
    work_cols = [col for col in age_columns if any(f"age_{i:02d}_{j:02d}" in age_mapping[col] 
                  for i, j in [(15, 19), (20, 24), (25, 29), (30, 34), (35, 39), 
                               (40, 44), (45, 49), (50, 54), (55, 59), (60, 64)])]
    elderly_cols = [col for col in age_columns if any(f"age_{i:02d}_{j:02d}" in age_mapping[col] 
                     for i, j in [(65, 69), (70, 74), (75, 79), (80, 84)]) or 
                     'age_85p' in age_mapping[col] or 'age_100p' in age_mapping[col]]
    
    df['youth_share'] = df[youth_cols].sum(axis=1) / df['pop_total'].replace(0, 1)
    df['work_share'] = df[work_cols].sum(axis=1) / df['pop_total'].replace(0, 1)
    df['elderly_share'] = df[elderly_cols].sum(axis=1) / df['pop_total'].replace(0, 1)
    df['dependency_ratio'] = (df['youth_share'] + df['elderly_share']) / df['work_share'].replace(0, 1e-6)
    
    return df


def create_general_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create general population features (regardless of age data availability).
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with general features added
    """
    df = df.copy()
    df = df.sort_values(['town', 'year'])
    
    # Basic population changes
    df['delta'] = df.groupby('town')['pop_total'].diff()
    df['growth_pct'] = df['delta'] / df.groupby('town')['pop_total'].shift().replace(0, 1.0)
    df['growth_log'] = np.log1p(df['pop_total']) - np.log1p(df.groupby('town')['pop_total'].shift())
    
    # City-level features
    city_pop = df.groupby('year')['pop_total'].sum()
    city_growth_log = np.log1p(city_pop) - np.log1p(city_pop.shift())
    df['city_pop'] = df['year'].map(city_pop)
    df['city_growth_log'] = df['year'].map(city_growth_log)
    df['growth_adj_log'] = df['growth_log'] - df['city_growth_log']
    
    # Momentum and volatility features
    df['delta_roll_mean_3'] = df.groupby('town')['delta'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df['delta_roll_std_3'] = df.groupby('town')['delta'].rolling(3, min_periods=1).std().reset_index(0, drop=True)
    df['delta_accel'] = df.groupby('town')['delta'].diff()
    df['delta_cum3'] = df.groupby('town')['delta'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    
    # Trend slope (3-year OLS)
    def calc_trend_slope(group):
        if len(group) < 2:
            return pd.Series([np.nan] * len(group), index=group.index)
        
        slopes = []
        for i in range(len(group)):
            if i < 2:
                slopes.append(np.nan)
            else:
                y = group.iloc[i-2:i+1].values
                x = np.arange(3)
                if len(y) == 3 and not np.any(pd.isna(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=group.index)
    
    df['trend_slope_3y'] = df.groupby('town')['pop_total'].apply(calc_trend_slope).reset_index(0, drop=True)
    
    return df


def build_features(input_path: str = "subject3/data/processed/panel_raw.csv",
                   output_path: str = "subject3/data/processed/features_panel.csv",
                   age_rescale_threshold: float = 0.1,
                   run_id: str = "default") -> pd.DataFrame:
    """
    Build features for anomaly detection from raw panel data.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        age_rescale_threshold: Threshold for age rescale logging
        run_id: Run ID for logging
        
    Returns:
        Dataframe with features
    """
    logger.info(f"Building features from {input_path}")
    
    # Read input data
    df = pd.read_csv(input_path)
    
    # Validate required columns
    required_cols = ['town', 'year', 'pop_total']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Detect age columns
    age_columns, age_mapping = detect_age_columns(df)
    logger.info(f"Detected {len(age_columns)} age columns: {age_columns}")
    
    # Rescale age columns if present
    if age_columns:
        df = rescale_age_columns(df, age_columns, age_mapping, age_rescale_threshold)
        
        # Log age mismatches
        log_dir = Path(f"logs/{run_id}")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "age_mismatch.log", "w") as f:
            f.write("Age rescale mismatches logged during feature building\n")
    
    # Create general features
    df = create_general_features(df)
    
    # Create age features if available
    if age_columns:
        df = create_age_features(df, age_columns, age_mapping)
    
    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Features saved to {output_path}")
    logger.info(f"Final dataframe shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    build_features()
