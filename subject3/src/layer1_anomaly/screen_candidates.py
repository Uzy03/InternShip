"""
Candidate Screening for Manual Investigation

This module screens population changes to identify candidates for manual investigation
using hybrid rules based on percentage and absolute changes.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_screening_rules(df: pd.DataFrame, 
                         surge_pct_min: float = 0.5,
                         surge_abs_min: int = 100,
                         surge_small_abs_min: int = 50,
                         large_up_pct_min: float = 0.2,
                         large_up_abs_min: int = 50,
                         large_up_small_abs_min: int = 25,
                         large_down_pct_max: float = -0.2,
                         large_down_abs_min: int = -50,
                         large_down_small_abs_min: int = -25,
                         crash_pct_max: float = -0.5,
                         crash_abs_min: int = -100,
                         crash_small_abs_min: int = -50,
                         jump_exception_abs: int = 200,
                         small_prev_threshold: int = 100) -> pd.DataFrame:
    """
    Apply screening rules to identify major population changes.
    
    Args:
        df: Input dataframe with population data
        surge_pct_min: Minimum percentage for surge classification
        surge_abs_min: Minimum absolute change for surge classification
        surge_small_abs_min: Minimum absolute change for small population surge
        large_up_pct_min: Minimum percentage for large increase classification
        large_up_abs_min: Minimum absolute change for large increase classification
        large_up_small_abs_min: Minimum absolute change for small population large increase
        large_down_pct_max: Maximum percentage for large decrease classification
        large_down_abs_min: Minimum absolute change for large decrease classification
        large_down_small_abs_min: Minimum absolute change for small population large decrease
        crash_pct_max: Maximum percentage for crash classification
        crash_abs_min: Minimum absolute change for crash classification
        crash_small_abs_min: Minimum absolute change for small population crash
        jump_exception_abs: Absolute threshold for jump exception
        small_prev_threshold: Threshold for small population special case
        
    Returns:
        Dataframe with screening results
    """
    df = df.copy()
    df = df.sort_values(['town', 'year'])
    
    # Calculate previous year population
    df['pop_prev'] = df.groupby('town')['pop_total'].shift()
    
    # Calculate changes
    df['delta'] = df['pop_total'] - df['pop_prev']
    df['growth_pct'] = df['delta'] / df['pop_prev'].replace(0, 1.0)
    
    # Initialize classification
    df['class'] = 'normal'
    df['reason'] = None
    
    # Apply rules
    for idx, row in df.iterrows():
        if pd.isna(row['pop_prev']) or pd.isna(row['delta']):
            continue
            
        prev = row['pop_prev']
        delta = row['delta']
        growth_pct = row['growth_pct']
        
        # Jump exception (highest priority)
        if abs(delta) >= jump_exception_abs:
            if delta > 0:
                df.at[idx, 'class'] = 'jump_exception_increase'
            else:
                df.at[idx, 'class'] = 'jump_exception_decrease'
            df.at[idx, 'reason'] = f'absolute_change_{abs(delta)}'
            continue
        
        # Small population special case
        is_small_pop = prev < small_prev_threshold
        
        # Surge (large increase)
        if growth_pct >= surge_pct_min and delta >= surge_abs_min:
            df.at[idx, 'class'] = 'surge'
            df.at[idx, 'reason'] = f'pct_{growth_pct:.3f}_abs_{delta}'
        elif is_small_pop and delta >= surge_small_abs_min:
            df.at[idx, 'class'] = 'surge'
            df.at[idx, 'reason'] = f'small_pop_abs_{delta}'
        
        # Large increase
        elif growth_pct >= large_up_pct_min and delta >= large_up_abs_min:
            df.at[idx, 'class'] = 'large_increase'
            df.at[idx, 'reason'] = f'pct_{growth_pct:.3f}_abs_{delta}'
        elif is_small_pop and delta >= large_up_small_abs_min:
            df.at[idx, 'class'] = 'large_increase'
            df.at[idx, 'reason'] = f'small_pop_abs_{delta}'
        
        # Large decrease
        elif growth_pct <= large_down_pct_max and delta <= large_down_abs_min:
            df.at[idx, 'class'] = 'large_decrease'
            df.at[idx, 'reason'] = f'pct_{growth_pct:.3f}_abs_{delta}'
        elif is_small_pop and delta <= large_down_small_abs_min:
            df.at[idx, 'class'] = 'large_decrease'
            df.at[idx, 'reason'] = f'small_pop_abs_{delta}'
        
        # Crash (large decrease)
        elif growth_pct <= crash_pct_max and delta <= crash_abs_min:
            df.at[idx, 'class'] = 'crash'
            df.at[idx, 'reason'] = f'pct_{growth_pct:.3f}_abs_{delta}'
        elif is_small_pop and delta <= crash_small_abs_min:
            df.at[idx, 'class'] = 'crash'
            df.at[idx, 'reason'] = f'small_pop_abs_{delta}'
    
    return df


def screen_candidates(input_path: str = "subject3/data/processed/features_panel.csv",
                     output_path: str = "subject3/data/processed/major_population_changes.csv",
                     surge_pct_min: float = 0.5,
                     surge_abs_min: int = 100,
                     surge_small_abs_min: int = 50,
                     large_up_pct_min: float = 0.2,
                     large_up_abs_min: int = 50,
                     large_up_small_abs_min: int = 25,
                     large_down_pct_max: float = -0.2,
                     large_down_abs_min: int = -50,
                     large_down_small_abs_min: int = -25,
                     crash_pct_max: float = -0.5,
                     crash_abs_min: int = -100,
                     crash_small_abs_min: int = -50,
                     jump_exception_abs: int = 200,
                     small_prev_threshold: int = 100) -> pd.DataFrame:
    """
    Screen population changes to identify candidates for manual investigation.
    
    Args:
        input_path: Path to features CSV file
        output_path: Path to output candidates CSV file
        surge_pct_min: Minimum percentage for surge classification
        surge_abs_min: Minimum absolute change for surge classification
        surge_small_abs_min: Minimum absolute change for small population surge
        large_up_pct_min: Minimum percentage for large increase classification
        large_up_abs_min: Minimum absolute change for large increase classification
        large_up_small_abs_min: Minimum absolute change for small population large increase
        large_down_pct_max: Maximum percentage for large decrease classification
        large_down_abs_min: Minimum absolute change for large decrease classification
        large_down_small_abs_min: Minimum absolute change for small population large decrease
        crash_pct_max: Maximum percentage for crash classification
        crash_abs_min: Minimum absolute change for crash classification
        crash_small_abs_min: Minimum absolute change for small population crash
        jump_exception_abs: Absolute threshold for jump exception
        small_prev_threshold: Threshold for small population special case
        
    Returns:
        Dataframe with screening results
    """
    logger.info(f"Screening candidates from {input_path}")
    
    # Read data
    df = pd.read_csv(input_path)
    
    # Validate required columns
    required_cols = ['town', 'year', 'pop_total']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Apply screening rules
    result_df = apply_screening_rules(
        df,
        surge_pct_min=surge_pct_min,
        surge_abs_min=surge_abs_min,
        surge_small_abs_min=surge_small_abs_min,
        large_up_pct_min=large_up_pct_min,
        large_up_abs_min=large_up_abs_min,
        large_up_small_abs_min=large_up_small_abs_min,
        large_down_pct_max=large_down_pct_max,
        large_down_abs_min=large_down_abs_min,
        large_down_small_abs_min=large_down_small_abs_min,
        crash_pct_max=crash_pct_max,
        crash_abs_min=crash_abs_min,
        crash_small_abs_min=crash_small_abs_min,
        jump_exception_abs=jump_exception_abs,
        small_prev_threshold=small_prev_threshold
    )
    
    # Filter to only candidates (non-normal)
    candidates_df = result_df[result_df['class'] != 'normal'].copy()
    
    # Select output columns
    output_cols = ['town', 'year', 'pop_prev', 'pop_total', 'delta', 'growth_pct', 'class', 'reason']
    candidates_df = candidates_df[output_cols].copy()
    
    # Rename pop_total to pop_now for clarity
    candidates_df = candidates_df.rename(columns={'pop_total': 'pop_now'})
    
    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(output_path, index=False)
    
    # Log statistics
    class_counts = candidates_df['class'].value_counts()
    logger.info(f"Candidate screening results saved to {output_path}")
    logger.info(f"Total candidates: {len(candidates_df)}")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count}")
    
    return candidates_df


if __name__ == "__main__":
    screen_candidates()
