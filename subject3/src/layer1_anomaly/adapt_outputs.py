"""
Output Format Adapter for Layer1 Anomaly Detection

This module adapts the output files to match the exact design specifications:
1. population_anomalies.csv: Convert to proper long format
2. major_population_changes.csv: Normalize to design schema
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def adapt_population_anomalies(input_path: str = "data/processed/population_anomalies.csv",
                              output_path: str = "data/processed/population_anomalies_adapted.csv") -> pd.DataFrame:
    """
    Adapt population anomalies to proper long format.
    
    Args:
        input_path: Path to input anomalies CSV file
        output_path: Path to output adapted CSV file
        
    Returns:
        Adapted dataframe
    """
    logger.info(f"Adapting population anomalies from {input_path}")
    
    # Read the current file
    df = pd.read_csv(input_path)
    
    # Check if it's already in the correct format
    expected_cols = ['town', 'year', 'z_ae', 'score_iso', 'flag_high', 'flag_low']
    if all(col in df.columns for col in expected_cols):
        logger.info("File is already in correct long format")
        # Ensure exact column order and types
        result_df = df[expected_cols].copy()
        result_df['year'] = result_df['year'].astype(int)
        result_df['z_ae'] = result_df['z_ae'].astype(float)
        result_df['score_iso'] = result_df['score_iso'].astype(float)
        result_df['flag_high'] = result_df['flag_high'].astype(int)
        result_df['flag_low'] = result_df['flag_low'].astype(int)
    else:
        # If it's in wide format, convert to long format
        logger.info("Converting from wide to long format")
        
        # Identify year columns (assuming they start with numbers)
        year_cols = [col for col in df.columns if col.isdigit() or 
                    (col.startswith('19') or col.startswith('20'))]
        
        if not year_cols:
            raise ValueError("No year columns found in wide format data")
        
        # Melt the dataframe
        id_vars = ['town'] if 'town' in df.columns else [df.columns[0]]
        result_df = pd.melt(df, id_vars=id_vars, value_vars=year_cols, 
                           var_name='year', value_name='z_ae')
        
        # Add missing columns with default values
        result_df['score_iso'] = 0.0
        result_df['flag_high'] = 0
        result_df['flag_low'] = 0
        result_df['year'] = result_df['year'].astype(int)
    
    # Sort by town and year
    result_df = result_df.sort_values(['town', 'year']).reset_index(drop=True)
    
    # Save adapted file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Adapted anomalies saved to {output_path}")
    logger.info(f"Shape: {result_df.shape}")
    
    return result_df


def adapt_major_population_changes(features_path: str = "data/processed/features_panel.csv",
                                  input_path: str = "data/processed/major_population_changes.csv",
                                  output_path: str = "data/processed/major_population_changes_adapted.csv") -> pd.DataFrame:
    """
    Adapt major population changes to design schema.
    
    Args:
        features_path: Path to features panel CSV file
        input_path: Path to input changes CSV file  
        output_path: Path to output adapted CSV file
        
    Returns:
        Adapted dataframe
    """
    logger.info(f"Adapting major population changes from {input_path}")
    
    # Read input files
    changes_df = pd.read_csv(input_path)
    features_df = pd.read_csv(features_path)
    
    # Check if already in correct format
    expected_cols = ['town', 'year', 'pop_prev', 'pop_now', 'delta', 'growth_pct', 'class']
    if all(col in changes_df.columns for col in expected_cols):
        logger.info("File is already in correct format")
        # Select and reorder columns
        result_df = changes_df[expected_cols].copy()
        
        # Ensure correct data types
        result_df['year'] = result_df['year'].astype(int)
        result_df['pop_prev'] = result_df['pop_prev'].astype(float)
        result_df['pop_now'] = result_df['pop_now'].astype(float)
        result_df['delta'] = result_df['delta'].astype(float)
        result_df['growth_pct'] = result_df['growth_pct'].astype(float)
        
    else:
        # If not in correct format, reconstruct from features
        logger.info("Reconstructing from features data")
        
        # Get population data with previous year
        pop_df = features_df[['town', 'year', 'pop_total']].copy()
        pop_df = pop_df.sort_values(['town', 'year'])
        pop_df['pop_prev'] = pop_df.groupby('town')['pop_total'].shift()
        pop_df['delta'] = pop_df['pop_total'] - pop_df['pop_prev']
        pop_df['growth_pct'] = pop_df['delta'] / pop_df['pop_prev'].replace(0, 1.0)
        
        # Rename columns to match schema
        pop_df = pop_df.rename(columns={'pop_total': 'pop_now'})
        
        # Merge with changes classification if available
        if 'class' in changes_df.columns:
            # Merge on town and year
            result_df = pop_df.merge(changes_df[['town', 'year', 'class']], 
                                   on=['town', 'year'], how='inner')
        else:
            # Add default class
            result_df = pop_df.copy()
            result_df['class'] = 'unknown'
        
        # Filter out rows with missing previous population
        result_df = result_df.dropna(subset=['pop_prev', 'delta'])
        
        # Select final columns
        result_df = result_df[expected_cols]
    
    # Sort by town and year
    result_df = result_df.sort_values(['town', 'year']).reset_index(drop=True)
    
    # Save adapted file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Adapted changes saved to {output_path}")
    logger.info(f"Shape: {result_df.shape}")
    logger.info(f"Classes: {result_df['class'].value_counts().to_dict()}")
    
    return result_df


def adapt_all_outputs(features_path: str = "data/processed/features_panel.csv",
                     anomalies_path: str = "data/processed/population_anomalies.csv",
                     changes_path: str = "data/processed/major_population_changes.csv",
                     output_dir: str = "data/processed") -> tuple:
    """
    Adapt all output files to design specifications.
    
    Args:
        features_path: Path to features panel CSV file
        anomalies_path: Path to anomalies CSV file
        changes_path: Path to changes CSV file
        output_dir: Output directory for adapted files
        
    Returns:
        Tuple of (adapted_anomalies_df, adapted_changes_df)
    """
    logger.info("Adapting all output files to design specifications")
    
    # Adapt population anomalies
    adapted_anomalies = adapt_population_anomalies(
        input_path=anomalies_path,
        output_path=f"{output_dir}/population_anomalies_adapted.csv"
    )
    
    # Adapt major population changes
    adapted_changes = adapt_major_population_changes(
        features_path=features_path,
        input_path=changes_path,
        output_path=f"{output_dir}/major_population_changes_adapted.csv"
    )
    
    logger.info("All files adapted successfully")
    
    return adapted_anomalies, adapted_changes


if __name__ == "__main__":
    # Adapt all output files
    adapt_all_outputs()
