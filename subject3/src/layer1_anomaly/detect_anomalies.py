"""
Anomaly Detection for Population Data

This module implements unsupervised anomaly detection using:
- Autoencoder reconstruction error
- Isolation Forest
- Ensemble voting for final anomaly flags
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Simple autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select numerical features for anomaly detection.
    
    Args:
        df: Input dataframe with features
        
    Returns:
        Tuple of (feature_df, feature_names)
    """
    # Base features (always included)
    base_features = [
        'delta', 'growth_log', 'growth_adj_log', 
        'delta_roll_mean_3', 'delta_roll_std_3', 
        'delta_accel', 'delta_cum3', 'trend_slope_3y'
    ]
    
    # Add log of population total
    df['log_pop_total'] = np.log1p(df['pop_total'])
    base_features.append('log_pop_total')
    
    # Add age features if available
    age_share_features = [col for col in df.columns if col.startswith('age_share_')]
    d_age_share_features = [col for col in df.columns if col.startswith('d_age_share_')]
    
    all_features = base_features + age_share_features + d_age_share_features
    
    # Filter to available features
    available_features = [f for f in all_features if f in df.columns]
    
    # Select features and handle missing values
    feature_df = df[available_features].copy()
    feature_df = feature_df.fillna(0)  # Simple imputation for missing values
    
    logger.info(f"Selected {len(available_features)} features for anomaly detection")
    
    return feature_df, available_features


def train_autoencoder(X: np.ndarray, hidden_dim: int = 128, latent_dim: int = 32, 
                     epochs: int = 100, patience: int = 10, random_state: int = 42) -> Autoencoder:
    """
    Train autoencoder for anomaly detection.
    
    Args:
        X: Input features
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        epochs: Maximum training epochs
        patience: Early stopping patience
        random_state: Random seed
        
    Returns:
        Trained autoencoder model
    """
    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X)
    
    # Split data
    X_train, X_val = train_test_split(X_tensor, test_size=0.2, random_state=random_state)
    
    # Create model
    model = Autoencoder(X.shape[1], hidden_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, X_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, X_val)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def calculate_robust_z_scores(reconstruction_errors: np.ndarray) -> np.ndarray:
    """
    Calculate robust Z-scores using median and MAD.
    
    Args:
        reconstruction_errors: Array of reconstruction errors
        
    Returns:
        Robust Z-scores
    """
    median = np.median(reconstruction_errors)
    mad = np.median(np.abs(reconstruction_errors - median))
    z_scores = (reconstruction_errors - median) / (mad * 1.4826)  # 1.4826 is the MAD to std conversion
    return z_scores


def detect_anomalies(input_path: str = "subject3/data/processed/features_panel.csv",
                    output_path: str = "subject3/data/processed/population_anomalies.csv",
                    pca_var: float = 0.95,
                    vote_top_percent: float = 0.05,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Detect anomalies using autoencoder and isolation forest.
    
    Args:
        input_path: Path to features CSV file
        output_path: Path to output anomalies CSV file
        pca_var: PCA variance threshold
        vote_top_percent: Top percent for voting
        random_state: Random seed
        
    Returns:
        Dataframe with anomaly scores and flags
    """
    logger.info(f"Detecting anomalies from {input_path}")
    
    # Read data
    df = pd.read_csv(input_path)
    
    # Select features
    feature_df, feature_names = select_features(df)
    X = feature_df.values
    
    # Preprocessing
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA if needed
    if X_scaled.shape[1] > 10:  # Only apply PCA if we have many features
        pca = PCA(n_components=pca_var, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        logger.info(f"PCA reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]}")
    else:
        X_pca = X_scaled
    
    # Autoencoder
    logger.info("Training autoencoder...")
    ae_model = train_autoencoder(X_pca, random_state=random_state)
    
    # Calculate reconstruction errors
    ae_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pca)
        reconstructed = ae_model(X_tensor)
        reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    
    # Calculate robust Z-scores
    z_ae = calculate_robust_z_scores(reconstruction_errors)
    
    # Isolation Forest
    logger.info("Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=400,
        contamination=0.02,
        random_state=random_state
    )
    score_iso = iso_forest.fit_predict(X_pca)
    # Convert to anomaly scores (higher = more anomalous)
    score_iso = -iso_forest.score_samples(X_pca)
    
    # Ensemble voting
    logger.info("Performing ensemble voting...")
    
    # Calculate thresholds
    ae_threshold = np.percentile(z_ae, (1 - vote_top_percent) * 100)
    iso_threshold = np.percentile(score_iso, (1 - vote_top_percent) * 100)
    
    # Create flags
    ae_anomalies = z_ae >= ae_threshold
    iso_anomalies = score_iso >= iso_threshold
    
    flag_high = (ae_anomalies & iso_anomalies).astype(int)
    flag_low = ((ae_anomalies | iso_anomalies) & ~(ae_anomalies & iso_anomalies)).astype(int)
    
    # Create output dataframe
    result_df = pd.DataFrame({
        'town': df['town'],
        'year': df['year'],
        'z_ae': z_ae,
        'score_iso': score_iso,
        'flag_high': flag_high,
        'flag_low': flag_low
    })
    
    # Save output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Anomaly detection results saved to {output_path}")
    logger.info(f"High confidence anomalies: {flag_high.sum()}")
    logger.info(f"Low confidence anomalies: {flag_low.sum()}")
    
    return result_df


if __name__ == "__main__":
    detect_anomalies()
