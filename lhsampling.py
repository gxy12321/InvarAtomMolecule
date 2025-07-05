
import pandas as pd
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist


def load_data(
    csv_path: str,
    gas: str = 'N2',
    feature_cols: list = None,
    target_col: str = 'tot_rate',
    n_train: int = 1500,
    mode: str = 'log'
) -> tuple:
    """
    Load data, filter by gas, log-transform target, and split into train/test sets.
    
    Args:
        csv_path (str): Path to the CSV file.
        gas (str): Gas type to filter by (default: 'N2').
        feature_cols (list): List of feature column names (default: None, uses predefined list).
        target_col (str): Target column name (default: 'tot_rate').
        n_train (int): Number of training samples (default: 3300).
    
    Returns:
        tuple: (X, y, train_idx, test_idx) where X is the feature matrix, y is the target array,
               train_idx and test_idx are arrays of training and test indices.
    """
    # Load and filter data
    df = pd.read_csv(csv_path)
    df = df[df['gas'] == gas]
    
    # Set default feature columns if none provided
    if feature_cols is None:
        feature_cols = ['de0', 're0', 'de2', 're2', 'c62', 'b']
    
    # Sort by target column and reset index
    df = df.sort_values(by=target_col).reset_index(drop=True)
    
    # Prepare feature matrix X and target array y
    X = df[feature_cols].values.astype(np.float32)
    if mode == 'log':
        y = np.log(df[target_col].values.astype(np.float32) + 1e-8)
    elif mode == 'no_log':
        y = df[target_col].values.astype(np.float32)
    else:
        raise ValueError('Mode is wrong: either "log" or "no_log"')
    
    # Perform train/test split using Latin Hypercube Sampling
    N = X.shape[0]  # Total number of data points
    d = X.shape[1]  # Number of features
    
    # Check if n_train is valid
    if n_train >= N:
        raise ValueError(f'n_train ({n_train}) must be less than the number of data points ({N})')
    
    # Generate LHS samples in [0,1]^d
    lhs = LatinHypercube(d)
    samples = lhs.random(n=n_train)
    
    # Scale samples to the range of X
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    scaled_samples = X_min + samples * (X_max - X_min)
    
    # Compute distance matrix between scaled samples and X
    dist_matrix = cdist(scaled_samples, X)
    
    # Greedily select unique data points closest to each LHS sample
    train_idx = []
    selected = np.zeros(N, dtype=bool)
    for i in range(n_train):
        # Get distances to unselected data points
        distances = dist_matrix[i, ~selected]
        min_j_local = np.argmin(distances)  # Index in the unselected subset
        min_j = np.where(~selected)[0][min_j_local]  # Global index
        train_idx.append(min_j)
        selected[min_j] = True
    
    # Convert train_idx to numpy array and get test_idx
    train_idx = np.array(train_idx)
    test_idx = np.setdiff1d(np.arange(N), train_idx)
    
    return X, y, train_idx, test_idx