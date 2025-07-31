"""
Utility functions for MBVL-Granger analysis
"""

import numpy as np
import warnings
from typing import Union, List

def validate_time_series(X: Union[list, np.ndarray], Y: Union[list, np.ndarray]) -> tuple:
    """
    Validate and prepare time series for analysis
    
    Parameters:
    -----------
    X, Y : array-like
        Input time series
        
    Returns:
    --------
    tuple
        Validated numpy arrays
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    
    if len(X) < 10:
        raise ValueError("Time series too short (minimum 10 samples)")
    
    # Handle NaN values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        warnings.warn("NaN values detected, using forward fill")
        import pandas as pd
        X = pd.Series(X).ffill().bfill().values
        Y = pd.Series(Y).ffill().bfill().values
    
    return X, Y

def check_sampling_frequency(fs: float, data_length: int) -> None:
    """
    Validate sampling frequency
    
    Parameters:
    -----------
    fs : float
        Sampling frequency
    data_length : int
        Length of time series
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    if fs < 2:
        warnings.warn("Very low sampling frequency detected")
    
    # Check Nyquist considerations
    duration = data_length / fs
    if duration < 1:
        warnings.warn("Very short time series duration")

def format_frequency_bands(bands: dict) -> dict:
    """
    Validate and format frequency bands
    
    Parameters:
    -----------
    bands : dict
        Frequency bands dictionary
        
    Returns:
    --------
    dict
        Validated frequency bands
    """
    formatted_bands = {}
    
    for name, (low, high) in bands.items():
        if low >= high:
            raise ValueError(f"Invalid band {name}: low freq >= high freq")
        if low < 0:
            raise ValueError(f"Invalid band {name}: negative frequency")
        
        formatted_bands[name] = (float(low), float(high))
    
    return formatted_bands