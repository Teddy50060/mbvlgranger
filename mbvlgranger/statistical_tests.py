"""
Statistical Tests and Meta-Analysis Methods for VL-Granger

This module contains statistical combination methods and the main mbvl_granger function.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, pearsonr
from typing import Dict, Tuple, Optional, List
import warnings

from .core import VLGrangerCausality
from .frequency_analysis import MultiBandVLGranger


def fishers_combined_test(p_values):
    """
    Fisher's method to combine p-values
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values to combine
        
    Returns:
    --------
    Tuple[float, float]
        Combined p-value, test statistic
    """
    # Remove any NaN values
    valid_p_values = p_values[~np.isnan(p_values)]
    
    if len(valid_p_values) == 0:
        return 1.0, 0.0  # No valid p-values
    
    # Fisher's formula: -2 * sum(ln(p_i))
    test_statistic = -2 * np.sum(np.log(valid_p_values))
    
    # Degrees of freedom = 2 * number of tests
    df = 2 * len(valid_p_values)
    
    # P-value from chi-square distribution
    combined_p_value = 1 - chi2.cdf(test_statistic, df)
    
    return combined_p_value, test_statistic


def stouffers_method(p_values):
    """
    Stouffer's method to combine p-values
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values to combine
        
    Returns:
    --------
    Tuple[float, float]
        Combined p-value, combined Z-score
    """
    valid_p_values = p_values[~np.isnan(p_values)]
    
    if len(valid_p_values) == 0:
        return 1.0, 0.0
    
    # Convert p-values to Z-scores
    z_scores = norm.ppf(1 - valid_p_values)  # One-tailed
    
    # Combined Z-score
    combined_z = np.sum(z_scores) / np.sqrt(len(valid_p_values))
    
    # Combined p-value
    combined_p_value = 1 - norm.cdf(combined_z)
    
    return combined_p_value, combined_z


def bonferroni_combination(p_values):
    """
    Bonferroni combination method
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values to combine
        
    Returns:
    --------
    Tuple[float, float]
        Combined p-value, minimum p-value
    """
    valid_p_values = p_values[~np.isnan(p_values)]
    
    if len(valid_p_values) == 0:
        return 1.0, 0.0
    
    # Bonferroni combination: min(1, k * min(p_i))
    min_p = np.min(valid_p_values)
    k = len(valid_p_values)
    combined_p = min(1.0, k * min_p)
    
    return combined_p, min_p


def estimate_noise_characteristics(X, Y):
    """
    Estimate noise level in the signals
    
    Parameters:
    -----------
    X, Y : array-like
        Input time series
        
    Returns:
    --------
    float
        Noise level (0-1, higher = more noisy)
    """
    # Signal-to-noise ratio estimation
    signal_power_x = np.var(X)
    signal_power_y = np.var(Y)
    
    # High-frequency noise estimation (above 100Hz)
    from scipy.signal import welch
    f_x, psd_x = welch(X, fs=250, nperseg=min(256, len(X)//4))
    f_y, psd_y = welch(Y, fs=250, nperseg=min(256, len(Y)//4))
    
    # Noise power (high frequency content)
    high_freq_mask = f_x > 100
    noise_power_x = np.mean(psd_x[high_freq_mask]) if np.any(high_freq_mask) else 0
    noise_power_y = np.mean(psd_y[high_freq_mask]) if np.any(high_freq_mask) else 0
    
    # Cross-correlation peak sharpness (sharp peak = clean signal)
    correlation = np.correlate(Y, X, mode='full')
    correlation = correlation / np.max(np.abs(correlation))
    peak_sharpness = np.max(correlation) / np.std(correlation)
    
    # Combine metrics (higher = more noisy)
    snr_x = signal_power_x / (noise_power_x + 1e-10)
    snr_y = signal_power_y / (noise_power_y + 1e-10)
    
    # Normalize to 0-1 scale
    avg_snr = (snr_x + snr_y) / 2
    noise_level = 1 / (1 + avg_snr/10)  # Sigmoid transformation
    
    # Adjust based on peak sharpness
    if peak_sharpness < 2:  # Broad correlation peak = noisy
        noise_level = min(1.0, noise_level * 1.5)
    
    return noise_level


def broadband_vl_granger_with_strict_mode(X, Y, fs, max_lag=25, strict_mode=False):
    """
    Broadband VL-Granger with optional strict mode for noisy signals
    
    Parameters:
    -----------
    X, Y : array-like
        Input time series
    fs : float
        Sampling frequency
    max_lag : int
        Maximum lag to consider
    strict_mode : bool
        Whether to use strict thresholds
        
    Returns:
    --------
    Dict
        Analysis results
    """
    vl_analyzer = VLGrangerCausality()
    
    # Adjust parameters for strict mode
    if strict_mode:
        alpha = 0.001  # Much stricter p-value
        gamma = 0.8    # Much stricter BIC ratio
    else:
        alpha = 0.05
        gamma = 0.6
    
    result = vl_analyzer.analyze_causality(
        Y, X,
        alpha=alpha,
        max_lag=max_lag,
        gamma=gamma,
        auto_lag=True
    )
    
    # Extract results
    causality = result.get('XgCsY', False)
    p_value = result.get('p_val', 1.0)
    bic_ratio = result.get('BIC_diff_ratio', 0.0)
    detected_lag = result.get('following_result', {}).get('opt_delay', 0)
    
    return {
        'causality': causality,
        'p_value': p_value,
        'bic_ratio': bic_ratio,
        'detected_lag': detected_lag,
        'method': 'broadband_strict' if strict_mode else 'broadband_normal'
    }


def residualize(target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    """
    Remove linear effects of predictors from target variable
    
    Parameters:
    -----------
    target : np.ndarray
        Target variable to residualize
    predictors : np.ndarray
        Predictor variables
        
    Returns:
    --------
    np.ndarray
        Residuals from linear regression
    """
    try:
        if predictors.size == 0 or len(target) != len(predictors):
            return target
            
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(predictors)), predictors])
        
        # Solve normal equations: (X'X)^-1 X'y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ target
        
        # Check for singularity
        if np.linalg.cond(XtX) > 1e12:
            return target  # Return original if matrix is singular
            
        beta = np.linalg.solve(XtX, Xty)
        fitted = X_with_intercept @ beta
        residuals = target - fitted
        
        return residuals
        
    except Exception as e:
        return target  # Return original if residualization fails


def test_partial_correlation(Y: np.ndarray, X: np.ndarray, target_lag: int, 
                           conditioning_lags: list, fs: float) -> float:
    """
    Test partial correlation: Y_t vs X_{t-target_lag} controlling for X_{t-conditioning_lags}
    
    Parameters:
    -----------
    Y : np.ndarray
        Target time series
    X : np.ndarray
        Source time series
    target_lag : int
        Target lag to test
    conditioning_lags : list
        Lags to condition on
    fs : float
        Sampling frequency
        
    Returns:
    --------
    float
        P-value of the partial correlation test
    """
    try:
        if target_lag >= len(X) or target_lag >= len(Y):
            return 1.0
            
        # Prepare target variables
        Y_target = Y[max(conditioning_lags + [target_lag]):]
        X_target_lag = X[max(conditioning_lags + [target_lag]) - target_lag:-target_lag] if target_lag > 0 else X[max(conditioning_lags):]
        
        if len(Y_target) < 10:  # Need minimum samples
            return 1.0
            
        # Prepare conditioning variables
        conditioning_vars = []
        for cond_lag in conditioning_lags:
            if cond_lag != target_lag:  # Don't condition on itself
                X_cond = X[max(conditioning_lags + [target_lag]) - cond_lag:-cond_lag] if cond_lag > 0 else X[max(conditioning_lags + [target_lag]):]
                if len(X_cond) == len(Y_target):
                    conditioning_vars.append(X_cond)
        
        if not conditioning_vars:
            # No valid conditioning variables, fall back to simple correlation
            correlation, p_value = pearsonr(Y_target, X_target_lag)
            return p_value
        
        # Create design matrix for partial correlation
        conditioning_matrix = np.column_stack(conditioning_vars)
        
        # Residualize Y and X with respect to conditioning variables
        Y_residual = residualize(Y_target, conditioning_matrix)
        X_residual = residualize(X_target_lag, conditioning_matrix)
        
        # Test correlation of residuals (partial correlation)
        if len(Y_residual) > 5 and len(X_residual) > 5:
            partial_corr, p_value = pearsonr(Y_residual, X_residual)
            return p_value
        else:
            return 1.0
            
    except Exception as e:
        return 1.0  # Conservative: assume no relationship if test fails


def pcmci_inspired_lag_selection(X: np.ndarray, Y: np.ndarray, frequency_band: Tuple[float, float], 
                                fs: float, alpha: float = 0.05, max_search_factor: float = 2.0) -> int:
    """
    PCMCI-inspired adaptive lag selection for VL-Granger causality
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Input time series
    frequency_band : tuple
        (low_freq, high_freq) for the frequency band
    fs : float
        Sampling frequency
    alpha : float
        Significance level for statistical tests
    max_search_factor : float
        Factor to determine maximum search range
        
    Returns:
    --------
    int
        Optimal maximum lag for this frequency band
    """
    low_freq, high_freq = frequency_band
    center_freq = np.sqrt(low_freq * high_freq)
    
    # Set frequency-specific search bounds
    if center_freq > 40:  # High gamma: max 20ms
        max_search = min(int(fs * 0.02), int(fs / low_freq * max_search_factor))
    elif center_freq > 30:  # Low gamma: max 25ms
        max_search = min(int(fs * 0.025), int(fs / low_freq * max_search_factor))
    elif center_freq > 13:  # Beta: max 50ms
        max_search = min(int(fs * 0.05), int(fs / low_freq * max_search_factor))
    elif center_freq > 8:   # Alpha: max 100ms
        max_search = min(int(fs * 0.1), int(fs / low_freq * max_search_factor))
    else:  # Theta/Delta: max 200ms
        max_search = min(int(fs * 0.2), int(fs / low_freq * max_search_factor))
    
    # Ensure reasonable bounds
    min_search = max(2, int(fs / high_freq / 8))  # Sub-cycle resolution
    max_search = max(min_search + 1, min(max_search, 50))  # Hard computational limit
    
    relevant_lags = []
    conditioning_lags = []
    
    # PCMCI Stage 1: PC1-inspired iterative lag testing
    for lag in range(min_search, max_search + 1):
        if lag >= len(X) or lag >= len(Y):
            break
            
        try:
            if len(conditioning_lags) == 0:
                # Unconditional test: simple correlation
                if lag < len(X) and lag < len(Y):
                    X_lagged = X[:-lag] if lag > 0 else X
                    Y_target = Y[lag:]
                    
                    if len(Y_target) < 10:  # Need minimum samples
                        continue
                        
                    correlation, p_value = pearsonr(Y_target, X_lagged)
                else:
                    continue
            else:
                # Conditional test: partial correlation controlling for other relevant lags
                p_value = test_partial_correlation(Y, X, lag, conditioning_lags, fs)
            
            # If significant (NOT independent), include in relevant set
            if p_value < alpha and not np.isnan(p_value):
                relevant_lags.append(lag)
                conditioning_lags.append(lag)
                
            # Limit conditioning set size to prevent overfitting (key PCMCI principle)
            if len(conditioning_lags) > 5:
                # Keep most informative lags (could be refined with better selection)
                conditioning_lags = conditioning_lags[-3:]  # Keep most recent 3
                
        except Exception as e:
            # Skip problematic lags
            continue
    
    # Return optimal max_lag
    if relevant_lags:
        optimal_lag = max(relevant_lags)
        
        # Apply additional frequency-specific constraints as safety bounds
        if center_freq > 30:  # Gamma bands
            optimal_lag = min(optimal_lag, int(fs * 0.03))  # 30ms max
        elif center_freq > 13:  # Beta
            optimal_lag = min(optimal_lag, int(fs * 0.06))  # 60ms max
            
        return max(2, optimal_lag)
    else:
        # Fallback to literature-based heuristic if no relevant lags found
        fallback_lag = max(2, int(fs / center_freq / 4))  # Cohen (2014) quarter-period rule
        return min(fallback_lag, max_search)


def mbvl_granger(X: np.ndarray, Y: np.ndarray, fs: float,
               max_lag: Optional[int] = None,
               bands: Optional[Dict[str, Tuple[float, float]]] = None,
               alpha: float = 0.05,
               gamma: float = 0.2,
               combination_method: str = 'fisher',
               adaptive_lag: bool = False,
               pcmci_alpha: float = 0.05,
               fallback_max_lag: int = 15,
               noise_aware: bool = True,
               noise_threshold: float = 1.0) -> Dict:
    """
    Enhanced VL-Granger with noise-aware hierarchical selection
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Input time series
    fs : float
        Sampling frequency
    max_lag : int, optional
        Maximum lag to consider
    bands : dict, optional
        Frequency bands to analyze
    alpha : float
        Significance level
    gamma : float
        BIC ratio threshold  
    combination_method : str
        Method to combine p-values ('fisher', 'stouffer', 'bonferroni')
    adaptive_lag : bool
        Whether to use adaptive lag selection
    pcmci_alpha : float
        Alpha for PCMCI lag selection
    fallback_max_lag : int
        Fallback maximum lag
    noise_aware : bool
        Whether to use noise-aware hierarchical approach
    noise_threshold : float
        Noise level threshold
        
    Returns:
    --------
    Dict
        Comprehensive analysis results
    """
    
    # NOISE-AWARE HIERARCHICAL SELECTION
    if noise_aware:
        noise_level = estimate_noise_characteristics(X, Y)
        
        if noise_level > noise_threshold:
            # High noise: use frequency bands (your method - lower false positives)
            method_used = 'frequency_bands_noisy'
            # Continue with existing frequency band analysis below...
            
        else:
            # Low noise: try broadband VL-Granger first
            broadband_result = broadband_vl_granger_with_strict_mode(X, Y, fs, max_lag or 25, strict_mode=False)
            
            if broadband_result['causality'] and broadband_result['bic_ratio'] > 0.7:
                # Strong broadband causality - return early
                return {
                    'overall_causality': True,
                    'combined_p_value': broadband_result['p_value'],
                    'method_used': 'broadband_primary',
                    'detected_lag': broadband_result['detected_lag'],
                    'bic_ratio': broadband_result['bic_ratio'],
                    'noise_level': noise_level,
                    'band_results': pd.DataFrame(),  # Empty for compatibility
                    'adaptive_lags_used': {'broadband': max_lag or 25}
                }
            else:
                # Weak/no broadband causality: fall back to frequency bands
                method_used = 'frequency_bands_fallback'
                # Continue with existing frequency band analysis below...
    else:
        method_used = 'frequency_bands_only'
        noise_level = None
        # Continue with existing frequency band analysis below...

    # Use provided bands or default frequency bands
    if bands is None:
        bands = {
            'low': (1, 80),
            'high': (81, 120)
        }
    
    analyzer = MultiBandVLGranger()
    results = []
    p_values = []
    adaptive_lags_used = {}  # Track what lags were selected for each band
    count_significant = 0
    
    for band_name, freq_range in bands.items():
        try:
            # Determine max_lag for this frequency band
            if adaptive_lag and max_lag is None:
                # Use PCMCI-inspired adaptive lag selection
                band_max_lag = pcmci_inspired_lag_selection(
                    X, Y, freq_range, fs, alpha=pcmci_alpha
                )
                adaptive_lags_used[band_name] = band_max_lag
            elif max_lag is not None:
                # Use provided fixed max_lag
                band_max_lag = max_lag
                adaptive_lags_used[band_name] = max_lag
            else:
                # Fallback to default
                band_max_lag = fallback_max_lag
                adaptive_lags_used[band_name] = fallback_max_lag
            
            # Run VL-Granger analysis with determined max_lag
            result = analyzer.single_band_vl_granger(
                X, Y, fs=fs,
                frequency_band=freq_range,
                max_lag=band_max_lag,  # Use adaptive or fixed lag
                gamma=gamma,
                alpha=alpha
            )
            
            p_val = result['p_value']
            bic_ratio = result['bic_ratio']
            
            # Individual band significance using dual criteria
            significant_individual = (p_val <= alpha) or (bic_ratio >= gamma)

            if significant_individual:
                count_significant += 1
            
            interval = f"{freq_range[0]}-{freq_range[1]}Hz"
            results.append({
                'interval': interval,
                'f_stat': result['f_statistic'],
                'p_value': p_val,
                'bic_ratio': bic_ratio,
                'detected_lag': result['detected_delay_avg'],
                'max_lag_used': band_max_lag,  # Track what max_lag was used
                'significant_individual': significant_individual
            })
            
            # Collect p-values for combination
            if not np.isnan(p_val):
                p_values.append(p_val)
                
        except Exception as e:
            # Handle errors gracefully
            interval = f"{freq_range[0]}-{freq_range[1]}Hz"
            fallback_lag = fallback_max_lag if not adaptive_lag or max_lag is None else max_lag
            
            results.append({
                'interval': interval,
                'f_stat': np.nan,
                'p_value': np.nan,
                'bic_ratio': np.nan,
                'detected_lag': np.nan,
                'max_lag_used': fallback_lag,
                'significant_individual': False
            })
            
            adaptive_lags_used[band_name] = fallback_lag
            warnings.warn(f"Error processing band {band_name}: {str(e)}")
    
    # Create DataFrame
    band_results = pd.DataFrame(results)
    
    # COMBINE P-VALUES for overall causality
    if len(p_values) > 0:
        p_values_array = np.array(p_values)
        if combination_method == 'fisher':
            combined_p_value, test_statistic = fishers_combined_test(p_values_array)
        elif combination_method == 'stouffer':
            combined_p_value, test_statistic = stouffers_method(p_values_array)
        elif combination_method == 'bonferroni':
            combined_p_value, test_statistic = bonferroni_combination(p_values_array)
        else:
            raise ValueError("combination_method must be 'fisher', 'stouffer', or 'bonferroni'")
    else:
        combined_p_value = 1.0
        test_statistic = 0.0
    
    # Overall causality decision
    overall_causality = combined_p_value <= alpha
    
    return {
        'band_results': band_results,
        'overall_causality': overall_causality,
        'combined_p_value': combined_p_value,
        'test_statistic': test_statistic,
        'combination_method': combination_method,
        'individual_p_values': p_values,
        'n_valid_bands': len(p_values),
        'adaptive_lags_used': adaptive_lags_used,  # New: shows what lags were selected
        'adaptive_lag_enabled': adaptive_lag,       # New: indicates if adaptive mode was used
        'noise_level': noise_level,          # NEW
        'method_used': method_used
    }


def print_mbvlgranger_results(results):
    """
    Print VL-Granger results in a clean format
    
    Parameters:
    -----------
    results : dict
        Results from mbvl_granger analysis
    """
    print("VL-Granger Frequency Causality Analysis Results")
    print("=" * 50)
    
    # Overall results
    print(f"Overall Causality: {results['overall_causality']}")
    print(f"Combined P-value: {results['combined_p_value']:.6f}")
    print(f"Method: {results['combination_method']}")
    print(f"Valid Bands: {results['n_valid_bands']}")
    
    # Individual band results
    print("\nIndividual Band Results:")
    print("-" * 50)
    band_results = results['band_results']
    
    for _, row in band_results.iterrows():
        sig_status = "YES" if row['significant_individual'] else "NO"
        print(f"{row['interval']:8} | F={row['f_stat']:6.3f} | p={row['p_value']:8.6f} | "
              f"BIC={row['bic_ratio']:7.3f} | Lag={row['detected_lag']:3.0f} | Sig={sig_status}")


def quick_mbvlgranger(x, y, fs=300, max_lag=30, bands=None, combination_method='fisher', print_results=True):
    """
    Ultra-simple one-liner function for VL-Granger analysis
    
    Parameters:
    -----------
    x, y : array-like
        Input time series
    fs : float
        Sampling frequency
    max_lag : int
        Maximum lag to consider
    bands : dict, optional
        Frequency bands to analyze
    combination_method : str
        Statistical combination method
    print_results : bool
        Whether to print results
        
    Returns:
    --------
    dict
        Analysis results with DataFrame
    """
    results = mbvl_granger(x, y, fs, max_lag, bands=bands, combination_method=combination_method)
    
    if print_results:
        print_mbvlgranger_results(results)
    
    return results