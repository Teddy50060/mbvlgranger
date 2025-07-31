"""
Core VL-Granger Causality Implementation

This module contains the main VLGrangerCausality class that implements
the Variable-Lag Granger Causality algorithm.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import f as f_dist
from dtaidistance import dtw
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
from typing import Dict, Tuple, Optional, Union

class VLGrangerCausality:
    """
    Variable-Lag Granger Causality implementation
    
    This class implements the Variable-Lag Granger Causality method that extends
    traditional Granger causality by detecting optimal and time-varying lags
    between time series using cross-correlation and DTW alignment.
    """
    
    def __init__(self):
        self.last_result = None
    
    def cross_correlation_analysis(self, Y: np.ndarray, X: np.ndarray, max_lag: int) -> Tuple[int, float, np.ndarray]:
        """
        Cross-correlation analysis without numpy.correlate bugs
        
        Parameters:
        -----------
        Y : np.ndarray
            Target time series
        X : np.ndarray
            Source time series  
        max_lag : int
            Maximum lag to consider
            
        Returns:
        --------
        Tuple[int, float, np.ndarray]
            Optimal lag, optimal correlation, full correlation array
        """
        Y = np.array(Y, dtype=float)
        X = np.array(X, dtype=float)
        
        # Center the signals
        Y_centered = Y - np.mean(Y)
        X_centered = X - np.mean(X)
        
        # Manual correlation computation
        correlations = []
        lags = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(Y_centered, X_centered)[0, 1]
            elif lag > 0:
                # Positive lag: X leads Y by 'lag' samples
                if len(Y_centered) > lag:
                    Y_chunk = Y_centered[lag:]
                    X_chunk = X_centered[:len(Y_chunk)]
                    if len(Y_chunk) > 10:  # Minimum overlap
                        corr = np.corrcoef(Y_chunk, X_chunk)[0, 1]
                    else:
                        corr = 0
                else:
                    corr = 0
            else:  # lag < 0
                # Negative lag: Y leads X
                abs_lag = abs(lag)
                if len(X_centered) > abs_lag:
                    X_chunk = X_centered[abs_lag:]
                    Y_chunk = Y_centered[:len(X_chunk)]
                    if len(X_chunk) > 10:
                        corr = np.corrcoef(Y_chunk, X_chunk)[0, 1]
                    else:
                        corr = 0
                else:
                    corr = 0
            
            correlations.append(corr)
            lags.append(lag)
        
        # Find optimal lag
        correlations = np.array(correlations)
        lags = np.array(lags)
        
        opt_idx = np.argmax(np.abs(correlations))
        opt_lag = lags[opt_idx]
        opt_corr = correlations[opt_idx]
        
        return int(opt_lag), float(opt_corr), correlations
    
    def _enhanced_causality_decision(self, vl_result):
        """
        Enhanced causality decision using dual BIC/F-test criteria
        
        Parameters:
        -----------
        vl_result : dict
            VL-Granger analysis result
            
        Returns:
        --------
        Tuple[bool, str, str]
            Final causality decision, confidence level, detection method
        """
        bic_causality = vl_result.get('XgCsY', False)
        ftest_causality = vl_result.get('XgCsY_ftest', False)
        
        # Dual criteria decision
        if bic_causality and ftest_causality:
            final_causality = True
            confidence = 'high'
            method = 'both_criteria'
        elif bic_causality:
            final_causality = True
            confidence = 'medium'
            method = 'bic_only'
        elif ftest_causality:
            final_causality = True
            confidence = 'medium'
            method = 'ftest_only'
        else:
            final_causality = False
            confidence = 'none'
            method = 'neither'
        
        return final_causality, confidence, method
    
    def dtw_variable_alignment(self, Y: np.ndarray, X: np.ndarray, window_size: int) -> np.ndarray:
        """
        DTW alignment with correct lag calculation
        
        Parameters:
        -----------
        Y : np.ndarray
            Target time series
        X : np.ndarray
            Source time series
        window_size : int
            DTW window size
            
        Returns:
        --------
        np.ndarray
            Array of DTW-derived lags
        """
        try:
            distance, paths = dtw.warping_paths(Y, X, window=window_size, use_pruning=True)
            if paths is None:
                return np.zeros(len(Y), dtype=int)
                
            path = dtw.best_path(paths)
            if not path:
                return np.zeros(len(Y), dtype=int)
            
            T = len(Y)
            dtw_lags = np.zeros(T, dtype=int)
            
            # Create mapping from path
            path_dict = {}
            for y_idx, x_idx in path:
                if 0 <= y_idx < T:
                    path_dict[y_idx] = x_idx
            
            # Fill lag sequence
            for t in range(T):
                if t in path_dict:
                    # Lag = how far back in X we look to match Y[t]
                    dtw_lags[t] = t - path_dict[t]
                else:
                    # Interpolate missing values
                    if t > 0:
                        dtw_lags[t] = dtw_lags[t-1]
                    else:
                        dtw_lags[t] = 0
                        
            return dtw_lags
            
        except Exception as e:
            warnings.warn(f"DTW failed: {e}")
            return np.zeros(len(Y), dtype=int)
    
    def following_relation_analysis(self, Y: np.ndarray, X: np.ndarray, 
                                  time_lag_window: Optional[int] = None, 
                                  lag_window: float = 0.2) -> Dict:
        """
        Following relation analysis matching R implementation exactly
        
        Parameters:
        -----------
        Y : np.ndarray
            Target time series
        X : np.ndarray
            Source time series
        time_lag_window : Optional[int]
            Time lag window size
        lag_window : float
            Lag window as fraction of series length
            
        Returns:
        --------
        Dict
            Following relation analysis results
        """
        Y = np.array(Y, dtype=float)
        X = np.array(X, dtype=float)
        T = len(Y)
        
        # Handle missing values
        Y = pd.Series(Y).ffill().bfill().values
        X = pd.Series(X).ffill().bfill().values
        
        if time_lag_window is None:
            time_lag_window = int(np.ceil(lag_window * T))
        
        # Cross-correlation analysis
        opt_delay, opt_corr, ccf_out = self.cross_correlation_analysis(Y, X, time_lag_window)
        
        # Initialize outputs
        nX = np.zeros(T)
        opt_index_vec = np.full(T, opt_delay, dtype=int)
        foll_val = 0.0
        VL_val = 0.0
        tmp_opt_delay = opt_delay
        
        # Handle edge cases
        if np.isnan(opt_corr):
            opt_delay = 0
            opt_corr = 0.0
            foll_val = 0.0
            VL_val = 0.0
            nX = X.copy()
            tmp_opt_delay = 0
        elif opt_delay < 0:
            foll_val = -1.0
            VL_val = 0.0
            nX = X.copy()
        else:
            # Handle negative correlation by flipping X
            X_work = -X if opt_corr < 0 else X.copy()
                
            # DTW analysis for variable lags
            dtw_index_vec = self.dtw_variable_alignment(Y, X_work, opt_delay)
            
            # Hybrid approach: choose best lag at each time point
            index_vec = np.full(T, opt_delay)  # CC lags
            
            for t in range(T):
                cc_lag = index_vec[t]
                dtw_lag = dtw_index_vec[t]
                
                # Ensure non-negative lags
                cc_lag = max(0, cc_lag)
                dtw_lag = max(0, dtw_lag)
                
                # Check bounds and choose best lag
                if t - cc_lag < 0 and t - dtw_lag < 0:
                    nX[t] = X_work[t] if t < len(X_work) else 0
                    opt_index_vec[t] = 0
                elif t - cc_lag >= 0 and t - dtw_lag >= 0:
                    # Choose lag with smaller error
                    cc_val = X_work[t - cc_lag]
                    dtw_val = X_work[t - dtw_lag]
                    
                    cc_dist = abs(Y[t] - cc_val)
                    dtw_dist = abs(Y[t] - dtw_val)
                    
                    if dtw_dist < cc_dist:
                        opt_index_vec[t] = dtw_lag
                        nX[t] = dtw_val
                    else:
                        opt_index_vec[t] = cc_lag
                        nX[t] = cc_val
                elif t - cc_lag >= 0:
                    opt_index_vec[t] = cc_lag
                    nX[t] = X_work[t - cc_lag]
                else:  # t - dtw_lag >= 0
                    opt_index_vec[t] = dtw_lag
                    nX[t] = X_work[t - dtw_lag] if t - dtw_lag < len(X_work) else 0
            
            # Calculate metrics
            VL_val = float(np.mean(opt_index_vec != opt_delay))
            foll_val = float(np.mean(np.sign(opt_index_vec)))
            tmp_opt_delay = np.mean(opt_index_vec)
        
        return {
            'foll_val': foll_val,
            'nX': nX,
            'opt_delay': float(tmp_opt_delay),
            'opt_corr': float(abs(opt_corr)),
            'opt_index_vec': opt_index_vec,
            'VL_val': VL_val,
            'ccf_out': ccf_out
        }
    
    def create_regression_matrices(self, Y: np.ndarray, X_warped: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create regression matrices matching R's ts.intersect exactly
        
        Parameters:
        -----------
        Y : np.ndarray
            Target time series
        X_warped : np.ndarray
            Warped source time series
        max_lag : int
            Maximum lag to consider
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Current Y values, H0 design matrix, H1 design matrix
        """
        T = len(Y)
        if T <= max_lag:
            raise ValueError(f"Time series too short: {T} <= max_lag {max_lag}")
        
        n_obs = T - max_lag
        
        # Current Y values (what we're predicting)
        y_current = Y[max_lag:]
        
        # H0: Y(t) ~ Y(t-1) + Y(t-2) + ... + Y(t-maxLag)
        X_h0 = np.zeros((n_obs, max_lag))
        for lag in range(1, max_lag + 1):
            # Correct indexing for lagged values
            X_h0[:, lag-1] = Y[max_lag-lag:T-lag]
        
        # H1: Y(t) ~ Y(t-1) + X(t-1) + Y(t-2) + X(t-2) + ... (interleaved)
        X_h1 = np.zeros((n_obs, 2 * max_lag))
        for lag in range(1, max_lag + 1):
            # Interleaved Y and X as specified in paper
            X_h1[:, 2*(lag-1)] = Y[max_lag-lag:T-lag]       # Y(t-lag)
            X_h1[:, 2*(lag-1)+1] = X_warped[max_lag-lag:T-lag]  # X(t-lag)
        
        return y_current, X_h0, X_h1
    
    def analyze_causality(self, Y: Union[list, np.ndarray], X: Union[list, np.ndarray], 
                         alpha: float = 0.05, max_lag: Optional[int] = None, 
                         gamma: float = 0.5, auto_lag: bool = True) -> Dict:
        """
        Main VL-Granger causality analysis with corrected signal handling
        
        Parameters:
        -----------
        Y : Union[list, np.ndarray]
            Target time series
        X : Union[list, np.ndarray]
            Source time series
        alpha : float
            Significance level for F-test
        max_lag : Optional[int]
            Maximum lag to consider
        gamma : float
            BIC ratio threshold
        auto_lag : bool
            Whether to automatically adjust max_lag
            
        Returns:
        --------
        Dict
            Comprehensive causality analysis results
        """
        Y = np.array(Y, dtype=float)
        X = np.array(X, dtype=float)
        
        if len(Y) != len(X):
            raise ValueError("Y and X must have the same length")
        
        T = len(Y)
        if T < 10:
            raise ValueError("Time series too short")
        
        if max_lag is None:
            max_lag = max(1, int(0.2 * T))
        max_lag = max(1, min(max_lag, T // 4))
        
        # Step 1: Following relation analysis
        following_result = self.following_relation_analysis(Y, X, max_lag)
        
        # Step 2: Auto-adjust max_lag based on detected optimal delay
        if auto_lag and following_result['opt_delay'] > 0:
            max_lag = max(max_lag, min(abs(following_result['opt_delay']), T // 4))
        
        # Step 3: Check sufficient data
        min_required = 2 * max_lag + 5
        if T <= min_required:
            warnings.warn(f"Insufficient data: T={T}, need ≥{min_required}")
            return self._create_insufficient_data_result(max_lag, following_result)
        
        # Step 4: Signal preparation matching R code exactly
        if following_result['opt_delay'] == 0:
            X_warped = following_result['nX']
        else:
            # Apply the signal shift as in R code: c(nX[-1], 0)
            nX = following_result['nX']
            X_warped = np.concatenate([nX[1:], [0]])
        
        try:
            # Step 5: Create regression matrices
            y_current, X_h0, X_h1 = self.create_regression_matrices(Y, X_warped, max_lag)
            
            # Step 6: Fit models with intercepts
            X_h0_const = add_constant(X_h0)
            X_h1_const = add_constant(X_h1)
            
            model_h0 = OLS(y_current, X_h0_const).fit()
            model_h1 = OLS(y_current, X_h1_const).fit()
            
            # Step 7: Calculate test statistics
            n = len(y_current)
            S0 = np.sum(model_h0.resid**2)  # RSS for H0
            S1 = np.sum(model_h1.resid**2)  # RSS for H1
            
            # F-test for H0 vs H1
            df_num = max_lag  # Additional X parameters in H1
            df_den = n - 2 * max_lag - 1  # Residual df for H1
            
            if df_den > 0 and S1 > 0 and S0 >= S1:
                ftest = ((S0 - S1) / df_num) / (S1 / df_den)
                p_val = 1 - f_dist.cdf(ftest, df_num, df_den)
            else:
                ftest = 0.0
                p_val = 1.0
            
            # BIC calculation using exact paper formula
            BIC_H0 = (S0/n) * (n**((max_lag + 1)/n))
            BIC_H1 = (S1/n) * (n**((2*max_lag + 1)/n))
            
            # Decision criteria
            XgCsY_ftest = p_val <= alpha
            XgCsY_BIC = BIC_H1 < BIC_H0
            
            # BIC difference ratio
            BIC_diff_ratio = (BIC_H0 - BIC_H1) / BIC_H0 if BIC_H0 > 0 else 0.0
            XgCsY = BIC_diff_ratio >= gamma
            
            result = {
                # Main causality decisions
                'XgCsY': XgCsY,
                'XgCsY_ftest': XgCsY_ftest,
                'XgCsY_BIC': XgCsY_BIC,
                
                # Test statistics
                'ftest': ftest,
                'p_val': p_val,
                'BIC_H0': BIC_H0,
                'BIC_H1': BIC_H1,
                'BIC_diff_ratio': BIC_diff_ratio,
                
                # Parameters
                'max_lag': max_lag,
                'alpha': alpha,
                'gamma': gamma,
                'n_obs': n,
                
                # Following relation results
                'following_result': following_result,
                
                # Model objects
                'model_H0': model_h0,
                'model_H1': model_h1
            }
            
            self.last_result = result
            return result
            
        except Exception as e:
            warnings.warn(f"Analysis failed: {e}")
            return self._create_failed_analysis_result(max_lag, following_result, str(e))
    
    def _create_insufficient_data_result(self, max_lag: int, following_result: Dict) -> Dict:
        """Create result structure for insufficient data case"""
        return {
            'XgCsY': False, 'XgCsY_ftest': False, 'XgCsY_BIC': False,
            'ftest': np.nan, 'p_val': 1.0, 'BIC_H0': np.inf, 'BIC_H1': np.inf,
            'BIC_diff_ratio': 0.0, 'max_lag': max_lag,
            'following_result': following_result, 'error': 'Insufficient data'
        }
    
    def _create_failed_analysis_result(self, max_lag: int, following_result: Dict, error_msg: str) -> Dict:
        """Create result structure for failed analysis case"""
        return {
            'XgCsY': False, 'XgCsY_ftest': False, 'XgCsY_BIC': False,
            'ftest': np.nan, 'p_val': 1.0, 'BIC_H0': np.inf, 'BIC_H1': np.inf,
            'BIC_diff_ratio': 0.0, 'max_lag': max_lag,
            'following_result': following_result, 'error': error_msg
        }


# Convenience function
def vl_granger_causality(Y, X, alpha=0.05, max_lag=None, gamma=0.5, auto_lag=True):
    """
    Convenience function for VL-Granger causality analysis
    
    Parameters:
    -----------
    Y : array-like
        Target time series
    X : array-like
        Source time series
    alpha : float
        Significance level
    max_lag : int, optional
        Maximum lag to consider
    gamma : float
        BIC ratio threshold
    auto_lag : bool
        Whether to automatically adjust max_lag
        
    Returns:
    --------
    Dict
        Analysis results
    """
    analyzer = VLGrangerCausality()
    return analyzer.analyze_causality(Y, X, alpha, max_lag, gamma, auto_lag)