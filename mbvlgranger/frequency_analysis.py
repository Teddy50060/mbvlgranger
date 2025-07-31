"""
Frequency-Band Specific VL-Granger Analysis

This module contains the MultiBandVLGranger class for analyzing causality
within specific frequency bands.
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Tuple, Optional, Union, List

from .core import VLGrangerCausality


class MultiBandVLGranger:
    """
    Frequency-Band Specific Variable-Lag Granger Causality Analysis
    
    This class extends VL-Granger causality to work within specific frequency bands,
    allowing for frequency-specific connectivity analysis.
    """
    
    # Default EEG frequency bands with better boundaries
    DEFAULT_EEG_BANDS = {
        'delta': (1, 4),
        'theta': (4, 8), 
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 50),
        'high_gamma': (50, 80)
    }
    
    def __init__(self, vl_granger_analyzer=None):
        if vl_granger_analyzer is None:
            self.vl_granger = VLGrangerCausality()
        else:
            self.vl_granger = vl_granger_analyzer
        self.last_result = None
    
    def design_bandpass_filter(self, low_freq: float, high_freq: float, 
                              fs: float, filter_order: int = 4) -> Tuple:
        """
        Better filter design with edge case handling
        
        Parameters:
        -----------
        low_freq : float
            Lower frequency bound
        high_freq : float
            Upper frequency bound
        fs : float
            Sampling frequency
        filter_order : int
            Filter order
            
        Returns:
        --------
        Tuple
            SOS filter coefficients, actual low freq, actual high freq
        """
        nyquist = fs / 2.0
        
        # Ensure minimum separation from boundaries
        low_norm = max(low_freq / nyquist, 0.01)    # At least 1% of Nyquist
        high_norm = min(high_freq / nyquist, 0.95)   # At most 95% of Nyquist
        
        # Ensure minimum bandwidth
        min_bandwidth = 0.02  # 2% of Nyquist minimum
        if (high_norm - low_norm) < min_bandwidth:
            center = (low_norm + high_norm) / 2
            low_norm = max(center - min_bandwidth/2, 0.01)
            high_norm = min(center + min_bandwidth/2, 0.95)
            
        if high_norm <= low_norm:
            raise ValueError(f"Invalid frequency band: {low_freq}-{high_freq} Hz at fs={fs} Hz")
        
        # Use cascaded second-order sections for better numerical stability
        sos = signal.butter(filter_order, [low_norm, high_norm], 
                           btype='band', output='sos')
        
        return sos, low_norm * nyquist, high_norm * nyquist
    
    def apply_bandpass_filter(self, data: np.ndarray, low_freq: float, 
                             high_freq: float, fs: float, 
                             filter_order: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        Apply bandpass filter with proper edge handling
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series
        low_freq : float
            Lower frequency bound
        high_freq : float
            Upper frequency bound
        fs : float
            Sampling frequency
        filter_order : int
            Filter order
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            Filtered data and filter information
        """
        try:
            sos, actual_low, actual_high = self.design_bandpass_filter(
                low_freq, high_freq, fs, filter_order)
            
            # Apply zero-phase filtering
            filtered_data = signal.sosfiltfilt(sos, data)
            
            filter_info = {
                'success': True,
                'actual_frequencies': (actual_low, actual_high),
                'requested_frequencies': (low_freq, high_freq),
                'filter_order': filter_order
            }
            
            return filtered_data, filter_info
            
        except Exception as e:
            warnings.warn(f"Bandpass filtering failed for {low_freq}-{high_freq} Hz: {e}")
            filter_info = {
                'success': False,
                'error': str(e),
                'actual_frequencies': (low_freq, high_freq),
                'requested_frequencies': (low_freq, high_freq)
            }
            return data.copy(), filter_info
    
    def assess_band_quality(self, low_freq: float, high_freq: float, 
                           data_length: int, fs: float) -> Dict:
        """
        Better quality assessment for frequency bands
        
        Parameters:
        -----------
        low_freq : float
            Lower frequency bound
        high_freq : float
            Upper frequency bound
        data_length : int
            Length of time series
        fs : float
            Sampling frequency
            
        Returns:
        --------
        Dict
            Quality assessment results
        """
        band_width = high_freq - low_freq
        quality_flags = []
        
        # Frequency-based checks
        if low_freq < 1:
            quality_flags.append('very_low_frequency')
        elif low_freq < 4:
            quality_flags.append('low_frequency_band')
        
        if band_width < 2:
            quality_flags.append('very_narrow_band')
        elif band_width < 4:
            quality_flags.append('narrow_band')
        
        # Data length checks (more conservative)
        min_cycles_needed = 20  # Need more cycles for robust filtering
        min_samples_needed = min_cycles_needed * fs / low_freq if low_freq > 0 else np.inf
        
        if data_length < min_samples_needed:
            quality_flags.append('insufficient_data_for_frequency')
        elif data_length < 2 * min_samples_needed:
            quality_flags.append('borderline_data_length')
        
        # Nyquist checks
        if high_freq > fs / 3:  # More conservative than 25%
            quality_flags.append('high_frequency_band')
        
        # Overall quality score
        if any(flag in ['very_low_frequency', 'very_narrow_band', 'insufficient_data_for_frequency'] 
               for flag in quality_flags):
            quality_score = 'poor'
        elif quality_flags:
            quality_score = 'warning'
        else:
            quality_score = 'good'
        
        return {
            'quality_score': quality_score,
            'quality_flags': quality_flags,
            'band_width': band_width,
            'min_samples_needed': min_samples_needed,
            'data_adequacy_ratio': data_length / min_samples_needed if min_samples_needed > 0 else np.inf
        }
    
    def single_band_vl_granger(self, X: np.ndarray, Y: np.ndarray, 
                              fs: float, frequency_band: Tuple[float, float],
                              filter_order: int = 4, **vl_granger_kwargs) -> Dict:
        """
        Single band analysis with better error handling
        
        Parameters:
        -----------
        X : np.ndarray
            Source time series
        Y : np.ndarray
            Target time series
        fs : float
            Sampling frequency
        frequency_band : Tuple[float, float]
            Frequency band (low, high)
        filter_order : int
            Filter order
        **vl_granger_kwargs
            Additional arguments for VL-Granger analysis
            
        Returns:
        --------
        Dict
            Single band analysis results
        """
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        
        low_freq, high_freq = frequency_band
        band_name = f"{low_freq}-{high_freq}Hz"
        
        # Step 1: Quality assessment
        quality_assessment = self.assess_band_quality(low_freq, high_freq, len(X), fs)
        
        # Step 2: Apply filtering
        X_filtered, x_filter_info = self.apply_bandpass_filter(X, low_freq, high_freq, fs, filter_order)
        Y_filtered, y_filter_info = self.apply_bandpass_filter(Y, low_freq, high_freq, fs, filter_order)
        
        filtering_success = x_filter_info['success'] and y_filter_info['success']
        
        if not filtering_success:
            quality_assessment['quality_flags'].append('filtering_failed')
            quality_assessment['quality_score'] = 'poor'
        
        # Step 3: VL-Granger analysis
        try:
            # Pass reasonable defaults if not specified
            default_kwargs = {'gamma': 0.3, 'alpha': 0.05, 'auto_lag': True}
            default_kwargs.update(vl_granger_kwargs)
            
            vl_result = self.vl_granger.analyze_causality(Y_filtered, X_filtered, **default_kwargs)
            analysis_success = True
            
            # Check if analysis actually succeeded (not just didn't crash)
            if 'error' in vl_result:
                analysis_success = False
                quality_assessment['quality_flags'].append('analysis_error')
            
        except Exception as e:
            warnings.warn(f"VL-Granger analysis failed for {band_name}: {e}")
            vl_result = self._create_failed_vl_result(str(e))
            analysis_success = False
            quality_assessment['quality_flags'].append('analysis_failed')
            quality_assessment['quality_score'] = 'poor'
        
        # Step 4: Extract key metrics safely
        try:
            causality = vl_result.get('XgCsY', False)
            causality_ftest = vl_result.get('XgCsY_ftest', False)
            causality_bic = vl_result.get('XgCsY_BIC', False)
            bic_ratio = float(vl_result.get('BIC_diff_ratio', 0.0))
            p_value = float(vl_result.get('p_val', 1.0))
            f_statistic = float(vl_result.get('ftest', 0.0))
            
            # Safe extraction of following_result
            following_result = vl_result.get('following_result', {})
            detected_delay = int(following_result.get('opt_delay', 0))
            detected_delay_avg = float(following_result.get('avg_used_lag', detected_delay))
            variable_lag_percentage = float(following_result.get('VL_val', 0.0))
            
        except (TypeError, ValueError, KeyError) as e:
            warnings.warn(f"Error extracting results for {band_name}: {e}")
            causality = causality_ftest = causality_bic = False
            bic_ratio = p_value = f_statistic = 0.0
            detected_delay = 0
            detected_delay_avg = 0.0
            variable_lag_percentage = 0.0
            analysis_success = False

        final_causality, confidence, method = self.vl_granger._enhanced_causality_decision(vl_result)
        
        # Step 5: Compile results
        band_result = {
            # Core causality results
            'causality': final_causality,
            'causality_ftest': causality_ftest,
            'causality_bic': causality_bic,
            'confidence': confidence,
            'detection_method': method,
            
            # Statistics
            'bic_ratio': bic_ratio,
            'p_value': p_value,
            'f_statistic': f_statistic,
            
            # Lag information  
            'detected_delay': detected_delay,
            'detected_delay_avg': detected_delay_avg,
            'variable_lag_percentage': variable_lag_percentage,
            
            # Band metadata
            'frequency_band': frequency_band,
            'band_name': band_name,
            'sampling_rate': fs,
            'quality': quality_assessment['quality_score'],
            'quality_flags': quality_assessment['quality_flags'],
            'band_width': quality_assessment['band_width'],
            'data_adequacy_ratio': quality_assessment['data_adequacy_ratio'],
            
            # Processing info
            'filtering_success': filtering_success,
            'analysis_success': analysis_success,
            'filter_info': {'X': x_filter_info, 'Y': y_filter_info},
            
            # Full results
            'full_vl_result': vl_result
        }
        
        return band_result
    
    def _create_failed_vl_result(self, error_msg: str) -> Dict:
        """Helper to create consistent failed result structure"""
        return {
            'XgCsY': False, 'XgCsY_ftest': False, 'XgCsY_BIC': False,
            'ftest': 0.0, 'p_val': 1.0, 'BIC_H0': np.inf, 'BIC_H1': np.inf,
            'BIC_diff_ratio': 0.0, 'max_lag': 1,
            'following_result': {'opt_delay': 0, 'VL_val': 0.0, 'opt_corr': 0.0},
            'error': error_msg
        }


# Convenience function with better defaults
def multiband_vl_granger_analysis(X, Y, fs, frequency_bands=None, 
                                      band_names=None, gamma=0.3, **kwargs):
    """
    Convenience function for frequency-band VL-Granger analysis
    
    Parameters:
    -----------
    X, Y : array-like
        Input time series
    fs : float
        Sampling frequency
    frequency_bands : list of tuples, optional
        Frequency bands to analyze
    band_names : list of str, optional
        Names for frequency bands
    gamma : float
        BIC ratio threshold
    **kwargs
        Additional arguments for VL-Granger analysis
        
    Returns:
    --------
    Dict
        Analysis results
    """
    analyzer = MultiBandVLGranger()
    return analyzer.multi_band_vl_granger(X, Y, fs, frequency_bands, band_names, 
                                        gamma=gamma, **kwargs)