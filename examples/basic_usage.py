#!/usr/bin/env python3
"""
Basic Usage Example for MBVL-Granger Causality Analysis

This example demonstrates the simplest way to use the mbvlgranger package
for analyzing causality between two time series.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mbvlgranger import quick_mbvlgranger, mbvl_granger, print_mbvlgranger_results

def example_1_simple_analysis():
    """Example 1: Ultra-simple one-liner analysis"""
    print("=" * 60)
    print("EXAMPLE 1: Ultra-Simple Analysis")
    print("=" * 60)
    
    # Generate synthetic data with known causality
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = np.zeros(n)
    
    # Create causality: y[t] = 0.8 * x[t-10] + noise
    lag = 10
    for t in range(n):
        if t >= lag:
            y[t] = 0.8 * x[t - lag] + 0.3 * np.random.randn()
        else:
            y[t] = 0.3 * np.random.randn()
    
    # One-liner analysis
    results = quick_mbvlgranger(x, y, fs=250, max_lag=30)
    
    print(f"\n✅ Analysis complete!")
    print(f"Overall Causality: {results['overall_causality']}")
    print(f"Combined p-value: {results['combined_p_value']:.6f}")

def example_2_custom_frequency_bands():
    """Example 2: Custom frequency bands for specific application"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Frequency Bands (Gas Furnace Data)")
    print("=" * 60)
    
    # Load real-world data (you'll need to have this file)
    try:
        mat_data = scipy.io.loadmat('data/gasfurnace.mat')
        x = np.array(mat_data['gasfurnace'][0]).flatten()
        y = np.array(mat_data['gasfurnace'][1]).flatten()
        
        print(f"Loaded gas furnace data: {len(x)} samples")
        
    except FileNotFoundError:
        print("Gas furnace data not found, generating synthetic process data...")
        # Generate synthetic process control data
        np.random.seed(123)
        n = 500
        t = np.linspace(0, 100, n)
        
        # Simulate process with multiple time scales
        x = (np.sin(0.1 * t) +           # Slow dynamics
             0.5 * np.sin(0.5 * t) +     # Medium dynamics  
             0.3 * np.sin(2.0 * t) +     # Fast dynamics
             0.2 * np.random.randn(n))   # Noise
        
        # Y responds to X with different lags at different frequencies
        y = np.zeros(n)
        for i in range(n):
            # Slow response (lag 20)
            if i >= 20:
                y[i] += 0.6 * x[i-20]
            # Fast response (lag 5)  
            if i >= 5:
                y[i] += 0.3 * x[i-5]
            # Add noise
            y[i] += 0.2 * np.random.randn()
    
    # Define frequency bands for process control
    process_bands = {
        'slow': (1, 10),        # Slow thermal dynamics
        'medium': (11, 25),     # Medium process dynamics
        'fast': (26, 50),       # Fast control responses
        'rapid': (51, 100)      # Rapid fluctuations
    }
    
    # Analyze with custom bands
    results = mbvl_granger(
        X=x, Y=y,
        fs=250,
        max_lag=50,
        alpha=0.05,
        gamma=0.6,
        bands=process_bands
    )
    
    print_mbvlgranger_results(results)

def example_3_eeg_analysis():
    """Example 3: EEG-style analysis with standard neuroscience bands"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: EEG-Style Analysis")
    print("=" * 60)
    
    # Generate synthetic EEG-like data
    np.random.seed(456)
    fs = 500  # 500 Hz sampling
    duration = 10  # 10 seconds
    n = fs * duration
    t = np.linspace(0, duration, n)
    
    # Create multi-frequency signals
    # Channel X: mixed frequency content
    x = (0.8 * np.sin(2 * np.pi * 10 * t) +    # Alpha (10 Hz)
         0.6 * np.sin(2 * np.pi * 40 * t) +    # Gamma (40 Hz)
         0.4 * np.random.randn(n))              # Noise
    
    # Channel Y: responds to X with frequency-specific lags
    y = np.zeros(n)
    alpha_lag = 25    # 50ms lag for alpha (25 samples at 500Hz)
    gamma_lag = 10    # 20ms lag for gamma (10 samples at 500Hz)
    
    for i in range(n):
        # Alpha coupling
        if i >= alpha_lag:
            y[i] += 0.5 * x[i - alpha_lag]
        # Gamma coupling  
        if i >= gamma_lag:
            y[i] += 0.3 * x[i - gamma_lag]
        # Noise
        y[i] += 0.3 * np.random.randn()
    
    # Standard EEG frequency bands
    eeg_bands = {
        'delta': (1, 4),
        'theta': (4, 8), 
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 50),
        'high_gamma': (50, 80)
    }
    
    # Analyze
    results = mbvl_granger(
        X=x, Y=y,
        fs=fs,
        max_lag=30,
        alpha=0.01,  # Stricter significance
        gamma=0.5,
        bands=eeg_bands,
        adaptive_lag=True  # Use adaptive lag selection
    )
    
    print_mbvlgranger_results(results)
    
    # Show which bands detected causality
    significant_bands = []
    for _, row in results['band_results'].iterrows():
        if row['significant_individual']:
            significant_bands.append(row['interval'])
    
    print(f"\n🧠 Significant EEG bands: {', '.join(significant_bands)}")
    print(f"Expected: 8-13Hz (alpha) and 30-50Hz (low_gamma)")

def example_4_comparison_methods():
    """Example 4: Compare different statistical combination methods"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Statistical Method Comparison")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(789)
    n = 800
    x = np.random.randn(n)
    y = np.zeros(n)
    
    # Add causality in specific frequency bands
    lag = 15
    for t in range(n):
        if t >= lag:
            y[t] = 0.7 * x[t - lag] + 0.25 * np.random.randn()
        else:
            y[t] = 0.25 * np.random.randn()
    
    # Compare different combination methods
    methods = ['fisher', 'stouffer', 'bonferroni']
    
    for method in methods:
        print(f"\n--- {method.upper()} METHOD ---")
        
        results = mbvl_granger(
            X=x, Y=y,
            fs=250,
            max_lag=25,
            combination_method=method,
            alpha=0.05,
            gamma=0.4,
            print_results=False  # Suppress detailed output
        )
        
        print(f"Overall Causality: {results['overall_causality']}")
        print(f"Combined p-value: {results['combined_p_value']:.6f}")
        print(f"Test statistic: {results['test_statistic']:.3f}")

def main():
    """Run all examples"""
    print("MBVL-Granger Basic Usage Examples")
    print("*" * 60)
    
    example_1_simple_analysis()
    example_2_custom_frequency_bands()  
    example_3_eeg_analysis()
    example_4_comparison_methods()
    
    print("\n" + "*" * 60)
    print("All examples completed successfully!")
    print("Next steps:")
    print("1. Try with your own data")
    print("2. Experiment with different frequency bands")
    print("3. Adjust parameters (alpha, gamma, max_lag)")
    print("4. Check out frequency_analysis.py for advanced usage")

if __name__ == "__main__":
    main()