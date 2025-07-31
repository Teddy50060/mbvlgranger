#!/usr/bin/env python3
"""
Gas Furnace Demo - Reproducing your exact workflow
"""

import numpy as np
import scipy.io
from mbvlgranger import quick_mbvlgranger

def main():
    """Reproduce your exact gas furnace analysis"""
    print("Gas Furnace MBVL-Granger Analysis Demo")
    print("=" * 40)
    
    # Load the data (you'll need to provide this file)
    try:
        mat_data = scipy.io.loadmat('data/gasfurnace.mat')
        print("✅ Loaded gasfurnace.mat successfully")
        
        x = np.array(mat_data['gasfurnace'][0]).flatten()
        y = np.array(mat_data['gasfurnace'][1]).flatten()
        
        print(f"Data shape: X={x.shape}, Y={y.shape}")
        
    except FileNotFoundError:
        print("❌ gasfurnace.mat not found. Generating synthetic data...")
        # Generate synthetic furnace-like data
        np.random.seed(42)
        n = 296  # Typical gas furnace dataset size
        t = np.linspace(0, 300, n)
        
        # Simulate furnace input (gas rate)
        x = np.sin(0.05 * t) + 0.3 * np.random.randn(n)
        
        # Simulate furnace output (CO2 concentration) with delay
        y = np.zeros(n)
        for i in range(n):
            if i >= 4:  # 4-sample delay
                y[i] = 0.7 * x[i-4] + 0.2 * np.random.randn()
            else:
                y[i] = 0.2 * np.random.randn()
    
    # Your exact analysis
    results = quick_mbvlgranger(
        x=x, y=y,
        fs=250,  # sampling rate 
        max_lag=50,
        bands={
            'slow': (1, 10),       # Slow thermal dynamics
            'medium': (11, 25),    # Medium process dynamics
            'fast': (26, 50),      # Fast control responses
            'rapid': (51, 100)     # Rapid fluctuations
        }
    )
    
    print("\n🎯 Analysis Results:")
    print(f"Overall Causality: {results['overall_causality']}")
    print(f"Combined p-value: {results['combined_p_value']:.6f}")
    
    # Show significant bands
    significant_bands = []
    for _, row in results['band_results'].iterrows():
        if row['significant_individual']:
            significant_bands.append(row['interval'])
    
    if significant_bands:
        print(f"Significant bands: {', '.join(significant_bands)}")
    else:
        print("No significant frequency bands detected")

if __name__ == "__main__":
    main()