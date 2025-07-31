# MBVL-Granger: MultiBand Variable-Lag Granger Causality Analysis

A Python framework to infer causality between time series using MultiBand Variable-Lag Granger causality with frequency-band decomposition.

Traditional Granger causality assumes fixed time delays between cause and effect. However, for non-stationary time series, this assumption often fails. For example, in neural signals, the delay between brain regions can vary over time due to changing network dynamics. 

We propose MBVL-Granger that allows variable-lags and analyzes causality within specific frequency bands to handle complex, non-stationary time series relationships.

## Installation

You can install our package from PyPI:

```bash
pip install mbvlgranger
```

For the newest version from GitHub:

```bash
git clone https://github.com/Teddy50060/mbvlgranger.git
cd mbvlgranger
pip install -e .
```

## Example: Gas Furnace Data Analysis

First, we load time series data where gas input rate (X) potentially causes CO2 concentration (Y) with variable delays.

```python
import numpy as np
import scipy.io
from mbvlgranger import quick_mbvlgranger

# Load gas furnace data
mat_data = scipy.io.loadmat('data/gasfurnace.mat')
x = np.array(mat_data['gasfurnace'][0]).flatten()  # Gas input rate
y = np.array(mat_data['gasfurnace'][1]).flatten()  # CO2 concentration
```

We use the following function to infer whether X causes Y across different frequency bands:

```python
# Run MBVL-Granger analysis
results = quick_mbvlgranger(
    x=x, y=y,
    fs=250,  # sampling frequency
    max_lag=50,
    bands={
        'slow': (1, 10),       # Slow thermal dynamics
        'medium': (11, 25),    # Medium process dynamics
        'fast': (26, 50),      # Fast control responses
        'rapid': (51, 100)     # Rapid fluctuations
    }
)
```

The result of MBVL-Granger causality analysis:

```python
print(f"Overall Causality: {results['overall_causality']}")
print(f"Combined p-value: {results['combined_p_value']:.6f}")

# Actual output:
# Overall Causality: True
# Combined p-value: 0.000000
```

If `results['overall_causality']` is True, then X MBVL-Granger-causes Y. The `combined_p_value` indicates statistical significance across all frequency bands.

For individual frequency band results:

```python
from mbvlgranger import print_mbvlgranger_results

# Print detailed results for each frequency band
print_mbvlgranger_results(results)

# Actual output:
# VL-Granger Frequency Causality Analysis Results
# ==================================================
# Overall Causality: True
# Combined P-value: 0.000000
# Method: fisher
# Valid Bands: 4
# 
# Individual Band Results:
# --------------------------------------------------
# 1-10Hz   | F=32.014 | p=0.000000 | BIC= 0.746 | Lag= 3 | Sig=YES
# 11-25Hz  | F= 0.000 | p=1.000000 | BIC=-27.633 | Lag= 4 | Sig=NO
# 26-50Hz  | F= 0.996 | p=0.492149 | BIC= -1.279 | Lag= 4 | Sig=NO
# 51-100Hz | F= 0.663 | p=0.952331 | BIC= -1.492 | Lag=12 | Sig=NO
```

## Core Analysis

For more control over the analysis:

```python
from mbvlgranger import VLGrangerCausality

# Single time series analysis with core algorithm
analyzer = VLGrangerCausality()
result = analyzer.analyze_causality(Y=y, X=x, max_lag=25, gamma=0.5)

print(f"Causality detected: {result['XgCsY']}")
print(f"Detected lag: {result['following_result']['opt_delay']}")
print(f"BIC ratio: {result['BIC_diff_ratio']:.3f}")
```
