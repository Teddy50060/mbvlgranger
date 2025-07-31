"""
VL-Granger: Variable-Lag Granger Causality Analysis

A Python package for performing Variable-Lag Granger Causality analysis with 
frequency-band decomposition and advanced statistical testing.

Main Classes:
- VLGrangerCausality: Core VL-Granger analysis
- MultiBandVLGranger: Frequency-band specific analysis

Main Functions:
- vlf_granger: Convenient frequency-band analysis
- quick_vlgranger: Ultra-simple one-liner analysis
- print_vlgranger_results: Pretty-print results
"""

from .core import VLGrangerCausality, vl_granger_causality
from .frequency_analysis import MultiBandVLGranger, multiband_vl_granger_analysis
from .statistical_tests import mbvl_granger, quick_mbvlgranger, print_mbvlgranger_results
from .data_generation import generate_complete_dataset, load_dataset_file
from .evaluation import run_comprehensive_test

__version__ = "0.1.3"
__author__ = "Chakattrai Sookkongwaree"
__email__ = "6632033821@student.chula.ac.th"

# Default frequency bands for EEG analysis
DEFAULT_EEG_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8), 
    'alpha': (8, 13),
    'beta': (13, 30),
    'low_gamma': (30, 50),
    'high_gamma': (50, 80)
}

# Make key classes and functions available at package level
__all__ = [
    # Core classes
    'VLGrangerCausality',
    'MultiBandVLGranger',
    
    # Main analysis functions
    'mbvl_granger',
    'quick_mbvlgranger',
    'vl_granger_causality',
    'multiband_vl_granger_analysis',
    
    # Utility functions
    'print_mbvlgranger_results',
    
    # Data and evaluation
    'generate_complete_dataset',
    'load_dataset_file',
    'run_comprehensive_test',
    
    # Constants
    'DEFAULT_EEG_BANDS',
]