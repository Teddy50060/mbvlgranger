"""
Tests for core MBVL-Granger functionality
"""

import pytest
import numpy as np
from mbvlgranger.core import VLGrangerCausality, vl_granger_causality

class TestVLGrangerCausality:
    
    def test_basic_initialization(self):
        """Test basic class initialization"""
        analyzer = VLGrangerCausality()
        assert analyzer.last_result is None
    
    def test_simple_causality_detection(self):
        """Test detection of simple causality"""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.zeros(n)
        
        # Create clear causality
        lag = 5
        for t in range(n):
            if t >= lag:
                y[t] = 0.8 * x[t - lag] + 0.2 * np.random.randn()
            else:
                y[t] = 0.2 * np.random.randn()
        
        analyzer = VLGrangerCausality()
        result = analyzer.analyze_causality(y, x, max_lag=10)
        
        # Should detect causality
        assert result['XgCsY'] == True or result['XgCsY_ftest'] == True
        
        # Should detect approximate lag
        detected_lag = result['following_result']['opt_delay']
        assert abs(detected_lag - lag) <= 3  # Within 3 samples
    
    def test_no_causality(self):
        """Test that no causality is correctly identified"""
        np.random.seed(123)
        n = 200
        x = np.random.randn(n)
        y = np.random.randn(n)  # Independent
        
        analyzer = VLGrangerCausality()
        result = analyzer.analyze_causality(y, x, max_lag=10, gamma=0.5)
        
        # Should not detect causality
        assert result['XgCsY'] == False
        assert result['p_val'] > 0.01  # Should have high p-value
    
    def test_convenience_function(self):
        """Test convenience function"""
        np.random.seed(456)
        n = 100
        x = np.random.randn(n)
        y = 0.5 * x + 0.3 * np.random.randn(n)  # Instantaneous causality
        
        result = vl_granger_causality(y, x, max_lag=5)
        
        assert isinstance(result, dict)
        assert 'XgCsY' in result
        assert 'p_val' in result

def test_input_validation():
    """Test input validation"""
    analyzer = VLGrangerCausality()
    
    # Test length mismatch
    with pytest.raises(ValueError):
        analyzer.analyze_causality([1, 2, 3], [1, 2])
    
    # Test too short series
    with pytest.raises(ValueError):
        analyzer.analyze_causality([1, 2], [3, 4])