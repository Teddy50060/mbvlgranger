#!/usr/bin/env python3
"""
Package Integrity Validation Script for MBVL-Granger

This script tests all major components to ensure the package works correctly
after renaming from vlgranger to mbvlgranger.
"""

import sys
import numpy as np
import traceback

def test_imports():
    """Test all import statements"""
    print("🔍 Testing Imports...")
    
    try:
        # Test main package import
        import mbvlgranger
        print("  ✅ Main package import: mbvlgranger")
        
        # Test core imports
        from mbvlgranger import VLGrangerCausality, vl_granger_causality
        print("  ✅ Core class imports")
        
        # Test main function imports
        from mbvlgranger import mbvl_granger, quick_mbvlgranger, print_mbvlgranger_results
        print("  ✅ Main function imports")
        
        # Test class imports
        from mbvlgranger import MultiBandVLGranger
        print("  ✅ MultiBand class import")
        
        # Test utility imports
        from mbvlgranger import generate_complete_dataset, run_comprehensive_test
        print("  ✅ Utility function imports")
        
        print("  🎉 All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print(f"  📍 Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_core_functionality():
    """Test core VL-Granger functionality"""
    print("🧠 Testing Core VL-Granger...")
    
    try:
        from mbvlgranger import VLGrangerCausality
        
        # Generate simple test data
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = np.zeros(n)
        
        # Create causality: y[t] = 0.8 * x[t-5] + noise
        lag = 5
        for t in range(n):
            if t >= lag:
                y[t] = 0.8 * x[t - lag] + 0.2 * np.random.randn()
            else:
                y[t] = 0.2 * np.random.randn()
        
        # Test analysis
        analyzer = VLGrangerCausality()
        result = analyzer.analyze_causality(y, x, max_lag=10)
        
        # Check result structure
        required_keys = ['XgCsY', 'p_val', 'BIC_diff_ratio', 'following_result']
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing key in result: {key}")
        
        print(f"  ✅ Core analysis completed")
        print(f"  📊 Detected causality: {result['XgCsY']}")
        print(f"  📊 P-value: {result['p_val']:.6f}")
        print(f"  📊 Detected lag: {result['following_result']['opt_delay']}")
        print("  🎉 Core functionality working!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Core functionality error: {e}")
        print(f"  📍 Traceback: {traceback.format_exc()}")
        return False

def test_main_functions():
    """Test main MBVL-Granger functions"""
    print("🚀 Testing Main Functions...")
    
    try:
        from mbvlgranger import mbvl_granger, quick_mbvlgranger, print_mbvlgranger_results
        
        # Generate test data
        np.random.seed(123)
        n = 300
        x = np.random.randn(n)
        y = 0.6 * x + 0.4 * np.random.randn(n)  # Instantaneous causality
        
        # Test mbvl_granger
        result1 = mbvl_granger(x, y, fs=250, max_lag=20)
        print("  ✅ mbvl_granger function works")
        
        # Test quick_mbvlgranger  
        result2 = quick_mbvlgranger(x, y, fs=250, max_lag=20, print_results=False)
        print("  ✅ quick_mbvlgranger function works")
        
        # Test print function
        print("  📊 Testing print function:")
        print_mbvlgranger_results(result2)
        print("  ✅ print_mbvlgranger_results function works")
        
        print("  🎉 Main functions working!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Main functions error: {e}")
        print(f"  📍 Traceback: {traceback.format_exc()}")
        return False

def test_multiband_class():
    """Test MultiBand class"""
    print("🎵 Testing MultiBand Class...")
    
    try:
        from mbvlgranger import MultiBandVLGranger
        
        # Generate test data with multiple frequencies
        np.random.seed(456)
        fs = 500
        t = np.linspace(0, 4, fs * 4)
        
        # Multi-frequency signal
        x = (np.sin(2 * np.pi * 10 * t) +    # 10 Hz
             np.sin(2 * np.pi * 40 * t) +    # 40 Hz  
             0.3 * np.random.randn(len(t)))
        
        y = 0.5 * x + 0.3 * np.random.randn(len(t))
        
        # Test single band analysis
        analyzer = MultiBandVLGranger()
        band_result = analyzer.single_band_vl_granger(
            x, y, fs=fs, 
            frequency_band=(8, 13),  # Alpha band
            max_lag=10
        )
        
        print("  ✅ Single band analysis works")
        print(f"  📊 Band causality: {band_result['causality']}")
        print(f"  📊 Band quality: {band_result['quality']}")
        
        print("  🎉 MultiBand class working!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ MultiBand class error: {e}")
        print(f"  📍 Traceback: {traceback.format_exc()}")
        return False

def test_your_exact_usage():
    """Test your exact usage pattern"""
    print("🎯 Testing Your Exact Usage Pattern...")
    
    try:
        from mbvlgranger import quick_mbvlgranger
        
        # Simulate your gas furnace data
        np.random.seed(789)
        n = 296
        x = np.random.randn(n)
        y = np.zeros(n)
        
        # Add some causality
        for i in range(n):
            if i >= 4:
                y[i] = 0.7 * x[i-4] + 0.3 * np.random.randn()
            else:
                y[i] = 0.3 * np.random.randn()
        
        # Your exact usage
        results = quick_mbvlgranger(
            x=x, y=y,
            fs=250,
            max_lag=50,
            bands={
                'slow': (1, 10),
                'medium': (11, 25),
                'fast': (26, 50),  
                'rapid': (51, 100)
            },
            print_results=False  # Suppress output for test
        )
        
        print("  ✅ Your exact usage pattern works!")
        print(f"  📊 Overall causality: {results['overall_causality']}")
        print(f"  📊 Combined p-value: {results['combined_p_value']:.6f}")
        print(f"  📊 Number of bands analyzed: {len(results['band_results'])}")
        
        print("  🎉 Your workflow is functional!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Your usage pattern error: {e}")
        print(f"  📍 Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("🔍 MBVL-GRANGER PACKAGE INTEGRITY VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Import Validation", test_imports),
        ("Core Functionality", test_core_functionality),
        ("Main Functions", test_main_functions),
        ("MultiBand Class", test_multiband_class),
        ("Your Exact Usage", test_your_exact_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Your package is ready!")
        return True
    else:
        print("⚠️  Some tests failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)