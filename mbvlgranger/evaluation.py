import numpy as np
import pandas as pd
import scipy.io
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_dataset_file(filepath: str) -> Dict:
    """
    Load a single .mat file from the test dataset
    """
    try:
        data = scipy.io.loadmat(filepath)
        return {
            'x': data['x'].flatten(),
            'y': data['y'].flatten(),
            'fs': float(data['fs'][0][0]),
            'metadata': data['metadata']
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_metadata(metadata) -> Dict:
    """
    Extract metadata from .mat file format
    """
    if isinstance(metadata, np.ndarray):
        # Handle structured array from .mat file
        meta_dict = {}
        for field in metadata.dtype.names:
            value = metadata[field][0][0]
            if isinstance(value, np.ndarray):
                if value.dtype.char == 'U':  # Unicode string
                    meta_dict[field] = str(value)
                elif value.size == 1:
                    meta_dict[field] = value.item()
                else:
                    # Convert arrays to lists for JSON compatibility
                    meta_dict[field] = value.tolist()
            else:
                meta_dict[field] = value
        return meta_dict
    elif isinstance(metadata, dict):
        # Already a dictionary - ensure lists are properly handled
        clean_dict = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                clean_dict[key] = value.tolist()
            else:
                clean_dict[key] = value
        return clean_dict
    else:
        # Try to convert to dict
        try:
            return dict(metadata) if hasattr(metadata, '__iter__') else {}
        except:
            return {}

def get_expected_band_causality(dataset_type: str, metadata: Dict) -> Dict:
    """
    Determine which bands should show causality based on dataset type
    """
    band_causality = {
        'delta': False, 'theta': False, 'alpha': False, 
        'beta': False, 'low_gamma': False, 'high_gamma': False
    }
    
    if dataset_type in ['random', 'ar']:  # Added 'ar' for AR noise
        # No bands should show causality for both random and AR noise
        return band_causality
    
    elif dataset_type in ['basic', 'variable']:  # Changed from 'basic_causation', 'variable_lag'
        # Broadband causality - all bands should show some effect
        for band in band_causality:
            band_causality[band] = True
        return band_causality
    
    elif dataset_type == 'broadband':
        # 1-50 Hz signal - affects delta, theta, alpha, beta, low_gamma
        for band in ['delta', 'theta', 'alpha', 'beta', 'low_gamma']:
            band_causality[band] = True
        return band_causality
    
    elif dataset_type == 'multifreq':
        # 10, 40, 80 Hz components
        frequencies = metadata.get('frequencies', [10, 40, 80])
        
        # Map frequencies to bands
        freq_to_band = {
            10: 'alpha',      # 10 Hz -> alpha (8-13 Hz)
            40: 'low_gamma',  # 40 Hz -> low_gamma (30-50 Hz)
            80: 'high_gamma'  # 80 Hz -> high_gamma (50-80 Hz)
        }
        
        for freq in frequencies:
            if freq in freq_to_band:
                band_causality[freq_to_band[freq]] = True
        
        return band_causality
    
    return band_causality

def get_expected_lag(dataset_type: str, metadata: Dict, band: str) -> float:
    """
    Get expected lag for a specific band based on dataset type
    """
    if dataset_type in ['random', 'ar']:  # Added 'ar' for AR noise
        return 0  # No meaningful lag for both random and AR noise
    
    elif dataset_type == 'basic':  # Changed from 'basic_causation'
        return metadata.get('true_lag', 20)
    
    elif dataset_type == 'variable':  # Changed from 'variable_lag'
        return metadata.get('true_lag_mean', 15)
    
    elif dataset_type == 'broadband':
        return metadata.get('true_lag', 7)
    
    elif dataset_type == 'multifreq':
        # Get frequency-specific lags
        frequencies = metadata.get('frequencies', [10, 40, 80])
        lags = metadata.get('true_lags', [15, 8, 4])
        
        # Map bands to expected lags
        band_to_freq = {
            'alpha': 10,      # 10 Hz
            'low_gamma': 40,  # 40 Hz
            'high_gamma': 80  # 80 Hz
        }
        
        if band in band_to_freq:
            freq = band_to_freq[band]
            if freq in frequencies:
                freq_idx = frequencies.index(freq)
                return lags[freq_idx]
        
        return 0  # No expected lag for this band
    
    return 0



def test_single_file(filepath: str, max_lag: int = 30) -> Dict:
    """
    Test a single dataset file
    """
    # Load data
    data = load_dataset_file(filepath)
    if data is None:
        return None
    
    # Extract info
    filename = os.path.basename(filepath)
    dataset_type = filename.split('_')[0]
    metadata = extract_metadata(data['metadata'])
    
    # Run VL-Granger analysis
    try:
        #max_lag = get_appropriate_max_lag(dataset_type, data['fs'])
        results = vlf_granger(
            data['x'], data['y'], 
            fs=data['fs'], 
            max_lag=25,
            #combination_method='fisher',
            alpha=0.01,
            gamma=0.6,
            adaptive_lag=False,
            noise_aware=False,
        )
        
        # Extract band results
        band_results = results['band_results']
        
        # Get ground truth
        expected_causality = get_expected_band_causality(dataset_type, metadata)
        overall_expected = any(expected_causality.values())
        
        # Calculate accuracy metrics
        band_predictions = []
        band_ground_truth = []
        lag_errors = []
        
        for _, row in band_results.iterrows():
            band_name = row['interval'].replace('Hz', '').replace('-', '_').lower()
            
            # Map interval to band name
            interval_to_band = {
                '1_4': 'delta', '4_8': 'theta', '8_13': 'alpha',
                '13_30': 'beta', '30_50': 'low_gamma', '50_80': 'high_gamma'
            }
            
            if band_name in interval_to_band:
                band_key = interval_to_band[band_name]
                predicted = row['significant_individual']
                expected = expected_causality[band_key]
                
                band_predictions.append(predicted)
                band_ground_truth.append(expected)
                
                # Calculate lag error if both predicted and expected causality
                if predicted and expected:
                    expected_lag = get_expected_lag(dataset_type, metadata, band_key)
                    detected_lag = row['detected_lag']
                    if not np.isnan(detected_lag) and expected_lag > 0:
                        lag_errors.append(abs(detected_lag - expected_lag))
        
        # Overall accuracy
        overall_predicted = results['overall_causality']
        overall_correct = (overall_predicted == overall_expected)
        
        # Band-level accuracy
        band_accuracy = accuracy_score(band_ground_truth, band_predictions) if band_predictions else 0
        
        return {
            'filename': filename,
            'dataset_type': dataset_type,
            'metadata': metadata,
            'overall_predicted': overall_predicted,
            'overall_expected': overall_expected,
            'overall_correct': overall_correct,
            'band_predictions': band_predictions,
            'band_ground_truth': band_ground_truth,
            'band_accuracy': band_accuracy,
            'lag_errors': lag_errors,
            'combined_p_value': results['combined_p_value'],
            'band_results': band_results,
            'full_results': results
        }
        
    except Exception as e:
        print(f"Error analyzing {filename}: {e}")
        return {
            'filename': filename,
            'dataset_type': dataset_type,
            'error': str(e),
            'overall_correct': False,
            'band_accuracy': 0,
            'lag_errors': []
        }

def test_dataset_directory(dataset_dir: str, max_lag: int = 30) -> pd.DataFrame:
    """
    Test all files in the dataset directory
    """
    # Find all .mat files
    mat_files = glob.glob(os.path.join(dataset_dir, "*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {dataset_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(mat_files)} files to test")
    
    # Test each file
    results = []
    failed_count = 0
    
    for i, filepath in enumerate(mat_files):
        filename = os.path.basename(filepath)
        print(f"Testing {i+1}/{len(mat_files)}: {filename}")
        
        result = test_single_file(filepath, max_lag)
        if result:
            # Ensure all required fields are present and valid
            required_fields = ['overall_predicted', 'overall_expected', 'overall_correct']
            valid_result = True
            
            for field in required_fields:
                if field not in result:
                    print(f"  Warning: Missing field {field} in {filename}")
                    valid_result = False
                elif result[field] is None:
                    print(f"  Warning: None value for {field} in {filename}")
                    result[field] = False  # Convert None to False
                elif isinstance(result[field], float) and np.isnan(result[field]):
                    print(f"  Warning: NaN value for {field} in {filename}")
                    result[field] = False  # Convert NaN to False
            
            if valid_result:
                results.append(result)
            else:
                failed_count += 1
                print(f"  Skipped {filename} due to invalid results")
        else:
            failed_count += 1
            print(f"  Failed to process {filename}")
    
    print(f"Successfully processed: {len(results)}/{len(mat_files)} files")
    print(f"Failed: {failed_count} files")
    
    return pd.DataFrame(results)

def calculate_performance_metrics(test_results: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive performance metrics
    """
    metrics = {}
    
    # Overall accuracy by dataset type
    for dataset_type in test_results['dataset_type'].unique():
        subset = test_results[test_results['dataset_type'] == dataset_type]
        accuracy = subset['overall_correct'].mean()
        metrics[f'{dataset_type}_accuracy'] = accuracy
    
    # Overall metrics
    all_overall_pred = test_results['overall_predicted'].tolist()
    all_overall_true = test_results['overall_expected'].tolist()
    
    if all_overall_pred and all_overall_true:
        metrics['overall_accuracy'] = accuracy_score(all_overall_true, all_overall_pred)
        metrics['overall_precision'] = precision_score(all_overall_true, all_overall_pred, zero_division=0)
        metrics['overall_recall'] = recall_score(all_overall_true, all_overall_pred, zero_division=0)
        metrics['overall_f1'] = f1_score(all_overall_true, all_overall_pred, zero_division=0)
    
    # Band-level metrics
    all_band_pred = []
    all_band_true = []
    
    for _, row in test_results.iterrows():
        if isinstance(row['band_predictions'], list) and isinstance(row['band_ground_truth'], list):
            all_band_pred.extend(row['band_predictions'])
            all_band_true.extend(row['band_ground_truth'])
    
    if all_band_pred and all_band_true:
        metrics['band_accuracy'] = accuracy_score(all_band_true, all_band_pred)
        metrics['band_precision'] = precision_score(all_band_true, all_band_pred, zero_division=0)
        metrics['band_recall'] = recall_score(all_band_true, all_band_pred, zero_division=0)
        metrics['band_f1'] = f1_score(all_band_true, all_band_pred, zero_division=0)
    
    # Lag accuracy for frequency-specific datasets
    multifreq_results = test_results[test_results['dataset_type'] == 'multifreq']
    broadband_results = test_results[test_results['dataset_type'] == 'broadband']
    
    for dataset_type, subset in [('multifreq', multifreq_results), ('broadband', broadband_results)]:
        if not subset.empty:
            all_lag_errors = []
            for _, row in subset.iterrows():
                if isinstance(row['lag_errors'], list):
                    all_lag_errors.extend(row['lag_errors'])
            
            if all_lag_errors:
                metrics[f'{dataset_type}_lag_error_mean'] = np.mean(all_lag_errors)
                metrics[f'{dataset_type}_lag_error_std'] = np.std(all_lag_errors)
    
    return metrics

def print_performance_report(test_results: pd.DataFrame, metrics: Dict):
    """
    Print a comprehensive performance report
    """
    
    print("VL-GRANGER PERFORMANCE EVALUATION REPORT")
    print("=" * 60)
    
    # Overall Performance
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Accuracy:  {metrics.get('overall_accuracy', 0):.3f}")
    print(f"  Precision: {metrics.get('overall_precision', 0):.3f}")
    print(f"  Recall:    {metrics.get('overall_recall', 0):.3f}")
    print(f"  F1 Score:  {metrics.get('overall_f1', 0):.3f}")
    
    # Performance by Dataset Type (including AR noise)
    print(f"\nPERFORMANCE BY DATASET TYPE:")
    for dataset_type in ['random', 'ar', 'basic', 'variable', 'broadband', 'multifreq']:
        key = f'{dataset_type}_accuracy'
        if key in metrics:
            subset = test_results[test_results['dataset_type'] == dataset_type]
            n_files = len(subset)
            accuracy = metrics[key]
            print(f"  {dataset_type:15}: {accuracy:.3f} ({n_files} files)")
    
    # Band-level Performance
    print(f"\nBAND-LEVEL PERFORMANCE:")
    print(f"  Accuracy:  {metrics.get('band_accuracy', 0):.3f}")
    print(f"  Precision: {metrics.get('band_precision', 0):.3f}")
    print(f"  Recall:    {metrics.get('band_recall', 0):.3f}")
    print(f"  F1 Score:  {metrics.get('band_f1', 0):.3f}")
    
    # Lag Detection Performance
    print(f"\nLAG DETECTION PERFORMANCE:")
    for dataset_type in ['multifreq', 'broadband']:
        mean_key = f'{dataset_type}_lag_error_mean'
        std_key = f'{dataset_type}_lag_error_std'
        if mean_key in metrics:
            mean_error = metrics[mean_key]
            std_error = metrics[std_key]
            print(f"  {dataset_type:10}: {mean_error:.2f} ± {std_error:.2f} samples")
    
    # False Positive Rate (now includes both random and AR noise)
    print(f"\nFALSE POSITIVE RATE:")
    
    # Random Noise False Positive Rate
    random_results = test_results[test_results['dataset_type'] == 'random']
    if not random_results.empty:
        random_fp_rate = random_results['overall_predicted'].mean()
        print(f"  Random Noise: {random_fp_rate:.3f} ({random_fp_rate*100:.1f}%)")
    
    # AR Noise False Positive Rate
    ar_results = test_results[test_results['dataset_type'] == 'ar']
    if not ar_results.empty:
        ar_fp_rate = ar_results['overall_predicted'].mean()
        print(f"  AR Noise:     {ar_fp_rate:.3f} ({ar_fp_rate*100:.1f}%)")
    
    # Combined Noise False Positive Rate
    noise_results = test_results[test_results['dataset_type'].isin(['random', 'ar'])]
    if not noise_results.empty:
        combined_fp_rate = noise_results['overall_predicted'].mean()
        print(f"  Combined:     {combined_fp_rate:.3f} ({combined_fp_rate*100:.1f}%)")
    
    # Confusion Matrix
    if 'overall_accuracy' in metrics:
        all_true = test_results['overall_expected'].tolist()
        all_pred = test_results['overall_predicted'].tolist()
        cm = confusion_matrix(all_true, all_pred)
        
        print(f"\nCONFUSION MATRIX (Overall):")
        print(f"               Predicted")
        print(f"              No   Yes")
        print(f"Actual No   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Yes  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Dataset file counts (including AR noise)
    print(f"\nDATASET FILE COUNTS:")
    type_counts = test_results['dataset_type'].value_counts()
    for dataset_type in ['random', 'ar', 'basic', 'variable', 'broadband', 'multifreq']:
        count = type_counts.get(dataset_type, 0)
        print(f"  {dataset_type:15}: {count:3d} files")
    
    # Additional breakdown for noise types
    print(f"\nNOISE TYPE BREAKDOWN:")
    noise_counts = test_results[test_results['dataset_type'].isin(['random', 'ar'])]['dataset_type'].value_counts()
    total_noise = len(test_results[test_results['dataset_type'].isin(['random', 'ar'])])
    
    if 'random' in noise_counts:
        print(f"  Random Noise:   {noise_counts['random']:3d} files")
    if 'ar' in noise_counts:
        print(f"  AR Noise:       {noise_counts['ar']:3d} files")
    print(f"  Total Noise:    {total_noise:3d} files")
    
    # Causality type breakdown
    causality_counts = test_results[test_results['dataset_type'].isin(['basic', 'variable', 'broadband', 'multifreq'])]['dataset_type'].value_counts()
    total_causality = len(test_results[test_results['dataset_type'].isin(['basic', 'variable', 'broadband', 'multifreq'])])
    
    print(f"\nCAUSALITY TYPE BREAKDOWN:")
    for ctype in ['basic', 'variable', 'broadband', 'multifreq']:
        if ctype in causality_counts:
            print(f"  {ctype.capitalize():15}: {causality_counts[ctype]:3d} files")
    print(f"  Total Causality: {total_causality:3d} files")

def run_comprehensive_test(dataset_dir: str = "vlgranger_test_data", max_lag: int = 30):
    """
    Run comprehensive test on the entire dataset
    """
    print("Starting comprehensive VL-Granger evaluation...")
    
    # Test all files
    test_results = test_dataset_directory(dataset_dir, max_lag)
    
    if test_results.empty:
        print("No results to analyze")
        return
    
    # Calculate metrics
    metrics = calculate_performance_metrics(test_results)
    
    # Print report
    print_performance_report(test_results, metrics)
    
    # Save results
    test_results.to_csv('vlgranger_test_results.csv', index=False)
    
    print(f"\nDetailed results saved to: vlgranger_test_results.csv")
    print(f"Total files tested: {len(test_results)}")
    
    return test_results, metrics

# Example usage
if __name__ == "__main__":
    # Run the comprehensive test
    results, metrics = run_comprehensive_test("vlgranger_test_data")
    
    # Print final F1 score
    print(f"\nFINAL F1 SCORE: {metrics.get('overall_f1', 0):.3f}")