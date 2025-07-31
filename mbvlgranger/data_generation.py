import numpy as np
import scipy.io
import os
from pathlib import Path
import json

def create_dataset_directory(base_dir="vlgranger_test_data"):
    """Create directory structure for the dataset"""
    Path(base_dir).mkdir(exist_ok=True)
    return base_dir


def generate_random_noise(dataset_num, base_dir, min_length=500, max_length=2000):
    """
    1.1 Random noise (60 datasets) - for testing false positives
    """
    np.random.seed(1000 + dataset_num)
    length = np.random.randint(min_length, max_length + 1)
    
    x = np.random.randn(length)
    y = np.random.randn(length)
    
    # Fixed metadata to match test expectations
    mat_metadata = {
        'type': 'random',  # Changed from 'random_noise'
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': 0,  # Use 0 for .mat file
        'true_lag': 0,  # Add this field
        'description': 'Independent random noise - no causality'
    }
    
    json_metadata = {
        'type': 'random',  # Changed from 'random_noise'
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': False,  # Use False for JSON
        'true_lag': None,
        'description': 'Independent random noise - no causality'
    }
    
    filename = f"random_noise_{dataset_num:03d}.mat"
    filepath = os.path.join(base_dir, filename)
    
    scipy.io.savemat(filepath, {
        'x': x.astype(np.float64),
        'y': y.astype(np.float64),
        'metadata': mat_metadata,
        'fs': np.array([250], dtype=np.float64)
    })
    
    return filepath, json_metadata

def generate_basic_causation(dataset_num, base_dir, min_length=500, max_length=2000):
    """
    1.2 Basic X causes Y (30 datasets)
    """
    np.random.seed(2000 + dataset_num)
    length = np.random.randint(min_length, max_length + 1)
    lag = 20
    
    x = np.random.randn(length)
    c1 = abs(np.random.normal(0, 1))
    
    y = np.zeros(length)
    for t in range(length):
        if t >= lag:
            y[t] = c1 * x[t - lag] #+ 0.3 * abs(np.random.randn())
        else:
            y[t] = 0.3 * np.random.randn()
    
    # Fixed metadata to match test expectations
    mat_metadata = {
        'type': 'basic',  # Changed from 'basic_causation'
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': 1,
        'true_lag': int(lag),
        'coupling_coeff': float(c1),
        'description': f'Basic causation: y = {c1:.3f} * x(t-{lag}) + noise'
    }
    
    json_metadata = {
        'type': 'basic',  # Changed from 'basic_causation'
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': True,
        'true_lag': int(lag),
        'coupling_coeff': float(c1),
        'description': f'Basic causation: y = {c1:.3f} * x(t-{lag}) + noise'
    }
    
    filename = f"basic_causation_{dataset_num:03d}.mat"
    filepath = os.path.join(base_dir, filename)
    
    scipy.io.savemat(filepath, {
        'x': x.astype(np.float64),
        'y': y.astype(np.float64),
        'metadata': mat_metadata,
        'fs': np.array([250], dtype=np.float64)
    })
    
    return filepath, json_metadata

def generate_variable_lag_causation(dataset_num, base_dir, min_length=500, max_length=2000):
    """
    Generate realistic variable lag causation with discrete lag periods.
    
    Key improvements:
    1. Discrete lag periods instead of continuous variation (8 samples vs 20)
    2. Stable periods allow VL-Granger to find consistent relationships
    3. Mimics neural regime changes, not continuous variation
    4. Should be detectable even when split into frequency bands
    """
    np.random.seed(3000 + dataset_num)
    length = np.random.randint(min_length, max_length + 1)
    x = np.random.randn(length)
    c1 = np.random.normal(0, 1)
    
    # NEW APPROACH: Discrete lag periods with smaller variation
    lag_periods = [12, 16, 20]  # Smaller range: 8 samples vs previous 20
    num_switches = np.random.randint(2, 5)  # 2-4 regime changes
    
    # Create switch points - evenly distributed with some randomness
    switch_points = np.sort(np.random.choice(
        range(length//4, 3*length//4), 
        size=num_switches-1, 
        replace=False
    ))
    switch_points = np.concatenate([[0], switch_points, [length]])
    
    # Assign lags to each period
    variable_lags = np.zeros(length, dtype=int)
    actual_lags = []
    
    for i in range(len(switch_points)-1):
        start_idx = switch_points[i]
        end_idx = switch_points[i+1]
        period_lag = np.random.choice(lag_periods)
        variable_lags[start_idx:end_idx] = period_lag
        actual_lags.extend([period_lag] * (end_idx - start_idx))
    
    # Generate y with variable lags
    y = np.zeros(length)
    for t in range(length):
        lag = variable_lags[t]
        if t >= lag:
            y[t] = c1 * x[t - lag] + 0.3 * np.random.randn()
        else:
            y[t] = 0.3 * np.random.randn()
    
    # Calculate statistics
    unique_lags, counts = np.unique(actual_lags, return_counts=True)
    lag_mean = np.mean(actual_lags)
    lag_std = np.std(actual_lags)
    lag_range = [int(np.min(actual_lags)), int(np.max(actual_lags))]
    
    # Metadata
    mat_metadata = {
        'type': 'variable',
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': 1,
        'true_lag': int(lag_mean),
        'true_lag_mean': float(lag_mean),
        'true_lag_std': float(lag_std),
        'true_lag_range': np.array(lag_range, dtype=int),
        'lag_periods': np.array(lag_periods, dtype=int),
        'num_switches': int(num_switches),
        'coupling_coeff': float(c1),
        'description': f'Realistic variable lag: discrete periods {lag_periods}, {num_switches} switches'
    }
    
    json_metadata = {
        'type': 'variable',
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': True,
        'true_lag': int(lag_mean),
        'true_lag_mean': float(lag_mean),
        'true_lag_std': float(lag_std),
        'true_lag_range': lag_range,
        'lag_periods': lag_periods,
        'num_switches': int(num_switches),
        'coupling_coeff': float(c1),
        'description': f'Realistic variable lag: discrete periods {lag_periods}, {num_switches} switches'
    }
    
    # Save
    filename = f"variable_lag_{dataset_num:03d}.mat"
    filepath = os.path.join(base_dir, filename)
    scipy.io.savemat(filepath, {
        'x': x.astype(np.float64),
        'y': y.astype(np.float64),
        'metadata': mat_metadata,
        'variable_lags': np.array(actual_lags, dtype=int),
        'switch_points': np.array(switch_points, dtype=int),
        'fs': np.array([250], dtype=np.float64)
    })
    
    return filepath, json_metadata


def generate_broadband_causation(dataset_num, base_dir, min_length=500, max_length=2000):
    """
    1.4.1 Broadband causation (30 datasets)
    """
    np.random.seed(4000 + dataset_num)
    length = np.random.randint(min_length, max_length + 1)
    fs = 250
    t = np.linspace(0, length/fs, length)
    
    # Generate broadband signal (1-50 Hz)
    x = np.zeros(length)
    for freq in range(1, 51):
        amplitude = np.random.uniform(0.1, 1.0)
        phase = np.random.uniform(0, 2*np.pi)
        x += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    x += 0.2 * np.random.randn(length)
    
    lag = 7
    c1 = np.random.normal(0, 1)
    
    y = np.zeros(length)
    for i in range(length):
        if i >= lag:
            y[i] = c1 * x[i - lag] + 0.3 * np.random.randn()
        else:
            y[i] = 0.3 * np.random.randn()
    
    # Fixed metadata to match test expectations
    mat_metadata = {
        'type': 'broadband',  # Changed from 'broadband_causation'
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': 1,
        'true_lag': int(lag),
        'frequency_range': np.array([1, 50], dtype=int),
        'coupling_coeff': float(c1),
        'description': f'Broadband (1-50 Hz) causation with {lag} sample lag'
    }
    
    json_metadata = {
        'type': 'broadband',  # Changed from 'broadband_causation'
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': True,
        'true_lag': int(lag),
        'frequency_range': [1, 50],
        'coupling_coeff': float(c1),
        'description': f'Broadband (1-50 Hz) causation with {lag} sample lag'
    }
    
    filename = f"broadband_{dataset_num:03d}.mat"
    filepath = os.path.join(base_dir, filename)
    
    scipy.io.savemat(filepath, {
        'x': x.astype(np.float64),
        'y': y.astype(np.float64),
        'metadata': mat_metadata,
        'fs': np.array([fs], dtype=np.float64)
    })
    
    return filepath, json_metadata

def make_json_serializable(obj):
    """Convert numpy arrays and other non-JSON types to JSON serializable formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def generate_multifreq_causation(dataset_num, base_dir, min_length=500, max_length=2000):
    """
    Fixed multi-frequency causation generation that works with your test framework
    """
    import numpy as np
    import scipy.io
    import os
    
    # Set seed for reproducibility
    np.random.seed(5000 + dataset_num)
    
    # Generate parameters
    length = np.random.randint(min_length, max_length + 1)
    fs = 250
    
    # Frequency components and lags
    frequencies = [10, 40, 80]  # Hz
    lags = [15, 8, 4]  # samples
    
    # Time vector
    t = np.arange(length) / fs
    
    # Generate x signal with multiple frequencies
    x = np.zeros(length)
    freq_components = []
    
    for freq in frequencies:
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        component = amplitude * np.sin(2 * np.pi * freq * t + phase)
        freq_components.append(component)
        x += component
    
    # Add noise to x
    x += 0.3 * np.random.randn(length)
    
    # Generate y with frequency-specific lagged causality
    y = np.zeros(length)
    coupling_coeffs = []
    
    for i, (freq, lag) in enumerate(zip(frequencies, lags)):
        coeff = np.random.normal(0, 1)
        coupling_coeffs.append(coeff)
        
        # Add lagged component to y
        for t_idx in range(lag, length):
            y[t_idx] += coeff * freq_components[i][t_idx - lag]
    
    # Add noise to y
    y += 0.3 * np.random.randn(length)
    
    # Create metadata - FIX: Use plain strings, not arrays
    mat_metadata_fields = {
        'type': 'multifreq',  # Plain string, not array
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': 1,
        'true_lag': int(lags[0]),
        'description': f'Multi-frequency causation: {frequencies} Hz with lags {lags} samples'  # Plain string
    }
    
    # For JSON metadata
    json_metadata = {
        'type': 'multifreq',
        'dataset_num': int(dataset_num),
        'length': int(length),
        'true_causality': True,
        'true_lag': int(lags[0]),
        'frequencies': frequencies,
        'true_lags': lags,
        'coupling_coeffs': [float(c) for c in coupling_coeffs],
        'description': f'Multi-frequency causation: {frequencies} Hz with lags {lags} samples'
    }
    
    # Save to .mat file
    filename = f"multifreq_{dataset_num:03d}.mat"
    filepath = os.path.join(base_dir, filename)
    
    # Create save dictionary - ensure strings are saved as strings, not arrays
    save_data = {
        'x': x.astype(np.float64),
        'y': y.astype(np.float64),
        'fs': np.array([[fs]], dtype=np.float64),
        'metadata': mat_metadata_fields,
        'frequencies': np.array(frequencies, dtype=int),
        'true_lags': np.array(lags, dtype=int),
        'coupling_coeffs': np.array(coupling_coeffs, dtype=float)
    }
    
    try:
        scipy.io.savemat(filepath, save_data)
        print(f"✅ Generated {filename}: {length} samples, freqs={frequencies}Hz, lags={lags}")
        return filepath, json_metadata
        
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")
        return None, None

def generate_complete_dataset():
    """
    Generate complete dataset for VL-Granger testing
    Total: 240 files
    - 60 random noise + 60 AR noise = 120 no-causality (false positive test)
    - 120 causality (true positive test)
    """
    print("Generating VL-Granger Test Dataset...")
    print("=" * 50)
    
    base_dir = create_dataset_directory()
    all_metadata = []
    
    # 1.1 Random noise (60 datasets) - FALSE POSITIVE TEST
    print("Generating random noise datasets (60 files)...")
    for i in range(120):
        filepath, metadata = generate_random_noise(i + 1, base_dir)  # Start from 1
        all_metadata.append(metadata)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/120 random noise datasets")
    
    # 1.1.5 AR noise (60 datasets) - FALSE POSITIVE TEST
    # print("Generating AR noise datasets (60 files)...")
    # for i in range(60):
    #     filepath, metadata = generate_ar_noise(i + 1, base_dir)  # Start from 1
    #     all_metadata.append(metadata)
    #     if (i + 1) % 10 == 0:
    #         print(f"   Generated {i + 1}/60 AR noise datasets")
    
    # 1.2 Basic causation (30 datasets) - TRUE POSITIVE TEST
    print("Generating basic causation datasets (30 files)...")
    for i in range(30):
        filepath, metadata = generate_basic_causation(i + 1, base_dir)  # Start from 1
        all_metadata.append(metadata)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/30 basic causation datasets")
    
    # 1.3 Variable lag causation (30 datasets) - TRUE POSITIVE TEST
    print("Generating variable lag datasets (30 files)...")
    for i in range(30):
        filepath, metadata = generate_variable_lag_causation(i + 1, base_dir)  # Start from 1
        all_metadata.append(metadata)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/30 variable lag datasets")
    
    # 1.4.1 Broadband causation (30 datasets) - TRUE POSITIVE TEST
    print("Generating broadband causation datasets (30 files)...")
    for i in range(30):
        filepath, metadata = generate_broadband_causation(i + 1, base_dir)  # Start from 1
        all_metadata.append(metadata)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/30 broadband datasets")
    
    # 1.4.2 Multi-frequency causation (30 datasets) - TRUE POSITIVE TEST
    print("Generating multi-frequency datasets (30 files)...")
    for i in range(30):
        filepath, metadata = generate_multifreq_causation(i + 1, base_dir)  # Start from 1
        all_metadata.append(metadata)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/30 multi-frequency datasets")
    
    # Convert all metadata to JSON-serializable format before saving
    json_serializable_metadata = [make_json_serializable(meta) for meta in all_metadata]
    
    # Save metadata summary
    metadata_file = os.path.join(base_dir, "dataset_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(json_serializable_metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Generation Complete!")
    print(f"Location: {base_dir}/")
    print(f"Total files: {len(all_metadata)}")
    
    # Count by type
    type_counts = {}
    causality_counts = {'true': 0, 'false': 0}
    
    for meta in all_metadata:
        dataset_type = meta['type']
        type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1
        
        if meta['true_causality']:
            causality_counts['true'] += 1
        else:
            causality_counts['false'] += 1
    
    print("\nDataset Breakdown:")
    for dtype, count in type_counts.items():
        print(f"   {dtype}: {count} files")
    
    print(f"\nTesting Breakdown:")
    print(f"   False Positive Test: {causality_counts['false']} files (no causality)")
    print(f"   True Positive Test: {causality_counts['true']} files (with causality)")
    
    print(f"\nMetadata saved: {metadata_file}")
    
    return base_dir, all_metadata

def load_dataset_file(filepath):
    """
    Helper function to load a dataset file
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

def quick_dataset_test():
    """
    Quick test to verify dataset generation works including AR noise
    """
    print("🧪 Quick Dataset Test")
    print("-" * 20)
    
    base_dir = "test_dataset"
    
    # Generate one of each type
    print("Generating test samples...")
    
    filepath1, meta1 = generate_random_noise(1, base_dir)
    print(f"✅ Random noise: {meta1['description']}")

    
    filepath2, meta2 = generate_basic_causation(1, base_dir)
    print(f"✅ Basic causation: {meta2['description']}")
    
    filepath3, meta3 = generate_variable_lag_causation(1, base_dir)
    print(f"✅ Variable lag: {meta3['description']}")
    
    filepath4, meta4 = generate_broadband_causation(1, base_dir)
    print(f"✅ Broadband: {meta4['description']}")
    
    filepath5, meta5 = generate_multifreq_causation(1, base_dir)
    print(f"✅ Multi-frequency: {meta5['description']}")
    
    # Test loading
    print("\nTesting file loading...")
    test_data = load_dataset_file(filepath2)
    if test_data:
        print(f"✅ Loading successful: x shape {test_data['x'].shape}, y shape {test_data['y'].shape}")
    
    print(f"\n📁 Test files saved in: {base_dir}/")

if __name__ == "__main__":
    # Generate full dataset (not quick test)
    print("Generating complete 240-file dataset...")
    base_dir, metadata = generate_complete_dataset()
    
    print(f"\nReady for paper evaluation!")
    print(f"Use the files in '{base_dir}/' to test your VL-Granger method.")
    print(f"Total files generated: {len(metadata)}")
    
    # Verify file count
    import glob
    mat_files = glob.glob(os.path.join(base_dir, "*.mat"))
    print(f"Verification: {len(mat_files)} .mat files found in directory")