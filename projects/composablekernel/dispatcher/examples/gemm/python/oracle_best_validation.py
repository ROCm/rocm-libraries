#!/usr/bin/env python3
"""
Oracle-Best Validation: Test ML model on training shapes with known oracle-best TFLOPS
"""

import sys
import time
import pandas as pd
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/workspace/rocm-libraries/projects/composablekernel/dispatcher/python')
sys.path.insert(0, '/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics')

from ctypes_utils import DispatcherLib, Dispatcher, KernelConfig
from predict import Predictor

# Load training data with oracle-best (use corrected version with fixed pad flags)
print("Loading training data with oracle-best TFLOPS...")
training_data = pd.read_parquet('/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/data/all_training_data_fixed.parquet')

# Get unique shapes with oracle-best
shape_groups = training_data.groupby(['m', 'n', 'k', 'dtype'])
oracle_best_data = []

for (m, n, k, dtype), group in shape_groups:
    oracle_best_tflops = group['measured_tflops'].max()
    oracle_best_data.append({
        'm': m, 'n': n, 'k': k, 'dtype': dtype,
        'oracle_best': oracle_best_tflops,
        'num_kernels_tested': len(group)
    })

oracle_df = pd.DataFrame(oracle_best_data)
print(f"✓ Loaded {len(oracle_df)} unique shapes with oracle-best data")
print()

# Filter for fp8 only (matching training data)
oracle_df = oracle_df[oracle_df['dtype'] == 'fp8']
print(f"Filtered to fp8: {len(oracle_df)} shapes")
print()

# Load ML predictor (P1-fixed model with corrected padding flags)
model_dir = '/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/models/gemm_universal_fp8_gfx950_p1_fixed'
predictor = Predictor(model_dir)
print(f"✓ Loaded ML predictor from {model_dir}")
print(f"  Log targets: {predictor._log_targets}")
print()

# Load dispatcher library
arch = 'gfx950'
try:
    lib = DispatcherLib.load()
    dispatcher = Dispatcher(lib, arch)
    print(f"✓ Loaded dispatcher library for {arch}")
except Exception as e:
    print(f"✗ Failed to load dispatcher: {e}")
    sys.exit(1)

print()
print("=" * 100)
print(f"  Testing {len(oracle_df)} Shapes with Oracle-Best Validation")
print("=" * 100)
print()

results = []
test_count = 0
skip_count = 0

for idx, row in oracle_df.iterrows():
    m, n, k = int(row['m']), int(row['n']), int(row['k'])
    dtype = row['dtype']
    oracle_best = row['oracle_best']
    
    test_count += 1
    
    # Progress indicator
    if test_count % 20 == 0:
        print(f"  [{test_count}/{len(oracle_df)}] Testing {dtype} {m}×{n}×{k}...")
    
    # Create problem
    problem = {
        'm': m, 'n': n, 'k': k,
        'dtype': dtype,
        'layout': 'rcr',
        'split_k': 1
    }
    
    # Use ML to select kernel (similar to sweep script)
    try:
        # Get all available kernels
        all_kernels = dispatcher.get_all_kernels(dtype, 'rcr')
        
        if not all_kernels:
            skip_count += 1
            continue
        
        # Rank kernels using ML model
        kernel_dicts = [k.__dict__ for k in all_kernels]
        ranked = predictor.rank_kernels(problem, kernel_dicts)
        
        if not ranked:
            skip_count += 1
            continue
        
        best_kernel_name, predicted_tflops = ranked[0]
        
        # Find the kernel config
        selected_kernel = next((k for k in all_kernels if k.name == best_kernel_name), None)
        
        if not selected_kernel:
            skip_count += 1
            continue
        
        # Run on GPU
        try:
            actual_time_ms = dispatcher.run_kernel(selected_kernel, problem)
            
            # Calculate actual TFLOPS
            flops = 2.0 * m * n * k
            actual_tflops = (flops / 1e12) / (actual_time_ms / 1000.0) if actual_time_ms > 0 else 0
            
            # Calculate efficiency vs oracle-best
            efficiency_pct = 100.0 * (actual_tflops / oracle_best) if oracle_best > 0 else 0
            
            results.append({
                'dtype': dtype,
                'm': m, 'n': n, 'k': k,
                'oracle_best': oracle_best,
                'predicted_tflops': predicted_tflops,
                'actual_tflops': actual_tflops,
                'efficiency_pct': efficiency_pct,
                'kernel': best_kernel_name,
                'status': 'SUCCESS'
            })
            
        except Exception as e:
            results.append({
                'dtype': dtype,
                'm': m, 'n': n, 'k': k,
                'oracle_best': oracle_best,
                'predicted_tflops': predicted_tflops,
                'actual_tflops': 0,
                'efficiency_pct': 0,
                'kernel': best_kernel_name,
                'status': f'RUN_FAIL: {str(e)}'
            })
            
    except Exception as e:
        skip_count += 1
        continue

print()
print(f"✓ Completed {test_count} tests")
print(f"  Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
print(f"  Failed: {sum(1 for r in results if 'FAIL' in r['status'])}")
print(f"  Skipped: {skip_count}")
print()

# Save results
output_file = 'oracle_best_validation_results.csv'
with open(output_file, 'w', newline='') as f:
    if results:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

print(f"✓ Results saved to {output_file}")
print()

# Calculate summary statistics
successful = [r for r in results if r['status'] == 'SUCCESS']

if successful:
    print("=" * 100)
    print("  Summary Statistics")
    print("=" * 100)
    print()
    
    efficiencies = [r['efficiency_pct'] for r in successful]
    
    import numpy as np
    print(f"Total successful tests: {len(successful)}")
    print(f"Mean efficiency (% of oracle-best): {np.mean(efficiencies):.2f}%")
    print(f"Median efficiency: {np.median(efficiencies):.2f}%")
    print(f"Min efficiency: {np.min(efficiencies):.2f}%")
    print(f"Max efficiency: {np.max(efficiencies):.2f}%")
    print(f"P10 efficiency: {np.percentile(efficiencies, 10):.2f}%")
    print(f"P90 efficiency: {np.percentile(efficiencies, 90):.2f}%")
    print()
    
    # Published target
    print("Published target: 98.28% mean efficiency")
    print(f"Our result: {np.mean(efficiencies):.2f}%")
    print()

print("=" * 100)

