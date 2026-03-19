#!/usr/bin/env python3
"""
Simple Oracle-Best Validation: Just use ML predictions from training data
Since we have oracle-best already, we don't need to re-run on GPU!
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics')

from predict import Predictor

print("=" * 100)
print("  P1-FIXED MODEL: Oracle-Best Validation (Prediction-Only)")
print("=" * 100)
print()

# Load training data (corrected version with fixed pad flags)
print("Loading training data...")
training_data = pd.read_parquet('/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/data/all_training_data_fixed.parquet')

# Filter for fp8 only (as published)
training_data = training_data[training_data['dtype'] == 'fp8']
training_data = training_data[training_data['layout'] == 'rcr']

print(f"✓ Loaded {len(training_data):,} benchmark runs (fp8, rcr)")
print()

# Get unique shapes with oracle-best
shape_groups = training_data.groupby(['m', 'n', 'k'])
print(f"Unique shapes: {len(shape_groups)}")
print()

# Load ML predictor (P1-fixed model)
model_dir = '/workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics/models/gemm_universal_fp8_gfx950_p1_fixed'
predictor = Predictor(model_dir)
print(f"✓ Loaded ML predictor")
print(f"  Log targets: {predictor._log_targets}")
print()

print("=" * 100)
print("  Computing Oracle-Best Efficiency for Each Shape")
print("=" * 100)
print()

results = []

for shape_idx, ((m, n, k), group) in enumerate(shape_groups):
    # Find oracle-best (max TFLOPS across all kernels tested)
    oracle_best_row = group.loc[group['measured_tflops'].idxmax()]
    oracle_best_tflops = oracle_best_row['measured_tflops']
    oracle_best_kernel = oracle_best_row['kernel_name']
    
    # Get all kernels tested for this shape
    kernel_configs = []
    for _, row in group.iterrows():
        kernel_dict = {
            'tile_m': row['tile_m'],
            'tile_n': row['tile_n'],
            'tile_k': row['tile_k'],
            'warp_m': row['warp_m'],
            'warp_n': row['warp_n'],
            'warp_k': row['warp_k'],
            'warp_tile_m': row['warp_tile_m'],
            'warp_tile_n': row['warp_tile_n'],
            'warp_tile_k': row['warp_tile_k'],
            'pipeline': row['pipeline'],
            'scheduler': row['scheduler'],
            'epilogue': row['epilogue'],
            'pad_m': row['pad_m'],
            'pad_n': row['pad_n'],
            'pad_k': row['pad_k'],
            'persistent': row['persistent'],
            'kernel_name': row['kernel_name']
        }
        kernel_configs.append(kernel_dict)
    
    # Use ML model to rank kernels
    problem = {'m': m, 'n': n, 'k': k, 'dtype': 'fp8', 'layout': 'rcr', 'split_k': 1}
    
    try:
        ranked = predictor.rank_kernels(problem, kernel_configs)
        
        if ranked:
            ml_best_kernel, ml_predicted_tflops = ranked[0]
            
            # Find actual TFLOPS for the ML-predicted kernel
            ml_kernel_row = group[group['kernel_name'] == ml_best_kernel]
            if len(ml_kernel_row) > 0:
                ml_actual_tflops = ml_kernel_row['measured_tflops'].values[0]
                
                # Calculate efficiency
                efficiency_pct = 100.0 * (ml_actual_tflops / oracle_best_tflops)
                
                results.append({
                    'm': m, 'n': n, 'k': k,
                    'oracle_best_tflops': oracle_best_tflops,
                    'oracle_best_kernel': oracle_best_kernel,
                    'ml_predicted_tflops': ml_predicted_tflops,
                    'ml_selected_kernel': ml_best_kernel,
                    'ml_actual_tflops': ml_actual_tflops,
                    'efficiency_pct': efficiency_pct,
                    'num_kernels': len(group)
                })
                
                if (shape_idx + 1) % 20 == 0:
                    print(f"  [{shape_idx + 1}/{len(shape_groups)}] {m}×{n}×{k}: {efficiency_pct:.2f}% efficiency")
    except Exception as e:
        print(f"  Error on shape {m}×{n}×{k}: {e}")
        continue

print()
print("=" * 100)
print("  Results Summary")
print("=" * 100)
print()

if results:
    df_results = pd.DataFrame(results)
    efficiencies = df_results['efficiency_pct'].values
    
    print(f"Total shapes tested: {len(results)}")
    print()
    print("Efficiency Statistics (% of Oracle-Best):")
    print(f"  Mean:   {np.mean(efficiencies):.2f}%")
    print(f"  Median: {np.median(efficiencies):.2f}%")
    print(f"  Min:    {np.min(efficiencies):.2f}%")
    print(f"  Max:    {np.max(efficiencies):.2f}%")
    print(f"  P10:    {np.percentile(efficiencies, 10):.2f}%")
    print(f"  P25:    {np.percentile(efficiencies, 25):.2f}%")
    print(f"  P75:    {np.percentile(efficiencies, 75):.2f}%")
    print(f"  P90:    {np.percentile(efficiencies, 90):.2f}%")
    print()
    
    print("=" * 100)
    print("  Comparison with Published Results")
    print("=" * 100)
    print()
    print(f"Published (README): 98.28% mean efficiency")
    print(f"Our result:         {np.mean(efficiencies):.2f}% mean efficiency")
    print()
    
    if np.mean(efficiencies) >= 95:
        print("✅ PASS - Matches published performance!")
    elif np.mean(efficiencies) >= 90:
        print("⚠️  CLOSE - Slightly below published performance")
    else:
        print("❌ FAIL - Significantly below published performance")
    
    print()
    
    # Save results
    df_results.to_csv('oracle_best_results.csv', index=False)
    print(f"✓ Results saved to oracle_best_results.csv")
    
    # Show top 10 and bottom 10
    print()
    print("Top 10 shapes (best efficiency):")
    print(df_results.nlargest(10, 'efficiency_pct')[['m', 'n', 'k', 'efficiency_pct', 'oracle_best_tflops']])
    print()
    print("Bottom 10 shapes (worst efficiency):")
    print(df_results.nsmallest(10, 'efficiency_pct')[['m', 'n', 'k', 'efficiency_pct', 'oracle_best_tflops']])

else:
    print("No results to display")

print()
print("=" * 100)

