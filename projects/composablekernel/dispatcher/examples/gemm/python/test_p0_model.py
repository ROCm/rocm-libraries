#!/usr/bin/env python3
"""Test P0 model on UNSUPPORTED cases to validate tile selection improvements."""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../../heuristics')

from predict import Predictor
from feature_engine import GemmUniversalFeatureEngine

def parse_tile_from_kernel_name(kernel_name):
    """Extract tile dimensions from kernel name."""
    if kernel_name.startswith('s_'):
        tile_m, tile_n = 64, 64
    elif kernel_name.startswith('m_'):
        tile_m, tile_n = 128, 128
    elif kernel_name.startswith('l_'):
        tile_m, tile_n = 256, 256
    elif kernel_name.startswith('r_'):
        # Parse from name like r_64x128_k64_v3
        parts = kernel_name.split('_')[1].split('x')
        tile_m = int(parts[0])
        tile_n = int(parts[1]) if len(parts) > 1 else 64
    else:
        tile_m, tile_n = 128, 128

    # Extract K tile
    if 'k16' in kernel_name:
        tile_k = 16
    elif 'k32' in kernel_name:
        tile_k = 32
    elif 'k64' in kernel_name:
        tile_k = 64
    elif 'k128' in kernel_name:
        tile_k = 128
    else:
        tile_k = 64

    return tile_m, tile_n, tile_k

def create_kernel_config(name):
    """Create kernel config dict from kernel name."""
    tm, tn, tk = parse_tile_from_kernel_name(name)

    # Determine pipeline
    if 'mem' in name or 'preshuffle' in name:
        pipeline = 'mem'
        pad_m, pad_n, pad_k = True, True, True
    elif 'v4' in name:
        pipeline = 'compv4'
        pad_m, pad_n, pad_k = False, False, False
    elif 'v3' in name:
        pipeline = 'compv3'
        pad_m, pad_n, pad_k = False, False, False
    else:
        pipeline = 'compv3'
        pad_m, pad_n, pad_k = False, False, False

    scheduler = 'interwave' if 'iw' in name else 'intrawave'

    return {
        'name': name,
        'kernel_name': name,  # predictor.rank_kernels() looks for 'kernel_name'
        'tile_m': tm,
        'tile_n': tn,
        'tile_k': tk,
        'pad_m': pad_m,
        'pad_n': pad_n,
        'pad_k': pad_k,
        'pipeline': pipeline,
        'scheduler': scheduler,
        'epilogue': 'default',
        'warp_m': 2,
        'warp_n': 2,
        'warp_k': 1,
        'warp_tile_m': 16,
        'warp_tile_n': 16,
        'warp_tile_k': 16,
        'persistent': False,
    }

def check_tile_compatibility(M, N, K, tile_m, tile_n, tile_k, pad_m, pad_n, pad_k):
    """Check if tile size is compatible with problem dimensions."""
    # For non-padded kernels, dimensions must be divisible by tile
    if not pad_m and M % tile_m != 0:
        return False, "M not divisible by tile_m (no padding)"
    if not pad_n and N % tile_n != 0:
        return False, "N not divisible by tile_n (no padding)"
    if not pad_k and K % tile_k != 0:
        return False, "K not divisible by tile_k (no padding)"

    # For padded kernels, we still want to avoid oversized tiles
    # Based on hardware tests: very small problems (N < tile_n) often fail even with padding
    if N < tile_n and N < 64:
        return False, f"N={N} too small for tile_n={tile_n}"

    return True, "Compatible"

def main():
    # Load test shapes
    test_csv = 'unsupported_test_shapes.csv'
    df = pd.read_csv(test_csv)

    print("="*90)
    print(f"P1 MODEL VALIDATION TEST - {len(df)} UNSUPPORTED SHAPES")
    print("="*90)
    print()

    # Load P1-fixed predictor (with corrected padding flags in training data)
    predictor = Predictor(model_dir='../../../heuristics/models/gemm_universal_fp8_gfx950_p1_fixed')

    # Define available kernel pool (same as original sweep)
    kernel_pool = [
        's_64x64_k64_mem',
        's_64x64_k128_mem',
        's_64x64_k32_v4',
        's_64x64_k64_v4',
        's_64x64_k128_v3',
        'm_128x128_k32_v4',
        'm_128x128_k64_mem',
        'm_128x128_k128_mem',
        'm_128x128_k128_iw_v3',
        'l_256x256_k64_v3',
    ]

    results = []
    improved = 0
    still_bad = 0

    print("Testing P0 model predictions...")
    print()

    for idx, row in df.iterrows():
        M, N, K = int(row['M']), int(row['N']), int(row['K'])
        old_kernel = row['selected_kernel']

        # Create problem dict
        problem = {
            'M': M, 'N': N, 'K': K,
            'm': M, 'n': N, 'k': K,
            'dtype': 'fp16',
            'layout': 'rcr',
            'split_k': 1,
        }

        # Create kernel configs
        kernel_configs = [create_kernel_config(k) for k in kernel_pool]

        # Filter out incompatible kernels (missing required padding)
        compatible_configs = []
        for kc in kernel_configs:
            tm, tn, tk = kc['tile_m'], kc['tile_n'], kc['tile_k']
            pad_m, pad_n, pad_k = kc['pad_m'], kc['pad_n'], kc['pad_k']

            # Check if kernel is compatible with problem dimensions
            needs_pad_m = (M % tm != 0)
            needs_pad_n = (N % tn != 0)
            needs_pad_k = (K % tk != 0)

            has_required_padding = (not needs_pad_m or pad_m) and \
                                   (not needs_pad_n or pad_n) and \
                                   (not needs_pad_k or pad_k)

            if has_required_padding:
                compatible_configs.append(kc)

        if not compatible_configs:
            print(f"  Warning: No compatible kernels for M={M} N={N} K={K}")
            compatible_configs = kernel_configs  # Fallback to all

        # Use P1 model to rank compatible kernels only
        try:
            ranked = predictor.rank_kernels(problem, compatible_configs)
            new_best_kernel = ranked[0][0]
            new_best_tflops = ranked[0][1]

            # Get tile sizes
            old_tm, old_tn, old_tk = parse_tile_from_kernel_name(old_kernel)
            new_tm, new_tn, new_tk = parse_tile_from_kernel_name(new_best_kernel)

            # Get padding info
            old_cfg = create_kernel_config(old_kernel)
            new_cfg = create_kernel_config(new_best_kernel)

            # Check compatibility
            old_compat, old_reason = check_tile_compatibility(
                M, N, K, old_tm, old_tn, old_tk,
                old_cfg['pad_m'], old_cfg['pad_n'], old_cfg['pad_k']
            )
            new_compat, new_reason = check_tile_compatibility(
                M, N, K, new_tm, new_tn, new_tk,
                new_cfg['pad_m'], new_cfg['pad_n'], new_cfg['pad_k']
            )

            # Count improvements
            if new_compat and not old_compat:
                improved += 1
                status = "✓ IMPROVED"
            elif new_compat and old_compat:
                status = "= BOTH OK"
            elif not new_compat and not old_compat:
                still_bad += 1
                status = "✗ STILL BAD"
            else:
                status = "? WORSE"

            results.append({
                'M': M, 'N': N, 'K': K,
                'old_kernel': old_kernel,
                'old_tile_n': old_tn,
                'old_compat': old_compat,
                'new_kernel': new_best_kernel,
                'new_tile_n': new_tn,
                'new_compat': new_compat,
                'new_tflops': new_best_tflops,
                'status': status,
            })

        except Exception as e:
            print(f"Error processing {M}x{N}x{K}: {e}")
            results.append({
                'M': M, 'N': N, 'K': K,
                'old_kernel': old_kernel,
                'status': f"ERROR: {e}",
            })

    # Summary
    print("="*90)
    print("RESULTS SUMMARY")
    print("="*90)
    print()
    print(f"Total test cases: {len(df)}")
    print(f"  ✓ IMPROVED (now compatible): {improved} ({100*improved/len(df):.1f}%)")
    print(f"  ✗ STILL BAD (still incompatible): {still_bad} ({100*still_bad/len(df):.1f}%)")
    print(f"  = Already OK or worse: {len(df) - improved - still_bad}")
    print()

    # Show sample improvements
    improved_cases = [r for r in results if '✓ IMPROVED' in r.get('status', '')]
    if improved_cases:
        print("Sample improvements (first 15):")
        print(f"{'M':>6} {'N':>6} {'K':>6} {'Old Kernel':<25} {'→ New Kernel':<25} {'Old TN':<7} {'New TN'}")
        print("-"*90)
        for r in improved_cases[:15]:
            print(f"{r['M']:6d} {r['N']:6d} {r['K']:6d} {r['old_kernel']:<25} → {r['new_kernel']:<25} {r['old_tile_n']:<7d} {r['new_tile_n']}")

    print()

    # Show remaining problems
    still_bad_cases = [r for r in results if '✗ STILL BAD' in r.get('status', '')]
    if still_bad_cases:
        print(f"\nRemaining problems ({len(still_bad_cases)} cases):")
        print(f"{'M':>6} {'N':>6} {'K':>6} {'New Kernel':<25} {'New TN':<7} {'Issue'}")
        print("-"*90)
        for r in still_bad_cases[:10]:
            issue = f"N={r['N']} < tile_n={r['new_tile_n']}" if r['N'] < r['new_tile_n'] else "Other"
            print(f"{r['M']:6d} {r['N']:6d} {r['K']:6d} {r['new_kernel']:<25} {r['new_tile_n']:<7d} {issue}")

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('p0_test_results.csv', index=False)
    print()
    print(f"Detailed results saved to p0_test_results.csv")

    return improved, still_bad, len(df)

if __name__ == '__main__':
    main()
