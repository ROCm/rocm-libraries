#!/usr/bin/env python3

"""
Analyze ML Heuristic Sweep Results

Filters and analyzes GEMM sweep results by shape characteristics,
comparing achieved TFLOPS against device peak performance.

Usage:
    python analyze_sweep_results.py sweep_results_1024_full.csv
    python analyze_sweep_results.py sweep_results_1024_full.csv --output analysis.txt
"""

import argparse
import sys
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

# Device Peak TFLOPS (MI300X / gfx950)
DEVICE_PEAK_TFLOPS = {
    'fp16': 1300,  # FP16 peak with matrix cores
    'bf16': 1300,  # BF16 peak (same as FP16)
    'fp8': 2600,   # FP8 peak (2x FP16)
}


@dataclass
class ShapeCharacteristics:
    """Classify shape characteristics"""
    is_square: bool
    is_power_of_2: bool
    has_odd_dim: bool
    is_small: bool  # Any dimension < 128
    is_large: bool  # Any dimension > 2048
    is_very_small: bool  # Any dimension < 32
    aspect_ratio: str  # tall_m, tall_n, tall_k, balanced
    size_class: str  # tiny, small, medium, large, huge


def classify_shape(M: int, N: int, K: int) -> ShapeCharacteristics:
    """Classify shape characteristics"""
    is_square = (M == N == K)

    # Check power of 2
    def is_pow2(n):
        return n > 0 and (n & (n - 1)) == 0

    is_power_of_2 = is_pow2(M) and is_pow2(N) and is_pow2(K)
    has_odd_dim = (M % 2 == 1) or (N % 2 == 1) or (K % 2 == 1)

    # Size classifications
    is_very_small = min(M, N, K) < 32
    is_small = min(M, N, K) < 128
    is_large = max(M, N, K) > 2048

    # Aspect ratio
    max_dim = max(M, N, K)
    if K == max_dim and K > max(M, N) * 2:
        aspect_ratio = 'tall_k'
    elif M == max_dim and M > max(N, K) * 2:
        aspect_ratio = 'tall_m'
    elif N == max_dim and N > max(M, K) * 2:
        aspect_ratio = 'tall_n'
    else:
        aspect_ratio = 'balanced'

    # Size class
    volume = M * N * K
    if volume < 1e6:
        size_class = 'tiny'
    elif volume < 1e7:
        size_class = 'small'
    elif volume < 1e8:
        size_class = 'medium'
    elif volume < 1e9:
        size_class = 'large'
    else:
        size_class = 'huge'

    return ShapeCharacteristics(
        is_square=is_square,
        is_power_of_2=is_power_of_2,
        has_odd_dim=has_odd_dim,
        is_small=is_small,
        is_large=is_large,
        is_very_small=is_very_small,
        aspect_ratio=aspect_ratio,
        size_class=size_class
    )


def analyze_results(csv_path: str, output_file=None):
    """Analyze sweep results"""

    # Read results
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                M = int(row['M'])
                N = int(row['N'])
                K = int(row['K'])
                dtype = row['dtype']
                status = row['status']
                tflops = float(row['actual_tflops']) if row['actual_tflops'] else 0
                pred_tflops = float(row['predicted_tflops'])

                chars = classify_shape(M, N, K)

                results.append({
                    'M': M, 'N': N, 'K': K,
                    'dtype': dtype,
                    'status': status,
                    'tflops': tflops,
                    'predicted_tflops': pred_tflops,
                    'kernel': row['selected_kernel'],
                    'chars': chars,
                })
            except (ValueError, KeyError) as e:
                continue

    # Setup output
    output = []
    def print_line(line=""):
        output.append(line)
        print(line)

    # Header
    print_line("=" * 100)
    print_line("  ML Heuristic Sweep - Results Analysis")
    print_line("=" * 100)
    print_line()

    # Overall statistics
    total = len(results)
    by_status = defaultdict(int)
    for r in results:
        by_status[r['status']] += 1

    print_line(f"Total Tests:     {total}")
    print_line(f"  ✓ SUCCESS:       {by_status['SUCCESS']} ({100*by_status['SUCCESS']/total:.1f}%)")
    print_line(f"  ○ UNSUPPORTED:   {by_status['UNSUPPORTED']} ({100*by_status['UNSUPPORTED']/total:.1f}%)")
    print_line(f"  ✗ RUN_FAIL:      {by_status['RUN_FAIL']} ({100*by_status['RUN_FAIL']/total:.1f}%)")
    print_line(f"  ✗ BUILD_FAIL:    {by_status['BUILD_FAIL']} ({100*by_status['BUILD_FAIL']/total:.1f}%)")
    print_line()

    # Filter successful results
    successful = [r for r in results if r['status'] == 'SUCCESS' and r['tflops'] > 0]

    if not successful:
        print_line("⚠️  No successful tests with valid TFLOPS!")
        return

    print_line("=" * 100)
    print_line("  Performance vs Device Peak")
    print_line("=" * 100)
    print_line()

    # Per-dtype analysis
    for dtype in ['fp16', 'bf16', 'fp8']:
        dtype_results = [r for r in successful if r['dtype'] == dtype]
        if not dtype_results:
            continue

        peak = DEVICE_PEAK_TFLOPS[dtype]
        tflops_values = [r['tflops'] for r in dtype_results]
        avg_tflops = sum(tflops_values) / len(tflops_values)
        max_tflops = max(tflops_values)

        avg_efficiency = 100 * avg_tflops / peak
        max_efficiency = 100 * max_tflops / peak

        print_line(f"{dtype.upper()}:")
        print_line(f"  Device Peak:     {peak:8.1f} TFLOPS")
        print_line(f"  Average:         {avg_tflops:8.1f} TFLOPS ({avg_efficiency:5.2f}% of peak)")
        print_line(f"  Maximum:         {max_tflops:8.1f} TFLOPS ({max_efficiency:5.2f}% of peak)")
        print_line(f"  Tests:           {len(dtype_results)}")
        print_line()

    print_line("=" * 100)
    print_line("  Results by Shape Characteristics")
    print_line("=" * 100)
    print_line()

    # Analysis by characteristics
    characteristics = [
        ('Power-of-2', lambda r: r['chars'].is_power_of_2),
        ('Square (M=N=K)', lambda r: r['chars'].is_square),
        ('Has Odd Dimension', lambda r: r['chars'].has_odd_dim),
        ('Very Small (<32)', lambda r: r['chars'].is_very_small),
        ('Small (<128)', lambda r: r['chars'].is_small),
        ('Large (>2048)', lambda r: r['chars'].is_large),
    ]

    for char_name, filter_fn in characteristics:
        matching = [r for r in successful if filter_fn(r)]
        if not matching:
            print_line(f"{char_name:20s}: No successful tests")
            continue

        tflops_values = [r['tflops'] for r in matching]
        avg = sum(tflops_values) / len(tflops_values)

        # Average efficiency across dtypes
        efficiencies = [100 * r['tflops'] / DEVICE_PEAK_TFLOPS[r['dtype']] for r in matching]
        avg_eff = sum(efficiencies) / len(efficiencies)

        print_line(f"{char_name:20s}: {len(matching):4d} tests | "
                   f"Avg {avg:7.1f} TFLOPS | {avg_eff:5.2f}% efficiency")

    print_line()

    # By aspect ratio
    print_line("By Aspect Ratio:")
    for aspect in ['balanced', 'tall_m', 'tall_n', 'tall_k']:
        matching = [r for r in successful if r['chars'].aspect_ratio == aspect]
        if not matching:
            continue

        tflops_values = [r['tflops'] for r in matching]
        avg = sum(tflops_values) / len(tflops_values)
        efficiencies = [100 * r['tflops'] / DEVICE_PEAK_TFLOPS[r['dtype']] for r in matching]
        avg_eff = sum(efficiencies) / len(efficiencies)

        print_line(f"  {aspect:12s}: {len(matching):4d} tests | "
                   f"Avg {avg:7.1f} TFLOPS | {avg_eff:5.2f}% efficiency")

    print_line()

    # By size class
    print_line("By Problem Size:")
    for size in ['tiny', 'small', 'medium', 'large', 'huge']:
        matching = [r for r in successful if r['chars'].size_class == size]
        if not matching:
            continue

        tflops_values = [r['tflops'] for r in matching]
        avg = sum(tflops_values) / len(tflops_values)
        efficiencies = [100 * r['tflops'] / DEVICE_PEAK_TFLOPS[r['dtype']] for r in matching]
        avg_eff = sum(efficiencies) / len(efficiencies)

        print_line(f"  {size:12s}: {len(matching):4d} tests | "
                   f"Avg {avg:7.1f} TFLOPS | {avg_eff:5.2f}% efficiency")

    print_line()
    print_line("=" * 100)
    print_line("  Top 20 Performers")
    print_line("=" * 100)
    print_line()

    # Sort by TFLOPS
    top_20 = sorted(successful, key=lambda r: r['tflops'], reverse=True)[:20]

    print_line(f"{'Rank':<6} {'Dtype':<6} {'Shape':<20} {'TFLOPS':<10} {'Efficiency':<12} {'Kernel':<25}")
    print_line("-" * 100)

    for i, r in enumerate(top_20, 1):
        shape = f"{r['M']}×{r['N']}×{r['K']}"
        efficiency = 100 * r['tflops'] / DEVICE_PEAK_TFLOPS[r['dtype']]
        kernel = r['kernel'][:24]

        print_line(f"{i:<6} {r['dtype']:<6} {shape:<20} {r['tflops']:<10.2f} {efficiency:<12.2f}% {kernel:<25}")

    print_line()
    print_line("=" * 100)
    print_line("  ML Model Accuracy (Predicted vs Actual)")
    print_line("=" * 100)
    print_line()

    # Filter valid predictions
    valid_pred = [r for r in successful if r['predicted_tflops'] > 0]

    if valid_pred:
        errors = []
        for r in valid_pred:
            actual = r['tflops']
            pred = r['predicted_tflops']
            error_pct = 100 * abs(actual - pred) / actual if actual > 0 else 0
            errors.append(error_pct)

        avg_error = sum(errors) / len(errors)
        print_line(f"Valid Predictions: {len(valid_pred)}")
        print_line(f"Average Error:     {avg_error:.2f}%")
    else:
        print_line("⚠️  No valid predictions found (all predictions ≤ 0)")
        print_line("    This suggests the ML model may need retraining or log transform adjustment")

    print_line()
    print_line("=" * 100)

    # Save output if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))
        print_line()
        print_line(f"Analysis saved to: {output_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description='Analyze ML heuristic sweep results')
    parser.add_argument('csv_file', help='Path to sweep results CSV file')
    parser.add_argument('--output', '-o', help='Output file for analysis report')

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return 1

    analyze_results(args.csv_file, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
