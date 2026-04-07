#!/usr/bin/env python3
"""
Compare ML predictions against oracle benchmark results for backward passes (BWD_DATA, BWD_WEIGHT).

This script:
1. Loads ML model predictions for 10 validation problems
2. Loads oracle results from validation_{variant}_bf16_gfx950.csv (when available)
3. Computes efficiency metrics: how close is ML's top-1 pick to oracle's best?
4. Optionally runs ML-picked kernels on hardware for actual TFLOPS measurement (--run-ml-hw)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "heuristics"))
sys.path.insert(0, str(_THIS_DIR / "problems"))

from predict import Predictor
from feature_engine_grouped_conv import GroupedConvFeatureEngine

# Backward kernels (20 total)
BACKWARD_KERNELS = [
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 16, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'mem'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'mem'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 32, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'mem'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 64, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
    {'block_size': 128, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'compv3'},
    {'block_size': 128, 'gemm_m_per_block': 64, 'gemm_n_per_block': 128, 'pipeline': 'mem'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'compv3'},
    {'block_size': 128, 'gemm_m_per_block': 128, 'gemm_n_per_block': 64, 'pipeline': 'mem'},
]


def problem_to_dict(p):
    """Convert GroupedConvProblem to dictionary."""
    return {
        'N': p.N, 'C': p.C, 'K': p.K, 'G': p.G,
        'Hi': p.Hi, 'Wi': p.Wi, 'Y': p.Y, 'X': p.X,
        'stride_h': p.stride_h, 'stride_w': p.stride_w,
        'pad_h': p.pad_h, 'pad_w': p.pad_w,
        'dtype': 'bf16'
    }


def format_problem(p):
    """Format problem for display."""
    return (f"N={p.N:3d} C={p.C:4d} K={p.K:4d} "
            f"{p.Hi:2d}x{p.Wi:2d}→{p.Ho:2d}x{p.Wo:2d} "
            f"f{p.Y}x{p.X}")


def kernel_to_name(kernel, variant='bwd_data', short=False):
    """Convert kernel dict to name.

    Args:
        kernel: Kernel configuration dict
        variant: 'bwd_data' or 'bwd_weight'
        short: If True, return short name (e.g., "16x64x64_compv3")
               If False, return oracle name (e.g., "grouped_conv_bwd_data_bf16_2d_16x64x64_compv3")
    """
    tile_name = f"{kernel['block_size']}x{kernel['gemm_m_per_block']}x{kernel['gemm_n_per_block']}"
    pipeline = kernel['pipeline']

    if short:
        return f"{tile_name}_{pipeline}"
    else:
        return f"grouped_conv_{variant}_bf16_2d_{tile_name}_{pipeline}"


def load_oracle_results(csv_path):
    """Load oracle benchmark results from CSV.

    Returns:
        dict: {problem_idx: [(kernel_name, tflops), ...]}
    """
    if not Path(csv_path).exists():
        return None

    results = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_idx = int(row['problem_idx'])
            kernel_name = row['kernel']
            tflops = float(row['tflops']) if row['tflops'] not in ['nan', ''] else 0.0

            results[problem_idx].append((kernel_name, tflops))

    # Sort each problem's results by TFLOPS descending
    for idx in results:
        results[idx].sort(key=lambda x: x[1], reverse=True)

    return dict(results)


def find_kernel_so_path(kernel_name, arch='gfx950'):
    """Find the .so path for a compiled kernel.

    Args:
        kernel_name: e.g., "32x128x64_compv3"
        arch: GPU architecture

    Returns:
        Path to .so file or None if not found
    """
    # Parse kernel name to construct .so filename
    # kernel_name format: "32x128x64_compv3"
    parts = kernel_name.split('_')
    tile = parts[0]  # "32x128x64"
    pipeline = parts[1]  # "compv3" or "mem"

    # The .so files are in _THIS_DIR / ".ck_dispatcher_build" / arch
    build_dir = _THIS_DIR / ".ck_dispatcher_build" / arch

    # Find matching .so - pattern: libdispatcher_conv_bwd_data_2d_bf16_*_{pipeline}_intrawave.so
    import glob
    pattern = str(build_dir / f"libdispatcher_conv_bwd_data_2d_bf16_*_{pipeline}_intrawave.so")
    matches = glob.glob(pattern)

    # Filter by tile size in the name
    for match in matches:
        if tile.replace('x', 'x') in match:
            return Path(match)

    return None


def run_ml_kernel_on_hw(problem_idx, problem, kernel_name):
    """Run a specific ML-picked kernel on hardware.

    Args:
        problem_idx: Problem index
        problem: GroupedConvProblem object
        kernel_name: Kernel name (e.g., "32x128x64_compv3")

    Returns:
        dict: {'status': 'SUCCESS'/'FAIL', 'tflops': float, 'latency_ms': float, 'error': str}
    """
    # Find the compiled .so for this kernel
    so_path = find_kernel_so_path(kernel_name)

    if so_path is None or not so_path.exists():
        return {'status': 'FAIL', 'error': f'Kernel .so not found: {kernel_name}'}

    # Prepare problem dict for subprocess
    prob_dict = {
        'N': problem.N, 'C': problem.C, 'K': problem.K, 'G': problem.G,
        'Hi': problem.Hi, 'Wi': problem.Wi, 'Y': problem.Y, 'X': problem.X,
        'stride_h': problem.stride_h, 'stride_w': problem.stride_w,
        'pad_h': problem.pad_h, 'pad_w': problem.pad_w,
        'direction': 'bwd_data'
    }

    # Run via subprocess
    runner_script = _THIS_DIR / "run_one_grouped_conv_kernel.py"

    input_data = json.dumps({
        "so_path": str(so_path),
        "problem": prob_dict,
        "kernel_name": kernel_name
    })

    try:
        env = os.environ.copy()
        env['GCONV_PYPATH'] = str(_DISPATCHER_ROOT / "python")
        env['ROCR_VISIBLE_DEVICES'] = '0'

        result = subprocess.run(
            ['python3', str(runner_script)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_THIS_DIR,
            env=env
        )

        if result.returncode != 0:
            return {'status': 'FAIL', 'error': f'Subprocess error: {result.stderr[:200]}'}

        # Parse JSON output
        output = result.stdout.strip()
        data = json.loads(output)

        if data.get('ok'):
            return {
                'status': 'SUCCESS',
                'tflops': data['tflops'],
                'latency_ms': data['ms']
            }
        else:
            return {'status': 'FAIL', 'error': data.get('error', 'Unknown error')}

    except subprocess.TimeoutExpired:
        return {'status': 'FAIL', 'error': 'Timeout (30s)'}
    except json.JSONDecodeError:
        return {'status': 'FAIL', 'error': f'JSON parse error: {result.stdout[:200]}'}
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)[:200]}


def main():
    parser = argparse.ArgumentParser(description="Validate ML heuristic predictions against oracle")
    parser.add_argument('--variant', choices=['bwd_data', 'bwd_weight'], default='bwd_data',
                       help='Variant to validate (bwd_data or bwd_weight)')
    parser.add_argument('--run-ml-hw', action='store_true',
                       help='Run ML-picked kernels on hardware for actual TFLOPS measurement')
    args = parser.parse_args()

    variant = args.variant

    # Load appropriate validation problems
    if variant == 'bwd_data':
        from bwd_data_test_validation import VALIDATION_PROBLEMS_BWD_DATA as validation_problems
    else:  # bwd_weight
        from bwd_weight_test_validation import VALIDATION_PROBLEMS_BWD_WEIGHT as validation_problems

    print()
    print("=" * 120)
    if args.run_ml_hw:
        print(f"  {variant.upper()} ML MODEL VALIDATION - Hardware Execution of ML Picks")
    else:
        print(f"  {variant.upper()} ML MODEL VALIDATION - Prediction vs Oracle Comparison")
    print("=" * 120)
    print()

    # Load ML model
    model_dir = _DISPATCHER_ROOT / "heuristics" / "models" / f"grouped_conv_{variant}_bf16_gfx950"
    if not model_dir.exists():
        print(f"ERROR: Model not found: {model_dir}")
        sys.exit(1)

    feature_engine = GroupedConvFeatureEngine()
    predictor = Predictor(model_dir, feature_engine=feature_engine)
    print(f"  ✓ ML model loaded: {model_dir.name}")
    print(f"  ✓ Validation problems: {len(validation_problems)}")
    print()

    # Try to load oracle results
    oracle_csv = _THIS_DIR / f"validation_{variant}_bf16_gfx950.csv"
    oracle_results = load_oracle_results(oracle_csv)

    if oracle_results is None:
        if args.run_ml_hw:
            print(f"ERROR: Oracle results required for --run-ml-hw mode: {oracle_csv}")
            print("Please run the oracle benchmark first.")
            sys.exit(1)
        print(f"  ⚠  Oracle results not yet available: {oracle_csv}")
        print("  ⚠  Will only show ML predictions (waiting for oracle benchmark to complete)")
        print()
        mode = "prediction_only"
    else:
        print(f"  ✓ Oracle results loaded: {oracle_csv}")
        print(f"  ✓ Oracle benchmarks: {sum(len(v) for v in oracle_results.values())} kernel measurements")
        print()
        mode = "hardware" if args.run_ml_hw else "comparison"

    # Get ML predictions for each problem
    if mode == "hardware":
        print(f"  {'Problem':<40} {'Oracle Best':<20} {'Oracle':>10} {'ML Pick':<20} {'ML Pred':>10} {'ML HW':>10} {'Efficiency':>11}")
        print("  " + "-" * 115)
    elif mode == "comparison":
        print(f"  {'Problem':<45} {'ML Pick':<25} {'ML Pred':>11} {'Oracle Best':<25} {'Oracle':>13} {'Efficiency':>11}")
        print("  " + "-" * (45 + 25 + 11 + 25 + 13 + 11 + 4))
    else:
        print(f"  {'Problem':<45} {'ML Pick':<25} {'ML TFLOPS':>11}")
        print("  " + "-" * (45 + 25 + 11 + 2))

    efficiencies = []

    for idx, problem in enumerate(validation_problems):
        prob_str = format_problem(problem)
        prob_dict = problem_to_dict(problem)

        # Get ML predictions
        predictions = []
        for kernel in BACKWARD_KERNELS:
            pred_tflops = predictor.predict_tflops(prob_dict, kernel)
            predictions.append({
                'kernel': kernel,
                'name': kernel_to_name(kernel, variant=variant, short=False),  # Use oracle format for matching
                'short_name': kernel_to_name(kernel, variant=variant, short=True),  # For display
                'pred_tflops': pred_tflops
            })

        # Sort by predicted TFLOPS
        predictions.sort(key=lambda x: x['pred_tflops'], reverse=True)

        ml_pick = predictions[0]
        ml_name = ml_pick['name']  # Oracle format for matching
        ml_short_name = ml_pick['short_name']  # Short format for display
        ml_pred_tflops = ml_pick['pred_tflops']

        if mode == "hardware":
            # Get oracle best
            oracle_list = oracle_results.get(idx, [])
            oracle_best_name, oracle_best_tflops = oracle_list[0] if oracle_list else ('N/A', 0.0)
            oracle_short = oracle_best_name.replace('grouped_conv_bwd_data_bf16_2d_', '') if oracle_best_name != 'N/A' else 'N/A'

            # Run ML pick on hardware
            hw_result = run_ml_kernel_on_hw(idx, problem, ml_short_name)

            if hw_result['status'] == 'SUCCESS':
                ml_hw_tflops = hw_result['tflops']
                efficiency = 100.0 * ml_hw_tflops / oracle_best_tflops if oracle_best_tflops > 0 else 0.0
                efficiencies.append(efficiency)
                print(f"  {prob_str:<40} {oracle_short:<20} {oracle_best_tflops:>10.2f} {ml_short_name:<20} {ml_pred_tflops:>10.2f} {ml_hw_tflops:>10.2f} {efficiency:>10.1f}%")
            else:
                print(f"  {prob_str:<40} {oracle_short:<20} {oracle_best_tflops:>10.2f} {ml_short_name:<20} {ml_pred_tflops:>10.2f} {'FAIL':>10} {'N/A':>11}")
                print(f"    Error: {hw_result['error']}")

        elif mode == "comparison":
            # Get oracle best and find ML pick in oracle results
            oracle_list = oracle_results.get(idx, [])
            if oracle_list:
                oracle_best_name, oracle_best_tflops = oracle_list[0]
                oracle_short = oracle_best_name.replace(f'grouped_conv_{variant}_bf16_2d_', '')

                # Find ML's pick in oracle results (match by full name)
                ml_oracle_tflops = None
                for name, tflops in oracle_list:
                    if name == ml_name:
                        ml_oracle_tflops = tflops
                        break

                if ml_oracle_tflops is not None and oracle_best_tflops > 0:
                    efficiency = 100.0 * ml_oracle_tflops / oracle_best_tflops
                    efficiencies.append(efficiency)
                    print(f"  {prob_str:<45} {ml_short_name:<25} {ml_pred_tflops:>11.2f} {oracle_short:<25} {oracle_best_tflops:>13.2f} {efficiency:>10.1f}%")
                else:
                    print(f"  {prob_str:<45} {ml_short_name:<25} {ml_pred_tflops:>11.2f} {oracle_short:<25} {oracle_best_tflops:>13.2f} {'N/A':>11}")
            else:
                print(f"  {prob_str:<45} {ml_short_name:<25} {ml_pred_tflops:>11.2f} {'N/A':<25} {'N/A':>13} {'N/A':>11}")
        else:
            print(f"  {prob_str:<45} {ml_short_name:<25} {ml_pred_tflops:>11.2f}")

    print()
    print("  " + "=" * 100)
    print()

    if mode == "comparison" and efficiencies:
        print("  EFFICIENCY METRICS:")
        print(f"  {'Metric':<30} {'Value':>15}")
        print("  " + "-" * 47)
        print(f"  {'Mean Efficiency':<30} {sum(efficiencies)/len(efficiencies):>14.1f}%")
        print(f"  {'P10 Efficiency':<30} {sorted(efficiencies)[len(efficiencies)//10]:>14.1f}%")
        print(f"  {'P50 Efficiency (Median)':<30} {sorted(efficiencies)[len(efficiencies)//2]:>14.1f}%")
        print(f"  {'Min Efficiency':<30} {min(efficiencies):>14.1f}%")
        print(f"  {'Max Efficiency':<30} {max(efficiencies):>14.1f}%")
        print()

        # Top-1 accuracy (how often ML picks the oracle best)
        top1_count = sum(1 for eff in efficiencies if eff >= 99.9)
        print(f"  {'Top-1 Accuracy':<30} {100*top1_count/len(efficiencies):>14.1f}% ({top1_count}/{len(efficiencies)})")
        print()
    elif mode == "prediction_only":
        # Show statistics on predictions
        all_pred_tflops = [p['pred_tflops'] for p in predictions]
        print("  ML PREDICTION STATISTICS (Oracle not yet available):")
        print(f"  {'Metric':<30} {'Value':>15}")
        print("  " + "-" * 47)

        avg_tflops = sum(predictions[0]['pred_tflops'] for _ in validation_problems) / len(validation_problems)
        print(f"  {'Avg Predicted TFLOPS':<30} {avg_tflops:>15.2f}")
        print()
        print("  Run oracle benchmark to enable efficiency comparison:")
        print("  python3 grouped_conv_full_benchmark.py --arch gfx950 \\")
        print(f"    --problems {variant}_test_validation \\")
        print(f"    --csv validation_{variant}_bf16_gfx950.csv \\")
        print(f"    --batch-size 5 configs/{variant}_bf16.json")
        print()

    print("  " + "=" * 100)
    print()


if __name__ == "__main__":
    main()
