#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ML Heuristic Sweep: Comprehensive GEMM Performance Evaluation

Sweeps across ~1024 problem shapes with multiple dtypes (fp16, bf16, fp8)
using ML-based kernel selection heuristics to measure TFLOPS performance.

Usage:
    python ml_heuristic_sweep.py --model model_tflops_log_big --output sweep_results.csv
    python ml_heuristic_sweep.py --dtypes fp16 bf16 --num_shapes 512 --dry_run
    python ml_heuristic_sweep.py --dtypes fp16 bf16 fp8 --output results.csv
"""

import sys
import argparse
import time
import csv
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "heuristics"))

import numpy as np

from ctypes_utils import (
    KernelConfig,
    setup_gemm_dispatcher,
    cleanup_gemm,
)

try:
    from predict import Predictor
    from feature_engine import GemmUniversalFeatureEngine
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("WARNING: ML heuristic modules not available. Will use first-fit selection.")


@dataclass
class KernelSpec:
    """Kernel specification for ML heuristic"""
    name: str
    tile_m: int
    tile_n: int
    tile_k: int
    pipeline: str = "compv3"
    scheduler: str = "intrawave"
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1
    warp_m: int = 32
    warp_n: int = 32
    warp_k: int = 16


# Comprehensive kernel pool covering diverse tile sizes and configurations
KERNEL_POOL = [
    # Small tiles (64x64)
    KernelSpec("s_64x64_k16_v3", 64, 64, 16, "compv3", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k32_v3", 64, 64, 32, "compv3", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k64_v3", 64, 64, 64, "compv3", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k128_v3", 64, 64, 128, "compv3", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k32_v4", 64, 64, 32, "compv4", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k64_v4", 64, 64, 64, "compv4", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k64_mem", 64, 64, 64, "mem", "intrawave", 2, 2, 1, 16, 16, 16),
    KernelSpec("s_64x64_k128_mem", 64, 64, 128, "mem", "intrawave", 2, 2, 1, 16, 16, 16),

    # Medium tiles (128x128)
    KernelSpec("m_128x128_k16_v3", 128, 128, 16, "compv3", "intrawave"),
    KernelSpec("m_128x128_k32_v3", 128, 128, 32, "compv3", "intrawave"),
    KernelSpec("m_128x128_k64_v3", 128, 128, 64, "compv3", "intrawave"),
    KernelSpec("m_128x128_k128_v3", 128, 128, 128, "compv3", "intrawave"),
    KernelSpec("m_128x128_k32_v4", 128, 128, 32, "compv4", "intrawave"),
    KernelSpec("m_128x128_k64_v4", 128, 128, 64, "compv4", "intrawave"),
    KernelSpec("m_128x128_k128_v4", 128, 128, 128, "compv4", "intrawave"),
    KernelSpec("m_128x128_k64_mem", 128, 128, 64, "mem", "intrawave"),
    KernelSpec("m_128x128_k128_mem", 128, 128, 128, "mem", "intrawave"),

    # Rectangular medium (M != N)
    KernelSpec("r_64x128_k32_v3", 64, 128, 32, "compv3", "intrawave", 2, 2, 1, 16, 32, 16),
    KernelSpec("r_128x64_k32_v3", 128, 64, 32, "compv3", "intrawave", 2, 2, 1, 32, 16, 16),
    KernelSpec("r_64x128_k64_v3", 64, 128, 64, "compv3", "intrawave", 2, 2, 1, 16, 32, 16),
    KernelSpec("r_128x64_k64_v3", 128, 64, 64, "compv3", "intrawave", 2, 2, 1, 32, 16, 16),
    KernelSpec("r_64x256_k32_v3", 64, 256, 32, "compv3", "intrawave", 2, 2, 1, 16, 32, 16),
    KernelSpec("r_256x64_k32_v3", 256, 64, 32, "compv3", "intrawave", 2, 2, 1, 32, 16, 16),

    # Large tiles (256x256)
    KernelSpec("l_256x128_k32_v3", 256, 128, 32, "compv3", "intrawave"),
    KernelSpec("l_128x256_k32_v3", 128, 256, 32, "compv3", "intrawave"),
    KernelSpec("l_256x256_k16_v3", 256, 256, 16, "compv3", "intrawave"),
    KernelSpec("l_256x256_k32_v3", 256, 256, 32, "compv3", "intrawave"),
    KernelSpec("l_256x256_k64_v3", 256, 256, 64, "compv3", "intrawave"),
    KernelSpec("l_256x256_k32_v4", 256, 256, 32, "compv4", "intrawave"),
    KernelSpec("l_256x256_k64_v4", 256, 256, 64, "compv4", "intrawave"),

    # Interwave variants
    KernelSpec("m_128x128_k64_iw_v3", 128, 128, 64, "compv3", "interwave"),
    KernelSpec("m_128x128_k128_iw_v3", 128, 128, 128, "compv3", "interwave"),
    KernelSpec("l_256x256_k32_iw_v3", 256, 256, 32, "compv3", "interwave"),
]


def generate_problem_shapes(num_shapes: int = 1024) -> List[Tuple[int, int, int]]:
    """
    Generate ~1024 diverse problem shapes covering:
    - Powers of 2 (square)
    - ML workloads (LLM attention, MLP)
    - Rectangular matrices
    - ODD DIMENSIONS (extensive coverage of non-power-of-2)
    - Edge cases (very small, very large, extreme aspect ratios)
    """
    shapes = []

    # 1. Powers of 2 - Square (64 to 8192) with K variations
    for p in range(6, 14):  # 2^6=64 to 2^13=8192
        dim = 2 ** p
        shapes.append((dim, dim, dim))
        if dim >= 128:
            shapes.append((dim, dim, dim // 2))
            shapes.append((dim, dim, dim * 2))
            shapes.append((dim, dim, dim // 4))

    # 2. ODD DIMENSIONS - Comprehensive coverage
    # Small odd numbers (primes and common odd values)
    small_odds = [3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 31, 33, 37, 41, 43, 47, 51,
                  59, 61, 63, 67, 71, 73, 77, 79, 83, 89, 91, 97, 99, 101, 103, 107,
                  109, 111, 113, 117, 119, 121, 123, 127]

    # Medium odd numbers
    medium_odds = [131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
                   197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
                   271, 277, 281, 283, 293, 299, 307, 311, 313, 317, 331, 337, 347,
                   349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
                   431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]

    # Large odd numbers
    large_odds = [501, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587,
                  593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
                  661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751,
                  757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
                  853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
                  941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
                  1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
                  1093, 1097, 1103, 1109, 1117, 1123, 1129]

    # Very large odd numbers
    xlarge_odds = [1151, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229,
                   1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301,
                   1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409,
                   1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481,
                   1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553,
                   1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619,
                   1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709,
                   1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789,
                   1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879,
                   1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
                   1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063,
                   2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137]

    # Extreme odd numbers (for stress testing)
    extreme_odds = [2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243,
                    2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333,
                    2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393,
                    2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477,
                    2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593,
                    2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683,
                    2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741,
                    2749, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2843,
                    2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939,
                    2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037,
                    3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137,
                    3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229,
                    3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
                    3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407,
                    3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511,
                    3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581,
                    3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671]

    all_odds = small_odds + medium_odds + large_odds + xlarge_odds + extreme_odds

    # Square odd matrices
    for odd in all_odds[:150]:  # Use first 150 odd numbers
        shapes.append((odd, odd, odd))

    # Rectangular odd matrices (odd x odd with different K)
    for i, m in enumerate(all_odds[:80]):
        for j, n in enumerate(all_odds[:80]):
            if i % 5 == j % 5:  # Stratified sampling
                for k in [128, 256, 512, 1024]:
                    shapes.append((m, n, k))

    # Odd M/N with power-of-2 K
    for odd in all_odds[:100]:
        for pow2 in [64, 128, 256, 512, 1024, 2048]:
            shapes.append((odd, odd, pow2))
            shapes.append((odd, pow2, odd))
            shapes.append((pow2, odd, odd))

    # Mixed odd and even
    for odd in [99, 101, 127, 199, 201, 255, 257, 299, 333, 399, 501, 511, 513,
                777, 999, 1023, 1025, 1111, 1333, 1501, 1777, 1999, 2001, 2047,
                2049, 2222, 2345, 2999, 3001, 3333, 3579, 3999]:
        shapes.append((odd, odd, odd))
        shapes.append((odd, odd * 2, odd))
        shapes.append((odd * 2, odd, odd))

    # 3. Small batch inference (1-256 batch, common hidden dims)
    hidden_dims = [768, 1024, 2048, 3072, 4096, 5120, 8192, 11008, 12288, 16384]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for hidden in hidden_dims:
        for batch in batch_sizes[:8]:
            shapes.append((batch, hidden, hidden))
            if hidden >= 4096:
                # LLM MLP projections
                shapes.append((batch, hidden, hidden * 3 // 4))
                shapes.append((batch, hidden * 3 // 4, hidden))

    # 4. Attention patterns (seq_len x head_dim)
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    head_dims = [64, 80, 96, 128, 256]
    num_heads = [8, 12, 16, 32, 40, 64]

    for seq in seq_lens:
        for head_dim in head_dims:
            for nh in num_heads[:4]:
                total_dim = nh * head_dim
                shapes.append((seq, total_dim, head_dim))
                shapes.append((seq, head_dim, total_dim))

    # 5. Rectangular matrices (extreme aspect ratios)
    for m in [64, 128, 256, 512, 1024, 2048]:
        for n in [64, 128, 256, 512, 1024, 2048]:
            if m != n:
                for k in [128, 512, 2048, 8192]:
                    shapes.append((m, n, k))

    # 6. Very tall K (memory-bound)
    for mn in [128, 256, 512, 1024]:
        for k in [4096, 8192, 16384]:
            shapes.append((mn, mn, k))

    # 7. Very short K (compute-bound)
    for mn in [512, 1024, 2048, 4096]:
        for k in [16, 32, 64, 128]:
            shapes.append((mn, mn, k))

    # 8. Very small (edge cases)
    for m in [1, 2, 4, 8, 16, 32]:
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for k in [16, 32, 64, 128, 256]:
                shapes.append((m, n, k))

    # 9. Stress test sizes
    stress_sizes = [
        (10000, 1000, 1000), (1000, 10000, 1000), (1000, 1000, 10000),
        (5555, 5555, 5555), (7777, 7777, 7777), (9999, 9999, 9999),
        (10001, 10001, 10001),
    ]
    shapes.extend(stress_sizes)

    # Remove duplicates while preserving order
    seen = set()
    unique_shapes = []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            unique_shapes.append(s)

    # Sample down to target number if we have too many
    if len(unique_shapes) > num_shapes:
        # Stratified sampling to preserve diversity
        step = len(unique_shapes) / num_shapes
        unique_shapes = [unique_shapes[int(i * step)] for i in range(num_shapes)]

    return unique_shapes


def spec_to_feature_dict(spec: KernelSpec, dtype: str, layout: str) -> dict:
    """Convert KernelSpec to feature dict for ML predictor"""
    return {
        "kernel_name": spec.name,
        "tile_m": spec.tile_m, "tile_n": spec.tile_n, "tile_k": spec.tile_k,
        "warp_m": spec.wave_m, "warp_n": spec.wave_n, "warp_k": spec.wave_k,
        "warp_tile_m": spec.warp_m, "warp_tile_n": spec.warp_n, "warp_tile_k": spec.warp_k,
        "pipeline": spec.pipeline, "scheduler": spec.scheduler,
        "epilogue": "cshuffle",
        "pad_m": False, "pad_n": False, "pad_k": False, "persistent": False,
        "dtype": dtype, "layout": layout,
    }


def spec_to_kernel_config(spec: KernelSpec, dtype: str, arch: str) -> KernelConfig:
    """Convert KernelSpec to KernelConfig for dispatcher"""
    return KernelConfig(
        dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, dtype_acc="fp32",
        layout_a="row", layout_b="col", layout_c="row",
        tile_m=spec.tile_m, tile_n=spec.tile_n, tile_k=spec.tile_k,
        wave_m=spec.wave_m, wave_n=spec.wave_n, wave_k=spec.wave_k,
        warp_m=spec.warp_m, warp_n=spec.warp_n, warp_k=spec.warp_k,
        pipeline=spec.pipeline, scheduler=spec.scheduler, epilogue="cshuffle",
        gfx_arch=arch,
    )


def ml_select_kernel(predictor, pool: List[KernelSpec], M: int, N: int, K: int,
                     dtype: str, layout: str) -> Tuple[KernelSpec, float]:
    """Use ML model to select best kernel"""
    if not HAS_ML or predictor is None:
        # Fallback: select first kernel
        return pool[0], 0.0

    problem = {"m": M, "n": N, "k": K, "dtype": dtype, "layout": layout, "split_k": 1}
    kernel_dicts = [spec_to_feature_dict(s, dtype, layout) for s in pool]

    ranked = predictor.rank_kernels(problem, kernel_dicts)
    if not ranked:
        return pool[0], 0.0

    best_name, best_tflops = ranked[0]
    best_spec = next((s for s in pool if s.name == best_name), pool[0])
    return best_spec, best_tflops


def run_single_gemm(M: int, N: int, K: int, dtype: str, arch: str,
                    predictor, dry_run: bool = False) -> dict:
    """Run a single GEMM with ML heuristic selection"""

    # Select kernel via ML heuristic
    t0 = time.time()
    best_spec, pred_tflops = ml_select_kernel(predictor, KERNEL_POOL, M, N, K, dtype, "rcr")
    select_time_ms = (time.time() - t0) * 1000

    result = {
        'M': M, 'N': N, 'K': K,
        'dtype': dtype,
        'selected_kernel': best_spec.name,
        'predicted_tflops': pred_tflops,
        'selection_time_ms': select_time_ms,
        'actual_time_ms': 0,
        'actual_tflops': 0,
        'status': 'SKIP' if dry_run else 'PENDING',
        'error': None,
    }

    if dry_run:
        return result

    # Build and run kernel
    config = spec_to_kernel_config(best_spec, dtype, arch)

    try:
        setup = setup_gemm_dispatcher(
            config=config,
            registry_name=f"sweep_{dtype}_{best_spec.name}",
            verbose=False,
            auto_rebuild=True,
        )

        if not setup.success:
            result['status'] = 'BUILD_FAIL'
            result['error'] = 'Failed to build kernel'
            cleanup_gemm()
            return result

        dispatcher = setup.dispatcher
        if not dispatcher.is_supported(M, N, K):
            result['status'] = 'UNSUPPORTED'
            result['error'] = 'Problem size not supported by kernel'
            cleanup_gemm()
            return result

        # Create input data
        np_dtype = {'fp16': np.float16, 'bf16': np.float16, 'fp8': np.float16}[dtype]
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np_dtype)
        B = (np.random.randn(K, N) * 0.1).astype(np_dtype)

        # Run GEMM
        exec_result = dispatcher.run(A, B, M, N, K)

        if exec_result.success:
            result['actual_time_ms'] = exec_result.time_ms
            result['actual_tflops'] = exec_result.tflops
            result['status'] = 'SUCCESS'
        else:
            result['status'] = 'RUN_FAIL'
            result['error'] = 'Kernel execution failed'

        cleanup_gemm()

    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)[:200]
        cleanup_gemm()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ML Heuristic Sweep: Test GEMM across many shapes and dtypes"
    )
    parser.add_argument('--dtypes', nargs='+', default=['fp16', 'bf16', 'fp8'],
                        choices=['fp16', 'bf16', 'fp8'],
                        help='Data types to test')
    parser.add_argument('--arch', default='gfx942', help='GPU architecture')
    parser.add_argument('--model', default='model_tflops_log_big',
                        help='Model name or path (default: model_tflops_log_big)')
    parser.add_argument('--model_dir', default=None,
                        help='Path to model directory (auto-detect if not specified)')
    parser.add_argument('--num_shapes', type=int, default=1024,
                        help='Number of problem shapes to test (default: 1024)')
    parser.add_argument('--output', default='ml_heuristic_sweep_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only predict, do not run kernels')

    args = parser.parse_args()

    # Setup ML predictor
    predictor = None
    if HAS_ML:
        if args.model_dir is None:
            # Auto-detect model directory
            model_dirs = [
                Path(__file__).parent.parent.parent.parent / "heuristics" / "models" / args.model,
                Path(__file__).parent.parent.parent.parent / "heuristics" / "models" / "gemm_universal_fp8_gfx950",
                Path(__file__).parent.parent.parent.parent / "heuristics" / "models" / "gemm_universal_fp16_gfx942",
                Path(__file__).parent.parent.parent.parent / "heuristics" / "models",
            ]
            for model_dir in model_dirs:
                if model_dir.exists():
                    args.model_dir = str(model_dir)
                    break

        if args.model_dir and Path(args.model_dir).exists():
            try:
                predictor = Predictor(args.model_dir)
                print(f"✓ Loaded ML model from: {args.model_dir}")
            except Exception as e:
                print(f"⚠ Failed to load ML model: {e}")
                print("  Will use first-fit selection instead")
        else:
            print(f"⚠ Model directory not found: {args.model_dir}")
            print("  Will use first-fit selection instead")

    # Generate problem shapes
    print(f"\nGenerating {args.num_shapes} problem shapes...")
    shapes = generate_problem_shapes(args.num_shapes)
    print(f"✓ Generated {len(shapes)} unique shapes")

    # Count odd dimension shapes
    odd_count = sum(1 for m, n, k in shapes if m % 2 == 1 or n % 2 == 1 or k % 2 == 1)
    print(f"  - Shapes with odd dimensions: {odd_count} ({100*odd_count/len(shapes):.1f}%)")

    # Print configuration
    print("\n" + "=" * 80)
    print("  ML Heuristic Sweep Configuration")
    print("=" * 80)
    print(f"  Model:         {args.model}")
    print(f"  Data types:    {', '.join(args.dtypes)}")
    print(f"  Architecture:  {args.arch}")
    print(f"  Kernel pool:   {len(KERNEL_POOL)} kernels")
    print(f"  Problem shapes: {len(shapes)}")
    print(f"  Total tests:   {len(shapes) * len(args.dtypes)}")
    print(f"  Mode:          {'DRY RUN (prediction only)' if args.dry_run else 'FULL RUN (execute kernels)'}")
    print(f"  Output:        {args.output}")
    print("=" * 80)

    # Open output CSV
    csv_file = open(args.output, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'dtype', 'M', 'N', 'K',
        'selected_kernel', 'predicted_tflops', 'selection_time_ms',
        'actual_time_ms', 'actual_tflops',
        'status', 'error'
    ])
    csv_writer.writeheader()

    # Run sweep
    total_tests = len(shapes) * len(args.dtypes)
    completed = 0
    start_time = time.time()

    print(f"\nStarting sweep... (Ctrl+C to stop and save partial results)\n")

    try:
        for dtype in args.dtypes:
            print(f"\n{'='*80}")
            print(f"  Testing dtype: {dtype.upper()}")
            print(f"{'='*80}\n")

            for i, (M, N, K) in enumerate(shapes):
                result = run_single_gemm(M, N, K, dtype, args.arch, predictor, args.dry_run)

                # Write to CSV
                csv_writer.writerow(result)
                csv_file.flush()

                completed += 1

                # Progress update
                if completed % 10 == 0 or result['status'] != 'SUCCESS':
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_tests - completed) / rate if rate > 0 else 0

                    status_emoji = {
                        'SUCCESS': '✓',
                        'SKIP': '→',
                        'BUILD_FAIL': '✗',
                        'UNSUPPORTED': '○',
                        'RUN_FAIL': '✗',
                        'ERROR': '✗',
                    }.get(result['status'], '?')

                    print(f"  [{completed:4d}/{total_tests}] {status_emoji} "
                          f"{dtype:4s} {M:5d}x{N:5d}x{K:5d} → "
                          f"{result['selected_kernel']:20s} "
                          f"pred={result['predicted_tflops']:6.1f} "
                          f"actual={result['actual_tflops']:6.1f} TFLOPS  "
                          f"[{rate:.1f} tests/s, ETA {eta/60:.1f}m]")

    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted! Saving partial results to {args.output}...")

    finally:
        csv_file.close()

    # Summary
    print("\n" + "=" * 80)
    print("  SWEEP COMPLETE")
    print("=" * 80)

    # Read back results and compute statistics
    results = []
    with open(args.output, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)

    print(f"\n  Total tests:     {len(results)}")
    print(f"  Output file:     {args.output}")

    if not args.dry_run:
        success = [r for r in results if r['status'] == 'SUCCESS']
        print(f"  Successful:      {len(success)} ({100*len(success)/len(results):.1f}%)")

        if success:
            avg_tflops = np.mean([float(r['actual_tflops']) for r in success])
            max_tflops = max([float(r['actual_tflops']) for r in success])
            print(f"  Avg TFLOPS:      {avg_tflops:.2f}")
            print(f"  Max TFLOPS:      {max_tflops:.2f}")

            # Per-dtype breakdown
            for dtype in args.dtypes:
                dtype_results = [r for r in success if r['dtype'] == dtype]
                if dtype_results:
                    avg = np.mean([float(r['actual_tflops']) for r in dtype_results])
                    print(f"    {dtype:4s}:          {avg:.2f} TFLOPS (n={len(dtype_results)})")

    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
