#!/usr/bin/env python3
"""
Validate ML Heuristic on Unseen Shapes (Generalization Test)

This is the true test of ML generalization:
1. Select convolution shapes NOT in training data
2. Run all 20 kernels to find oracle best (on hardware)
3. Use ML to predict best kernel
4. Compare ML selected vs oracle best

This shows how well the model generalizes to unseen problems.
"""

import sys
import json
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # heuristics
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "tile_engine" / "ops" / "grouped_conv" / "problems"))

import pandas as pd
import numpy as np

from predict import Predictor
from feature_engine_grouped_conv import GroupedConvFeatureEngine
from grouped_conv_utils import (
    GroupedConvKernelConfig,
    setup_multiple_grouped_conv_dispatchers,
)

# Import MIOpen production shapes
from forward_training_miopen import TRAINING_PROBLEMS_FORWARD_MIOPEN


@dataclass
class KernelSpec:
    """Grouped convolution kernel specification"""

    name: str
    block_size: int
    gemm_m_per_block: int
    gemm_n_per_block: int
    pipeline: str = "compv3"

    def to_kernel_config(
        self, dtype: str = "bf16", arch: str = "gfx950"
    ) -> GroupedConvKernelConfig:
        """Convert to GroupedConvKernelConfig for building."""
        return GroupedConvKernelConfig(
            variant="forward",
            dtype=dtype,
            ndim_spatial=2,
            layout="NHWGC_KYXGC_NHWGK",
            arch=arch,
            tile_m=self.block_size,
            tile_n=self.gemm_m_per_block,
            tile_k=self.gemm_n_per_block,
            wave_m=2,
            wave_n=2,
            wave_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
            pipeline=self.pipeline,
            scheduler="default",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            pad_k=True,
        )


# Full kernel pool (20 kernels)
KERNEL_POOL = [
    # Block size 16
    KernelSpec("k16_64x64_v3", 16, 64, 64, "compv3"),
    KernelSpec("k16_64x64_v4", 16, 64, 64, "compv4"),
    KernelSpec("k16_64x128_v3", 16, 64, 128, "compv3"),
    KernelSpec("k16_64x128_v4", 16, 64, 128, "compv4"),
    # Block size 32
    KernelSpec("k32_64x64_v3", 32, 64, 64, "compv3"),
    KernelSpec("k32_64x64_v4", 32, 64, 64, "compv4"),
    KernelSpec("k32_64x128_v3", 32, 64, 128, "compv3"),
    KernelSpec("k32_64x128_v4", 32, 64, 128, "compv4"),
    KernelSpec("k32_128x64_v3", 32, 128, 64, "compv3"),
    KernelSpec("k32_128x64_v4", 32, 128, 64, "compv4"),
    # Block size 64
    KernelSpec("k64_64x64_v3", 64, 64, 64, "compv3"),
    KernelSpec("k64_64x64_v4", 64, 64, 64, "compv4"),
    KernelSpec("k64_64x128_v3", 64, 64, 128, "compv3"),
    KernelSpec("k64_64x128_v4", 64, 64, 128, "compv4"),
    KernelSpec("k64_128x64_v3", 64, 128, 64, "compv3"),
    KernelSpec("k64_128x64_v4", 64, 128, 64, "compv4"),
    # Block size 128
    KernelSpec("k128_64x128_v3", 128, 64, 128, "compv3"),
    KernelSpec("k128_64x128_v4", 128, 64, 128, "compv4"),
    KernelSpec("k128_128x64_v3", 128, 128, 64, "compv3"),
    KernelSpec("k128_128x64_v4", 128, 128, 64, "compv4"),
]


def build_kernel(
    spec: KernelSpec, dtype: str, arch: str, verbose: bool = False
) -> Optional[Path]:
    """Build a kernel on-demand using JIT compilation."""
    kernel_config = spec.to_kernel_config(dtype=dtype, arch=arch)

    lib_paths = setup_multiple_grouped_conv_dispatchers(
        [kernel_config], verbose=verbose, max_workers=1
    )

    if not lib_paths or lib_paths[0] is None:
        return None

    return lib_paths[0]


def run_kernel_on_hw(so_path: Path, problem: dict, kernel_name: str) -> dict:
    """Run a kernel on hardware via subprocess."""
    script_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "tile_engine"
        / "ops"
        / "grouped_conv"
        / "run_one_grouped_conv_kernel.py"
    )

    input_data = {
        "so_path": str(so_path),
        "problem": {**problem, "direction": "forward"},
        "kernel_name": kernel_name,
    }

    env = {
        **os.environ,
        "GCONV_PYPATH": str(Path(__file__).parent.parent.parent.parent / "python"),
    }

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stdout, stderr = proc.communicate(input=json.dumps(input_data).encode(), timeout=30)

    try:
        result = json.loads(stdout.decode().strip())
        return result
    except:
        return {"ok": False, "error": f"Failed to parse output"}


def is_shape_in_training(problem: dict, training_df: pd.DataFrame) -> bool:
    """Check if a problem shape exists in training data."""
    shape_cols = ["N", "C", "K", "G", "Hi", "Wi", "Y", "X", "stride_h", "stride_w", "pad_h", "pad_w"]

    mask = True
    for col in shape_cols:
        mask = mask & (training_df[col] == problem[col])

    return len(training_df[mask]) > 0


def find_oracle_best(
    problem: dict, kernels: List[KernelSpec], dtype: str, arch: str, verbose: bool = False
) -> Tuple[Optional[KernelSpec], float, Dict[str, float]]:
    """
    Run all kernels on hardware to find oracle best.

    Returns:
        (best_spec, best_tflops, all_results_dict)
    """
    results = {}

    for i, spec in enumerate(kernels):
        # Progress indicator
        print(f"    [{i+1:2d}/{len(kernels)}] {spec.name:<20}", end=" ", flush=True)

        # Build kernel
        so_path = build_kernel(spec, dtype, arch, verbose=False)
        if not so_path:
            print("SKIP (build)")
            continue

        # Run on hardware
        kernel_name = so_path.stem[3:] if so_path.stem.startswith("lib") else so_path.stem
        hw_result = run_kernel_on_hw(so_path, problem, kernel_name)

        if hw_result.get("ok"):
            tflops = hw_result["tflops"]
            results[spec.name] = tflops
            print(f"{tflops:>10.2f} TFLOPS")
        else:
            print("SKIP (run)")

    if not results:
        return None, 0.0, {}

    # Find best
    best_name = max(results, key=results.get)
    best_tflops = results[best_name]
    best_spec = next(s for s in kernels if s.name == best_name)

    return best_spec, best_tflops, results


def main():
    print("=" * 110)
    print("  ML Heuristic Generalization Test - Unseen Shapes")
    print("=" * 110)

    # Load training data to identify unseen shapes
    data_path = (
        Path(__file__).parent.parent.parent.parent
        / "heuristics"
        / "data"
        / "grouped_conv_forward_bf16_gfx950"
        / "training_data.parquet"
    )
    training_df = pd.read_parquet(data_path)

    print(f"\nLoaded training data: {len(training_df)} samples")

    # Get unique training shapes
    shape_cols = ["N", "C", "K", "G", "Hi", "Wi", "Y", "X", "stride_h", "stride_w", "pad_h", "pad_w"]
    training_shapes = training_df[shape_cols].drop_duplicates()
    print(f"Training shapes: {len(training_shapes)} unique problems")

    # Load ML model
    model_dir = (
        Path(__file__).parent.parent.parent.parent
        / "heuristics"
        / "models"
        / "grouped_conv_forward_bf16_gfx950"
    )
    feature_engine = GroupedConvFeatureEngine()
    predictor = Predictor(model_dir, feature_engine=feature_engine)
    print(f"Loaded ML model from {model_dir}")

    # Load MIOpen production shapes and convert to dicts
    print(f"\nLoading MIOpen production shapes...")
    miopen_shapes = []
    for prob in TRAINING_PROBLEMS_FORWARD_MIOPEN:
        shape_dict = {
            "N": prob.N,
            "C": prob.C,
            "K": prob.K,
            "G": prob.G,
            "Hi": prob.Hi,
            "Wi": prob.Wi,
            "Y": prob.Y,
            "X": prob.X,
            "stride_h": prob.stride_h,
            "stride_w": prob.stride_w,
            "pad_h": prob.pad_h,
            "pad_w": prob.pad_w,
        }
        miopen_shapes.append(shape_dict)

    print(f"Loaded {len(miopen_shapes)} MIOpen production shapes")

    # Filter for unseen shapes (not in training data)
    unseen_shapes = [s for s in miopen_shapes if not is_shape_in_training(s, training_df)]

    print(f"  Unseen shapes (not in training): {len(unseen_shapes)}")
    print(f"  Seen shapes (in training): {len(miopen_shapes) - len(unseen_shapes)}")

    if len(unseen_shapes) < 10:
        print(f"\n⚠ Warning: Only {len(unseen_shapes)} unseen shapes available")
        num_test = len(unseen_shapes)
    else:
        num_test = 10

    # Randomly select test shapes
    import random
    random.seed(42)
    unseen_shapes = random.sample(unseen_shapes, num_test)
    print(f"\nRandomly selected {num_test} unseen MIOpen shapes for testing")

    print()

    # Test each unseen shape
    results = []

    for shape_idx, problem in enumerate(unseen_shapes, 1):
        problem["dtype"] = "bf16"

        # Format problem description
        Ho = (problem["Hi"] + 2*problem["pad_h"] - problem["Y"]) // problem["stride_h"] + 1
        Wo = (problem["Wi"] + 2*problem["pad_w"] - problem["X"]) // problem["stride_w"] + 1
        prob_str = (
            f"C{problem['C']:4d}→K{problem['K']:4d} "
            f"{problem['Hi']:3d}x{problem['Wi']:3d}→{Ho:2d}x{Wo:2d} "
            f"f{problem['Y']}x{problem['X']} s{problem['stride_h']}x{problem['stride_w']}"
        )

        print(f"[{shape_idx}/{num_test}] {prob_str}")
        print("-" * 110)

        # Step 1: Find oracle best by running all kernels
        print(f"  Step 1: Finding oracle best (running all {len(KERNEL_POOL)} kernels)...")
        oracle_spec, oracle_tflops, all_results = find_oracle_best(
            problem, KERNEL_POOL, "bf16", "gfx950", verbose=False
        )

        if not oracle_spec:
            print(f"  ❌ SKIP: No kernels succeeded\n")
            continue

        print(f"  ✓ Oracle best: {oracle_spec.name} ({oracle_tflops:.2f} TFLOPS)")

        # Step 2: ML prediction
        print(f"\n  Step 2: ML prediction...")
        kernel_dicts = [
            {
                "kernel_name": s.name,
                "block_size": s.block_size,
                "gemm_m_per_block": s.gemm_m_per_block,
                "gemm_n_per_block": s.gemm_n_per_block,
                "pipeline": s.pipeline,
                "dtype": "bf16",
            }
            for s in KERNEL_POOL
        ]

        ranked = predictor.rank_kernels(problem, kernel_dicts)
        ml_name, ml_pred_tflops = ranked[0]

        # Get actual HW TFLOPS for ML selected kernel
        ml_actual_tflops = all_results.get(ml_name, 0.0)

        efficiency = (ml_actual_tflops / oracle_tflops) * 100 if oracle_tflops > 0 else 0

        print(f"  ✓ ML selected: {ml_name} (predicted {ml_pred_tflops:.2f} TFLOPS, actual {ml_actual_tflops:.2f} TFLOPS)")
        print(f"  ✓ Efficiency: {efficiency:.1f}% of oracle\n")

        results.append({
            "problem": prob_str,
            "oracle_name": oracle_spec.name,
            "oracle_tflops": oracle_tflops,
            "ml_name": ml_name,
            "ml_pred_tflops": ml_pred_tflops,
            "ml_actual_tflops": ml_actual_tflops,
            "efficiency": efficiency,
            "same_kernel": oracle_spec.name == ml_name,
            "num_kernels": len(all_results),
        })

    # Summary
    print("=" * 110)
    print("  SUMMARY")
    print("=" * 110)

    if results:
        print(f"\nTests completed: {len(results)}")

        # Detailed results table
        print(f"\n{'Problem':<45} {'Oracle':<20} {'ML':<20} {'Or TFLOPS':>10} {'ML TFLOPS':>10} {'Efficiency':>12}")
        print("-" * 110)
        for r in results:
            match_mark = "✓" if r["same_kernel"] else " "
            print(
                f"{r['problem']:<45} {r['oracle_name']:<20} {r['ml_name']:<20} "
                f"{r['oracle_tflops']:>10.2f} {r['ml_actual_tflops']:>10.2f} {r['efficiency']:>11.1f}% {match_mark}"
            )

        # Statistics
        avg_efficiency = np.mean([r["efficiency"] for r in results])
        same_kernel_count = sum(1 for r in results if r["same_kernel"])

        print(f"\n{'Metric':<40} {'Value':>20}")
        print("-" * 60)
        print(f"{'ML selected same as oracle':<40} {same_kernel_count}/{len(results)} ({(same_kernel_count/len(results))*100:.1f}%)")
        print(f"{'Average efficiency (ML vs Oracle)':<40} {avg_efficiency:>19.2f}%")

        avg_oracle = np.mean([r["oracle_tflops"] for r in results])
        avg_ml = np.mean([r["ml_actual_tflops"] for r in results])
        print(f"{'Average Oracle TFLOPS':<40} {avg_oracle:>20.2f}")
        print(f"{'Average ML TFLOPS':<40} {avg_ml:>20.2f}")

        # Prediction accuracy
        pred_accuracy = np.mean(
            [(r["ml_actual_tflops"] / r["ml_pred_tflops"]) * 100 for r in results if r["ml_pred_tflops"] > 0]
        )
        print(f"{'ML Prediction Accuracy':<40} {pred_accuracy:>19.1f}%")

        # Grade performance
        print()
        if avg_efficiency >= 95:
            print("✓ EXCELLENT: ML achieves >95% efficiency on unseen shapes!")
        elif avg_efficiency >= 90:
            print("✓ GOOD: ML achieves >90% efficiency on unseen shapes")
        elif avg_efficiency >= 85:
            print("⚠ FAIR: ML achieves >85% efficiency - room for improvement")
        else:
            print("⚠ NEEDS WORK: ML efficiency <85% - consider more training data or features")

    print("=" * 110)
    return 0 if results else 1


if __name__ == "__main__":
    main()
