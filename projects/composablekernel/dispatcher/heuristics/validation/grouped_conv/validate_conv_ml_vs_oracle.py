#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validate a grouped-conv-forward ML model against a benchmark parquet.

The parquet must be long-format: one row per (conv shape × kernel config),
with a measured 'tflops' column and hw_* hardware columns embedded by
convert_csv_to_parquet.py.  Hardware parameters are read directly from the
parquet so no --arch flag is needed; a single parquet always covers one arch.

The model directory must contain model_tflops.lgbm (not .lgbm.gz — the C++
scorer cannot auto-decompress at runtime).

Methodology:
  1. Load the parquet and read hardware constants from its hw_* columns.
  2. Group rows by conv shape.
  3. For each shape, the oracle-best is the row with maximum measured TFLOPS.
  4. The model scores all kernels measured for that shape; the top-scored
     kernel's measured TFLOPS gives the model's efficiency for that shape.
  5. Report percentile efficiency, oracle-match rate, and the 10 worst shapes.

Usage:
    python validation/grouped_conv/validate_conv_ml_vs_oracle.py \\
        --model        models/grouped_conv_forward_fp16_gfx942 \\
        --oracle-parquet /path/to/conv_fp16_gfx942.parquet

    python validation/grouped_conv/validate_conv_ml_vs_oracle.py \\
        --model        models/grouped_conv_forward_bf16_gfx950 \\
        --oracle-parquet /path/to/conv_bf16_gfx950.parquet \\
        --output       /tmp/results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).parent
_HEURISTICS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_HEURISTICS_DIR))

from feature_engine_grouped_conv import GroupedConvFeatureEngine  # noqa: E402
from predict import Predictor  # noqa: E402

# Columns that identify a unique conv problem.
_SHAPE_COLS = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
               "stride_h", "stride_w", "pad_h", "pad_w"]

# hw_* columns written by convert_csv_to_parquet.py; strip prefix for
# GroupedConvFeatureEngine constructor kwargs.
_HW_COLS = [
    "hw_num_cus", "hw_lds_capacity", "hw_max_clock_mhz", "hw_simds_per_cu",
    "hw_shader_engines", "hw_max_waves_per_cu", "hw_wavefront_size",
    "hw_l1_cache_kb", "hw_l2_cache_kb", "hw_l3_cache_kb", "hw_num_xcd",
]


def _hw_kwargs_from_parquet(df: pd.DataFrame) -> dict:
    """Read hardware constants from the first parquet row and strip hw_ prefix."""
    missing = [c for c in _HW_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet is missing hardware columns: {missing}\n"
            "  Was it produced by convert_csv_to_parquet.py?"
        )
    row = df.iloc[0]
    return {col[len("hw_"):]: int(row[col]) for col in _HW_COLS}


def _problem_dict(row: pd.Series) -> dict:
    return {
        "N": int(row["N"]),
        "C": int(row["C"]),
        "K": int(row["K"]),
        "G": int(row.get("G", 1)),
        "Hi": int(row["Hi"]),
        "Wi": int(row["Wi"]),
        "Y": int(row["Y"]),
        "X": int(row["X"]),
        "stride_h": int(row["stride_h"]),
        "stride_w": int(row["stride_w"]),
        "pad_h": int(row["pad_h"]),
        "pad_w": int(row["pad_w"]),
        "dtype": str(row.get("dtype", "fp16")),
        # Pin depth dimensions: kConvSelectionDim == 2 in ConvImplicitGemmScorer.cpp
        "Di": 1, "Z": 1, "stride_d": 1, "pad_d": 0, "dilation_d": 1,
        "dilation_h": int(row.get("dilation_h", 1)),
        "dilation_w": int(row.get("dilation_w", 1)),
    }


def _kernel_dict(row: pd.Series) -> dict:
    return {
        "block_size": int(row.get("block_size", 256)),
        "gemm_m_per_block": int(row.get("gemm_m_per_block", 64)),
        "gemm_n_per_block": int(row.get("gemm_n_per_block", 64)),
        "pipeline": str(row.get("pipeline", "mem")),
        "wave_mode": str(row.get("wave_mode", "intrawave")),
        "has_dsb": int(row.get("has_dsb", 0)),
        "has_si": int(row.get("has_si", 0)),
    }


def validate(parquet: Path, model_dir: Path, output: Path) -> int:
    print("=" * 80)
    print("  Grouped-Conv Forward ML vs Oracle Validation")
    print("=" * 80)
    print(f"  Parquet : {parquet}")
    print(f"  Model   : {model_dir}")
    print()

    # Verify the plain .lgbm exists — the C++ scorer (ConvImplicitGemmScorer)
    # loads it at the path baked in at CMake time and cannot auto-decompress
    # .lgbm.gz.  A missing .lgbm causes the C++ scorer to silently fall back
    # to the analytic policy for every shape.
    lgbm_path = model_dir / "model_tflops.lgbm"
    if not lgbm_path.exists():
        gz_path = model_dir / "model_tflops.lgbm.gz"
        if gz_path.exists():
            print(f"ERROR: plain model not found at {lgbm_path}")
            print(f"  Decompress first: gunzip -k {gz_path}")
        else:
            print(f"ERROR: model not found at {lgbm_path}")
        return 1

    df = pd.read_parquet(parquet)
    print(f"Loaded {len(df):,} benchmark rows, {df['kernel_name'].nunique()} unique kernels")

    try:
        hw_kwargs = _hw_kwargs_from_parquet(df)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Hardware: num_cus={hw_kwargs['num_cus']}  "
          f"clock={hw_kwargs['max_clock_mhz']} MHz  "
          f"lds={hw_kwargs['lds_capacity']//1024}KB")
    print()

    df = df[df["tflops"].notna() & (df["tflops"] > 0)].copy()

    feature_engine = GroupedConvFeatureEngine(**hw_kwargs)
    predictor = Predictor(model_dir, feature_engine=feature_engine)
    print(f"Loaded predictor from {model_dir.name}")
    print(f"  Log targets: {predictor._log_targets}")
    print()

    present_shape_cols = [c for c in _SHAPE_COLS if c in df.columns]
    groups = df.groupby(present_shape_cols)
    print(f"Unique conv shapes: {len(groups)}")
    print()

    results = []
    skipped = 0

    for _shape_key, group in groups:
        oracle_row = group.loc[group["tflops"].idxmax()]
        oracle_tflops = float(oracle_row["tflops"])
        oracle_kernel = str(oracle_row["kernel_name"])

        problem = _problem_dict(oracle_row)

        best_pred_tflops = -1.0
        best_kernel_name = ""
        best_actual_tflops = 0.0

        for _, krow in group.iterrows():
            try:
                pred = predictor.predict_tflops(problem, _kernel_dict(krow))
            except Exception:
                continue
            if pred > best_pred_tflops:
                best_pred_tflops = pred
                best_kernel_name = str(krow["kernel_name"])
                best_actual_tflops = float(krow["tflops"])

        if best_kernel_name == "":
            skipped += 1
            continue

        efficiency = best_actual_tflops / oracle_tflops if oracle_tflops > 0 else 0.0
        results.append({
            "N": problem["N"], "G": problem["G"],
            "C": problem["C"], "K": problem["K"],
            "Hi": problem["Hi"], "Wi": problem["Wi"],
            "Y": problem["Y"], "X": problem["X"],
            "stride_h": problem["stride_h"], "pad_h": problem["pad_h"],
            "oracle_kernel": oracle_kernel,
            "oracle_tflops": oracle_tflops,
            "ml_kernel": best_kernel_name,
            "ml_predicted_tflops": best_pred_tflops,
            "ml_actual_tflops": best_actual_tflops,
            "efficiency": efficiency,
            "match": best_kernel_name == oracle_kernel,
            "n_kernels": len(group),
        })

    if skipped:
        print(f"  (skipped {skipped} shapes with no scoreable kernels)")
        print()

    print("=" * 80)
    print("  Results Summary")
    print("=" * 80)
    print()

    if not results:
        print("ERROR: no results to report")
        return 1

    df_r = pd.DataFrame(results)
    eff = df_r["efficiency"].values
    matches = df_r["match"].sum()

    print(f"Shapes evaluated : {len(df_r)}")
    print(f"Kernels per shape: {df_r['n_kernels'].mean():.1f} avg")
    print()
    print("Efficiency (ml_actual_tflops / oracle_tflops):")
    print(f"  Mean   : {np.mean(eff):.4f}  ({np.mean(eff)*100:.2f}%)")
    print(f"  Median : {np.median(eff):.4f}  ({np.median(eff)*100:.2f}%)")
    print(f"  P10    : {np.percentile(eff, 10):.4f}  ({np.percentile(eff, 10)*100:.2f}%)")
    print(f"  P90    : {np.percentile(eff, 90):.4f}  ({np.percentile(eff, 90)*100:.2f}%)")
    print(f"  Min    : {np.min(eff):.4f}  ({np.min(eff)*100:.2f}%)")
    print(f"  Max    : {np.max(eff):.4f}  ({np.max(eff)*100:.2f}%)")
    print()
    print(f"Oracle-best match: {matches}/{len(df_r)} ({100*matches/len(df_r):.1f}%)")
    print()

    for label, mask in [
        ("1x1 conv", (df_r["Y"] == 1) & (df_r["X"] == 1)),
        ("3x3 conv", (df_r["Y"] == 3) & (df_r["X"] == 3)),
        ("other   ", ~((df_r["Y"] == 1) & (df_r["X"] == 1)) &
                      ~((df_r["Y"] == 3) & (df_r["X"] == 3))),
    ]:
        sub = df_r[mask]
        if len(sub):
            print(f"  {label}: n={len(sub):4d}  "
                  f"mean={sub['efficiency'].mean()*100:.2f}%  "
                  f"P10={sub['efficiency'].quantile(0.1)*100:.2f}%")
    print()

    print("Worst 10 shapes (lowest efficiency):")
    print()
    for _, row in df_r.nsmallest(10, "efficiency").iterrows():
        print(f"  N={int(row['N'])} G={int(row['G'])} C={int(row['C'])} K={int(row['K'])} "
              f"H={int(row['Hi'])} W={int(row['Wi'])} "
              f"R={int(row['Y'])} S={int(row['X'])} "
              f"s={int(row['stride_h'])} p={int(row['pad_h'])}")
        print(f"    oracle : {row['oracle_tflops']:8.3f} TFLOPS  {row['oracle_kernel']}")
        print(f"    ml pick: {row['ml_actual_tflops']:8.3f} TFLOPS  {row['ml_kernel']}")
        print(f"    efficiency: {row['efficiency']*100:.2f}%")
        print()

    output.parent.mkdir(parents=True, exist_ok=True)
    df_r.to_csv(output, index=False)
    print(f"Results saved to: {output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, type=Path, metavar="DIR",
                   help="Model directory containing model_tflops.lgbm")
    p.add_argument("--oracle-parquet", required=True, type=Path, metavar="FILE",
                   help="Long-format benchmark parquet produced by convert_csv_to_parquet.py")
    p.add_argument("--output", type=Path, metavar="FILE",
                   default=Path("conv_ml_vs_oracle_results.csv"),
                   help="Output CSV (default: conv_ml_vs_oracle_results.csv)")
    args = p.parse_args()

    if not args.oracle_parquet.exists():
        print(f"ERROR: parquet not found: {args.oracle_parquet}")
        return 1
    if not args.model.exists():
        print(f"ERROR: model directory not found: {args.model}")
        return 1

    return validate(args.oracle_parquet, args.model, args.output)


if __name__ == "__main__":
    sys.exit(main())
