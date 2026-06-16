#!/usr/bin/env python3
"""Convert DSL candidate sweep CSV to parquet for LightGBM training.

The DSL sweep binary (conv_candidate_sweep) writes one row per
(shape, candidate) with explicit tile columns:

    N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,
    tile_m,tile_n,tile_k,pipeline,tflops,latency_us

This differs from the ckProfiler format (which has a kernel_name string).
This script converts the DSL format to the parquet schema expected by
projects/composablekernel/dispatcher/heuristics/train.py.
"""

import argparse
import sys
import pandas as pd

# gfx942 (MI300X) hardware profile used for all DSL training data.
GFX942_HW = {
    "hw_num_cus": 228,
    "hw_simds_per_cu": 4,
    "hw_shader_engines": 28,
    "hw_max_clock_mhz": 2100,
    "hw_max_waves_per_cu": 32,
    "hw_wavefront_size": 64,
    "hw_lds_capacity": 65536,
    "hw_l1_cache_kb": 32,
    "hw_l2_cache_kb": 4096,
    "hw_l3_cache_kb": 262144,
    "hw_num_xcd": 8,
}

DSL_CSV_COLUMNS = [
    "N", "G", "C", "K", "Hi", "Wi", "Y", "X",
    "stride_h", "stride_w", "pad_h", "pad_w",
    "tile_m", "tile_n", "tile_k", "pipeline", "tflops", "latency_us",
]


def convert(input_path: str, output_path: str, arch: str, run_id: int, dtype: str = "fp16") -> None:
    df = pd.read_csv(input_path, header=0, names=DSL_CSV_COLUMNS,
                     on_bad_lines="skip")

    # Drop duplicate header rows that appear when shards are cat'd together.
    df = df[df["N"] != "N"].reset_index(drop=True)

    # Cast numeric columns.
    int_cols = ["N", "G", "C", "K", "Hi", "Wi", "Y", "X",
                "stride_h", "stride_w", "pad_h", "pad_w",
                "tile_m", "tile_n", "tile_k"]
    float_cols = ["tflops", "latency_us"]
    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)

    # Rename tile columns to match feature_engine_grouped_conv.py expectations.
    df = df.rename(columns={
        "tile_m": "gemm_m_per_block",
        "tile_n": "gemm_n_per_block",
        "tile_k": "gemm_k_per_block",
    })
    # CEngine conv always uses warp_m=2, warp_n=2, wavefront_size=64 → block_size=256.
    df["block_size"] = 256

    # DSL gfx942 kernels are all intrawave, no DSB or SI variants.
    df["wave_mode"] = "intrawave"
    df["has_dsb"] = 0
    df["has_si"] = 0

    # Timing: convert latency_us → latency_ms.
    df["latency_ms"] = df["latency_us"] / 1000.0
    df.drop(columns=["latency_us"], inplace=True)

    # Metadata.
    df["op_type"] = "grouped_conv"
    df["variant"] = "fwd"
    df["dtype"] = dtype
    df["arch"] = arch
    df["ndim_spatial"] = 2
    df["is_valid"] = (df["tflops"] > 0) & (df["latency_ms"] > 0)
    df["run_id"] = run_id

    # Synthetic kernel_name for compatibility with downstream tooling.
    df["kernel_name"] = (
        f"grouped_conv_fwd_{dtype}_nhwgc_2d_"
        + df["pipeline"] + "_intrawave_"
        + df["gemm_m_per_block"].astype(str) + "x"
        + df["gemm_n_per_block"].astype(str) + "x"
        + df["gemm_k_per_block"].astype(str)
    )

    # Hardware profile.
    for col, val in GFX942_HW.items():
        df[col] = val

    print(f"Rows: {len(df):,}")
    print(f"Unique shapes: {df.groupby(['N','G','C','K','Hi','Wi','Y','X','stride_h','stride_w','pad_h','pad_w']).ngroups:,}")
    print(f"TFLOPS range: {df['tflops'].min():.3f} – {df['tflops'].max():.3f}")
    print(f"Valid rows: {df['is_valid'].sum():,} / {len(df):,}")

    df.to_parquet(output_path, index=False)
    print(f"Written: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Merged CSV from cat shard_*.csv")
    parser.add_argument("--output", required=True,
                        help="Output parquet path")
    parser.add_argument("--arch", default="gfx942",
                        help="GPU architecture tag written into parquet (default: gfx942)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"],
                        help="Data type written into parquet (default: fp16)")
    parser.add_argument("--run-id", type=int, default=1,
                        help="Integer run identifier written into parquet (default: 1)")
    args = parser.parse_args()

    convert(args.input, args.output, args.arch, args.run_id, args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
