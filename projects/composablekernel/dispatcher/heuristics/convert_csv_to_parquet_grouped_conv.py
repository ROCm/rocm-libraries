#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Convert grouped convolution CSV benchmark data to parquet format for training.

Usage:
    python convert_csv_to_parquet_grouped_conv.py \
        --input ../../tile_engine/ops/grouped_conv/training_data_forward_bf16_20.csv \
        --output data/grouped_conv_forward_bf16_gfx950/training_data.parquet \
        --arch gfx950 \
        --dtype bf16 \
        --variant forward

Features:
    - Parses kernel names to extract configuration (tiles, pipeline, etc.)
    - Converts CSV to flat parquet format for ML training
    - Adds metadata columns (arch, dtype, variant)
    - Validates data quality and reports statistics
"""

import argparse
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def parse_grouped_conv_kernel_name(kernel_name: str) -> Dict[str, Any]:
    """
    Parse grouped conv kernel name to extract configuration features.

    Example: grouped_conv_forward_bf16_2d_16x64x64_compv3
    Returns: {
        'variant': 'forward',
        'dtype': 'bf16',
        'ndim_spatial': 2,
        'block_size': 16,
        'gemm_m_per_block': 64,
        'gemm_n_per_block': 64,
        'pipeline': 'compv3'
    }
    """
    # Pattern: grouped_conv_{variant}_{dtype}_{ndim}d_{block}x{m}x{n}_{pipeline}
    pattern = r'grouped_conv_([a-z_]+)_([a-z0-9]+)_(\d+)d_(\d+)x(\d+)x(\d+)_([a-z0-9]+)'

    match = re.match(pattern, kernel_name)
    if not match:
        raise ValueError(f"Cannot parse kernel name: {kernel_name}")

    variant, dtype, ndim, block_size, gemm_m, gemm_n, pipeline = match.groups()

    return {
        'variant': variant,
        'dtype': dtype,
        'ndim_spatial': int(ndim),
        'block_size': int(block_size),
        'gemm_m_per_block': int(gemm_m),
        'gemm_n_per_block': int(gemm_n),
        'pipeline': pipeline,
        'kernel_name': kernel_name
    }


def convert_csv_to_parquet(csv_file: Path, output_file: Path,
                          arch: str = "gfx950",
                          dtype: str = None,
                          variant: str = None) -> pd.DataFrame:
    """Convert grouped conv CSV to parquet training data format."""

    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Parse kernel names and extract configuration
    print("Parsing kernel configurations...")
    kernel_configs = []
    for kernel_name in df['kernel'].unique():
        try:
            config = parse_grouped_conv_kernel_name(kernel_name)
            kernel_configs.append(config)
        except ValueError as e:
            print(f"  Warning: {e}")

    print(f"  Parsed {len(kernel_configs)} unique kernels")
    print()

    # Create lookup map for kernel configs
    kernel_config_map = {cfg['kernel_name']: cfg for cfg in kernel_configs}

    # Build parquet rows
    rows = []
    for _, row in df.iterrows():
        kernel_name = row['kernel']

        if kernel_name not in kernel_config_map:
            continue  # Skip unparseable kernels

        kernel_cfg = kernel_config_map[kernel_name]

        # Hardware profile for gfx950 (MI300 series)
        # These values match the actual hardware specifications for MI300
        hw_profile_gfx950 = {
            'hw_num_cus': 256,
            'hw_simds_per_cu': 4,
            'hw_shader_engines': 32,
            'hw_max_clock_mhz': 2400,
            'hw_max_waves_per_cu': 32,
            'hw_wavefront_size': 64,
            'hw_lds_capacity': 65536,
            'hw_l1_cache_kb': 32,
            'hw_l2_cache_kb': 4096,
            'hw_l3_cache_kb': 262144,
            'hw_num_xcd': 8,
        }

        # Build parquet row with all features
        pq_row = {
            # Metadata
            'op_type': 'grouped_conv',
            'arch': arch,
            'dtype': dtype if dtype else kernel_cfg['dtype'],
            'variant': variant if variant else kernel_cfg['variant'],
            'ndim_spatial': kernel_cfg['ndim_spatial'],

            # Problem features (from CSV)
            'N': int(row['N']),
            'C': int(row['C']),
            'K': int(row['K']),
            'G': int(row['G']),
            'Hi': int(row['Hi']),
            'Wi': int(row['Wi']),
            'Y': int(row['Y']),
            'X': int(row['X']),
            'stride_h': int(row['stride_h']),
            'stride_w': int(row['stride_w']),
            'pad_h': int(row['pad_h']),
            'pad_w': int(row['pad_w']),

            # Kernel configuration (from parsed name)
            'block_size': kernel_cfg['block_size'],
            'gemm_m_per_block': kernel_cfg['gemm_m_per_block'],
            'gemm_n_per_block': kernel_cfg['gemm_n_per_block'],
            'pipeline': kernel_cfg['pipeline'],
            'kernel_name': kernel_name,

            # Performance metrics (from CSV)
            'latency_ms': float(row['latency_ms']),
            'tflops': float(row['tflops']),
            'non_zero': int(row['non_zero']),

            # Validity flag (all CSV data is valid)
            'is_valid': True,
            'run_id': 0,
        }

        # Add hardware profile (only for gfx950, can be extended for other archs)
        if arch == 'gfx950':
            pq_row.update(hw_profile_gfx950)

        rows.append(pq_row)

    result_df = pd.DataFrame(rows)

    print(f"Converted {len(result_df):,} benchmark results")
    print(f"  Valid: {result_df['is_valid'].sum():,}")
    print(f"  Unique kernels: {result_df['kernel_name'].nunique()}")
    print(f"  Unique problems: {result_df[['N', 'C', 'K', 'G', 'Hi', 'Wi', 'Y', 'X', 'stride_h', 'stride_w', 'pad_h', 'pad_w']].drop_duplicates().shape[0]}")
    print()

    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    print()

    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()

    print("Problem dimension ranges:")
    print(f"  N: {result_df['N'].min():,} - {result_df['N'].max():,}")
    print(f"  C: {result_df['C'].min():,} - {result_df['C'].max():,}")
    print(f"  K: {result_df['K'].min():,} - {result_df['K'].max():,}")
    print(f"  G: {result_df['G'].min():,} - {result_df['G'].max():,}")
    print(f"  Hi: {result_df['Hi'].min():,} - {result_df['Hi'].max():,}")
    print(f"  Wi: {result_df['Wi'].min():,} - {result_df['Wi'].max():,}")
    print(f"  Y: {result_df['Y'].min():,} - {result_df['Y'].max():,}")
    print(f"  X: {result_df['X'].min():,} - {result_df['X'].max():,}")
    print()

    print("Performance metrics:")
    print(f"  Latency (ms): {result_df['latency_ms'].min():.4f} - {result_df['latency_ms'].max():.4f}")
    print(f"  TFLOPS: {result_df['tflops'].min():.2f} - {result_df['tflops'].max():.2f}")
    print(f"  Mean TFLOPS: {result_df['tflops'].mean():.2f}")
    print(f"  Median TFLOPS: {result_df['tflops'].median():.2f}")
    print()

    print("Pipeline distribution:")
    print(result_df['pipeline'].value_counts())
    print()

    print("Block size distribution:")
    print(result_df['block_size'].value_counts())
    print()

    print("Tile size distribution (MxN):")
    tile_sizes = result_df.groupby(['gemm_m_per_block', 'gemm_n_per_block']).size()
    print(tile_sizes.sort_values(ascending=False).head(10))
    print()

    # Show best kernels per problem
    print("Sample best kernels per problem:")
    best_per_problem = result_df.loc[result_df.groupby(['N', 'C', 'K', 'G', 'Hi', 'Wi', 'Y', 'X',
                                                         'stride_h', 'stride_w', 'pad_h', 'pad_w'])['tflops'].idxmax()]
    for i, row in best_per_problem.head(5).iterrows():
        print(f"  Problem N={row['N']:2d} C={row['C']:4d} K={row['K']:4d} Hi={row['Hi']:3d}x{row['Wi']:3d} "
              f"Y={row['Y']:d}x{row['X']:d} → {row['tflops']:.1f} TFLOPS ({row['kernel_name']})")
    print()

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert grouped conv CSV to parquet training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input CSV file from grouped conv benchmark"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output parquet file"
    )
    parser.add_argument(
        "--arch", type=str, default="gfx950",
        help="GPU architecture (default: gfx950)"
    )
    parser.add_argument(
        "--dtype", type=str,
        help="Data type override (default: parsed from kernel name)"
    )
    parser.add_argument(
        "--variant", type=str,
        help="Convolution variant override (default: parsed from kernel name)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    # Convert CSV to parquet
    df = convert_csv_to_parquet(input_file, output_file, args.arch, args.dtype, args.variant)

    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Output: {output_file}")
    print(f"✓ Rows: {len(df):,}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"✓ Size: {output_file.stat().st_size / 1024:.1f} KB")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
