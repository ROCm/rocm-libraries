#!/usr/bin/env bash
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set -eux

# ensure this script is in the cwd
cd "$(dirname "${BASH_SOURCE[0]}")"

output_dir=rocwmma-benchmarks
build_dir=../../build/test/gemm/

if [ -d "$build_dir" ]; then
  # setup output directory for benchmarks
  mkdir -p "$output_dir"

  gemm_bench=("gemm_PGR0_LB0_MP0_SB_NC" "gemm_PGR0_LB0_MP0_MB_NC" "gemm_PGR1_LB2_MP0_MB_CP_BLK" "gemm_PGR1_LB2_MP0_MB_CP_WG" "gemm_PGR1_LB2_MP0_MB_CP_WV")

  # run benchmarks
  for f in ${gemm_bench[@]}; do
    if [[ -e $build_dir/$f-bench && ! -L $build_dir/$f-bench ]]; then
      mkdir -p $output_dir/rocWMMA_$f
      $build_dir$f"-bench" --output_stream "$output_dir/rocWMMA_$f/${f}-benchmark.csv"
    fi
  done
fi

