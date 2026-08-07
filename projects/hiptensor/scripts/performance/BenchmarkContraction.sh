#!/usr/bin/env bash
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set -eux

# Check if three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <binary_dir> <config_dir> <output_dir>"
    exit 1
fi

binary_dir="${1%/}/"
config_dir="${2%/}/"

# Both directories must exist; fail early (and loudly) otherwise. The config
# directory is passed in, so this works whether the benchmark yamls come from
# the source tree (test/01_contraction/configs/bench) or the install tree
# (bin/hiptensor/configs/bench/01_contraction).
if [ ! -d "$binary_dir" ]; then
    echo "ERROR: binary directory does not exist: $binary_dir" >&2
    exit 1
fi
if [ ! -d "$config_dir" ]; then
    echo "ERROR: config directory does not exist: $config_dir" >&2
    exit 1
fi

output_dir="${3%/}/"

cold_runs=1
hot_runs=5

validate=OFF

if [ -d "$binary_dir" ]; then
    # setup output directory for benchmarks
    mkdir -p "$output_dir"

    tests=("bilinear_contraction_test_m1n1k1"
         "bilinear_contraction_test_m2n2k2"
         "bilinear_contraction_test_m3n3k3"
         "bilinear_contraction_test_m4n4k4"
         "bilinear_contraction_test_m5n5k5"
         "bilinear_contraction_test_m6n6k6"
         "complex_bilinear_contraction_test_m1n1k1"
         "complex_bilinear_contraction_test_m2n2k2"
         "complex_bilinear_contraction_test_m3n3k3"
         "complex_bilinear_contraction_test_m4n4k4"
         "complex_bilinear_contraction_test_m5n5k5"
         "complex_bilinear_contraction_test_m6n6k6"
         "scale_contraction_test_m1n1k1"
         "scale_contraction_test_m2n2k2"
         "scale_contraction_test_m3n3k3"
         "scale_contraction_test_m4n4k4"
         "scale_contraction_test_m5n5k5"
         "scale_contraction_test_m6n6k6"
         "complex_scale_contraction_test_m1n1k1"
         "complex_scale_contraction_test_m2n2k2"
         "complex_scale_contraction_test_m3n3k3"
         "complex_scale_contraction_test_m4n4k4"
         "complex_scale_contraction_test_m5n5k5"
         "complex_scale_contraction_test_m6n6k6")

    configs=("bilinear_test_params_rank1.yaml"
             "bilinear_test_params_rank2.yaml"
             "bilinear_test_params_rank3.yaml"
             "bilinear_test_params_rank4.yaml"
             "bilinear_test_params_rank5.yaml"
             "bilinear_test_params_rank6.yaml"
             "complex_bilinear_test_params_rank1.yaml"
             "complex_bilinear_test_params_rank2.yaml"
             "complex_bilinear_test_params_rank3.yaml"
             "complex_bilinear_test_params_rank4.yaml"
             "complex_bilinear_test_params_rank5.yaml"
             "complex_bilinear_test_params_rank6.yaml"
             "scale_test_params_rank1.yaml"
             "scale_test_params_rank2.yaml"
             "scale_test_params_rank3.yaml"
             "scale_test_params_rank4.yaml"
             "scale_test_params_rank5.yaml"
             "scale_test_params_rank6.yaml"
             "complex_scale_test_params_rank1.yaml"
             "complex_scale_test_params_rank2.yaml"
             "complex_scale_test_params_rank3.yaml"
             "complex_scale_test_params_rank4.yaml"
             "complex_scale_test_params_rank5.yaml"
             "complex_scale_test_params_rank6.yaml")

    arrayLength=${#tests[@]}

    # run benchmarks
    for (( i=0; i<${arrayLength}; i++ )); do
        if [[ -e $binary_dir && ! -L $binary_dir/${tests[$i]} ]]; then
            $binary_dir${tests[$i]} -y $config_dir/${configs[$i]} \
            -o $output_dir${tests[$i]}".csv" --cold_runs $cold_runs --hot_runs $hot_runs -v $validate
        fi
    done
fi

