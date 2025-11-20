#!/usr/bin/env bash
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

set -eux

# Check if two arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <binary_dir> <config_dir> <output_dir>"
    exit 1
fi

binary_dir="${1%/}/"
config_dir="${2%/}/"

# Check if the folders exist
if [ -d "$binary_dir" ] && [ -d "$config_dir" ]; then
    echo "Both folders exist:"
    echo "$binary_dir"
    echo "$config_dir"
else
    echo "One or both folders do not exist."
    if [ ! -d "$binary_dir" ]; then
        echo "$binary_dir does not exist."
    fi
    if [ ! -d "$config_dir" ]; then
        echo "$config_dir does not exist."
    fi
fi


output_dir="${3%/}/"

cold_runs=1
hot_runs=5

validate=OFF

if [ -d "$binary_dir" ]; then
    # setup output directory for benchmarks
    mkdir -p "$output_dir"

    tests=("rank1_reduction_test"
           "rank2_reduction_test"
           "rank3_reduction_test"
           "rank4_reduction_test"
           "rank5_reduction_test"
           "rank6_reduction_test")

    configs=("rank1_test_params.yaml"
             "rank2_test_params.yaml"
             "rank3_test_params.yaml"
             "rank4_test_params.yaml"
             "rank5_test_params.yaml"
             "rank6_test_params.yaml")

    arrayLength=${#tests[@]}

    # run benchmarks
    for (( i=0; i<${arrayLength}; i++ )); do
        if [[ -e $binary_dir && ! -L $binary_dir/${tests[$i]} ]]; then
            $binary_dir${tests[$i]} -y $config_dir/${configs[$i]} \
            -o $output_dir${tests[$i]}".csv" --cold_runs $cold_runs --hot_runs $hot_runs -v $validate
        fi
    done
fi

