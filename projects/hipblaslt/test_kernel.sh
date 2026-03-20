#!/bin/bash

# Optional: <m> <n> <k> (default: 256 256 256)
m="${1:-256}"
n="${2:-256}"
k="${3:-256}"

# Use the build's libhipblaslt and librocroller so the in-tree rocroller (with
# Operations::Tensor 4-arg constructor) is loaded instead of /opt/rocm's older lib.
export LD_LIBRARY_PATH="${PWD}/build/library:${PWD}/build/rocroller${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# # Point to Tensile library dir so custom kernels (YAML + .co) are found at runtime.
# export HIPBLASLT_TENSILE_LIBPATH="${PWD}/Tensile"
cd /workspace/rocm-libraries/projects/hipblaslt/build
ninja hipblaslt-bench && ./clients/hipblaslt-bench \
    --api_method c \
    -m "$m" -n "$n" -k "$k" \
    --alpha 1 --beta 0 \
    --transA T --transB N \
    --batch_count 1 \
    --scaleA 1001 --scaleB 1001 \
    --a_type f4_r --b_type f4_r \
    --c_type bf16_r --d_type bf16_r \
    --compute_type f32_r \
    --rotating 0 --cold_iters 1 --iters 1 \
    --print_kernel_info \
    --use_gpu_timer \
    --swizzleA \
    --verify 