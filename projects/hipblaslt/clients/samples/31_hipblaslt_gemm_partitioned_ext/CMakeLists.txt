# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_partitioned_ext "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_partitioned_ext.cpp")
# PartitionedGemm is experimental: roc::hipblaslt_experimental defines ROCM_ENABLE_EXPERIMENTAL_APIS.
target_link_libraries(sample_hipblaslt_gemm_partitioned_ext PRIVATE sample-helper roc::hipblaslt_experimental)
