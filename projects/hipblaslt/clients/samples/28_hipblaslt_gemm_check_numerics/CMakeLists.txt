# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_check_numerics
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_check_numerics.cpp")
target_link_libraries(sample_hipblaslt_gemm_check_numerics PRIVATE sample-helper)
