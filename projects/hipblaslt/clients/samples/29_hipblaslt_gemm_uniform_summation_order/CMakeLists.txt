# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_uniform_summation_order "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_uniform_summation_order.cpp")
target_link_libraries(sample_hipblaslt_gemm_uniform_summation_order PRIVATE sample-helper)
