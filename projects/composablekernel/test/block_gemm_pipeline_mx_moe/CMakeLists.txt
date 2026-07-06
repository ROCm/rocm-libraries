# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# gfx950 compile guard for the MX MoE B-preshuffle v1 blockwise GEMM pipeline
# (protects against the ck::IndexEval<int, N> regression fixed by wrapping the
#  XDL pack constants in ck::Number<>). Header-only guard, links only utility.
add_gtest_executable(test_blockwise_gemm_pipeline_mx_moe_v1_gfx950
    test_blockwise_gemm_pipeline_mx_moe_v1_gfx950.cpp)
if(result EQUAL 0)
    target_link_libraries(test_blockwise_gemm_pipeline_mx_moe_v1_gfx950 PRIVATE utility)
endif()
