# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently TDM is only supported on gfx1250
if(GPU_TARGETS MATCHES "gfx1250")
    add_gtest_executable(test_ck_tile_grouped_gemm_mx_tdm test_mx_grouped_gemm.cpp)
    # target_compile_options(test_ck_tile_grouped_gemm_mx_tdm PRIVATE --save-temps)
endif()

# MX FLATMM grouped GEMM: MXFlatmmPipelineAGmemBGmemCRegV1 on gfx950 and gfx1250,
# WeightPreshufflePipelineAGmemBGmemCRegTDM on gfx1250 only
if(GPU_TARGETS MATCHES "gfx950|gfx1250")
    add_gtest_executable(test_ck_tile_grouped_gemm_mx_flatmm test_grouped_gemm_mx_flatmm.cpp)
endif()
