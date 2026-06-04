# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently TDM is only supported on gfx1250
if(GPU_TARGETS MATCHES "gfx1250")
    add_gtest_executable(test_ck_tile_grouped_gemm_mx_tdm test_mx_grouped_gemm.cpp)
    # target_compile_options(test_ck_tile_grouped_gemm_mx_tdm PRIVATE --save-temps)
    add_gtest_executable(test_ck_tile_grouped_gemm_mx_flatmm_tdm test_grouped_gemm_mx_flatmm_tdm.cpp)
endif()

if(GPU_TARGETS MATCHES "gfx950|gfx1250")
    add_gtest_executable(test_ck_tile_grouped_gemm_mx_flatmm_non_tdm
                         test_grouped_gemm_mx_flatmm_non_tdm.cpp)
endif()

#   - non-TDM (MXFlatmmPipelineAGmemBGmemCRegV1):       gfx950 and gfx1250
#   - TDM    (WeightPreshufflePipelineAGmemBGmemCRegTDM): gfx1250 only

if(GPU_TARGETS MATCHES "gfx950|gfx1250")
    add_custom_target(test_ck_tile_grouped_gemm_mx_flatmm)
    add_dependencies(test_ck_tile_grouped_gemm_mx_flatmm
                     test_ck_tile_grouped_gemm_mx_flatmm_non_tdm)
    if(GPU_TARGETS MATCHES "gfx1250")
        add_dependencies(test_ck_tile_grouped_gemm_mx_flatmm
                         test_ck_tile_grouped_gemm_mx_flatmm_tdm)
    endif()
endif()

