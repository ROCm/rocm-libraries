# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_ck_tile_gemm_decode test_gemm_decode.cpp)

    if(CK_USE_OCP_FP8)
        target_compile_options(test_ck_tile_gemm_decode PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endif()
