# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")

    add_custom_target(test_ck_tile_batched_gemm)

    add_gtest_executable(test_ck_tile_batched_gemm_f16 test_batched_gemm_f16.cpp)
    add_gtest_executable(test_ck_tile_batched_gemm_bf16 test_batched_gemm_bf16.cpp)

    add_dependencies(test_ck_tile_batched_gemm
        test_ck_tile_batched_gemm_f16
        test_ck_tile_batched_gemm_bf16)

    if(GPU_TARGETS MATCHES "gfx9|gfx12")
        set(CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS)
        if(CK_USE_OCP_FP8)
            list(APPEND CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
        endif()
        add_gtest_executable(test_ck_tile_batched_gemm_f8 test_batched_gemm_f8.cpp)
        add_gtest_executable(test_ck_tile_batched_gemm_bf8 test_batched_gemm_bf8.cpp)
        target_compile_options(test_ck_tile_batched_gemm_f8
            PRIVATE ${CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_batched_gemm_bf8
            PRIVATE ${CK_TILE_BATCHED_GEMM_F8_COMPILE_OPTIONS})
        add_dependencies(test_ck_tile_batched_gemm
            test_ck_tile_batched_gemm_f8
            test_ck_tile_batched_gemm_bf8)
    endif()

    if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12")
        add_gtest_executable(test_ck_tile_batched_gemm_int8 test_batched_gemm_int8.cpp)
        add_dependencies(test_ck_tile_batched_gemm test_ck_tile_batched_gemm_int8)
    endif()
endif()
