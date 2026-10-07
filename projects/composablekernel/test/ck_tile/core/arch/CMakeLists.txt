# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_subdirectory(mma)

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

if(GPU_TARGETS MATCHES "gfx9|amdgcnspirv")
    add_gtest_executable(test_arch test_arch.cpp)
    target_compile_options(test_arch PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping test_arch tests for current target")
endif()

# Pure compile-time checks of the VGPR budget model; valid for every target.
add_gtest_executable(test_vgpr_budget test_vgpr_budget.cpp)
