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

# Ungated: the assertions are compile-time only and never instantiate the gfx1250-only
# device entry points, whose non-gfx125 branch is a hard static_assert. Still compiled as
# HIP for every SUPPORTED_GPU_TARGETS, so keep device-side instantiations out of this file.
add_gtest_executable(test_barrier test_barrier.cpp)
