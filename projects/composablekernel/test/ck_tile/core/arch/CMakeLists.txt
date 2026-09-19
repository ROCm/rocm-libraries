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

# Buffer addressing is not architecture specific, so this is built for every
# target. cast_to_amdgpu_buffer_rsrc_t() is defined in two mutually exclusive
# headers selected by CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN, and only one of them
# can be reached from a given translation unit, so the same source is built once
# per setting to keep both definitions covered on any compiler.
add_gtest_executable(test_buffer_addressing test_buffer_addressing.cpp)
if(result EQUAL 0)
    target_compile_options(test_buffer_addressing PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
endif()

add_gtest_executable(test_buffer_addressing_no_builtin test_buffer_addressing.cpp)
if(result EQUAL 0)
    target_compile_options(test_buffer_addressing_no_builtin PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_definitions(test_buffer_addressing_no_builtin
                               PRIVATE CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN=0)
endif()
