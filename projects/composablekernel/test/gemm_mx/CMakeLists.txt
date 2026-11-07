# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_mx test_gemm_mx.cpp)
if(result EQUAL 0)
    target_compile_options(test_gemm_mx PRIVATE -mavx512f)
    target_link_libraries(test_gemm_mx PRIVATE utility device_gemm_mx_instance)
endif()

# Large M shapes need an Instinct GPU: gfx950 only (kept off the gfx1250 emulator stage).
if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_gemm_mx_fp4_large_tensor test_gemm_mx_fp4_large_tensor.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_mx_fp4_large_tensor PRIVATE utility)
    endif()
endif()
