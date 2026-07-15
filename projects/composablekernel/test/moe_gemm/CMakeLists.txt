# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_moe_gemm1_xdl_fp16 test_moe_gemm1_xdl_fp16.cpp)
if(result EQUAL 0)
    target_link_libraries(test_moe_gemm1_xdl_fp16 PRIVATE utility)
endif()

add_gtest_executable(test_moe_gemm1_xdl_bf16 test_moe_gemm1_xdl_bf16.cpp)
if(result EQUAL 0)
    target_link_libraries(test_moe_gemm1_xdl_bf16 PRIVATE utility)
endif()
