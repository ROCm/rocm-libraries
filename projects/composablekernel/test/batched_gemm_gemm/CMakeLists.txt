# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_batched_gemm_gemm_fp16 test_batched_gemm_gemm_fp16.cpp)
if(result EQUAL 0)
  target_link_libraries(test_batched_gemm_gemm_fp16 PRIVATE utility device_batched_gemm_gemm_instance)
endif()

add_gtest_executable(test_batched_gemm_gemm_bf16_wmma_cshuffle_v3 test_batched_gemm_gemm_bf16_wmma_cshuffle_v3.cpp)
if(result EQUAL 0)
  target_link_libraries(test_batched_gemm_gemm_bf16_wmma_cshuffle_v3 PRIVATE utility device_batched_gemm_gemm_instance)
endif()
