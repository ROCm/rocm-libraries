# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_batched_gemm test_batched_gemm.cpp)
if(result EQUAL 0)
   target_link_libraries(test_batched_gemm PRIVATE utility device_batched_gemm_instance)
endif()
