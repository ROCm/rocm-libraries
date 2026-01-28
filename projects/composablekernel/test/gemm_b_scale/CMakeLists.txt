# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_b_scale test_gemm_b_scale.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_b_scale PRIVATE utility device_gemm_b_scale_instance)
endif()
