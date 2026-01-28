# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_gemm_multi_abd test_gemm_multi_abd.cpp)
if(result EQUAL 0)
    target_link_libraries(test_gemm_multi_abd PRIVATE utility device_gemm_multi_abd_instance)
endif()