# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_subdirectory(arch)
add_subdirectory(container)

add_gtest_executable(test_l2_swizzle test_l2_swizzle.cpp)
if(result EQUAL 0)
  target_compile_options(test_l2_swizzle PRIVATE -std=c++17 -O2)
endif()
