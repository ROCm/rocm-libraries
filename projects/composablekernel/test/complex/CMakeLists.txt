# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_complex)

add_gtest_executable(test_complex_type test_complex_type.cpp)
if(result EQUAL 0)
  add_dependencies(test_complex test_complex_type)
endif()
