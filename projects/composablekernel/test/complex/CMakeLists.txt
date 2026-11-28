# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_complex)

add_gtest_executable(test_complex_type test_complex_type.cpp)
if(result EQUAL 0)
  add_dependencies(test_complex test_complex_type)
endif()

add_gtest_executable(test_complex_load_store test_complex_load_store.cpp)
if(result EQUAL 0)
  target_link_libraries(test_complex_load_store PRIVATE utility)
  add_dependencies(test_complex test_complex_load_store)
endif()

add_gtest_executable(test_complex_arithmetic_device test_complex_arithmetic_device.cpp)
if(result EQUAL 0)
  target_link_libraries(test_complex_arithmetic_device PRIVATE utility)
  add_dependencies(test_complex test_complex_arithmetic_device)
endif()
