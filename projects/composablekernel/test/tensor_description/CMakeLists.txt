# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_tensor_descriptor_element_space_size
                     test_tensor_descriptor_element_space_size.cpp)
if(result EQUAL 0)
   target_link_libraries(test_tensor_descriptor_element_space_size PRIVATE utility)
endif()
