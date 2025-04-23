###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

# Include path for base unit tests files
set(ROCWMMA_TEST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${ROCWMMA_TEST_INCLUDE_DIRS})

# Unit common test sources
set(UnitCommonSources ${ROCWMMA_COMMON_TEST_SOURCES}
                      ${CMAKE_CURRENT_SOURCE_DIR}/unit_kernel_base.cpp)

# Custom target to build all rocWMMA unit tests
add_custom_target(rocwmma_unit_tests)

# Unit tests that are linked to custom target
function(add_rocwmma_unit_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_rocwmma_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add unit include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${ROCWMMA_TEST_INCLUDE_DIRS})

  # Add dependency to custom target
  add_dependencies(rocwmma_unit_tests ${TEST_TARGET})
endfunction()

# Add unit tests
add_subdirectory(contamination_test)
add_subdirectory(layout_test)
add_subdirectory(layout_traits_test)
add_subdirectory(map_util_test)
add_subdirectory(load_store_matrix_sync_test)
add_subdirectory(load_store_matrix_coop_sync_test)
add_subdirectory(fill_fragment_test)
add_subdirectory(vector_iterator_test)
add_subdirectory(vector_test)
add_subdirectory(vector_util_test)
add_subdirectory(pack_util_test)
add_subdirectory(io_traits_test)
add_subdirectory(cross_lane_ops_test)
add_subdirectory(io_shape_test)
add_subdirectory(tuple_test)
add_subdirectory(transforms_test)
add_subdirectory(unpack_util_test)
