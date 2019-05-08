# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(add_rocprim_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_include_directories(${TEST_TARGET} SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
  )
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      rocprim_hip
      ${GTEST_BOTH_LIBRARIES}
  )
  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
  set_target_properties(${TEST_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/rocprim"
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

# HIP basic test, which also checks if there are no linkage problems when there are multiple sources
add_rocprim_test("rocprim.basic_test" "test_basic.cpp;detail/get_rocprim_version.cpp")

add_rocprim_test("rocprim.arg_index_iterator" test_arg_index_iterator.cpp)
add_rocprim_test("rocprim.block_discontinuity" test_block_discontinuity.cpp)
add_rocprim_test("rocprim.block_exchange" test_block_exchange.cpp)
add_rocprim_test("rocprim.block_histogram" test_block_histogram.cpp)
add_rocprim_test("rocprim.block_load_store" test_block_load_store.cpp)
add_rocprim_test("rocprim.block_radix_sort" test_block_radix_sort.cpp)
add_rocprim_test("rocprim.block_reduce" test_block_reduce.cpp)
add_rocprim_test("rocprim.block_scan" test_block_scan.cpp)
add_rocprim_test("rocprim.block_sort" test_block_sort.cpp)
add_rocprim_test("rocprim.constant_iterator" test_constant_iterator.cpp)
add_rocprim_test("rocprim.counting_iterator" test_counting_iterator.cpp)
add_rocprim_test("rocprim.device_binary_search" test_device_binary_search.cpp)
add_rocprim_test("rocprim.device_histogram" test_device_histogram.cpp)
add_rocprim_test("rocprim.device_merge" test_device_merge.cpp)
add_rocprim_test("rocprim.device_merge_sort" test_device_merge_sort.cpp)
add_rocprim_test("rocprim.device_partition" test_device_partition.cpp)
add_rocprim_test("rocprim.device_radix_sort" test_device_radix_sort.cpp)
add_rocprim_test("rocprim.device_reduce_by_key" test_device_reduce_by_key.cpp)
add_rocprim_test("rocprim.device_reduce" test_device_reduce.cpp)
add_rocprim_test("rocprim.device_run_length_encode" test_device_run_length_encode.cpp)
add_rocprim_test("rocprim.device_scan" test_device_scan.cpp)
add_rocprim_test("rocprim.device_segmented_radix_sort" test_device_segmented_radix_sort.cpp)
add_rocprim_test("rocprim.device_segmented_reduce" test_device_segmented_reduce.cpp)
add_rocprim_test("rocprim.device_segmented_scan" test_device_segmented_scan.cpp)
add_rocprim_test("rocprim.device_select" test_device_select.cpp)
add_rocprim_test("rocprim.device_transform" test_device_transform.cpp)
add_rocprim_test("rocprim.discard_iterator" test_discard_iterator.cpp)
add_rocprim_test("rocprim.texture_cache_iterator" test_texture_cache_iterator.cpp)
add_rocprim_test("rocprim.transform_iterator" test_transform_iterator.cpp)
add_rocprim_test("rocprim.intrinsics" test_intrinsics.cpp)
add_rocprim_test("rocprim.warp_reduce" test_warp_reduce.cpp)
add_rocprim_test("rocprim.warp_scan" test_warp_scan.cpp)
add_rocprim_test("rocprim.warp_sort" test_warp_sort.cpp)
add_rocprim_test("rocprim.zip_iterator" test_zip_iterator.cpp)
