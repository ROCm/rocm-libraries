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

function(meta_add_rocprim_test TEST_NAME TEST_SOURCES ROCPRIM_TARGET)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_include_directories(${TEST_TARGET} SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
  )
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      ${ROCPRIM_TARGET}
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

function(add_rocprim_test_hc TEST_NAME TEST_SOURCES)
  meta_add_rocprim_test(${TEST_NAME} "${TEST_SOURCES}" rocprim_hc)
endfunction()

function(add_rocprim_test_hip TEST_NAME TEST_SOURCES)
  meta_add_rocprim_test(${TEST_NAME} "${TEST_SOURCES}" rocprim_hip)
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

#
# rocPRIM HC API tests
#

# HCP basic test, which also checks if there are no linkage problems when there are multiple sources
if(HIP_COMPILER STREQUAL "HCC")
add_rocprim_test_hc("rocprim.hc.basic_test" "test_hc_basic.cpp;detail/get_rocprim_version_hc.cpp")

add_rocprim_test_hc("rocprim.hc.arg_index_iterator" test_hc_arg_index_iterator.cpp)
add_rocprim_test_hc("rocprim.hc.block_discontinuity" test_hc_block_discontinuity.cpp)
add_rocprim_test_hc("rocprim.hc.block_exchange" test_hc_block_exchange.cpp)
add_rocprim_test_hc("rocprim.hc.block_histogram" test_hc_block_histogram.cpp)
add_rocprim_test_hc("rocprim.hc.block_load_store" test_hc_block_load_store.cpp)
add_rocprim_test_hc("rocprim.hc.block_radix_sort" test_hc_block_radix_sort.cpp)
add_rocprim_test_hc("rocprim.hc.block_reduce" test_hc_block_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.block_scan" test_hc_block_scan.cpp)
add_rocprim_test_hc("rocprim.hc.block_sort" test_hc_block_sort.cpp)
add_rocprim_test_hc("rocprim.hc.constant_iterator" test_hc_constant_iterator.cpp)
add_rocprim_test_hc("rocprim.hc.counting_iterator" test_hc_counting_iterator.cpp)
add_rocprim_test_hc("rocprim.hc.device_histogram" test_hc_device_histogram.cpp)
add_rocprim_test_hc("rocprim.hc.device_merge" test_hc_device_merge.cpp)
add_rocprim_test_hc("rocprim.hc.device_merge_sort" test_hc_device_merge_sort.cpp)
add_rocprim_test_hc("rocprim.hc.device_partition" test_hc_device_partition.cpp)
add_rocprim_test_hc("rocprim.hc.device_radix_sort" test_hc_device_radix_sort.cpp)
add_rocprim_test_hc("rocprim.hc.device_reduce_by_key" test_hc_device_reduce_by_key.cpp)
add_rocprim_test_hc("rocprim.hc.device_reduce" test_hc_device_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.device_run_length_encode" test_hc_device_run_length_encode.cpp)
add_rocprim_test_hc("rocprim.hc.device_scan" test_hc_device_scan.cpp)
add_rocprim_test_hc("rocprim.hc.device_segmented_radix_sort" test_hc_device_segmented_radix_sort.cpp)
add_rocprim_test_hc("rocprim.hc.device_segmented_reduce" test_hc_device_segmented_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.device_segmented_scan" test_hc_device_segmented_scan.cpp)
add_rocprim_test_hc("rocprim.hc.device_select" test_hc_device_select.cpp)
add_rocprim_test_hc("rocprim.hc.device_transform" test_hc_device_transform.cpp)
add_rocprim_test_hc("rocprim.hc.discard_iterator" test_hc_discard_iterator.cpp)
add_rocprim_test_hc("rocprim.hc.intrinsics" test_hc_intrinsics.cpp)
add_rocprim_test_hc("rocprim.hc.transform_iterator" test_hc_transform_iterator.cpp)
add_rocprim_test_hc("rocprim.hc.tuple" test_hc_tuple.cpp)
add_rocprim_test_hc("rocprim.hc.warp_reduce" test_hc_warp_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.warp_scan" test_hc_warp_scan.cpp)
add_rocprim_test_hc("rocprim.hc.warp_sort" test_hc_warp_sort.cpp)
add_rocprim_test_hc("rocprim.hc.zip_iterator" test_hc_zip_iterator.cpp)
endif()

#
# rocPRIM HIP API tests
#

# HIP basic test, which also checks if there are no linkage problems when there are multiple sources
add_rocprim_test_hip("rocprim.hip.basic_test" "test_hip_basic.cpp;detail/get_rocprim_version_hip.cpp")

add_rocprim_test_hip("rocprim.hip.arg_index_iterator" test_hip_arg_index_iterator.cpp)
add_rocprim_test_hip("rocprim.hip.block_discontinuity" test_hip_block_discontinuity.cpp)
add_rocprim_test_hip("rocprim.hip.block_exchange" test_hip_block_exchange.cpp)
add_rocprim_test_hip("rocprim.hip.block_histogram" test_hip_block_histogram.cpp)
add_rocprim_test_hip("rocprim.hip.block_load_store" test_hip_block_load_store.cpp)
add_rocprim_test_hip("rocprim.hip.block_radix_sort" test_hip_block_radix_sort.cpp)
add_rocprim_test_hip("rocprim.hip.block_reduce" test_hip_block_reduce.cpp)
add_rocprim_test_hip("rocprim.hip.block_scan" test_hip_block_scan.cpp)
add_rocprim_test_hip("rocprim.hip.block_sort" test_hip_block_sort.cpp)
add_rocprim_test_hip("rocprim.hip.constant_iterator" test_hip_constant_iterator.cpp)
add_rocprim_test_hip("rocprim.hip.counting_iterator" test_hip_counting_iterator.cpp)
add_rocprim_test_hip("rocprim.hip.device_histogram" test_hip_device_histogram.cpp)
add_rocprim_test_hip("rocprim.hip.device_merge" test_hip_device_merge.cpp)
add_rocprim_test_hip("rocprim.hip.device_merge_sort" test_hip_device_merge_sort.cpp)
add_rocprim_test_hip("rocprim.hip.device_partition" test_hip_device_partition.cpp)
add_rocprim_test_hip("rocprim.hip.device_radix_sort" test_hip_device_radix_sort.cpp)
add_rocprim_test_hip("rocprim.hip.device_reduce_by_key" test_hip_device_reduce_by_key.cpp)
add_rocprim_test_hip("rocprim.hip.device_reduce" test_hip_device_reduce.cpp)
add_rocprim_test_hip("rocprim.hip.device_run_length_encode" test_hip_device_run_length_encode.cpp)
add_rocprim_test_hip("rocprim.hip.device_scan" test_hip_device_scan.cpp)
add_rocprim_test_hip("rocprim.hip.device_segmented_radix_sort" test_hip_device_segmented_radix_sort.cpp)
add_rocprim_test_hip("rocprim.hip.device_segmented_reduce" test_hip_device_segmented_reduce.cpp)
add_rocprim_test_hip("rocprim.hip.device_segmented_scan" test_hip_device_segmented_scan.cpp)
add_rocprim_test_hip("rocprim.hip.device_select" test_hip_device_select.cpp)
add_rocprim_test_hip("rocprim.hip.device_transform" test_hip_device_transform.cpp)
add_rocprim_test_hip("rocprim.hip.discard_iterator" test_hip_discard_iterator.cpp)
add_rocprim_test_hip("rocprim.hip.texture_cache_iterator" test_hip_texture_cache_iterator.cpp)
add_rocprim_test_hip("rocprim.hip.transform_iterator" test_hip_transform_iterator.cpp)
add_rocprim_test_hip("rocprim.hip.intrinsics" test_hip_intrinsics.cpp)
add_rocprim_test_hip("rocprim.hip.warp_reduce" test_hip_warp_reduce.cpp)
add_rocprim_test_hip("rocprim.hip.warp_scan" test_hip_warp_scan.cpp)
add_rocprim_test_hip("rocprim.hip.warp_sort" test_hip_warp_sort.cpp)
add_rocprim_test_hip("rocprim.hip.zip_iterator" test_hip_zip_iterator.cpp)
