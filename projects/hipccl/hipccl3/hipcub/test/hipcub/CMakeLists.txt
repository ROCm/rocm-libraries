# MIT License
#
# Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.10
if((HIP_COMPILER STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.10"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/hipcub/include/hipcub"
        "${PROJECT_BINARY_DIR}/hipcub/include"
        "${PROJECT_SOURCE_DIR}/hipcub/include"
        ${GTEST_INCLUDE_DIRS}
        ${CUB_INCLUDE_DIR}
    )
endif()

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_include_directories(${TEST_TARGET} SYSTEM BEFORE
      PUBLIC
        ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        hipcub
        ${GTEST_BOTH_LIBRARIES}
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${TEST_TARGET}
        PRIVATE
          --amdgpu-target=${amdgpu_target}
      )
    endforeach()
  else()
    set_source_files_properties(${TEST_SOURCES}
      PROPERTIES
          CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    CUDA_ADD_EXECUTABLE(${TEST_TARGET}
      ${TEST_SOURCES}
      OPTIONS
        --expt-extended-lambda
    )
    target_include_directories(${TEST_TARGET} SYSTEM BEFORE
      PUBLIC
        ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_TARGET}
      hipcub_cub
      ${GTEST_BOTH_LIBRARIES}
    )
  endif()
  set_target_properties(${TEST_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/hipcub"
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

# HIP basic test, which also checks if there are no linkage problems when there are multiple sources
add_hipcub_test("hipcub.BasicTest" "test_hipcub_basic.cpp;detail/get_hipcub_version.cpp")

add_hipcub_test("hipcub.CachingDeviceAllocator" test_hipcub_caching_device_allocator.cpp)
add_hipcub_test("hipcub.BlockDiscontinuity" test_hipcub_block_discontinuity.cpp)
add_hipcub_test("hipcub.BlockExchange" test_hipcub_block_exchange.cpp)
add_hipcub_test("hipcub.BlockHistogram" test_hipcub_block_histogram.cpp)
add_hipcub_test("hipcub.BlockLoadStore" test_hipcub_block_load_store.cpp)
add_hipcub_test("hipcub.BlockRadixSort" test_hipcub_block_radix_sort.cpp)
add_hipcub_test("hipcub.BlockReduce" test_hipcub_block_reduce.cpp)
add_hipcub_test("hipcub.BlockScan" test_hipcub_block_scan.cpp)
add_hipcub_test("hipcub.DeviceHistogram" test_hipcub_device_histogram.cpp)
add_hipcub_test("hipcub.DeviceRadixSort" test_hipcub_device_radix_sort.cpp)
add_hipcub_test("hipcub.DeviceReduce" test_hipcub_device_reduce.cpp)
add_hipcub_test("hipcub.DeviceRunLengthEncode" test_hipcub_device_run_length_encode.cpp)
add_hipcub_test("hipcub.DeviceReduceByKey" test_hipcub_device_reduce_by_key.cpp)
add_hipcub_test("hipcub.DeviceScan" test_hipcub_device_scan.cpp)
add_hipcub_test("hipcub.DeviceSegmentedRadixSort" test_hipcub_device_segmented_radix_sort.cpp)
add_hipcub_test("hipcub.DeviceSegmentedReduce" test_hipcub_device_segmented_reduce.cpp)
add_hipcub_test("hipcub.DeviceSelect" test_hipcub_device_select.cpp)
add_hipcub_test("hipcub.UtilPtx" test_hipcub_util_ptx.cpp)
add_hipcub_test("hipcub.WarpReduce" test_hipcub_warp_reduce.cpp)
add_hipcub_test("hipcub.WarpScan" test_hipcub_warp_scan.cpp)
