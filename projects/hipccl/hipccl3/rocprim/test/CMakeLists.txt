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
  target_include_directories(${TEST_TARGET} SYSTEM
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
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
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

# HC and HIP tests without using rocPRIM
add_rocprim_test_hc("hip.device_api" test_hc_api.cpp)
add_rocprim_test_hip("hip.device_api" test_hip_api.cpp)

# HC and HIP basic tests, which also checks if there are no linkage problems when there are multiple sources
add_rocprim_test_hip("rocprim.hip.basic_test" "test_rocprim_hip_basic.cpp;detail/get_rocprim_version_hip.cpp")
add_rocprim_test_hc("rocprim.hc.basic_test" "test_rocprim_hc_basic.cpp;detail/get_rocprim_version_hc.cpp")

# rocPRIM HC API tests
add_rocprim_test_hc("rocprim.hc.block_discontinuity" test_rocprim_hc_block_discontinuity.cpp)
add_rocprim_test_hc("rocprim.hc.block_exchange" test_rocprim_hc_block_exchange.cpp)
add_rocprim_test_hc("rocprim.hc.block_load_store" test_rocprim_hc_block_load_store.cpp)
add_rocprim_test_hc("rocprim.hc.block_radix_sort" test_rocprim_hc_block_radix_sort.cpp)
add_rocprim_test_hc("rocprim.hc.block_reduce" test_rocprim_hc_block_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.block_scan" test_rocprim_hc_block_scan.cpp)
add_rocprim_test_hc("rocprim.hc.device_radix_sort" test_rocprim_hc_device_radix_sort.cpp)
add_rocprim_test_hc("rocprim.hc.device_reduce_by_key" test_rocprim_hc_device_reduce_by_key.cpp)
add_rocprim_test_hc("rocprim.hc.device_reduce" test_rocprim_hc_device_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.device_scan" test_rocprim_hc_device_scan.cpp)
add_rocprim_test_hc("rocprim.hc.device_transform" test_rocprim_hc_device_transform.cpp)
add_rocprim_test_hc("rocprim.hc.intrinsics" test_rocprim_hc_intrinsics.cpp)
add_rocprim_test_hc("rocprim.hc.warp_reduce" test_rocprim_hc_warp_reduce.cpp)
add_rocprim_test_hc("rocprim.hc.warp_scan" test_rocprim_hc_warp_scan.cpp)
add_rocprim_test_hc("rocprim.hc.warp_sort" test_rocprim_hc_warp_sort.cpp)

# rocPRIM HIP API tests
add_rocprim_test_hip("rocprim.hip.block_discontinuity" test_rocprim_hip_block_discontinuity.cpp)
add_rocprim_test_hip("rocprim.hip.block_exchange" test_rocprim_hip_block_exchange.cpp)
add_rocprim_test_hip("rocprim.hip.block_load_store" test_rocprim_hip_block_load_store.cpp)
add_rocprim_test_hip("rocprim.hip.block_radix_sort" test_rocprim_hip_block_radix_sort.cpp)
add_rocprim_test_hip("rocprim.hip.device_radix_sort" test_rocprim_hip_device_radix_sort.cpp)
add_rocprim_test_hip("rocprim.hip.device_reduce_by_key" test_rocprim_hip_device_reduce_by_key.cpp)
add_rocprim_test_hip("rocprim.hip.device_reduce" test_rocprim_hip_device_reduce.cpp)
add_rocprim_test_hip("rocprim.hip.device_scan" test_rocprim_hip_device_scan.cpp)
add_rocprim_test_hip("rocprim.hip.device_transform" test_rocprim_hip_device_transform.cpp)
add_rocprim_test_hip("rocprim.hip.warp_scan" test_rocprim_hip_warp_scan.cpp)
