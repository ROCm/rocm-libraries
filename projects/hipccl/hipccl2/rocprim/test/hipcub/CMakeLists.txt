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

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  if(HIP_PLATFORM STREQUAL "hcc")
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_include_directories(${TEST_TARGET} SYSTEM
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
    target_include_directories(${TEST_TARGET} SYSTEM
      PUBLIC
        ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(${TEST_TARGET}
      hipcub
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

add_hipcub_test("hipcub.BlockDiscontinuity" test_hipcub_block_discontinuity.cpp)
add_hipcub_test("hipcub.BlockExchange" test_hipcub_block_exchange.cpp)
add_hipcub_test("hipcub.BlockRadixSort" test_hipcub_block_radix_sort.cpp)
add_hipcub_test("hipcub.BlockScan" test_hipcub_block_scan.cpp)
add_hipcub_test("hipcub.DeviceRadixSort" test_hipcub_device_radix_sort.cpp)
add_hipcub_test("hipcub.DeviceScan" test_hipcub_device_scan.cpp)
add_hipcub_test("hipcub.WarpScan" test_hipcub_warp_scan.cpp)
