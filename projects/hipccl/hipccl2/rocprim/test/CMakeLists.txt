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

function(add_hip_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_include_directories(${TEST_TARGET} SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
  )

  # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
  if(TARGET hip::device)
    target_link_libraries(${TEST_TARGET} PRIVATE hip::device)
  else()
    target_link_libraries(${TEST_TARGET} PRIVATE hip::hip_hcc hip::hip_device)
  endif()

  target_link_libraries(${TEST_TARGET}
    PRIVATE
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

function(add_hc_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_include_directories(${TEST_TARGET} SYSTEM BEFORE
    PUBLIC
      ${GTEST_INCLUDE_DIRS}
  )
  if(HIP_COMPILER STREQUAL "HCC")
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        hcc::hccrt
        hcc::hc_am
        ${GTEST_BOTH_LIBRARIES}
    )
  elseif(HIP_COMPILER STREQUAL "clang")
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        ${GTEST_BOTH_LIBRARIES}
    )
  endif()
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

# ****************************************************************************
# Tests
# ****************************************************************************

# HC and HIP tests without using rocPRIM
if(HIP_PLATFORM STREQUAL "hcc")
  add_hip_test("hip.device_api"  test_hip_api.cpp)
  if(HIP_COMPILER STREQUAL "HCC")
    add_hc_test ("hc.device_api" test_hc_api.cpp)
  endif()
endif()

# rocPRIM test (run only on ROCm/hcc)
if(HIP_PLATFORM STREQUAL "hcc")
  add_subdirectory(rocprim)
endif()

# hipCUB tests
add_subdirectory(hipcub)
