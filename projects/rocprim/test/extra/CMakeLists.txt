# MIT License
#
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

include(CheckLanguage)
include(CMakeDependentOption)

check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)
if(USE_HIPCXX)
  set(PROJECT_LANG HIP)
else()
  set(PROJECT_LANG CXX)
endif()

# This project includes tests that should be run after
# rocPRIM is installed from package or using `make install`
project(rocprim_package_install_test ${PROJECT_LANG})

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
  ${HIP_PATH}/lib/cmake/hip ${HIP_PATH}/cmake /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake # FindHIP.cmake
)

# Use target ID syntax if supported for AMDGPU_TARGETS
if(TARGET_ID_SUPPORT)
  set(AMDGPU_TARGETS gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx1030 CACHE STRING "List of specific machine types for library to target")
else()
  set(AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target")
endif()

# Verify that supported compilers are being used
include(VerifyCompiler)

# Find rocPRIM
find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")

# Build CXX flags
if (NOT DEFINED CMAKE_${PROJECT_LANG}_STANDARD)
  set(CMAKE_${PROJECT_LANG}_STANDARD 17)
endif()
set(CMAKE_${PROJECT_LANG}_STANDARD_REQUIRED ON)
set(CMAKE_${PROJECT_LANG}_EXTENSIONS OFF)
set(CMAKE_${PROJECT_LANG}_FLAGS "${CMAKE_${PROJECT_LANG}_FLAGS} -Wall -Wextra -Werror")

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

# Enable testing (ctest)
enable_testing()

# Build
function(add_rocprim_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      ${rocprim_LIBRARIES} # roc::rocprim_hip
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()


add_rocprim_test("test_rocprim_package" test_rocprim_package.cpp)
