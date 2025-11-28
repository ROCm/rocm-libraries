# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Package test
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(rocrand_package_test CXX)

# CMake modules
list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} $ENV{ROCM_PATH}/hip)
list(APPEND CMAKE_MODULE_PATH
    $ENV{ROCM_PATH}/lib/cmake/hip
    ${HIP_PATH}/cmake $ENV{ROCM_PATH}/hip/cmake # FindHIP.cmake
)

if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

include(cmake/Dependencies.cmake)

# Find rocRAND
find_package(rocrand REQUIRED CONFIG HINTS ${rocrand_DIR} PATHS "$ENV{ROCM_PATH}/rocrand")

# Get sources
file(GLOB rocrand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../test_rocrand*.cpp)
# Exclude kernel tests
list(FILTER rocrand_pkg_TEST_SRCS EXCLUDE REGEX ".*test_rocrand_kernel.*\\.cpp$")

# Enable testing (ctest)
enable_testing()

function(add_relative_test test_name test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")
    add_test(NAME "${test_name}" COMMAND "./${rel_path}")
endfunction()

# Build
foreach(test_src ${rocrand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    if(USE_SYSTEM_LIB)
        target_link_libraries(${test_name} PRIVATE roc::rocrand GTest::gtest GTest::gtest_main)
    else()
        target_link_libraries(${test_name} PRIVATE rocrand GTest::gtest GTest::gtest_main)
    endif()
    target_link_libraries(${test_name} PRIVATE hip::host)
    add_relative_test(${test_name} ${test_name})
endforeach()
