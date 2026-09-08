# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

include_guard(DIRECTORY)

include(${CMAKE_CURRENT_LIST_DIR}/rocm-cmake.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

filter_cxx_flags_for_deps(CMAKE_CXX_FLAGS filtered_cmake_cxx_flags)
override_variable(ROCM_DISABLE_CHECKS ON)
override_variable(CMAKE_CXX_FLAGS ${filtered_cmake_cxx_flags})

if(WIN32)
  find_package(GTest 1.11.0 REQUIRED)
else()
  find_package(GTest)
endif()

if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
  # GTest not found, resort to downloading it ourselves.
  override_variable(BUILD_SHARED_LIBS OFF)

  set(BUILD_GTEST ON)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  if(EXISTS /usr/src/googletest)
    FetchContent_Declare(
      googletest
      SOURCE_DIR /usr/src/googletest
    )
  else()
    message(STATUS "Google Test not found. Fetching...")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.11.0
    )
  endif()
  FetchContent_MakeAvailable(googletest)

  add_library(GTest::GTest ALIAS gtest)
  add_library(GTest::Main ALIAS gtest_main)

  restore_variable(BUILD_SHARED_LIBS)
else()
  # 'find_package(GTest)' can return different targets depending on the CMake
  # version. See the following documentation pages:
  # * https://cmake.org/cmake/help/v3.19/module/FindGTest.html
  # * https://cmake.org/cmake/help/v3.20/module/FindGTest.html
  if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
    add_library(GTest::GTest ALIAS GTest::gtest)
    add_library(GTest::Main  ALIAS GTest::gtest_main)
  endif()
endif()

restore_variable(CMAKE_CXX_FLAGS)
restore_variable(ROCM_DISABLE_CHECKS)
