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

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/rocm-cmake.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

filter_cxx_flags_for_deps(CMAKE_CXX_FLAGS filtered_cmake_cxx_flags)
override_variable(ROCM_DISABLE_CHECKS ON)
override_variable(CMAKE_CXX_FLAGS ${filtered_cmake_cxx_flags})

if(WIN32)
  find_package(GTest 1.11.0 REQUIRED)
else()
  find_package(GTest QUIET)
endif()

# Google Test has created a mess with legacy FindGTest.cmake and newer
# GTestConfig.cmake
#
# FindGTest.cmake defines:
#   GTest::GTest, GTest::Main, GTEST_FOUND
#
# GTestConfig.cmake defines:
#   GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
#
# Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode,
# because targets will be duplicately defined.
#
# The following snippet first tries to find Google Test binary either in
# MODULE or CONFIG modes.If neither succeeds it goes on to import Google Test
# into this build either from a system source package (apt install googletest
# on Ubuntu 18.04 only) or GitHub and defines the MODULE mode targets.
# Otherwise if MODULE or CONFIG succeeded, then it prints the result to the
# console via a non-QUIET find_package call and if CONFIG succeeded, creates
# ALIAS targets with the MODULE IMPORTED names.
if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)

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
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main  ALIAS gtest_main)
  endif()

  restore_variable(BUILD_SHARED_LIBS)
else()
  find_package(GTest REQUIRED)
  if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
    add_library(GTest::GTest ALIAS GTest::gtest)
    add_library(GTest::Main  ALIAS GTest::gtest_main)
  endif()
endif()

restore_variable(CMAKE_CXX_FLAGS)
restore_variable(ROCM_DISABLE_CHECKS)
