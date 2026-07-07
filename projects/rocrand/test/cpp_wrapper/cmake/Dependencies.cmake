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

cmake_minimum_required(VERSION 3.16)

# Dependencies

# For downloading and building required dependencies
include(FetchContent)

# Google Test (https://github.com/google/googletest)
# NOTE: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
#
# FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
#
# GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
#
# NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
#        will be duplicately defined.
if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  find_package(GTest QUIET)
endif()

if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
  message(STATUS "Google Test not found or force download on. Fetching...")
  option(BUILD_GTEST "Builds the googletest subproject" ON)
  option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
  option(INSTALL_GTEST "Enable installation of googletest" OFF)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )
  FetchContent_MakeAvailable(googletest)
endif()
