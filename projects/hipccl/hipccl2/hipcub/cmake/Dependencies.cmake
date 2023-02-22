# MIT License
#
# Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

# ###########################
# hipCUB dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# CUB (only for CUDA platform)
if(HIP_COMPILER STREQUAL "nvcc")

  if(NOT DOWNLOAD_CUB)
    find_package(cub QUIET)
    find_package(thrust QUIET)
  endif()

  if(NOT DEFINED CUB_INCLUDE_DIR)
    file(
      DOWNLOAD https://github.com/NVIDIA/cub/archive/1.17.2.zip
      ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2.zip
      STATUS cub_download_status LOG cub_download_log
    )
    list(GET cub_download_status 0 cub_download_error_code)
    if(cub_download_error_code)
      message(FATAL_ERROR "Error: downloading "
        "https://github.com/NVIDIA/cub/archive/1.17.2.zip failed "
        "error_code: ${cub_download_error_code} "
        "log: ${cub_download_log} "
      )
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2.zip
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE cub_unpack_error_code
    )
    if(cub_unpack_error_code)
      message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2.zip failed")
    endif()
    set(CUB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2/ CACHE PATH "")
  endif()

  if(NOT DEFINED THRUST_INCLUDE_DIR)
    file(
      DOWNLOAD https://github.com/NVIDIA/thrust/archive/1.17.2.zip
      ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2.zip
      STATUS thrust_download_status LOG thrust_download_log
    )
    list(GET thrust_download_status 0 thrust_download_error_code)
    if(thrust_download_error_code)
      message(FATAL_ERROR "Error: downloading "
        "https://github.com/NVIDIA/thrust/archive/1.17.2.zip failed "
        "error_code: ${thrust_download_error_code} "
        "log: ${thrust_download_log} "
      )
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2.zip
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE thrust_unpack_error_code
    )
    if(thrust_unpack_error_code)
      message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2.zip failed")
    endif()
    set(THRUST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2/ CACHE PATH "")
  endif()
else()
  # rocPRIM (only for ROCm platform)
  if(NOT DOWNLOAD_ROCPRIM)
    find_package(rocprim QUIET)
  endif()
  if(NOT rocprim_FOUND)
    message(STATUS "Downloading and building rocprim.")
    download_project(
      PROJ                rocprim
      GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
      GIT_TAG             develop
      GIT_SHALLOW         TRUE
      INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
      CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    find_package(rocprim REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim NO_DEFAULT_PATH)
  endif()
endif()

# Test dependencies
if(BUILD_TEST)
  # NOTE1: Google Test has has two incompatible find_package paths: 
  #        the legacy FindGTest.cmake and the newer GTestConfig.cmake
  #
  # FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
  #
  # GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
  #        will be duplicately defined.
  #
  # NOTE3: The following snippet first tries to find Google Test binary either in MODULE or CONFIG modes.
  #        If neither succeeds it goes on to download Google Test from GitHub and defines the CONFIG
  #        mode targets. 
  #        Otherwise, if MODULE or CONFIG succeeded, then it prints the result to the
  #        console via a non-QUIET find_package call and if MODULE succeeded, creates ALIAS targets
  #        with the CONFIG names.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest 1.11.0 QUIET)
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    if(WIN32)
      set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=cl")
    endif()
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      GIT_SHALLOW         TRUE
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    list(APPEND CMAKE_PREFIX_PATH ${GTEST_ROOT})
    find_package(GTest 1.11.0 EXACT CONFIG REQUIRED PATHS ${GTEST_ROOT})
  else()
    find_package(GTest 1.11.0 REQUIRED)
    if(TARGET GTest::Main AND NOT TARGET GTest::gtest_main)
      add_library(GTest::gtest       ALIAS GTest::GTest)
      add_library(GTest::gtest_main  ALIAS GTest::Main)
    endif()
  endif()
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Benchmark (https://github.com/google/benchmark.git)
    find_package(benchmark QUIET)
  endif()

  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found or force download Google Benchmark on. Downloading and building Google Benchmark.")
    if(CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR "DownloadProject.cmake doesn't support multi-configuration generators.")
    endif()
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/googlebenchmark CACHE PATH "")
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
      # hip-clang cannot compile googlebenchmark for some reason
      if(WIN32)
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=cl")
      else()
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
      endif()
    endif()

    download_project(
      PROJ           googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.6.1
      GIT_SHALLOW    TRUE
      INSTALL_DIR    ${GOOGLEBENCHMARK_ROOT}
      CMAKE_ARGS     -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_STANDARD=14 ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD   TRUE
      LOG_CONFIGURE  TRUE
      LOG_BUILD      TRUE
      LOG_INSTALL    TRUE
      BUILD_PROJECT  TRUE
      UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()
