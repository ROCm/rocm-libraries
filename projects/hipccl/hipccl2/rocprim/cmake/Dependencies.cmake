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

# Dependencies

# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# HIP and nvcc configuration
find_package(HIP REQUIRED)
if(HIP_PLATFORM STREQUAL "nvcc")
  message(FATAL_ERROR "HIP_PLATFORM must be 'hcc' (AMD ROCm platform).")
elseif(HIP_PLATFORM STREQUAL "hcc")
  # Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
  # (Thanks to rocBLAS devs for finding workaround for this problem!)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
  # Ignore hcc warning: argument unused during compilation: '-isystem /opt/rocm/hip/include'
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
  find_package(hcc REQUIRED CONFIG PATHS /opt/rocm)
  find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
endif()

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

if (CMAKE_VERSION VERSION_LESS 3.2)
  set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
  set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED TRUE")
endif()

# Test dependencies
if(BUILD_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()

  if(NOT GTEST_FOUND)
    message(STATUS "GTest not found. Downloading and building GTest.")
    # Download, build and install googletest library
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
    download_project(PROJ   googletest
             GIT_REPOSITORY https://github.com/google/googletest.git
             GIT_TAG        master
             INSTALL_DIR    ${GTEST_ROOT}
             CMAKE_ARGS     -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD   TRUE
             LOG_CONFIGURE  TRUE
             LOG_BUILD      TRUE
             LOG_INSTALL    TRUE
             ${UPDATE_DISCONNECTED_IF_AVAILABLE}
    )
  endif()
  find_package(GTest REQUIRED)
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark QUIET)
  endif()
  
  if(NOT BENCHMARK_FOUND)
    message(STATUS "Google Benchmark not found. Downloading and building Google Benchmark.")
    # Download, build and install googlebenchmark library
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark CACHE PATH "")
    download_project(PROJ   googlebenchmark
             GIT_REPOSITORY https://github.com/google/benchmark.git
             GIT_TAG        master
             INSTALL_DIR    ${GOOGLEBENCHMARK_ROOT}
             CMAKE_ARGS     -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD   TRUE
             LOG_CONFIGURE  TRUE
             LOG_BUILD      TRUE
             LOG_INSTALL    TRUE
             ${UPDATE_DISCONNECTED_IF_AVAILABLE}
    )
  endif()
endif()
