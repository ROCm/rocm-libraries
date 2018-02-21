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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

# CMake modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Select hcc as a C++ compiler
include(cmake/SetToolchain.cmake)

# rocPRIM project
project(rocprim VERSION 0.1.0.0 LANGUAGES CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# Build options
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build examples" OFF)

# Get dependencies
include(cmake/Dependencies.cmake)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# rocPRIM works only on hcc
if(HIP_PLATFORM STREQUAL "hcc")
  # rocPRIM library
  add_subdirectory(library/rocprim)
endif()

# hipCUB library
add_subdirectory(library/hipcub)

# Tests
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks and examples are only for rocPRIM
if(HIP_PLATFORM STREQUAL "hcc")
  # Benchmarks
  if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
  endif()

  # Examples
  if(BUILD_EXAMPLE)
    add_subdirectory(example)
  endif()
endif()
