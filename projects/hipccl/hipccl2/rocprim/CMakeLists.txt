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

cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

# rocPRIM project
project(rocprim VERSION 0.3.2.0 LANGUAGES CXX)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Verify that hcc compiler is used on ROCM platform
include(cmake/VerifyCompiler.cmake)

# Set CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# Build options
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build examples" OFF)
# Disables building tests, benchmarks, examples
option(ONLY_INSTALL "Only install" OFF)

# Get dependencies
include(cmake/Dependencies.cmake)

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# rocPRIM works only on hcc
if(HIP_PLATFORM STREQUAL "hcc")
  # rocPRIM library
  add_subdirectory(rocprim)
endif()

# hipCUB library
add_subdirectory(hipcub)

# Tests
if(BUILD_TEST AND NOT ONLY_INSTALL)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks and examples are only for rocPRIM
if(HIP_PLATFORM STREQUAL "hcc")
  # Benchmarks
  if(BUILD_BENCHMARK AND NOT ONLY_INSTALL)
    add_subdirectory(benchmark)
  endif()

  # Examples
  if(BUILD_EXAMPLE AND NOT ONLY_INSTALL)
    add_subdirectory(example)
  endif()
endif()

# Package

if(HIP_PLATFORM STREQUAL "hcc")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.4)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.4")
else()
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_nvcc (>= 1.3)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip_nvcc >= 1.3")
endif()
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" )

# For CUDA backend the package contains only hipcub
if(HIP_PLATFORM STREQUAL "hcc")
  rocm_create_package(
    NAME rocprim
    DESCRIPTION "Radeon Open Compute Parallel Primitives Libary"
    MAINTAINER "Jakub Szuppe <jakub@streamhpc.com>"
  )
else()
  rocm_create_package(
    NAME rocprim-hipcub
    DESCRIPTION "Radeon Open Compute Parallel Primitives Libary (hipCUB only)"
    MAINTAINER "Jakub Szuppe <jakub@streamhpc.com>"
  )
endif()
