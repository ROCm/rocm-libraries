# MIT License
#
# Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

#Adding CMAKE_PREFIX_PATH
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )

# rocPRIM project
project(rocprim LANGUAGES CXX)

# Build options
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build examples" OFF)
option(USE_HIP_CPU "Prefer HIP-CPU runtime instead of HW acceleration" OFF)
# Disables building tests, benchmarks, examples
option(ONLY_INSTALL "Only install" OFF)

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

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

if(NOT USE_HIP_CPU)
  # Get dependencies (required here to get rocm-cmake)
  include(cmake/Dependencies.cmake)
  # Set the AMDGPU_TARGETS with backward compatiblity
  # Use target ID syntax if supported for AMDGPU_TARGETS
  set(AMDGPU_TARGETS "all" CACHE STRING "Compile for which gpu architectures?")
  if(COMMAND rocm_check_target_ids)
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
    )
  else()
    # Detect compiler support for target ID
    # This section is deprecated. Please use rocm_check_target_ids for future use.
    if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
        OUTPUT_VARIABLE CXX_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
      string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
    endif()
    if(TARGET_ID_SUPPORT)
      set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx1030")
    else()
      set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908;gfx1030")
    endif()
  endif()
  if (AMDGPU_TARGETS)
    if( AMDGPU_TARGETS STREQUAL "all" )
      set( gpus "${DEFAULT_AMDGPU_TARGETS}")
    else()
      set( gpus "${AMDGPU_TARGETS}")
    endif()
    # must FORCE set this AMDGPU_TARGETS before any find_package( hip ...), in this file
    # to override CACHE var and set --offload-arch flags via hip-config.cmake hip::device dependency
    set( AMDGPU_TARGETS "${gpus}" CACHE STRING "AMD GPU targets to compile for" FORCE )
  endif()

  # Verify that hcc compiler is used on ROCM platform
  # TODO: Fix VerifyCompiler for HIP on Windows
  if (NOT WIN32)
    include(cmake/VerifyCompiler.cmake)
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm /opt/rocm/llvm /opt/rocm/hip)
  find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)
endif()
# Build options
# Disable -Werror
option(DISABLE_WERROR "Disable building with Werror" ON)
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build examples" OFF)
# Disables building tests, benchmarks, examples
option(ONLY_INSTALL "Only install" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)

if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

# Set CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_HIP_CPU)
  # Get dependencies
  include(cmake/Dependencies.cmake)
endif()

# Setup VERSION
set(VERSION_STRING "2.10.9")
rocm_setup_version(VERSION ${VERSION_STRING})

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# rocPRIM library
add_subdirectory(rocprim)

# Address Sanitizer
if(BUILD_ADDRESS_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# Tests
if(BUILD_TEST AND NOT ONLY_INSTALL)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(BUILD_BENCHMARK AND NOT ONLY_INSTALL)
  add_subdirectory(benchmark)
endif()

# Examples
if(BUILD_EXAMPLE AND NOT ONLY_INSTALL)
  add_subdirectory(example)
endif()

# Package

set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" )

rocm_create_package(
  NAME rocprim
  DESCRIPTION "Radeon Open Compute Parallel Primitives Library"
  MAINTAINER "rocPRIM Maintainer <rocprim-maintainer@amd.com>"
  HEADER_ONLY
)


#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > # run your tests
  #  > make coverage (generate html documentation)
  #

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage
    DEPENDS rocprim
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

