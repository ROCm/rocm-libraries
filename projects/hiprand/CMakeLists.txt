# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)


if(DEFINED ENV{ROCM_PATH})
  set (ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "The path to the ROCm installation.")
endif()
set (ROCM_PATH "/opt/rocm" CACHE PATH "The path to the ROCm installation.")

# Install prefix
if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" )
  SET( INSTALL_PREFIX "C:/hipSDK" )
else()
  set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix, prepended onto install directories")
  list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} )
endif()

#
# hipRAND project
#
project(hipRAND CXX)
set(hipRAND_VERSION "3.1.0")

# Build options
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(BUILD_FORTRAN_WRAPPER "Build Fortran wrapper" OFF)
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(DEPENDENCIES_FORCE_DOWNLOAD "Do not use system installed dependencies, always download and build them" OFF)
option(DISABLE_WERROR "Disable building with Werror" ON)
option(DOWNLOAD_ROCRAND "Download rocRAND if it is not found in the default install directory or the manually-specified directory" OFF)
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
option(BUILD_SHARED_LIBS "Build shared" ON)

set(BUILD_WITH_LIB "ROCM" CACHE STRING "Select backend library to build with (ROCM or CUDA)")
set_property(CACHE BUILD_WITH_LIB PROPERTY STRINGS "ROCM" "CUDA")

if (BUILD_WITH_LIB STREQUAL "CUDA")
  # Not relevant for CUDA builds
  set(ROCRAND_PATH "" CACHE INTERNAL "")
  set(ROCRAND_ROOT "" CACHE INTERNAL "")
else()
  set(ROCRAND_PATH "" CACHE PATH "Path to rocRAND install.")
  set(ROCRAND_ROOT "${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand" CACHE PATH "Path to downloaded rocRAND install.")
endif()

set(HIPRAND_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

if (BUILD_TEST)
  set(GTEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/deps/gtest" CACHE PATH "Path to place downloaded googletest.")
else()
  set(GTEST_ROOT "" CACHE INTERNAL "")
endif()

mark_as_advanced(CMAKE_INSTALL_RPATH_USE_LINK_PATH)
mark_as_advanced(HIPRAND_CONFIG_DIR)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${HIP_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip # FindHIP.cmake
    ${ROCM_PATH}/llvm
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(cmake/ROCmCMakeBuildToolsDependency.cmake)

# Set target IDs for compilation
if (NOT BUILD_WITH_LIB STREQUAL "CUDA")
  if (NOT DEFINED AMDGPU_TARGETS)
    set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
  else()
    set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
  endif()
  set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

  if(GPU_TARGETS STREQUAL "all")
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"
    )
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
else()
  set(GPU_TARGETS "all" CACHE INTERNAL "")
endif()

# Verify that hipcc or HIP-aware clang is used on ROCM software
# TODO: Fix VerifyCompiler for Windows
if (NOT WIN32)
  include(cmake/VerifyCompiler.cmake)
else()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm)
  find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH})
endif()

# Include cmake scripts
include(cmake/Dependencies.cmake)

# Build CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(DISABLE_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

if (CMAKE_CXX_STANDARD EQUAL 14)
  message(WARNING "C++14 will be deprecated in the next major release")
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# HIP on Windows: xhip is required with clang++ to get __half defined
if (WIN32)
  add_compile_options(-xhip)
endif()

if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

if(BUILD_CODE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage -lgcov)
endif()

# Set version variables
rocm_setup_version( VERSION "${hipRAND_VERSION}" )
set ( hiprand_VERSION ${hipRAND_VERSION} )
# Old-style version number used within the library's API. rocrand_get_version should be modified.
math(EXPR hiprand_VERSION_NUMBER "${hipRAND_VERSION_MAJOR} * 100000 + ${hipRAND_VERSION_MINOR} * 100 + ${hipRAND_VERSION_PATCH}")
set(hiprand_SOVERSION 1.1)

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# hipRAND library
# This adds library targets: hiprand,
# also includes Fortran wrapper
add_subdirectory(library)

# Tests
if(BUILD_TEST)
    enable_testing()
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(tests)
    add_subdirectory(test)
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

if( HIP_RUNTIME_LOWER STREQUAL "rocclr" )
    if(BUILD_ADDRESS_SANITIZER)
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
    else()
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
    endif()
    rocm_package_add_dependencies("${DEPENDS_HIP_RUNTIME} >= 4.5.0")
elseif( HIP_RUNTIME STREQUAL "cuda" )
    rocm_package_add_dependencies("hip-nvcc >= 3.5.0")
endif( )
rocm_package_add_dependencies("rocrand >= ${hipRAND_VERSION}")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

if (BUILD_WITH_LIB STREQUAL "CUDA")
    set(package_name hiprand-alt)
else()
    set(package_name hiprand)
endif()

rocm_create_package(
        NAME ${package_name}
        DESCRIPTION "GPU-based random number generation library, written in HIP"
        MAINTAINER "hipRAND Maintainer <hiprand-maintainer@amd.com>"
        LDCONFIG
        LDCONFIG_DIR ${HIPRAND_CONFIG_DIR}
)

if(BUILD_CODE_COVERAGE)
  #
  # Run coverage analysis
  #
  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ctest --output-on-failure --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hiprand)

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory ./hiprand --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" "'*/test/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

  add_custom_target(coverage DEPENDS coverage_output)

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
