# MIT License
#
# Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_policy(VERSION 3.16...3.25)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

# rocPRIM project
project(rocprim LANGUAGES CXX)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(ROCPRIM_PROJECT_IS_TOP_LEVEL TRUE)
else()
  set(ROCPRIM_PROJECT_IS_TOP_LEVEL FALSE)
endif()

#Adding CMAKE_PREFIX_PATH
if(WIN32)
  set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

# Build options
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_NAIVE_BENCHMARK "Build naive benchmarks" OFF)
option(BUILD_EXAMPLE "Build examples" OFF)
option(BUILD_DOCS "Build documentation (requires sphinx)" OFF)
# Disables building tests, benchmarks, examples
option(ONLY_INSTALL "Only install" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(ROCPRIM_INSTALL "Enable installation of rocPRIM (projects embedding rocPRIM may want to turn this OFF)" ON)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${ROCM_PATH}/lib/cmake/hip ${HIP_PATH}/cmake ${ROCM_ROOT}/lib/cmake/hip ${ROCM_ROOT}/hip/cmake # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Set CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_STANDARD EQUAL 14)
  message(WARNING "C++14 will be deprecated in the next major release")
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

if(DEFINED BUILD_SHARED_LIBS)
  set(PKG_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
else()  
  set(PKG_BUILD_SHARED_LIBS ON)
endif()  
set(BUILD_SHARED_LIBS OFF) # don't build client dependencies as shared

# Get dependencies (required here to get rocm-cmake)
include(cmake/Dependencies.cmake)
# Use target ID syntax if supported for GPU_TARGETS
if (NOT DEFINED AMDGPU_TARGETS)
  set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
else()
  set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
endif()
set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

if(GPU_TARGETS STREQUAL "all")
  if(BUILD_ADDRESS_SANITIZER)
    # ASAN builds require xnack
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx940:xnack+;gfx941:xnack+;gfx942:xnack+"
    )
  else()
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"
    )
  endif()

  set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
endif()

# TODO: Fix VerifyCompiler for HIP on Windows
if (NOT WIN32)
  include(cmake/VerifyCompiler.cmake)
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm ${ROCM_ROOT}/llvm ${ROCM_ROOT} ${ROCM_ROOT}/hip)
find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
if(ROCPRIM_INSTALL AND BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_dir(
    "${PROJECT_SOURCE_DIR}/rocprim/include/rocprim"
    WRAPPER_LOCATIONS rocprim/include/rocprim
    OUTPUT_LOCATIONS rocprim/wrapper/include/rocprim
    PATTERNS *.hpp
  )
endif()

if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

# Setup VERSION
set(VERSION_STRING "3.3.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# rocPRIM library
add_subdirectory(rocprim)

if(ROCPRIM_PROJECT_IS_TOP_LEVEL AND NOT ONLY_INSTALL AND (BUILD_TEST OR BUILD_BENCHMARK))
  rocm_package_setup_component(clients)
endif()

# Tests
if(BUILD_TEST AND NOT ONLY_INSTALL)
  if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(tests)
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(BUILD_BENCHMARK AND NOT ONLY_INSTALL)
  if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(benchmarks)
  endif()
  add_subdirectory(benchmark)
endif()

# Examples
if(BUILD_EXAMPLE AND NOT ONLY_INSTALL)
  add_subdirectory(example)
endif()

# Docs
if(BUILD_DOCS AND NOT ONLY_INSTALL)
  add_subdirectory(docs)
endif()

# set BUILD_SHARED_LIBS for packaging
set(BUILD_SHARED_LIBS ${PKG_BUILD_SHARED_LIBS})
# Package
if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
  # add dependency on HIP runtime
  set(HIP_RUNTIME_MINIMUM 4.5.0)
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
  endif()

  rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")

  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

  rocm_create_package(
    NAME rocprim
    DESCRIPTION "Radeon Open Compute Parallel Primitives Library"
    MAINTAINER "rocPRIM Maintainer <rocprim-maintainer@amd.com>"
    HEADER_ONLY
  )
endif()

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
