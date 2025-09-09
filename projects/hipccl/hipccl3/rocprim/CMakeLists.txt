# MIT License
#
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Set CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set HIP flags
set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

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

include(CheckLanguage)
include(CMakeDependentOption)

# Build options
# Disables building tests, benchmarks, examples
option(ONLY_INSTALL "Only install" OFF)
cmake_dependent_option(BUILD_TEST "Build tests (requires googletest)" OFF "NOT ONLY_INSTALL" OFF)
option(WITH_ROCRAND "Build tests with device-side data generation(requires rocRAND)" OFF)
cmake_dependent_option(BUILD_BENCHMARK "Build benchmarks" OFF "NOT ONLY_INSTALL" OFF)
cmake_dependent_option(BUILD_EXAMPLE "Build examples" OFF "NOT ONLY_INSTALL" OFF)
option(BUILD_NAIVE_BENCHMARK "Build naive benchmarks" OFF)
cmake_dependent_option(BUILD_DOCS "Build documentation (requires sphinx)" OFF "NOT ONLY_INSTALL" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(ROCPRIM_INSTALL "Enable installation of rocPRIM (projects embedding rocPRIM may want to turn this OFF)" ON)
option(ROCPRIM_ENABLE_ASSERTS "Enable asserts in release build)" OFF)
option(BUILD_OFFLOAD_COMPRESS "Build rocPRIM with offload compression" ON)
cmake_dependent_option(USE_SYSTEM_LIB "Use installed ROCm libs when building tests" OFF BUILD_TEST OFF)

check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)

include(CheckCXXCompilerFlag)

if(BUILD_OFFLOAD_COMPRESS)
  check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-compress")
  else()
    message(STATUS "Warning: BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
  endif()
endif()

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

if(ROCPRIM_ENABLE_ASSERTS)
  if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
      string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
      set(BUILD_TYPE_CXX_FLAGS "CMAKE_CXX_FLAGS_${BUILD_TYPE}")
      set(BUILD_TYPE_C_FLAGS "CMAKE_C_FLAGS_${BUILD_TYPE}")
  endif()

  string(REGEX REPLACE "-DNDEBUG( |$)" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-DNDEBUG( |$)" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
      string(REGEX REPLACE "-DNDEBUG( |$)" "" ${BUILD_TYPE_CXX_FLAGS} "${${BUILD_TYPE_CXX_FLAGS}}")
      string(REGEX REPLACE "-DNDEBUG( |$)" "" ${BUILD_TYPE_C_FLAGS} "${${BUILD_TYPE_C_FLAGS}}")
  endif()
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

if(DEFINED BUILD_SHARED_LIBS)
  set(PKG_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
else()  
  set(PKG_BUILD_SHARED_LIBS ON)
endif()  
set(BUILD_SHARED_LIBS OFF) # don't build client dependencies as shared

# Get dependencies (required here to get rocm-cmake)
include(cmake/Dependencies.cmake)

# Use target ID syntax if supported for GPU_TARGETS
if(USE_HIPCXX)
  enable_language(HIP)
else()
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
        TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
      )
    else()
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
        TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"
      )
    endif()
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
endif()

# Compressed offload binaries are currently not working with the SPIR-V target
if("amdgcnspirv" IN_LIST GPU_TARGETS)
  if(BUILD_OFFLOAD_COMPRESS)
    message(FATAL_ERROR "Cannot combine SPIR-V and BUILD_OFFLOAD_COMPRESS")
  endif()
endif()

# TODO: Fix VerifyCompiler for HIP on Windows
if (NOT WIN32)
  include(cmake/VerifyCompiler.cmake)
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm ${ROCM_ROOT}/llvm ${ROCM_ROOT} ${ROCM_ROOT}/hip)
find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

# Setup VERSION
set(VERSION_STRING "4.1.0")
rocm_setup_version(VERSION ${VERSION_STRING})
math(EXPR rocprim_VERSION_NUMBER "${rocprim_VERSION_MAJOR} * 100000 + ${rocprim_VERSION_MINOR} * 100 + ${rocprim_VERSION_PATCH}")

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# rocPRIM library
add_subdirectory(rocprim)

if(ROCPRIM_PROJECT_IS_TOP_LEVEL AND (BUILD_TEST OR BUILD_BENCHMARK))
  rocm_package_setup_component(clients)
endif()

# Tests
if(BUILD_TEST)
  if(USE_SYSTEM_LIB)
    find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
    if (NOT ${rocprim_VERSION} VERSION_EQUAL ${VERSION_STRING})
      message(WARNING "The installed rocprim version, ${rocprim_VERSION}, does not match project version ${VERSION_STRING}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
    endif()
  endif()

  if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(tests)
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(BUILD_BENCHMARK)
  if (ROCPRIM_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(benchmarks)
  endif()
  add_subdirectory(benchmark)
endif()

# Examples
if(BUILD_EXAMPLE)
  add_subdirectory(example)
endif()

# Docs
if(BUILD_DOCS)
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

  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")

  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

  rocm_create_package(
    NAME rocprim
    DESCRIPTION "rocPRIM is a header-only library that provides HIP parallel primitives." 
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
