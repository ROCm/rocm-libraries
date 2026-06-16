# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18...3.25)

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

# ROCm libraries root (for shared/ctest and other shared resources)
# Only set default when not already defined (e.g. by superproject / rocm-libraries).
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
  set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
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
cmake_dependent_option(ROCPRIM_ENABLE_TOPK "Enable the device level topk and segmented topk algorithms" OFF BUILD_TEST ON)
option(BUILD_OFFLOAD_COMPRESS "Build rocPRIM with offload compression" ON)
cmake_dependent_option(USE_SYSTEM_LIB "Use installed ROCm libs when building tests" OFF BUILD_TEST OFF)

check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)

include(CheckCXXCompilerFlag)

if(BUILD_OFFLOAD_COMPRESS)
  check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
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

# Check if mold linker is available and use it if so
find_program(MOLD_LINKER mold)
if(MOLD_LINKER)
    message(STATUS "Using mold linker")
    add_link_options(-fuse-ld=mold)
else()
    message(STATUS "mold not found, using default linker")
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
        TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250"
      )
    endif()
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
endif()

# TODO: Fix VerifyCompiler for HIP on Windows
if (NOT WIN32)
  include(cmake/VerifyCompiler.cmake)
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm ${ROCM_ROOT}/llvm ${ROCM_ROOT} ${ROCM_ROOT}/hip)
find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

# Setup VERSION
set(VERSION_STRING "4.4.0")
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

  # Used by tests to define their RESOURCE_GROUPS
  set(DEFAULT_GPU_RESOURCE "gpus")

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

if(BUILD_CODE_COVERAGE)
  target_compile_options(rocprim INTERFACE
    -w
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(rocprim INTERFACE
    -fprofile-instr-generate
  )
endif()

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  add_custom_target(
    coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ctest --output-on-failure -E "rocprim.linking*" --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  get_property(LLVM_COV_OBJECT_ARGS GLOBAL PROPERTY LLVM_COV_OBJECT_ARGS)
  add_custom_target(
    coverage
    DEPENDS coverage_analysis
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/rocprim-coverage_*.profraw -o ./coverage-report/rocprim.profdata
    COMMAND ${LLVM_COV} report  ${LLVM_COV_OBJECT_ARGS} --ignore-filename-regex="test_*" -instr-profile=./coverage-report/rocprim.profdata
    COMMAND ${LLVM_COV} show  ${LLVM_COV_OBJECT_ARGS}  --ignore-filename-regex="test_*" -instr-profile=./coverage-report/rocprim.profdata -format=html -output-dir=coverage-report
    COMMAND ${LLVM_COV} export  ${LLVM_COV_OBJECT_ARGS} --ignore-filename-regex="test_*" -instr-profile=./coverage-report/rocprim.profdata -format=lcov > ./coverage-report/coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
