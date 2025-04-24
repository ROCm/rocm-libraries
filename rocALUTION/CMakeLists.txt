# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Needed to pickup static library files
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH})

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${ROCM_PATH}/lib/cmake/hip
     ${ROCM_PATH}/hip/cmake
     ${ROCM_PATH}/cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocALUTION project
project(rocalution LANGUAGES CXX)

# Build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build rocALUTION as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks (requires boost)" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" ON)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
option(BUILD_GLOBALTYPE_64 "Support global number of rows / columns exceeding 32 bits" OFF)
option(BUILD_LOCALTYPE_64 "Support local number of rows / columns exceeding 32 bits" OFF)
option(BUILD_PTRTYPE_64 "Support local number of non-zeros exceeding 32 bits" OFF)
option(BUILD_OPTCPU "Enable all instruction subsets supported by the local machine" OFF)

# Dependencies
include(cmake/Dependencies.cmake)

if(BUILD_ADDRESS_SANITIZER)
  set(DEFAULT_AMDGPU_TARGETS
    gfx908:xnack+
    gfx90a:xnack+
    gfx942:xnack+
  )
else()
  set(DEFAULT_AMDGPU_TARGETS
    gfx803
    gfx900:xnack-
    gfx906:xnack-
    gfx908:xnack-
    gfx90a:xnack-
    gfx90a:xnack+
    gfx942
    gfx950
    gfx1030
    gfx1100
    gfx1101
    gfx1102
    gfx1151
    gfx1200
    gfx1201
  )
endif()
if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
  message( DEPRECATION "AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS." )
endif()
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "Target default GPUs if AMDGPU_TARGETS is not defined. (Deprecated, prefer GPU_TARGETS)")
# Don't force, users should be able to override GPU_TARGETS at the command line if desired
set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to build for")

include(CheckLanguage)
include(CMakeDependentOption)
check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)
if(USE_HIPCXX)
  enable_language(HIP)
else()
  find_package(HIP MODULE) # hip_add_library is only provided by the find module
  if(NOT HIP_FOUND)
    message("-- HIP not found. Compiling WITHOUT HIP support.")
  endif()
endif()

cmake_dependent_option(SUPPORT_HIP "Compile WITH HIP support" ON "USE_HIPCXX OR HIP_FOUND" OFF)

# HIP related library dependencies
if(SUPPORT_HIP)
  if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
  endif( )
  find_package(rocblas REQUIRED)
  find_package(rocsparse REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(rocrand REQUIRED)
endif()

if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage -lgcov)
endif()

if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
    add_link_options(-fuse-ld=lld)
endif()


# Setup version
rocm_setup_version(VERSION "3.2.4")

if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests)
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks)
  endif()
endif()

# rocALUTION library
add_subdirectory(src)

# Trigger client builds if selected
if(BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

# Package specific CPACK vars
if(SUPPORT_HIP)
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan >= 4.5.0" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd >= 4.5.0" )
  endif()
  rocm_package_add_dependencies(DEPENDS "${DEPENDS_HIP_RUNTIME}" "rocsparse >= 1.12.10" "rocblas >= 2.22.0" "rocrand >= 2.1.0")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  set(INSTALL_PREFIX "C:/hipSDK")
  set(CPACK_SET_DESTDIR OFF)
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
  set(CPACK_PACKAGING_INSTALL_PREFIX "")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
  if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

# Package name
set(package_name rocalution)

set(ROCALUTION_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "ROCm library for sparse linear systems"
  MAINTAINER "rocALUTION Maintainer <rocalution-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCALUTION_CONFIG_DIR}
)

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  #  > make coverage_output (generate html documentation)
  #

  #
  # Run coverage analysis
  #
  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/rocalution-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis rocalution)

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'${CMAKE_SOURCE_DIR}/src/utils/rocsparseio.*'" "'${CMAKE_SOURCE_DIR}/src/base/host/host_io.*'" "'${CMAKE_SOURCE_DIR}/clients/*'" "'${CMAKE_SOURCE_DIR}/build/*'" "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
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
