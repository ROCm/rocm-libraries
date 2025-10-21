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

# The ROCm platform requires at least Ubuntu 16.04 or Fedora 24, which have cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Adding CMAKE_PREFIX_PATH, needed for static builds
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake)

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

# rocSPARSE project
project(rocsparse LANGUAGES CXX C)

# Determine if CXX Compiler is hip-clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using hip-clang to build for amdgpu backend")
else()
  message(FATAL_ERROR "'hip-clang' compiler required to compile for ROCm platform.")
endif()

# Set CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CXX standard
if (CMAKE_CXX_STANDARD EQUAL 14)
  message( DEPRECATION "Builds using the C++14 standard will no longer be supported in the next major release" )
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# Build options
option(BUILD_SHARED_LIBS "Build rocSPARSE as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)
option(BUILD_CLIENTS_ONLY "Build only clients" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build rocSPARSE with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build rocSPARSE with address sanitizer enabled" OFF)
option(BUILD_MEMSTAT "Build rocSPARSE with memory statistics enabled" OFF)
option(BUILD_ROCSPARSE_ILP64 "Build rocSPARSE with rocsparse_int equal to int64_t" OFF)
option(BUILD_COMPRESSED_DBG "Enable compressed debug symbols" ON)
option(BUILD_WITH_ROCBLAS "Enable building rocSPARSE with rocBLAS" ON)
option(BUILD_WITH_ROCTX "Enable building rocSPARSE with rocTX" ON)
option(BUILD_WITH_OFFLOAD_COMPRESS
       "Enable offload compression during compilation" ON)

# Clients utilize rocsparse fortran API and a fortran compiler
if( NOT BUILD_FORTRAN_CLIENTS )
  option( BUILD_FORTRAN_CLIENTS "Build rocSPARSE clients requiring Fortran capabilities" ON )
endif( )
option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_WITH_OFFLOAD_COMPRESS)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("--offload-compress"
                          CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(NOT CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    message(
      FATAL_ERROR
        "The option '--offload-compress' is not supported from the CXX compiler, turn off the option BUILD_WITH_OFFLOAD_COMPRESS."
    )
  endif()
endif()

#
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage -lgcov)
endif()

if(BUILD_COMPRESSED_DBG)
  add_compile_options($<$<CONFIG:Debug>:-gz> $<$<CONFIG:RelWithDebInfo>:-gz>)
endif()

if(BUILD_ADDRESS_SANITIZER)

  # CXX
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -fuse-ld=lld")

  # C
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_LINK_EXECUTABLE "${CMAKE_C_LINK_EXECUTABLE} -fuse-ld=lld")

  # Fortran not supported, add_link_options below invalid for fortran linking
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

# Dependencies
include(cmake/Dependencies.cmake)

# Detect compiler support for target ID
if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Detect compiler support for target ID
# This section is deprecated. Please use rocm_check_target_ids for future use.
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
        OUTPUT_VARIABLE CXX_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

#Set the GPU_TARGETS with backward compatibility
if(COMMAND rocm_check_target_ids)
    if(BUILD_ADDRESS_SANITIZER)
      rocm_check_target_ids(DEFAULT_GPU_TARGETS
        TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+")
    else()
      rocm_check_target_ids(DEFAULT_GPU_TARGETS
          TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")
    endif()
else()
    # Use target ID syntax if supported for GPU_TARGETS
    if(TARGET_ID_SUPPORT)
        set(DEFAULT_GPU_TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")
    else()
        set(DEFAULT_GPU_TARGETS "gfx803;gfx900;gfx906;gfx908")
    endif()
endif()

#Support AMDGPU_TARGETS as deprecated
if( AMDGPU_TARGETS AND NOT GPU_TARGETS )
  message( DEPRECATION "AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS." )
  set( GFX_LIST ${AMDGPU_TARGETS} )
elseif( GPU_TARGETS )
  set( GFX_LIST ${GPU_TARGETS} )
else ()
  set( GFX_LIST ${DEFAULT_AMDGPU_TARGETS} ) # default if not set
endif()

set(GPU_TARGETS "${GFX_LIST}" CACHE STRING "GPU architectures to build for")

if (GPU_TARGETS)
  set(TMPGPU_TARGETS "${GPU_TARGETS}")
  if(TMPGPU_TARGETS STREQUAL "all" )
    set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target" FORCE)
  else()
    set(GPU_TARGETS "${TMPGPU_TARGETS}" CACHE STRING "AMD GPU targets to compile for" FORCE)
  endif()
else()
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()

message(STATUS "GPU_TARGETS: ${GPU_TARGETS}")

# Find rocprim package
find_package(rocprim 4.0.0 REQUIRED)

# Find rocblas package
if (BUILD_WITH_ROCBLAS)
  find_package(rocblas 4.1.0 QUIET)
else()
  message("-- Build rocSPARSE with rocBLAS is disabled")
endif()

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
endif( )

# Setup version
set(VERSION_STRING "4.2.0")
set(SOVERSION_STRING "1.0")

rocm_setup_version(VERSION ${VERSION_STRING})
set(rocsparse_SOVERSION ${SOVERSION_STRING})

if( BUILD_CLIENTS_ONLY OR BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  set( BUILD_CLIENTS ON )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  find_package(OpenMP QUIET)
  if(OPENMP_FOUND)
    set(OPENMP_RPM "libgomp")
    set(OPENMP_DEB "libomp-dev")
  endif()
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(CLIENTS_OS STREQUAL "sles" AND OPENMP_FOUND)
    set(OPENMP_RPM "libgomp1")
  elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()

  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if( BUILD_CLIENTS_TESTS )
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()
  if( BUILD_CLIENTS_BENCHMARKS )
  rocm_package_setup_client_component(
    benchmarks
    DEPENDS
      COMPONENT clients-common
      DEB "${OPENMP_DEB}"
      RPM "${OPENMP_RPM}"
  )
  endif()
  if(BUILD_FORTRAN_CLIENTS)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
  endif()
endif()

# rocSPARSE library
if (NOT BUILD_CLIENTS_ONLY)
  add_subdirectory(library)
endif()

# Trigger client builds if selected
if(BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

# Build docs
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Package specific CPACK vars
## add dependency on hip runtime for shared libraries
set(HIP_RUNTIME_MINIMUM 4.5.0)
if(BUILD_ADDRESS_SANITIZER)
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
else()
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
endif()
rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")

## add dependency on hip runtime for static libraries
rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

## add dependency on rocBLAS
if (rocblas_FOUND)
  set(ROCBLAS_MINIMUM "4.1.0")
  rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${ROCBLAS_MINIMUM}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${ROCBLAS_MINIMUM}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${ROCBLAS_MINIMUM}")
else()
  message("-- Build rocSPARSE with rocBLAS is disabled since rocBLAS is not found")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  endif()
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
set(package_name rocsparse)

set(ROCSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "ROCm SPARSE library"
  MAINTAINER "rocSPARSE Maintainer <rocsparse-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCSPARSE_CONFIG_DIR}
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
    COMMAND ./clients/staging/rocsparse-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis rocsparse)

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
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'${CMAKE_SOURCE_DIR}/clients/*'" "'${CMAKE_SOURCE_DIR}/build/*'" "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
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
