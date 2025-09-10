# ##########################################################################
# Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

cmake_minimum_required(VERSION 3.13)

# This has to be initialized before the project() command appears
# Set the default build type to Release
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
  set(CMAKE_Fortran_COMPILER  "gfortran")
endif()

# ROCM_BUILD_ID is added to the package name by rocm-cmake. Unsetting it prevents that.
unset(ENV{ROCM_BUILD_ID})

# Disable ROCMClang detection to make CMake v3.21 work the same as CMake v3.20 and earlier.
# https://gitlab.kitware.com/cmake/cmake/-/merge_requests/6533
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0 AND CMAKE_VERSION VERSION_LESS 3.21.3)
  set(__skip_rocmclang ON)
endif()
message(STATUS "Using CMake ${CMAKE_VERSION}")

project(rocsolver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

option(ROCSOLVER_EMBED_FMT "Hide libfmt symbols" ON)
option(OPTIMAL "Build specialized kernels for small matrix sizes" ON)
option(ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG "Skip module mode search for LAPACK" ON)
option(ROCSOLVER_USE_INTERNAL_BLAS "Use internal implementation of GEMM and TRSM for debugging." OFF)
option(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER "Use LAPACK like implementation of secular equations solver" ON)

# Add our CMake helper files to the lookup path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

# ########################################################################
# Main
# ########################################################################

# Get rocm-cmake
include(get-rocm-cmake)

# Include the rocm-cmake components we use
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
include(ROCMHeaderWrapper)

include(os-detection)
get_os_id(OS_ID)
message(STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set(VERSION_STRING "3.31.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Workaround until llvm and hip CMake modules fix symlink logic in their config files
list(APPEND CMAKE_PREFIX_PATH
  ${ROCM_PATH}
  ${ROCM_PATH}/llvm
  ${ROCM_PATH}/hip
  /opt/rocm
  /opt/rocm/llvm
  /opt/rocm/hip
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEFAULT_ARMOR_LEVEL 1)
else()
  set(DEFAULT_ARMOR_LEVEL 0)
endif()
set(ARMOR_LEVEL "${DEFAULT_ARMOR_LEVEL}" CACHE STRING "Enables increasingly expensive runtime correctness checks")
include(armor-config)

# This option only works for make, nmake and ninja, but no reason it shouldn't be on all the time
# It creates a compile_commands.json file for use with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# BUILD_SHARED_LIBS is a cmake built-in
# Make it an explicit option such that it shows in cmake-gui
option(BUILD_SHARED_LIBS "Build rocSOLVER as a shared library" ON)

# Include helper functions and wrapper functions
include(util)
include(CheckLanguage)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)

option(BUILD_TESTING "Build rocSOLVER tests" OFF)
if(BUILD_TESTING)
  enable_testing()
endif()

if(BUILD_SHARED_LIBS)
  set(BUILD_WITH_SPARSE_DEFAULT OFF)
else()
  set(BUILD_WITH_SPARSE_DEFAULT ON)
endif()

option(BUILD_LIBRARY "Build rocSOLVER library" ON)
option_opposite(BUILD_LIBRARY SKIP_LIBRARY)
option(BUILD_WITH_SPARSE "Build with rocsparse available at build time" "${BUILD_WITH_SPARSE_DEFAULT}")
option(BUILD_CLIENTS_TESTS "Build rocSOLVER test client" "${BUILD_TESTING}")
option(BUILD_CLIENTS_BENCHMARKS "Build rocSOLVER benchmark client" OFF)
option(BUILD_CLIENTS_SAMPLES "Build rocSOLVER samples" OFF)
cmake_dependent_option(BUILD_CLIENTS_EXTRA_TESTS "Build extra tests" OFF BUILD_TESTING OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_CODE_COVERAGE "Build rocSOLVER with code coverage enabled" OFF)
option(WERROR "Treat warnings as errors" OFF)
option(BUILD_COMPRESSED_DBG "Enable compressed debug symbols" ON)
check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
cmake_dependent_option(BUILD_OFFLOAD_COMPRESS "Build with offload compression" ON CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS OFF)

message(STATUS "Tests: ${BUILD_CLIENTS_TESTS}")
message(STATUS "Benchmarks: ${BUILD_CLIENTS_BENCHMARKS}")
message(STATUS "Samples: ${BUILD_CLIENTS_SAMPLES}")

check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)
if(USE_HIPCXX)
  enable_language(HIP)
else()
  if(NOT DEFINED AMDGPU_TARGETS)
    set(XNACK_PLUS_TARGETS
      gfx90a:xnack+
      gfx908:xnack+
      gfx942:xnack+
      gfx950:xnack+)
    set(XNACK_MINUS_TARGETS gfx90a:xnack-)
    set(MISC_TARGETS
      gfx942
      gfx950
      gfx10-1-generic
      gfx10-3-generic
      gfx1100
      gfx1101
      gfx1102
      gfx1151
      gfx12-generic
    )
    if(BUILD_ADDRESS_SANITIZER)
      set(OPTIONAL_TARGETS_QUERY ${XNACK_PLUS_TARGETS})
    else()
      set(OPTIONAL_TARGETS_QUERY ${XNACK_MINUS_TARGETS} ${MISC_TARGETS})
      set(DEFAULT_TARGETS
        gfx900
        gfx906:xnack-
        gfx908:xnack-
      )
    endif()
    # Query for compiler support of GPU archs
    rocm_check_target_ids(OPTIONAL_AMDGPU_TARGETS TARGETS ${OPTIONAL_TARGETS_QUERY})
    set(AMDGPU_TARGETS_INIT ${OPTIONAL_AMDGPU_TARGETS} ${DEFAULT_TARGETS})
  endif()

  # Set this before finding hip so that hip::device has the required arch flags
  # added as usage requirements on its interface
  set(AMDGPU_TARGETS "${AMDGPU_TARGETS_INIT}"
    CACHE STRING "List of specific machine types for library to target")
endif()

# Find HIP dependencies
find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)

find_package(rocblas REQUIRED CONFIG PATHS ${ROCM_PATH})
get_imported_target_location(location roc::rocblas)
message(STATUS "Found rocBLAS: ${location}")
set(rocblas_minimum 5.1)
rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}")
rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}")

if(BUILD_WITH_SPARSE)
  find_package(rocsparse REQUIRED CONFIG PATHS ${ROCM_PATH})
  get_imported_target_location(location roc::rocsparse)
  message(STATUS "Found rocSPARSE: ${location}")
  set(rocsparse_minimum 2.2)
  rocm_package_add_dependencies(SHARED_DEPENDS "rocsparse >= ${rocsparse_minimum}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocsparse-static-devel >= ${rocsparse_minimum}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocsparse-static-dev >= ${rocsparse_minimum}")
else()
  list(APPEND CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "rocsparse")
  list(APPEND CPACK_RPM_RUNTIME_PACKAGE_SUGGESTS "rocsparse")
endif()

find_package(rocprim REQUIRED CONFIG PATHS ${ROCM_PATH})
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocprim-static-devel")
rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocprim-static-dev")

add_subdirectory(common)

if(BUILD_LIBRARY)
  add_subdirectory(library)
endif()

if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SAMPLES)
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR  CLIENTS_OS STREQUAL "almalinux")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()
  rocm_package_setup_component(clients)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests DEPENDS DEB "${GFORTRAN_DEB}" RPM "${GFORTRAN_RPM}")
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks DEPENDS DEB "${GFORTRAN_DEB}" RPM "${GFORTRAN_RPM}")
  endif()
  add_subdirectory(clients)
endif()

if(OS_ID_sles)
  rocm_package_add_rpm_dependencies("libLLVM >= 7.0.1")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "BSD")

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
    set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}")
  endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
  "\${CPACK_PACKAGING_INSTALL_PREFIX}"
)

set(ROCSOLVER_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME rocsolver
  DESCRIPTION "AMD ROCm SOLVER library"
  MAINTAINER "RocSOLVER maintainer <rocsolver-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCSOLVER_CONFIG_DIR}
)

# Code Coverage Build Commands:
#   make coverage_cleanup (clean coverage related files)
#   make coverage GTEST_FILTER=<>
#   make coverage_analysis GTEST_FILTER=<> (analyze tests)
#   make coverage_output (generate html documentation)
if(BUILD_CODE_COVERAGE)
  # Run coverage analysis
  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/rocsolver-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_dependencies(coverage_analysis rocsolver)

  # Generate gcov-tool script
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
  )

  # Generate code coverage report
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml --ignore-errors source lcoverage/main_coverage.info --output-directory lcoverage
  )
  add_custom_target(coverage DEPENDS coverage_output)

  # Delete gcov data files
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
