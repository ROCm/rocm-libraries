# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.22...3.25.2 FATAL_ERROR)

# Build documentation
option(BUILD_DOCS "Build documentation" OFF)

option(LEGACY_HIPBLAS_DIRECT "Use hipblas headers instead of hipblas-common." OFF )

# This will add compile option: -std=c++17
set(CMAKE_CXX_STANDARD 17 )
# Without this line, it will add -std=gnu++17 instead, which may have issues.
set(CMAKE_CXX_EXTENSIONS OFF )
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

if(NOT BUILD_CUDA)
# Adding CMAKE_PREFIX_PATH, needed for static builds
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )
endif()

if( NOT DEFINED ENV{HIP_PATH})
    set( HIP_PATH "/opt/rocm/hip" )
else( )
    set (HIP_PATH $ENV{HIP_PATH} )
endif( )

if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
endif()

if (NOT DEFINED ENV{CXX} AND NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++")
endif()

if (NOT DEFINED ENV{CC} AND NOT CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER "${rocm_bin}/amdclang")
endif()

# TODO: move FC and CXX and CC compiler vars above to new toolchain-linux.cmake (Fortran for clients)
if (NOT DEFINED ENV{FC} AND NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER "gfortran")
endif()

# hipBLASLt project
project(hipblaslt LANGUAGES CXX)

# Dependencies
include(cmake/Dependencies.cmake)

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_PATH}/cmake )

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

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)


# Build options
option(BUILD_SHARED_LIBS "Build hipBLASLt as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build hipBLASLt with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build hipBLASLt with address sanitizer enabled" OFF)
# Find CUDA if the user wants a CUDA version.
option(USE_CUDA "Look for CUDA and use that as a backend if found" OFF)
# For lazy library loading
option(Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON)
option(Tensile_SEPARATE_ARCHITECTURES "Tensile to use GPU architecture specific files?" ON)
option(Tensile_NO_LAZY_LIBRARY_LOADING "Diasble loading kernels on demand?" OFF)
# For roctx
include(CMakeDependentOption)
cmake_dependent_option(HIPBLASLT_ENABLE_MARKER "Enable roctx marker in hipBLASLt" ON "BUILD_SHARED_LIBS" OFF)
# For rocRoller
option(USE_ROCROLLER "Build with the rocRoller library" ON)

if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage -lgcov)
endif()

if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
    add_link_options(-fuse-ld=lld)
endif()

if(NOT BUILD_CUDA)
    # Determine if CXX Compiler is amdclang
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Using amdclang to build for amdgpu backend")
        if( CMAKE_CXX_COMPILER MATCHES ".*hipcc.*" )
          message( STATUS "WARNING: hipcc compiler use is deprecated. Use amdclang++ directly." )
        endif()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )
        if (CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT WIN32)
            set (CMAKE_CXX_FLAGS_DEBUG "-O0 ${CMAKE_CXX_FLAGS_DEBUG} -gsplit-dwarf -ggdb" )
        endif()
        if (CMAKE_BUILD_TYPE MATCHES "Release" AND NOT WIN32)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" )
        endif()
    else()
        message(FATAL_ERROR "'amdclang' compiler required to compile for ROCm software.")
    endif()
endif()

if (LINK_BLIS)
  include(cmake/findBLIS.cmake)
endif()

if (USE_CUDA)
	find_package( CUDA REQUIRED )
    # Hip headers required of all clients; clients use hip to allocate device memory
    find_package( HIP MODULE REQUIRED )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
else()

    if (NOT BUILD_ADDRESS_SANITIZER)
      #Set the AMDGPU_TARGETS with backward compatiblity
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
          TARGETS "gfx908:xnack+;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx942;gfx950;gfx1100;gfx1101;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
      )
    else()
      #build xnack-supported targets only
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
          TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
      )
    endif()

    if (AMDGPU_TARGETS)
        set(TMPAMDGPU_TARGETS "${AMDGPU_TARGETS}")
        if(TMPAMDGPU_TARGETS STREQUAL "all" )
            set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target" FORCE)
        else()
            set(AMDGPU_TARGETS "${TMPAMDGPU_TARGETS}" CACHE STRING "AMD GPU targets to compile for" FORCE)
        endif()
    else()
        set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
    endif()

    message(STATUS "AMDGPU_TARGETS: ${AMDGPU_TARGETS}")

    if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
        find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
    endif( )

    option( Tensile_SKIP_BUILD "Skip building Tensile" OFF )

    # we will have expanded "all" for tensile to ensure consistency as we have local rules
    set( Tensile_ARCHITECTURE "${AMDGPU_TARGETS}" CACHE STRING "Tensile to use which architecture?" FORCE)

    set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")
    set( Tensile_CODE_OBJECT_VERSION "4" CACHE STRING "Tensile code_object_version")
    set( Tensile_COMPILER "amdclang++" CACHE STRING "Tensile compiler")
    set( Tensile_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format")
    set( Tensile_CPU_THREADS "" CACHE STRING "Number of threads for Tensile parallel build")

    option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
    option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
    option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

    if (NOT Tensile_BUILD_ID)
      set(Tensile_BUILD_ID "sha1" CACHE STRING "Build ID Kind for Tensile" FORCE)
    endif()

    set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )

    set_property( CACHE Tensile_LOGIC PROPERTY STRINGS aldebaran asm_full asm_lite asm_miopen hip_lite other )
    set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS default V4 V5 )
    set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc hipcc)
    set_property( CACHE Tensile_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)

    if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
      option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
      option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." OFF)
    else()
      option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." OFF)
      option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
    endif()

    if (WIN32)
      set( Tensile_ROOT "${CMAKE_BINARY_DIR}/virtualenv/Lib/site-packages/Tensile" )
    endif()

    include(virtualenv)
    if (Tensile_TEST_LOCAL_PATH)
      virtualenv_install(${Tensile_TEST_LOCAL_PATH})
      message (STATUS "using local Tensile from ${Tensile_TEST_LOCAL_PATH}, copied to ${Tensile_ROOT}")
    else()
      virtualenv_install(${CMAKE_SOURCE_DIR}/tensilelite)
    endif()
    message(STATUS "Adding ${VIRTUALENV_HOME_DIR} to CMAKE_PREFIX_PATH")
    list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})
    if (TENSILE_VERSION)
      find_package(Tensile ${TENSILE_VERSION} EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
    else()
      find_package(Tensile 4.33.0 EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
    endif()

    # setup hipblaslt defines used for both the library and clients
    if(Tensile_SEPARATE_ARCHITECTURES)
      list(APPEND TENSILE_DEFINES ROCBLAS_TENSILE_SEPARATE_ARCH=1)
    else()
      list(APPEND TENSILE_DEFINES ROCBLAS_TENSILE_SEPARATE_ARCH=0)
    endif()
    if(Tensile_NO_LAZY_LIBRARY_LOADING)
      list(APPEND TENSILE_DEFINES ROCBLASLT_TENSILE_LAZY_LOAD=0)
    else()
      list(APPEND TENSILE_DEFINES ROCBLASLT_TENSILE_LAZY_LOAD=1)
    endif()
endif()

cmake_host_system_information(RESULT OS_PLATFORM QUERY DISTRIB_ID)
cmake_host_system_information(RESULT OS_INFO QUERY DISTRIB_INFO)

if( ${OS_PLATFORM} STREQUAL "rhel")
  if( ${OS_INFO_VERSION_ID} VERSION_LESS "9.5")
    message(WARNING "RHEL version ${OS_INFO_VERSION_ID} too old, not building RocRoller")
    set(USE_ROCROLLER OFF)
  endif()
endif()

if( LEGACY_HIPBLAS_DIRECT )
  find_package( hipblas REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)
else()
  find_package( hipblas-common REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)
endif()

if(HIPBLASLT_ENABLE_MARKER)
  find_library(rocTracer roctx64)
  if(NOT rocTracer)
    message(FATAL_ERROR "roctracer not found, but HIPBLASLT_ENABLE_MARKER is enabled")
  endif()
  add_definitions(-DHIPBLASLT_ENABLE_MARKER)
  rocm_package_add_dependencies(DEPENDS "roctracer >= 1.0.0")
endif()

# Setup version
set(VERSION_STRING "0.14.0")
rocm_setup_version(VERSION ${VERSION_STRING})
set(hipblaslt_SOVERSION 0.14)


if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  set( BUILD_CLIENTS ON )
endif()

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include
  )
endif()

# hipBLASLt library
add_subdirectory( library )

# Trigger client builds if selected
# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()

  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(BUILD_CLIENTS_SAMPLES)
    rocm_package_setup_client_component(samples)
  endif()
  add_subdirectory( clients )
endif( )

if( NOT BUILD_CUDA )
  if( LEGACY_HIPBLAS_DIRECT )
    rocm_package_add_dependencies(DEPENDS "hipblas >= 0.50.0")
  else()
    set(hipblas_common_minimum 1.0.0)
    if(BUILD_SHARED_LIBS)
    rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-dev >= ${hipblas_common_minimum}")
    rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-devel >= ${hipblas_common_minimum}")
    else()
      rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-dev >= ${hipblas_common_minimum}")
      rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-devel >= ${hipblas_common_minimum}")
    endif()
  endif()
endif( )

# Build docs
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Package specific CPACK vars
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT")

if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
endif()

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

# work around code object stripping failure if using /usr/bin/strip
set( CPACK_RPM_SPEC_MORE_DEFINE "%define __strip ${rocm_bin}/../llvm/bin/llvm-strip")

# Give hipblaslt compiled for CUDA backend a different name
if( NOT USE_CUDA )
    set( package_name hipblaslt )
else( )
    set( package_name hipblaslt-alt )
endif( )

set( HIPBLASLT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "HIP library for GEMM operations with extended functionality"
    MAINTAINER "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLASLT_CONFIG_DIR}
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
  set(coverage_test ./clients/staging/hipblaslt-test)

  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/hipblaslt-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hipblaslt)

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
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
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
