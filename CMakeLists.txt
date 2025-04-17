# ########################################################################
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required( VERSION 3.16.8 )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option(BUILD_SHARED_LIBS "Build hipSparseLt as a shared library" ON )
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
# Find CUDA if the user wants a CUDA version.
option(BUILD_CUDA "Look for CUDA and use that as a backend if found" OFF)
include(CMakeDependentOption)
cmake_dependent_option(HIPSPARSELT_ENABLE_MARKER "Enable roctx marker in hipSPARSELt" ON "BUILD_SHARED_LIBS" OFF)

# IF cuda backend disable clients tests.
if(BUILD_CUDA)
    set(BUILD_CLIENTS_TESTS OFF)
endif()

# This will add compile option: -std=c++17
set(CMAKE_CXX_STANDARD 17 )
# Without this line, it will add -std=gnu++17 instead, which may have issues.
set(CMAKE_CXX_EXTENSIONS OFF )
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

if(NOT BUILD_CUDA)
# Adding CMAKE_PREFIX_PATH, needed for static builds
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip /opt/rocm/llvm /opt/rocm /opt/rocm/hip )
endif()

if( NOT DEFINED ENV{HIP_PATH})
    set( HIP_PATH "/opt/rocm/hip" )
else( )
    set (HIP_PATH $ENV{HIP_PATH} )
endif( )

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

project(hipsparselt LANGUAGES CXX)

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
    add_link_options(-fuse-ld=lld)
endif()

if(NOT BUILD_CUDA)
    # Determine if CXX Compiler is hip-clang
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Using amdclang to build for amdgpu backend")
        if( CMAKE_CXX_COMPILER MATCHES ".*hipcc.*" )
          message( STATUS "WARNING: hipcc compiler use is deprecated. Use amdclang++ directly." )
        endif()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )
        if (CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT WIN32)
            set (CMAKE_CXX_FLAGS_DEBUG "-O1 ${CMAKE_CXX_FLAGS_DEBUG} -gsplit-dwarf -ggdb" )
        endif()
        if (CMAKE_BUILD_TYPE MATCHES "Release" AND NOT WIN32)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" )
        endif()
    else()
        message(FATAL_ERROR "'amdclang' compiler required to compile for ROCm software.")
    endif()
endif()
# Dependencies
include(cmake/Dependencies.cmake)

if (BUILD_CUDA)
    set (HIP_ROOT_DIR ${HIP_PATH})
    find_package( CUDA REQUIRED )
    # Hip headers required of all clients; clients use hip to allocate device memory
    find_package( HIP MODULE REQUIRED )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )

else()
    if (BUILD_ADDRESS_SANITIZER)
      #Set the AMDGPU_TARGETS for ASAN build
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
          TARGETS "gfx942:xnack+;gfx950:xnack+;"
      )
    else()
      #Set the AMDGPU_TARGETS with backward compatiblity
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
          TARGETS "gfx942;gfx950;"
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

    option( BUILD_WITH_TENSILE "Build full functionality which requires tensile?" ON )

    if( BUILD_WITH_TENSILE )
      # we will have expanded "all" for tensile to ensure consistency as we have local rules
      set( Tensile_ARCHITECTURE "${AMDGPU_TARGETS}" CACHE STRING "Tensile to use which architecture?" FORCE)

      set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")

      if ((Tensile_CODE_OBJECT_VERSION MATCHES "default") OR (Tensile_CODE_OBJECT_VERSION MATCHES "V4"))
        set( Tensile_CODE_OBJECT_VERSION "4" STRING "Tensile code_object_version")
      elseif(Tensile_CODE_OBJECT_VERSION MATCHES "V5")
        set( Tensile_CODE_OBJECT_VERSION "5" STRING "Tensile code_object_version")
      else()
        set( Tensile_CODE_OBJECT_VERSION "4" CACHE STRING "Tensile code_object_version")
      endif()
      set( Tensile_COMPILER "amdclang++" CACHE STRING "Tensile compiler")
      set( Tensile_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format")
      set( Tensile_CPU_THREADS "" CACHE STRING "Number of threads for Tensile parallel build")

      option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
      option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
      option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

      set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )

      set_property( CACHE Tensile_LOGIC PROPERTY STRINGS aldebaran asm_full asm_lite asm_miopen hip_lite other )
      set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS default V4 V5 )
      set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc amdclang++)
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

      if (NOT Tensile_BUILD_ID)
        set(Tensile_BUILD_ID "sha1" CACHE STRING "Build ID Kind for Tensile" FORCE)
      endif()

      include(virtualenv)
      if (Tensile_TEST_LOCAL_PATH)
        virtualenv_install(${Tensile_TEST_LOCAL_PATH})
        message (STATUS "using local Tensile from ${Tensile_TEST_LOCAL_PATH}, copied to ${Tensile_ROOT}")
      else()
        # Use the virtual-env setup and download package from specified repot:
        set( tensile_fork "ROCm" CACHE STRING "Tensile fork to use" )
        file (STRINGS "tensilelite_tag.txt" read_tensile_tag)
        set( tensile_tag ${read_tensile_tag} CACHE STRING "Tensile tag to download" )
        virtualenv_install("git+https://github.com/${tensile_fork}/hipBLASLt.git@${tensile_tag}#subdirectory=tensilelite")

        message (STATUS "using GIT Tensile fork=${tensile_fork} from branch=${tensile_tag}")
      endif()
      message(STATUS "Adding ${VIRTUALENV_HOME_DIR} to CMAKE_PREFIX_PATH")
      list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})
      if (TENSILE_VERSION)
        find_package(Tensile ${TENSILE_VERSION} EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
      else()
        find_package(Tensile 4.33.0 EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
      endif()
    endif()
endif()

find_package( hipsparse REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

# rocm-cmake helpers
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds )
include( ROCMHeaderWrapper )
include( ROCMClients )


include (os-detection)
get_os_id(OS_ID)
message (STATUS "OS detected is ${OS_ID}")

if(HIPSPARSELT_ENABLE_MARKER)
  find_library(rocTracer roctx64 PATHS ${ROCM_PATH}/lib)
  if(NOT rocTracer)
    message(FATAL_ERROR "roctracer not found, but HIPSPARSELT_ENABLE_MARKER is enabled")
  endif()
  add_definitions(-DHIPSPARSELT_ENABLE_MARKER)
  rocm_package_add_dependencies(DEPENDS "roctracer >= 1.0.0")
endif()

# Setup version
set (VERSION_STRING "0.2.4" )
rocm_setup_version( VERSION ${VERSION_STRING} )
set(hipsparselt_SOVERSION 0.2)

# setup rocsparselt defines used for both the library and clients
if( BUILD_WITH_TENSILE )
    list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE=1)
    if(Tensile_NO_LAZY_LIBRARY_LOADING)
      list(APPEND TENSILE_DEFINES ROCSPARSELT_TENSILE_LAZY_LOAD=0)
    else()
      list(APPEND TENSILE_DEFINES ROCSPARSELT_TENSILE_LAZY_LOAD=1)
    endif()
else()
    list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE=0)
endif()

if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
  set(OPENMP_RPM "libgomp")
  set(OPENMP_DEB "libomp-dev")
  set(GFORTRAN_RPM "libgfortran")
  set(GFORTRAN_DEB "libgfortran5")
  if(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()
  elseif(CLIENTS_OS STREQUAL "sles")
    set(OPENMP_RPM "libgomp1")
  endif()

  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component( clients-common )
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()
  if(BUILD_CLIENTS_SAMPLES)
    rocm_package_setup_client_component(samples)
  endif()
endif()

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" ON)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include
  )
endif()
add_subdirectory( library )

# Trigger client builds if selected
if(BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
endif( )

# Package specific CPACK vars
if( NOT BUILD_CUDA )
  rocm_package_add_dependencies(DEPENDS "hipsparse >= 2.1.0")
endif( )

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT")

if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
  endif()
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

# Give hipsparselt compiled for CUDA backend a different name
if( NOT BUILD_CUDA )
    set( package_name hipsparselt )
else( )
    set( package_name hipsparselt-alt )
endif( )

set( HIPSPARSELTS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm Structured Sparsity Matrix Multiplication marshalling library"
    MAINTAINER "hipSPARSELt Maintainer <hipsparselt-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPSPARSELTS_CONFIG_DIR}
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
  set(coverage_test ./clients/staging/hipsparselt-test)

  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hipsparselt)

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
    COMMAND lcov --directory . --base-directory ${CMAKE_BINARY_DIR} --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" "'*/Tensile/Source/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage --prefix ${CMAKE_BINARY_DIR}
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

