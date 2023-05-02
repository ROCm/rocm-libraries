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

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(VERSION 3.16...3.21)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

#Adding CMAKE_PREFIX_PATH
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm ${ROCM_PATH})

# hipCUB project
project(hipcub LANGUAGES CXX)

#Adding CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# rocm-cmake contains common cmake code for rocm projects to help
# setup and install
include(cmake/RocmCmakeDependence.cmake)

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMHeaderWrapper )
include( ROCMCheckTargetIds )
include( ROCMClients )

# Setup GPU targets for rocm platform
if(NOT (CMAKE_CXX_COMPILER MATCHES ".*nvcc$" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
  if (NOT DEFINED AMDGPU_TARGETS)
      set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
  else()
    set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
  endif()
  set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

  if(GPU_TARGETS STREQUAL "all")
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030;gfx1100;gfx1101;gfx1102"
    )
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
endif()

# Verify that hip-clang is used on ROCM platform
if (NOT WIN32)
  include(cmake/VerifyCompiler.cmake)
endif()

# Set CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(DOWNLOAD_ROCPRIM "Download rocPRIM and do not search for rocPRIM package" OFF)
option(DOWNLOAD_CUB "Download CUB and thrust. Do not search for CUB package" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build Examples" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
#Set the header wrapper ON by default. 
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" ON)

if(BUILD_ADDRESS_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# Get dependencies
include(cmake/Dependencies.cmake)

# Setup VERSION
set(VERSION_STRING "2.13.1")
rocm_setup_version(VERSION ${VERSION_STRING})

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# hipCUB library
add_subdirectory(hipcub)

if(BUILD_TEST OR (BUILD_BENCHMARK AND NOT ONLY_INSTALL))
  rocm_package_setup_component(clients)
endif()

# Tests
if(BUILD_TEST)
  enable_testing()
  rocm_package_setup_client_component(tests)
  add_subdirectory(test)
endif()

# Examples
if(BUILD_EXAMPLE)
  add_subdirectory(examples)
endif()

# Benchmarks
if(BUILD_BENCHMARK AND NOT ONLY_INSTALL)
  rocm_package_setup_client_component(benchmarks)
  add_subdirectory(benchmark)
endif()

# Create header wrapper for backward compatibility
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_dir(
            ${PROJECT_SOURCE_DIR}/hipcub/include/hipcub/
	    PATTERNS "*.h"
	    PATTERN "*.hpp"
	    GUARDS SYMLINK WRAPPER
	    WRAPPER_LOCATIONS cub/${CMAKE_INSTALL_INCLUDEDIR}/hipcub/
            OUTPUT_LOCATIONS cub/wrapper/include/hipcub/
	  )
endif()

# Package
if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
  rocm_package_add_deb_dependencies(DEPENDS "rocprim-dev >= 2.10.1")
  rocm_package_add_rpm_dependencies(DEPENDS "rocprim-devel >= 2.10.1")
  set(CPACK_DEBIAN_PACKAGE_REPLACES "cub-hip")
  set(CPACK_RPM_PACKAGE_OBSOLETES "cub-hip")
else()
  rocm_package_add_dependencies(DEPENDS "hip-dev >= 4.4")
endif()
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RPM_PACKAGE_LICENSE "BSD")
# if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
#   set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
# endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

if(HIP_COMPILER STREQUAL "clang")
  rocm_create_package(
    NAME hipcub
    DESCRIPTION "hipCUB (rocPRIM backend)"
    MAINTAINER "hipcub-maintainer@amd.com"
    HEADER_ONLY
  )
else()
  rocm_create_package(
    NAME hipcub_nvcc
    DESCRIPTION "hipCUB (CUB backend)"
    MAINTAINER "hipcub-maintainer@amd.com"
    HEADER_ONLY
  )
endif()
