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

# hipCUB project
project(hipcub LANGUAGES CXX)

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

# Set HIP flags
set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

include(CheckLanguage)
include(CMakeDependentOption)

# Build options
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(CODE_COVERAGE "Enable code coverage" OFF)
option(DEPENDENCIES_FORCE_DOWNLOAD "Download dependencies and do not search for packages" OFF)
option(DOWNLOAD_CUB "Download CUB and thrust. Do not search for CUB package" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build Examples" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)

check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)

# Set the ROCM install directory.
if(WIN32)
  set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()


# Set the header wrapper OFF by default.
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)

# Add hipCUB's CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CODE_COVERAGE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# rocm-cmake has to be included early so that it's available to set GPU_TARGETS
# If hip is included prior to setting that then it defaults to building only for the current architecture
include(ROCmCMakeBuildToolsDependency)

# Setup GPU targets for rocm platform
if(USE_HIPCXX)
  enable_language(HIP)
else()
  # Setup GPU targets for rocm platform
  if(NOT (CMAKE_CXX_COMPILER MATCHES ".*nvcc$" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
    if(NOT DEFINED AMDGPU_TARGETS)
      set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
    else()
      set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
    endif()
    set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

    if(GPU_TARGETS STREQUAL "all")
      if(BUILD_ADDRESS_SANITIZER)
        # ASAN builds require xnack
        rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
          TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+"
        )
      else()
        rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
          TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"
        )
      endif()
      set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
    endif()
  endif()
endif()

# Find and verify HIP.
include(VerifyCompiler)

# Get dependencies (except rocm-cmake, included earlier)
include(Dependencies)

if(BUILD_ADDRESS_SANITIZER)
  add_compile_options(-fsanitize=address -shared-libasan)
  add_link_options(-fuse-ld=lld)
endif()

# Setup VERSION
set(VERSION_STRING "3.5.0")
rocm_setup_version(VERSION ${VERSION_STRING})
math(EXPR hipcub_VERSION_NUMBER "${hipcub_VERSION_MAJOR} * 100000 + ${hipcub_VERSION_MINOR} * 100 + ${hipcub_VERSION_PATCH}")

# hipCUB library
add_subdirectory(hipcub)

if(CODE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fprofile-instr-generate -fcoverage-mapping")
endif()

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
if(HIP_COMPILER STREQUAL "clang")
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

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()
