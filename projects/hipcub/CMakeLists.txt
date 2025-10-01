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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18...3.25)

# --------------------------------------
# Update these variables at release time
#
# Set the library version
set(VERSION_STRING "4.1.0")
# Set the minimum required rocPRIM version
set(MIN_ROCPRIM_PACKAGE_VERSION "4.1.0" CACHE STRING "Minimum version of rocPRIM to search for when ROCPRIM_FETCH_METHOD is set to PACKAGE.")
# Set download branch for dependency rocPRIM
set(ROCM_DEP_RELEASE_BRANCH "release/rocm-rel-7.1" CACHE STRING "Download branch for ROCm dependencies")
# --------------------------------------

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
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

include(CheckLanguage)
include(CMakeDependentOption)

# Build options
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(CODE_COVERAGE "Enable code coverage" OFF)
option(EXTERNAL_DEPS_FORCE_DOWNLOAD "Download non-ROCm dependencies and do not search for packages" OFF)
option(DOWNLOAD_CUB "Download CUB and thrust. Do not search for CUB package" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_EXAMPLE "Build Examples" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_OFFLOAD_COMPRESS "Build hipCUB with offload compression" ON)
option(BUILD_COMPUTE_SANITIZER "Build tests with cuda's compute sanitizer enabled" OFF)
cmake_dependent_option(USE_SYSTEM_LIB "Use installed hipCUB when building tests" OFF BUILD_TEST OFF)

# Check and test cuda compiler, defines 'CMAKE_HIP_COMPILER'
check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)

# Check and test cuda compiler, defines 'CMAKE_CUDA_COMPILER'
check_language(CUDA)

# Set the ROCM install directory.
if(WIN32)
  set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

# Set up options for obtaining dependency rocPRIM.
# PACKAGE: Search for an install package that contains the dependency.
# MONOREPO: Assume this is a monorepo checkout and search for the dependency in the directory at ../../projects/.
# DOWNLOAD: Download the dependency from the monorepo.
set(FETCH_METHOD_OPTIONS "PACKAGE" "MONOREPO" "DOWNLOAD")

set(ROCPRIM_FETCH_METHOD "PACKAGE" CACHE STRING "How to obtain the rocPRIM dependency")

# This function checks to see if the fetch method variable it's passed is defined, and contains a valid value.
# If it does not contain a valid value, it issues a fatal failure with an error message.
function(check_fetch_method method)
  if (DEFINED ${method} AND NOT ${${method}} IN_LIST FETCH_METHOD_OPTIONS)
    message(FATAL_ERROR "Unrecognized ${method}: \"${${method}}\". Valid options are: ${FETCH_METHOD_OPTIONS}.")
  endif()
endfunction()

check_fetch_method(ROCPRIM_FETCH_METHOD)

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

# Detect compiler through use of result from 'check_language(...)'
if(USE_HIPCXX)
  enable_language(HIP)
elseif(NOT (CMAKE_CXX_COMPILER MATCHES ".*nvcc$" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
  # Detected HIP through archaic match by checking passed CXX compiler. This can
  # be removed once we bump minimum CMake version to 3.21 or higher.

  # Setup GPU targets for rocm platform
  message(STATUS "CMake could not derive language via 'check_language'. Falling back to legacy compiler checks.")
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
        TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
      )
    else()
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
        TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"
      )
    endif()
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
elseif(CMAKE_CUDA_COMPILER)
  # We haven't detected HIP, so surely we must be on a CUDA-compatible compiler.
  # Contrary to HIP, 'enable_language(CUDA)' is supported from CMake 3.8 and higher.
  enable_language(CUDA)

  # Hack: let CMake think that our 'hip' files are actually CUDA files.
  set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS hip;cu)
endif()

# Compressed offload binaries are currently not working with the SPIR-V target
if("amdgcnspirv" IN_LIST GPU_TARGETS)
  if(BUILD_OFFLOAD_COMPRESS)
    message(FATAL_ERROR "Cannot combine SPIR-V and BUILD_OFFLOAD_COMPRESS")
  endif()
endif()

# Setup the library version
rocm_setup_version(VERSION ${VERSION_STRING})
math(EXPR hipcub_VERSION_NUMBER "${hipcub_VERSION_MAJOR} * 100000 + ${hipcub_VERSION_MINOR} * 100 + ${hipcub_VERSION_PATCH}")

# Find and verify HIP.
include(VerifyCompiler)

# Get dependencies (except rocm-cmake, included earlier)
include(Dependencies)

if(BUILD_ADDRESS_SANITIZER)
  add_compile_options(-fsanitize=address -shared-libasan)
  add_link_options(-fuse-ld=lld)
endif()

include(CheckCXXCompilerFlag)

if(BUILD_OFFLOAD_COMPRESS)
  # We need to pass '-x hip' since check_cxx_compiler_flag assumes c++ and not HIP. 
  check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-compress")
  else()
    message(STATUS "Warning: BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
  endif()
endif()

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
  if(USE_SYSTEM_LIB)
    # On ROCm hipCUB requires rocPRIM
    if(HIP_COMPILER STREQUAL "clang")
      find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/lib/cmake/rocprim")
      if (${rocprim_VERSION} VERSION_LESS ${MIN_ROCPRIM_PACKAGE_VERSION})
        message(WARNING "The installed rocprim version, ${rocprim_VERSION}, is less than the minimum required version ${MIN_ROCPRIM_PACKAGE_VERSION}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
      endif()
    endif()
    find_package(hipcub REQUIRED CONFIG PATHS "/opt/rocm/lib/cmake/hipcub")
    if (NOT ${hipcub_VERSION} VERSION_EQUAL ${VERSION_STRING})
      message(WARNING "The installed hipcub version, ${hipcub_VERSION}, does not match project version ${VERSION_STRING}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
    endif()
  endif()

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
