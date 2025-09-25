# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Build options
include(CMakeDependentOption)

option(BUILD_FORTRAN_WRAPPER "Build Fortran wrapper" OFF)
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
cmake_dependent_option(BUILD_BENCHMARK_TUNING
"Build extra benchmarks for kernel configuration tuning" OFF "BUILD_BENCHMARK" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(CODE_COVERAGE "Build with code coverage flags (clang only)" OFF)
option(DEPENDENCIES_FORCE_DOWNLOAD "Don't search the system for dependencies, always download them" OFF)
cmake_dependent_option(RUN_SLOW_TESTS "Run extra tests with CTest. These cover niche functionality and take long time" OFF "BUILD_TEST" OFF)
cmake_dependent_option(USE_SYSTEM_LIB "Use existing system ROCm library installation when building tests" OFF BUILD_TEST OFF)

if (NOT DEFINED ENV{ROCM_PATH})
#Path to ROCm installation
  set(ENV{ROCM_PATH} "/opt/rocm")
endif()
  
# Install prefix
set(CMAKE_INSTALL_PREFIX "$ENV{ROCM_PATH}" CACHE PATH "Install path prefix, prepended onto install directories")

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path")
    set(INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{ROCM_PATH}" CACHE PATH "Install path prefix, prepended onto install directories")
    #Adding CMAKE_PREFIX_PATH
    if(NOT DEFINED CMAKE_PREFIX_PATH)
      list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/llvm $ENV{ROCM_PATH})
    endif()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH
${CMAKE_CURRENT_SOURCE_DIR}/cmake
${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
$ENV{ROCM_PATH}/lib/cmake/hip
${HIP_PATH}/cmake $ENV{ROCM_PATH}/hip/cmake # FindHIP.cmake
$ENV{ROCM_PATH}/llvm
)

#
# rocRAND project
#
project(rocRAND CXX)
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Set the default value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared")

include(cmake/HaveASMIncbin.cmake)

rocrand_check_have_asm_incbin(ROCRAND_HAVE_ASM_INCBIN)

# Get dependencies
include(cmake/Dependencies.cmake)

# Detect compiler support for target ID
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Verify that hipcc compiler is used on ROCM platform
# TODO: Fix VerifyCompiler for Windows
# This will run for NVIDIA too, which is unnecessary, but not harmful, since we use the CUDA cmake
# language in that case which will ignore GPU_TARGETS.
# The reason for this is that this has to happen before find_package(hip), because we're providing
# our own defaults, but because of that we have no reasonable way to detect the platform we're
# targeting yet.
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
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1151"
    )
  endif()

  set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
endif()

# Verify the compiler settings and include the HIP CMake package
include(cmake/VerifyCompiler.cmake)

# Build option to disable -Werror
option(DISABLE_WERROR "Disable building with Werror" ON)

# Build CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(DISABLE_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()
if(CODE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    add_definitions(-DCODE_COVERAGE_ENABLED)
endif()

if (CMAKE_CXX_STANDARD EQUAL 14)
  message(WARNING "C++14 will be deprecated in the next major release")
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# HIP on Windows: xhip is required with clang++ to get __half defined
if (WIN32)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-xhip>")
endif()

cmake_dependent_option(BUILD_LEGACY_BENCHMARK "Build the legacy benchmarks" ON "BUILD_BENCHMARK" OFF)
if(BUILD_LEGACY_BENCHMARK)
  message(DEPRECATION "Building the legacy benchmarks will be removed in a future release. Consider using the new benchmark suite.")
endif()
if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# Set version variables
rocm_setup_version( VERSION "4.1.0" )
set ( rocrand_VERSION ${rocRAND_VERSION} )
# Old-style version number used within the library's API. rocrand_get_version should be modified.
math(EXPR rocrand_VERSION_NUMBER "${rocRAND_VERSION_MAJOR} * 100000 + ${rocRAND_VERSION_MINOR} * 100 + ${rocRAND_VERSION_PATCH}")
set(rocrand_SOVERSION 1.1)

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# The "tools" and "library" targets may be skipped under certain conditions when USE_SYSTEM_LIB=ON
if ((NOT BUILD_BENCHMARK) AND (NOT BUILD_FORTRAN_WRAPPER) AND BUILD_TEST AND USE_SYSTEM_LIB)
  message(STATUS "Skipping the 'tools' and 'library' targets as only BUILD_TEST=ON and USE_SYSTEM_LIB=ON were specified")
else()
  # Tools
  add_subdirectory(tools)

  # rocRAND library, adds library target "rocrand" and includes Fortran wrapper
  add_subdirectory(library)
endif()

if(BUILD_TEST OR BUILD_BENCHMARK)
  rocm_package_setup_component(clients)
endif()

# Tests
if (BUILD_TEST)
    if(USE_SYSTEM_LIB)
        find_package(rocrand REQUIRED CONFIG PATHS "$ENV{ROCM_PATH}/rocrand")
        if (NOT ${rocrand_VERSION} VERSION_EQUAL ${rocRAND_VERSION})
            message(WARNING "The installed rocrand version, ${rocrand_VERSION}, does not match project version ${rocRAND_VERSION}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
        endif()
    endif()

    enable_testing()
    rocm_package_setup_client_component(tests)
    add_subdirectory(test)
endif()

# Benchmarks
if (BUILD_BENCHMARK)
    rocm_package_setup_client_component(benchmarks)
    add_subdirectory(benchmark)
endif()

# Package (make package)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if( HIP_RUNTIME_LOWER STREQUAL "rocclr" )
    if(BUILD_ADDRESS_SANITIZER)
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
    else()
        set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
    endif()
    rocm_package_add_dependencies("${DEPENDS_HIP_RUNTIME} >= 4.5.0")
elseif( HIP_RUNTIME STREQUAL "cuda" )
    rocm_package_add_dependencies("hip-nvcc >= 3.5.0")
endif( )

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

if(HIP_COMPILER STREQUAL "nvcc")
    set(package_name rocrand-alt)
else()
    set(package_name rocrand)
endif()

set(ROCRAND_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "rocRAND is a ROCm random number generation library"
    MAINTAINER "rocRAND Maintainer <hiprand-maintainer@amd.com>"
    LDCONFIG
    # LDCONFIG_DIR ${ROCRAND_CONFIG_DIR}
)
