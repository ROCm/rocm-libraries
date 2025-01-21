###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

cmake_minimum_required(VERSION 3.14)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# NOTE: This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." )
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE )

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif()
if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install path prefix for packages.")
  set(CPACK_SET_DESTDIR OFF)
endif()

# This helps cmake properly find hip-config.cmake
list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} ${ROCM_PATH} /opt/rocm )
# Append our library helper cmake path and the cmake path for hip (for convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# Build shared libs
set(BUILD_SHARED_LIBS ON)

project(hiptensor)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#Set clang - driver mode to g++
set(CLANG_DRIVER_MODE "--driver-mode=g++")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2") # clang++ crashes without -O2
set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 -DNDEBUG") # clang++ failed to build the project with the default -Os

# Project options
if( CMAKE_PROJECT_NAME STREQUAL "hiptensor" )
  option( HIPTENSOR_BUILD_TESTS "Build hiptensor tests" ON )
  option( HIPTENSOR_BUILD_SAMPLES "Build hiptensor samples" ON )
  option( HIPTENSOR_BUILD_COMPRESSED_DBG "Enable compressed debug symbols" ON)
  option( HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR "Set hiptensor default strides to column major" ON )
  option(BUILD_OFFLOAD_COMPRESS "Build hiptensor with offload compression" ON)
endif()

# Setup output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

# ROCm setup
find_package(ROCM 0.8 CONFIG REQUIRED)
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)

include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(CheckCXXCompilerFlag)

# check if ROCm supports `__hip_fp8_e5m2` and `__hip_fp8_e4m3`
include(cmake/Macros/CheckF8.cmake)
check_f8(F8_EXISTS)
if(F8_EXISTS)
    message( STATUS "Performing Test `__hip_fp8_e5m2` and `__hip_fp8_e4m3` - Success" )
else()
    message(FATAL_ERROR "The detected ROCm does not support data type `__hip_fp8_e5m2` or `__hip_fp8_e4m3`.")
endif()

# check if asan is enabled
if (NOT DEFINED ADDRESS_SANITIZER AND DEFINED ENV{ADDRESS_SANITIZER})
    set(ADDRESS_SANITIZER $ENV{ADDRESS_SANITIZER})
endif()
if (ADDRESS_SANITIZER OR CMAKE_CXX_FLAGS MATCHES "-fsanitize=address")
    set(ADDRESS_SANITIZER_ENABLED ON)
else()
    set(ADDRESS_SANITIZER_ENABLED OFF)
endif()

if (ADDRESS_SANITIZER_ENABLED)
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
        TARGETS "gfx90a:xnack+;gfx942:xnack+" )
else()
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
        TARGETS "gfx908;gfx90a;gfx942" )
endif()

# Check if offload compression is supported
include(CheckCXXCompilerFlag)
if (BUILD_OFFLOAD_COMPRESS)
  check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if (NOT CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    message(WARNING "BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option." )
    set(CMAKE_NO_BUILTIN_CHRPATH ON)
  else()
    set(CMAKE_NO_BUILTIN_CHRPATH OFF)
  endif()
endif()

# Variable GPU_TARGET must be a cached variable and must be specified before calling find_package(hip)
# This is because hip-config.cmake sets --offload-arch via GPU_TARGET cached variable __after__ setting
# default cached variable GPU_TARGET to "gfx900;gfx906;gfx908;gfx1100;gfx1101;gfx1102", where not all archs are compatible with MFMA instructions
#
# By rule, once cached variable is set, it cannot be overridden unless we use the FORCE option
if(GPU_TARGETS)
  set(GPU_TARGETS "${GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
elseif(AMDGPU_TARGETS)
  set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
  message(DEPRECATION "AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS.")
else()
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message(STATUS "GPU_TARGETS=${GPU_TARGETS}")

if(HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR)
  add_compile_definitions(HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=1)
else()
  add_compile_definitions(HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=0)
endif()
message(STATUS "HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=${HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR}")

# Setup HIP
find_package(hip REQUIRED )
math(EXPR hip_VERSION_FLAT "(${hip_VERSION_MAJOR} * 1000 + ${hip_VERSION_MINOR}) * 100000 + ${hip_VERSION_PATCH}")
message(STATUS "hip_version_flat=${hip_VERSION_FLAT}")

# No assumption that HIP kernels are launched with uniform block size for backward compatibility
# SWDEV-413293 and https://reviews.llvm.org/D155213
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 500723302)
  message(STATUS "Adding the fno-offload-uniform-block compiler flag")
  add_compile_options(-fno-offload-uniform-block)
endif()
# Add optimization flags needed by backend
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600140090)
  message(STATUS "Adding the enable-post-misched=0 compiler flag")
  add_compile_options("SHELL: -mllvm -enable-post-misched=0")
endif()
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
  message(STATUS "Adding -amdgpu-early-inline-all=true and -amdgpu-function-calls=false")
  add_compile_options("SHELL: -mllvm -amdgpu-early-inline-all=true")
  add_compile_options("SHELL: -mllvm -amdgpu-function-calls=false")
endif()

# Configure library build
add_subdirectory(library/src)

# Configure testing setup
if(HIPTENSOR_BUILD_TESTS OR HIPTENSOR_BUILD_SAMPLES)
  enable_testing()
  rocm_package_setup_component(clients)
endif()

# Configure tests build
if(HIPTENSOR_BUILD_TESTS)
  rocm_package_setup_component(tests PARENT clients)
  add_subdirectory(test)
endif()

# Configure clients build
if(HIPTENSOR_BUILD_SAMPLES)
  rocm_package_setup_component(samples PARENT clients)
  add_subdirectory(samples)
endif()

# Versioning via rocm-cmake
set ( VERSION_STRING "1.5.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# configure a header file to pass the CMake version settings to the source
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/library/include/hiptensor/internal/hiptensor-version.hpp.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/library/include/hiptensor/hiptensor-version.hpp" )

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfigVersion.cmake"
    VERSION "${VERSION_STRING}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
     "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfig.cmake"
     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
     NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

rocm_create_package(
    NAME hiptensor
    DESCRIPTION "AMD high-performance HIP library for tensor primitives"
    MAINTAINER "hiptensor Maintainer <hiptensor-maintainer@amd.com>"
    LDCONFIG
)
