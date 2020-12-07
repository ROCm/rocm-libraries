# ########################################################################
# Copyright 2019-2020 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

# Install prefix
if(WIN32)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/package CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Thrust project
project(rocthrust LANGUAGES CXX)

#Adding CMAKE_PREFIX_PATH
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Detect compiler support for target ID
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Use target ID syntax if supported for AMDGPU_TARGETS
if(TARGET_ID_SUPPORT)
  set(AMDGPU_TARGETS gfx900:xnack-;gfx906:xnack-;gfx908:xnack- CACHE STRING "List of specific machine types for library to target")
else()
  set(AMDGPU_TARGETS gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target")
endif()

# Verify that hcc or hipcc compiler is used on ROCM platform
include(cmake/VerifyCompiler.cmake)

# Build options
# Disable -Werror
option(DISABLE_WERROR "Disable building with Werror" ON)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(DOWNLOAD_ROCPRIM "Download rocPRIM and do not search for rocPRIM package" OFF)
set(RNG_SEED_COUNT 0 CACHE STRING "Number of true random sequences to test each input size for")
set(PRNG_SEEDS 1 CACHE STRING "Seeds of pseudo random sequences to test each input size for")

set(THRUST_HOST_SYSTEM_OPTIONS CPP OMP TBB)
set(THRUST_HOST_SYSTEM CPP CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_HOST_SYSTEM
  PROPERTY STRINGS ${THRUST_HOST_SYSTEM_OPTIONS}
)

if (NOT THRUST_HOST_SYSTEM IN_LIST THRUST_HOST_SYSTEM_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_HOST_SYSTEM must be one of ${THRUST_HOST_SYSTEM_OPTIONS}"
  )
endif ()

# Set CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get dependencies
include(cmake/Dependencies.cmake)

# Setup VERSION
rocm_setup_version(VERSION "2.10.7")

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# Thrust (with HIP backend)
add_subdirectory(thrust)

# Tests
if(BUILD_TEST)
  # We still want the testing to be compiled to catch some errors
  add_subdirectory(testing)
  enable_testing()
  add_subdirectory(test)
endif()

# Examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(internal/benchmark)
endif()

set(THRUST_OPTIONS_DEBUG ${THRUST_OPTIONS_WARNINGS})
set(THRUST_OPTIONS_RELEASE ${THRUST_OPTIONS_WARNINGS})

# Package
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocprim (>= 2.10.1)") # 1.5.18263 is HIP version in ROCm 1.8.2
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_REQUIRES "rocprim >= 2.10.1")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

rocm_create_package(
  NAME rocthrust
  DESCRIPTION "Radeon Open Compute Thrust library"
  MAINTAINER "rocthrust-maintainer@amd.com"
)
