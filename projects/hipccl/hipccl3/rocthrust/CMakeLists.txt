# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
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

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Verify that hcc compiler is used on ROCM platform
include(cmake/VerifyCompiler.cmake)

# Set CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-function")

# Build options
option(BUILD_TEST "Build tests" ON)

# Get dependencies
include(cmake/Dependencies.cmake)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target")

# Setup VERSION
rocm_setup_version(VERSION "2.8.0")

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# Thrust (with HIP backend)
add_subdirectory(thrust)

# Tests
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(testing)
endif()

# Package
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.18263), rocprim (>= 2.5)") # 1.5.18263 is HIP version in ROCm 1.8.2
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.18263, rocprim >= 2.5")

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
