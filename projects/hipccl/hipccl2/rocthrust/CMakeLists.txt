# Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

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
set(AMDGPU_TARGETS gfx803;gfx900;gfx906 CACHE STRING "List of specific machine types for library to target")

# Setup VERSION
set ( VERSION_STRING "2.6.0." )

# Check if BUILD_NUMBER is defined in a Jenkins envirnment
if ($ENV{BUILD_NUMBER} )
  string(CONCAT BUILD_VERSION ${VERSION_STRING} $ENV{BUILD_NUMBER})
else ()
  string(CONCAT BUILD_VERSION ${VERSION_STRING} "0")
endif ()
rocm_setup_version( VERSION ${BUILD_VERSION} NO_GIT_TAG_VERSION )

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
  MAINTAINER "Saad Rahim <saad.rahim@amd.com>"
)
