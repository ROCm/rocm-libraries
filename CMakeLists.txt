###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

cmake_minimum_required( VERSION 3.14 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()
message( VERBOSE "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set( CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE )

### Project ROCWMMA
project( rocwmma LANGUAGES CXX )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#Set Clang C++ flags.
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2") # clang++ crashes without -O2
set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 -DNDEBUG") # clang++ failed to build the project with the default -Os
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --driver-mode=g++ -Xclang -fallow-half-arguments-and-returns -D__HIP_HCC_COMPAT_MODE__=1 -Wno-format-nonliteral -parallel-jobs=4 -fclang-abi-compat=17")

# Top level configs
if( CMAKE_PROJECT_NAME STREQUAL "rocwmma" )
  option( ROCWMMA_BUILD_TESTS "Build rocWMMA tests" ON )
  option( ROCWMMA_BUILD_SAMPLES "Build rocWMMA samples" ON )
  option( ROCWMMA_BUILD_ASSEMBLY "Output assembly files" OFF )
  option( BUILD_OFFLOAD_COMPRESS "Build rocWMMA with offload compression" ON )
endif()

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif()
if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install path prefix for packages.")
  set(CPACK_SET_DESTDIR OFF)
endif()
set(BUILD_SHARED_LIBS ON)

# This helps cmake properly find hip-config.cmake
list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} ${ROCM_PATH} /opt/rocm )
# Append our library helper cmake path and the cmake path for hip (for convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

find_package(ROCmCMakeBuildTools 0.7 CONFIG REQUIRED)
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)
include(ROCMClients)

# Versioning via rocm-cmake
set ( VERSION_STRING "1.7.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

set(MINIMUM_REQUIRED_ROCM_VERSION 6.3.0)
if(NOT DEFINED ROCM_PLATFORM_VERSION OR ${ROCM_PLATFORM_VERSION} VERSION_LESS ${MINIMUM_REQUIRED_ROCM_VERSION})
  message(FATAL_ERROR "${PROJECT_NAME} ${VERSION_STRING} and later versions require at least ROCm version ${MINIMUM_REQUIRED_ROCM_VERSION} for support.\nThe detected ROCm version is ${ROCM_PLATFORM_VERSION}.")
endif()

# configure a header file to pass the CMake version settings to the source
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/library/include/rocwmma/internal/rocwmma-version.hpp.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/library/include/rocwmma/rocwmma-version.hpp" )

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
    #TODO: Remove next line when rocm-cmake fix is available
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
        TARGETS "gfx90a:xnack+;gfx942:xnack+" )
else()
    rocm_check_target_ids(DEFAULT_GPU_TARGETS
        TARGETS "gfx908;gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" )
endif()

# Check if offload compression is supported
include(CheckCXXCompilerFlag)
if (BUILD_OFFLOAD_COMPRESS)
  check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
endif()

# TODO: Remove next line when rocm-cmake fix is available
# Currently fixes linking issues with large executables
set(CMAKE_NO_BUILTIN_CHRPATH ON)

# Variable GPU_TARGET must be a cached variable and must be specified before calling find_package(hip)
# This is because hip-config.cmake sets --offload-arch via GPU_TARGET cached variable __after__ setting
# default cached variable GPU_TARGET to DEFAULT_GPU_TARGETS, where not all archs are compatible with MFMA instructions
#
# By rule, once cached variable is set, it cannot be overridden unless we use the FORCE option
if(GPU_TARGETS)
  set(GPU_TARGETS "${GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
elseif(AMDGPU_TARGETS)
  set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
  message(STATUS "WARNING: AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS.")
else()
  set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message( VERBOSE "GPU_TARGETS=${GPU_TARGETS}")

find_package( hip REQUIRED )
find_package( hiprtc REQUIRED )
find_package( OpenMP REQUIRED )

find_path(ROCM_SMI_ROOT "include/rocm_smi/rocm_smi.h"
    PATHS "${ROCM_ROOT}"
    PATH_SUFFIXES "rocm_smi"
    )

find_library(ROCM_SMI_LIBRARY rocm_smi64
    PATHS "${ROCM_SMI_ROOT}/lib")

add_library(rocwmma INTERFACE)
target_link_libraries(rocwmma INTERFACE hip::device hip::host OpenMP::OpenMP_CXX ${ROCM_SMI_LIBRARY})

rocm_install_targets(
  TARGETS rocwmma
  INCLUDE library/include
)

if(ROCWMMA_BUILD_SAMPLES OR ROCWMMA_BUILD_TESTS)
  enable_testing()
  rocm_package_setup_component(clients)
endif()

if(ROCWMMA_BUILD_SAMPLES)
  rocm_package_setup_component(samples PARENT clients)
  add_subdirectory(samples)
endif()


if(ROCWMMA_BUILD_TESTS)
  rocm_package_setup_component(tests PARENT clients)
  add_subdirectory(test)
endif()

# Package
if(BUILD_ADDRESS_SANITIZER)
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
else()
  set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
endif()
rocm_package_add_dependencies("${DEPENDS_HIP_RUNTIME} >= 4.5.0")
rocm_package_add_deb_dependencies("libomp-dev")
rocm_package_add_rpm_dependencies("libomp-devel")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
  NAME rocwmma
  DESCRIPTION "AMD C++ library for facilitating GEMM, or GEMM-like 2D matrix multiplications on GPU leveraging MFMA instructions executing on matrix cores."
  MAINTAINER "rocWMMA Maintainer <rocwmma-maintainer@amd.com>"
  HEADER_ONLY
)
