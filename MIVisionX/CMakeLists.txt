# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
  set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()
set(VERSION "3.2.0")

# Set Project Version and Language
project(mivisionx VERSION ${VERSION} LANGUAGES CXX)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "MIVisionX default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# MIVisionX Default Options
option(ENHANCED_MESSAGE "MIVisionX Enhanced Message Option"        ON)
option(NEURAL_NET       "Build MIVisionX with Neural Net Support"  ON)
option(LOOM             "Build MIVisionX with LOOM Support"        ON)
option(GPU_SUPPORT      "Build MIVisionX with GPU Support"         ON)
option(MIGRAPHX         "Build MIVisionX with MIGraphX Support"    ON)

if(WIN32)
  set(BACKEND "OpenCL")
  set(ENHANCED_MESSAGE OFF)
  message("-- Windows Detected -- Backend Set to OpenCL")
endif()
set(DEFAULT_BUILD_TYPE "Release")

if(ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(BoldBlue    "${Esc}[1;34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
endif()
message("-- ${BoldBlue}MIVisionX Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}MIVisionX Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(GPU_SUPPORT OFF)
  set(BACKEND "CPU")
  message("-- ${Magenta}Apple macOS Detected -- GPU Support turned OFF${ColourReset}")
endif()

if(NOT DEFINED BACKEND)
  set(BACKEND "HIP")   # set default backend to HIP
else()
  if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL" OR "${BACKEND}" STREQUAL "opencl")
    set(BACKEND "OPENCL")
  elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
    set(BACKEND "HIP")
  elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
    set(BACKEND "CPU")
    set(GPU_SUPPORT OFF)
  else()
    message("-- ${Red}Warning: MIVisionX backend option: ${BACKEND} -- unknown${ColourReset}")
    message("-- ${Red}Warning: MIVisionX default backend will be enforced${ColourReset}")
    set(BACKEND "HIP")
  endif()
endif()
message("-- ${BoldBlue}MIVisionX Backend set to -- ${BACKEND}${ColourReset}")

# MIVisionX Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "MIVisionX Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
  # -O0  -- Don't Optimize output file
  # -g -- generate debugging information
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
  # -O3 -- Optimize output file
  # -DNDEBUG -- turn off asserts
  # -fPIC -- Generate position-independent code if possible
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
endif()
message("-- ${BoldBlue}MIVisionX Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
find_package(HALF QUIET)

if(HALF_FOUND)
  add_definitions(-DAMD_FP16_SUPPORT)
  include_directories(${HALF_INCLUDE_DIRS})
  message("-- ${Blue}MIVisionX Built with float16 Support -- -DAMD_FP16_SUPPORT definition added${ColourReset}")
else()
  message("-- ${Blue}MIVisionX Built with float16 Support OFF${ColourReset}")
endif()

# HIP backend defaults
if("${BACKEND}" STREQUAL "HIP")
  set(LOOM OFF)
  # Set supported GPU Targets
  if(NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
      find_package(ROCmCMakeBuildTools QUIET)
      if(NOT ROCmCMakeBuildTools_FOUND)
          find_package(ROCM QUIET)
      endif()
      include(ROCMCheckTargetIds OPTIONAL RESULT_VARIABLE HAS_ROCM_CHECK_TARGET_IDS)

      set(OPTIONAL_GPU_TARGETS "gfx950;gfx1200;gfx1201")
      if(HAS_ROCM_CHECK_TARGET_IDS)
          rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
      else() # if we don't have rocm_check_target_ids, just assume the targets are available
          set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
      endif()
      set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
  endif()

  # Set AMD GPU_TARGETS
  if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
    message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColourReset}")
    if(DEFINED ENV{AMDGPU_TARGETS})
      set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
    else()
      set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
  endif()

  if(DEFINED ENV{GPU_ARCHS})
      set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
  elseif(GPU_TARGETS)
      message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColourReset}")
  else()
      set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
  endif()
  message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColourReset}")
endif()

message("-- ${Cyan}MIVisionX Developer Options${ColourReset}")
message("-- ${Cyan}     -D NEURAL_NET=${NEURAL_NET} [Turn ON/OFF Neural Net Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D LOOM=${LOOM} [Turn ON/OFF LOOM OpenCL Modules (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D GPU_SUPPORT=${GPU_SUPPORT} [Turn ON/OFF GPU support (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D MIGRAPHX=${MIGRAPHX} [Turn ON/OFF MIGraphX Module (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D BACKEND=${BACKEND} [Select MIVisionX Backend [options:CPU/OPENCL/HIP](default:HIP)]${ColourReset}")

add_subdirectory(amd_openvx)
add_subdirectory(amd_openvx_extensions)
add_subdirectory(utilities)

# install tests
install(FILES tests/CMakeLists.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
install(DIRECTORY cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
install(DIRECTORY tests/openvx_api_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
install(DIRECTORY tests/vision_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
install(DIRECTORY data/videos DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)

# CTests
enable_testing()
include(CTest)
option(BUILT_WITH_SOURCE "MIVisionX built with source"        ON)
add_subdirectory(tests)

# set package information
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "MIVisionX Support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD ROCm")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://github.com/ROCm/MIVisionX")

if(ENABLE_ASAN_PACKAGING)
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD MIVisionX address sanitizer libraries")
  set(CPACK_RPM_PACKAGE_SUMMARY "AMD MIVisionX address sanitizer libraries")
else()
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit")
  set(CPACK_RPM_PACKAGE_SUMMARY "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit")
endif()

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

# set license information
set(CPACK_RPM_PACKAGE_LICENSE    "MIT")
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# set license information - {ROCM_PATH}/share/doc/mivisionx
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# Set the dependent packages
set(MIVISIONX_RUNTIME_PACKAGE_LIST  "rocm-hip-runtime, openmp-extras-runtime, rpp, rocblas, miopen-hip, migraphx")

# Set the dev dependent packages
set(MIVISIONX_DEBIAN_DEV_PACKAGE_LIST  "half, rocm-hip-runtime-dev, openmp-extras-dev, rpp-dev, rocblas-dev, miopen-hip-dev, migraphx-dev, pkg-config, libavcodec-dev, libavformat-dev, libavutil-dev, libswscale-dev, libopencv-dev")
# TBD - Some RPM packages need Fusion Packages - libavcodec-devel, libavformat-devel, libavutil-devel, libswscale-devel, libopencv
set(MIVISIONX_RPM_DEV_PACKAGE_LIST  "half, rocm-hip-runtime-devel, openmp-extras-devel, rpp-devel, rocblas-devel, miopen-hip-devel, migraphx-devel, pkg-config")

# Add OS specific dependencies
if(EXISTS "/etc/os-release")
  file(READ "/etc/os-release" OS_RELEASE)
  string(REGEX MATCH "22.04" UBUNTU_22_FOUND ${OS_RELEASE})
  if(UBUNTU_22_FOUND)
    set(MIVISIONX_DEBIAN_DEV_PACKAGE_LIST "${MIVISIONX_DEBIAN_DEV_PACKAGE_LIST}, libstdc++-12-dev")
  endif()
endif()

# '%{?dist}' breaks manual builds on debian systems due to empty Provides
execute_process(
    COMMAND rpm --eval %{?dist}
    RESULT_VARIABLE PROC_RESULT
    OUTPUT_VARIABLE EVAL_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

# set package 
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_URL ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_AUTOREQPROV "no")
# Debian package - COMPONENT
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${MIVISIONX_RUNTIME_PACKAGE_LIST}")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS 
  "rocm-core, ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}, ${MIVISIONX_DEBIAN_DEV_PACKAGE_LIST}")
# Debian package - specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan, ${MIVISIONX_RUNTIME_PACKAGE_LIST}" )
# Debian package - Test
set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_DEV_PACKAGE_NAME}" )
# RPM package - COMPONENT
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${MIVISIONX_RUNTIME_PACKAGE_LIST}")
set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_RUNTIME_PACKAGE_NAME}, ${MIVISIONX_RPM_DEV_PACKAGE_LIST}")
# RPM package - specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, ${MIVISIONX_RUNTIME_PACKAGE_LIST}" )
# RPM package specific variable for ASAN
set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_DEV_PACKAGE_NAME}" )

if(NOT ROCM_DEP_ROCMCORE)
  string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_TEST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_TEST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
endif()

if(ENABLE_ASAN_PACKAGING)
  # ASAN Package requires asan component with only libraries and license file
  set(CPACK_COMPONENTS_ALL asan)
else()
  set(CPACK_COMPONENTS_ALL runtime dev test)
endif()

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
set(CPACK_RPM_PACKAGE_RELEASE "local")

if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

if(APPLE)
  set(CPACK_GENERATOR "Bundle")
  set(CPACK_BUNDLE_NAME "${PROJECT_NAME}")
  set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/.github/info.plist")
  set(CPACK_BUNDLE_ICON "${CMAKE_SOURCE_DIR}/docs/data/MIVisionX.bmp")
else()
  set(CPACK_GENERATOR "TGZ;ZIP")
  if(EXISTS ${MAKE_NSIS_EXE})
    list(APPEND CPACK_GENERATOR "NSIS")
  endif()
  if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
    message("-- ${White}AMD ROCm MIVisionX RunTime Package -- ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES}${ColourReset}")
    message("-- ${White}AMD ROCm MIVisionX Dev Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
  endif()
  if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
    message("-- ${White}AMD ROCm MIVisionX RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
    message("-- ${White}AMD ROCm MIVisionX Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
  endif()
endif()

include(CPack)

cpack_add_component(runtime
                  DISPLAY_NAME "MIVisionX Runtime Package"
                  DESCRIPTION "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit. \
MIVisionX runtime package provides MIVisionX libraries and license.txt")

cpack_add_component(dev
                  DISPLAY_NAME "MIVisionX Develop Package"
                  DESCRIPTION "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit. \
MIVisionX develop package provides MIVisionX libraries, header files, samples, and license.txt")

cpack_add_component(asan
                  DISPLAY_NAME "MIVisionX ASAN Package"
                  DESCRIPTION "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit. \
MIVisionX ASAN package provides MIVisionX ASAN libraries")

cpack_add_component(test
                  DISPLAY_NAME "MIVisionX Test Package"
                  DESCRIPTION "AMD MIVisionX is a comprehensive Computer Vision and ML Inference Toolkit. \
MIVisionX Test package provides MIVisionX Test Components")
