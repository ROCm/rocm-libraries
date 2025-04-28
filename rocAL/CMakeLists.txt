# Copyright (c) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_CXX_COMPILER clang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

# rocAL Version
set(VERSION "2.3.0")
# Set Project Version and Language
project(rocal VERSION ${VERSION} LANGUAGES CXX)

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
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "rocAL default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# rocAL Default Options
option(ENHANCED_MESSAGE "rocAL Enhanced Message Option"        ON)
option(GPU_SUPPORT      "Build rocAL with GPU Support"         ON)
option(BUILD_PYPACKAGE  "Build rocAL Python Package"           ON)
option(PYTHON_VERSION_SUGGESTED "Python version to build rocal" "")

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
message("-- ${BoldBlue}rocAL Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}rocAL Install Path -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")

if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
  set(GPU_SUPPORT OFF)
  set(BACKEND "CPU")
  message("-- ${Magenta}Apple macOS Detected -- GPU Support turned OFF${ColourReset}")
else()
  # Changes for RPATH Removal from Binaries:
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  set(CMAKE_SKIP_INSTALL_RPATH TRUE)
endif()

if(NOT DEFINED BACKEND)
  set(BACKEND "HIP")   # set default backend to HIP
else()
  if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL")
    set(BACKEND "OPENCL")
  elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
    set(BACKEND "HIP")
  elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
    set(BACKEND "CPU")
    set(GPU_SUPPORT OFF)
  else()
    message("-- ${Red}Warning: rocAL backend option: ${BACKEND} -- unknown${ColourReset}")
    message("-- ${Red}Warning: rocAL default backend will be enforced${ColourReset}")
    set(BACKEND "HIP")
  endif()
endif()
message("-- ${BoldBlue}rocAL Backend set to -- ${BACKEND}${ColourReset}")

# rocAL Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "rocAL Default Build Type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
  # -O0  -- Don't Optimize output file
  # -g -- generate debugging information
  # -Og -- Optimize for debugging experience rather than speed or size
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Og")
else()
  # -O3 -- Optimize output file
  # -DNDEBUG -- turn off asserts
  # -fPIC -- Generate position-independent code if possible
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
endif()
message("-- ${BoldBlue}rocAL Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
find_package(HALF QUIET)

if(HALF_FOUND)
  add_definitions(-DAMD_FP16_SUPPORT)
  include_directories(${HALF_INCLUDE_DIRS})
  message("-- ${Blue}rocAL Built with float16 Support -- -DAMD_FP16_SUPPORT definition added${ColourReset}")
else()
  message("-- ${Blue}rocAL Built with float16 Support OFF${ColourReset}")
endif()

message("-- ${Cyan}rocAL Developer Options${ColourReset}")
message("-- ${Cyan}     -D GPU_SUPPORT=${GPU_SUPPORT} [Turn ON/OFF GPU support (default:ON)]${ColourReset}")
message("-- ${Cyan}     -D BACKEND=${BACKEND} [Select rocAL Backend [options:CPU/OPENCL/HIP](default:HIP)]${ColourReset}")
message("-- ${Cyan}     -D BUILD_PYPACKAGE=${BUILD_PYPACKAGE} [rocAL Python Package(default:ON)]${ColourReset}")
message("-- ${Cyan}     -D PYTHON_VERSION_SUGGESTED=${PYTHON_VERSION_SUGGESTED} [User provided python version to use for rocAL Python Bindings(default:System Version)]${ColourReset}")

add_subdirectory(rocAL)
if(BUILD_PYPACKAGE)
  add_subdirectory(rocAL_pybind)
else()
  message("-- ${Cyan}rocAL Python Module turned OFF by user option -D BUILD_PYPACKAGE=OFF ${ColourReset}")
endif()

# install rocAL docs -- {ROCM_PATH}/${CMAKE_INSTALL_DATADIR}/doc/rocal/
install(FILES docs/README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/rocal COMPONENT runtime)

# test package
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp_api/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
# CTest - Needs rocAL Installed
enable_testing()
include(CTest)
add_subdirectory(tests/cpp_api)

# set package information
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT       "rocAL Support <mivisionx.support@amd.com>")
set(CPACK_PACKAGE_VENDOR        "AMD ROCm")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://github.com/ROCm/rocAL")

if(ENABLE_ASAN_PACKAGING)
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocAL address sanitizer libraries")
  set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocAL address sanitizer libraries")
else()
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "AMD rocAL is a comprehensive augmentation library")
  set(CPACK_RPM_PACKAGE_SUMMARY "AMD rocAL is a comprehensive augmentation library")
endif()

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

# set license information
set(CPACK_RPM_PACKAGE_LICENSE    "MIT")
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# set license information - {ROCM_PATH}/share/doc/rocal
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
set(ROCAL_DEBIAN_PACKAGE_LIST  "mivisionx")
set(ROCAL_RPM_PACKAGE_LIST     "mivisionx")
# Set the dev dependent packages
set(ROCAL_DEBIAN_DEV_PACKAGE_LIST  "mivisionx-dev, liblmdb-dev, libprotobuf-dev, libturbojpeg0-dev")
set(ROCAL_RPM_DEV_PACKAGE_LIST  "mivisionx-devel, lmdb-devel, protobuf-devel")

# Add OS specific dependencies
if (EXISTS "/etc/os-release")
  file(READ "/etc/os-release" OS_RELEASE)
  string(REGEX MATCH "rhel" RHEL_FOUND ${OS_RELEASE})
  string(REGEX MATCH "Mariner" MARINER_FOUND ${OS_RELEASE})
  string(REGEX MATCH "Ubuntu" UBUNTU_FOUND ${OS_RELEASE})
  string(REGEX MATCH "Debian" DEBIAN_FOUND ${OS_RELEASE})
  # Add TurboJPEG
  if(RHEL_FOUND)
    set(ROCAL_RPM_PACKAGE_LIST     "${ROCAL_RPM_PACKAGE_LIST}, turbojpeg")
    set(ROCAL_RPM_DEV_PACKAGE_LIST  "${ROCAL_RPM_DEV_PACKAGE_LIST}, turbojpeg-devel")
  endif()
  if(UBUNTU_FOUND)
    set(ROCAL_DEBIAN_PACKAGE_LIST   "${ROCAL_DEBIAN_PACKAGE_LIST}, libturbojpeg:amd64")
  elseif(DEBIAN_FOUND)
    set(ROCAL_DEBIAN_PACKAGE_LIST   "${ROCAL_DEBIAN_PACKAGE_LIST}, libturbojpeg0")
  endif()
  # Add rocDecode and rocJpeg deps for Non-Mariner
  if(NOT MARINER_FOUND)
    set(ROCAL_DEBIAN_PACKAGE_LIST  "${ROCAL_DEBIAN_PACKAGE_LIST}, rocdecode, rocjpeg")
    set(ROCAL_RPM_PACKAGE_LIST     "${ROCAL_RPM_PACKAGE_LIST}, rocdecode, rocjpeg")
    set(ROCAL_DEBIAN_DEV_PACKAGE_LIST  "${ROCAL_DEBIAN_DEV_PACKAGE_LIST}, rocdecode-dev, rocjpeg-dev")
    set(ROCAL_RPM_DEV_PACKAGE_LIST  "${ROCAL_RPM_DEV_PACKAGE_LIST}, rocdecode-devel, rocjpeg-devel")
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
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${ROCAL_DEBIAN_PACKAGE_LIST}")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS 
  "rocm-core, ${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}, ${ROCAL_DEBIAN_DEV_PACKAGE_LIST}")
# Debian package - specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan, ${ROCAL_DEBIAN_PACKAGE_LIST}" )
# Debian package - Test
set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${CPACK_DEBIAN_DEV_PACKAGE_NAME}" )
# RPM package - COMPONENT
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${ROCAL_RPM_PACKAGE_LIST}")
set(CPACK_RPM_RUNTIME_PACKAGE_PROVIDES "${PROJECT_NAME}")
set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_RUNTIME_PACKAGE_NAME}, ${ROCAL_RPM_DEV_PACKAGE_LIST}")
set(CPACK_RPM_DEV_PACKAGE_PROVIDES "${PROJECT_NAME}-devel")
# RPM package - specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, ${ROCAL_RPM_PACKAGE_LIST}" )
set(CPACK_RPM_ASAN_PACKAGE_PROVIDES "${PROJECT_NAME}-asan")
# RPM package specific variable for ASAN
set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test" )
set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${CPACK_RPM_DEV_PACKAGE_NAME}" )
set(CPACK_RPM_TEST_PACKAGE_PROVIDES "${PROJECT_NAME}-test")

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
  set(CPACK_BUNDLE_ICON "${CMAKE_SOURCE_DIR}/docs/data/rocAL_logo.png")
  set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/.github/info.plist")
else()
  set(CPACK_GENERATOR "TGZ;ZIP")
  if(EXISTS ${MAKE_NSIS_EXE})
    list(APPEND CPACK_GENERATOR "NSIS")
  endif()
  if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
    message("-- ${White}AMD ROCm rocAL RPM RunTime Package -- ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES}${ColourReset}")
    message("-- ${White}AMD ROCm rocAL RPM Dev Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
  endif()
  if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
    message("-- ${White}AMD ROCm rocAL DEB RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
    message("-- ${White}AMD ROCm rocAL DEB Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
endif()
endif()

include(CPack)

cpack_add_component(runtime
                  DISPLAY_NAME "rocAL Runtime Package"
                  DESCRIPTION "AMD rocAL is a comprehensive augmentation library. \
rocAL runtime package provides rocAL libraries and license.txt")

cpack_add_component(dev
                  DISPLAY_NAME "rocAL Develop Package"
                  DESCRIPTION "AMD rocAL is a comprehensive augmentation library. \
rocAL develop package provides rocAL libraries, header files, samples, and license.txt")

cpack_add_component(asan
                  DISPLAY_NAME "rocAL ASAN Package"
                  DESCRIPTION "AMD rocAL is a comprehensive augmentation library. \
rocAL ASAN package provides rocAL ASAN libraries")

cpack_add_component(test
                  DISPLAY_NAME "rocAL Test Package"
                  DESCRIPTION "AMD rocAL is a comprehensive augmentation library. \
rocAL Test package provides rocAL Test Components")
