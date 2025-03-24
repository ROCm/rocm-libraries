# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################

cmake_minimum_required( VERSION 3.11 )

# Build documentation
option(BUILD_DOCS "Build documentation" OFF)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( hipblas-common LANGUAGES CXX )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_DIR}/cmake )

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
include(dependencies)

set ( VERSION_STRING "1.1.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

option( BUILD_VERBOSE "Output additional build information" OFF )

add_subdirectory( library )

# Build docs
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
endif( )

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT")

if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
  endif()
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

set( package_name hipblas-common )

set( HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Common files for hipBLAS libraries"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
    HEADER_ONLY
)
