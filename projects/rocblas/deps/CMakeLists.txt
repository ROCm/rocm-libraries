# ########################################################################
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#
# ########################################################################
# Helper cmake script to automate building dependencies for rocblas
# This script can be invoked manually by the user with 'cmake -P'

cmake_minimum_required( VERSION 3.9 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# The superbuild does not build anything itself; all compiling is done in external projects
project( rocblas-dependencies NONE )

option( BUILD_GTEST "Download and build googletest library" ON )
# option( BUILD_VERBOSE "Print helpful build debug information" OFF )

# if( BUILD_VERBOSE )
#   message( STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}" )
#   message( STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}" )
#   message( STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}" )
#   message( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
#   message( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
#   message( STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}" )
#   message( STATUS "CMAKE_CURRENT_LIST_FILE: ${CMAKE_CURRENT_LIST_FILE}" )
# endif( )

# This module scrapes the CMakeCache.txt file and attempts to get all the cli options the user specified to cmake invocation
include( get-cli-arguments )

# The following is a series of super-build projects; this cmake project will download and build
if( BUILD_GTEST )
  include( external-gtest )

  list( APPEND rocblas_dependencies googletest )
  set( gtest_custom_target COMMAND cd ${GTEST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . )
  set( gtest_install_target COMMAND cd ${GTEST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

add_custom_target( build_deps
  ${gtest_custom_target}
  DEPENDS ${rocblas_dependencies}
)

add_custom_target( install_deps
  ${gtest_install_target}
  DEPENDS ${build_deps}
)
