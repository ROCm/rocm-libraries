# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper cmake script to automate building dependencies for rocsparselt
# This script can be invoked manually by the user with 'cmake -P'

# AMD ROCm software requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake )

# Consider removing this in the future
# It can be annoying for visual studio developers to build a project that tries to install into 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# The superbuild does not build anything itself; all compiling is done in external projects
project( hipsparselt_dependencies-dependencies NONE )

# This module scrapes the CMakeCache.txt file and attempts to get all the cli options the user specified to cmake invocation
# include( get-cli-arguments )

# The following is a series of super-build projects; this cmake project will download and build

option( BUILD_LAPACK "Download and build lapack library" ON )

if( BUILD_LAPACK )
  include( external-lapack )

  list( APPEND hipsparselt_dependencies lapack )
  set( lapack_custom_target COMMAND cd ${LAPACK_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )


# POLICY CMP0037 - "Target names should not be reserved and should match a validity pattern"
# Familiar target names like 'install' should be OK at the super-build level
if( POLICY CMP0037 )
  cmake_policy( SET CMP0037 OLD )
endif( )

add_custom_target( install
  ${lapack_custom_target}
  DEPENDS ${hipsparselt_dependencies}
)