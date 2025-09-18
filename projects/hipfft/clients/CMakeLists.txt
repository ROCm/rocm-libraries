# #############################################################################
# Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
# #############################################################################

# CMake version according to latest ROCm platform requirements
cmake_minimum_required( VERSION 3.16 )

# We use C++17 features, this will add compile option: -std=c++17
set( CMAKE_CXX_STANDARD 17 )
set(CMAKE_CXX_EXTENSIONS OFF)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX
    "${PROJECT_BINARY_DIR}/package"
    CACHE
    PATH
    "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX
    "/opt/rocm"
    CACHE
    PATH
    "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does
# not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

set( HIPFFT_CLIENTS_BUILD_SCOPE ON )

# This project may compile dependencies for clients
project( hipfft-clients LANGUAGES CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

if(NOT (CMAKE_CXX_COMPILER MATCHES ".*hipcc$" OR
        CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+" OR
	CMAKE_CXX_COMPILER MATCHES ".*nvcc" OR
	CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+"
  	)
  )
  if(BUILD_CLIENTS)
    message( FATAL_ERROR "Using BUILD_CLIENTS=ON requires a compiler capable of building device code (hipcc, clang, nvcc, nvc++)." )
  endif()
endif()


# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on
# all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# if hipfft is not a target, then we know clients are built separately from the library and we must
# search for the hipfft package
if( NOT TARGET hipfft )
  find_package( hipfft REQUIRED CONFIG PATHS )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_TESTS )
  find_package( GTest QUIET )
  include( ExternalProject )
  if( NOT GTest_FOUND )
    set( GTEST_INCLUDE_DIRS
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest/googletest/include
      )
    set( GTEST_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/gtest-build/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
      )

    ExternalProject_Add( gtest
      URL https://github.com/google/googletest/archive/release-1.11.0.tar.gz
      URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${GTEST_LIBRARIES}
      )
    ExternalProject_Get_Property( gtest source_dir binary_dir )
  endif()
  add_subdirectory( tests )
endif( )

if( BUILD_CLIENTS_BENCH )
  add_subdirectory( bench )
endif( )
