# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required( VERSION 3.16 )

# This should appear before the project command, because it does not
# use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
  set( CPACK_PACKAGING_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
  set( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

set( ROCFFT_CLIENTS_BUILD_SCOPE ON )

# This project may compile dependencies for clients
project( rocfft-clients LANGUAGES CXX C )

set(CMAKE_CXX_STANDARD 17)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( NOT ROCmCMakeBuildTools_FOUND )
  find_package( ROCmCMakeBuildTools REQUIRED )
endif()

include( ROCMInstallTargets )

# Adding Version File to rocfft-client, this avoids empty rocfft-client package
file ( WRITE "${PROJECT_BINARY_DIR}/package/client-version"
             "${rocfft_VERSION_MAJOR}.${rocfft_VERSION_MINOR}.${rocfft_VERSION_PATCH}-${BUILD_ID}\n" )
rocm_install ( FILES ${PROJECT_BINARY_DIR}/package/client-version DESTINATION .info COMPONENT clients)


# This option only works for make/nmake and the ninja generators, but
# no reason it shouldn't be on all the time.
# This tells cmake to create a compile_commands.json file that can be
# used with clang tooling or vim.
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if(NOT ROCFFT_BUILD_SCOPE AND 
   NOT BUILD_CLIENTS_SAMPLES AND 
   NOT BUILD_CLIENTS_TESTS AND 
   NOT BUILD_CLIENTS_BENCH)
  set( BUILD_CLIENTS_SAMPLES ON )
  set( BUILD_CLIENTS_TESTS ON )
  set( BUILD_CLIENTS_BENCH ON )
endif()

# each backend requires different libraries for host and device code
if( USE_CUDA )
  if( NOT DEFINED CUDA_PREFIX )
    message( FATAL_ERROR "CUDA_PREFIX variable is required." )
  endif()
  if( NOT DEFINED CUDA_ARCH )
    message( FATAL_ERROR "CUDA_ARCH variable is required." )
  endif()
  add_compile_options(-I${HIP_ROOT_DIR}/include -I${CUDA_PREFIX}/include -D__HIP_PLATFORM_NVIDIA__)
  add_link_options(-L${CUDA_PREFIX}/lib64 -pthread)
  add_compile_options(--cuda-path=${CUDA_PREFIX} --cuda-gpu-arch=${CUDA_ARCH} -xcuda)
  set( ROCFFT_CLIENTS_HOST_LINK_LIBS -lcudart -ldl -lrt )
else()
  set( ROCFFT_CLIENTS_HOST_LINK_LIBS hip::host )
  set( ROCFFT_CLIENTS_DEVICE_LINK_LIBS hip::device )
endif()

if( ROCFFT_MPI_ENABLE )
  find_package( MPI REQUIRED )
endif()

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_TESTS )
  add_subdirectory( tests )
endif( )

if( BUILD_CLIENTS_BENCH )
  add_subdirectory( bench )
endif( )
