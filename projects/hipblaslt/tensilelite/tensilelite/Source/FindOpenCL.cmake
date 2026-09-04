################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

# Author: Kent Knox <kent.knox@amd dot com>
# Locate an OpenCL implementation.
# Currently supports AMD APP SDK (http://developer.amd.com/sdks/AMDAPPSDK/Pages/default.aspx/)
#
# Defines the following variables:
#
#   OPENCL_FOUND - Found the OPENCL framework
#   OPENCL_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   OPENCL_LIBRARIES - libopencl
#
# Accepts the following variables as input:
#
#   OPENCL_ROOT - (as a CMake or environment variable)
#                The root directory of the OpenCL implementation found
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether findOpenCL should search for
#                              64bit or 32bit libs
#-----------------------
# Example Usage:
#
#    find_package(OPENCL REQUIRED)
#    include_directories(${OPENCL_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${OPENCL_LIBRARIES})
#
#-----------------------
include( CheckSymbolExists )
include( CMakePushCheckState )

if( DEFINED OPENCL_ROOT OR DEFINED ENV{OPENCL_ROOT})
  message( STATUS "Defined OPENCL_ROOT: ${OPENCL_ROOT}, ENV{OPENCL_ROOT}: $ENV{OPENCL_ROOT}" )
endif( )

find_path(OPENCL_INCLUDE_DIRS
  NAMES OpenCL/cl.h CL/cl.h
  HINTS
    ${OPENCL_ROOT}/include
    $ENV{OPENCL_ROOT}/include
    $ENV{AMDAPPSDKROOT}/include
    $ENV{CUDA_PATH}/include
  PATHS
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    /opt/rocm/opencl/include
  DOC "OpenCL header file path"
)
mark_as_advanced( OPENCL_INCLUDE_DIRS )
message( STATUS "OPENCL_INCLUDE_DIRS: ${OPENCL_INCLUDE_DIRS}" )

set( OpenCL_VERSION "0.0" )

cmake_push_check_state( RESET )
set( CMAKE_REQUIRED_INCLUDES "${OPENCL_INCLUDE_DIRS}" )

# Bug in check_symbol_exists prevents us from specifying a list of files, so we loop
# Only 1 of these files will exist on a system, so the other file will not clobber the output variable
if( APPLE )
   set( CL_HEADER_FILE "OpenCL/cl.h" )
else( )
   set( CL_HEADER_FILE "CL/cl.h" )
endif( )

check_symbol_exists( CL_VERSION_2_0 ${CL_HEADER_FILE} HAVE_CL_2_0 )
check_symbol_exists( CL_VERSION_1_2 ${CL_HEADER_FILE} HAVE_CL_1_2 )
check_symbol_exists( CL_VERSION_1_1 ${CL_HEADER_FILE} HAVE_CL_1_1 )
# message( STATUS "HAVE_CL_2_0: ${HAVE_CL_2_0}" )
# message( STATUS "HAVE_CL_1_2: ${HAVE_CL_1_2}" )
# message( STATUS "HAVE_CL_1_1: ${HAVE_CL_1_1}" )

# set OpenCL_VERSION to the highest detected version
if( HAVE_CL_2_0 )
  set( OpenCL_VERSION "2.0" )
elseif( HAVE_CL_1_2 )
  set( OpenCL_VERSION "1.2" )
elseif( HAVE_CL_1_1 )
  set( OpenCL_VERSION "1.1" )
endif( )

cmake_pop_check_state( )

# Search for 64bit/32bit libs
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set(LIB64 ON)
  message( STATUS "FindOpenCL searching for 64-bit libraries" )
else( )
  set(LIB64 OFF)
  message( STATUS "FindOpenCL searching for 32-bit libraries" )
endif( )

if( LIB64 )
  find_library( OPENCL_LIBRARIES
    NAMES OpenCL
    HINTS
      ${OPENCL_ROOT}/lib
      $ENV{OPENCL_ROOT}/lib
      $ENV{AMDAPPSDKROOT}/lib
      $ENV{CUDA_PATH}/lib
    DOC "OpenCL dynamic library path"
    PATH_SUFFIXES x86_64 x64
    PATHS
    /usr/lib
    /usr/local/cuda/lib
    /opt/rocm/opencl/lib
  )
else( )
  find_library( OPENCL_LIBRARIES
    NAMES OpenCL
    HINTS
      ${OPENCL_ROOT}/lib
      $ENV{OPENCL_ROOT}/lib
      $ENV{AMDAPPSDKROOT}/lib
      $ENV{CUDA_PATH}/lib
    DOC "OpenCL dynamic library path"
    PATH_SUFFIXES x86 Win32
    PATHS
    /usr/lib
    /usr/local/cuda/lib
    /opt/rocm/opencl/lib
  )
endif( )
mark_as_advanced( OPENCL_LIBRARIES )

# If we asked for OpenCL 1.2, and we found a version installed greater than that, pass the 'use deprecated' flag
if( (OpenCL_FIND_VERSION VERSION_LESS "2.0") AND (OpenCL_VERSION VERSION_GREATER OpenCL_FIND_VERSION) )
    add_definitions( -DCL_USE_DEPRECATED_OPENCL_2_0_APIS )
    add_definitions( -DCL_USE_DEPRECATED_OPENCL_1_2_APIS )

    # If we asked for OpenCL 1.1, and we found a version installed greater than that, pass the 'use deprecated' flag
    if( (OpenCL_FIND_VERSION VERSION_LESS "1.2") AND (OpenCL_VERSION VERSION_GREATER OpenCL_FIND_VERSION) )
        add_definitions( -DCL_USE_DEPRECATED_OPENCL_1_1_APIS )
    endif( )
endif( )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OPENCL
    REQUIRED_VARS OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS
    VERSION_VAR OpenCL_VERSION
    )

if( NOT OPENCL_FOUND )
  message( STATUS "FindOpenCL looked for libraries named: OpenCL" )
else( )
  message(STATUS "FindOpenCL ${OPENCL_LIBRARIES}, ${OPENCL_INCLUDE_DIRS}")

  # UNKNOWN implies on windows I can use the .lib file for IMPORTED_LOCATION
  # we don't care about the location of opencl.dll, it's installed in system directories
  if (WIN32)
    add_library( opencl UNKNOWN IMPORTED )
  endif()

  set_target_properties( opencl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${OPENCL_LIBRARIES}"
  )

endif()
