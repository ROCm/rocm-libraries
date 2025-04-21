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

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "hipblas_VERSION: ${hipblas_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )

  message( STATUS "\t==>CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )
endif( )

# Copy Public Headers to Build Dir
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblas-common.h" "${PROJECT_BINARY_DIR}/include/hipblas-common/hipblas-common.h" COPYONLY)

set( hipblas_headers_public
  include/hipblas-common.h
)

source_group( "Header Files\\Public" FILES ${hipblas_headers_public} )

add_library( hipblas-common INTERFACE )

target_include_directories( hipblas-common
  INTERFACE
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

rocm_install_targets(
  TARGETS hipblas-common
  INCLUDE
    ${CMAKE_BINARY_DIR}/include
)

rocm_export_targets(TARGETS roc::hipblas-common
                      NAMESPACE roc::)
