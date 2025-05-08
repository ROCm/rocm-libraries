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

# This is incremented when the ABI to the library changes
set( rocfft_SOVERSION 0.1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but
# no reason it shouldn't be on all the time.
# This tells cmake to create a compile_commands.json file that can be
# used with clang tooling or vim.
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "rocfft_VERSION: ${rocfft_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_CXX_COMPILER: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS_RELEASE: " ${CMAKE_EXE_LINKER_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: " ${CMAKE_SHARED_LINKER_FLAGS_RELEASE} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_C_FLAGS: ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_CXX_FLAGS: ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_DEBUG: ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
endif( )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/rocfft/rocfft-version.h.in" "${PROJECT_BINARY_DIR}/include/rocfft/rocfft-version.h" )
# Copy rocfft from source dir path to build dir
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/rocfft/rocfft.h" "${PROJECT_BINARY_DIR}/include/rocfft/rocfft.h" COPYONLY )

set( rocfft_headers_public
  ${PROJECT_BINARY_DIR}/include/rocfft/rocfft.h
  ${PROJECT_BINARY_DIR}/include/rocfft/rocfft-version.h
)

source_group( "Header Files\\Public" FILES ${rocfft_headers_public} )

# Build into subdirectories
add_subdirectory( src )
