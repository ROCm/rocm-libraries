# Copyright (C) 2021 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Build GTest from source as a fallback when no installed GTest is found.
#
# Callers must set GTEST_TARGET to the CMake target that needs gtest.
# Set GTEST_LINK_MAIN to ON to also link gtest_main (for targets that
# do not provide their own main).

include( ExternalProject )

# Set the external project's prefix relative to the root build directory to avoid building in each
# subdirectory where we need gtest.
set( _gtest_prefix ${PROJECT_BINARY_DIR}/googletest )
set( _gtest_lib
  ${_gtest_prefix}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX} )
set( _gtest_main_lib
  ${_gtest_prefix}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX} )

if( NOT TARGET googletest )
  ExternalProject_Add( googletest
    PREFIX ${_gtest_prefix}
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${_gtest_prefix} -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    BUILD_BYPRODUCTS ${_gtest_lib} ${_gtest_main_lib}
    DOWNLOAD_NO_PROGRESS YES
    DOWNLOAD_EXTRACT_TIMESTAMP YES
  )
endif()

target_include_directories( ${GTEST_TARGET} PRIVATE ${_gtest_prefix}/include )
target_link_libraries( ${GTEST_TARGET} PRIVATE ${_gtest_lib} )
if( GTEST_LINK_MAIN )
  target_link_libraries( ${GTEST_TARGET} PRIVATE ${_gtest_main_lib} )
endif()
add_dependencies( ${GTEST_TARGET} googletest )
