# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# CMake v3.21.0 can identify ROCMClang, but there are some issues with feature detection.
# For now, disable ROCMClang detection to make it work the same as CMake v3.20 and earlier.
set(__skip_rocmclang ON)


# This will add compile option: -std=c++17
set( CMAKE_CXX_STANDARD 17 )
# Without this line, it will add -std=gnu++17 instead, which may have issues.
set( CMAKE_CXX_EXTENSIONS OFF )

# disable this next option and test build whenever compiler changes to verify unused arguments are safe to ignore
# add_compile_options( -Wno-unused-command-line-argument )

# TODO: address [[nodiscard]] warnings
add_compile_options( $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result> )


if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  # Determine if CXX Compiler is amdclang
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
  string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
endif()

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message( STATUS "Use amdclang to build for amdgpu backend" )

  if( CMAKE_CXX_COMPILER MATCHES ".*hipcc.*" )
    message( STATUS "WARNING: hipcc compiler use is deprecated. Use amdclang++ directly." )
  endif()

  # set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fallow-half-arguments-and-returns" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )

  if ( CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT WIN32 )
    set ( CMAKE_CXX_FLAGS_DEBUG "-O1 ${CMAKE_CXX_FLAGS_DEBUG} -gz -ggdb" )
  endif()

elseif( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  message( SEND_ERROR "ERROR: HCC compiler is no longer supported!" )
else()
  message( SEND_ERROR "ERROR: Unsupported compiler detected" )
endif( )


# FOR OPTIONAL CODE COVERAGE
if(BUILD_CODE_COVERAGE)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  #add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
  #add_compile_options(-fprofile-arcs -ftest-coverage)
  #add_link_options(--coverage -lgcov)
endif()

if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  # Fortran not supported, add_link_options below invalid for fortran linking
  set(BUILD_FORTRAN_CLIENTS OFF)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()
