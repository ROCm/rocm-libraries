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

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

project( hipfft-clients-bench LANGUAGES CXX )

set( hipfft_bench_source bench.cpp ../../shared/array_validator.cpp )
set( hipfft_bench_includes bench.h ../../shared/array_validator.h )

add_executable( hipfft-bench ${hipfft_bench_source} ${hipfft_bench_includes} )

target_compile_options( hipfft-bench PRIVATE ${WARNING_FLAGS} )

set_target_properties( hipfft-bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON )

target_include_directories( hipfft-bench
  PRIVATE
  $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../rocFFT/library/include>
  )
  
if((NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$") OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
    if( WIN32 )
      find_package( HIP CONFIG REQUIRED )
    else()
      find_package( HIP MODULE REQUIRED )
    endif()

    target_link_libraries( hipfft-bench PRIVATE hip::host hip::device )
  else()
    target_compile_definitions( hipfft-bench PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories( hipfft-bench PRIVATE ${HIP_INCLUDE_DIRS})
  endif()
endif()

if ( BUILD_WITH_LIB STREQUAL "CUDA" )
  if( CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+$" )
    target_compile_options( hipfft-bench PRIVATE -cuda -Xptxas=-w)
    target_link_options( hipfft-bench PRIVATE -cuda)
  else()
    target_compile_options( hipfft-bench PRIVATE -arch sm_53 -gencode=arch=compute_53,code=sm_53 -Xptxas=-w)
  endif()
  target_link_libraries( hipfft-bench PRIVATE ${CUDA_LIBRARIES} )
else()
  if( USE_HIPRAND AND NOT hiprand_FOUND )
    find_package( hiprand REQUIRED )
  endif()
  if( USE_HIPRAND )
    target_link_libraries( hipfft-bench PRIVATE hip::hiprand )
  endif()
endif()

target_link_libraries( hipfft-bench PRIVATE hip::hipfft )

set_target_properties( hipfft-bench PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipfft-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

if( HIPFFT_BUILD_SCOPE )
  set( BENCH_OUT_DIR "/../staging" )
elseif( HIPFFT_CLIENTS_BUILD_SCOPE )
  set( BENCH_OUT_DIR "/../bin" )
else()
  set( BENCH_OUT_DIR "/bin")
endif()
string( CONCAT BENCH_OUT_DIR "${PROJECT_BINARY_DIR}" ${BENCH_OUT_DIR} )

set_target_properties( hipfft-bench
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
  ${BENCH_OUT_DIR} )

rocm_install(TARGETS hipfft-bench COMPONENT benchmarks)
