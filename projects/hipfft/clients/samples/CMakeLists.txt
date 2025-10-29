# #############################################################################
# Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

CMAKE_MINIMUM_REQUIRED( VERSION 3.16 )

project( hipfft-clients-samples-rocfft LANGUAGES CXX )

# We use C++17 features, this will add compile option: -std=c++17
set( CMAKE_CXX_STANDARD 17 )

if( NOT TARGET hipfft )
  find_package( hipfft REQUIRED CONFIG PATHS )
endif( )

set( sample_list
  hipfft_1d_z2z
  hipfft_1d_d2z
  hipfft_2d_z2z
  hipfft_2d_d2z
  hipfft_3d_z2z
  hipfft_3d_d2z
  hipfft_planmany_2d_z2z
  hipfft_planmany_2d_r2c
  hipfft_multigpu_2d_z2z
  hipfft_setworkarea
  )

# callback sample has its own HIP code, so it needs to be built with hipcc or clang++
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  # on cuFFT backend, use of callbacks requires linking against the
  # static cuFFT library
  if( NOT (BUILD_WITH_LIB STREQUAL "CUDA") OR NOT BUILD_SHARED_LIBS )
    list( APPEND sample_list hipfft_callback )
  else()
    message( STATUS "hipfft_callback sample disabled on non-static CUDA build" )
  endif()
else()
  message( STATUS "hipfft_callback sample disabled, requires hipcc or Clang++ build" )
endif()

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cpp )

  set_target_properties( ${sample} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON )

  target_link_libraries( ${sample} PRIVATE hip::hipfft )

  target_compile_options( ${sample} PRIVATE ${WARNING_FLAGS} )
  
  if( NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    if( WIN32 )
      find_package( HIP CONFIG REQUIRED )
    else()
      find_package( HIP MODULE REQUIRED )
    endif()
    if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
      target_link_libraries( ${sample} PRIVATE hip::host hip::device )
    else()
      target_compile_definitions( ${sample} PRIVATE __HIP_PLATFORM_NVIDIA__)
      target_include_directories( ${sample} PRIVATE ${HIP_INCLUDE_DIRS})
    endif()
  endif()

  if ( BUILD_WITH_LIB STREQUAL "CUDA" )
    if( CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+$" )
      target_compile_options( ${sample} PRIVATE -cuda -Xptxas=-w)
      target_link_options( ${sample} PRIVATE -cuda)
    else()
      target_compile_options( ${sample} PRIVATE -arch sm_53 -gencode=arch=compute_53,code=sm_53 -Xptxas=-w)
    endif()
    target_link_libraries( ${sample} PRIVATE ${CUDA_LIBRARIES} )
  endif()

  target_include_directories( ${sample}
    PRIVATE
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    ${HIP_ROOT_DIR}
    )
  
  set_target_properties( ${sample} PROPERTIES CXX_EXTENSIONS NO )
  
  if( HIPFFT_BUILD_SCOPE )
    set( SAMPLES_OUT_DIR "/../staging" )
  elseif( HIPFFT_CLIENTS_BUILD_SCOPE )
    set( SAMPLES_OUT_DIR "/../bin" )
  else()
    set( SAMPLES_OUT_DIR "/bin" )
  endif()
  string( CONCAT SAMPLES_OUT_DIR "${PROJECT_BINARY_DIR}" ${SAMPLES_OUT_DIR} )

  set_target_properties( ${sample}
                         PROPERTIES 
                         RUNTIME_OUTPUT_DIRECTORY 
                         ${SAMPLES_OUT_DIR} )
  
endforeach()

# callback code must be compiled as relocatable device code
if( hipfft_callback IN_LIST sample_list )
  if( BUILD_WITH_LIB STREQUAL "CUDA" )
    target_compile_options( hipfft_callback PRIVATE -dc )
  else()
    # -fgpu-rdc causes failure at link stage on Windows
    if (NOT WIN32)
      target_compile_options( hipfft_callback PRIVATE -fgpu-rdc )
      target_link_options( hipfft_callback PRIVATE -fgpu-rdc )
    endif()
  endif()
endif()
