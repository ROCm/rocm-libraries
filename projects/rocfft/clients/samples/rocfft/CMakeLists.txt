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

cmake_minimum_required( VERSION 3.16 )

# This should appear before the project command, because it does not
# use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( rocfft-clients-samples-rocfft LANGUAGES CXX )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( NOT TARGET rocfft )
  find_package( rocfft REQUIRED CONFIG PATHS )
endif( )

if( NOT HIP_FOUND )
  find_package( hip REQUIRED PATHS /opt/rocm/lib/cmake/hip/ )
endif()

if( USE_HIPRAND AND NOT hiprand_FOUND )
  find_package( hiprand REQUIRED )
endif()

set( sample_list rocfft_example_complexcomplex rocfft_example_realcomplex rocfft_example_set_stream
     rocfft_example_callback )

foreach( sample ${sample_list} )

  add_executable( ${sample} ${sample}.cpp )

  target_include_directories(
    ${sample}
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  )

  target_link_libraries(
    ${sample}
    PRIVATE roc::rocfft
    )

  if( USE_HIPRAND )
    target_link_libraries(
      ${sample}
      PRIVATE
      hip::hiprand
      )
    target_compile_definitions( ${sample} PRIVATE USE_HIPRAND )
  endif()

  target_compile_options( ${sample} PRIVATE ${WARNING_FLAGS} -Wno-cpp )

  set_target_properties( ${sample} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )

  if( ROCFFT_BUILD_SCOPE )
    set( SAMPLES_ROCFFT_OUT_DIR "/../../staging" )
  elseif( ROCFFT_CLIENTS_BUILD_SCOPE )
    set( SAMPLES_ROCFFT_OUT_DIR "/../../bin" )
  elseif( ROCFFT_CLIENTS_SAMPLES_BUILD_SCOPE )
    set( SAMPLES_ROCFFT_OUT_DIR "/../bin" )
  else()
    set( SAMPLES_ROCFFT_OUT_DIR "/bin" )
  endif()
  string( CONCAT SAMPLES_ROCFFT_OUT_DIR "${PROJECT_BINARY_DIR}" ${SAMPLES_ROCFFT_OUT_DIR} )

  set_target_properties(${sample}
                        PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY 
                        ${SAMPLES_ROCFFT_OUT_DIR})

  if( CUDA_FOUND )
    target_include_directories( ${sample}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
      )
    target_compile_definitions( ${sample} PRIVATE __HIP_PLATFORM_NVCC__ )
  endif( )
  target_link_libraries( ${sample} PRIVATE ${ROCFFT_CLIENTS_HOST_LINK_LIBS} ${ROCFFT_CLIENTS_DEVICE_LINK_LIBS} )

endforeach( )

# callback functions need to be built as relocatable device code
# (causes failure at link stage on Windows)
if (NOT WIN32)
  target_compile_options( rocfft_example_callback PRIVATE -fgpu-rdc )
  target_link_options( rocfft_example_callback PRIVATE -fgpu-rdc )
endif()
