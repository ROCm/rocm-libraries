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

project( rocfft-clients-bench LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( NOT TARGET rocfft )
  find_package( rocfft REQUIRED CONFIG PATHS )
endif( )

if( NOT HIP_FOUND )
  find_package( hip REQUIRED PATHS /opt/rocm/lib/cmake/hip/ )
endif()

if( NOT ROCmCMakeBuildTools_FOUND )
  find_package( ROCmCMakeBuildTools REQUIRED )
endif()

if( USE_HIPRAND AND NOT hiprand_FOUND )
  find_package( hiprand REQUIRED )
endif()

include( ROCMInstallTargets )

set( bench_list rocfft-bench dyna-rocfft-bench )
foreach( bench ${bench_list})
  if(${bench} STREQUAL "rocfft-bench")
    add_executable( ${bench} ../../shared/array_validator.cpp bench.cpp bench.h )
  else()
    add_executable( ${bench} ../../shared/array_validator.cpp dyna-bench.cpp bench.h )
  endif()

  target_compile_options( ${bench} PRIVATE ${WARNING_FLAGS} -Wno-cpp )

  # NB: hip-clang includes omp.h, so we need to specify the location
  # of ROCM_CLANG_ROOT at cmake config time if we are using clang++.

  target_include_directories( ${bench}
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/include>
    ${HIP_CLANG_ROOT}/include
    ${ROCM_CLANG_ROOT}/include
    )

  if(${bench} STREQUAL "rocfft-bench")
    target_link_libraries( ${bench}
      PRIVATE
      hip::device
      roc::rocfft
      )
  else()
    target_link_libraries( ${bench}
      PRIVATE
      ${CMAKE_DL_LIBS}
      hip::device
      )
  endif()
    
  if( USE_HIPRAND )
    target_link_libraries( ${bench}
    PRIVATE
    hip::hiprand
    )
    target_compile_definitions( ${bench} PRIVATE USE_HIPRAND )
  endif()

  # We need to include both rocfft.h and rocfft-export.h
  target_include_directories( ${bench}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../library/include/
    ${HIP_CLANG_ROOT}/include
    )

  target_link_libraries( ${bench} PUBLIC
    ${ROCFFT_CLIENTS_HOST_LINK_LIBS}
  )

  if( ROCFFT_MPI_ENABLE )
    target_link_libraries( ${bench}
    PRIVATE
    MPI::MPI_CXX
    )
  if ( ROCFFT_CRAY_MPI_ENABLE)
    target_link_libraries( ${bench}
      PRIVATE
      "mpi_gtl_hsa"
      )
      get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY )
      target_link_directories( ${bench}
	PRIVATE
	${MPI_LIBDIR}/../../../../gtl/lib )
    endif()
  endif()

  set_target_properties( ${bench} PROPERTIES
    CXX_STANDARD_REQUIRED ON
  )

  if( ROCFFT_BUILD_SCOPE )
    set( BENCH_OUT_DIR "/../staging" )
  elseif( ROCFFT_CLIENTS_BUILD_SCOPE )
    set( BENCH_OUT_DIR "/../bin" )
  else()
    set( BENCH_OUT_DIR "/bin")
  endif()
  string( CONCAT BENCH_OUT_DIR "${PROJECT_BINARY_DIR}" ${BENCH_OUT_DIR} )

  set_target_properties(${bench}
                        PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                        ${BENCH_OUT_DIR} )

  rocm_install(TARGETS ${bench} COMPONENT benchmarks)
endforeach()

# Link dyna-rocfft-bench to the experimental filesystem library if
# it's not available in the standard library.
include( ../../cmake/std-filesystem.cmake )
target_link_std_experimental_filesystem( dyna-rocfft-bench )
