# #############################################################################
# Copyright (C) 2020 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# ############################################################################


# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on
# all the time.
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Print verbose compiler flags
if( BUILD_VERBOSE )
  include( ../cmake/verbose.cmake )
endif()

find_package(HIP REQUIRED)
set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")

# Configure a header file to pass the hipFFT version
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipfft/hipfft-version.h.in"
  "${PROJECT_BINARY_DIR}/include/hipfft/hipfft-version.h"
  )

# Public hipFFT headers
set( hipfft_headers_public
  include/hipfft/hipfft.h
  include/hipfft/hipfftXt.h
  include/hipfft/hipfftMp.h
  ${PROJECT_BINARY_DIR}/include/hipfft/hipfft-version.h
  )

source_group( "Header Files\\Public" FILES ${hipfft_headers_public} )

# Include sources
include( src/CMakeLists.txt )

# Create hipFFT library (and hipfftw on ROCm)
add_library( hipfft ${hipfft_source} ${hipfft_headers_public} )
add_library( hip::hipfft ALIAS hipfft )
set( HIPFFT_LIBRARY_TARGETS hipfft )
set( HIPFFT_EXPORT_TARGETS hip::hipfft )
if( BUILD_WITH_LIB STREQUAL "CUDA" )
  # static hipfft build should link against static cufft
  if( BUILD_SHARED_LIBS )
    if( NVHPC_FOUND )
      find_library( CUFFTW_LIB cufftw REQUIRED PATHS ${NVHPC_ROOT_DIR}/math_libs/lib64 )
      if( HIPFFT_MPI_ENABLE )
        target_link_libraries( hipfft PRIVATE NVHPC::CUFFTMP )
      else()
        target_link_libraries( hipfft PRIVATE NVHPC::CUFFT )
      endif()
    else()
      set( CUFFTW_LIB CUDA::cufftw )
      target_link_libraries( hipfft PRIVATE CUDA::cufft )
    endif()
  else()
    if( NVHPC_FOUND )
      find_library( CUFFTW_LIB cufftw_static REQUIRED PATHS ${NVHPC_ROOT_DIR}/math_libs/lib64 )
      target_link_libraries( hipfft PRIVATE NVHPC::CUFFT_static )
    else()
      set( CUFFTW_LIB CUDA::cufftw_static )
      target_link_libraries( hipfft PRIVATE CUDA::cufft_static )
    endif()
  endif()
  add_library( hipfftw ALIAS ${CUFFTW_LIB} )
  add_library( hip::hipfftw ALIAS ${CUFFTW_LIB} )
else()
  add_library( hipfftw ${hipfftw_source} ${hipfftw_headers_public} )
  add_library( hip::hipfftw ALIAS hipfftw )
  list( APPEND HIPFFT_LIBRARY_TARGETS hipfftw )
  list( APPEND HIPFFT_EXPORT_TARGETS hip::hipfftw )
  # hipfftw.h header uses the same hipfft-export.h as hipfft.h
  target_compile_definitions( hipfftw PRIVATE hipfft_EXPORTS )
endif()

set(static_depends)

# Target compile definitions
if( BUILD_WITH_COMPILER STREQUAL "HOST-DEFAULT" )
  if( BUILD_WITH_LIB STREQUAL "ROCM" )
    target_compile_definitions( hipfft PRIVATE __HIP_PLATFORM_AMD__ )
  else()
    target_compile_definitions( hipfft PRIVATE __HIP_PLATFORM_NVIDIA__ )
  endif()
endif()

if( HIPFFT_MPI_ENABLE )
  target_compile_definitions( hipfft PRIVATE HIPFFT_MPI_ENABLE )
endif()

# Target include directories
# Hip header files installed in cmake install includedir
# HIP_INCLUDE_DIRS is not required
foreach( lib ${HIPFFT_LIBRARY_TARGETS} )
  target_include_directories( ${lib}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipfft>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

  # Target link libraries - on ROCm we just use rocFFT for
  # everything.  CUDA has to be handled separately for normal and
  # "Mp" libs
  if( BUILD_WITH_LIB STREQUAL "ROCM" )
    list(APPEND static_depends PACKAGE rocfft)
    target_link_libraries( ${lib} PRIVATE roc::rocfft )
    if( WIN32 )
      target_link_libraries( ${lib} PRIVATE hip::device )
    endif()
    target_link_libraries( ${lib} PUBLIC hip::host )
  endif()

  target_compile_options( ${lib} PRIVATE ${WARNING_FLAGS} )

  # Target properties
  set_target_properties( ${lib} PROPERTIES CXX_EXTENSIONS NO )
  set_target_properties( ${lib} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

  if (BUILD_CODE_COVERAGE)
    target_compile_options( ${lib} PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping )
    target_link_options( ${lib} PUBLIC -fprofile-instr-generate )
  endif()

endforeach()

if( BUILD_WITH_LIB STREQUAL "CUDA" )
  target_include_directories( hipfft PUBLIC $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}> )
endif()

if (BUILD_WITH_COMPILER STREQUAL "HIP-NVCC" )
  set_property(TARGET hipfft PROPERTY POSITION_INDEPENDENT_CODE ON)
  set ( CXX_FLAGS "${CXX_FLAGS} -Xcompiler=-fPIC" )
endif()

#TODO:
# hipcc(with nvcc backend) build has problem for share library visibility,
# need to figure out the reason and enable visibility "hidden" for nvcc eventually.
if( NOT BUILD_WITH_COMPILER STREQUAL "HIP-NVCC" )
  set_target_properties( hipfft
    PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
endif()

# nvcc can not recognize shared library file name with suffix other than *.so when linking.
if (NOT BUILD_WITH_COMPILER STREQUAL "HIP-NVCC")
  rocm_set_soversion(hipfft ${hipfft_SOVERSION})
endif()

# Generate export header
include( GenerateExportHeader )
generate_export_header( hipfft EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipfft/hipfft-export.h )

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/library/include ${PROJECT_BINARY_DIR}/include)

if( ROCmCMakeBuildTools_FOUND )

  rocm_install_targets( TARGETS ${HIPFFT_LIBRARY_TARGETS}
    INCLUDE
    ${CMAKE_BINARY_DIR}/include
  )

  rocm_export_targets( TARGETS hip::hipfft
    DEPENDS PACKAGE hip
    STATIC_DEPENDS
      ${static_depends}
    NAMESPACE hip:: )
endif()
