# #############################################################################
# Copyright (C) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# #############################################################################

# HIP
if( NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
    if( WIN32 )
      find_package( HIP CONFIG REQUIRED )
    else()
      find_package( HIP REQUIRED )
    endif()
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
  endif()
else()
  if( BUILD_WITH_LIB STREQUAL "CUDA" )
    set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
  else()
    if( WIN32 )
      find_package( HIP CONFIG REQUIRED )
    else()
      find_package( HIP REQUIRED )
    endif()
  endif()
endif()
  
# Either rocfft or cufft is required
if(NOT BUILD_WITH_LIB STREQUAL "CUDA")
  if( HIPFFT_MPI_ENABLE )
    find_package( MPI REQUIRED )
  endif()
  find_package(rocfft REQUIRED)
else()
  # cufft may be in the HPC SDK or ordinary CUDA
  if( HIPFFT_MPI_ENABLE )
    if( NOT BUILD_SHARED_LIBS )
      message( FATAL_ERROR "cufftMp is shared-only, static build is not possible" )
    endif()
    # MPI support is only in HPC SDK
    find_package(NVHPC REQUIRED COMPONENTS CUDA MATH MPI)
  else()
    find_package(NVHPC QUIET COMPONENTS CUDA MATH)
  endif()
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  find_package(CUDAToolkit REQUIRED)
endif()

# ROCm
find_package( ROCmCMakeBuildTools CONFIG PATHS /opt/rocm )
if(NOT ROCmCMakeBuildTools_FOUND)
  set( PROJECT_EXTERN_DIR "${CMAKE_CURRENT_BINARY_DIR}/extern" )

  include( FetchContent )

  FetchContent_Declare( rocm_cmake_local
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake
    GIT_TAG rocm-6.4.1
    GIT_SHALLOW ON
  )

  FetchContent_MakeAvailable( rocm_cmake_local )

  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${rocm_cmake_local_SOURCE_DIR} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build ${rocm_cmake_local_SOURCE_DIR} --target install
    WORKING_DIRECTORY ${rocm_cmake_local_SOURCE_DIR} )

  find_package( ROCmCMakeBuildTools REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()
if( ROCmCMakeBuildTools_FOUND )
  message(STATUS "Found ROCm")
  include(ROCMSetupVersion)
  include(ROCMCreatePackage)
  include(ROCMInstallTargets)
  include(ROCMPackageConfigHelpers)
  include(ROCMInstallSymlinks)
  include(ROCMCheckTargetIds)
  include(ROCMClients)
  include(ROCMHeaderWrapper)
else()
  message(WARNING "Could not find rocm-cmake, packaging will fail.")
endif( )
