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
find_package( ROCM 0.7.3 CONFIG PATHS /opt/rocm )
if(NOT ROCM_FOUND)
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  set( PROJECT_EXTERN_DIR "${CMAKE_CURRENT_BINARY_DIR}/extern" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
      ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(WARNING "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  else()
    message(STATUS "downloading... done")

    execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzvf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )
    execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
    execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

    find_package( ROCM 0.7.3 CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
  endif()
endif()
if( ROCM_FOUND )
  message(STATUS "Found ROCm")
  include(ROCMSetupVersion)
  include(ROCMCreatePackage)
  include(ROCMInstallTargets)
  include(ROCMPackageConfigHelpers)
  include(ROCMInstallSymlinks)
  include(ROCMClients)
  include(ROCMHeaderWrapper)
else()
  message(WARNING "Could not find rocm-cmake, packaging will fail.")
endif( )
