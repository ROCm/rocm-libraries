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

project( hipfft-clients-tests LANGUAGES CXX )


find_package( Boost REQUIRED)

set( Boost_USE_STATIC_LIBS OFF )

find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT DOUBLE )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

set( hipfft-test_source
  gtest_main.cpp
  hipfft_accuracy_test.cpp
  simple_test.cpp
  accuracy_test_1D.cpp
  accuracy_test_2D.cpp
  accuracy_test_3D.cpp
  accuracy_test_callback.cpp
  multi_device_test.cpp
  ../../shared/array_validator.cpp
  )

add_executable( hipfft-test ${hipfft-test_source} ${hipfft-test_includes} )

set( TEST_TARGETS hipfft-test )

# MPI worker for MPI tests
if( HIPFFT_MPI_ENABLE )
  # build MPI worker to support the tests
  add_executable( hipfft_mpi_worker hipfft_mpi_worker.cpp )
  list( APPEND TEST_TARGETS hipfft_mpi_worker )
  target_include_directories( hipfft_mpi_worker
    PRIVATE
    ${MPI_C_INCLUDE_PATH}
  )
  add_compile_definitions( HIPFFT_MPI_ENABLE )
endif()

if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
  if( WIN32 )
    find_package( HIP CONFIG REQUIRED )
  else()
    find_package( HIP MODULE REQUIRED )
  endif()
endif()

if( HIPFFT_BUILD_SCOPE )
  set( TESTS_OUT_DIR "/../staging" )
elseif( HIPFFT_CLIENTS_BUILD_SCOPE )
  set( TESTS_OUT_DIR "/../bin" )
else()
  set( TESTS_OUT_DIR "/bin" )
endif()
string( CONCAT TESTS_OUT_DIR "${PROJECT_BINARY_DIR}" ${TESTS_OUT_DIR} )

option( BUILD_CLIENTS_TESTS_OPENMP "Build tests with OpenMP" ON )

foreach( target ${TEST_TARGETS} )
  set_target_properties( ${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    )

  if( BUILD_WITH_LIB STREQUAL "ROCM" )
    target_compile_options( ${target} PRIVATE ${WARNING_FLAGS} )
    target_link_libraries( ${target}
      PRIVATE
      hip::host
      hip::device
    )
    foreach( gpu_target ${AMDGPU_TARGETS} )
      target_compile_options( ${target} PRIVATE --offload-arch=${gpu_target} )
    endforeach()

    if( NOT hiprand_FOUND )
      find_package( hiprand REQUIRED )
    endif()
    if( USE_HIPRAND )
      target_link_libraries( ${target} PRIVATE hip::hiprand )
    endif()
  else()
    target_compile_definitions( ${target} PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories( ${target} PRIVATE ${HIP_INCLUDE_DIRS})

    if( CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+$" )
      target_compile_options( ${target} PRIVATE -cuda -Xptxas=-w)
      target_link_options( ${target} PRIVATE -cuda)
    else()
      target_compile_options( ${target} PRIVATE -arch sm_53 -gencode=arch=compute_53,code=sm_53 -Xptxas=-w)
    endif()
    if( NVHPC_FOUND )
      target_link_libraries( ${target}  PRIVATE NVHPC::CUDART )
    else()
      target_link_libraries( ${target}  PRIVATE CUDA::cudart )
    endif()
    target_compile_definitions( ${target} PUBLIC _CUFFT_BACKEND )
  endif()

  if( BUILD_CLIENTS_TESTS_OPENMP )
    if( BUILD_WITH_LIB STREQUAL "CUDA" )
      message( STATUS "OpenMP is not supported on CUDA, building tests without it" )
    else()
      target_compile_options( ${target} PRIVATE -DBUILD_CLIENTS_TESTS_OPENMP )
      if(NOT (CMAKE_CXX_COMPILER MATCHES ".*hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+"))
        target_compile_options( ${target} PRIVATE -fopenmp )
        target_link_libraries( ${target} PRIVATE -fopenmp -L${HIP_CLANG_ROOT}/lib -Wl,-rpath=${HIP_CLANG_ROOT}/lib )
        target_include_directories( ${target} PRIVATE ${HIP_CLANG_ROOT}/include )
      else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
          target_compile_options( ${target} PRIVATE -fopenmp=libomp )
          target_link_options( ${target} PRIVATE -fopenmp=libomp )
        endif()
      endif()
    endif()
  endif()

  target_include_directories( ${target}
    PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
  )

  target_link_libraries( ${target}
    PRIVATE
    hip::hipfft
    ${FFTW_LIBRARIES}
  )

  if( HIPFFT_MPI_ENABLE )
    target_link_libraries( ${target}
      PRIVATE
      MPI::MPI_CXX
    )
  endif()

  set_target_properties(${target}
                        PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY 
                        ${TESTS_OUT_DIR})

  rocm_install(TARGETS ${target} COMPONENT tests)
endforeach()

if( GTEST_FOUND )
  target_include_directories( hipfft-test PRIVATE $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}> )
  target_link_libraries( hipfft-test PRIVATE ${GTEST_LIBRARIES} )
else()
  add_dependencies( hipfft-test gtest )
  target_include_directories( hipfft-test PRIVATE hipfft-test_include_dirs ${GTEST_INCLUDE_DIRS} )
  target_link_libraries( hipfft-test PRIVATE ${GTEST_LIBRARIES} )
endif()

if(FFTW_MULTITHREAD)
  target_compile_options( hipfft-test PRIVATE -DFFTW_MULTITHREAD )
endif( )

target_link_libraries( hipfft-test
  PRIVATE
  Threads::Threads
  )

if (WIN32)

  # Ensure tests run with HIP DLLs and not anything the driver owns
  # in system32.  Libraries like amdhip64.dll are also in the HIP
  # runtime, and we need run with those.  But the only way to make a
  # same-named DLL override something in system32 is to have it next
  # to the executable.  So copy them in.
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET hipfft-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} $<TARGET_FILE_DIR:hipfft-test> )
  endforeach( file_i )
endif()
