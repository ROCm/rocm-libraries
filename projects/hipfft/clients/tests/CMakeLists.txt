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

# Dependencies

find_package( ROCmCMakeBuildTools REQUIRED CONFIG PATHS /opt/rocm )
include(ROCMInstallTargets)
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake )


# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( hipfft-clients-tests LANGUAGES CXX )

if( NOT HIPFFT_BUILD_SCOPE )
  find_package( hipfft REQUIRED CONFIG PATHS )
endif()

find_package( Boost REQUIRED)

set( Boost_USE_STATIC_LIBS OFF )

# searching FFTW with FindFFTW.cmake that is provided by project
# because distro versions of FFTW does not include the cmake specific
# helper files
find_package( FFTW 3.0 MODULE COMPONENTS FLOAT DOUBLE )
if( FFTW_FOUND )
  message(STATUS "FFTW found by FindFFTW.cmake")
else()
  # searching fftw versions build from source code which contain
  # cmake helper files like FFTW3Config.cmake
  find_package(FFTW3 CONFIG QUIET )
  find_package(FFTW3f CONFIG QUIET )
  if( FFTW3_FOUND AND FFTW3f_FOUND )
    message(STATUS "FFTW3 found by using the FFTW3Config.cmake")
    if ( NOT WIN32 )
      get_target_property(fftw3_libs FFTW3::fftw3 IMPORTED_LOCATION_RELEASE)
      get_target_property(fftw3f_libs FFTW3::fftw3f IMPORTED_LOCATION_RELEASE)
    else()
      get_target_property(fftw3_libs FFTW3::fftw3 IMPORTED_IMPLIB_RELEASE)
      get_target_property(fftw3f_libs FFTW3::fftw3f IMPORTED_IMPLIB_RELEASE)
    endif()

    # Look for omp (preferred) or thread libraries. These are not a
    # hard requirement, but are nice to have to make FFTW run faster.
    # To simplify the testing, we will not accept a mixed combination
    # where FFTW is only build agauns the OMP library support and
    # FFTWF is only build with the THREADS library support.
    find_library(FFTW_OMP_LIBRARY NAMES fftw3_omp HINTS ${FFTW3_LIBRARY_DIRS})
    find_library(FFTWF_OMP_LIBRARY NAMES fftw3f_omp HINTS ${FFTW3f_LIBRARY_DIRS})
    if ( FFTW_OMP_LIBRARY AND FFTWF_OMP_LIBRARY )
      list( APPEND fftw3_libs ${FFTW_OMP_LIBRARY} )
      list( APPEND fftw3f_libs ${FFTWF_OMP_LIBRARY} )
      set( FFTW_MULTITHREAD TRUE )
    else()
      find_library( FFTW_THREADS_LIBRARY fftw3_threads HINTS ${FFTW3_LIBRARY_DIRS} )
      find_library( FFTWF_THREADS_LIBRARY fftw3f_threads HINTS ${FFTW3f_LIBRARY_DIRS} )
      if ( FFTW_THREADS_LIBRARY AND FFTWF_THREADS_LIBRARY )
        list( APPEND fftw3_libs ${FFTW_THREADS_LIBRARY} )
        list( APPEND fftw3f_libs ${FFTWF_THREADS_LIBRARY} )
        set( FFTW_MULTITHREAD TRUE )
      endif()
    endif()

    set (FFTW_FOUND ${FFTW3_FOUND})
    set (FFTW_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS})
    set (FFTW_LIBRARIES ${fftw3_libs} )
    list( APPEND FFTW_LIBRARIES ${fftw3f_libs} )
  endif()
endif()
message(STATUS "FFTW_FOUND: ${FFTW_FOUND}")
message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}")
message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}")

set( BUILD_WITH_LIB "ROCM" CACHE STRING "Build ${PROJECT_NAME} with ROCM or CUDA libraries" )

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
  hipfftw_test.cpp
  multi_device_test.cpp
  multi_stream_test.cpp
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
    find_package( hip REQUIRED CONFIG PATHS /opt/rocm/lib/cmake/hip/ )
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
if( BUILD_CLIENTS_TESTS_OPENMP AND NOT BUILD_WITH_LIB STREQUAL "CUDA" )
  # Attempt to find a config version, which provides openmp_LIB_DIR.
  find_package( OpenMP CONFIG QUIET PATHS "${HIP_CLANG_ROOT}/lib/cmake" )
  if( NOT OpenMP_FOUND OR NOT DEFINED openmp_LIB_DIR )
     # Fall-back to module mode.
     find_package( OpenMP MODULE REQUIRED )
     set( BUILD_RPATH "${HIP_CLANG_ROOT}/lib" )
     set( INSTALL_RPATH "$ORIGIN/../llvm/lib" )
  else()
     set( BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}" )
     set( INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}" )

  endif()
endif()

foreach( target ${TEST_TARGETS} )
  set_target_properties( ${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    )

  if( BUILD_CLIENTS_TESTS_OPENMP )
    set_target_properties( ${TEST_TARGETS} PROPERTIES
       BUILD_RPATH "${BUILD_RPATH}"
    )
    set_target_properties( ${TEST_TARGETS} PROPERTIES
      INSTALL_RPATH "${INSTALL_RPATH}"
    )
  endif()

  if( USE_HIPRAND )
    if( NOT hiprand_FOUND )
      find_package( hiprand REQUIRED )
    endif()
    
    if( BUILD_WITH_LIB STREQUAL "CUDA" )
      # nvcc doesn't know what do with hiprand.so.1.1, but we can try linking it to hiprand.so,
      # assuming that it's in the same directory.
      get_target_property( HIPRAND_FILE_PATH hip::hiprand LOCATION )
      string( REGEX REPLACE ".so(.[0-9]*)*" ".so" HIPRAND_FILE_PATH_NOVER "${HIPRAND_FILE_PATH}" )
      message( "Renamed hiprand so filename: ${HIPRAND_FILE_PATH_NOVER}" )
      
      target_link_libraries( ${target} PRIVATE ${HIPRAND_FILE_PATH_NOVER} )
    else()
      target_link_libraries( ${target} PRIVATE hip::hiprand )
    endif()
  endif()
  
  if( BUILD_WITH_LIB STREQUAL "ROCM" )
    target_compile_options( ${target} PRIVATE ${WARNING_FLAGS} )
    target_link_libraries( ${target}
      PRIVATE
      hip::host
      hip::device
    )
    foreach( gpu_target ${GPU_TARGETS} )
      target_compile_options( ${target} PRIVATE --offload-arch=${gpu_target} )
    endforeach()

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

  target_include_directories( ${target}
    PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
  )

  target_link_libraries( ${target}
    PRIVATE
    hip::hipfft
    ${FFTW_LIBRARIES}
  )
  
  if( BUILD_CLIENTS_TESTS_OPENMP )
    if( BUILD_WITH_LIB STREQUAL "CUDA" )
      message( STATUS "OpenMP is not supported on CUDA, building tests without it" )
    else()
      if( DEFINED ${openmp_LIB_DIR} )
        set_target_properties( ${target} PROPERTIES BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}" )
        set_target_properties( ${target} PROPERTIES INSTALL_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}" )
      endif()
      if( TARGET OpenMP::omp )
        set( HIPFFT_OPENMP_LIBRARY OpenMP::omp )
      else()
        set( HIPFFT_OPENMP_LIBRARY OpenMP::OpenMP_CXX )
      endif()
      target_link_libraries( ${target} PRIVATE ${HIPFFT_OPENMP_LIBRARY} )
    endif()
  endif()

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

find_package( GTest 1.11.0 )

if( GTest_FOUND )
  target_include_directories( hipfft-test PRIVATE $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}> )
  target_link_libraries( hipfft-test PRIVATE GTest::gtest )
else()
  # gtest build by the hipFFT
  add_dependencies( hipfft-test gtest )
  target_include_directories( hipfft-test PRIVATE hipfft-test_include_dirs ${GTEST_INCLUDE_DIRS} )
  target_link_libraries( hipfft-test PRIVATE ${GTEST_LIBRARIES} )
endif()

# tests have callback functions, which need to be built as relocatable device code
if( BUILD_WITH_LIB STREQUAL "CUDA" )
  target_compile_options( hipfft-test PRIVATE -dc )
else()
  # -fgpu-rdc causes failure at link stage on Windows
  if (NOT WIN32)
    target_compile_options( hipfft-test PRIVATE -fgpu-rdc )
    target_link_options( hipfft-test PRIVATE -fgpu-rdc )
  endif()
endif()

if(FFTW_MULTITHREAD)
  target_compile_options( hipfft-test PRIVATE -DFFTW_MULTITHREAD )
endif( )

target_link_libraries( hipfft-test
  PRIVATE
  Threads::Threads
  ${CMAKE_DL_LIBS}
  )

# hipfft-test will opens the hipfftw library but does not link to it
if( TARGET hipfftw )
  add_dependencies( hipfft-test hipfftw )
endif()

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

option(BUILD_CODE_COVERAGE "Build with code coverage flags (clang only)" OFF)
set(COVERAGE_TEST_OPTIONS "--smoketest;--gtest_filter=-*call*" CACHE STRING "Command line arguments for hipfft-test when generating a code coverage report (Note: an additional run of hipfft-test targeting multi_gpu* and callback* tests is always executed and coverage results are aggregated)")
if (BUILD_CODE_COVERAGE)
  add_custom_target(
    code_cov_tests
    DEPENDS hipfft-test
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/hipfft-coverage_%p.profraw" GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:hipfft-test> --precompile=./clients/staging/hipfft-test-precompile.db ${COVERAGE_TEST_OPTIONS}
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/hipfft-coverage_%p.profraw" GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:hipfft-test> --precompile=./clients/staging/hipfft-test-precompile-multi_gpu-plus-callback.db --gtest_filter=multi_gpu*:callback*
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/hipfft-coverage_*.profraw -o ./coverage-report/hipfft.profdata
    COMMAND ${LLVM_COV} report -object ./library/libhipfftw.so -object ./library/libhipfft.so -instr-profile=./coverage-report/hipfft.profdata
    COMMAND ${LLVM_COV} show -object ./library/libhipfftw.so -object ./library/libhipfft.so -instr-profile=./coverage-report/hipfft.profdata -format=html -output-dir=coverage-report
    COMMAND ${LLVM_COV} export -object ./library/libhipfftw.so -object ./library/libhipfft.so -instr-profile=./coverage-report/hipfft.profdata -format=lcov > ./coverage-report/coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

endif()
