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

project( rocfft-clients-tests LANGUAGES CXX )

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

set( rocfft-test_source
  gtest_main.cpp
  rocfft_accuracy_test.cpp
  bitwise_repro/bitwise_repro_test.cpp
  accuracy_test.cpp
  accuracy_test_1D.cpp	
  accuracy_test_2D.cpp
  accuracy_test_3D.cpp
  accuracy_test_adhoc.cpp
  accuracy_test_emulation.cpp
  accuracy_test_callback.cpp
  accuracy_test_checkstride.cpp
  multithread_test.cpp
  multi_device_test.cpp
  hermitian_test.cpp
  hipGraph_test.cpp
  callback_change_type.cpp
  default_callbacks_test.cpp
  unit_test.cpp
  buffer_hash_test.cpp
  validate_length_stride.cpp
  random.cpp
  ../../shared/array_validator.cpp
  )

add_executable( rocfft-test ${rocfft-test_source} ${rocfft-test_includes} )
add_executable( rtc_helper_crash rtc_helper_crash.cpp )

# rocFFT device code builds with -O3 by default.  rocfft-test
# contains device code for callback functions, so ensure the device
# code is built with the same optimization level to minimize chance
# of a mismatch
target_compile_options( rocfft-test PRIVATE -Xarch_device -O3 )

# callback functions need to be built as relocatable device code
# (causes failure at link stage on Windows)
if (NOT WIN32)
  target_compile_options( rocfft-test PRIVATE -fgpu-rdc )
  target_link_options( rocfft-test PRIVATE -fgpu-rdc )
endif()

find_package( Boost REQUIRED )
set( Boost_DEBUG ON )
set( Boost_DETAILED_FAILURE_MSG ON )

option( BUILD_FFTW "Download and build FFTW" OFF )

# look for installed FFTW if we weren't asked to build it
if( NOT BUILD_FFTW )
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
endif()

include( ExternalProject )
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  # use extract timestamp for fetched files instead of timestamps in the archive
  cmake_policy(SET CMP0135 NEW)
endif()

# also try to build FFTW if FFTW isn't present
if( BUILD_FFTW OR NOT FFTW_FOUND )
  set(FFTW_LIBRARIES_DOUBLE
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_double-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3_threads${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_double-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(FFTW_LIBRARIES_SINGLE
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_single-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3f_threads${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_single-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3f${CMAKE_SHARED_LIBRARY_SUFFIX})

  set(FFTW_CMAKE_ARGS_COMMON
      -DDISABLE_FORTRAN=ON
      -DENABLE_AVX2=ON
      -DENABLE_THREADS=ON
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_TESTS=OFF
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER})

  set(FFTW_SRC_URL http://www.fftw.org/fftw-3.3.9.tar.gz CACHE STRING "Location of FFTW source code")
  set(FFTW_SRC_SHA256 bf2c7ce40b04ae811af714deb512510cc2c17b9ab9d6ddcf49fe4487eea7af3d CACHE STRING "SHA256 hash of FFTW source code")

  # build double-precision FFTW
  ExternalProject_Add(fftw_double
                      URL ${FFTW_SRC_URL}
                      URL_HASH SHA256=${FFTW_SRC_SHA256}
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fftw
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
                      CMAKE_ARGS ${FFTW_CMAKE_ARGS_COMMON}
                      INSTALL_COMMAND ""
                      BUILD_BYPRODUCTS ${FFTW_LIBRARIES_DOUBLE})
  ExternalProject_Get_Property( fftw_double source_dir binary_dir )

  # also build single-precision fftw from the same source dir
  ExternalProject_Add(fftw_single
                      DOWNLOAD_COMMAND ""
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fftw
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
                      CMAKE_ARGS ${FFTW_CMAKE_ARGS_COMMON} -DENABLE_FLOAT=ON
                      INSTALL_COMMAND ""
                      BUILD_BYPRODUCTS ${FFTW_LIBRARIES_SINGLE}
                      DEPENDS fftw_double)
  ExternalProject_Get_Property( fftw_single source_dir binary_dir )

  set(FFTW_INCLUDE_DIRS
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw/api)
  set(FFTW_LIBRARIES
      ${FFTW_LIBRARIES_DOUBLE}
      ${FFTW_LIBRARIES_SINGLE})

  # FFTW we build is always threaded
  set( FFTW_MULTITHREAD TRUE )

  add_dependencies( rocfft-test fftw_double fftw_single )
  rocm_install(
    FILES ${FFTW_LIBRARIES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/fftw
    COMPONENT clients-common
  )
else()
  include_directories(${FFTW_INCLUDE_DIRS})
endif()

set( rocfft-test_include_dirs
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
  ${ROCM_CLANG_ROOT}/include
  )

set( rocfft-test_link_libs
  ${FFTW_LIBRARIES}
  )

option( BUILD_CLIENTS_TESTS_OPENMP "Build tests with OpenMP" ON )
if( BUILD_CLIENTS_TESTS_OPENMP )
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
  if( TARGET OpenMP::omp )
    set( ROCFFT_OPENMP_LIBRARY OpenMP::omp )
  else()
    set( ROCFFT_OPENMP_LIBRARY OpenMP::OpenMP_CXX )
  endif()
  list( APPEND rocfft-test_link_libs ${ROCFFT_OPENMP_LIBRARY} )

  set_target_properties( rocfft-test PROPERTIES
     BUILD_RPATH "{$BUILD_RPATH}"
  )
  set_target_properties( rocfft-test PROPERTIES
    INSTALL_RPATH "${INSTALL_RPATH}"
  )
endif()

find_package( GTest QUIET )

if( GTest_FOUND )
    target_link_libraries( rocfft-test PRIVATE GTest::gtest )
else()
  include( ../cmake/build-gtest.cmake )
  add_dependencies( rocfft-test gtest )
  list( APPEND rocfft-test_include_dirs ${GTEST_INCLUDE_DIRS} )
  list( APPEND rocfft-test_link_libs ${GTEST_LIBRARIES} )
endif()

target_compile_options( rocfft-test PRIVATE ${WARNING_FLAGS} -Wno-cpp )

target_include_directories( rocfft-test
  PRIVATE
  ${rocfft-test_include_dirs}
  )

if( NOT BUILD_SHARED_LIBS )
  list(APPEND rocfft-test_link_libs ${ROCFFT_CLIENTS_HOST_LINK_LIBS} ${ROCFFT_CLIENTS_DEVICE_LINK_LIBS})
endif()

if( NOT ROCFFT_BUILD_SCOPE )
  find_package(SQLite3 REQUIRED)
  set( ROCFFT_SQLITE_LIB SQLite::SQLite3)
endif()

target_link_libraries( rocfft-test
  PRIVATE
  hip::device
  roc::rocfft
  ${ROCFFT_SQLITE_LIB}
  ${rocfft-test_link_libs}
  )

if ( USE_HIPRAND )
  target_link_libraries( rocfft-test
  PRIVATE
  hip::hiprand
  )
  target_compile_definitions( rocfft-test PRIVATE USE_HIPRAND )
endif()

if( ROCFFT_MPI_ENABLE )
  target_link_libraries( rocfft-test
    PRIVATE
    MPI::MPI_CXX
    )
  add_compile_definitions( ROCFFT_MPI_ENABLE )

  if ( ROCFFT_CRAY_MPI_ENABLE )
    target_link_libraries( rocfft-test
      PRIVATE
      "mpi_gtl_hsa"
      )
    get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY )
    target_link_directories( rocfft-test
      PRIVATE
      ${MPI_LIBDIR}/../../../../gtl/lib )
  endif()
endif()


include( ../../cmake/std-filesystem.cmake )
target_link_std_experimental_filesystem( rocfft-test )

if( USE_CUDA )
  target_include_directories( rocfft-test
    PRIVATE
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-test PRIVATE __HIP_PLATFORM_NVCC__ )
endif( )
target_link_libraries( rocfft-test PRIVATE
   ${ROCFFT_CLIENTS_HOST_LINK_LIBS}
   ${ROCFFT_CLIENTS_DEVICE_LINK_LIBS} )

include( ../../cmake/sqlite.cmake )
target_link_libraries( rocfft-test PUBLIC ${ROCFFT_SQLITE_LIB} )
target_include_directories( rocfft-test PRIVATE ${sqlite_local_SOURCE_DIR} )

set_property( TARGET rocfft-test APPEND PROPERTY LINK_LIBRARIES ${ROCFFT_SQLITE_LIB} )

if(FFTW_MULTITHREAD)
  target_compile_options( rocfft-test PRIVATE -DFFTW_MULTITHREAD )
endif( )

set_target_properties( rocfft-test PROPERTIES
  CXX_STANDARD_REQUIRED ON
)

if( ROCFFT_BUILD_SCOPE )
  set( TESTS_OUT_DIR "/../staging" )
elseif( ROCFFT_CLIENTS_BUILD_SCOPE )
  set( TESTS_OUT_DIR "/../bin" )
else()
  set( TESTS_OUT_DIR "/bin" )
endif()
string( CONCAT TESTS_OUT_DIR "${PROJECT_BINARY_DIR}" ${TESTS_OUT_DIR} )

set_target_properties(rocfft-test
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY 
                      ${TESTS_OUT_DIR})
set_target_properties(rtc_helper_crash
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY 
                      ${TESTS_OUT_DIR})


rocm_install(TARGETS rocfft-test rtc_helper_crash COMPONENT tests)

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
    add_custom_command( TARGET rocfft-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} $<TARGET_FILE_DIR:rocfft-test> )
  endforeach( file_i )
endif()

if( ROCFFT_MPI_ENABLE )
  # normal and dynamic-loading MPI worker processes
  foreach(worker rocfft_mpi_worker dyna_rocfft_mpi_worker)
    add_executable( ${worker} rocfft_mpi_worker.cpp )

    if( BUILD_CLIENTS_TESTS_OPENMP )
      set_target_properties( ${worker} PROPERTIES
         BUILD_RPATH "${BUILD_RPATH}"
      )
      set_target_properties( ${worker} PROPERTIES
        INSTALL_RPATH "${INSTALL_RPATH}"
      )
    endif()
    if( BUILD_FFTW OR NOT FFTW_FOUND )
      add_dependencies( ${worker} fftw_double fftw_single )
    endif()
    target_include_directories( ${worker}
      PRIVATE
      ${CMAKE_BINARY_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../library/include/
      ${MPI_C_INCLUDE_PATH}
      $<BUILD_INTERFACE:${FFTW_INCLUDE_DIRS}>
    )
    target_compile_options( ${worker} PRIVATE ${WARNING_FLAGS} )

    if ( ROCFFT_CRAY_MPI_ENABLE )
      target_link_libraries( ${worker} PRIVATE
        ${ROCFFT_OPENMP_LIBRARY}
        hip::hiprand
        hip::device
        MPI::MPI_CXX
        ${FFTW_LIBRARIES}
        "mpi_gtl_hsa"
      )
        get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY )
        target_link_directories( ${worker}
  	PRIVATE
  	${MPI_LIBDIR}/../../../../gtl/lib )
    else()
      target_link_libraries( ${worker} PRIVATE
        ${ROCFFT_OPENMP_LIBRARY}
        hip::hiprand
        hip::device
        MPI::MPI_CXX
        ${FFTW_LIBRARIES}
      )    
    endif()

    target_compile_definitions( ${worker} PRIVATE USE_HIPRAND )

    set_target_properties(${worker}
                          PROPERTIES 
                          RUNTIME_OUTPUT_DIRECTORY 
                          ${TESTS_OUT_DIR})
    rocm_install(TARGETS ${worker} COMPONENT tests)

  endforeach()

  # link normal MPI worker against rocFFT
  target_link_libraries( rocfft_mpi_worker PRIVATE
    roc::rocfft
  )

  # dyna worker only needs to dynamically load libraries
  target_compile_definitions( dyna_rocfft_mpi_worker PRIVATE ROCFFT_DYNA_MPI_WORKER )
  target_link_libraries( dyna_rocfft_mpi_worker PRIVATE
    ${CMAKE_DL_LIBS}
  )

endif()

set(COVERAGE_TEST_OPTIONS "--smoketest;--gtest_filter=-*call*" CACHE STRING "Command line arguments for rocfft-test when generating a code coverage report")

if(BUILD_CODE_COVERAGE)
  # Coverage won't work in a standalone build of the tests, as we can't
  # guarantee the library was built with coverage enabled
  if( NOT TARGET rocfft )
    message( FATAL_ERROR "BUILD_CODE_COVERAGE requires building from the root of rocFFT" )
  endif()

  add_custom_target(
    code_cov_tests
    DEPENDS rocfft-test rocfft_rtc_helper
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/rocfft-coverage_%p.profraw" GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:rocfft-test> --precompile=rocfft-test-precompile.db ${COVERAGE_TEST_OPTIONS}
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
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/rocfft-coverage_*.profraw -o ./coverage-report/rocfft.profdata
    COMMAND ${LLVM_COV} report -object ./library/src/librocfft.so -instr-profile=./coverage-report/rocfft.profdata
    COMMAND ${LLVM_COV} show -object ./library/src/librocfft.so -instr-profile=./coverage-report/rocfft.profdata -format=html -output-dir=coverage-report
    COMMAND ${LLVM_COV} export -object ./library/src/librocfft.so -instr-profile=./coverage-report/rocfft.profdata -format=lcov > ./coverage-report/coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

endif()
