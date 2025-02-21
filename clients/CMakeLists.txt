# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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
# ########################################################################

cmake_minimum_required( VERSION 3.16.8 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( hipsparselt-clients LANGUAGES CXX C Fortran )


if( SKIP_LIBRARY )
  include_directories(${HIPSPARSELT_LIBRARY_DIR}/include)
else()
  include_directories(${CMAKE_BINARY_DIR}/include)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# if it fails to find OpenMP compile and link flags in strange configurations it can just use non-parallel reference computation
# if there is no omp.h to find the client compilation will fail and this should be obvious, used to be REQUIRED
find_package(OpenMP)

if (TARGET OpenMP::OpenMP_CXX)
  set( COMMON_LINK_LIBS "OpenMP::OpenMP_CXX")
  list( APPEND COMMON_LINK_LIBS "-L${HIP_CLANG_ROOT}/lib;-Wl,-rpath=${HIP_CLANG_ROOT}/lib")
endif()

if (TARGET Threads::Threads)
  list( APPEND COMMON_LINK_LIBS "Threads::Threads")
endif()

message(STATUS "CLIENT COMMON CXX_OPTIONS: ${COMMON_CXX_OPTIONS}")
message(STATUS "CLIENT COMMON LINK: ${COMMON_LINK_LIBS}")

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

#include( build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( NOT TARGET hipsparselt )
  find_package( hipsparselt REQUIRED CONFIG PATHS ${ROCM_PATH}/hipsparselt /opt/rocm/hipsparselt ${HIPSPARSELT_LIBRARY_DIR})
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} /opt/rocm )
if ( NOT hip_FOUND )
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} )
endif( )


if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)

  # Linking lapack library requires fortran flags
  find_package( cblas REQUIRED CONFIG )
  if (LINK_BLIS)
    set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
    set( BLIS_CPP ../common/blis_interface.cpp )
    set( BLAS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.a )
  else()
    set( BLAS_LIBRARY "blas" )
  endif()

  # common source files used in subdirectories benchmarks and gtest thus ../common
  set( hipsparselt_test_bench_common
      ../common/singletons.cpp
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/argument_model.cpp
      ../common/hipsparselt_parse_data.cpp
      ../common/hipsparselt_arguments.cpp
      ../common/hipsparselt_random.cpp
      ${BLIS_CPP}
    )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )

endif()


set( HIPSPARSELT_COMMON "${PROJECT_BINARY_DIR}/staging/hipsparselt_common.yaml")
add_custom_command( OUTPUT "${HIPSPARSELT_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipsparselt_common.yaml "${HIPSPARSELT_COMMON}"
                    DEPENDS include/hipsparselt_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPSPARSELT_TEMPLATE "${PROJECT_BINARY_DIR}/staging/hipsparselt_template.yaml")
add_custom_command( OUTPUT "${HIPSPARSELT_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipsparselt_template.yaml "${HIPSPARSELT_TEMPLATE}"
                    DEPENDS include/hipsparselt_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPSPARSELT_GENTEST "${PROJECT_BINARY_DIR}/staging/hipsparselt_gentest.py")
add_custom_command( OUTPUT "${HIPSPARSELT_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/hipsparselt_gentest.py "${HIPSPARSELT_GENTEST}"
                    DEPENDS common/hipsparselt_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

add_custom_target( hipsparselt-common DEPENDS "${HIPSPARSELT_COMMON}" "${HIPSPARSELT_TEMPLATE}" "${HIPSPARSELT_GENTEST}" )


rocm_install(
  FILES ${HIPSPARSELT_COMMON} ${HIPSPARSELT_TEMPLATE}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clients-common
)

rocm_install(
  PROGRAMS ${HIPSPARSELT_GENTEST}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clients-common
)
