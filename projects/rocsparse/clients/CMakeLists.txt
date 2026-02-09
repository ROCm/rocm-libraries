# ########################################################################
# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

function( apply_omp_settings lib_target_ )
  if(NOT WIN32)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::OpenMP_CXX)
      set_target_properties( ${lib_target_} PROPERTIES
        BUILD_RPATH "${HIP_CLANG_ROOT}/lib"
      )
      set_target_properties( ${lib_target_} PROPERTIES
        INSTALL_RPATH "$ORIGIN/../llvm/lib"
      )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
      set_target_properties( ${lib_target_} PROPERTIES
        BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}"
      )
      set_target_properties( ${lib_target_} PROPERTIES
        INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}"
      )
    endif()
  endif()
endfunction()

if (rocblas_FOUND)
  message("Build rocSPARSE client with rocBLAS")
  add_compile_options(-DROCSPARSE_WITH_ROCBLAS)
endif()

if (BUILD_CLIENTS_ONLY)
  find_package(rocsparse 4.0.0 REQUIRED)
endif()

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
  set( CMAKE_Fortran_COMPILER "gfortran" )
endif()

# This project may compile dependencies for clients
if ( BUILD_FORTRAN_CLIENTS )
  set( fortran_language "Fortran" )
endif()

# This project may compile dependencies for clients
project(rocsparse-clients LANGUAGES CXX ${fortran_language})

if ( BUILD_FORTRAN_CLIENTS )
  set(rocsparse_fortran_source
      ../library/src/rocsparse_enums.f90
      ../library/src/rocsparse.f90
  )

  # Set Fortran module output directory
  set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/rocsparse)

  # Create rocBLAS Fortran module
  add_library(rocsparse_fortran OBJECT ${rocsparse_fortran_source})

endif()

# OpenMP
if(NOT WIN32)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Look for openmp config in ROCm install to populate openmp_LIB_DIR and openmp_LIB_INSTALL_DIR
    find_package(OpenMP CONFIG PATHS "${HIP_CLANG_ROOT}/lib/cmake")
  endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
    set(ROCSPARSE_CLIENT_LIBS "OpenMP::omp")
    message(STATUS "Found openmp-config.cmake at ${OpenMP_DIR}")
  else()
    # if it fails to find OpenMP compile and link flags in strange configurations it can just use non-parallel reference computation
    # if there is no omp.h to find the client compilation will fail and this should be obvious, used to be REQUIRED
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
      set(ROCSPARSE_CLIENT_LIBS "OpenMP::OpenMP_CXX")
    endif()
  endif()
else()
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(ROCSPARSE_CLIENT_LIBS "libomp")
  endif()
endif()

# Determine if CXX Compiler is hip-clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using hip-clang to build for amdgpu backend")
else()
  message(FATAL_ERROR "'hipcc' or 'amdclang++' compiler required to compile for ROCm platform.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET rocsparse)
  find_package(rocsparse REQUIRED CONFIG PATHS /opt/rocm/rocsparse)

  option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
  option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" OFF)
  option(BUILD_CLIENTS_SAMPLES "Build examples" ON)
endif()

# HIP headers required of all clients; clients use hip to allocate device memory
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(HIP REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})

# Set CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CXX standard
if (CMAKE_CXX_STANDARD EQUAL 14)
  message( DEPRECATION "Builds using the C++14 standard will no longer be supported in the next major release" )
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

#
# Build tools
#
add_subdirectory(tools)

if(BUILD_CLIENTS_TESTS)
  enable_testing()
  add_subdirectory(tests)

  if(NOT CONVERT_SOURCE)
    set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp)
  endif()

  ## determine separator for rpath argument
  if("${CMAKE_CXX_COMPILER}" MATCHES ".*amdclang")
  set(rpath_sep "=")
  else()
  # default to old behaviour
  set(rpath_sep ",")
  endif()
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -O3 -Wl,--build-id=sha1 -o ${PROJECT_BINARY_DIR}/mtx2csr.exe RESULT_VARIABLE STATUS)
  if(STATUS AND NOT STATUS EQUAL 0)
    message(FATAL_ERROR "mtx2csr.exe failed to build, aborting.")
  endif()

  set(ROCSPARSE_CLIENTMATRICES "${CMAKE_SOURCE_DIR}/cmake/rocsparse_clientmatrices.cmake")

  set(ROCSPARSE_CONVERT "${PROJECT_BINARY_DIR}/rocsparse_mtx2csr")
  add_custom_command(OUTPUT "${ROCSPARSE_CONVERT}"
                    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/mtx2csr.exe" "${ROCSPARSE_CONVERT}"
                    DEPENDS "${PROJECT_BINARY_DIR}/mtx2csr.exe"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCSPARSE_COMMON "${PROJECT_BINARY_DIR}/staging/rocsparse_common.yaml")
  add_custom_command(OUTPUT "${ROCSPARSE_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparse_common.yaml "${ROCSPARSE_COMMON}"
                    DEPENDS include/rocsparse_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCSPARSE_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocsparse_template.yaml")
  add_custom_command(OUTPUT "${ROCSPARSE_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparse_template.yaml "${ROCSPARSE_TEMPLATE}"
                    DEPENDS include/rocsparse_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCSPARSE_SMOKE "${PROJECT_BINARY_DIR}/staging/rocsparse_smoke.yaml")
  set(ROCSPARSE_REGRESSION "${PROJECT_BINARY_DIR}/staging/rocsparse_regression.yaml")
  set(ROCSPARSE_EXTENDED "${PROJECT_BINARY_DIR}/staging/rocsparse_extended.yaml")
  add_custom_command(OUTPUT "${ROCSPARSE_SMOKE}" "${ROCSPARSE_REGRESSION}" "${ROCSPARSE_EXTENDED}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparse_smoke.yaml "${ROCSPARSE_SMOKE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparse_regression.yaml "${ROCSPARSE_REGRESSION}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocsparse_extended.yaml "${ROCSPARSE_EXTENDED}"
                    DEPENDS include/rocsparse_smoke.yaml include/rocsparse_regression.yaml include/rocsparse_extended.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(ROCSPARSE_GENTEST "${PROJECT_BINARY_DIR}/staging/rocsparse_gentest.py")
  add_custom_command(OUTPUT "${ROCSPARSE_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/rocsparse_gentest.py "${ROCSPARSE_GENTEST}"
                    DEPENDS common/rocsparse_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target(rocsparse-common DEPENDS "${ROCSPARSE_COMMON}" "${ROCSPARSE_TEMPLATE}" "${ROCSPARSE_SMOKE}" "${ROCSPARSE_GENTEST}" "${ROCSPARSE_CONVERT}")

  rocm_install(
    FILES "${ROCSPARSE_COMMON}" "${ROCSPARSE_TEMPLATE}" "${ROCSPARSE_SMOKE}"
    COMPONENT clients-common
    DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsparse/test"
    )
  #
  rocm_install(
    PROGRAMS "${ROCSPARSE_GENTEST}"
    COMPONENT clients-common
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/rocsparse/test"
    )
  #
  rocm_install(
    PROGRAMS "${ROCSPARSE_CONVERT}"
    COMPONENT clients-common
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  #
  rocm_install(
    FILES "${ROCSPARSE_CLIENTMATRICES}"
    COMPONENT clients-common
    DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsparse/test"
    )

endif()
