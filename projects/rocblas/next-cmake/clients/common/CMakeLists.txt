# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/common)

find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
# TODO: Review the usage of these libraries, and whether they are needed
find_package(BLAS REQUIRED)
find_package(cblas REQUIRED)
find_package(LAPACK REQUIRED)

add_library(rocblas-clients-common OBJECT)

if(WARN_NOT_ILP64_PREFERRED)
  target_compile_definitions(rocblas-clients-common
                             PRIVATE ROCBLAS_REFERENCE_LIB_WARN)
endif()
target_compile_definitions(
  rocblas-clients-common
  PUBLIC ROCBLAS_REFERENCE_LIB=${BLAS_LIBRARY} ROCM_USE_FLOAT16
         ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS
         ROCBLAS_TENSILE_SEPARATE_ARCH=1 ROCBLAS_TENSILE_LAZY_LOAD=1
  PRIVATE GOOGLE_TEST)

target_include_directories(
  rocblas-clients-common
  PUBLIC
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    # Do not include /blas3, /blas2, /blas1 as they are not needed
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../../library/src>)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas-clients-common PRIVATE)
  target_compile_definitions(rocblas-clients-common PRIVATE ASAN_BUILD)
endif()

if(rocm_smi_FOUND)
  target_link_libraries(rocblas-clients-common PUBLIC rocm_smi64)
endif()
target_link_libraries(
  rocblas-clients-common
  PUBLIC roc::rocblas
         hip::host
         hip::device
         BLIS::BLIS
         OpenMP::OpenMP_CXX
         Threads::Threads
         GTest::gtest)

if(ROCBLAS_ENABLE_FORTRAN)
  add_library(rocblas-fortran-client OBJECT)
  add_library(rocblas::rocblas-fortran-client ALIAS rocblas-fortran-client)
  if(ROCBLAS_ENABLE_TENSILE)
    target_sources(
      rocblas-fortran-client
      PRIVATE
        ${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90)
  endif()

  target_sources(
    rocblas-fortran-client
    PRIVATE
      ${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90)

  target_link_libraries(rocblas-fortran-client
    PRIVATE
      rocblas::rocblas-fortran-module)

  # Make the object library available to the main client library
  set_target_properties(rocblas-fortran-client
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(rocblas-clients-common
    PRIVATE
      rocblas-fortran-client)
else()
  target_compile_definitions(rocblas-clients-common PUBLIC CLIENTS_NO_FORTRAN)
endif()

add_subdirectory(include)
add_subdirectory(src)
