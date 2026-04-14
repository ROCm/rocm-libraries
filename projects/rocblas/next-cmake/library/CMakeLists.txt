# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEMP_HOST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../library)
set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_HOST_SOURCE_DIR})

add_library(rocblas-interface INTERFACE)

target_include_directories(
  rocblas-interface
  INTERFACE
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
    # This is marked 'internal', but needs to be public for the clients
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    # TODO: Below include may not be required?
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/include/internal>
    $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/src/src64>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(
  rocblas-interface INTERFACE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API
                              ROCBLAS_BETA_FEATURES_API)

# NOTE: We assume separate architectures and lazy library loading are always
# enabled.
if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(
    rocblas-interface
    INTERFACE BUILD_WITH_TENSILE ROCBLAS_TENSILE_SEPARATE_ARCH=1
              ROCBLAS_TENSILE_LAZY_LOAD=1)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
  target_link_libraries(rocblas-interface INTERFACE roc::hipblaslt)
  target_compile_definitions(rocblas-interface INTERFACE BUILD_WITH_HIPBLASLT)
endif()

target_compile_features(rocblas-interface INTERFACE cxx_std_17)

target_link_libraries(rocblas-interface INTERFACE hip::host hip::device)

add_library(rocblas-helper OBJECT)

target_link_libraries(rocblas-helper PRIVATE roc::tensile-host rocblas-interface hip::host
                                             hip::device)
target_compile_features(rocblas-helper PRIVATE cxx_std_17)

if(ROCBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
  set_target_properties(rocblas-helper PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(rocblas64 OBJECT)

if(ROCBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
  set_target_properties(rocblas64 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_definitions(rocblas64 PRIVATE ROCBLAS_INTERNAL_ILP64)

target_link_libraries(rocblas64 PUBLIC rocblas-helper PRIVATE rocblas-interface)

if(ROCBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
  add_library(rocblas SHARED)
else()
  add_library(rocblas STATIC)
endif()

rocm_set_soversion(rocblas ${rocblas_SOVERSION})
add_library(roc::rocblas ALIAS rocblas)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas PRIVATE)
  rocblas_target_configure_sanitizers(rocblas64 PRIVATE)
endif()


target_include_directories(
  rocblas
  PUBLIC $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
         # This is marked 'internal', but needs to be public for the clients
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/internal>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         # TODO: Below include may not be required?
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas PRIVATE)
endif()

set_target_properties(rocblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                         ${PROJECT_BINARY_DIR}/staging)

if(ROCBLAS_ENABLE_MARKER)
  target_link_libraries(rocblas PRIVATE ${ROCTX64_LIBRARY})
endif()

target_link_libraries(rocblas PRIVATE rocblas-helper roc::tensile-host rocblas-interface rocblas64)

if(ROCBLAS_ENABLE_FORTRAN)
  add_library(rocblas-fortran-module OBJECT)
  add_library(rocblas::rocblas-fortran-module ALIAS rocblas-fortran-module)
  target_sources(
    rocblas-fortran-module
    PRIVATE
      ${_CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_module.f90
  )
  set_target_properties(rocblas-fortran-module PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
    POSITION_INDEPENDENT_CODE ON)

  target_include_directories(rocblas-fortran-module
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>)
endif()

add_subdirectory(include)
add_subdirectory(src)

# TODO: Fix the source code that uses inconsistent include paths and then remove
# the `configure_file` calls below. This is improper practice and allows clients
# to use wrong (or at least inconsistent) include paths. Refer to file:
# testing_gemm_batched_ex.hpp, which uses the following include
# ~~~
# #include <rocblas/internal/rocblas-beta.h>`
# ~~~
configure_file("${_CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-beta.h"
               "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal" COPYONLY)

configure_file("${_CMAKE_CURRENT_SOURCE_DIR}/include/rocblas.h"
               "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas" COPYONLY)

configure_file("${_CMAKE_CURRENT_SOURCE_DIR}/src/include/rocblas_device_malloc.hpp"
               "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas" COPYONLY)
