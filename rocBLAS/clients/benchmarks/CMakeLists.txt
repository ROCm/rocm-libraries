# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

set(rocblas_bench_source
  client.cpp
  )

add_executable( rocblas-bench ${rocblas_bench_source} )

rocblas_client_library_settings( rocblas-bench )


if( BUILD_WITH_TENSILE )
  set(rocblas_gemm_tune_source
    gemm_tune/gemm_tune_client.cpp
    gemm_tune/gemm_tuners.cpp
    )

  add_executable( rocblas-gemm-tune ${rocblas_gemm_tune_source} )

  target_include_directories( rocblas-gemm-tune
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
  )
  target_include_directories( rocblas-gemm-tune
    SYSTEM BEFORE PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
  )
endif()

target_link_libraries( rocblas-bench PRIVATE rocblas_clients_testing_common rocblas_clients_common )
if( BUILD_FORTRAN_CLIENTS )
  target_link_libraries( rocblas-bench PRIVATE rocblas_fortran_client )
endif( )
target_link_libraries( rocblas-bench PRIVATE roc::rocblas hip::host hip::device ${BLAS_LIBRARY} ${GTEST_BOTH_LIBRARIES} )

if( BUILD_WITH_TENSILE )
  target_link_libraries( rocblas-gemm-tune PRIVATE rocblas_clients_testing_common rocblas_clients_common )
  if( BUILD_FORTRAN_CLIENTS )
    target_link_libraries( rocblas-gemm-tune PRIVATE rocblas_fortran_client )
  endif( )
  target_link_libraries( rocblas-gemm-tune PRIVATE roc::rocblas hip::host hip::device ${BLAS_LIBRARY} ${GTEST_BOTH_LIBRARIES} )
endif()

if (NOT WIN32)
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs" )
endif()


target_link_libraries( rocblas-bench PRIVATE ${COMMON_LINK_LIBS} )
if( BUILD_WITH_TENSILE )
  target_link_libraries( rocblas-gemm-tune PRIVATE ${COMMON_LINK_LIBS} )
endif()

target_compile_options(rocblas-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
if( BUILD_WITH_TENSILE )
  target_compile_options(rocblas-gemm-tune PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)
endif()

target_compile_definitions( rocblas-bench PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS ${TENSILE_DEFINES} GOOGLE_TEST ${BLIS_DEFINES} )
if( BUILD_WITH_TENSILE )
  target_compile_definitions( rocblas-gemm-tune PRIVATE ROCBLAS_BENCH ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS ${TENSILE_DEFINES} )
endif()

if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( rocblas-bench PRIVATE CLIENTS_NO_FORTRAN )
  if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocblas-gemm-tune PRIVATE CLIENTS_NO_FORTRAN )
  endif()
endif()

set_target_properties( rocblas-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
if( BUILD_WITH_TENSILE )
  set_target_properties( rocblas-gemm-tune PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endif()

add_dependencies( rocblas-bench rocblas-common )
if( BUILD_WITH_TENSILE )
  add_dependencies( rocblas-gemm-tune rocblas-common )
endif()

add_subdirectory ( ./perf_script )

rocm_install(TARGETS rocblas-bench COMPONENT benchmarks)
if( BUILD_WITH_TENSILE )
  rocm_install(TARGETS rocblas-gemm-tune COMPONENT benchmarks)
endif()
