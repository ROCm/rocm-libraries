# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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


# ########################################################################
# standardized common object library settings
# ########################################################################
function( rocblas_client_library_settings lib_target_ )

  message(STATUS "** rocblas_client_library_settings: ${lib_target_}")

  target_include_directories( ${lib_target_}
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
  )

  # External header includes included as system files
  target_include_directories( ${lib_target_}
    SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
  )

  target_include_directories( ${lib_target_}
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/common>
  )

  target_compile_definitions( ${lib_target_} PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS )
  target_compile_definitions( ${lib_target_} PRIVATE ${TENSILE_DEFINES} ${HIPBLASLT_DEFINES} GOOGLE_TEST )
  # GOOGLE_TEST left off for reuse of this function by by gemm tuner
  if ( NOT BUILD_FORTRAN_CLIENTS )
    target_compile_definitions( ${lib_target_} PRIVATE CLIENTS_NO_FORTRAN )
  endif()

  # We widely use extensions and AMD Clang warnings are too noisy
  target_compile_options( ${lib_target_} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++17-extensions> )
  target_compile_options( ${lib_target_} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}> )

  if(BUILD_ADDRESS_SANITIZER)
    target_compile_definitions( ${lib_target_} PRIVATE ASAN_BUILD )
  endif()
endfunction()

set(rocblas_common_source
    # general
    singletons.cpp
    client_omp.cpp
    client_utility.cpp
    frequency_monitor.cpp
    rocblas_arguments.cpp
    argument_model.cpp
    rocblas_random.cpp
    rocblas_parse_data.cpp
    host_alloc.cpp
    gtest_helpers.cpp
    rocblas_matrix.cpp
)

set(rocblas_testing_common_source
    # general
    cblas_interface.cpp
    ${BLIS_CPP}
    # blas1
    blas1/common_asum.cpp
    blas1/common_axpy.cpp
    blas1/common_copy.cpp
    blas1/common_dot.cpp
    blas1/common_iamax_iamin.cpp
    blas1/common_nrm2.cpp
    blas1/common_rot.cpp
    blas1/common_scal.cpp
    blas1/common_swap.cpp
    # blas1_ex
    blas_ex/common_axpy_ex.cpp
    blas_ex/common_dot_ex.cpp
    blas_ex/common_nrm2_ex.cpp
    blas_ex/common_rot_ex.cpp
    blas_ex/common_scal_ex.cpp
    # blas2
    blas2/common_trsv.cpp
    blas2/common_gbmv.cpp
    blas2/common_gemv.cpp
    blas2/common_hbmv.cpp
    blas2/common_hemv.cpp
    blas2/common_her.cpp
    blas2/common_her2.cpp
    blas2/common_hpmv.cpp
    blas2/common_hpr.cpp
    blas2/common_hpr2.cpp
    blas2/common_trmv.cpp
    blas2/common_tpmv.cpp
    blas2/common_tbmv.cpp
    blas2/common_tbsv.cpp
    blas2/common_tpsv.cpp
    blas2/common_ger.cpp
    blas2/common_spr.cpp
    blas2/common_spr2.cpp
    blas2/common_syr.cpp
    blas2/common_syr2.cpp
    blas2/common_sbmv.cpp
    blas2/common_spmv.cpp
    blas2/common_symv.cpp
    # blas3 may use tensile or source gemm
    blas3/common_gemm.cpp
    blas_ex/common_gemm_ex.cpp
    blas_ex/common_trsm_ex.cpp
    blas3/common_symm_hemm.cpp
    blas3/common_trsm.cpp
    blas3/common_trtri.cpp
    blas3/common_trmm.cpp
    blas3/common_syrk.cpp
    blas3/common_syr2k.cpp # syrkx
    blas3/common_herk.cpp
    blas3/common_her2k.cpp # herkx
    blas3/common_dgmm.cpp
    blas3/common_geam.cpp
    blas_ex/common_gemmt.cpp
    blas_ex/common_geam_ex.cpp
)

add_library(rocblas_clients_common OBJECT ${rocblas_testing_common_tensile_source} ${rocblas_common_source})

rocblas_client_library_settings( rocblas_clients_common )

add_library(rocblas_clients_testing_common OBJECT ${rocblas_testing_common_source})

rocblas_client_library_settings( rocblas_clients_testing_common )

if ( WARN_NOT_ILP64_PREFERRED )
  target_compile_definitions( rocblas_clients_common PRIVATE ROCBLAS_REFERENCE_LIB_WARN )
endif()
target_compile_definitions( rocblas_clients_common PRIVATE ROCBLAS_REFERENCE_LIB=${BLAS_LIBRARY} )


# target is not linked as OBJECT library but we want any interface flags from dependencies, e.g. COMMON_LINK_LIBS may bring OpenMP
target_link_libraries( rocblas_clients_common PUBLIC roc::rocblas hip::host hip::device $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest> $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main> ${COMMON_LINK_LIBS} )
if(BUILD_WITH_HIPBLASLT)
    target_link_libraries( rocblas_clients_common PRIVATE roc::hipblaslt )
endif()
target_link_libraries( rocblas_clients_testing_common roc::rocblas hip::host hip::device ${BLAS_LIBRARY} $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest> $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main> ${COMMON_LINK_LIBS} )
