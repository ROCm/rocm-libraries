# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/gtest/src)

if(ROCBLAS_ENABLE_TENSILE)
  target_sources(
    rocblas-test
    PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../multiheaded_gtest.cpp"
            "${_CMAKE_CURRENT_SOURCE_DIR}/../atomics_mode_gtest.cpp"
            "${_CMAKE_CURRENT_SOURCE_DIR}/../get_solutions_gtest.cpp")
endif()

target_sources(
  rocblas-test
  PRIVATE # general
          "${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_gtest_main.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_test.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../general_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../set_get_pointer_mode_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../set_get_atomics_mode_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../logging_mode_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../ostream_threadsafety_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../set_get_vector_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../set_get_matrix_gtest.cpp"
          # blas1
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/asum_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/axpy_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/copy_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/dot_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/iamaxmin_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/nrm2_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/rot_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/scal_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas1/swap_gtest.cpp"
          # blas1_ex
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/axpy_ex_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/dot_ex_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/nrm2_ex_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/rot_ex_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/scal_ex_gtest.cpp"
          # blas2
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/trsv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/gbmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/gemv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/hbmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/hemv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/her_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/her2_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/hpmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/hpr_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/hpr2_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/trmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/tpmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/tbmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/tbsv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/tpsv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/ger_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/geru_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/gerc_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/spr_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/spr2_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/syr_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/syr2_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/sbmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/spmv_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas2/symv_gtest.cpp"
          # blas3 may use tensile or source gemm
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/gemm_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/gemm_ex_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/symm_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/hemm_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/trsm_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/trtri_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/trmm_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/syrk_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/syrkx_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/syr2k_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/herk_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/herkx_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/her2k_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/dgmm_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas3/geam_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/gemmt_gtest.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/geam_ex_gtest.cpp")
