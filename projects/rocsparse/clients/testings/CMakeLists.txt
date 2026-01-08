# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(ROCSPARSE_BUILD_TESTING OR BUILD_TESTING OR ROCSPARSE_ENABLE_BENCHMARKS)
    # Helper object avoids compiling the same sources twice
    add_library(rocsparse-test-bench-common OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_axpyi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_doti.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dotci.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gthr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gthrz.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_roti.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sctr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrxmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrsv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_coomv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrmv_managed.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrsv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csritsv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_ellmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_hybmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gebsrmv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrmm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gebsrmm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrmm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrsm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrsm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gemmi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrgeam.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrgemm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrgeam.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrgemm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrgemm_reuse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsric0.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrilu0.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csric0.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrilu0.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csritilu0.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csritilu0_ex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gtsv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gtsv_no_pivot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gtsv_no_pivot_strided_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gtsv_interleaved_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gpsv_interleaved_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gebsr2gebsc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gebsr2gebsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2ell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2hyb.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2bsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2gebsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_coo2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_ell2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_hyb2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsr2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gebsr2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2csr_compress.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_prune_csr2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_prune_csr2csr_by_percentage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_identity.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_inverse_permutation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrsort.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_cscsort.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_coosort.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csricsv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrilusv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_nnz.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dense2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dense2coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_prune_dense2csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_prune_dense2csr_by_percentage.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dense2csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csr2dense.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csc2dense.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_coo2dense.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmv_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmv_coo_aos.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmv_bsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmv_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmv_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmv_ell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_coo_aos.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_bsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_ell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_v2_spmv_sell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spsv_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sptrsv_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spitsv_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spsv_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sptrsv_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sptrsm_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sptrsm_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spsm_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spsm_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_bsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_bell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_batched_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_batched_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_batched_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmm_batched_bell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sparse_to_dense_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sparse_to_dense_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sparse_to_dense_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dense_to_sparse_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dense_to_sparse_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dense_to_sparse_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sparse_to_sparse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_extract.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgemm_bsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgemm_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgemm_reuse_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgeam_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgeam_reuse_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgeam_csr_2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spgeam_reuse_csr_2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gemvi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_sddmm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_csrcolor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_csr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_coo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_gebsr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_gebsc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_csc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_ell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_matrix_hyb.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_check_spmat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_bsrpad_value.cpp
    )

    target_compile_features(rocsparse-test-bench-common PRIVATE cxx_std_17)

    target_compile_options(rocsparse-test-bench-common PRIVATE
        -Wno-deprecated
        -Wno-unused-command-line-argument
        -Wall
    )

    target_include_directories(rocsparse-test-bench-common PRIVATE
        ${PROJECT_SOURCE_DIR}/clients/include
    )

    target_link_libraries(rocsparse-test-bench-common PRIVATE
        roc::rocsparse
        hip::host
        hip::device
    )

    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(rocsparse-test-bench-common PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(ROCSPARSE_ENABLE_MEMSTAT)
        target_compile_definitions(rocsparse-test-bench-common PRIVATE ROCSPARSE_WITH_MEMSTAT)
    endif()

    if(ROCSPARSE_ENABLE_ILP64)
        target_compile_definitions(rocsparse-test-bench-common PRIVATE rocsparse_ILP64)
    endif()

    if(ROCSPARSE_ENABLE_ASAN)
        rocsparse_target_configure_sanitizers(rocsparse-test-bench-common PRIVATE)
    endif()

    if(ROCSPARSE_ENABLE_MARKER)
        target_compile_definitions(rocsparse-test-bench-common PRIVATE ROCSPARSE_CLIENTS_WITH_ROCTX)
        target_include_directories(rocsparse-test-bench-common PRIVATE ${ROCTRACER_INCLUDE_DIR})
        target_link_libraries(rocsparse-test-bench-common PRIVATE ${rocTracer})
    endif()
endif()

if(ROCSPARSE_BUILD_TESTING OR BUILD_TESTING)
    target_link_libraries(rocsparse-test PRIVATE rocsparse-test-bench-common)
    target_sources(rocsparse-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_axpby.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_gather.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_scatter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_rot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spvec_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spmat_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dnvec_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_dnmat_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_const_spvec_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_const_spmat_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_const_dnvec_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_const_dnmat_descr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_spvv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testing_copy_info.cpp
    )
endif()

if(ROCSPARSE_ENABLE_BENCHMARKS)
    target_link_libraries(rocsparse-bench PRIVATE rocsparse-test-bench-common)
endif()
