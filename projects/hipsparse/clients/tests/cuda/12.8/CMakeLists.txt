# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(HIPSPARSE_TEST_SOURCES
    # Level 2
    test_bsrmv.cpp
    test_bsrsv2.cpp
    test_bsrxmv.cpp
    test_gemvi.cpp
    # Level 3
    test_bsrmm.cpp
    test_bsrsm2.cpp
    # Precond  
    test_bsric02.cpp
    test_bsrilu02.cpp
    test_csric02.cpp
    test_csrilu02.cpp
    test_gpsv_interleaved_batch.cpp
    test_gtsv.cpp
    test_gtsv2_nopivot.cpp
    test_gtsv2_strided_batch.cpp
    test_gtsv_interleaved_batch.cpp
    # Extra
    test_csrgeam2.cpp
    # Reorder
    test_csrcolor.cpp
    # Conversion
    test_bsr2csr.cpp
    test_coo2csr.cpp
    test_coosort.cpp
    test_cscsort.cpp
    test_csr2bsr.cpp
    test_csr2coo.cpp
    test_csr2gebsr.cpp
    test_csrsort.cpp
    test_csru2csr.cpp
    test_gebsr2csr.cpp
    test_gebsr2gebsc.cpp
    test_gebsr2gebsr.cpp
    test_identity.cpp
    test_nnz.cpp
    test_prune_csr2csr.cpp
    test_prune_csr2csr_by_percentage.cpp
    test_prune_dense2csr.cpp
    test_prune_dense2csr_by_percentage.cpp
    # Generic
    test_axpby.cpp
    test_const_dnmat_descr.cpp
    test_const_dnvec_descr.cpp
    test_const_spmat_descr.cpp
    test_const_spvec_descr.cpp
    test_dense_to_sparse_coo.cpp
    test_dense_to_sparse_csc.cpp
    test_dense_to_sparse_csr.cpp
    test_dnmat_descr.cpp
    test_dnvec_descr.cpp
    test_gather.cpp
    test_rot.cpp
    test_scatter.cpp
    test_sddmm_coo.cpp
    test_sddmm_coo_aos.cpp
    test_sddmm_csc.cpp
    test_sddmm_csr.cpp
    test_sparse_to_dense_coo.cpp
    test_sparse_to_dense_csc.cpp
    test_sparse_to_dense_csr.cpp
    test_spgemm_csr.cpp
    test_spgemmreuse_csr.cpp
    test_spmat_descr.cpp
    test_spmm_batched_coo.cpp
    test_spmm_batched_csc.cpp
    test_spmm_batched_csr.cpp
    test_spmm_bell.cpp
    test_spmm_coo.cpp
    test_spmm_csc.cpp
    test_spmm_csr.cpp
    test_spmm_bsr.cpp
    test_spmv_coo.cpp
    test_spmv_coo_aos.cpp
    test_spmv_csr.cpp
    test_spmv_csr_reuse_descr.cpp
    test_spmv_sell.cpp
    test_spsm_coo.cpp
    test_spsm_csr.cpp
    test_spsv_coo.cpp
    test_spsv_csr.cpp
    test_spvec_descr.cpp
    test_spvv.cpp
)

set(HIPSPARSE_TEST_YAML_INCLUDE cuda/12.8/hipsparse_test.yaml)

set(HIPSPARSE_TEST_YAMLS 
    # Level 2
    cuda/12.8/test_bsrmv.yaml
    cuda/12.8/test_bsrxmv.yaml
    cuda/12.8/test_bsrsv2.yaml
    cuda/12.8/test_gemvi.yaml
    # Level 3
    cuda/12.8/test_bsrmm.yaml
    cuda/12.8/test_bsrsm2.yaml
    # Precond
    cuda/12.8/test_bsric02.yaml
    cuda/12.8/test_bsrilu02.yaml
    cuda/12.8/test_csric02.yaml
    cuda/12.8/test_csrilu02.yaml
    cuda/12.8/test_gpsv_interleaved_batch.yaml
    cuda/12.8/test_gtsv.yaml
    cuda/12.8/test_gtsv2_nopivot.yaml
    cuda/12.8/test_gtsv2_strided_batch.yaml
    cuda/12.8/test_gtsv_interleaved_batch.yaml
    # Extra
    cuda/12.8/test_csrgeam2.yaml
    # Reorder
    cuda/12.8/test_csrcolor.yaml
    # Conversion
    cuda/12.8/test_bsr2csr.yaml
    cuda/12.8/test_coo2csr.yaml
    cuda/12.8/test_coosort.yaml
    cuda/12.8/test_cscsort.yaml
    cuda/12.8/test_csr2bsr.yaml
    cuda/12.8/test_csr2coo.yaml
    cuda/12.8/test_csr2csc.yaml
    cuda/12.8/test_csr2csc_ex2.yaml
    cuda/12.8/test_csr2gebsr.yaml
    cuda/12.8/test_csrsort.yaml
    cuda/12.8/test_csru2csr.yaml
    cuda/12.8/test_gebsr2csr.yaml
    cuda/12.8/test_gebsr2gebsc.yaml
    cuda/12.8/test_gebsr2gebsr.yaml
    cuda/12.8/test_identity.yaml
    cuda/12.8/test_nnz.yaml
    cuda/12.8/test_prune_csr2csr.yaml
    cuda/12.8/test_prune_csr2csr_by_percentage.yaml
    cuda/12.8/test_prune_dense2csr.yaml
    cuda/12.8/test_prune_dense2csr_by_percentage.yaml
    # Generic
    cuda/12.8/test_axpby.yaml
    cuda/12.8/test_dense_to_sparse_coo.yaml
    cuda/12.8/test_dense_to_sparse_csc.yaml
    cuda/12.8/test_dense_to_sparse_csr.yaml
    cuda/12.8/test_gather.yaml
    cuda/12.8/test_rot.yaml
    cuda/12.8/test_scatter.yaml
    cuda/12.8/test_sddmm_coo.yaml
    cuda/12.8/test_sddmm_coo_aos.yaml
    cuda/12.8/test_sddmm_csc.yaml
    cuda/12.8/test_sddmm_csr.yaml
    cuda/12.8/test_sparse_to_dense_coo.yaml
    cuda/12.8/test_sparse_to_dense_csc.yaml
    cuda/12.8/test_sparse_to_dense_csr.yaml
    cuda/12.8/test_spgemm_csr.yaml
    cuda/12.8/test_spgemmreuse_csr.yaml
    cuda/12.8/test_spmm_batched_coo.yaml
    cuda/12.8/test_spmm_batched_csc.yaml
    cuda/12.8/test_spmm_batched_csr.yaml
    cuda/12.8/test_spmm_coo.yaml
    cuda/12.8/test_spmm_csc.yaml
    cuda/12.8/test_spmm_csr.yaml
    cuda/12.8/test_spmm_bsr.yaml
    cuda/12.8/test_spmv_coo.yaml
    cuda/12.8/test_spmv_coo_aos.yaml
    cuda/12.8/test_spmv_csr.yaml
    cuda/12.8/test_spmv_csr_reuse_descr.yaml
    cuda/12.8/test_spmv_sell.yaml
    cuda/12.8/test_spsm_coo.yaml
    cuda/12.8/test_spsm_csr.yaml
    cuda/12.8/test_spsv_coo.yaml
    cuda/12.8/test_spsv_csr.yaml
    cuda/12.8/test_spvv.yaml)
