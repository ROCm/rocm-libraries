# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(HIPSPARSE_TEST_SOURCES
    # Level 1
    test_axpyi.cpp
    test_gthr.cpp
    test_gthrz.cpp
    test_roti.cpp
    test_sctr.cpp
    test_doti.cpp
    test_dotci.cpp
    # Level 2
    test_bsrmv.cpp
    test_bsrsv2.cpp
    test_bsrxmv.cpp
    test_csrsv2.cpp
    test_csrmv.cpp
    test_gemvi.cpp
    test_hybmv.cpp
    # Level 3
    test_bsrmm.cpp
    test_bsrsm2.cpp
    test_csrmm.cpp
    test_csrsm2.cpp
    test_gemmi.cpp
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
    test_csrgeam.cpp
    test_csrgeam2.cpp
    test_csrgemm.cpp
    test_csrgemm2_a.cpp
    test_csrgemm2_b.cpp
    # Reorder
    test_csrcolor.cpp
    # Conversion
    test_bsr2csr.cpp
    test_coo2csr.cpp
    test_coosort.cpp
    test_csc2dense.cpp
    test_cscsort.cpp
    test_csr2bsr.cpp
    test_csr2coo.cpp
    test_csr2csc.cpp
    test_csr2csc_ex2.cpp
    test_csr2csr_compress.cpp
    test_csr2dense.cpp
    test_csr2gebsr.cpp
    test_csr2hyb.cpp
    test_csrilusv.cpp
    test_csrsort.cpp
    test_csru2csr.cpp
    test_dense2csc.cpp
    test_dense2csr.cpp
    test_gebsr2csr.cpp
    test_gebsr2gebsc.cpp
    test_gebsr2gebsr.cpp
    test_hyb2csr.cpp
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
    test_spmm_bsr.cpp
    test_spmm_coo.cpp
    test_spmm_csc.cpp
    test_spmm_csr.cpp
    test_spmv_coo.cpp
    test_spmv_coo_aos.cpp
    test_spmv_csr.cpp
    test_spmv_csr_reuse_descr.cpp
    test_spmv_csr_pytorch_compat.cpp
    test_spmv_sell.cpp
    test_spsm_coo.cpp
    test_spsm_csr.cpp
    test_spsv_coo.cpp
    test_spsv_csr.cpp
    test_spvec_descr.cpp
    test_spvv.cpp
)

set(HIPSPARSE_TEST_YAML_INCLUDE rocm/hipsparse_test.yaml)

set(HIPSPARSE_TEST_YAMLS
    # Level 1
    rocm/test_axpyi.yaml
    rocm/test_dotci.yaml
    rocm/test_doti.yaml
    rocm/test_gthr.yaml
    rocm/test_gthrz.yaml
    rocm/test_roti.yaml
    rocm/test_sctr.yaml
    # Level 2    
    rocm/test_bsrmv.yaml
    rocm/test_bsrxmv.yaml
    rocm/test_bsrsv2.yaml
    rocm/test_csrmv.yaml
    rocm/test_csrsv2.yaml
    rocm/test_gemvi.yaml
    rocm/test_hybmv.yaml
    # Level 3
    rocm/test_bsrmm.yaml
    rocm/test_bsrsm2.yaml
    rocm/test_csrmm.yaml
    rocm/test_csrsm2.yaml
    rocm/test_gemmi.yaml
    # Precond
    rocm/test_bsric02.yaml
    rocm/test_bsrilu02.yaml
    rocm/test_csric02.yaml
    rocm/test_csrilu02.yaml
    rocm/test_gpsv_interleaved_batch.yaml
    rocm/test_gtsv.yaml
    rocm/test_gtsv2_nopivot.yaml
    rocm/test_gtsv2_strided_batch.yaml
    rocm/test_gtsv_interleaved_batch.yaml
    # Extra
    rocm/test_csrgeam.yaml
    rocm/test_csrgeam2.yaml
    rocm/test_csrgemm.yaml
    rocm/test_csrgemm2_a.yaml
    rocm/test_csrgemm2_b.yaml
    # Reorder
    rocm/test_csrcolor.yaml
    # Conversion
    rocm/test_bsr2csr.yaml
    rocm/test_coo2csr.yaml
    rocm/test_coosort.yaml
    rocm/test_csc2dense.yaml
    rocm/test_cscsort.yaml
    rocm/test_csr2bsr.yaml
    rocm/test_csr2coo.yaml
    rocm/test_csr2csc.yaml
    rocm/test_csr2csc_ex2.yaml
    rocm/test_csr2csr_compress.yaml
    rocm/test_csr2dense.yaml
    rocm/test_csr2gebsr.yaml
    rocm/test_csr2hyb.yaml
    rocm/test_csrsort.yaml
    rocm/test_csru2csr.yaml
    rocm/test_dense2csc.yaml
    rocm/test_dense2csr.yaml
    rocm/test_gebsr2csr.yaml
    rocm/test_gebsr2gebsc.yaml
    rocm/test_gebsr2gebsr.yaml
    rocm/test_hyb2csr.yaml
    rocm/test_identity.yaml
    rocm/test_nnz.yaml
    rocm/test_prune_csr2csr.yaml
    rocm/test_prune_csr2csr_by_percentage.yaml
    rocm/test_prune_dense2csr.yaml
    rocm/test_prune_dense2csr_by_percentage.yaml
    # Generic
    rocm/test_axpby.yaml
    rocm/test_dense_to_sparse_coo.yaml
    rocm/test_dense_to_sparse_csc.yaml
    rocm/test_dense_to_sparse_csr.yaml
    rocm/test_gather.yaml
    rocm/test_rot.yaml
    rocm/test_scatter.yaml
    rocm/test_sddmm_coo.yaml
    rocm/test_sddmm_coo_aos.yaml
    rocm/test_sddmm_csc.yaml
    rocm/test_sddmm_csr.yaml
    rocm/test_sparse_to_dense_coo.yaml
    rocm/test_sparse_to_dense_csc.yaml
    rocm/test_sparse_to_dense_csr.yaml
    rocm/test_spgemm_csr.yaml
    rocm/test_spgemmreuse_csr.yaml
    rocm/test_spmm_batched_coo.yaml
    rocm/test_spmm_batched_csc.yaml
    rocm/test_spmm_batched_csr.yaml
    rocm/test_spmm_bsr.yaml
    rocm/test_spmm_coo.yaml
    rocm/test_spmm_csc.yaml
    rocm/test_spmm_csr.yaml
    rocm/test_spmv_coo.yaml
    rocm/test_spmv_coo_aos.yaml
    rocm/test_spmv_csr.yaml
    rocm/test_spmv_csr_reuse_descr.yaml
    rocm/test_spmv_sell.yaml
    rocm/test_spsm_coo.yaml
    rocm/test_spsm_csr.yaml
    rocm/test_spsm_ex_coo.yaml
    rocm/test_spsm_ex_csr.yaml
    rocm/test_spsv_coo.yaml
    rocm/test_spsv_csr.yaml
    rocm/test_spvv.yaml)

if(HIPSPARSE_ENABLE_SPMV_BSR)
    list(APPEND HIPSPARSE_TEST_SOURCES test_spmv_bsr.cpp)
    list(APPEND HIPSPARSE_TEST_YAMLS rocm/test_spmv_bsr.yaml)
endif()
