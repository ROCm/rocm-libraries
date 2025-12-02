# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

set(HIPSPARSE_TEST_SOURCES
    # Level 1
    test_axpyi.cpp
    test_gthr.cpp
    test_gthrz.cpp
    test_roti.cpp
    test_sctr.cpp
    # Level 2
    test_bsrmv.cpp
    test_bsrsv2.cpp
    test_bsrxmv.cpp
    test_csrsv2.cpp
    test_gemvi.cpp
    # Level 3
    test_bsrmm.cpp
    test_bsrsm2.cpp
    test_csrsm2.cpp
    test_gemmi.cpp
    # Extra    
    test_csrgeam2.cpp
    test_csrgemm2_a.cpp
    test_csrgemm2_b.cpp
    # Rorder
    test_csrcolor.cpp
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
    # Conversion
    test_bsr2csr.cpp
    test_coo2csr.cpp
    test_coosort.cpp
    test_csc2dense.cpp
    test_cscsort.cpp
    test_csr2bsr.cpp
    test_csr2coo.cpp
    test_csr2csr_compress.cpp
    test_csr2dense.cpp
    test_csr2gebsr.cpp
    test_csrilusv.cpp
    test_csrsort.cpp
    test_csru2csr.cpp
    test_dense2csc.cpp
    test_dense2csr.cpp
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
    test_spmv_coo.cpp
    test_spmv_coo_aos.cpp
    test_spmv_csr.cpp
    test_spmv_sell.cpp
    test_spsm_coo.cpp
    test_spsm_csr.cpp
    test_spsv_coo.cpp
    test_spsv_csr.cpp
    test_spvec_descr.cpp
    test_spvv.cpp
)

set(HIPSPARSE_TEST_YAMLS
    # Level 1
    test_axpyi.yaml
    test_dotci.yaml
    test_doti.yaml
    test_gthr.yaml
    test_gthrz.yaml
    test_roti.yaml
    test_sctr.yaml
    # Level 2    
    test_bsrmv.yaml
    test_bsrsv2.yaml
    test_csrmv.yaml
    test_csrsv2.yaml
    test_gemvi.yaml
    test_hybmv.yaml
    # Level 3
    test_bsrmm.yaml
    test_bsrsm2.yaml
    test_csrmm.yaml
    test_csrsm2.yaml
    test_gemmi.yaml
    # Precond
    test_bsric02.yaml
    test_bsrilu02.yaml
    test_csric02.yaml
    test_csrilu02.yaml
    test_gpsv_interleaved_batch.yaml
    test_gtsv.yaml
    test_gtsv2_nopivot.yaml
    test_gtsv2_strided_batch.yaml
    test_gtsv_interleaved_batch.yaml
    # Conversion
    test_bsr2csr.yaml
    test_coo2csr.yaml
    test_coosort.yaml
    test_csc2dense.yaml
    test_cscsort.yaml
    test_csr2bsr.yaml
    test_csr2coo.yaml
    test_csr2csc.yaml
    test_csr2csc_ex2.yaml
    test_csr2csr_compress.yaml
    test_csr2dense.yaml
    test_csr2gebsr.yaml
    test_csr2hyb.yaml
    test_csrsort.yaml
    test_csru2csr.yaml
    test_dense2csc.yaml
    test_dense2csr.yaml
    test_gebsr2csr.yaml
    test_gebsr2gebsc.yaml
    test_gebsr2gebsr.yaml
    test_hyb2csr.yaml
    test_identity.yaml
    test_nnz.yaml
    test_prune_csr2csr.yaml
    test_prune_csr2csr_by_percentage.yaml
    test_prune_dense2csr.yaml
    test_prune_dense2csr_by_percentage.yaml
    # Generic
    test_axpby.yaml
    test_dense_to_sparse_coo.yaml
    test_dense_to_sparse_csc.yaml
    test_dense_to_sparse_csr.yaml
    test_gather.yaml
    test_rot.yaml
    test_scatter.yaml
    test_sddmm_coo.yaml
    test_sddmm_coo_aos.yaml
    test_sddmm_csc.yaml
    test_sddmm_csr.yaml
    test_sparse_to_dense_coo.yaml
    test_sparse_to_dense_csc.yaml
    test_sparse_to_dense_csr.yaml
    test_spgemm_csr.yaml
    test_spgemmreuse_csr.yaml
    test_spmm_batched_coo.yaml
    test_spmm_batched_csc.yaml
    test_spmm_batched_csr.yaml
    test_spmm_coo.yaml
    test_spmm_csc.yaml
    test_spmm_csr.yaml
    test_spmv_coo.yaml
    test_spmv_coo_aos.yaml
    test_spmv_csr.yaml
    test_spmv_sell.yaml
    test_spsm_coo.yaml
    test_spsm_csr.yaml
    test_spsv_coo.yaml
    test_spsv_csr.yaml
    test_spvv.yaml)