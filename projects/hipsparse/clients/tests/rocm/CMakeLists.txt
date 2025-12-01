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
    rocm/test_axpyi.cpp
    rocm/test_gthr.cpp
    rocm/test_gthrz.cpp
    rocm/test_roti.cpp
    rocm/test_sctr.cpp
    # Level 2
    rocm/test_bsrmv.cpp
    rocm/test_bsrsv2.cpp
    rocm/test_bsrxmv.cpp
    rocm/test_csrsv2.cpp
    rocm/test_gemvi.cpp
    # Level 3
    rocm/test_bsrmm.cpp
    rocm/test_bsrsm2.cpp
    rocm/test_csrsm2.cpp
    rocm/test_gemmi.cpp
    # Extra    
    rocm/test_csrgeam2.cpp
    rocm/test_csrgemm2_a.cpp
    rocm/test_csrgemm2_b.cpp
    # Rorder
    rocm/test_csrcolor.cpp
    # Precond 
    rocm/test_bsric02.cpp
    rocm/test_bsrilu02.cpp
    rocm/test_csric02.cpp
    rocm/test_csrilu02.cpp
    rocm/test_gpsv_interleaved_batch.cpp
    rocm/test_gtsv.cpp
    rocm/test_gtsv2_nopivot.cpp
    rocm/test_gtsv2_strided_batch.cpp
    rocm/test_gtsv_interleaved_batch.cpp
    # Conversion
    rocm/test_bsr2csr.cpp
    rocm/test_coo2csr.cpp
    rocm/test_coosort.cpp
    rocm/test_csc2dense.cpp
    rocm/test_cscsort.cpp
    rocm/test_csr2bsr.cpp
    rocm/test_csr2coo.cpp
    rocm/test_csr2csr_compress.cpp
    rocm/test_csr2dense.cpp
    rocm/test_csr2gebsr.cpp
    rocm/test_csrilusv.cpp
    rocm/test_csrsort.cpp
    rocm/test_csru2csr.cpp
    rocm/test_dense2csc.cpp
    rocm/test_dense2csr.cpp
    rocm/test_gebsr2csr.cpp
    rocm/test_gebsr2gebsc.cpp
    rocm/test_gebsr2gebsr.cpp
    rocm/test_identity.cpp
    rocm/test_nnz.cpp
    rocm/test_prune_csr2csr.cpp
    rocm/test_prune_csr2csr_by_percentage.cpp
    rocm/test_prune_dense2csr.cpp
    rocm/test_prune_dense2csr_by_percentage.cpp
    # Generic
    rocm/test_axpby.cpp
    rocm/test_const_dnmat_descr.cpp
    rocm/test_const_dnvec_descr.cpp
    rocm/test_const_spmat_descr.cpp
    rocm/test_const_spvec_descr.cpp
    rocm/test_dense_to_sparse_coo.cpp
    rocm/test_dense_to_sparse_csc.cpp
    rocm/test_dense_to_sparse_csr.cpp
    rocm/test_dnmat_descr.cpp
    rocm/test_dnvec_descr.cpp
    rocm/test_gather.cpp
    rocm/test_rot.cpp
    rocm/test_scatter.cpp
    rocm/test_sddmm_coo.cpp
    rocm/test_sddmm_coo_aos.cpp
    rocm/test_sddmm_csc.cpp
    rocm/test_sddmm_csr.cpp
    rocm/test_sparse_to_dense_coo.cpp
    rocm/test_sparse_to_dense_csc.cpp
    rocm/test_sparse_to_dense_csr.cpp
    rocm/test_spgemm_csr.cpp
    rocm/test_spgemmreuse_csr.cpp
    rocm/test_spmat_descr.cpp
    rocm/test_spmm_batched_coo.cpp
    rocm/test_spmm_batched_csc.cpp
    rocm/test_spmm_batched_csr.cpp
    rocm/test_spmm_bell.cpp
    rocm/test_spmm_coo.cpp
    rocm/test_spmm_csc.cpp
    rocm/test_spmm_csr.cpp
    rocm/test_spmv_coo.cpp
    rocm/test_spmv_coo_aos.cpp
    rocm/test_spmv_csr.cpp
    rocm/test_spmv_sell.cpp
    rocm/test_spsm_coo.cpp
    rocm/test_spsm_csr.cpp
    rocm/test_spsv_coo.cpp
    rocm/test_spsv_csr.cpp
    rocm/test_spvec_descr.cpp
    rocm/test_spvv.cpp
)

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
    rocm/test_spmm_coo.yaml
    rocm/test_spmm_csc.yaml
    rocm/test_spmm_csr.yaml
    rocm/test_spmv_coo.yaml
    rocm/test_spmv_coo_aos.yaml
    rocm/test_spmv_csr.yaml
    rocm/test_spmv_sell.yaml
    rocm/test_spsm_coo.yaml
    rocm/test_spsm_csr.yaml
    rocm/test_spsv_coo.yaml
    rocm/test_spsv_csr.yaml
    rocm/test_spvv.yaml)