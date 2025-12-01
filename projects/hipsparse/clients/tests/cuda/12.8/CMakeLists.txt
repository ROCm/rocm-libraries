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
    # Level 2
    cuda/12.8/test_bsrmv.cpp
    cuda/12.8/test_bsrsv2.cpp
    cuda/12.8/test_bsrxmv.cpp
    cuda/12.8/test_gemvi.cpp
    # Level 3
    cuda/12.8/test_bsrmm.cpp
    cuda/12.8/test_bsrsm2.cpp
    # Extra
    cuda/12.8/test_csrgeam2.cpp
    # Reorder
    cuda/12.8/test_csrcolor.cpp
    # Precond  
    cuda/12.8/test_bsric02.cpp
    cuda/12.8/test_bsrilu02.cpp
    cuda/12.8/test_csric02.cpp
    cuda/12.8/test_csrilu02.cpp
    cuda/12.8/test_gpsv_interleaved_batch.cpp
    cuda/12.8/test_gtsv.cpp
    cuda/12.8/test_gtsv2_nopivot.cpp
    cuda/12.8/test_gtsv2_strided_batch.cpp
    cuda/12.8/test_gtsv_interleaved_batch.cpp
    # Conversion
    cuda/12.8/test_bsr2csr.cpp
    cuda/12.8/test_coo2csr.cpp
    cuda/12.8/test_coosort.cpp
    cuda/12.8/test_cscsort.cpp
    cuda/12.8/test_csr2bsr.cpp
    cuda/12.8/test_csr2coo.cpp
    cuda/12.8/test_csr2gebsr.cpp
    cuda/12.8/test_csrsort.cpp
    cuda/12.8/test_csru2csr.cpp
    cuda/12.8/test_gebsr2csr.cpp
    cuda/12.8/test_gebsr2gebsc.cpp
    cuda/12.8/test_gebsr2gebsr.cpp
    cuda/12.8/test_identity.cpp
    cuda/12.8/test_nnz.cpp
    cuda/12.8/test_prune_csr2csr.cpp
    cuda/12.8/test_prune_csr2csr_by_percentage.cpp
    cuda/12.8/test_prune_dense2csr.cpp
    cuda/12.8/test_prune_dense2csr_by_percentage.cpp
    # Generic
    cuda/12.8/test_axpby.cpp
    cuda/12.8/test_const_dnmat_descr.cpp
    cuda/12.8/test_const_dnvec_descr.cpp
    cuda/12.8/test_const_spmat_descr.cpp
    cuda/12.8/test_const_spvec_descr.cpp
    cuda/12.8/test_dense_to_sparse_coo.cpp
    cuda/12.8/test_dense_to_sparse_csc.cpp
    cuda/12.8/test_dense_to_sparse_csr.cpp
    cuda/12.8/test_dnmat_descr.cpp
    cuda/12.8/test_dnvec_descr.cpp
    cuda/12.8/test_gather.cpp
    cuda/12.8/test_rot.cpp
    cuda/12.8/test_scatter.cpp
    cuda/12.8/test_sddmm_coo.cpp
    cuda/12.8/test_sddmm_coo_aos.cpp
    cuda/12.8/test_sddmm_csc.cpp
    cuda/12.8/test_sddmm_csr.cpp
    cuda/12.8/test_sparse_to_dense_coo.cpp
    cuda/12.8/test_sparse_to_dense_csc.cpp
    cuda/12.8/test_sparse_to_dense_csr.cpp
    cuda/12.8/test_spgemm_csr.cpp
    cuda/12.8/test_spgemmreuse_csr.cpp
    cuda/12.8/test_spmat_descr.cpp
    cuda/12.8/test_spmm_batched_coo.cpp
    cuda/12.8/test_spmm_batched_csc.cpp
    cuda/12.8/test_spmm_batched_csr.cpp
    cuda/12.8/test_spmm_bell.cpp
    cuda/12.8/test_spmm_coo.cpp
    cuda/12.8/test_spmm_csc.cpp
    cuda/12.8/test_spmm_csr.cpp
    cuda/12.8/test_spmv_coo.cpp
    cuda/12.8/test_spmv_coo_aos.cpp
    cuda/12.8/test_spmv_csr.cpp
    cuda/12.8/test_spmv_sell.cpp
    cuda/12.8/test_spsm_coo.cpp
    cuda/12.8/test_spsm_csr.cpp
    cuda/12.8/test_spsv_coo.cpp
    cuda/12.8/test_spsv_csr.cpp
    cuda/12.8/test_spvec_descr.cpp
    cuda/12.8/test_spvv.cpp
)

set(HIPSPARSE_TEST_YAMLS 
    # Level 2
    cuda/12.8/test_bsrmv.yaml
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
    cuda/12.8/test_spmv_coo.yaml
    cuda/12.8/test_spmv_coo_aos.yaml
    cuda/12.8/test_spmv_csr.yaml
    cuda/12.8/test_spmv_sell.yaml
    cuda/12.8/test_spsm_coo.yaml
    cuda/12.8/test_spsm_csr.yaml
    cuda/12.8/test_spsv_coo.yaml
    cuda/12.8/test_spsv_csr.yaml
    cuda/12.8/test_spvv.yaml)
