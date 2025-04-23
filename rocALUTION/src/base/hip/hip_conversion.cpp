/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hip_conversion.hpp"
#include "../../utils/def.hpp"
#include "../matrix_formats.hpp"
#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"
#include "hip_kernels_conversion.hpp"
#include "hip_sparse.hpp"
#include "hip_utils.hpp"
#include "rocalution/utils/types.hpp"

#include <hip/hip_runtime_api.h>
#include <rocprim/rocprim.hpp>

#include <complex>

namespace rocalution
{
    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_coo_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixCOO<ValueType, IndexType>*                    dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.row_offset != NULL);
        assert(src.col != NULL);
        assert(src.val != NULL);

        assert(dst != NULL);
        assert(backend != NULL);

        allocate_hip(nnz, &dst->row);
        allocate_hip(nnz, &dst->col);
        allocate_hip(nnz, &dst->val);

        copy_d2d(nnz, src.col, dst->col, true, HIPSTREAM(backend->HIP_stream_current));
        copy_d2d(nnz, src.val, dst->val, true, HIPSTREAM(backend->HIP_stream_current));

        rocsparse_status status = rocsparse_csr2coo(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                                    src.row_offset,
                                                    nnz,
                                                    nrow,
                                                    dst->row,
                                                    rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool coo_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                        int64_t                                       nnz,
                        IndexType                                     nrow,
                        IndexType                                     ncol,
                        const MatrixCOO<ValueType, IndexType>&        src,
                        MatrixCSR<ValueType, IndexType, PointerType>* dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(backend != NULL);

        allocate_hip(nrow + 1, &dst->row_offset);
        allocate_hip(nnz, &dst->col);
        allocate_hip(nnz, &dst->val);

        copy_d2d(nnz, src.col, dst->col, true, HIPSTREAM(backend->HIP_stream_current));
        copy_d2d(nnz, src.val, dst->val, true, HIPSTREAM(backend->HIP_stream_current));

        rocsparse_status status = rocsparse_coo2csr(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                                    src.row,
                                                    nnz,
                                                    nrow,
                                                    dst->row_offset,
                                                    rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                backend,
                         int64_t                                             nnz,
                         IndexType                                           nrow,
                         IndexType                                           ncol,
                         const MatrixCSR<ValueType, IndexType, PointerType>& src,
                         const rocsparse_mat_descr                           src_descr,
                         MatrixBCSR<ValueType, IndexType>*                   dst,
                         const rocsparse_mat_descr                           dst_descr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(backend != NULL);

        IndexType blockdim = dst->blockdim;

        assert(blockdim > 1);

        // Matrix dimensions must be a multiple of blockdim
        if((nrow % blockdim) != 0 || (ncol % blockdim) != 0)
        {
            return false;
        }

        // BCSR row blocks
        IndexType mb = (nrow + blockdim - 1) / blockdim;
        IndexType nb = (ncol + blockdim - 1) / blockdim;
        IndexType nnzb;

        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        allocate_hip(mb + 1, &dst->row_offset);

        rocsparse_status status
            = rocsparse_csr2bsr_nnz(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                    dir,
                                    nrow,
                                    ncol,
                                    src_descr,
                                    src.row_offset,
                                    src.col,
                                    blockdim,
                                    dst_descr,
                                    dst->row_offset,
                                    &nnzb);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nnzb, &dst->col);
        allocate_hip(nnzb * blockdim * blockdim, &dst->val);

        status = rocsparseTcsr2bsr(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                   dir,
                                   nrow,
                                   ncol,
                                   src_descr,
                                   src.val,
                                   src.row_offset,
                                   src.col,
                                   blockdim,
                                   dst_descr,
                                   dst->val,
                                   dst->row_offset,
                                   dst->col);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        dst->nrowb = mb;
        dst->ncolb = nb;
        dst->nnzb  = nnzb;

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                         int64_t                                       nnz,
                         IndexType                                     nrow,
                         IndexType                                     ncol,
                         const MatrixBCSR<ValueType, IndexType>&       src,
                         const rocsparse_mat_descr                     src_descr,
                         MatrixCSR<ValueType, IndexType, PointerType>* dst,
                         rocsparse_mat_descr                           dst_descr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(backend != NULL);

        IndexType blockdim = src.blockdim;

        assert(blockdim > 1);

        // Allocate device memory for uncompressed CSR matrix
        // IndexType* csr_row_offset = NULL;
        // IndexType* csr_col_ind = NULL;
        // ValueType* csr_val = NULL;
        // allocate_hip(nrow + 1, &csr_row_offset);
        // allocate_hip(nnz, &csr_col_ind);
        // allocate_hip(nnz, &csr_val);
        allocate_hip(nrow + 1, &dst->row_offset);
        allocate_hip(nnz, &dst->col);
        allocate_hip(nnz, &dst->val);

        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        rocsparse_status status = rocsparseTbsr2csr(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                                    dir,
                                                    src.nrowb,
                                                    src.ncolb,
                                                    src_descr,
                                                    src.val,
                                                    src.row_offset,
                                                    src.col,
                                                    blockdim,
                                                    dst_descr,
                                                    dst->val /*csr_val*/,
                                                    dst->row_offset /*csr_row_offset*/,
                                                    dst->col /*csr_col_ind*/);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // // Compress the output CSR matrix
        // IndexType nnz_C;
        // IndexType* nnz_per_row = NULL;
        // allocate_hip(nnz, &nnz_per_row);

        // status = rocsparseTnnz_compress(ROCSPARSE_HANDLE(backend->ROC_sparse_handle), nrow, dst_descr, csr_val, csr_row_offset, nnz_per_row, &nnz_C, static_cast<ValueType>(0));
        // CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // // Allocate device memory for the compressed version of the CSR matrix
        // allocate_hip(nrow + 1, &dst->row_offset);
        // allocate_hip(nnz_C, &dst->col);
        // allocate_hip(nnz_C, &dst->val);

        // // Finish compression
        // status = rocsparseTcsr2csr_compress(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
        //                                         nrow,
        //                                         ncol,
        //                                         dst_descr,
        //                                         csr_val,
        //                                         csr_row_offset,
        //                                         csr_col_ind,
        //                                         nnz,
        //                                         nnz_per_row,
        //                                         dst->val,
        //                                         dst->row_offset,
        //                                         dst->col,
        //                                         static_cast<ValueType>(0));
        // CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        const rocsparse_mat_descr                           src_descr,
                        MatrixELL<ValueType, IndexType>*                    dst,
                        const rocsparse_mat_descr                           dst_descr,
                        int64_t*                                            nnz_ell)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(nnz_ell != NULL);
        assert(backend != NULL);
        assert(src_descr != NULL);
        assert(dst_descr != NULL);

        rocsparse_status status;

        // Determine ELL width
        status = rocsparse_csr2ell_width(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                         nrow,
                                         src_descr,
                                         src.row_offset,
                                         dst_descr,
                                         &dst->max_row);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Synchronize stream to make sure, result is available on the host
        hipStreamSynchronize(HIPSTREAM(backend->HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Limit ELL size to 5 times CSR nnz
        if(dst->max_row > 5 * (nnz / nrow))
        {
            return false;
        }

        // Compute ELL non-zeros
        *nnz_ell = dst->max_row * nrow;

        // Allocate ELL matrix
        allocate_hip(*nnz_ell, &dst->col);
        allocate_hip(*nnz_ell, &dst->val);

        // Conversion
        status = rocsparseTcsr2ell(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                   nrow,
                                   src_descr,
                                   src.val,
                                   src.row_offset,
                                   src.col,
                                   dst_descr,
                                   dst->max_row,
                                   dst->val,
                                   dst->col);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                        int64_t                                       nnz,
                        IndexType                                     nrow,
                        IndexType                                     ncol,
                        const MatrixELL<ValueType, IndexType>&        src,
                        const rocsparse_mat_descr                     src_descr,
                        MatrixCSR<ValueType, IndexType, PointerType>* dst,
                        const rocsparse_mat_descr                     dst_descr,
                        int64_t*                                      nnz_csr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(nnz_csr != NULL);
        assert(backend != NULL);
        assert(src_descr != NULL);
        assert(dst_descr != NULL);

        rocsparse_status status;

        // Allocate CSR row offset structure
        allocate_hip(nrow + 1, &dst->row_offset);

        // Determine CSR nnz
        IndexType nnz32;
        status = rocsparse_ell2csr_nnz(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                       nrow,
                                       ncol,
                                       src_descr,
                                       src.max_row,
                                       src.col,
                                       dst_descr,
                                       dst->row_offset,
                                       &nnz32);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        assert(nnz32 <= std::numeric_limits<IndexType>::max());

        *nnz_csr = nnz32;

        if(*nnz_csr < 0)
        {
            free_hip(&dst->row_offset);
            return false;
        }

        // Allocate CSR column and value structures
        allocate_hip(*nnz_csr, &dst->col);
        allocate_hip(*nnz_csr, &dst->val);

        // Conversion
        status = rocsparseTell2csr(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                   nrow,
                                   ncol,
                                   src_descr,
                                   src.max_row,
                                   src.val,
                                   src.col,
                                   dst_descr,
                                   dst->val,
                                   dst->row_offset,
                                   dst->col);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_dia_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixDIA<ValueType, IndexType>*                    dst,
                        int64_t*                                            nnz_dia,
                        IndexType*                                          num_diag)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);
        assert(backend != NULL);

        assert(dst != NULL);
        assert(nnz_dia != NULL);
        assert(num_diag != NULL);

        assert(nrow + ncol <= std::numeric_limits<int>::max());

        // Get blocksize
        int blocksize = backend->HIP_block_size;

        // Get stream
        hipStream_t stream = HIPSTREAM(backend->HIP_stream_current);

        // Get diagonal mapping vector
        IndexType* diag_idx = NULL;
        allocate_hip(nrow + ncol, &diag_idx);
        set_to_zero_hip(blocksize, nrow + ncol, diag_idx);

        kernel_dia_diag_idx<<<(nrow - 1) / blocksize + 1, blocksize, 0, stream>>>(
            nrow, src.row_offset, src.col, diag_idx);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Reduction to obtain number of occupied diagonals
        IndexType* d_num_diag = NULL;
        allocate_hip(1, &d_num_diag);

        size_t rocprim_size   = 0;
        char*  rocprim_buffer = NULL;

        // Get reduction buffer size
        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        diag_idx,
                        d_num_diag,
                        0,
                        nrow + ncol,
                        rocprim::plus<IndexType>(),
                        stream);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Allocate rocprim buffer
        allocate_hip(rocprim_size, &rocprim_buffer);

        // Do reduction
        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        diag_idx,
                        d_num_diag,
                        0,
                        nrow + ncol,
                        rocprim::plus<IndexType>(),
                        stream);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Clear rocprim buffer
        free_hip(&rocprim_buffer);

        // Copy result to host
        copy_d2h(1, d_num_diag, num_diag);

        // Free device memory
        free_hip(&d_num_diag);

        // Conversion fails if DIA nnz exceeds 5 times CSR nnz
        IndexType size = (nrow > ncol) ? nrow : ncol;
        if(*num_diag > 5 * (nnz / size))
        {
            free_hip(&diag_idx);
            return false;
        }

        *nnz_dia = *num_diag * size;

        // Allocate DIA matrix
        allocate_hip(*num_diag, &dst->offset);
        allocate_hip(*nnz_dia, &dst->val);

        // Initialize values with zero
        set_to_zero_hip(blocksize, *num_diag, dst->offset);
        set_to_zero_hip(blocksize, *nnz_dia, dst->val);

        // Inclusive sum to obtain diagonal offsets
        IndexType* work = NULL;
        allocate_hip(nrow + ncol, &work);
        rocprim_buffer = NULL;
        rocprim_size   = 0;

        // Obtain rocprim buffer size
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                diag_idx,
                                work,
                                0,
                                nrow + ncol,
                                rocprim::plus<IndexType>(),
                                stream);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Allocate rocprim buffer
        rocprim_buffer = NULL;
        allocate_hip(rocprim_size, &rocprim_buffer);

        // Do inclusive sum
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                diag_idx,
                                work,
                                0,
                                nrow + ncol,
                                rocprim::plus<IndexType>(),
                                stream);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Clear rocprim buffer
        free_hip(&rocprim_buffer);

        // Fill DIA structures
        kernel_dia_fill_offset<<<(nrow + ncol) / blocksize + 1, blocksize, 0, stream>>>(
            nrow, ncol, diag_idx, work, dst->offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&work);

        kernel_dia_convert<<<(nrow - 1) / blocksize + 1, blocksize, 0, stream>>>(
            nrow, *num_diag, src.row_offset, src.col, src.val, diag_idx, dst->val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Clear
        free_hip(&diag_idx);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_hyb_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixHYB<ValueType, IndexType>*                    dst,
                        int64_t*                                            nnz_hyb,
                        int64_t*                                            nnz_ell,
                        int64_t*                                            nnz_coo)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);
        assert(backend != NULL);

        assert(dst != NULL);
        assert(nnz_hyb != NULL);
        assert(nnz_ell != NULL);
        assert(nnz_coo != NULL);

        // Get blocksize
        int blocksize = backend->HIP_block_size;

        // Get stream
        hipStream_t stream = HIPSTREAM(backend->HIP_stream_current);

        // Determine ELL width by average nnz per row
        if(dst->ELL.max_row == 0)
        {
            dst->ELL.max_row = (nnz - 1) / nrow + 1;
        }

        // ELL nnz is ELL width times nrow
        *nnz_ell = dst->ELL.max_row * nrow;
        *nnz_coo = 0;

        // Allocate ELL part
        allocate_hip(*nnz_ell, &dst->ELL.col);
        allocate_hip(*nnz_ell, &dst->ELL.val);

        // Array to gold COO part nnz per row
        PointerType* coo_row_nnz = NULL;
        allocate_hip(nrow + 1, &coo_row_nnz);

        // If there is no ELL part, its easy
        if(*nnz_ell == 0)
        {
            *nnz_coo = nnz;
            copy_d2d(nrow + 1,
                     src.row_offset,
                     coo_row_nnz,
                     true,
                     HIPSTREAM(backend->HIP_stream_current));
        }
        else
        {
            kernel_hyb_coo_nnz<<<(nrow - 1) / blocksize + 1, blocksize, 0, stream>>>(
                nrow, dst->ELL.max_row, src.row_offset, coo_row_nnz);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Inclusive sum on coo_row_nnz
            size_t rocprim_size   = 0;
            char*  rocprim_buffer = NULL;

            // Obtain rocprim buffer size
            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    coo_row_nnz,
                                    coo_row_nnz,
                                    0,
                                    nrow + 1,
                                    rocprim::plus<PointerType>(),
                                    stream);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Allocate rocprim buffer
            allocate_hip(rocprim_size, &rocprim_buffer);

            // Do exclusive sum
            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    coo_row_nnz,
                                    coo_row_nnz,
                                    0,
                                    nrow + 1,
                                    rocprim::plus<PointerType>(),
                                    stream);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Clear rocprim buffer
            free_hip(&rocprim_buffer);

            // Copy result to host
            PointerType nnz;
            copy_d2h(1, coo_row_nnz + nrow, &nnz);
            *nnz_coo = nnz;
        }

        *nnz_hyb = *nnz_coo + *nnz_ell;

        if(*nnz_hyb <= 0)
        {
            return false;
        }

        // Allocate COO part
        allocate_hip(*nnz_coo, &dst->COO.row);
        allocate_hip(*nnz_coo, &dst->COO.col);
        allocate_hip(*nnz_coo, &dst->COO.val);

        // Fill HYB structures
        kernel_hyb_csr2hyb<<<(nrow - 1) / blocksize + 1, blocksize, 0, stream>>>(nrow,
                                                                                 src.val,
                                                                                 src.row_offset,
                                                                                 src.col,
                                                                                 dst->ELL.max_row,
                                                                                 dst->ELL.col,
                                                                                 dst->ELL.val,
                                                                                 dst->COO.row,
                                                                                 dst->COO.col,
                                                                                 dst->COO.val,
                                                                                 coo_row_nnz);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&coo_row_nnz);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*                backend,
                          IndexType                                           nrow,
                          IndexType                                           ncol,
                          const MatrixCSR<ValueType, IndexType, PointerType>& src,
                          const rocsparse_mat_descr                           src_descr,
                          MatrixDENSE<ValueType>*                             dst)
    {
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(backend != NULL);
        assert(src_descr != NULL);

        allocate_hip(nrow * ncol, &dst->val);

        if(DENSE_IND_BASE == 0)
        {
            rocsparse_status status
                = rocsparseTcsr2dense(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                      nrow,
                                      ncol,
                                      src_descr,
                                      src.val,
                                      src.row_offset,
                                      src.col,
                                      dst->val,
                                      nrow);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            ValueType* temp = NULL;
            allocate_hip(nrow * ncol, &temp);

            rocsparse_status sparse_status
                = rocsparseTcsr2dense(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                      nrow,
                                      ncol,
                                      src_descr,
                                      src.val,
                                      src.row_offset,
                                      src.col,
                                      temp,
                                      nrow);
            CHECK_ROCSPARSE_ERROR(sparse_status, __FILE__, __LINE__);

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta  = static_cast<ValueType>(0);

            // Not actually used in following geam call as beta is zero
            ValueType* B;

            // transpose matrix so that dst values are row major
            rocblas_status blas_status = rocblasTgeam(ROCBLAS_HANDLE(backend->ROC_blas_handle),
                                                      rocblas_operation_transpose,
                                                      rocblas_operation_none,
                                                      nrow,
                                                      ncol,
                                                      &alpha,
                                                      temp,
                                                      nrow,
                                                      &beta,
                                                      B,
                                                      nrow,
                                                      dst->val,
                                                      nrow);
            CHECK_ROCBLAS_ERROR(blas_status, __FILE__, __LINE__);

            free_hip(&temp);
        }

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                          IndexType                                     nrow,
                          IndexType                                     ncol,
                          const MatrixDENSE<ValueType>&                 src,
                          MatrixCSR<ValueType, IndexType, PointerType>* dst,
                          const rocsparse_mat_descr                     dst_descr,
                          int64_t*                                      nnz_csr)
    {
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(backend != NULL);
        assert(dst_descr != NULL);

        IndexType  nnz_total;
        IndexType* nnz_per_row = NULL;

        if(DENSE_IND_BASE == 0)
        {
            rocsparse_status status;

            allocate_hip(nrow, &nnz_per_row);

            status = rocsparseTnnz(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                   rocsparse_direction_row,
                                   nrow,
                                   ncol,
                                   dst_descr,
                                   src.val,
                                   nrow,
                                   nnz_per_row,
                                   &nnz_total);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            allocate_hip(nrow + 1, &dst->row_offset);
            allocate_hip(nnz_total, &dst->col);
            allocate_hip(nnz_total, &dst->val);

            status = rocsparseTdense2csr(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                         nrow,
                                         ncol,
                                         dst_descr,
                                         src.val,
                                         nrow,
                                         nnz_per_row,
                                         nnz_total == 0 ? (ValueType*)0x4 : dst->val,
                                         dst->row_offset,
                                         nnz_total == 0 ? (IndexType*)0x4 : dst->col);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            free_hip(&nnz_per_row);
        }
        else
        {
            ValueType* temp = NULL;
            allocate_hip(nrow * ncol, &temp);

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta  = static_cast<ValueType>(0);

            // Not actually used in following geam call as beta is zero
            ValueType* B;

            // transpose matrix so that src values are column major
            rocblas_status blas_status = rocblasTgeam(ROCBLAS_HANDLE(backend->ROC_blas_handle),
                                                      rocblas_operation_transpose,
                                                      rocblas_operation_none,
                                                      nrow,
                                                      ncol,
                                                      &alpha,
                                                      src.val,
                                                      nrow,
                                                      &beta,
                                                      B,
                                                      nrow,
                                                      temp,
                                                      nrow);
            CHECK_ROCBLAS_ERROR(blas_status, __FILE__, __LINE__);

            allocate_hip(nrow, &nnz_per_row);

            rocsparse_status sparse_status;

            sparse_status = rocsparseTnnz(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                          rocsparse_direction_row,
                                          nrow,
                                          ncol,
                                          dst_descr,
                                          temp,
                                          nrow,
                                          nnz_per_row,
                                          &nnz_total);
            CHECK_ROCSPARSE_ERROR(sparse_status, __FILE__, __LINE__);

            allocate_hip(nrow + 1, &dst->row_offset);
            allocate_hip(nnz_total, &dst->col);
            allocate_hip(nnz_total, &dst->val);

            sparse_status = rocsparseTdense2csr(ROCSPARSE_HANDLE(backend->ROC_sparse_handle),
                                                nrow,
                                                ncol,
                                                dst_descr,
                                                temp,
                                                nrow,
                                                nnz_per_row,
                                                nnz_total == 0 ? (ValueType*)0x4 : dst->val,
                                                dst->row_offset,
                                                nnz_total == 0 ? (IndexType*)0x4 : dst->col);
            CHECK_ROCSPARSE_ERROR(sparse_status, __FILE__, __LINE__);

            free_hip(&temp);
            free_hip(&nnz_per_row);
        }

        // Sync memcopy
        hipDeviceSynchronize();

        *nnz_csr = nnz_total;

        return true;
    }

    // csr_to_coo
    template bool csr_to_coo_hip(const Rocalution_Backend_Descriptor*  backend,
                                 int64_t                               nnz,
                                 int                                   nrow,
                                 int                                   ncol,
                                 const MatrixCSR<float, int, PtrType>& src,
                                 MatrixCOO<float, int>*                dst);

    template bool csr_to_coo_hip(const Rocalution_Backend_Descriptor*   backend,
                                 int64_t                                nnz,
                                 int                                    nrow,
                                 int                                    ncol,
                                 const MatrixCSR<double, int, PtrType>& src,
                                 MatrixCOO<double, int>*                dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_coo_hip(const Rocalution_Backend_Descriptor*                backend,
                                 int64_t                                             nnz,
                                 int                                                 nrow,
                                 int                                                 ncol,
                                 const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                 MatrixCOO<std::complex<float>, int>*                dst);

    template bool csr_to_coo_hip(const Rocalution_Backend_Descriptor*                 backend,
                                 int64_t                                              nnz,
                                 int                                                  nrow,
                                 int                                                  ncol,
                                 const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                 MatrixCOO<std::complex<double>, int>*                dst);
#endif

    // coo_to_csr
    template bool coo_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                 int64_t                              nnz,
                                 int                                  nrow,
                                 int                                  ncol,
                                 const MatrixCOO<float, int>&         src,
                                 MatrixCSR<float, int, PtrType>*      dst);

    template bool coo_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                 int64_t                              nnz,
                                 int                                  nrow,
                                 int                                  ncol,
                                 const MatrixCOO<double, int>&        src,
                                 MatrixCSR<double, int, PtrType>*     dst);

#ifdef SUPPORT_COMPLEX
    template bool coo_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                                 int64_t                                       nnz,
                                 int                                           nrow,
                                 int                                           ncol,
                                 const MatrixCOO<std::complex<float>, int>&    src,
                                 MatrixCSR<std::complex<float>, int, PtrType>* dst);

    template bool coo_to_csr_hip(const Rocalution_Backend_Descriptor*           backend,
                                 int64_t                                        nnz,
                                 int                                            nrow,
                                 int                                            ncol,
                                 const MatrixCOO<std::complex<double>, int>&    src,
                                 MatrixCSR<std::complex<double>, int, PtrType>* dst);
#endif

    // csr_to_bcsr
    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*  backend,
                                  int64_t                               nnz,
                                  int                                   nrow,
                                  int                                   ncol,
                                  const MatrixCSR<float, int, PtrType>& src,
                                  const rocsparse_mat_descr             src_descr,
                                  MatrixBCSR<float, int>*               dst,
                                  const rocsparse_mat_descr             dst_descr);

    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*   backend,
                                  int64_t                                nnz,
                                  int                                    nrow,
                                  int                                    ncol,
                                  const MatrixCSR<double, int, PtrType>& src,
                                  const rocsparse_mat_descr              src_descr,
                                  MatrixBCSR<double, int>*               dst,
                                  const rocsparse_mat_descr              dst_descr);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                backend,
                                  int64_t                                             nnz,
                                  int                                                 nrow,
                                  int                                                 ncol,
                                  const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                  const rocsparse_mat_descr                           src_descr,
                                  MatrixBCSR<std::complex<float>, int>*               dst,
                                  const rocsparse_mat_descr                           dst_descr);

    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                 backend,
                                  int64_t                                              nnz,
                                  int                                                  nrow,
                                  int                                                  ncol,
                                  const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                  const rocsparse_mat_descr                            src_descr,
                                  MatrixBCSR<std::complex<double>, int>*               dst,
                                  const rocsparse_mat_descr                            dst_descr);
#endif

    // bcsr_to_csr
    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                  int64_t                              nnz,
                                  int                                  nrow,
                                  int                                  ncol,
                                  const MatrixBCSR<float, int>&        src,
                                  const rocsparse_mat_descr            src_descr,
                                  MatrixCSR<float, int, PtrType>*      dst,
                                  rocsparse_mat_descr                  dst_descr);

    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                  int64_t                              nnz,
                                  int                                  nrow,
                                  int                                  ncol,
                                  const MatrixBCSR<double, int>&       src,
                                  const rocsparse_mat_descr            src_descr,
                                  MatrixCSR<double, int, PtrType>*     dst,
                                  rocsparse_mat_descr                  dst_descr);

#ifdef SUPPORT_COMPLEX
    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                                  int64_t                                       nnz,
                                  int                                           nrow,
                                  int                                           ncol,
                                  const MatrixBCSR<std::complex<float>, int>&   src,
                                  const rocsparse_mat_descr                     src_descr,
                                  MatrixCSR<std::complex<float>, int, PtrType>* dst,
                                  rocsparse_mat_descr                           dst_descr);

    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*           backend,
                                  int64_t                                        nnz,
                                  int                                            nrow,
                                  int                                            ncol,
                                  const MatrixBCSR<std::complex<double>, int>&   src,
                                  const rocsparse_mat_descr                      src_descr,
                                  MatrixCSR<std::complex<double>, int, PtrType>* dst,
                                  rocsparse_mat_descr                            dst_descr);
#endif

    // csr_to_ell
    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*  backend,
                                 int64_t                               nnz,
                                 int                                   nrow,
                                 int                                   ncol,
                                 const MatrixCSR<float, int, PtrType>& src,
                                 const rocsparse_mat_descr             src_descr,
                                 MatrixELL<float, int>*                dst,
                                 const rocsparse_mat_descr             dst_descr,
                                 int64_t*                              nnz_ell);

    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*   backend,
                                 int64_t                                nnz,
                                 int                                    nrow,
                                 int                                    ncol,
                                 const MatrixCSR<double, int, PtrType>& src,
                                 const rocsparse_mat_descr              src_descr,
                                 MatrixELL<double, int>*                dst,
                                 const rocsparse_mat_descr              dst_descr,
                                 int64_t*                               nnz_ell);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                backend,
                                 int64_t                                             nnz,
                                 int                                                 nrow,
                                 int                                                 ncol,
                                 const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                 const rocsparse_mat_descr                           src_descr,
                                 MatrixELL<std::complex<float>, int>*                dst,
                                 const rocsparse_mat_descr                           dst_descr,
                                 int64_t*                                            nnz_ell);

    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                 backend,
                                 int64_t                                              nnz,
                                 int                                                  nrow,
                                 int                                                  ncol,
                                 const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                 const rocsparse_mat_descr                            src_descr,
                                 MatrixELL<std::complex<double>, int>*                dst,
                                 const rocsparse_mat_descr                            dst_descr,
                                 int64_t*                                             nnz_ell);
#endif

    // ell_to_csr
    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                 int64_t                              nnz,
                                 int                                  nrow,
                                 int                                  ncol,
                                 const MatrixELL<float, int>&         src,
                                 const rocsparse_mat_descr            src_descr,
                                 MatrixCSR<float, int, PtrType>*      dst,
                                 const rocsparse_mat_descr            dst_descr,
                                 int64_t*                             nnz_csr);

    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                 int64_t                              nnz,
                                 int                                  nrow,
                                 int                                  ncol,
                                 const MatrixELL<double, int>&        src,
                                 const rocsparse_mat_descr            src_descr,
                                 MatrixCSR<double, int, PtrType>*     dst,
                                 const rocsparse_mat_descr            dst_descr,
                                 int64_t*                             nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                                 int64_t                                       nnz,
                                 int                                           nrow,
                                 int                                           ncol,
                                 const MatrixELL<std::complex<float>, int>&    src,
                                 const rocsparse_mat_descr                     src_descr,
                                 MatrixCSR<std::complex<float>, int, PtrType>* dst,
                                 const rocsparse_mat_descr                     dst_descr,
                                 int64_t*                                      nnz_csr);

    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*           backend,
                                 int64_t                                        nnz,
                                 int                                            nrow,
                                 int                                            ncol,
                                 const MatrixELL<std::complex<double>, int>&    src,
                                 const rocsparse_mat_descr                      src_descr,
                                 MatrixCSR<std::complex<double>, int, PtrType>* dst,
                                 const rocsparse_mat_descr                      dst_descr,
                                 int64_t*                                       nnz_csr);
#endif

    // csr_to_dia
    template bool csr_to_dia_hip(const Rocalution_Backend_Descriptor*  backend,
                                 int64_t                               nnz,
                                 int                                   nrow,
                                 int                                   ncol,
                                 const MatrixCSR<float, int, PtrType>& src,
                                 MatrixDIA<float, int>*                dst,
                                 int64_t*                              nnz_dia,
                                 int*                                  num_diag);

    template bool csr_to_dia_hip(const Rocalution_Backend_Descriptor*   backend,
                                 int64_t                                nnz,
                                 int                                    nrow,
                                 int                                    ncol,
                                 const MatrixCSR<double, int, PtrType>& src,
                                 MatrixDIA<double, int>*                dst,
                                 int64_t*                               nnz_dia,
                                 int*                                   num_diag);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dia_hip(const Rocalution_Backend_Descriptor*                backend,
                                 int64_t                                             nnz,
                                 int                                                 nrow,
                                 int                                                 ncol,
                                 const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                 MatrixDIA<std::complex<float>, int>*                dst,
                                 int64_t*                                            nnz_dia,
                                 int*                                                num_diag);

    template bool csr_to_dia_hip(const Rocalution_Backend_Descriptor*                 backend,
                                 int64_t                                              nnz,
                                 int                                                  nrow,
                                 int                                                  ncol,
                                 const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                 MatrixDIA<std::complex<double>, int>*                dst,
                                 int64_t*                                             nnz_dia,
                                 int*                                                 num_diag);
#endif

    // csr_to_hyb
    template bool csr_to_hyb_hip(const Rocalution_Backend_Descriptor*  backend,
                                 int64_t                               nnz,
                                 int                                   nrow,
                                 int                                   ncol,
                                 const MatrixCSR<float, int, PtrType>& src,
                                 MatrixHYB<float, int>*                dst,
                                 int64_t*                              nnz_hyb,
                                 int64_t*                              nnz_ell,
                                 int64_t*                              nnz_coo);

    template bool csr_to_hyb_hip(const Rocalution_Backend_Descriptor*   backend,
                                 int64_t                                nnz,
                                 int                                    nrow,
                                 int                                    ncol,
                                 const MatrixCSR<double, int, PtrType>& src,
                                 MatrixHYB<double, int>*                dst,
                                 int64_t*                               nnz_hyb,
                                 int64_t*                               nnz_ell,
                                 int64_t*                               nnz_coo);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_hyb_hip(const Rocalution_Backend_Descriptor*                backend,
                                 int64_t                                             nnz,
                                 int                                                 nrow,
                                 int                                                 ncol,
                                 const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                 MatrixHYB<std::complex<float>, int>*                dst,
                                 int64_t*                                            nnz_hyb,
                                 int64_t*                                            nnz_ell,
                                 int64_t*                                            nnz_coo);

    template bool csr_to_hyb_hip(const Rocalution_Backend_Descriptor*                 backend,
                                 int64_t                                              nnz,
                                 int                                                  nrow,
                                 int                                                  ncol,
                                 const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                 MatrixHYB<std::complex<double>, int>*                dst,
                                 int64_t*                                             nnz_hyb,
                                 int64_t*                                             nnz_ell,
                                 int64_t*                                             nnz_coo);
#endif

    // csr_to_dense
    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*  ackend,
                                   int                                   row,
                                   int                                   col,
                                   const MatrixCSR<float, int, PtrType>& src,
                                   const rocsparse_mat_descr             src_descr,
                                   MatrixDENSE<float>*                   dst);

    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*   backend,
                                   int                                    nrow,
                                   int                                    ncol,
                                   const MatrixCSR<double, int, PtrType>& src,
                                   const rocsparse_mat_descr              src_descr,
                                   MatrixDENSE<double>*                   dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*                backend,
                                   int                                                 nrow,
                                   int                                                 ncol,
                                   const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                   const rocsparse_mat_descr                           src_descr,
                                   MatrixDENSE<std::complex<float>>*                   dst);

    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*                 backend,
                                   int                                                  nrow,
                                   int                                                  ncol,
                                   const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                   const rocsparse_mat_descr                            src_descr,
                                   MatrixDENSE<std::complex<double>>*                   dst);
#endif

    // dense_to_csr
    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                   int                                  nrow,
                                   int                                  ncol,
                                   const MatrixDENSE<float>&            src,
                                   MatrixCSR<float, int, PtrType>*      dst,
                                   const rocsparse_mat_descr            dst_descr,
                                   int64_t*                             nnz_csr);

    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                   int                                  nrow,
                                   int                                  ncol,
                                   const MatrixDENSE<double>&           src,
                                   MatrixCSR<double, int, PtrType>*     dst,
                                   const rocsparse_mat_descr            dst_descr,
                                   int64_t*                             nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                                   int                                           nrow,
                                   int                                           ncol,
                                   const MatrixDENSE<std::complex<float>>&       src,
                                   MatrixCSR<std::complex<float>, int, PtrType>* dst,
                                   const rocsparse_mat_descr                     dst_descr,
                                   int64_t*                                      nnz_csr);

    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*           backend,
                                   int                                            nrow,
                                   int                                            ncol,
                                   const MatrixDENSE<std::complex<double>>&       src,
                                   MatrixCSR<std::complex<double>, int, PtrType>* dst,
                                   const rocsparse_mat_descr                      dst_descr,
                                   int64_t*                                       nnz_csr);
#endif

} // namespace rocalution
