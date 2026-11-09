/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "hip_kernels_conversion.hpp"
#include "hip_sparse.hpp"
#include "hip_utils.hpp"
#include "rocalution/utils/types.hpp"

#include <hip/hip_runtime_api.h>
#include <rocprim/rocprim.hpp>

#include <complex>

namespace rocalution
{
    // Runs the analysis and compute stages of rocsparse_sparse_to_sparse. The target sizes are
    // only known after analysis, so allocate_target(nnz) is called in between and has to
    // allocate the remaining target arrays and attach them with rocsparse_*_set_pointers.
    // Returning false from allocate_target skips the compute stage and makes the conversion
    // fail. Analysis already writes the row pointer of CSR and BSR targets, which therefore has
    // to be attached when the target descriptor is created.
    template <typename AllocateTarget>
    static bool sparse_to_sparse_hip(const Rocalution_Backend_Descriptor* backend,
                                     rocsparse_const_spmat_descr          source,
                                     rocsparse_spmat_descr                target,
                                     AllocateTarget&&                     allocate_target)
    {
        rocsparse_handle handle = ROCSPARSE_HANDLE(backend->ROC_sparse_handle);

        rocsparse_sparse_to_sparse_descr descr;
        rocsparse_status                 status = rocsparse_create_sparse_to_sparse_descr(
            &descr, source, target, rocsparse_sparse_to_sparse_alg_default);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size;
        char*  buffer = NULL;

        status = rocsparse_sparse_to_sparse_buffer_size(
            handle, descr, source, target, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_sparse_to_sparse(handle,
                                            descr,
                                            source,
                                            target,
                                            rocsparse_sparse_to_sparse_stage_analysis,
                                            buffer_size,
                                            buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        int64_t target_nrow;
        int64_t target_ncol;
        int64_t target_nnz;

        status = rocsparse_spmat_get_size(target, &target_nrow, &target_ncol, &target_nnz);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(allocate_target(target_nnz) == false)
        {
            status = rocsparse_destroy_sparse_to_sparse_descr(descr);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return false;
        }

        status = rocsparse_sparse_to_sparse_buffer_size(
            handle, descr, source, target, rocsparse_sparse_to_sparse_stage_compute, &buffer_size);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_sparse_to_sparse(handle,
                                            descr,
                                            source,
                                            target,
                                            rocsparse_sparse_to_sparse_stage_compute,
                                            buffer_size,
                                            buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_destroy_sparse_to_sparse_descr(descr);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

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

        rocsparse_const_spmat_descr source;
        rocsparse_status            status
            = rocsparse_create_const_csr_descr(&source,
                                               nrow,
                                               ncol,
                                               nnz,
                                               src.row_offset,
                                               src.col,
                                               src.val,
                                               rocalution_indextype_traits<PointerType>::value,
                                               rocalution_indextype_traits<IndexType>::value,
                                               rocsparse_index_base_zero,
                                               rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        rocsparse_spmat_descr target;
        status = rocsparse_create_coo_descr(&target,
                                            nrow,
                                            ncol,
                                            0,
                                            NULL,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnz) {
            assert(target_nnz == nnz);

            allocate_hip(target_nnz, &dst->row);
            allocate_hip(target_nnz, &dst->col);
            allocate_hip(target_nnz, &dst->val);

            status = rocsparse_coo_set_pointers(target, dst->row, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
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

        assert(src.row != NULL);
        assert(src.col != NULL);
        assert(src.val != NULL);

        assert(dst != NULL);
        assert(backend != NULL);

        rocsparse_const_spmat_descr source;
        rocsparse_status            status
            = rocsparse_create_const_coo_descr(&source,
                                               nrow,
                                               ncol,
                                               nnz,
                                               src.row,
                                               src.col,
                                               src.val,
                                               rocalution_indextype_traits<IndexType>::value,
                                               rocsparse_index_base_zero,
                                               rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nrow + 1, &dst->row_offset);

        rocsparse_spmat_descr target;
        status = rocsparse_create_csr_descr(&target,
                                            nrow,
                                            ncol,
                                            0,
                                            dst->row_offset,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<PointerType>::value,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnz) {
            assert(target_nnz == nnz);

            allocate_hip(target_nnz, &dst->col);
            allocate_hip(target_nnz, &dst->val);

            status = rocsparse_csr_set_pointers(target, dst->row_offset, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_csc_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixCSR<ValueType, IndexType, PointerType>*       dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.row_offset != NULL);
        assert(src.col != NULL);
        assert(src.val != NULL);

        assert(dst != NULL);
        assert(backend != NULL);

        rocsparse_const_spmat_descr source;
        rocsparse_status            status
            = rocsparse_create_const_csr_descr(&source,
                                               nrow,
                                               ncol,
                                               nnz,
                                               src.row_offset,
                                               src.col,
                                               src.val,
                                               rocalution_indextype_traits<PointerType>::value,
                                               rocalution_indextype_traits<IndexType>::value,
                                               rocsparse_index_base_zero,
                                               rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(ncol + 1, &dst->row_offset);

        rocsparse_spmat_descr target;
        status = rocsparse_create_csc_descr(&target,
                                            nrow,
                                            ncol,
                                            0,
                                            dst->row_offset,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<PointerType>::value,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnz) {
            assert(target_nnz == nnz);

            allocate_hip(target_nnz, &dst->col);
            allocate_hip(target_nnz, &dst->val);

            status = rocsparse_csc_set_pointers(target, dst->row_offset, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                backend,
                         int64_t                                             nnz,
                         IndexType                                           nrow,
                         IndexType                                           ncol,
                         const MatrixCSR<ValueType, IndexType, PointerType>& src,
                         MatrixBCSR<ValueType, IndexType, PointerType>*      dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.row_offset != NULL);
        assert(src.col != NULL);
        assert(src.val != NULL);

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

        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        rocsparse_const_spmat_descr source;
        rocsparse_status            status
            = rocsparse_create_const_csr_descr(&source,
                                               nrow,
                                               ncol,
                                               nnz,
                                               src.row_offset,
                                               src.col,
                                               src.val,
                                               rocalution_indextype_traits<PointerType>::value,
                                               rocalution_indextype_traits<IndexType>::value,
                                               rocsparse_index_base_zero,
                                               rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(mb + 1, &dst->row_offset);

        rocsparse_spmat_descr target;
        status = rocsparse_create_bsr_descr(&target,
                                            mb,
                                            nb,
                                            0,
                                            dir,
                                            blockdim,
                                            dst->row_offset,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<PointerType>::value,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnzb) {
            allocate_hip(target_nnzb, &dst->col);
            allocate_hip(target_nnzb * blockdim * blockdim, &dst->val);

            status = rocsparse_bsr_set_pointers(target, dst->row_offset, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            dst->nrowb = mb;
            dst->ncolb = nb;
            dst->nnzb  = target_nnzb;

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*                 backend,
                         int64_t                                              nnz,
                         IndexType                                            nrow,
                         IndexType                                            ncol,
                         const MatrixBCSR<ValueType, IndexType, PointerType>& src,
                         MatrixCSR<ValueType, IndexType, PointerType>*        dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.row_offset != NULL);
        assert(src.col != NULL);
        assert(src.val != NULL);

        assert(dst != NULL);
        assert(backend != NULL);

        IndexType blockdim = src.blockdim;

        assert(blockdim > 1);

        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        // Not a const descriptor: the BSR to CSR conversion of rocSPARSE reads the column
        // indices of the source through the non-const pointer, which a const descriptor leaves
        // NULL.
        rocsparse_spmat_descr source;
        rocsparse_status      status
            = rocsparse_create_bsr_descr(&source,
                                         src.nrowb,
                                         src.ncolb,
                                         src.nnzb,
                                         dir,
                                         blockdim,
                                         src.row_offset,
                                         src.col,
                                         src.val,
                                         rocalution_indextype_traits<PointerType>::value,
                                         rocalution_indextype_traits<IndexType>::value,
                                         rocsparse_index_base_zero,
                                         rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nrow + 1, &dst->row_offset);

        rocsparse_spmat_descr target;
        status = rocsparse_create_csr_descr(&target,
                                            nrow,
                                            ncol,
                                            0,
                                            dst->row_offset,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<PointerType>::value,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnz) {
            assert(target_nnz == nnz);

            allocate_hip(target_nnz, &dst->col);
            allocate_hip(target_nnz, &dst->val);

            status = rocsparse_csr_set_pointers(target, dst->row_offset, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                backend,
                        int64_t                                             nnz,
                        IndexType                                           nrow,
                        IndexType                                           ncol,
                        const MatrixCSR<ValueType, IndexType, PointerType>& src,
                        MatrixELL<ValueType, IndexType>*                    dst,
                        int64_t*                                            nnz_ell)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.row_offset != NULL);
        assert(src.col != NULL);
        assert(src.val != NULL);

        assert(dst != NULL);
        assert(nnz_ell != NULL);
        assert(backend != NULL);

        rocsparse_const_spmat_descr source;
        rocsparse_status            status
            = rocsparse_create_const_csr_descr(&source,
                                               nrow,
                                               ncol,
                                               nnz,
                                               src.row_offset,
                                               src.col,
                                               src.val,
                                               rocalution_indextype_traits<PointerType>::value,
                                               rocalution_indextype_traits<IndexType>::value,
                                               rocsparse_index_base_zero,
                                               rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        rocsparse_spmat_descr target;
        status = rocsparse_create_ell_descr(&target,
                                            nrow,
                                            ncol,
                                            NULL,
                                            NULL,
                                            0,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnz) {
            int64_t              ell_nrow;
            int64_t              ell_ncol;
            int64_t              ell_width;
            void*                ell_col;
            void*                ell_val;
            rocsparse_indextype  idx_type;
            rocsparse_index_base idx_base;
            rocsparse_datatype   data_type;

            status = rocsparse_ell_get(target,
                                       &ell_nrow,
                                       &ell_ncol,
                                       &ell_col,
                                       &ell_val,
                                       &ell_width,
                                       &idx_type,
                                       &idx_base,
                                       &data_type);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Limit ELL size to 5 times CSR nnz
            if(ell_width > 5 * (nnz / nrow))
            {
                return false;
            }

            assert(target_nnz == ell_width * nrow);

            dst->max_row = ell_width;
            *nnz_ell     = target_nnz;

            allocate_hip(target_nnz, &dst->col);
            allocate_hip(target_nnz, &dst->val);

            status = rocsparse_ell_set_pointers(target, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                        int64_t                                       nnz,
                        IndexType                                     nrow,
                        IndexType                                     ncol,
                        const MatrixELL<ValueType, IndexType>&        src,
                        MatrixCSR<ValueType, IndexType, PointerType>* dst,
                        int64_t*                                      nnz_csr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.col != NULL);
        assert(src.val != NULL);

        assert(dst != NULL);
        assert(nnz_csr != NULL);
        assert(backend != NULL);

        rocsparse_spmat_descr source;
        rocsparse_status      status
            = rocsparse_create_ell_descr(&source,
                                         nrow,
                                         ncol,
                                         src.col,
                                         src.val,
                                         src.max_row,
                                         rocalution_indextype_traits<IndexType>::value,
                                         rocsparse_index_base_zero,
                                         rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nrow + 1, &dst->row_offset);

        rocsparse_spmat_descr target;
        status = rocsparse_create_csr_descr(&target,
                                            nrow,
                                            ncol,
                                            0,
                                            dst->row_offset,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<PointerType>::value,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        bool converted = sparse_to_sparse_hip(backend, source, target, [&](int64_t target_nnz) {
            *nnz_csr = target_nnz;

            allocate_hip(target_nnz, &dst->col);
            allocate_hip(target_nnz, &dst->val);

            status = rocsparse_csr_set_pointers(target, dst->row_offset, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            return true;
        });

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return converted;
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
        hipStream_t stream = HIPSTREAM(_get_backend_descriptor()->HIP_stream_current);

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
        CHECK_HIP_STATUS(rocprim::reduce(rocprim_buffer,
                                         rocprim_size,
                                         diag_idx,
                                         d_num_diag,
                                         0,
                                         nrow + ncol,
                                         rocprim::plus<IndexType>(),
                                         stream));

        // Allocate rocprim buffer
        allocate_hip(rocprim_size, &rocprim_buffer);

        // Do reduction
        CHECK_HIP_STATUS(rocprim::reduce(rocprim_buffer,
                                         rocprim_size,
                                         diag_idx,
                                         d_num_diag,
                                         0,
                                         nrow + ncol,
                                         rocprim::plus<IndexType>(),
                                         stream));

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
        CHECK_HIP_STATUS(rocprim::exclusive_scan(rocprim_buffer,
                                                 rocprim_size,
                                                 diag_idx,
                                                 work,
                                                 0,
                                                 nrow + ncol,
                                                 rocprim::plus<IndexType>(),
                                                 stream));

        // Allocate rocprim buffer
        rocprim_buffer = NULL;
        allocate_hip(rocprim_size, &rocprim_buffer);

        // Do inclusive sum
        CHECK_HIP_STATUS(rocprim::exclusive_scan(rocprim_buffer,
                                                 rocprim_size,
                                                 diag_idx,
                                                 work,
                                                 0,
                                                 nrow + ncol,
                                                 rocprim::plus<IndexType>(),
                                                 stream));

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
        hipStream_t stream = HIPSTREAM(_get_backend_descriptor()->HIP_stream_current);

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
            copy_d2d(nrow + 1, src.row_offset, coo_row_nnz, true, stream);
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
            CHECK_HIP_STATUS(rocprim::exclusive_scan(rocprim_buffer,
                                                     rocprim_size,
                                                     coo_row_nnz,
                                                     coo_row_nnz,
                                                     0,
                                                     nrow + 1,
                                                     rocprim::plus<PointerType>(),
                                                     stream));

            // Allocate rocprim buffer
            allocate_hip(rocprim_size, &rocprim_buffer);

            // Do exclusive sum
            CHECK_HIP_STATUS(rocprim::exclusive_scan(rocprim_buffer,
                                                     rocprim_size,
                                                     coo_row_nnz,
                                                     coo_row_nnz,
                                                     0,
                                                     nrow + 1,
                                                     rocprim::plus<PointerType>(),
                                                     stream));

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
                          int64_t                                             nnz,
                          IndexType                                           nrow,
                          IndexType                                           ncol,
                          const MatrixCSR<ValueType, IndexType, PointerType>& src,
                          MatrixDENSE<ValueType>*                             dst)
    {
        assert(nnz >= 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.row_offset != NULL);
        assert(nnz == 0 || src.col != NULL);
        assert(nnz == 0 || src.val != NULL);

        assert(dst != NULL);
        assert(backend != NULL);

        rocsparse_handle handle = ROCSPARSE_HANDLE(backend->ROC_sparse_handle);

        rocsparse_const_spmat_descr source;
        rocsparse_status            status
            = rocsparse_create_const_csr_descr(&source,
                                               nrow,
                                               ncol,
                                               nnz,
                                               src.row_offset,
                                               src.col,
                                               src.val,
                                               rocalution_indextype_traits<PointerType>::value,
                                               rocalution_indextype_traits<IndexType>::value,
                                               rocsparse_index_base_zero,
                                               rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nrow * ncol, &dst->val);

        rocsparse_dnmat_descr target;
        status = rocsparse_create_dnmat_descr(&target,
                                              nrow,
                                              ncol,
                                              DENSE_IND_BASE == 0 ? nrow : ncol,
                                              dst->val,
                                              rocalution_datatype_traits<ValueType>::value,
                                              DENSE_IND_BASE == 0 ? rocsparse_order_column
                                                                  : rocsparse_order_row);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size;
        char*  buffer = NULL;

        status = rocsparse_sparse_to_dense(
            handle, source, target, rocsparse_sparse_to_dense_alg_default, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_sparse_to_dense(
            handle, source, target, rocsparse_sparse_to_dense_alg_default, &buffer_size, buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_destroy_dnmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_spmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                          IndexType                                     nrow,
                          IndexType                                     ncol,
                          const MatrixDENSE<ValueType>&                 src,
                          MatrixCSR<ValueType, IndexType, PointerType>* dst,
                          int64_t*                                      nnz_csr)
    {
        assert(nrow > 0);
        assert(ncol > 0);

        assert(src.val != NULL);

        assert(dst != NULL);
        assert(nnz_csr != NULL);
        assert(backend != NULL);

        rocsparse_handle handle = ROCSPARSE_HANDLE(backend->ROC_sparse_handle);

        rocsparse_const_dnmat_descr source;
        rocsparse_status            status = rocsparse_create_const_dnmat_descr(
            &source,
            nrow,
            ncol,
            DENSE_IND_BASE == 0 ? nrow : ncol,
            src.val,
            rocalution_datatype_traits<ValueType>::value,
            DENSE_IND_BASE == 0 ? rocsparse_order_column : rocsparse_order_row);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nrow + 1, &dst->row_offset);

        rocsparse_spmat_descr target;
        status = rocsparse_create_csr_descr(&target,
                                            nrow,
                                            ncol,
                                            0,
                                            dst->row_offset,
                                            NULL,
                                            NULL,
                                            rocalution_indextype_traits<PointerType>::value,
                                            rocalution_indextype_traits<IndexType>::value,
                                            rocsparse_index_base_zero,
                                            rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size;
        char*  buffer = NULL;

        status = rocsparse_dense_to_sparse(
            handle, source, target, rocsparse_dense_to_sparse_alg_default, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        // Analysis
        status = rocsparse_dense_to_sparse(
            handle, source, target, rocsparse_dense_to_sparse_alg_default, NULL, buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        int64_t target_nrow;
        int64_t target_ncol;
        int64_t target_nnz;

        status = rocsparse_spmat_get_size(target, &target_nrow, &target_ncol, &target_nnz);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(target_nnz, &dst->col);
        allocate_hip(target_nnz, &dst->val);

        if(target_nnz > 0)
        {
            status = rocsparse_csr_set_pointers(target, dst->row_offset, dst->col, dst->val);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Compute
            status = rocsparse_dense_to_sparse(handle,
                                               source,
                                               target,
                                               rocsparse_dense_to_sparse_alg_default,
                                               &buffer_size,
                                               buffer);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            // rocSPARSE requires column and value arrays for the compute stage even without
            // non-zeros, an empty matrix only needs zero row offsets
            set_to_zero_hip(backend->HIP_block_size, nrow + 1, dst->row_offset);
        }

        free_hip(&buffer);

        status = rocsparse_destroy_spmat_descr(target);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_dnmat_descr(source);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        *nnz_csr = target_nnz;

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

    // csr_to_csc
    template bool csr_to_csc_hip(const Rocalution_Backend_Descriptor*  backend,
                                 int64_t                               nnz,
                                 int                                   nrow,
                                 int                                   ncol,
                                 const MatrixCSR<float, int, PtrType>& src,
                                 MatrixCSR<float, int, PtrType>*       dst);

    template bool csr_to_csc_hip(const Rocalution_Backend_Descriptor*   backend,
                                 int64_t                                nnz,
                                 int                                    nrow,
                                 int                                    ncol,
                                 const MatrixCSR<double, int, PtrType>& src,
                                 MatrixCSR<double, int, PtrType>*       dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_csc_hip(const Rocalution_Backend_Descriptor*                backend,
                                 int64_t                                             nnz,
                                 int                                                 nrow,
                                 int                                                 ncol,
                                 const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                 MatrixCSR<std::complex<float>, int, PtrType>*       dst);

    template bool csr_to_csc_hip(const Rocalution_Backend_Descriptor*                 backend,
                                 int64_t                                              nnz,
                                 int                                                  nrow,
                                 int                                                  ncol,
                                 const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                 MatrixCSR<std::complex<double>, int, PtrType>*       dst);
#endif

    // csr_to_bcsr
    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*  backend,
                                  int64_t                               nnz,
                                  int                                   nrow,
                                  int                                   ncol,
                                  const MatrixCSR<float, int, PtrType>& src,
                                  MatrixBCSR<float, int, PtrType>*      dst);

    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*   backend,
                                  int64_t                                nnz,
                                  int                                    nrow,
                                  int                                    ncol,
                                  const MatrixCSR<double, int, PtrType>& src,
                                  MatrixBCSR<double, int, PtrType>*      dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                backend,
                                  int64_t                                             nnz,
                                  int                                                 nrow,
                                  int                                                 ncol,
                                  const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                  MatrixBCSR<std::complex<float>, int, PtrType>*      dst);

    template bool csr_to_bcsr_hip(const Rocalution_Backend_Descriptor*                 backend,
                                  int64_t                                              nnz,
                                  int                                                  nrow,
                                  int                                                  ncol,
                                  const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                  MatrixBCSR<std::complex<double>, int, PtrType>*      dst);
#endif

    // bcsr_to_csr
    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*   backend,
                                  int64_t                                nnz,
                                  int                                    nrow,
                                  int                                    ncol,
                                  const MatrixBCSR<float, int, PtrType>& src,
                                  MatrixCSR<float, int, PtrType>*        dst);

    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*    backend,
                                  int64_t                                 nnz,
                                  int                                     nrow,
                                  int                                     ncol,
                                  const MatrixBCSR<double, int, PtrType>& src,
                                  MatrixCSR<double, int, PtrType>*        dst);

#ifdef SUPPORT_COMPLEX
    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*                 backend,
                                  int64_t                                              nnz,
                                  int                                                  nrow,
                                  int                                                  ncol,
                                  const MatrixBCSR<std::complex<float>, int, PtrType>& src,
                                  MatrixCSR<std::complex<float>, int, PtrType>*        dst);

    template bool bcsr_to_csr_hip(const Rocalution_Backend_Descriptor*                  backend,
                                  int64_t                                               nnz,
                                  int                                                   nrow,
                                  int                                                   ncol,
                                  const MatrixBCSR<std::complex<double>, int, PtrType>& src,
                                  MatrixCSR<std::complex<double>, int, PtrType>*        dst);
#endif

    // csr_to_ell
    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*  backend,
                                 int64_t                               nnz,
                                 int                                   nrow,
                                 int                                   ncol,
                                 const MatrixCSR<float, int, PtrType>& src,
                                 MatrixELL<float, int>*                dst,
                                 int64_t*                              nnz_ell);

    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*   backend,
                                 int64_t                                nnz,
                                 int                                    nrow,
                                 int                                    ncol,
                                 const MatrixCSR<double, int, PtrType>& src,
                                 MatrixELL<double, int>*                dst,
                                 int64_t*                               nnz_ell);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                backend,
                                 int64_t                                             nnz,
                                 int                                                 nrow,
                                 int                                                 ncol,
                                 const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                 MatrixELL<std::complex<float>, int>*                dst,
                                 int64_t*                                            nnz_ell);

    template bool csr_to_ell_hip(const Rocalution_Backend_Descriptor*                 backend,
                                 int64_t                                              nnz,
                                 int                                                  nrow,
                                 int                                                  ncol,
                                 const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                 MatrixELL<std::complex<double>, int>*                dst,
                                 int64_t*                                             nnz_ell);
#endif

    // ell_to_csr
    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                 int64_t                              nnz,
                                 int                                  nrow,
                                 int                                  ncol,
                                 const MatrixELL<float, int>&         src,
                                 MatrixCSR<float, int, PtrType>*      dst,
                                 int64_t*                             nnz_csr);

    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                 int64_t                              nnz,
                                 int                                  nrow,
                                 int                                  ncol,
                                 const MatrixELL<double, int>&        src,
                                 MatrixCSR<double, int, PtrType>*     dst,
                                 int64_t*                             nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                                 int64_t                                       nnz,
                                 int                                           nrow,
                                 int                                           ncol,
                                 const MatrixELL<std::complex<float>, int>&    src,
                                 MatrixCSR<std::complex<float>, int, PtrType>* dst,
                                 int64_t*                                      nnz_csr);

    template bool ell_to_csr_hip(const Rocalution_Backend_Descriptor*           backend,
                                 int64_t                                        nnz,
                                 int                                            nrow,
                                 int                                            ncol,
                                 const MatrixELL<std::complex<double>, int>&    src,
                                 MatrixCSR<std::complex<double>, int, PtrType>* dst,
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
    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*  backend,
                                   int64_t                               nnz,
                                   int                                   nrow,
                                   int                                   ncol,
                                   const MatrixCSR<float, int, PtrType>& src,
                                   MatrixDENSE<float>*                   dst);

    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*   backend,
                                   int64_t                                nnz,
                                   int                                    nrow,
                                   int                                    ncol,
                                   const MatrixCSR<double, int, PtrType>& src,
                                   MatrixDENSE<double>*                   dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*                backend,
                                   int64_t                                             nnz,
                                   int                                                 nrow,
                                   int                                                 ncol,
                                   const MatrixCSR<std::complex<float>, int, PtrType>& src,
                                   MatrixDENSE<std::complex<float>>*                   dst);

    template bool csr_to_dense_hip(const Rocalution_Backend_Descriptor*                 backend,
                                   int64_t                                              nnz,
                                   int                                                  nrow,
                                   int                                                  ncol,
                                   const MatrixCSR<std::complex<double>, int, PtrType>& src,
                                   MatrixDENSE<std::complex<double>>*                   dst);
#endif

    // dense_to_csr
    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                   int                                  nrow,
                                   int                                  ncol,
                                   const MatrixDENSE<float>&            src,
                                   MatrixCSR<float, int, PtrType>*      dst,
                                   int64_t*                             nnz_csr);

    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor* backend,
                                   int                                  nrow,
                                   int                                  ncol,
                                   const MatrixDENSE<double>&           src,
                                   MatrixCSR<double, int, PtrType>*     dst,
                                   int64_t*                             nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*          backend,
                                   int                                           nrow,
                                   int                                           ncol,
                                   const MatrixDENSE<std::complex<float>>&       src,
                                   MatrixCSR<std::complex<float>, int, PtrType>* dst,
                                   int64_t*                                      nnz_csr);

    template bool dense_to_csr_hip(const Rocalution_Backend_Descriptor*           backend,
                                   int                                            nrow,
                                   int                                            ncol,
                                   const MatrixDENSE<std::complex<double>>&       src,
                                   MatrixCSR<std::complex<double>, int, PtrType>* dst,
                                   int64_t*                                       nnz_csr);
#endif

} // namespace rocalution
