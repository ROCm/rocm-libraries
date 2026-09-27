/*! \file */
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
#include "internal/conversion/rocsparse_csrsort.h"
#include "rocsparse_utility.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "csrsort_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsort.hpp"
#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    // Number of bits needed to represent the column indices, which are at most n.
    static uint32_t csrsort_endbit(int64_t n)
    {
        // __builtin_clzll is undefined for n == 0
        return (n == 0) ? 0 : 64 - __builtin_clzll(static_cast<unsigned long long>(n));
    }

    template <typename I, typename J>
    static rocsparse_status csrsort_buffer_size_template(rocsparse_handle handle,
                                                         int64_t          m,
                                                         int64_t          n,
                                                         int64_t          nnz,
                                                         const I*         csr_row_ptr,
                                                         const J*         csr_col_ind,
                                                         size_t*          buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || nnz == 0)
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        const uint32_t startbit = 0;
        const uint32_t endbit   = rocsparse::csrsort_endbit(n);

        // We do not know if sort_pairs or sort_keys will be called, so use the largest buffer between the two
        size_t size1;
        size_t size2;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
                handle, nnz, m, startbit, endbit, &size1)));
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
                handle, nnz, m, startbit, endbit, &size2)));

        *buffer_size = rocsparse::align_size<char>(rocsparse::max(size1, size2));

        // rocPRIM does not support in-place sorting, so we need additional buffer
        // for all temporary arrays

        // columns buffer
        *buffer_size += rocsparse::align_size<J>(nnz);
        // perm buffer
        *buffer_size += rocsparse::align_size<I>(nnz);
        // segm buffer
        *buffer_size += rocsparse::align_size<I>(m + 1);

        return rocsparse_status_success;
    }

    // Sorts the column indices within each row. If perm is not null, the sort applies its
    // reordering to perm.
    template <typename I, typename J>
    static rocsparse_status csrsort_template(rocsparse_handle     handle,
                                             int64_t              m,
                                             int64_t              n,
                                             int64_t              nnz,
                                             rocsparse_index_base idx_base,
                                             const I*             csr_row_ptr,
                                             J*                   csr_col_ind,
                                             I*                   perm,
                                             void*                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }

        // Stream
        hipStream_t stream = handle->stream;

        const uint32_t startbit = 0;
        const uint32_t endbit   = rocsparse::csrsort_endbit(n);
        size_t         size;

        if(perm != nullptr)
        {
            // Sort pairs, if permutation vector is present
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
                    handle, nnz, m, startbit, endbit, &size)));
        }
        else
        {
            // Sort keys, if no permutation vector is present
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
                    handle, nnz, m, startbit, endbit, &size)));
        }

        // Temporary buffer entry points
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        // columns buffer
        J* tmp_cols = reinterpret_cast<J*>(ptr);
        ptr += rocsparse::align_size<J>(nnz);

        // perm buffer
        I* tmp_perm = reinterpret_cast<I*>(ptr);
        ptr += rocsparse::align_size<I>(nnz);

        // segm buffer
        I* tmp_segm = reinterpret_cast<I*>(ptr);
        ptr += rocsparse::align_size<I>(m + 1);

        // Index base one requires shift of offset positions
        if(idx_base == rocsparse_index_base_one)
        {
#define CSRSORT_DIM 512
            dim3 csrsort_blocks((m + 1 - 1) / CSRSORT_DIM + 1);
            dim3 csrsort_threads(CSRSORT_DIM);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrsort_shift_kernel<CSRSORT_DIM>),
                                               csrsort_blocks,
                                               csrsort_threads,
                                               0,
                                               stream,
                                               m + 1,
                                               csr_row_ptr,
                                               tmp_segm);
#undef CSRSORT_DIM
        }

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        // Switch between offsets
        const I* offsets = (idx_base == rocsparse_index_base_one) ? tmp_segm : csr_row_ptr;

        // Sort by columns and obtain permutation vector

        if(perm != nullptr)
        {
            // Sort by pairs, if permutation vector is present
            rocsparse::primitives::double_buffer<J> keys(csr_col_ind, tmp_cols);
            rocsparse::primitives::double_buffer<I> vals(perm, tmp_perm);

            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::primitives::segmented_radix_sort_pairs(handle,
                                                                  keys,
                                                                  vals,
                                                                  nnz,
                                                                  m,
                                                                  offsets,
                                                                  offsets + 1,
                                                                  startbit,
                                                                  endbit,
                                                                  size,
                                                                  tmp_rocprim));

            if(keys.current() != csr_col_ind)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                    csr_col_ind, keys.current(), sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
            }
            if(vals.current() != perm)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                    perm, vals.current(), sizeof(I) * nnz, hipMemcpyDeviceToDevice, stream));
            }
        }
        else
        {
            // Sort by keys, if no permutation vector is present
            rocsparse::primitives::double_buffer<J> keys(csr_col_ind, tmp_cols);

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(
                handle, keys, nnz, m, offsets, offsets + 1, startbit, endbit, size, tmp_rocprim));

            if(keys.current() != csr_col_ind)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                    csr_col_ind, keys.current(), sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
            }
        }
        return rocsparse_status_success;
    }

}

extern "C" rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrsort_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::csrsort_buffer_size_template<rocsparse_int, rocsparse_int>(
            handle, m, n, nnz, csr_row_ptr, csr_col_ind, buffer_size)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csrsort(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int*      csr_row_ptr,
                                              rocsparse_int*            csr_col_ind,
                                              rocsparse_int*            perm,
                                              void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csrsort",
                         m,
                         n,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsort_template<rocsparse_int, rocsparse_int>(
        handle, m, n, nnz, descr->base, csr_row_ptr, csr_col_ind, perm, temp_buffer)));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    static rocsparse_status gcsrsort_buffer_size(rocsparse_handle    handle,
                                                 int64_t             m,
                                                 int64_t             n,
                                                 int64_t             nnz,
                                                 rocsparse_indextype ptr_type,
                                                 rocsparse_indextype ind_type,
                                                 const void*         ptr,
                                                 const void*         ind,
                                                 size_t*             buffer_size)
    {
        if(ptr_type == rocsparse_indextype_i32 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_buffer_size_template(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        reinterpret_cast<const int32_t*>(ptr),
                                                        reinterpret_cast<const int32_t*>(ind),
                                                        buffer_size));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_buffer_size_template(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        reinterpret_cast<const int64_t*>(ptr),
                                                        reinterpret_cast<const int32_t*>(ind),
                                                        buffer_size));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_buffer_size_template(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        reinterpret_cast<const int64_t*>(ptr),
                                                        reinterpret_cast<const int64_t*>(ind),
                                                        buffer_size));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the combination of offsets and indices types is not supported");
    }

    static rocsparse_status gcsrsort(rocsparse_handle     handle,
                                     int64_t              m,
                                     int64_t              n,
                                     int64_t              nnz,
                                     rocsparse_index_base idx_base,
                                     rocsparse_indextype  ptr_type,
                                     rocsparse_indextype  ind_type,
                                     const void*          ptr,
                                     void*                ind,
                                     void*                perm,
                                     void*                buffer)
    {
        if(ptr_type == rocsparse_indextype_i32 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_template(handle,
                                            m,
                                            n,
                                            nnz,
                                            idx_base,
                                            reinterpret_cast<const int32_t*>(ptr),
                                            reinterpret_cast<int32_t*>(ind),
                                            reinterpret_cast<int32_t*>(perm),
                                            buffer));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_template(handle,
                                            m,
                                            n,
                                            nnz,
                                            idx_base,
                                            reinterpret_cast<const int64_t*>(ptr),
                                            reinterpret_cast<int32_t*>(ind),
                                            reinterpret_cast<int64_t*>(perm),
                                            buffer));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_template(handle,
                                            m,
                                            n,
                                            nnz,
                                            idx_base,
                                            reinterpret_cast<const int64_t*>(ptr),
                                            reinterpret_cast<int64_t*>(ind),
                                            reinterpret_cast<int64_t*>(perm),
                                            buffer));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the combination of offsets and indices types is not supported");
    }

    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the indices are sorted, followed by scratch space shared by the index sort
    // and the value permutation.
    static size_t csxsort_perm_size(int64_t nnz, rocsparse_indextype perm_indextype)
    {
        return rocsparse::align_size<char>(rocsparse::indextype_sizeof(perm_indextype) * nnz);
    }
}

rocsparse_status rocsparse::csxsort_buffer_size(rocsparse_handle            handle,
                                                rocsparse_direction         dir,
                                                rocsparse_const_spmat_descr source,
                                                rocsparse_const_spmat_descr target,
                                                size_t*                     buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    // A CSC matrix is sorted as the CSR layout of its transpose.
    const bool                by_row   = (dir == rocsparse_direction_row);
    const int64_t             m        = by_row ? target->rows : target->cols;
    const int64_t             n        = by_row ? target->cols : target->rows;
    const int64_t             nnz      = target->nnz;
    const rocsparse_indextype ptr_type = by_row ? target->row_type : target->col_type;
    const rocsparse_indextype ind_type = by_row ? target->col_type : target->row_type;
    const void*               ptr      = by_row ? target->const_row_data : target->const_col_data;
    const void*               ind      = by_row ? target->const_col_data : target->const_row_data;

    size_t sort_buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsrsort_buffer_size(
        handle, m, n, nnz, ptr_type, ind_type, ptr, ind, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (target->const_val_data == source->const_val_data)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(target->data_type) * nnz)
              : 0;

    *buffer_size_in_bytes = rocsparse::csxsort_perm_size(nnz, ptr_type)
                            + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csxsort(rocsparse_handle            handle,
                                    rocsparse_direction         dir,
                                    rocsparse_const_spmat_descr source,
                                    rocsparse_spmat_descr       target,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    size_t required_buffer_size;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csxsort_buffer_size(handle, dir, source, target, &required_buffer_size));
    if(buffer_size_in_bytes < required_buffer_size)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_size,
            "the buffer is smaller than the size returned by the buffer size query");
    }

    // A CSC matrix is sorted as the CSR layout of its transpose.
    const bool                 by_row    = (dir == rocsparse_direction_row);
    const int64_t              m         = by_row ? target->rows : target->cols;
    const int64_t              n         = by_row ? target->cols : target->rows;
    const int64_t              nnz       = target->nnz;
    const rocsparse_index_base idx_base  = target->idx_base;
    const rocsparse_indextype  ptr_type  = by_row ? target->row_type : target->col_type;
    const rocsparse_indextype  ind_type  = by_row ? target->col_type : target->row_type;
    const rocsparse_datatype   data_type = target->data_type;

    const void* ptr_source = by_row ? source->const_row_data : source->const_col_data;
    const void* ind_source = by_row ? source->const_col_data : source->const_row_data;
    void*       ptr_target = by_row ? target->row_data : target->col_data;
    void*       ind_target = by_row ? target->col_data : target->row_data;

    const size_t ptr_size = rocsparse::indextype_sizeof(ptr_type);
    const size_t ind_size = rocsparse::indextype_sizeof(ind_type);
    const size_t val_size = rocsparse::datatype_sizeof(data_type);

    void* perm = buffer;
    void* sort_buffer
        = reinterpret_cast<char*>(buffer) + rocsparse::csxsort_perm_size(nnz, ptr_type);

    // The batches run one after the other on the handle stream, so they share the buffer.
    for(int64_t batch = 0; batch < source->batch_count; ++batch)
    {
        const int64_t offsets_offset_source = batch * source->offsets_batch_stride;
        const int64_t offsets_offset_target = batch * target->offsets_batch_stride;
        const int64_t cv_offset_source      = batch * source->columns_values_batch_stride;
        const int64_t cv_offset_target      = batch * target->columns_values_batch_stride;

        const void* batch_ptr_source
            = reinterpret_cast<const char*>(ptr_source) + offsets_offset_source * ptr_size;
        const void* batch_ind_source
            = reinterpret_cast<const char*>(ind_source) + cv_offset_source * ind_size;
        const void* batch_val_source
            = reinterpret_cast<const char*>(source->const_val_data) + cv_offset_source * val_size;
        void* batch_ptr_target
            = reinterpret_cast<char*>(ptr_target) + offsets_offset_target * ptr_size;
        void* batch_ind_target = reinterpret_cast<char*>(ind_target) + cv_offset_target * ind_size;
        void* batch_val_target
            = reinterpret_cast<char*>(target->val_data) + cv_offset_target * val_size;

        // The index sort works in place, so the offsets and indices of source are first copied
        // into target. A matrix without rows may have a null offsets array.
        if(m > 0 && batch_ptr_target != batch_ptr_source)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_ptr_target,
                                                         batch_ptr_source,
                                                         ptr_size * (m + 1),
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
        }
        if(batch_ind_target != batch_ind_source)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_ind_target,
                                                         batch_ind_source,
                                                         ind_size * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
        }

        // The index sort applies its reordering to perm, so it must start as the identity.
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, nnz, ptr_type, perm));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsrsort(handle,
                                                      m,
                                                      n,
                                                      nnz,
                                                      idx_base,
                                                      ptr_type,
                                                      ind_type,
                                                      batch_ptr_target,
                                                      batch_ind_target,
                                                      perm,
                                                      sort_buffer));

        // The gather cannot write over its own input, so in place values go through scratch.
        const bool in_place_val = (batch_val_target == batch_val_source);
        void*      sorted_val   = in_place_val ? sort_buffer : batch_val_target;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                                  nnz,
                                                  data_type,
                                                  batch_val_source,
                                                  data_type,
                                                  sorted_val,
                                                  ptr_type,
                                                  perm,
                                                  rocsparse_index_base_zero));

        if(in_place_val)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_val_target,
                                                         sorted_val,
                                                         val_size * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
        }
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsort_buffer_size(rocsparse_handle            handle,
                                                rocsparse_csrsort_alg       alg,
                                                rocsparse_const_spmat_descr source,
                                                rocsparse_const_spmat_descr target,
                                                size_t*                     buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csxsort_buffer_size(
        handle, rocsparse_direction_row, source, target, buffer_size_in_bytes));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsort(rocsparse_handle            handle,
                                    rocsparse_csrsort_alg       alg,
                                    rocsparse_const_spmat_descr source,
                                    rocsparse_spmat_descr       target,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csxsort(
        handle, rocsparse_direction_row, source, target, buffer_size_in_bytes, buffer));
    return rocsparse_status_success;
}
