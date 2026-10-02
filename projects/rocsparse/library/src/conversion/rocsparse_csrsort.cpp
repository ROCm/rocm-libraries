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

    static bool csrsort_is_supported(rocsparse_indextype ptr_type, rocsparse_indextype ind_type)
    {
        return (ptr_type == rocsparse_indextype_i32 && ind_type == rocsparse_indextype_i32)
               || (ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i32)
               || (ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i64);
    }

    template <typename I, typename J>
    static rocsparse_status csrsort_rocprim_buffer_size_template(rocsparse_handle handle,
                                                                 int64_t          m,
                                                                 int64_t          nnz,
                                                                 uint32_t         startbit,
                                                                 uint32_t         endbit,
                                                                 bool             with_perm,
                                                                 size_t*          buffer_size)
    {
        if(with_perm)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, I, I>(
                    handle, nnz, m, startbit, endbit, buffer_size)));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
                    handle, nnz, m, startbit, endbit, buffer_size)));
        }
        return rocsparse_status_success;
    }

    static rocsparse_status csrsort_rocprim_buffer_size(rocsparse_handle    handle,
                                                        rocsparse_indextype ptr_type,
                                                        rocsparse_indextype ind_type,
                                                        int64_t             m,
                                                        int64_t             nnz,
                                                        uint32_t            startbit,
                                                        uint32_t            endbit,
                                                        bool                with_perm,
                                                        size_t*             buffer_size)
    {
        if(ptr_type == rocsparse_indextype_i32 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((csrsort_rocprim_buffer_size_template<int32_t, int32_t>(
                handle, m, nnz, startbit, endbit, with_perm, buffer_size)));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((csrsort_rocprim_buffer_size_template<int64_t, int32_t>(
                handle, m, nnz, startbit, endbit, with_perm, buffer_size)));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR((csrsort_rocprim_buffer_size_template<int64_t, int64_t>(
                handle, m, nnz, startbit, endbit, with_perm, buffer_size)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the combination of offsets and indices types is not supported");
    }

    // rocPRIM does not sort in place, so the sorted indices and permutation end up in either
    // the input arrays or the temporary arrays, and are copied back into the input arrays.
    template <typename I, typename J>
    static rocsparse_status csrsort_rocprim_sort_template(rocsparse_handle handle,
                                                          int64_t          m,
                                                          int64_t          nnz,
                                                          const void*      offsets,
                                                          void*            ind,
                                                          void*            tmp_ind,
                                                          void*            perm,
                                                          void*            tmp_perm,
                                                          uint32_t         startbit,
                                                          uint32_t         endbit,
                                                          size_t           buffer_size,
                                                          void*            buffer)
    {
        const I* offsets_ = reinterpret_cast<const I*>(offsets);
        J*       ind_     = reinterpret_cast<J*>(ind);
        I*       perm_    = reinterpret_cast<I*>(perm);

        rocsparse::primitives::double_buffer<J> keys(ind_, reinterpret_cast<J*>(tmp_ind));

        if(perm_ != nullptr)
        {
            rocsparse::primitives::double_buffer<I> vals(perm_, reinterpret_cast<I*>(tmp_perm));

            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::primitives::segmented_radix_sort_pairs(handle,
                                                                  keys,
                                                                  vals,
                                                                  nnz,
                                                                  m,
                                                                  offsets_,
                                                                  offsets_ + 1,
                                                                  startbit,
                                                                  endbit,
                                                                  buffer_size,
                                                                  buffer));

            if(vals.current() != perm_)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(perm_,
                                                             vals.current(),
                                                             sizeof(I) * nnz,
                                                             hipMemcpyDeviceToDevice,
                                                             handle->stream));
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(handle,
                                                                                       keys,
                                                                                       nnz,
                                                                                       m,
                                                                                       offsets_,
                                                                                       offsets_ + 1,
                                                                                       startbit,
                                                                                       endbit,
                                                                                       buffer_size,
                                                                                       buffer));
        }

        if(keys.current() != ind_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                ind_, keys.current(), sizeof(J) * nnz, hipMemcpyDeviceToDevice, handle->stream));
        }
        return rocsparse_status_success;
    }

    static rocsparse_status csrsort_rocprim_sort(rocsparse_handle    handle,
                                                 rocsparse_indextype ptr_type,
                                                 rocsparse_indextype ind_type,
                                                 int64_t             m,
                                                 int64_t             nnz,
                                                 const void*         offsets,
                                                 void*               ind,
                                                 void*               tmp_ind,
                                                 void*               perm,
                                                 void*               tmp_perm,
                                                 uint32_t            startbit,
                                                 uint32_t            endbit,
                                                 size_t              buffer_size,
                                                 void*               buffer)
    {
        if(ptr_type == rocsparse_indextype_i32 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((csrsort_rocprim_sort_template<int32_t, int32_t>(handle,
                                                                                       m,
                                                                                       nnz,
                                                                                       offsets,
                                                                                       ind,
                                                                                       tmp_ind,
                                                                                       perm,
                                                                                       tmp_perm,
                                                                                       startbit,
                                                                                       endbit,
                                                                                       buffer_size,
                                                                                       buffer)));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((csrsort_rocprim_sort_template<int64_t, int32_t>(handle,
                                                                                       m,
                                                                                       nnz,
                                                                                       offsets,
                                                                                       ind,
                                                                                       tmp_ind,
                                                                                       perm,
                                                                                       tmp_perm,
                                                                                       startbit,
                                                                                       endbit,
                                                                                       buffer_size,
                                                                                       buffer)));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64 && ind_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR((csrsort_rocprim_sort_template<int64_t, int64_t>(handle,
                                                                                       m,
                                                                                       nnz,
                                                                                       offsets,
                                                                                       ind,
                                                                                       tmp_ind,
                                                                                       perm,
                                                                                       tmp_perm,
                                                                                       startbit,
                                                                                       endbit,
                                                                                       buffer_size,
                                                                                       buffer)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the combination of offsets and indices types is not supported");
    }

    // Shifts one based offsets to zero based offsets, which the segmented sort expects.
    static rocsparse_status csrsort_shift_offsets(rocsparse_handle    handle,
                                                  rocsparse_indextype ptr_type,
                                                  int64_t             size,
                                                  const void*         in,
                                                  void*               out)
    {
#define CSRSORT_DIM 512
        dim3 csrsort_blocks((size - 1) / CSRSORT_DIM + 1);
        dim3 csrsort_threads(CSRSORT_DIM);

        if(ptr_type == rocsparse_indextype_i32)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrsort_shift_kernel<CSRSORT_DIM>),
                                               csrsort_blocks,
                                               csrsort_threads,
                                               0,
                                               handle->stream,
                                               static_cast<int32_t>(size),
                                               reinterpret_cast<const int32_t*>(in),
                                               reinterpret_cast<int32_t*>(out));
            return rocsparse_status_success;
        }
        if(ptr_type == rocsparse_indextype_i64)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrsort_shift_kernel<CSRSORT_DIM>),
                                               csrsort_blocks,
                                               csrsort_threads,
                                               0,
                                               handle->stream,
                                               size,
                                               reinterpret_cast<const int64_t*>(in),
                                               reinterpret_cast<int64_t*>(out));
            return rocsparse_status_success;
        }
#undef CSRSORT_DIM
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "the offsets type is not supported");
    }

    static rocsparse_status csrsort_buffer_size_compute(rocsparse_handle    handle,
                                                        int64_t             m,
                                                        int64_t             n,
                                                        int64_t             nnz,
                                                        rocsparse_indextype ptr_type,
                                                        rocsparse_indextype ind_type,
                                                        size_t*             buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(!rocsparse::csrsort_is_supported(ptr_type, ind_type))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the combination of offsets and indices types is not supported");
        }

        if(m == 0 || n == 0 || nnz == 0)
        {
            *buffer_size_in_bytes = 0;
            return rocsparse_status_success;
        }

        const uint32_t startbit = 0;
        const uint32_t endbit   = rocsparse::csrsort_endbit(n);

        // We do not know if sort_pairs or sort_keys will be called, so use the largest buffer between the two
        size_t size1 = std::numeric_limits<size_t>::max();
        size_t size2 = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_rocprim_buffer_size(
            handle, ptr_type, ind_type, m, nnz, startbit, endbit, true, &size1));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_rocprim_buffer_size(
            handle, ptr_type, ind_type, m, nnz, startbit, endbit, false, &size2));

        const size_t ptr_size = rocsparse::indextype_sizeof(ptr_type);
        const size_t ind_size = rocsparse::indextype_sizeof(ind_type);

        // rocPRIM does not support in-place sorting, so we need additional buffer
        // for all temporary arrays
        *buffer_size_in_bytes = rocsparse::align_size<char>(rocsparse::max(size1, size2));

        // columns buffer
        *buffer_size_in_bytes += rocsparse::align_size<char>(ind_size * nnz);
        // perm buffer
        *buffer_size_in_bytes += rocsparse::align_size<char>(ptr_size * nnz);
        // segm buffer
        *buffer_size_in_bytes += rocsparse::align_size<char>(ptr_size * (m + 1));

        return rocsparse_status_success;
    }

    // Sorts the column indices within each row. If perm is not null, the sort applies its
    // reordering to perm.
    static rocsparse_status csrsort_compute(rocsparse_handle     handle,
                                            int64_t              m,
                                            int64_t              n,
                                            int64_t              nnz,
                                            rocsparse_index_base idx_base,
                                            rocsparse_indextype  ptr_type,
                                            rocsparse_indextype  ind_type,
                                            const void*          csr_row_ptr,
                                            void*                csr_col_ind,
                                            void*                perm,
                                            size_t               buffer_size_in_bytes,
                                            void*                buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        size_t required_buffer_size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_buffer_size_compute(
            handle, m, n, nnz, ptr_type, ind_type, &required_buffer_size));
        if(buffer_size_in_bytes < required_buffer_size)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_size,
                "the buffer is smaller than the size required by the index sort");
        }

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }

        const uint32_t startbit = 0;
        const uint32_t endbit   = rocsparse::csrsort_endbit(n);

        size_t rocprim_size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_rocprim_buffer_size(
            handle, ptr_type, ind_type, m, nnz, startbit, endbit, perm != nullptr, &rocprim_size));

        const size_t ptr_size = rocsparse::indextype_sizeof(ptr_type);
        const size_t ind_size = rocsparse::indextype_sizeof(ind_type);

        // Temporary buffer entry points
        char* ptr = reinterpret_cast<char*>(buffer);

        // columns buffer
        void* tmp_cols = ptr;
        ptr += rocsparse::align_size<char>(ind_size * nnz);

        // perm buffer
        void* tmp_perm = ptr;
        ptr += rocsparse::align_size<char>(ptr_size * nnz);

        // segm buffer
        void* tmp_segm = ptr;
        ptr += rocsparse::align_size<char>(ptr_size * (m + 1));

        // rocprim buffer
        void* tmp_rocprim = ptr;

        // Index base one requires shift of offset positions
        const void* offsets = csr_row_ptr;
        if(idx_base == rocsparse_index_base_one)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrsort_shift_offsets(handle, ptr_type, m + 1, csr_row_ptr, tmp_segm));
            offsets = tmp_segm;
        }

        // Sort by columns and obtain permutation vector
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_rocprim_sort(handle,
                                                                  ptr_type,
                                                                  ind_type,
                                                                  m,
                                                                  nnz,
                                                                  offsets,
                                                                  csr_col_ind,
                                                                  tmp_cols,
                                                                  perm,
                                                                  tmp_perm,
                                                                  startbit,
                                                                  endbit,
                                                                  rocprim_size,
                                                                  tmp_rocprim));
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
        rocsparse::csrsort_buffer_size_compute(handle,
                                               m,
                                               n,
                                               nnz,
                                               rocsparse::get_indextype<rocsparse_int>(),
                                               rocsparse::get_indextype<rocsparse_int>(),
                                               buffer_size));

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

    // The legacy API does not take the size of temp_buffer, which is assumed to be the size
    // returned by rocsparse_csrsort_buffer_size.
    size_t buffer_size = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrsort_buffer_size_compute(handle,
                                               m,
                                               n,
                                               nnz,
                                               rocsparse::get_indextype<rocsparse_int>(),
                                               rocsparse::get_indextype<rocsparse_int>(),
                                               &buffer_size));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_compute(handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         descr->base,
                                                         rocsparse::get_indextype<rocsparse_int>(),
                                                         rocsparse::get_indextype<rocsparse_int>(),
                                                         csr_row_ptr,
                                                         csr_col_ind,
                                                         perm,
                                                         buffer_size,
                                                         temp_buffer));

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
    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the indices are sorted, followed by scratch space shared by the index sort
    // and the value permutation.
    static size_t csxsort_perm_size_in_bytes(int64_t nnz, rocsparse_indextype perm_indextype)
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

    size_t sort_buffer_size = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_buffer_size_compute(
        handle, m, n, nnz, ptr_type, ind_type, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (target->const_val_data == source->const_val_data)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(target->data_type) * nnz)
              : 0;

    *buffer_size_in_bytes = rocsparse::csxsort_perm_size_in_bytes(nnz, ptr_type)
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

    const char* ptr_source
        = reinterpret_cast<const char*>(by_row ? source->const_row_data : source->const_col_data);
    const char* ind_source
        = reinterpret_cast<const char*>(by_row ? source->const_col_data : source->const_row_data);
    const char* val_source = reinterpret_cast<const char*>(source->const_val_data);
    char*       ptr_target = reinterpret_cast<char*>(by_row ? target->row_data : target->col_data);
    char*       ind_target = reinterpret_cast<char*>(by_row ? target->col_data : target->row_data);
    char*       val_target = reinterpret_cast<char*>(target->val_data);

    const size_t ptr_size = rocsparse::indextype_sizeof(ptr_type);
    const size_t ind_size = rocsparse::indextype_sizeof(ind_type);
    const size_t val_size = rocsparse::datatype_sizeof(data_type);

    const size_t val_stride_source = source->columns_values_batch_stride * val_size;
    const size_t ptr_stride_target = target->offsets_batch_stride * ptr_size;
    const size_t ind_stride_target = target->columns_values_batch_stride * ind_size;
    const size_t val_stride_target = target->columns_values_batch_stride * val_size;

    const size_t perm_size        = rocsparse::csxsort_perm_size_in_bytes(nnz, ptr_type);
    const size_t sort_buffer_size = buffer_size_in_bytes - perm_size;
    void*        perm             = buffer;
    void*        sort_buffer      = reinterpret_cast<char*>(buffer) + perm_size;

    // Only uniform batches are supported, so every batch has the sparsity pattern of the first
    // one. Its indices are sorted once, and the resulting permutation is applied to the values
    // of every batch. A single source matrix has zero batch strides, so its values are sorted
    // into every batch of target.

    // The index sort works in place, so the offsets and indices of source are first copied into
    // target. A matrix without rows may have a null offsets array.
    if(m > 0 && ptr_target != ptr_source)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            ptr_target, ptr_source, ptr_size * (m + 1), hipMemcpyDeviceToDevice, handle->stream));
    }
    if(ind_target != ind_source)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            ind_target, ind_source, ind_size * nnz, hipMemcpyDeviceToDevice, handle->stream));
    }

    // The index sort applies its reordering to perm, so it must start as the identity.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcreate_identity_permutation(handle, nnz, ptr_type, perm));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_compute(handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         idx_base,
                                                         ptr_type,
                                                         ind_type,
                                                         ptr_target,
                                                         ind_target,
                                                         perm,
                                                         sort_buffer_size,
                                                         sort_buffer));

    // The batches run one after the other on the handle stream, so they share the buffer.
    for(int64_t batch = 0; batch < target->batch_count; ++batch)
    {
        if(batch > 0)
        {
            // The offsets of the target are shared by all its batches when their stride is zero.
            char* batch_ptr_target = ptr_target + batch * ptr_stride_target;
            if(m > 0 && batch_ptr_target != ptr_target)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_ptr_target,
                                                             ptr_target,
                                                             ptr_size * (m + 1),
                                                             hipMemcpyDeviceToDevice,
                                                             handle->stream));
            }
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(ind_target + batch * ind_stride_target,
                                                         ind_target,
                                                         ind_size * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
        }

        const char* batch_val_source = val_source + batch * val_stride_source;
        char*       batch_val_target = val_target + batch * val_stride_target;

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
