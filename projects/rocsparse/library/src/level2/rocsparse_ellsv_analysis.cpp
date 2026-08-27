/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_gcreate_identity_permutation.hpp"
#include "ellsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_ellsv.hpp"
#include "rocsparse_ellsv_info.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_trm_data_t.hpp"
#include "rocsparse_trm_info.hpp"
#include "rocsparse_utility.hpp"

#include <vector>

namespace rocsparse
{
    static rocsparse_fill_mode ellsv_flip_fill(rocsparse_fill_mode fill_mode)
    {
        return (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                        : rocsparse_fill_mode_lower;
    }

    static void ellsv_clear_trm_slots(rocsparse_ellsv_info ei)
    {
        const rocsparse_operation operations[]
            = {rocsparse_operation_none, rocsparse_operation_transpose};
        const rocsparse_fill_mode fill_modes[]
            = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};

        for(const rocsparse_operation op : operations)
        {
            for(const rocsparse_fill_mode fm : fill_modes)
            {
                rocsparse::trm_info_t* trm_info = ei->get(op, fm);
                if(trm_info != nullptr)
                {
                    delete trm_info;
                    ei->set(op, fm, nullptr);
                }
            }
        }
    }

    static bool ellsv_trm_info_matches(const rocsparse::trm_info_t* trm_info,
                                       rocsparse_const_spmat_descr  A)
    {
        if(trm_info == nullptr || trm_info->get_row_map() == nullptr)
        {
            return false;
        }

        if(trm_info->get_m() != A->rows || trm_info->get_index_indextype() != A->col_type
           || trm_info->get_descr() != A->descr)
        {
            return false;
        }

        // A cached transpose was materialized from the values of A, so it is only
        // reusable while the value type still agrees.
        rocsparse_const_spmat_descr transposed = trm_info->get_transposed_matrix();

        return transposed == nullptr || transposed->data_type == A->data_type;
    }

    static bool ellsv_info_matrix_matches(rocsparse_ellsv_info ei, rocsparse_const_spmat_descr A)
    {
        if(ei == nullptr)
        {
            return true;
        }

        const rocsparse_operation operations[]
            = {rocsparse_operation_none, rocsparse_operation_transpose};
        const rocsparse_fill_mode fill_modes[]
            = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};

        for(const rocsparse_operation op : operations)
        {
            for(const rocsparse_fill_mode fm : fill_modes)
            {
                const rocsparse::trm_info_t* trm_info = ei->get(op, fm);
                if(trm_info != nullptr && !rocsparse::ellsv_trm_info_matches(trm_info, A))
                {
                    return false;
                }
            }
        }

        return true;
    }

    // Hands the trm info a freshly allocated, still uninitialized ELL matrix to
    // hold the transpose of the analyzed matrix. Triangular solve only accepts
    // square matrices, so the transpose has the same dimensions as the original,
    // and it is described with the opposite fill mode.
    static rocsparse_status ellsv_create_transposed(rocsparse::trm_info_t*    trm_info,
                                                    int64_t                   m,
                                                    int64_t                   width,
                                                    rocsparse_indextype       index_type,
                                                    rocsparse_datatype        value_type,
                                                    rocsparse_index_base      base,
                                                    const rocsparse_mat_descr descr,
                                                    hipStream_t               stream)
    {
        trm_info->clear_transposed_matrix();

        const int64_t count = rocsparse::max(m * width, static_cast<int64_t>(1));
        const size_t  col_bytes
            = rocsparse::indextype_sizeof(index_type) * static_cast<size_t>(count);
        const size_t val_bytes
            = rocsparse::datatype_sizeof(value_type) * static_cast<size_t>(count);

        void* col_ind = nullptr;
        void* val     = nullptr;
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&col_ind, col_bytes, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&val, val_bytes, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        rocsparse_spmat_descr  transposed = nullptr;
        const rocsparse_status status     = rocsparse_create_ell_descr(
            &transposed, m, m, col_ind, val, width, index_type, base, value_type);

        if(status != rocsparse_status_success)
        {
            WARNING_IF_HIP_ERROR(rocsparse_hipFreeAsync(col_ind, stream));
            WARNING_IF_HIP_ERROR(rocsparse_hipFreeAsync(val, stream));
            RETURN_IF_ROCSPARSE_ERROR(status);
        }

        transposed->descr->fill_mode = rocsparse::ellsv_flip_fill(descr->fill_mode);
        transposed->descr->diag_type = descr->diag_type;

        trm_info->set_transposed_matrix(transposed);

        return rocsparse_status_success;
    }

    template <uint32_t WF_SIZE, bool SLEEP, typename I>
    static rocsparse_status ellsv_launch_analysis(rocsparse_handle     handle,
                                                  I                    m,
                                                  I                    n,
                                                  int64_t              ell_width,
                                                  const I*             ell_col_ind,
                                                  rocsparse_index_base base,
                                                  rocsparse_fill_mode  fill_mode,
                                                  rocsparse_diag_type  diag_type,
                                                  rocsparse_indextype  index_type,
                                                  I*                   row_map,
                                                  I*                   zero_pivot,
                                                  size_t               buffer_size,
                                                  void*                temp_buffer)
    {
        constexpr uint32_t BLOCKSIZE = 1024;

        hipStream_t  stream   = handle->stream;
        const size_t sizeof_I = sizeof(I);

        const uint32_t startbit = 0;
        const uint32_t endbit   = rocsparse::clz(static_cast<int64_t>(m));

        size_t rocprim_size = 0;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, I>(
            handle, m, startbit, endbit, &rocprim_size)));

        const size_t done_bytes       = ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));
        const size_t workspace_bytes  = ellsv_align256(sizeof_I * static_cast<size_t>(m));
        const size_t workspace2_bytes = ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));

        // Must stay in sync with rocsparse::ellsv_analysis_buffer_size.
        if(buffer_size < done_bytes + workspace_bytes + workspace2_bytes + rocprim_size)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
        }

        char*    ptr        = reinterpret_cast<char*>(temp_buffer);
        int32_t* done_array = reinterpret_cast<int32_t*>(ptr);
        ptr += done_bytes;

        void* workspace = ptr;
        ptr += workspace_bytes;

        void* workspace2 = ptr;
        ptr += workspace2_bytes;

        void* rocprim_buffer = ptr;

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(done_array, 0, done_bytes, stream));

        dim3 blocks((static_cast<size_t>(m) * WF_SIZE - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::ellsv_analysis_lower_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I>),
                blocks,
                threads,
                0,
                stream,
                m,
                n,
                ell_width,
                ell_col_ind,
                done_array,
                zero_pivot,
                base,
                diag_type);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::ellsv_analysis_upper_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I>),
                blocks,
                threads,
                0,
                stream,
                m,
                n,
                ell_width,
                ell_col_ind,
                done_array,
                zero_pivot,
                base,
                diag_type);
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, m, index_type, workspace));

        rocsparse::primitives::double_buffer<int32_t> keys(done_array,
                                                           reinterpret_cast<int32_t*>(workspace2));
        rocsparse::primitives::double_buffer<I> vals(reinterpret_cast<I*>(workspace), row_map);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, m, startbit, endbit, rocprim_size, rocprim_buffer));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        if(vals.current() != row_map)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(row_map,
                                                         vals.current(),
                                                         sizeof_I * static_cast<size_t>(m),
                                                         hipMemcpyDeviceToDevice,
                                                         stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
        }

        return rocsparse_status_success;
    }

    template <typename I>
    static rocsparse_status ellsv_analysis_dispatch(rocsparse_handle     handle,
                                                    bool                 sleep,
                                                    uint32_t             wfsize,
                                                    I                    m,
                                                    I                    n,
                                                    int64_t              ell_width,
                                                    const void*          ell_col_ind,
                                                    rocsparse_index_base base,
                                                    rocsparse_fill_mode  fill_mode,
                                                    rocsparse_diag_type  diag_type,
                                                    rocsparse_indextype  index_type,
                                                    void*                row_map,
                                                    void*                zero_pivot,
                                                    size_t               buffer_size,
                                                    void*                temp_buffer)
    {
        const I* col_ind = reinterpret_cast<const I*>(ell_col_ind);
        I*       map     = reinterpret_cast<I*>(row_map);
        I*       pivot   = reinterpret_cast<I*>(zero_pivot);

        if(sleep)
        {
            return rocsparse::ellsv_launch_analysis<64, true, I>(handle,
                                                                 m,
                                                                 n,
                                                                 ell_width,
                                                                 col_ind,
                                                                 base,
                                                                 fill_mode,
                                                                 diag_type,
                                                                 index_type,
                                                                 map,
                                                                 pivot,
                                                                 buffer_size,
                                                                 temp_buffer);
        }
        else if(wfsize == 64)
        {
            return rocsparse::ellsv_launch_analysis<64, false, I>(handle,
                                                                  m,
                                                                  n,
                                                                  ell_width,
                                                                  col_ind,
                                                                  base,
                                                                  fill_mode,
                                                                  diag_type,
                                                                  index_type,
                                                                  map,
                                                                  pivot,
                                                                  buffer_size,
                                                                  temp_buffer);
        }

        return rocsparse::ellsv_launch_analysis<32, false, I>(handle,
                                                              m,
                                                              n,
                                                              ell_width,
                                                              col_ind,
                                                              base,
                                                              fill_mode,
                                                              diag_type,
                                                              index_type,
                                                              map,
                                                              pivot,
                                                              buffer_size,
                                                              temp_buffer);
    }

    template <typename I, typename T>
    static rocsparse_status ellsv_build_transpose(rocsparse_handle          handle,
                                                  I                         m,
                                                  I                         n,
                                                  int64_t                   ell_width,
                                                  const void*               ell_col_ind,
                                                  const void*               ell_val,
                                                  rocsparse_index_base      base,
                                                  bool                      conj,
                                                  const rocsparse_mat_descr descr,
                                                  rocsparse::trm_info_t*    trm_info)
    {
        constexpr uint32_t BLOCKSIZE = 256;

        hipStream_t stream = handle->stream;

        unsigned long long* counts = nullptr;
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &counts, sizeof(unsigned long long) * static_cast<size_t>(m), stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            counts, 0, sizeof(unsigned long long) * static_cast<size_t>(m), stream));

        dim3 blocks((static_cast<size_t>(m) - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellsv_transpose_count_kernel<BLOCKSIZE, I>),
                                           blocks,
                                           threads,
                                           0,
                                           stream,
                                           m,
                                           n,
                                           ell_width,
                                           reinterpret_cast<const I*>(ell_col_ind),
                                           base,
                                           counts);

        std::vector<unsigned long long> h_counts(static_cast<size_t>(m));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(h_counts.data(),
                                     counts,
                                     sizeof(unsigned long long) * static_cast<size_t>(m),
                                     hipMemcpyDeviceToHost,
                                     stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        int64_t t_width = 0;
        for(int64_t i = 0; i < static_cast<int64_t>(m); ++i)
        {
            t_width = rocsparse::max(t_width, static_cast<int64_t>(h_counts[i]));
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_create_transposed(trm_info,
                                                                     static_cast<int64_t>(m),
                                                                     t_width,
                                                                     rocsparse::get_indextype<I>(),
                                                                     rocsparse::get_datatype<T>(),
                                                                     base,
                                                                     descr,
                                                                     stream));

        rocsparse_spmat_descr transposed = trm_info->get_transposed_matrix();

        const int64_t total = static_cast<int64_t>(m) * t_width;
        if(total > 0)
        {
            dim3 fill_blocks((total - 1) / BLOCKSIZE + 1);
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ellsv_fill_col_ind_kernel<BLOCKSIZE, I>),
                                               fill_blocks,
                                               threads,
                                               0,
                                               stream,
                                               total,
                                               reinterpret_cast<I*>(transposed->col_data),
                                               static_cast<I>(m + base));
        }

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            counts, 0, sizeof(unsigned long long) * static_cast<size_t>(m), stream));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::ellsv_transpose_scatter_kernel<BLOCKSIZE, I, T>),
            blocks,
            threads,
            0,
            stream,
            m,
            n,
            ell_width,
            reinterpret_cast<const I*>(ell_col_ind),
            reinterpret_cast<const T*>(ell_val),
            base,
            conj,
            counts,
            reinterpret_cast<I*>(transposed->col_data),
            reinterpret_cast<T*>(transposed->val_data));

        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(counts, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    static rocsparse_status gellsv_analysis_typed(rocsparse_handle          handle,
                                                  rocsparse_operation       trans,
                                                  I                         m,
                                                  I                         n,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  ell_val,
                                                  const I*                  ell_col_ind,
                                                  int64_t                   ell_width,
                                                  rocsparse_index_base      idx_base,
                                                  bool                      sleep,
                                                  uint32_t                  wfsize,
                                                  rocsparse::trm_info_t*    trm_info,
                                                  void*                     zero_pivot,
                                                  size_t                    buffer_size,
                                                  void*                     temp_buffer)
    {
        rocsparse_fill_mode fill = descr->fill_mode;

        const void* col_ind  = ell_col_ind;
        I           n_solver = n;
        int64_t     width    = ell_width;

        if(trans != rocsparse_operation_none)
        {
            const bool conjugate = (trans == rocsparse_operation_conjugate_transpose);

            RETURN_IF_ROCSPARSE_ERROR((rocsparse::ellsv_build_transpose<I, T>(handle,
                                                                              m,
                                                                              n,
                                                                              ell_width,
                                                                              ell_col_ind,
                                                                              ell_val,
                                                                              idx_base,
                                                                              conjugate,
                                                                              descr,
                                                                              trm_info)));

            // The transpose describes itself, including the flipped fill mode.
            rocsparse_const_spmat_descr transposed = trm_info->get_transposed_matrix();

            col_ind  = transposed->const_col_data;
            n_solver = static_cast<I>(transposed->cols);
            width    = transposed->ell_width;
            fill     = transposed->descr->fill_mode;
        }

        return rocsparse::ellsv_analysis_dispatch<I>(handle,
                                                     sleep,
                                                     wfsize,
                                                     m,
                                                     n_solver,
                                                     width,
                                                     col_ind,
                                                     idx_base,
                                                     fill,
                                                     descr->diag_type,
                                                     rocsparse::get_indextype<I>(),
                                                     trm_info->get_row_map(),
                                                     zero_pivot,
                                                     buffer_size,
                                                     temp_buffer);
    }

    rocsparse_status gellsv_analysis(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int64_t                   m,
                                     int64_t                   n,
                                     const rocsparse_mat_descr descr,
                                     rocsparse_datatype        ell_val_datatype,
                                     const void*               ell_val,
                                     rocsparse_indextype       ell_col_ind_indextype,
                                     const void*               ell_col_ind,
                                     int64_t                   ell_width,
                                     rocsparse_index_base      idx_base,
                                     rocsparse::trm_info_t*    info,
                                     rocsparse::pivot_info_t*  pivot_info,
                                     size_t                    buffer_size,
                                     void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;

        info->set_m(m);
        info->set_descr(descr);
        info->set_offset_indextype(ell_col_ind_indextype);
        info->set_index_indextype(ell_col_ind_indextype);

        const size_t num_bytes
            = rocsparse::indextype_sizeof(ell_col_ind_indextype) * static_cast<size_t>(m);
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(info->get_ref_row_map(), num_bytes, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        bool     sleep  = false;
        uint32_t wfsize = 0;
        rocsparse::ellsv_select_launch(handle, &sleep, &wfsize);

        pivot_info->create_zero_pivot_async(ell_col_ind_indextype, stream);
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(pivot_info->get_batch_count(),
                                                              ell_col_ind_indextype,
                                                              pivot_info->get_position(),
                                                              stream));

#define GELLSV_ANALYSIS_DISPATCH(ITYPE, TTYPE)                                       \
    gellsv_analysis_typed<ITYPE, TTYPE>(handle,                                      \
                                        trans,                                       \
                                        static_cast<ITYPE>(m),                       \
                                        static_cast<ITYPE>(n),                       \
                                        descr,                                       \
                                        reinterpret_cast<const TTYPE*>(ell_val),     \
                                        reinterpret_cast<const ITYPE*>(ell_col_ind), \
                                        ell_width,                                   \
                                        idx_base,                                    \
                                        sleep,                                       \
                                        wfsize,                                      \
                                        info,                                        \
                                        pivot_info->get_position(),                  \
                                        buffer_size,                                 \
                                        temp_buffer)

        switch(ell_col_ind_indextype)
        {
        case rocsparse_indextype_i32:
        {
            switch(ell_val_datatype)
            {
            case rocsparse_datatype_f32_r:
                RETURN_IF_ROCSPARSE_ERROR(GELLSV_ANALYSIS_DISPATCH(int32_t, float));
                break;
            case rocsparse_datatype_f64_r:
                RETURN_IF_ROCSPARSE_ERROR(GELLSV_ANALYSIS_DISPATCH(int32_t, double));
                break;
            case rocsparse_datatype_f32_c:
                RETURN_IF_ROCSPARSE_ERROR(
                    GELLSV_ANALYSIS_DISPATCH(int32_t, rocsparse_float_complex));
                break;
            case rocsparse_datatype_f64_c:
                RETURN_IF_ROCSPARSE_ERROR(
                    GELLSV_ANALYSIS_DISPATCH(int32_t, rocsparse_double_complex));
                break;
            default:
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                // LCOV_EXCL_STOP
            }
            break;
        }
        case rocsparse_indextype_i64:
        {
            switch(ell_val_datatype)
            {
            case rocsparse_datatype_f32_r:
                RETURN_IF_ROCSPARSE_ERROR(GELLSV_ANALYSIS_DISPATCH(int64_t, float));
                break;
            case rocsparse_datatype_f64_r:
                RETURN_IF_ROCSPARSE_ERROR(GELLSV_ANALYSIS_DISPATCH(int64_t, double));
                break;
            case rocsparse_datatype_f32_c:
                RETURN_IF_ROCSPARSE_ERROR(
                    GELLSV_ANALYSIS_DISPATCH(int64_t, rocsparse_float_complex));
                break;
            case rocsparse_datatype_f64_c:
                RETURN_IF_ROCSPARSE_ERROR(
                    GELLSV_ANALYSIS_DISPATCH(int64_t, rocsparse_double_complex));
                break;
            default:
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                // LCOV_EXCL_STOP
            }
            break;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        }

#undef GELLSV_ANALYSIS_DISPATCH

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::ellsv_analysis_buffer_size(rocsparse_handle            handle,
                                                       rocsparse_operation         trans,
                                                       rocsparse_const_spmat_descr A,
                                                       size_t* buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, buffer_size_in_bytes);

    if(A->rows == 0 || A->batch_count == 0)
    {
        *buffer_size_in_bytes = 0;
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    const int64_t m        = A->rows;
    const size_t  sizeof_I = rocsparse::indextype_sizeof(A->col_type);

    size_t size = rocsparse::ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));
    size += rocsparse::ellsv_align256(sizeof_I * static_cast<size_t>(m));
    size += rocsparse::ellsv_align256(sizeof(int32_t) * static_cast<size_t>(m));

    const uint32_t startbit = 0;
    const uint32_t endbit   = rocsparse::clz(m);

    size_t rocprim_size = 0;
    auto   calculate_rocprim_size
        = rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32, A->col_type);
    RETURN_IF_ROCSPARSE_ERROR(
        (calculate_rocprim_size(handle, m, startbit, endbit, &rocprim_size, true)));

    size += rocprim_size;

    *buffer_size_in_bytes = size;

    return rocsparse_status_success;
}

rocsparse_status rocsparse::ellsv_analysis(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_analysis_policy   analysis_policy,
                                           rocsparse_ellsv_info*       p_ellsv_info,
                                           size_t                      buffer_size,
                                           void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_ENUM(3, analysis_policy);
    ROCSPARSE_CHECKARG_POINTER(4, p_ellsv_info);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(5, (A->rows > 0 && A->batch_count > 0), temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellsv_check(A));

    rocsparse_mat_descr descr = A->descr;
    rocsparse_mat_info  info  = A->info;

    if(analysis_policy == rocsparse_analysis_policy_reuse)
    {
        rocsparse::trm_info_t* trm_info = info->get_ellsv_info(trans, descr->fill_mode);

        if(trm_info != nullptr && rocsparse::ellsv_trm_info_matches(trm_info, A))
        {
            return rocsparse_status_success;
        }
    }

    rocsparse_ellsv_info ei = p_ellsv_info[0];
    if(ei != nullptr && !rocsparse::ellsv_info_matrix_matches(ei, A))
    {
        rocsparse::ellsv_clear_trm_slots(ei);
    }

    if(ei == nullptr)
    {
        ei              = new _rocsparse_ellsv_info();
        p_ellsv_info[0] = ei;
    }

    RETURN_IF_ROCSPARSE_ERROR(ei->recreate(handle,
                                           trans,
                                           A->rows,
                                           A->cols,
                                           descr,
                                           A->data_type,
                                           A->const_val_data,
                                           A->col_type,
                                           A->const_col_data,
                                           A->ell_width,
                                           A->idx_base,
                                           buffer_size,
                                           temp_buffer));

    return rocsparse_status_success;
}
