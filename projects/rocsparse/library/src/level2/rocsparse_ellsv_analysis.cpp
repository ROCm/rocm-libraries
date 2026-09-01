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

namespace rocsparse
{
    // Whether a cached analysis can be reused for the matrix A, that is, whether
    // it was built from a matrix with the same sparsity pattern layout.
    static bool ellsv_trm_info_matches(const rocsparse::trm_info_t* trm_info,
                                       rocsparse_const_spmat_descr  A)
    {
        if(trm_info == nullptr || trm_info->get_row_map() == nullptr)
        {
            return false;
        }

        // The analysis only levels the sparsity pattern, so a cached result stays
        // valid whatever the values are.
        return trm_info->get_m() == A->rows && trm_info->get_index_indextype() == A->col_type
               && trm_info->get_descr() == A->descr;
    }

    template <uint32_t WF_SIZE, bool SLEEP, typename I>
    static rocsparse_status ellsv_launch_analysis(rocsparse_handle     handle,
                                                  I                    m,
                                                  I                    n,
                                                  I                    ell_width,
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
        static constexpr uint32_t BLOCKSIZE = 1024;

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
                                                    I                    ell_width,
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

    static rocsparse_status gellsv_analysis(rocsparse_handle          handle,
                                            int64_t                   m,
                                            int64_t                   n,
                                            const rocsparse_mat_descr descr,
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

        // Levelling only reads the sparsity pattern, so the values and their type
        // play no part here.
#define GELLSV_ANALYSIS_DISPATCH(ITYPE)                                      \
    rocsparse::ellsv_analysis_dispatch<ITYPE>(handle,                        \
                                              sleep,                         \
                                              wfsize,                        \
                                              static_cast<ITYPE>(m),         \
                                              static_cast<ITYPE>(n),         \
                                              static_cast<ITYPE>(ell_width), \
                                              ell_col_ind,                   \
                                              idx_base,                      \
                                              descr->fill_mode,              \
                                              descr->diag_type,              \
                                              ell_col_ind_indextype,         \
                                              info->get_row_map(),           \
                                              pivot_info->get_position(),    \
                                              buffer_size,                   \
                                              temp_buffer)

        switch(ell_col_ind_indextype)
        {
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(GELLSV_ANALYSIS_DISPATCH(int32_t));
            break;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(GELLSV_ANALYSIS_DISPATCH(int64_t));
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

rocsparse::trm_info_t* rocsparse::trm_data_t::create(rocsparse_handle          handle,
                                                     rocsparse_operation       trans,
                                                     int64_t                   m,
                                                     int64_t                   n,
                                                     const rocsparse_mat_descr descr,
                                                     rocsparse_indextype  ell_col_ind_indextype,
                                                     const void*          ell_col_ind,
                                                     int64_t              ell_width,
                                                     rocsparse_index_base idx_base,
                                                     size_t               buffer_size,
                                                     void*                temp_buffer)
{
    rocsparse::trm_info_t* trm_info = new rocsparse::trm_info_t();

    THROW_IF_ROCSPARSE_ERROR(rocsparse::gellsv_analysis(handle,
                                                        m,
                                                        n,
                                                        descr,
                                                        ell_col_ind_indextype,
                                                        ell_col_ind,
                                                        ell_width,
                                                        idx_base,
                                                        trm_info,
                                                        this,
                                                        buffer_size,
                                                        temp_buffer));
    return trm_info;
}

rocsparse_status rocsparse::trm_data_t::recreate(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int64_t                   m,
                                                 int64_t                   n,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_indextype       ell_col_ind_indextype,
                                                 const void*               ell_col_ind,
                                                 int64_t                   ell_width,
                                                 rocsparse_index_base      idx_base,
                                                 size_t                    buffer_size,
                                                 void*                     temp_buffer)
{
    return this->recreate(trans,
                          descr->fill_mode,
                          handle,
                          trans,
                          m,
                          n,
                          descr,
                          ell_col_ind_indextype,
                          ell_col_ind,
                          ell_width,
                          idx_base,
                          buffer_size,
                          temp_buffer);
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

    // Transposing an ELL matrix is not a width-preserving operation, so a
    // transposed solve is left unsupported rather than paying for a transpose
    // whose width is unbounded.
    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    if(A->rows == 0)
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

    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    if(A->rows == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(5, A->rows, temp_buffer);

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
                                           A->col_type,
                                           A->const_col_data,
                                           A->ell_width,
                                           A->idx_base,
                                           buffer_size,
                                           temp_buffer));

    return rocsparse_status_success;
}
