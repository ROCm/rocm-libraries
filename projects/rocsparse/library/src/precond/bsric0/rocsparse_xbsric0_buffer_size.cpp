/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/precond/rocsparse_bsric0.h"
#include "rocsparse_bsric0.hpp"
#include "rocsparse_utility.hpp"
namespace rocsparse
{
    template <typename T>
    static rocsparse_status xbsric0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        ROCSPARSE_CHECKARG_ENUM(1, dir);

        ROCSPARSE_CHECKARG_SIZE(2, mb);

        ROCSPARSE_CHECKARG_SIZE(3, nnzb);

        ROCSPARSE_CHECKARG_POINTER(4, descr);

        ROCSPARSE_CHECKARG(4,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(4,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(5, nnzb, bsr_val);

        ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);

        ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);

        ROCSPARSE_CHECKARG_SIZE(8, block_dim);

        ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG_POINTER(9, info);

        ROCSPARSE_CHECKARG_POINTER(10, buffer_size);

        _rocsparse_spmat_descr bsr(rocsparse_format_bsr,
                                   false,
                                   static_cast<int64_t>(1),
                                   mb,
                                   mb,
                                   nnzb,
                                   dir,
                                   block_dim,
                                   rocsparse::get_datatype<T>(),
                                   bsr_val,
                                   nullptr,
                                   static_cast<int64_t>(0),
                                   rocsparse::get_indextype<rocsparse_int>(),
                                   bsr_row_ptr,
                                   nullptr,
                                   static_cast<int64_t>(0),
                                   rocsparse::get_indextype<rocsparse_int>(),
                                   bsr_col_ind,
                                   nullptr,
                                   static_cast<int64_t>(0),
                                   descr->base,
                                   descr,
                                   info);

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::bsric0_analysis_buffer_size(handle, &bsr, buffer_size)));

        size_t solve_buffer_size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::bsric0_solve_buffer_size(handle, &bsr, &solve_buffer_size)));
        buffer_size[0] = std::max(buffer_size[0], solve_buffer_size);
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_sbsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nnzb,
                                                          const rocsparse_mat_descr descr,
                                                          const float*              bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             block_dim,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsric0_buffer_size(handle,
                                                             dir,
                                                             mb,
                                                             nnzb,
                                                             descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             block_dim,
                                                             info,
                                                             buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dbsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nnzb,
                                                          const rocsparse_mat_descr descr,
                                                          const double*             bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             block_dim,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsric0_buffer_size(handle,
                                                             dir,
                                                             mb,
                                                             nnzb,
                                                             descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             block_dim,
                                                             info,
                                                             buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cbsric0_buffer_size(rocsparse_handle               handle,
                                                          rocsparse_direction            dir,
                                                          rocsparse_int                  mb,
                                                          rocsparse_int                  nnzb,
                                                          const rocsparse_mat_descr      descr,
                                                          const rocsparse_float_complex* bsr_val,
                                                          const rocsparse_int* bsr_row_ptr,
                                                          const rocsparse_int* bsr_col_ind,
                                                          rocsparse_int        block_dim,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsric0_buffer_size(handle,
                                                             dir,
                                                             mb,
                                                             nnzb,
                                                             descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             block_dim,
                                                             info,
                                                             buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zbsric0_buffer_size(rocsparse_handle                handle,
                                                          rocsparse_direction             dir,
                                                          rocsparse_int                   mb,
                                                          rocsparse_int                   nnzb,
                                                          const rocsparse_mat_descr       descr,
                                                          const rocsparse_double_complex* bsr_val,
                                                          const rocsparse_int* bsr_row_ptr,
                                                          const rocsparse_int* bsr_col_ind,
                                                          rocsparse_int        block_dim,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsric0_buffer_size(handle,
                                                             dir,
                                                             mb,
                                                             nnzb,
                                                             descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             block_dim,
                                                             info,
                                                             buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
