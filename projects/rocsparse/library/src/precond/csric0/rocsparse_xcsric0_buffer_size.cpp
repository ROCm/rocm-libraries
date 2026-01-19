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

#include "../level2/rocsparse_csrsv.hpp"
#include "rocsparse_csric0.hpp"
#include "rocsparse_utility.hpp"

#include "internal/precond/rocsparse_csric0.h"

namespace rocsparse
{
    template <typename T>
    static rocsparse_status xcsric0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        ROCSPARSE_CHECKARG_SIZE(1, m);

        ROCSPARSE_CHECKARG_SIZE(2, nnz);

        ROCSPARSE_CHECKARG_POINTER(3, descr);

        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);

        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);

        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);

        ROCSPARSE_CHECKARG_POINTER(7, info);

        ROCSPARSE_CHECKARG_POINTER(8, buffer_size);

        _rocsparse_spmat_descr csr(rocsparse_format_csr,
                                   false,
                                   static_cast<int64_t>(1),
                                   m,
                                   m,
                                   nnz,
                                   rocsparse::get_datatype<T>(),
                                   csr_val,
                                   nullptr,
                                   static_cast<int64_t>(0),
                                   rocsparse::get_indextype<rocsparse_int>(),
                                   csr_row_ptr,
                                   nullptr,
                                   static_cast<int64_t>(0),
                                   rocsparse::get_indextype<rocsparse_int>(),
                                   csr_col_ind,
                                   nullptr,
                                   static_cast<int64_t>(0),
                                   descr->base,
                                   descr,
                                   info);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_analysis_buffer_size(
            handle, rocsparse_operation_none, &csr, buffer_size)));

        size_t solve_buffer_size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_solve_buffer_size(
            handle, rocsparse_operation_none, &csr, &solve_buffer_size)));

        buffer_size[0] = std::max(buffer_size[0], solve_buffer_size);

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_scsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_int             m,
                                                          rocsparse_int             nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const float*              csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          const rocsparse_int*      csr_col_ind,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsric0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsric0_buffer_size(rocsparse_handle          handle,
                                                          rocsparse_int             m,
                                                          rocsparse_int             nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const double*             csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          const rocsparse_int*      csr_col_ind,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsric0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsric0_buffer_size(rocsparse_handle               handle,
                                                          rocsparse_int                  m,
                                                          rocsparse_int                  nnz,
                                                          const rocsparse_mat_descr      descr,
                                                          const rocsparse_float_complex* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsric0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsric0_buffer_size(rocsparse_handle                handle,
                                                          rocsparse_int                   m,
                                                          rocsparse_int                   nnz,
                                                          const rocsparse_mat_descr       descr,
                                                          const rocsparse_double_complex* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info   info,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsric0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
