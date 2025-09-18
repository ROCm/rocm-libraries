/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "internal/conversion/rocsparse_dense2coo.h"
#include "rocsparse_csr2coo.hpp"
#include "rocsparse_dense2coo.hpp"

#include "rocsparse_dense2csx_impl.hpp"

namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status dense2coo_checkarg(rocsparse_handle          handle, //0
                                        I                         m, //1
                                        I                         n, //2
                                        const rocsparse_mat_descr descr, //3
                                        const T*                  A, //4
                                        int64_t                   ld, //5
                                        const I*                  nnz_per_rows, //6
                                        T*                        coo_val, //7
                                        I*                        coo_row_ind, //8
                                        I*                        coo_col_ind) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(4,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(5, ld, (ld < m), rocsparse_status_invalid_size);

        // Quick return if possible
        if(m == 0 || n == 0)
        {
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(4, A);
        ROCSPARSE_CHECKARG_ARRAY(6, m, nnz_per_rows);
        ROCSPARSE_CHECKARG_POINTER(7, coo_val);
        ROCSPARSE_CHECKARG_POINTER(8, coo_row_ind);
        ROCSPARSE_CHECKARG_POINTER(9, coo_col_ind);
        return rocsparse_status_continue;
    }
}

template <typename I, typename T>
rocsparse_status rocsparse::dense2coo_template(rocsparse_handle          handle, //0
                                               rocsparse_order           order, //1
                                               I                         m, //2
                                               I                         n, //3
                                               const rocsparse_mat_descr descr, //4
                                               const T*                  A, //5
                                               int64_t                   ld, //6
                                               const I*                  nnz_per_rows, //7
                                               T*                        coo_val, //8
                                               I*                        coo_row_ind, //9
                                               I*                        coo_col_ind) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xdense2coo"),
                         order,
                         m,
                         n,
                         descr,
                         (const void*&)A,
                         ld,
                         (const void*&)nnz_per_rows,
                         (const void*&)coo_val,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind);

    I* row_ptr;
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&row_ptr, sizeof(I) * (m + 1), handle->stream));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2csx_impl<rocsparse_direction_row>(
        handle, order, m, n, descr, A, ld, nnz_per_rows, coo_val, row_ptr, coo_col_ind));

    I start;
    I end;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &row_ptr[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    const I nnz = end - start;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csr2coo_template(handle, row_ptr, nnz, m, coo_row_ind, descr->base));

    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(row_ptr, handle->stream));

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                          \
    template rocsparse_status rocsparse::dense2coo_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                  \
        rocsparse_order           order,                                   \
        ITYPE                     m,                                       \
        ITYPE                     n,                                       \
        const rocsparse_mat_descr descr,                                   \
        const TTYPE*              A,                                       \
        int64_t                   ld,                                      \
        const ITYPE*              nnz_per_rows,                            \
        TTYPE*                    coo_val,                                 \
        ITYPE*                    coo_row_ind,                             \
        ITYPE*                    coo_col_ind);

INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sdense2coo(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              A,
                                                 rocsparse_int             ld,
                                                 const rocsparse_int*      nnz_per_rows,
                                                 float*                    coo_val,
                                                 rocsparse_int*            coo_row_ind,
                                                 rocsparse_int*            coo_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::dense2coo_checkarg(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2coo_template(handle,
                                                            rocsparse_order_column,
                                                            m,
                                                            n,
                                                            descr,
                                                            A,
                                                            ld,
                                                            nnz_per_rows,
                                                            coo_val,
                                                            coo_row_ind,
                                                            coo_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ddense2coo(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             A,
                                                 rocsparse_int             ld,
                                                 const rocsparse_int*      nnz_per_rows,
                                                 double*                   coo_val,
                                                 rocsparse_int*            coo_row_ind,
                                                 rocsparse_int*            coo_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::dense2coo_checkarg(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2coo_template(handle,
                                                            rocsparse_order_column,
                                                            m,
                                                            n,
                                                            descr,
                                                            A,
                                                            ld,
                                                            nnz_per_rows,
                                                            coo_val,
                                                            coo_row_ind,
                                                            coo_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cdense2coo(rocsparse_handle               handle,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  n,
                                                 const rocsparse_mat_descr      descr,
                                                 const rocsparse_float_complex* A,
                                                 rocsparse_int                  ld,
                                                 const rocsparse_int*           nnz_per_rows,
                                                 rocsparse_float_complex*       coo_val,
                                                 rocsparse_int*                 coo_row_ind,
                                                 rocsparse_int*                 coo_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::dense2coo_checkarg(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2coo_template(handle,
                                                            rocsparse_order_column,
                                                            m,
                                                            n,
                                                            descr,
                                                            A,
                                                            ld,
                                                            nnz_per_rows,
                                                            coo_val,
                                                            coo_row_ind,
                                                            coo_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zdense2coo(rocsparse_handle                handle,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   n,
                                                 const rocsparse_mat_descr       descr,
                                                 const rocsparse_double_complex* A,
                                                 rocsparse_int                   ld,
                                                 const rocsparse_int*            nnz_per_rows,
                                                 rocsparse_double_complex*       coo_val,
                                                 rocsparse_int*                  coo_row_ind,
                                                 rocsparse_int*                  coo_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::dense2coo_checkarg(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2coo_template(handle,
                                                            rocsparse_order_column,
                                                            m,
                                                            n,
                                                            descr,
                                                            A,
                                                            ld,
                                                            nnz_per_rows,
                                                            coo_val,
                                                            coo_row_ind,
                                                            coo_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
