/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_csr2ell.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2ell.hpp"
#include "rocsparse_utility.hpp"

#include "csr2ell_device.h"

rocsparse_status rocsparse::csr2ell_quickreturn(rocsparse_handle          handle,
                                                int64_t                   m,
                                                const rocsparse_mat_descr csr_descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                const rocsparse_mat_descr ell_descr,
                                                int64_t                   ell_width,
                                                void*                     ell_val,
                                                void*                     ell_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || ell_width == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    rocsparse_status csr2ell_checkarg(rocsparse_handle          handle, //0
                                      int64_t                   m, //1
                                      const rocsparse_mat_descr csr_descr, //2
                                      const void*               csr_val, //3
                                      const void*               csr_row_ptr, //4
                                      const void*               csr_col_ind, //5
                                      const rocsparse_mat_descr ell_descr, //6
                                      int64_t                   ell_width, //7
                                      void*                     ell_val, //8
                                      void*                     ell_col_ind) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(7, ell_width);

        const rocsparse_status status = rocsparse::csr2ell_quickreturn(handle,
                                                                       m,
                                                                       csr_descr,
                                                                       csr_val,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       ell_descr,
                                                                       ell_width,
                                                                       ell_val,
                                                                       ell_col_ind);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(2, csr_descr);
        ROCSPARSE_CHECKARG(2,
                           csr_descr,
                           (csr_descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(2,
                           csr_descr,
                           (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_POINTER(6, ell_descr);
        ROCSPARSE_CHECKARG(6,
                           ell_descr,
                           (ell_descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(6,
                           ell_descr,
                           (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(7, ell_width);
        ROCSPARSE_CHECKARG_ARRAY(3, ell_width, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(5, ell_width, csr_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(8, ell_width, ell_val);
        ROCSPARSE_CHECKARG_ARRAY(9, ell_width, ell_col_ind);

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csr2ell_core(rocsparse_handle          handle,
                                         J                         m,
                                         const rocsparse_mat_descr csr_descr,
                                         const T*                  csr_val,
                                         const I*                  csr_row_ptr,
                                         const J*                  csr_col_ind,
                                         const rocsparse_mat_descr ell_descr,
                                         J                         ell_width,
                                         T*                        ell_val,
                                         J*                        ell_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

#define CSR2ELL_DIM 512
    dim3 csr2ell_blocks((m - 1) / CSR2ELL_DIM + 1);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2ell_kernel<CSR2ELL_DIM>),
                                       csr2ell_blocks,
                                       csr2ell_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_descr->base,
                                       ell_width,
                                       ell_col_ind,
                                       ell_val,
                                       ell_descr->base);
#undef CSR2ELL_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, J)                                    \
    template rocsparse_status rocsparse::csr2ell_core<T, I, J>( \
        rocsparse_handle          handle,                       \
        J                         m,                            \
        const rocsparse_mat_descr csr_descr,                    \
        const T*                  csr_val,                      \
        const I*                  csr_row_ptr,                  \
        const J*                  csr_col_ind,                  \
        const rocsparse_mat_descr ell_descr,                    \
        J                         ell_width,                    \
        T*                        ell_val,                      \
        J*                        ell_col_ind)

INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t);
INSTANTIATE(int32_t, int32_t, int64_t);
INSTANTIATE(int32_t, int64_t, int64_t);

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int32_t, int64_t);
INSTANTIATE(float, int64_t, int64_t);

INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int32_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);

INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);

INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);

namespace rocsparse
{
    template <typename... P>
    rocsparse_status csr2ell_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_Xcsr2ell", p...);
        const rocsparse_status status = rocsparse::csr2ell_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_core(p...));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsr2ell(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               const rocsparse_mat_descr csr_descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               float*                    ell_val,
                                               rocsparse_int*            ell_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_impl(handle,
                                                      m,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsr2ell(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               const rocsparse_mat_descr csr_descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               const rocsparse_mat_descr ell_descr,
                                               rocsparse_int             ell_width,
                                               double*                   ell_val,
                                               rocsparse_int*            ell_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_impl(handle,
                                                      m,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsr2ell(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               const rocsparse_mat_descr      csr_descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               const rocsparse_mat_descr      ell_descr,
                                               rocsparse_int                  ell_width,
                                               rocsparse_float_complex*       ell_val,
                                               rocsparse_int*                 ell_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_impl(handle,
                                                      m,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsr2ell(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               const rocsparse_mat_descr       csr_descr,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               const rocsparse_mat_descr       ell_descr,
                                               rocsparse_int                   ell_width,
                                               rocsparse_double_complex*       ell_val,
                                               rocsparse_int*                  ell_col_ind)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_impl(handle,
                                                      m,
                                                      csr_descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind,
                                                      ell_descr,
                                                      ell_width,
                                                      ell_val,
                                                      ell_col_ind));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
