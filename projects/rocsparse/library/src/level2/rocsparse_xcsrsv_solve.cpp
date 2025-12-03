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

#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_utility.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
namespace rocsparse
{

    static rocsparse_status xcsrsv_solve_checkarg(rocsparse_handle          handle,
                                                  rocsparse_operation       trans,
                                                  rocsparse_int             m,
                                                  rocsparse_int             nnz,
                                                  const void*               alpha_device_host,
                                                  const rocsparse_mat_descr descr,
                                                  const void*               csr_val,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  const void*               x,
                                                  void*                     y,
                                                  rocsparse_solve_policy    policy,
                                                  void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        // Check for valid handle and matrix descriptor
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);
        ROCSPARSE_CHECKARG_POINTER(5, descr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(9, info);
        ROCSPARSE_CHECKARG_ARRAY(10, m, x);
        ROCSPARSE_CHECKARG_ARRAY(11, m, y);
        ROCSPARSE_CHECKARG_ENUM(12, policy);
        ROCSPARSE_CHECKARG_POINTER(13, temp_buffer);
        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status xcsrsv_solve(rocsparse_handle          handle,
                                         rocsparse_operation       trans,
                                         rocsparse_int             m,
                                         rocsparse_int             nnz,
                                         const T*                  alpha,
                                         const rocsparse_mat_descr descr,
                                         const T*                  csr_val,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_int*      csr_col_ind,
                                         rocsparse_mat_info        info,
                                         const T*                  x,
                                         T*                        y,
                                         rocsparse_solve_policy    policy,
                                         void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0)
            return rocsparse_status_success;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrsv_solve_checkarg(handle,
                                                                   trans,
                                                                   m,
                                                                   nnz,
                                                                   alpha,
                                                                   descr,
                                                                   csr_val,
                                                                   csr_row_ptr,
                                                                   csr_col_ind,
                                                                   info,
                                                                   x,
                                                                   y,
                                                                   policy,
                                                                   temp_buffer));

        rocsparse_csrsv_info   csrsv_info = (info != nullptr) ? info->get_csrsv_info() : nullptr;
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

                                   //
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

        _rocsparse_dnvec_descr dnvec_descr_x(1, m, rocsparse::get_datatype<T>(), x, nullptr, 1, 0);
        _rocsparse_dnvec_descr dnvec_descr_y(1, m, rocsparse::get_datatype<T>(), y, y, 1, 0);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                         trans,
                                                         rocsparse::get_datatype<T>(),
                                                         alpha,
                                                         static_cast<int64_t>(0),
                                                         &csr,
                                                         &dnvec_descr_x,
                                                         &dnvec_descr_y,
                                                         policy,
                                                         csrsv_info,
                                                         temp_buffer));

        return rocsparse_status_success;
    }
}

#define C_IMPL(NAME, T)                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             m,           \
                                     rocsparse_int             nnz,         \
                                     const T*                  alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const T*                  csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_mat_info        info,        \
                                     const T*                  x,           \
                                     T*                        y,           \
                                     rocsparse_solve_policy    policy,      \
                                     void*                     temp_buffer) \
    try                                                                     \
    {                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrsv_solve(handle,           \
                                                          trans,            \
                                                          m,                \
                                                          nnz,              \
                                                          alpha,            \
                                                          descr,            \
                                                          csr_val,          \
                                                          csr_row_ptr,      \
                                                          csr_col_ind,      \
                                                          info,             \
                                                          x,                \
                                                          y,                \
                                                          policy,           \
                                                          temp_buffer));    \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_scsrsv_solve, float);
C_IMPL(rocsparse_dcsrsv_solve, double);
C_IMPL(rocsparse_ccsrsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_solve, rocsparse_double_complex);

#undef C_IMPL
