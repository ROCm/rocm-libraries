/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_csritsv.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csritsv.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csrmv.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status csritsv_solve_impl(rocsparse_handle                     handle,
                                        rocsparse_int*                       host_nmaxiter,
                                        const rocsparse::floating_data_t<T>* host_tol,
                                        rocsparse::floating_data_t<T>*       host_history,
                                        rocsparse_operation                  trans,
                                        J                                    m,
                                        I                                    nnz,
                                        const T*                             alpha_device_host,
                                        const rocsparse_mat_descr            descr,
                                        const T*                             csr_val,
                                        const I*                             csr_row_ptr,
                                        const J*                             csr_col_ind,
                                        rocsparse_mat_info                   info,
                                        const T*                             x,
                                        T*                                   y,
                                        rocsparse_solve_policy               policy,
                                        void*                                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle and matrix descriptor
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(8, descr);
        ROCSPARSE_CHECKARG_POINTER(12, info);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsritsv_solve"),
                             (const void*&)host_nmaxiter,
                             (const void*&)host_tol,
                             (const void*&)host_history,
                             trans,
                             m,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)info,
                             (const void*&)x,
                             (const void*&)y,
                             policy,
                             (const void*&)temp_buffer);

        ROCSPARSE_CHECKARG_ENUM(4, trans);
        ROCSPARSE_CHECKARG_ENUM(15, policy);

        // Check matrix type
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->type != rocsparse_matrix_type_general
                            && descr->type != rocsparse_matrix_type_triangular),
                           rocsparse_status_not_implemented);

        // Check matrix sorting mode

        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        // Check sizes
        ROCSPARSE_CHECKARG_SIZE(5, m);
        ROCSPARSE_CHECKARG_SIZE(6, nnz);

        ROCSPARSE_CHECKARG_ARRAY(9, nnz, csr_val);

        ROCSPARSE_CHECKARG_ARRAY(10, m, csr_row_ptr);

        ROCSPARSE_CHECKARG_ARRAY(11, nnz, csr_col_ind);

        ROCSPARSE_CHECKARG(16,
                           temp_buffer,
                           (m > 0 && nnz > 0 && temp_buffer == nullptr),
                           rocsparse_status_invalid_pointer);

        ROCSPARSE_CHECKARG_POINTER(1, host_nmaxiter);

        ROCSPARSE_CHECKARG_POINTER(7, alpha_device_host);

        ROCSPARSE_CHECKARG_ARRAY(13, m, x);

        ROCSPARSE_CHECKARG_ARRAY(14, m, y);

        ROCSPARSE_CHECKARG(
            12, info, (m > 0 && info->csritsv_info == nullptr), rocsparse_status_invalid_pointer);

        static constexpr rocsparse_int host_nfreeiter = 0;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritsv_solve_ex_template(handle,
                                                                       host_nmaxiter,
                                                                       host_nfreeiter,
                                                                       host_tol,
                                                                       host_history,
                                                                       trans,
                                                                       m,
                                                                       nnz,
                                                                       alpha_device_host,
                                                                       descr,
                                                                       csr_val,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       info,
                                                                       x,
                                                                       y,
                                                                       policy,
                                                                       temp_buffer));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(I, J, T)                                          \
    template rocsparse_status rocsparse::csritsv_solve_impl<I, J, T>( \
        rocsparse_handle handle,                                      \
        rocsparse_int * host_nmaxiter,                                \
        const rocsparse::floating_data_t<T>* host_tol,                \
        rocsparse::floating_data_t<T>*       host_history,            \
        rocsparse_operation                  trans,                   \
        J                                    m,                       \
        I                                    nnz,                     \
        const T*                             alpha_device_host,       \
        const rocsparse_mat_descr            descr,                   \
        const T*                             csr_val,                 \
        const I*                             csr_row_ptr,             \
        const J*                             csr_col_ind,             \
        rocsparse_mat_info                   info,                    \
        const T*                             x,                       \
        T*                                   y,                       \
        rocsparse_solve_policy               policy,                  \
        void*                                temp_buffer)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, T)                                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle                     handle,            \
                                     rocsparse_int*                       host_nmaxiter,     \
                                     const rocsparse::floating_data_t<T>* host_tol,          \
                                     rocsparse::floating_data_t<T>*       host_history,      \
                                     rocsparse_operation                  trans,             \
                                     rocsparse_int                        m,                 \
                                     rocsparse_int                        nnz,               \
                                     const T*                             alpha_device_host, \
                                     const rocsparse_mat_descr            descr,             \
                                     const T*                             csr_val,           \
                                     const rocsparse_int*                 csr_row_ptr,       \
                                     const rocsparse_int*                 csr_col_ind,       \
                                     rocsparse_mat_info                   info,              \
                                     const T*                             x,                 \
                                     T*                                   y,                 \
                                     rocsparse_solve_policy               policy,            \
                                     void*                                temp_buffer)       \
    try                                                                                      \
    {                                                                                        \
        ROCSPARSE_ROUTINE_TRACE;                                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritsv_solve_impl(handle,                      \
                                                                host_nmaxiter,               \
                                                                host_tol,                    \
                                                                host_history,                \
                                                                trans,                       \
                                                                m,                           \
                                                                nnz,                         \
                                                                alpha_device_host,           \
                                                                descr,                       \
                                                                csr_val,                     \
                                                                csr_row_ptr,                 \
                                                                csr_col_ind,                 \
                                                                info,                        \
                                                                x,                           \
                                                                y,                           \
                                                                policy,                      \
                                                                temp_buffer));               \
        return rocsparse_status_success;                                                     \
    }                                                                                        \
    catch(...)                                                                               \
    {                                                                                        \
        RETURN_ROCSPARSE_EXCEPTION();                                                        \
    }

C_IMPL(rocsparse_scsritsv_solve, float);
C_IMPL(rocsparse_dcsritsv_solve, double);
C_IMPL(rocsparse_ccsritsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zcsritsv_solve, rocsparse_double_complex);

#undef C_IMPL
