/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_coosv.hpp"
#include "control.h"
#include "rocsparse_csrsv.hpp"
#include "utility.h"

#include "../conversion/rocsparse_coo2csr.hpp"

template <typename I, typename T>
rocsparse_status rocsparse::coosv_buffer_size_template(rocsparse_handle          handle,
                                                       rocsparse_operation       trans,
                                                       I                         m,
                                                       int64_t                   nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const T*                  coo_val,
                                                       const I*                  coo_row_ind,
                                                       const I*                  coo_col_ind,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_POINTER(8, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoosv_buffer_size"),
                         trans,
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)coo_val,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)info,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check matrix type
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);

    // Check for valid buffer_size pointer
    ROCSPARSE_CHECKARG_POINTER(9, buffer_size);

    // Quick return if possible
    if(m == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    // All must be null (zero matrix) or none null
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_val);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, coo_col_ind);

    // Call CSR buffer size
    *buffer_size = 0;

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        // Trick since it is not used in csrsv_buffer_size, otherwise we need to create a proper ptr array for nothing.
        const int32_t* ptr = (const int32_t*)0x4;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size_template(
            handle, trans, m, (int32_t)nnz, descr, coo_val, ptr, coo_col_ind, info, buffer_size));

        // For coosv we first convert from COO to CSR format.
        *buffer_size += sizeof(int32_t) * (m / 256 + 1) * 256;
    }
    else
    {
        // Trick since it is not used in csrsv_buffer_size, otherwise we need to create a proper ptr array for nothing.
        const int64_t* ptr = (const int64_t*)0x4;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size_template(
            handle, trans, m, nnz, descr, coo_val, ptr, coo_col_ind, info, buffer_size));

        // For coosv we first convert from COO to CSR format.
        *buffer_size += sizeof(int64_t) * (m / 256 + 1) * 256;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                    \
    template rocsparse_status rocsparse::coosv_buffer_size_template( \
        rocsparse_handle          handle,                            \
        rocsparse_operation       trans,                             \
        ITYPE                     m,                                 \
        int64_t                   nnz,                               \
        const rocsparse_mat_descr descr,                             \
        const TTYPE*              coo_val,                           \
        const ITYPE*              coo_row_ind,                       \
        const ITYPE*              coo_col_ind,                       \
        rocsparse_mat_info        info,                              \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

template <typename I, typename T>
rocsparse_status rocsparse::coosv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    I                         m,
                                                    int64_t                   nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_analysis_policy analysis,
                                                    rocsparse_solve_policy    solve,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_POINTER(8, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoosv_analysis"),
                         trans,
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)coo_val,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)info,
                         solve,
                         analysis,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(9, analysis);
    ROCSPARSE_CHECKARG_ENUM(10, solve);

    // Check matrix type
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);
    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(4,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(11, temp_buffer);

    // All must be null (zero matrix) or none null
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_val);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, coo_col_ind);

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        // convert to csr
        int32_t* csr_row_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += sizeof(int32_t) * (m / 256 + 1) * 256;

        const I* csr_col_ind = coo_col_ind;
        const T* csr_val     = coo_val;

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coo2csr_template(
            handle, coo_row_ind, (int32_t)nnz, m, csr_row_ptr, descr->base));

        // Call CSR analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis_template(handle,
                                                                     trans,
                                                                     m,
                                                                     (int32_t)nnz,
                                                                     descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
                                                                     info,
                                                                     analysis,
                                                                     solve,
                                                                     ptr));
    }
    else
    {
        // convert to csr
        int64_t* csr_row_ptr = reinterpret_cast<int64_t*>(ptr);
        ptr += sizeof(int64_t) * (m / 256 + 1) * 256;

        const I* csr_col_ind = coo_col_ind;
        const T* csr_val     = coo_val;

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::coo2csr_template(handle, coo_row_ind, nnz, m, csr_row_ptr, descr->base));

        // Call CSR analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis_template(handle,
                                                                     trans,
                                                                     m,
                                                                     nnz,
                                                                     descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind,
                                                                     info,
                                                                     analysis,
                                                                     solve,
                                                                     ptr));
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                 \
    template rocsparse_status rocsparse::coosv_analysis_template( \
        rocsparse_handle          handle,                         \
        rocsparse_operation       trans,                          \
        ITYPE                     m,                              \
        int64_t                   nnz,                            \
        const rocsparse_mat_descr descr,                          \
        const TTYPE*              coo_val,                        \
        const ITYPE*              coo_row_ind,                    \
        const ITYPE*              coo_col_ind,                    \
        rocsparse_mat_info        info,                           \
        rocsparse_analysis_policy analysis,                       \
        rocsparse_solve_policy    solve,                          \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

template <typename I, typename T>
rocsparse_status rocsparse::coosv_solve_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 I                         m,
                                                 int64_t                   nnz,
                                                 const T*                  alpha_device_host,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  coo_val,
                                                 const I*                  coo_row_ind,
                                                 const I*                  coo_col_ind,
                                                 rocsparse_mat_info        info,
                                                 const T*                  x,
                                                 T*                        y,
                                                 rocsparse_solve_policy    policy,
                                                 void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(5, descr);
    ROCSPARSE_CHECKARG_POINTER(9, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoosv"),
                         trans,
                         m,
                         nnz,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)coo_val,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)info,
                         (const void*&)x,
                         (const void*&)y,
                         policy,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(12, policy);

    // Check matrix type
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(13, temp_buffer);
    ROCSPARSE_CHECKARG_ARRAY(10, m, x);
    ROCSPARSE_CHECKARG_ARRAY(11, m, y);

    // All must be null (zero matrix) or none null
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, coo_col_ind);

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        int32_t* csr_row_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += sizeof(int32_t) * (m / 256 + 1) * 256;

        const I* csr_col_ind = coo_col_ind;
        const T* csr_val     = coo_val;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve_template(handle,
                                                                  trans,
                                                                  m,
                                                                  (int32_t)nnz,
                                                                  alpha_device_host,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  info,
                                                                  x,
                                                                  (int64_t)1,
                                                                  y,
                                                                  policy,
                                                                  ptr));
        return rocsparse_status_success;
    }
    else
    {
        int64_t* csr_row_ptr = reinterpret_cast<int64_t*>(ptr);
        ptr += sizeof(int64_t) * (m / 256 + 1) * 256;

        const I* csr_col_ind = coo_col_ind;
        const T* csr_val     = coo_val;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve_template(handle,
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
                                                                  (int64_t)1,
                                                                  y,
                                                                  policy,
                                                                  ptr));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, TTYPE)                                            \
    template rocsparse_status rocsparse::coosv_solve_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans,                                     \
        ITYPE                     m,                                         \
        int64_t                   nnz,                                       \
        const TTYPE*              alpha_device_host,                         \
        const rocsparse_mat_descr descr,                                     \
        const TTYPE*              coo_val,                                   \
        const ITYPE*              coo_row_ind,                               \
        const ITYPE*              coo_col_ind,                               \
        rocsparse_mat_info        info,                                      \
        const TTYPE*              x,                                         \
        TTYPE*                    y,                                         \
        rocsparse_solve_policy    policy,                                    \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE
