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

#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsv_buffer_size_template(rocsparse_handle          handle,
                                                       rocsparse_operation       trans,
                                                       int64_t                   m_,
                                                       int64_t                   nnz_,
                                                       const rocsparse_mat_descr descr,
                                                       const void*               csr_val_,
                                                       const void*               csr_row_ptr_,
                                                       const void*               csr_col_ind_,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m           = static_cast<J>(m_);
    const I  nnz         = static_cast<I>(nnz_);
    const T* csr_val     = reinterpret_cast<const T*>(csr_val_);
    const I* csr_row_ptr = reinterpret_cast<const I*>(csr_row_ptr_);
    const J* csr_col_ind = reinterpret_cast<const J*>(csr_col_ind_);

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(4, descr);
    ROCSPARSE_CHECKARG_POINTER(8, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrsv_buffer_size"),
                         trans,
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
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

    // Check pointer arguments
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);

    // rocsparse_int max_nnz
    *buffer_size = 256;

    // rocsparse_int done_array[m]
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    // rocsparse_int workspace
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;

    // rocsparse_int workspace2
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(m);

    size_t rocprim_size = 0;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int, J>(
        handle, m, startbit, endbit, &rocprim_size)));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // On transposed case, we might need more temporary storage for transposing
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        size_t transpose_size;

        // Determine rocprim buffer size
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, I>(
            handle, nnz, startbit, endbit, &transpose_size)));

        // rocPRIM does not support in-place sorting, so we need an additional buffer
        transpose_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
        transpose_size += ((rocsparse::max(sizeof(I), sizeof(T)) * nnz - 1) / 256 + 1) * 256;

        *buffer_size = rocsparse::max(*buffer_size, transpose_size);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                  \
    template rocsparse_status rocsparse::csrsv_buffer_size_template<I, J, T>( \
        rocsparse_handle          handle,                                     \
        rocsparse_operation       trans,                                      \
        int64_t                   m,                                          \
        int64_t                   nnz,                                        \
        const rocsparse_mat_descr descr,                                      \
        const void*               csr_val,                                    \
        const void*               csr_row_ptr,                                \
        const void*               csr_col_ind,                                \
        rocsparse_mat_info        info,                                       \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

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
#define C_IMPL(NAME, T)                                                                            \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                             \
                                     rocsparse_operation       trans,                              \
                                     rocsparse_int             m,                                  \
                                     rocsparse_int             nnz,                                \
                                     const rocsparse_mat_descr descr,                              \
                                     const T*                  csr_val,                            \
                                     const rocsparse_int*      csr_row_ptr,                        \
                                     const rocsparse_int*      csr_col_ind,                        \
                                     rocsparse_mat_info        info,                               \
                                     size_t*                   buffer_size)                        \
    try                                                                                            \
    {                                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                                   \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_buffer_size_template<rocsparse_int,            \
                                                                         rocsparse_int,            \
                                                                         T>(                       \
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size))); \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    catch(...)                                                                                     \
    {                                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                                              \
    }

C_IMPL(rocsparse_scsrsv_buffer_size, float);
C_IMPL(rocsparse_dcsrsv_buffer_size, double);
C_IMPL(rocsparse_ccsrsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_buffer_size, rocsparse_double_complex);

#undef C_IMPL
