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

#include "internal/conversion/rocsparse_csr2csc.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2csc.hpp"
#include "rocsparse_utility.hpp"

#include "csr2csc_device.h"
#include "rocsparse_common.h"
#include "rocsparse_coo2csr.hpp"
#include "rocsparse_csr2coo.hpp"
#include "rocsparse_identity.hpp"
#include "rocsparse_primitives.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csr2csc_core(rocsparse_handle     handle,
                                         J                    m,
                                         J                    n,
                                         I                    nnz,
                                         const T*             csr_val,
                                         const I*             csr_row_ptr_begin,
                                         const I*             csr_row_ptr_end,
                                         const J*             csr_col_ind,
                                         T*                   csc_val,
                                         J*                   csc_row_ind,
                                         I*                   csc_col_ptr,
                                         rocsparse_action     copy_values,
                                         rocsparse_index_base idx_base,
                                         void*                temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(n);

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // work1 buffer
    J* tmp_work1 = reinterpret_cast<J*>(ptr);
    ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

    // Load CSR column indices into work1 buffer
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(tmp_work1, csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));

    if(copy_values == rocsparse_action_symbolic)
    {
        // action symbolic

        // work2 buffer
        J* tmp_work2 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        // perm buffer
        J* tmp_perm = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2coo_core(
            handle, csr_row_ptr_begin, csr_row_ptr_end, nnz, m, csc_row_ind, idx_base));

        // Stable sort COO by columns
        rocsparse::primitives::double_buffer<J> keys(tmp_work1, tmp_perm);
        rocsparse::primitives::double_buffer<J> vals(csc_row_ind, tmp_work2);

        size_t size = 0;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
            handle, nnz, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, nnz, startbit, endbit, size, tmp_rocprim));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::coo2csr_core(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

        // Copy csc_row_ind if not current
        if(vals.current() != csc_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csc_row_ind, vals.current(), sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }
    else
    {
        // action numeric

        // work2 buffer
        I* tmp_work2 = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * nnz - 1) / 256 + 1) * 256;

        // perm buffer
        I* tmp_perm = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * nnz - 1) / 256 + 1) * 256;

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        // Create identitiy permutation
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_core(handle, nnz, tmp_perm));

        // Stable sort COO by columns
        rocsparse::primitives::double_buffer<J> keys(tmp_work1, csc_row_ind);
        rocsparse::primitives::double_buffer<I> vals(tmp_perm, tmp_work2);

        size_t size = 0;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, I>(
            handle, nnz, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, nnz, startbit, endbit, size, tmp_rocprim));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::coo2csr_core(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2coo_core(
            handle, csr_row_ptr_begin, csr_row_ptr_end, nnz, m, tmp_work1, idx_base));

// Permute row indices and values
#define CSR2CSC_DIM 512
        dim3 csr2csc_blocks((nnz - 1) / CSR2CSC_DIM + 1);
        dim3 csr2csc_threads(CSR2CSC_DIM);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2csc_permute_kernel<CSR2CSC_DIM>),
                                           csr2csc_blocks,
                                           csr2csc_threads,
                                           0,
                                           stream,
                                           nnz,
                                           tmp_work1,
                                           csr_val,
                                           vals.current(),
                                           csc_row_ind,
                                           csc_val);
#undef CSR2CSC_DIM
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csr2csc_template(rocsparse_handle     handle,
                                             J                    m,
                                             J                    n,
                                             I                    nnz,
                                             const T*             csr_val,
                                             const I*             csr_row_ptr,
                                             const J*             csr_col_ind,
                                             T*                   csc_val,
                                             J*                   csc_row_ind,
                                             I*                   csc_col_ptr,
                                             rocsparse_action     copy_values,
                                             rocsparse_index_base idx_base,
                                             void*                temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        if(nnz == 0 && csc_col_ptr != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, n + 1, static_cast<I>(idx_base), csc_col_ptr));
        }
        return rocsparse_status_success;
    }

    if(temp_buffer == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_core(handle,
                                                      m,
                                                      n,
                                                      nnz,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_row_ptr + 1,
                                                      csr_col_ind,
                                                      csc_val,
                                                      csc_row_ind,
                                                      csc_col_ptr,
                                                      copy_values,
                                                      idx_base,
                                                      temp_buffer));
    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csr2csc_impl(rocsparse_handle     handle, //0
                                         J                    m, //1
                                         J                    n, //2
                                         I                    nnz, //3
                                         const T*             csr_val, //4
                                         const I*             csr_row_ptr, //5
                                         const J*             csr_col_ind, //6
                                         T*                   csc_val, //7
                                         J*                   csc_row_ind, //8
                                         I*                   csc_col_ptr, //9
                                         rocsparse_action     copy_values, //10
                                         rocsparse_index_base idx_base, //11
                                         void*                temp_buffer) //12
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsr2csc"),
                         m,
                         n,
                         nnz,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)csc_val,
                         (const void*&)csc_row_ind,
                         (const void*&)csc_col_ptr,
                         copy_values,
                         idx_base,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(9, n, csc_col_ptr);
    ROCSPARSE_CHECKARG_ENUM(10, copy_values);
    ROCSPARSE_CHECKARG_ENUM(11, idx_base);

    if(copy_values == rocsparse_action_numeric)
    {
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csc_val);
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        if(nnz == 0 && csc_col_ptr != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, n + 1, static_cast<I>(idx_base), csc_col_ptr));
        }
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(12, temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_template(handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          csc_val,
                                                          csc_row_ind,
                                                          csc_col_ptr,
                                                          copy_values,
                                                          idx_base,
                                                          temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                        \
    template rocsparse_status rocsparse::csr2csc_core<ITYPE, JTYPE, TTYPE>(     \
        rocsparse_handle     handle,                                            \
        JTYPE                m,                                                 \
        JTYPE                n,                                                 \
        ITYPE                nnz,                                               \
        const TTYPE*         csr_val,                                           \
        const ITYPE*         csr_row_ptr_begin,                                 \
        const ITYPE*         csr_row_ptr_end,                                   \
        const JTYPE*         csr_col_ind,                                       \
        TTYPE*               csc_val,                                           \
        JTYPE*               csc_row_ind,                                       \
        ITYPE*               csc_col_ptr,                                       \
        rocsparse_action     copy_values,                                       \
        rocsparse_index_base idx_base,                                          \
        void*                temp_buffer);                                                     \
    template rocsparse_status rocsparse::csr2csc_impl<ITYPE, JTYPE, TTYPE>(     \
        rocsparse_handle     handle,                                            \
        JTYPE                m,                                                 \
        JTYPE                n,                                                 \
        ITYPE                nnz,                                               \
        const TTYPE*         csr_val,                                           \
        const ITYPE*         csr_row_ptr,                                       \
        const JTYPE*         csr_col_ind,                                       \
        TTYPE*               csc_val,                                           \
        JTYPE*               csc_row_ind,                                       \
        ITYPE*               csc_col_ptr,                                       \
        rocsparse_action     copy_values,                                       \
        rocsparse_index_base idx_base,                                          \
        void*                temp_buffer);                                                     \
    template rocsparse_status rocsparse::csr2csc_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle     handle,                                            \
        JTYPE                m,                                                 \
        JTYPE                n,                                                 \
        ITYPE                nnz,                                               \
        const TTYPE*         csr_val,                                           \
        const ITYPE*         csr_row_ptr,                                       \
        const JTYPE*         csr_col_ind,                                       \
        TTYPE*               csc_val,                                           \
        JTYPE*               csc_row_ind,                                       \
        ITYPE*               csc_col_ptr,                                       \
        rocsparse_action     copy_values,                                       \
        rocsparse_index_base idx_base,                                          \
        void*                temp_buffer)

INSTANTIATE(int32_t, int32_t, int8_t);
INSTANTIATE(int64_t, int32_t, int8_t);
INSTANTIATE(int32_t, int64_t, int8_t);
INSTANTIATE(int64_t, int64_t, int8_t);

INSTANTIATE(int32_t, int32_t, uint8_t);
INSTANTIATE(int64_t, int32_t, uint8_t);
INSTANTIATE(int32_t, int64_t, uint8_t);
INSTANTIATE(int64_t, int64_t, uint8_t);

INSTANTIATE(int32_t, int32_t, uint32_t);
INSTANTIATE(int64_t, int32_t, uint32_t);
INSTANTIATE(int32_t, int64_t, uint32_t);
INSTANTIATE(int64_t, int64_t, uint32_t);

INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int64_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t);
INSTANTIATE(int64_t, int64_t, int32_t);

INSTANTIATE(int32_t, int32_t, int64_t);
INSTANTIATE(int64_t, int32_t, int64_t);
INSTANTIATE(int32_t, int64_t, int64_t);
INSTANTIATE(int64_t, int64_t, int64_t);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, float);

INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, double);

INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);

INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE

template <typename I, typename J>
rocsparse_status rocsparse::csr2csc_buffer_size_core(rocsparse_handle handle,
                                                     J                m,
                                                     J                n,
                                                     I                nnz,
                                                     const I*         csr_row_ptr_begin,
                                                     const I*         csr_row_ptr_end,
                                                     const J*         csr_col_ind,
                                                     rocsparse_action copy_values,
                                                     size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(n);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
        handle, nnz, startbit, endbit, buffer_size)));

    *buffer_size = ((*buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays
    *buffer_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
    *buffer_size += ((rocsparse::max(sizeof(I), sizeof(J)) * nnz - 1) / 256 + 1) * 256;
    *buffer_size += ((rocsparse::max(sizeof(I), sizeof(J)) * nnz - 1) / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2csc_buffer_size_template(rocsparse_handle handle,
                                                         J                m,
                                                         J                n,
                                                         I                nnz,
                                                         const I*         csr_row_ptr,
                                                         const J*         csr_col_ind,
                                                         rocsparse_action copy_values,
                                                         size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_buffer_size_core(
        handle, m, n, nnz, csr_row_ptr, csr_row_ptr + 1, csr_col_ind, copy_values, buffer_size));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2csc_buffer_size_impl(rocsparse_handle handle,
                                                     J                m,
                                                     J                n,
                                                     I                nnz,
                                                     const I*         csr_row_ptr,
                                                     const J*         csr_col_ind,
                                                     rocsparse_action copy_values,
                                                     size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csr2csc_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         copy_values,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ENUM(6, copy_values);
    ROCSPARSE_CHECKARG_POINTER(7, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_buffer_size_template(
        handle, m, n, nnz, csr_row_ptr, csr_col_ind, copy_values, buffer_size));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE)                                                    \
    template rocsparse_status rocsparse::csr2csc_buffer_size_core<ITYPE, JTYPE>(     \
        rocsparse_handle handle,                                                     \
        JTYPE            m,                                                          \
        JTYPE            n,                                                          \
        ITYPE            nnz,                                                        \
        const ITYPE*     csr_row_ptr_begin,                                          \
        const ITYPE*     csr_row_ptr_end,                                            \
        const JTYPE*     csr_col_ind,                                                \
        rocsparse_action copy_values,                                                \
        size_t*          buffer_size);                                                        \
                                                                                     \
    template rocsparse_status rocsparse::csr2csc_buffer_size_template<ITYPE, JTYPE>( \
        rocsparse_handle handle,                                                     \
        JTYPE            m,                                                          \
        JTYPE            n,                                                          \
        ITYPE            nnz,                                                        \
        const ITYPE*     csr_row_ptr,                                                \
        const JTYPE*     csr_col_ind,                                                \
        rocsparse_action copy_values,                                                \
        size_t*          buffer_size);                                                        \
                                                                                     \
    template rocsparse_status rocsparse::csr2csc_buffer_size_impl<ITYPE, JTYPE>(     \
        rocsparse_handle handle,                                                     \
        JTYPE            m,                                                          \
        JTYPE            n,                                                          \
        ITYPE            nnz,                                                        \
        const ITYPE*     csr_row_ptr,                                                \
        const JTYPE*     csr_col_ind,                                                \
        rocsparse_action copy_values,                                                \
        size_t*          buffer_size)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_action     copy_values,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_buffer_size_impl(
        handle, m, n, nnz, csr_row_ptr, csr_col_ind, copy_values, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#define CIMPL(NAME, T)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle,        \
                                     rocsparse_int        m,             \
                                     rocsparse_int        n,             \
                                     rocsparse_int        nnz,           \
                                     const T*             csr_val,       \
                                     const rocsparse_int* csr_row_ptr,   \
                                     const rocsparse_int* csr_col_ind,   \
                                     T*                   csc_val,       \
                                     rocsparse_int*       csc_row_ind,   \
                                     rocsparse_int*       csc_col_ptr,   \
                                     rocsparse_action     copy_values,   \
                                     rocsparse_index_base idx_base,      \
                                     void*                temp_buffer)   \
    try                                                                  \
    {                                                                    \
        ROCSPARSE_ROUTINE_TRACE;                                         \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_impl(handle,        \
                                                          m,             \
                                                          n,             \
                                                          nnz,           \
                                                          csr_val,       \
                                                          csr_row_ptr,   \
                                                          csr_col_ind,   \
                                                          csc_val,       \
                                                          csc_row_ind,   \
                                                          csc_col_ptr,   \
                                                          copy_values,   \
                                                          idx_base,      \
                                                          temp_buffer)); \
        return rocsparse_status_success;                                 \
    }                                                                    \
    catch(...)                                                           \
    {                                                                    \
        RETURN_ROCSPARSE_EXCEPTION();                                    \
    }

CIMPL(rocsparse_scsr2csc, float);
CIMPL(rocsparse_ccsr2csc, rocsparse_float_complex);
CIMPL(rocsparse_dcsr2csc, double);
CIMPL(rocsparse_zcsr2csc, rocsparse_double_complex);
#undef CIMPL
