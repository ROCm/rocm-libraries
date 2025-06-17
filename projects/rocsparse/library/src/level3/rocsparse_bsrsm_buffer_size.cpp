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

#include "internal/level3/rocsparse_bsrsm.h"
#include "rocsparse_bsrsm.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_primitives.hpp"

rocsparse_status rocsparse::bsrsm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_operation       trans_X,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nrhs,
                                                          rocsparse_int             nnzb,
                                                          const rocsparse_mat_descr descr,
                                                          const void*               bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             block_dim,
                                                          rocsparse_mat_info        info,
                                                          size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(mb == 0 || nrhs == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status bsrsm_buffer_size_checkarg(rocsparse_handle          handle, //0
                                                       rocsparse_direction       dir, //1
                                                       rocsparse_operation       trans_A, //2
                                                       rocsparse_operation       trans_X, //3
                                                       rocsparse_int             mb, //4
                                                       rocsparse_int             nrhs, //5
                                                       rocsparse_int             nnzb, //6
                                                       const rocsparse_mat_descr descr, //7
                                                       const void*               bsr_val, //8
                                                       const rocsparse_int*      bsr_row_ptr, //9
                                                       const rocsparse_int*      bsr_col_ind, //10
                                                       rocsparse_int             block_dim, //11
                                                       rocsparse_mat_info        info, //12
                                                       size_t*                   buffer_size) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans_A);
        ROCSPARSE_CHECKARG_ENUM(3, trans_X);
        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, nrhs);

        const rocsparse_status status = rocsparse::bsrsm_buffer_size_quickreturn(handle,
                                                                                 dir,
                                                                                 trans_A,
                                                                                 trans_X,
                                                                                 mb,
                                                                                 nrhs,
                                                                                 nnzb,
                                                                                 descr,
                                                                                 bsr_val,
                                                                                 bsr_row_ptr,
                                                                                 bsr_col_ind,
                                                                                 block_dim,
                                                                                 info,
                                                                                 buffer_size);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_SIZE(6, nnzb);
        ROCSPARSE_CHECKARG_POINTER(7, descr);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_col_ind);
        ROCSPARSE_CHECKARG_SIZE(11, block_dim);
        ROCSPARSE_CHECKARG(11, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(12, info);
        ROCSPARSE_CHECKARG_POINTER(13, buffer_size);

        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::bsrsm_buffer_size_core(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_X,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nrhs,
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

    // rocsparse_int max_nnz
    *buffer_size = 256;

    // 16 columns per block seem to work very well
    static constexpr uint32_t ncol = 16;

    const int narrays = (nrhs - 1) / ncol + 1;

    // int done_array
    *buffer_size += ((sizeof(int) * size_t(mb) * narrays - 1) / 256 + 1) * 256;

    // rocsparse_int workspace
    *buffer_size += ((sizeof(rocsparse_int) * mb - 1) / 256 + 1) * 256;

    // int workspace2
    *buffer_size += ((sizeof(int) * mb - 1) / 256 + 1) * 256;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(mb);

    size_t rocprim_size;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::radix_sort_pairs_buffer_size<int, rocsparse_int>(
            handle, mb, startbit, endbit, &rocprim_size)));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // Additional buffer to store transpose of B, if trans_X == rocsparse_operation_none
    if(trans_X == rocsparse_operation_none)
    {
        *buffer_size += ((sizeof(T) * size_t(mb) * block_dim * nrhs - 1) / 256 + 1) * 256;
    }

    // Additional buffer to store transpose A, if transA == rocsparse_operation_transpose
    if(trans_A == rocsparse_operation_transpose)
    {
        size_t transpose_size;

        // Determine rocprim buffer size
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::radix_sort_pairs_buffer_size<rocsparse_int, rocsparse_int>(
                handle, nnzb, startbit, endbit, &transpose_size)));

        // rocPRIM does not support in-place sorting, so we need an additional buffer
        transpose_size += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;
        transpose_size += ((sizeof(T) * size_t(nnzb) * block_dim * block_dim - 1) / 256 + 1) * 256;

        *buffer_size += transpose_size;
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename... P>
    static rocsparse_status bsrsm_buffer_size_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_Xbsrsm_buffer_size", p...);

        const rocsparse_status status = rocsparse::bsrsm_buffer_size_checkarg(p...);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsm_buffer_size_core(p...));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,             \
                                     rocsparse_direction       dir,                \
                                     rocsparse_operation       trans_A,            \
                                     rocsparse_operation       trans_X,            \
                                     rocsparse_int             mb,                 \
                                     rocsparse_int             nrhs,               \
                                     rocsparse_int             nnzb,               \
                                     const rocsparse_mat_descr descr,              \
                                     const TYPE*               bsr_val,            \
                                     const rocsparse_int*      bsr_row_ptr,        \
                                     const rocsparse_int*      bsr_col_ind,        \
                                     rocsparse_int             block_dim,          \
                                     rocsparse_mat_info        info,               \
                                     size_t*                   buffer_size)        \
    try                                                                            \
    {                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsm_buffer_size_impl(handle,        \
                                                                    dir,           \
                                                                    trans_A,       \
                                                                    trans_X,       \
                                                                    mb,            \
                                                                    nrhs,          \
                                                                    nnzb,          \
                                                                    descr,         \
                                                                    bsr_val,       \
                                                                    bsr_row_ptr,   \
                                                                    bsr_col_ind,   \
                                                                    block_dim,     \
                                                                    info,          \
                                                                    buffer_size)); \
        return rocsparse_status_success;                                           \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                              \
    }

C_IMPL(rocsparse_sbsrsm_buffer_size, float);
C_IMPL(rocsparse_dbsrsm_buffer_size, double);
C_IMPL(rocsparse_cbsrsm_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsm_buffer_size, rocsparse_double_complex);

#undef C_IMPL
