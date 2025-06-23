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

#include "internal/conversion/rocsparse_gebsr2gebsc.h"
#include "internal/conversion/rocsparse_coo2csr.h"
#include "internal/conversion/rocsparse_csr2coo.h"
#include "internal/conversion/rocsparse_inverse_permutation.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_gebsr2gebsc.hpp"
#include "rocsparse_utility.hpp"

#include "gebsr2gebsc_device.h"
#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    static rocsparse_status gebsr2gebsc_quickreturn(rocsparse_handle     handle,
                                                    rocsparse_int        mb,
                                                    rocsparse_int        nb,
                                                    rocsparse_int        nnzb,
                                                    const void*          bsr_val,
                                                    const rocsparse_int* bsr_row_ptr,
                                                    const rocsparse_int* bsr_col_ind,
                                                    rocsparse_int        row_block_dim,
                                                    rocsparse_int        col_block_dim,
                                                    void*                bsc_val,
                                                    rocsparse_int*       bsc_row_ind,
                                                    rocsparse_int*       bsc_col_ptr,
                                                    rocsparse_action     copy_values,
                                                    rocsparse_index_base idx_base,
                                                    void*                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(mb == 0 || nb == 0)
        {
            if(bsc_col_ptr != nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                    handle, nb + 1, static_cast<rocsparse_int>(idx_base), bsc_col_ptr));
            }

            return rocsparse_status_success;
        }

        if(nnzb == 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                handle, nb + 1, static_cast<rocsparse_int>(idx_base), bsc_col_ptr));

            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::gebsr2gebsc_template(rocsparse_handle     handle, //0
                                                 rocsparse_int        mb, //1
                                                 rocsparse_int        nb, //2
                                                 rocsparse_int        nnzb, //3
                                                 const T*             bsr_val, //4
                                                 const rocsparse_int* bsr_row_ptr, //5
                                                 const rocsparse_int* bsr_col_ind, //6
                                                 rocsparse_int        row_block_dim, //7
                                                 rocsparse_int        col_block_dim, //8
                                                 T*                   bsc_val, //9
                                                 rocsparse_int*       bsc_row_ind, //10
                                                 rocsparse_int*       bsc_col_ptr, //11
                                                 rocsparse_action     copy_values, //12
                                                 rocsparse_index_base idx_base, //13
                                                 void*                temp_buffer) //14
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgebsr2gebsc"),
                         mb,
                         nb,
                         nnzb,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         row_block_dim,
                         col_block_dim,
                         (const void*&)bsc_val,
                         (const void*&)bsc_row_ind,
                         (const void*&)bsc_col_ptr,
                         copy_values,
                         idx_base,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(12, copy_values);
    ROCSPARSE_CHECKARG_ENUM(13, idx_base);
    ROCSPARSE_CHECKARG_SIZE(1, mb);
    ROCSPARSE_CHECKARG_SIZE(2, nb);
    ROCSPARSE_CHECKARG_SIZE(3, nnzb);
    ROCSPARSE_CHECKARG_SIZE(7, row_block_dim);
    ROCSPARSE_CHECKARG_SIZE(8, col_block_dim);
    ROCSPARSE_CHECKARG(7, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(8, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse::gebsr2gebsc_quickreturn(handle,
                                                                       mb,
                                                                       nb,
                                                                       nnzb,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       row_block_dim,
                                                                       col_block_dim,
                                                                       bsc_val,
                                                                       bsc_row_ind,
                                                                       bsc_col_ptr,
                                                                       copy_values,
                                                                       idx_base,
                                                                       temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(5, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_col_ind);

    ROCSPARSE_CHECKARG_ARRAY(11, nb, bsc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsc_row_ind);

    if(copy_values == rocsparse_action_numeric)
    {
        ROCSPARSE_CHECKARG_ARRAY(4, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsc_val);
    }

    ROCSPARSE_CHECKARG_ARRAY(14, nnzb, temp_buffer);

    // Stream
    hipStream_t stream   = handle->stream;
    uint32_t    startbit = 0;
    uint32_t    endbit   = rocsparse::clz(nb);

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // work1 buffer
    rocsparse_int* tmp_work1 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;

    // work2 buffer
    rocsparse_int* tmp_work2 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Load CSR column indices into work1 buffer
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        tmp_work1, bsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyDeviceToDevice, stream));

    if(copy_values == rocsparse_action_symbolic)
    {
        // action symbolic

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, bsr_row_ptr, nnzb, mb, bsc_row_ind, idx_base));
        // Stable sort COO by columns
        rocsparse::primitives::double_buffer<rocsparse_int> keys(tmp_work1, tmp_perm);
        rocsparse::primitives::double_buffer<rocsparse_int> vals(bsc_row_ind, tmp_work2);

        size_t size = 0;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::radix_sort_pairs_buffer_size<rocsparse_int, rocsparse_int>(
                handle, nnzb, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, nnzb, startbit, endbit, size, tmp_rocprim));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnzb, nb, bsc_col_ptr, idx_base));

        // Copy bsc_row_ind if not current
        if(vals.current() != bsc_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(bsc_row_ind,
                                               vals.current(),
                                               sizeof(rocsparse_int) * nnzb,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
    }
    else
    {
        // action numeric

        // Create identitiy permutation
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnzb, tmp_perm));

        // Stable sort COO by columns
        rocsparse::primitives::double_buffer<rocsparse_int> keys(tmp_work1, bsc_row_ind);
        rocsparse::primitives::double_buffer<rocsparse_int> vals(tmp_perm, tmp_work2);

        size_t size = 0;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::radix_sort_pairs_buffer_size<rocsparse_int, rocsparse_int>(
                handle, nnzb, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, nnzb, startbit, endbit, size, tmp_rocprim));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnzb, nb, bsc_col_ptr, idx_base));

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, bsr_row_ptr, nnzb, mb, tmp_work1, idx_base));

// Permute row indices and values
#define GEBSR2GEBSC_DIM 512
        dim3 gebsr2gebsc_blocks((nnzb - 1) / GEBSR2GEBSC_DIM + 1);
        dim3 gebsr2gebsc_threads(GEBSR2GEBSC_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gebsr2gebsc_permute_kernel<GEBSR2GEBSC_DIM>),
                                           gebsr2gebsc_blocks,
                                           gebsr2gebsc_threads,
                                           0,
                                           stream,
                                           nnzb,
                                           col_block_dim * row_block_dim,
                                           tmp_work1,
                                           bsr_val,
                                           vals.current(),
                                           bsc_row_ind,
                                           bsc_val);
#undef GEBSR2GEBSC_DIM
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    static rocsparse_status gebsr2gebsc_buffer_size_quickreturn(rocsparse_handle handle,
                                                                rocsparse_int    mb,
                                                                rocsparse_int    nb,
                                                                rocsparse_int    nnzb,
                                                                const void*      bsr_val,
                                                                const void*      bsr_row_ptr,
                                                                const void*      bsr_col_ind,
                                                                rocsparse_int    row_block_dim,
                                                                rocsparse_int    col_block_dim,
                                                                size_t*          p_buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(mb == 0 || nb == 0)
        {
            *p_buffer_size = 0;
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T>
    rocsparse_status gebsr2gebsc_buffer_size_template(rocsparse_handle     handle, //0
                                                      rocsparse_int        mb, //1
                                                      rocsparse_int        nb, //2
                                                      rocsparse_int        nnzb, //3
                                                      const T*             bsr_val, //4
                                                      const rocsparse_int* bsr_row_ptr, //5
                                                      const rocsparse_int* bsr_col_ind, //6
                                                      rocsparse_int        row_block_dim, //7
                                                      rocsparse_int        col_block_dim, //8
                                                      size_t*              p_buffer_size) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging
        rocsparse::log_trace(handle,
                             "rocsparse_gebsr2gebsc_buffer_size",
                             mb,
                             nb,
                             nnzb,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             row_block_dim,
                             col_block_dim,
                             (const void*&)p_buffer_size);

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, mb);
        ROCSPARSE_CHECKARG_SIZE(2, nb);
        ROCSPARSE_CHECKARG_SIZE(3, nnzb);
        ROCSPARSE_CHECKARG_SIZE(7, row_block_dim);
        ROCSPARSE_CHECKARG_SIZE(8, col_block_dim);
        ROCSPARSE_CHECKARG(7, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(8, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(9, p_buffer_size);

        // Quick return if possible
        const rocsparse_status status
            = rocsparse::gebsr2gebsc_buffer_size_quickreturn(handle,
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             p_buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(4, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(5, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_col_ind);

        uint32_t startbit = 0;
        uint32_t endbit   = rocsparse::clz(nb);

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::radix_sort_pairs_buffer_size<rocsparse_int, rocsparse_int>(
                handle, nnzb, startbit, endbit, p_buffer_size)));

        *p_buffer_size = ((*p_buffer_size - 1) / 256 + 1) * 256;

        // rocPRIM does not support in-place sorting, so we need additional buffer
        // for all temporary arrays
        *p_buffer_size += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;
        *p_buffer_size += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;
        *p_buffer_size += ((sizeof(rocsparse_int) * nnzb - 1) / 256 + 1) * 256;

        return rocsparse_status_success;
    }
}

//
// EXTERN C WRAPPING OF THE TEMPLATE.
//

#define C_IMPL(NAME, TYPE)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle,                              \
                                     rocsparse_int        mb,                                  \
                                     rocsparse_int        nb,                                  \
                                     rocsparse_int        nnzb,                                \
                                     const TYPE*          bsr_val,                             \
                                     const rocsparse_int* bsr_row_ptr,                         \
                                     const rocsparse_int* bsr_col_ind,                         \
                                     rocsparse_int        row_block_dim,                       \
                                     rocsparse_int        col_block_dim,                       \
                                     size_t*              p_buffer_size)                       \
    try                                                                                        \
    {                                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                                               \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsc_buffer_size_template(handle,          \
                                                                              mb,              \
                                                                              nb,              \
                                                                              nnzb,            \
                                                                              bsr_val,         \
                                                                              bsr_row_ptr,     \
                                                                              bsr_col_ind,     \
                                                                              row_block_dim,   \
                                                                              col_block_dim,   \
                                                                              p_buffer_size)); \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                                          \
    }

C_IMPL(rocsparse_sgebsr2gebsc_buffer_size, float);
C_IMPL(rocsparse_dgebsr2gebsc_buffer_size, double);
C_IMPL(rocsparse_cgebsr2gebsc_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgebsr2gebsc_buffer_size, rocsparse_double_complex);

#undef C_IMPL

//
// EXTERN C WRAPPING OF THE TEMPLATE.
//
#define C_IMPL(NAME, TYPE)                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle,                \
                                     rocsparse_int        mb,                    \
                                     rocsparse_int        nb,                    \
                                     rocsparse_int        nnzb,                  \
                                     const TYPE*          bsr_val,               \
                                     const rocsparse_int* bsr_row_ptr,           \
                                     const rocsparse_int* bsr_col_ind,           \
                                     rocsparse_int        row_block_dim,         \
                                     rocsparse_int        col_block_dim,         \
                                     TYPE*                bsc_val,               \
                                     rocsparse_int*       bsc_row_ind,           \
                                     rocsparse_int*       bsc_col_ptr,           \
                                     rocsparse_action     copy_values,           \
                                     rocsparse_index_base idx_base,              \
                                     void*                buffer)                \
    try                                                                          \
    {                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                 \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsc_template(handle,        \
                                                                  mb,            \
                                                                  nb,            \
                                                                  nnzb,          \
                                                                  bsr_val,       \
                                                                  bsr_row_ptr,   \
                                                                  bsr_col_ind,   \
                                                                  row_block_dim, \
                                                                  col_block_dim, \
                                                                  bsc_val,       \
                                                                  bsc_row_ind,   \
                                                                  bsc_col_ptr,   \
                                                                  copy_values,   \
                                                                  idx_base,      \
                                                                  buffer));      \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

C_IMPL(rocsparse_sgebsr2gebsc, float);
C_IMPL(rocsparse_dgebsr2gebsc, double);
C_IMPL(rocsparse_cgebsr2gebsc, rocsparse_float_complex);
C_IMPL(rocsparse_zgebsr2gebsc, rocsparse_double_complex);
#undef C_IMPL
