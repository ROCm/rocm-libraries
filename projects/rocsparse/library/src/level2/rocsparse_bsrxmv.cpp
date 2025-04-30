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

#include "internal/level2/rocsparse_bsrxmv.h"
#include "rocsparse_bsrxmv.hpp"
#include "rocsparse_bsrxmv_spzl.hpp"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::bsrxmv_template_dispatch(rocsparse_handle          handle,
                                                     rocsparse_direction       dir,
                                                     rocsparse_operation       trans,
                                                     J                         size_of_mask,
                                                     J                         mb,
                                                     J                         nb,
                                                     I                         nnzb,
                                                     const T*                  alpha_device_host,
                                                     const rocsparse_mat_descr descr,
                                                     const T*                  bsr_val,
                                                     const J*                  bsr_mask_ptr,
                                                     const I*                  bsr_row_ptr,
                                                     const I*                  bsr_end_ptr,
                                                     const J*                  bsr_col_ind,
                                                     J                         block_dim,
                                                     const T*                  x,
                                                     const T*                  beta_device_host,
                                                     T*                        y)
{
    ROCSPARSE_ROUTINE_TRACE;

    // LCOV_EXCL_START
    // Run different bsrxmv kernels
    if(handle->wavefront_size == 32)
    {
        rocsparse::bsrxmvn_general(handle,
                                   dir,
                                   mb,
                                   alpha_device_host,
                                   size_of_mask,
                                   bsr_mask_ptr,
                                   bsr_row_ptr,
                                   bsr_end_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   block_dim,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
        return rocsparse_status_success;
    }
    // LCOV_EXCL_STOP

    if(block_dim == 2)
    {
        rocsparse::bsrxmvn_2x2(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               size_of_mask,
                               bsr_mask_ptr,
                               bsr_row_ptr,
                               bsr_end_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else if(block_dim == 3)
    {
        rocsparse::bsrxmvn_3x3(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               size_of_mask,
                               bsr_mask_ptr,
                               bsr_row_ptr,
                               bsr_end_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else if(block_dim == 4)
    {
        rocsparse::bsrxmvn_4x4(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               size_of_mask,
                               bsr_mask_ptr,
                               bsr_row_ptr,
                               bsr_end_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else if(block_dim == 5)
    {
        rocsparse::bsrxmvn_5x5(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               size_of_mask,
                               bsr_mask_ptr,
                               bsr_row_ptr,
                               bsr_end_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else if(block_dim == 8)
    {
        rocsparse::bsrxmvn_8x8(handle,
                               dir,
                               mb,
                               nnzb,
                               alpha_device_host,
                               size_of_mask,
                               bsr_mask_ptr,
                               bsr_row_ptr,
                               bsr_end_ptr,
                               bsr_col_ind,
                               bsr_val,
                               x,
                               beta_device_host,
                               y,
                               descr->base);
    }
    else if(block_dim == 16)
    {
        rocsparse::bsrxmvn_16x16(handle,
                                 dir,
                                 mb,
                                 nnzb,
                                 alpha_device_host,
                                 size_of_mask,
                                 bsr_mask_ptr,
                                 bsr_row_ptr,
                                 bsr_end_ptr,
                                 bsr_col_ind,
                                 bsr_val,
                                 x,
                                 beta_device_host,
                                 y,
                                 descr->base);
    }
    else if(block_dim > 16 && block_dim <= 32)
    {
        rocsparse::bsrxmvn_17_32(handle,
                                 dir,
                                 mb,
                                 nnzb,
                                 alpha_device_host,
                                 size_of_mask,
                                 bsr_mask_ptr,
                                 bsr_row_ptr,
                                 bsr_end_ptr,
                                 bsr_col_ind,
                                 bsr_val,
                                 block_dim,
                                 x,
                                 beta_device_host,
                                 y,
                                 descr->base);
    }
    else
    {
        rocsparse::bsrxmvn_general(handle,
                                   dir,
                                   mb,
                                   alpha_device_host,
                                   size_of_mask,
                                   bsr_mask_ptr,
                                   bsr_row_ptr,
                                   bsr_end_ptr,
                                   bsr_col_ind,
                                   bsr_val,
                                   block_dim,
                                   x,
                                   beta_device_host,
                                   y,
                                   descr->base);
    }
    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::bsrxmv_template(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans,
                                            J                         size_of_mask,
                                            J                         mb,
                                            J                         nb,
                                            I                         nnzb,
                                            const T*                  alpha_device_host,
                                            const rocsparse_mat_descr descr,
                                            const T*                  bsr_val,
                                            const J*                  bsr_mask_ptr,
                                            const I*                  bsr_row_ptr,
                                            const I*                  bsr_end_ptr,
                                            const J*                  bsr_col_ind,
                                            J                         block_dim,
                                            const T*                  x,
                                            const T*                  beta_device_host,
                                            T*                        y)
{
    ROCSPARSE_ROUTINE_TRACE;

    //
    // Check for valid handle and matrix descriptor
    //
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(8, descr);

    //
    // Logging
    //
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsrxmv"),
                         dir,
                         trans,
                         size_of_mask,
                         mb,
                         nb,
                         nnzb,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_mask_ptr,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_end_ptr,
                         (const void*&)bsr_col_ind,
                         block_dim,
                         (const void*&)x,
                         LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                         (const void*&)y);

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_ENUM(2, trans);
    ROCSPARSE_CHECKARG(
        2, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

    if(block_dim == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    // Check matrix type
    ROCSPARSE_CHECKARG(
        8, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(8,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    //
    // Check sizes
    //
    ROCSPARSE_CHECKARG_SIZE(3, size_of_mask);
    ROCSPARSE_CHECKARG_SIZE(4, mb);
    ROCSPARSE_CHECKARG_SIZE(5, nb);
    ROCSPARSE_CHECKARG_SIZE(6, nnzb);
    ROCSPARSE_CHECKARG_SIZE(14, block_dim);
    ROCSPARSE_CHECKARG(14, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    //
    // Quick return if possible
    //
    if(mb == 0 || nb == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize = (bsr_mask_ptr == nullptr) ? block_dim * mb : block_dim * size_of_mask;
        if(ysize > 0)
        {
            if(y == nullptr && beta_device_host == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrxmv_scale_array<256>),
                                               dim3((ysize - 1) / 256 + 1),
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               mb,
                                               size_of_mask,
                                               block_dim,
                                               bsr_mask_ptr,
                                               y,
                                               beta_device_host,
                                               descr->base);
        }

        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    ROCSPARSE_CHECKARG_POINTER(7, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(16, beta_device_host);

    //
    // Another quick return.
    //
    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //

    ROCSPARSE_CHECKARG_ARRAY(10, size_of_mask, bsr_mask_ptr);
    ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(12, mb, bsr_end_ptr);
    ROCSPARSE_CHECKARG_ARRAY(15, nb, x);
    ROCSPARSE_CHECKARG_ARRAY(17, mb, y);

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(13, nnzb, bsr_col_ind);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrxmv_template_dispatch(handle,
                                                                  dir,
                                                                  trans,
                                                                  size_of_mask,
                                                                  mb,
                                                                  nb,
                                                                  nnzb,
                                                                  alpha_device_host,
                                                                  descr,
                                                                  bsr_val,
                                                                  bsr_mask_ptr,
                                                                  bsr_row_ptr,
                                                                  bsr_end_ptr,
                                                                  bsr_col_ind,
                                                                  block_dim,
                                                                  x,
                                                                  beta_device_host,
                                                                  y));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,       \
                                     rocsparse_direction       dir,          \
                                     rocsparse_operation       trans,        \
                                     rocsparse_int             size_of_mask, \
                                     rocsparse_int             mb,           \
                                     rocsparse_int             nb,           \
                                     rocsparse_int             nnzb,         \
                                     const TYPE*               alpha,        \
                                     const rocsparse_mat_descr descr,        \
                                     const TYPE*               bsr_val,      \
                                     const rocsparse_int*      bsr_mask_ptr, \
                                     const rocsparse_int*      bsr_row_ptr,  \
                                     const rocsparse_int*      bsr_end_ptr,  \
                                     const rocsparse_int*      bsr_col_ind,  \
                                     rocsparse_int             block_dim,    \
                                     const TYPE*               x,            \
                                     const TYPE*               beta,         \
                                     TYPE*                     y)            \
    try                                                                      \
    {                                                                        \
        ROCSPARSE_ROUTINE_TRACE;                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrxmv_template(handle,         \
                                                             dir,            \
                                                             trans,          \
                                                             size_of_mask,   \
                                                             mb,             \
                                                             nb,             \
                                                             nnzb,           \
                                                             alpha,          \
                                                             descr,          \
                                                             bsr_val,        \
                                                             bsr_mask_ptr,   \
                                                             bsr_row_ptr,    \
                                                             bsr_end_ptr,    \
                                                             bsr_col_ind,    \
                                                             block_dim,      \
                                                             x,              \
                                                             beta,           \
                                                             y));            \
        return rocsparse_status_success;                                     \
    }                                                                        \
    catch(...)                                                               \
    {                                                                        \
        RETURN_ROCSPARSE_EXCEPTION();                                        \
    }

C_IMPL(rocsparse_sbsrxmv, float);
C_IMPL(rocsparse_dbsrxmv, double);
C_IMPL(rocsparse_cbsrxmv, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrxmv, rocsparse_double_complex);

#undef C_IMPL
