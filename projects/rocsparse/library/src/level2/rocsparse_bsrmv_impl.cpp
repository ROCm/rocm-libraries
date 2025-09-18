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

#include "internal/level2/rocsparse_bsrmv.h"
#include "internal/level2/rocsparse_csrmv.h"
#include "rocsparse_bsrmv.hpp"
#include "rocsparse_bsrxmv_spzl.hpp"
#include "rocsparse_common.h"
#include "rocsparse_csrmv.hpp"

template <typename I, typename J, typename A>
rocsparse_status rocsparse::bsrmv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans,
                                                    int64_t                   mb_,
                                                    int64_t                   nb_,
                                                    int64_t                   nnzb_,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               bsr_val_,
                                                    const void*               bsr_row_ptr_,
                                                    const void*               bsr_col_ind_,
                                                    int64_t                   block_dim_,
                                                    rocsparse_bsrmv_info*     p_bsrmv_info)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  mb          = static_cast<J>(mb_);
    const J  nb          = static_cast<J>(nb_);
    const I  nnzb        = static_cast<I>(nnzb_);
    const A* bsr_val     = reinterpret_cast<const A*>(bsr_val_);
    const I* bsr_row_ptr = reinterpret_cast<const I*>(bsr_row_ptr_);
    const J* bsr_col_ind = reinterpret_cast<const J*>(bsr_col_ind_);
    const J  block_dim   = static_cast<J>(block_dim_);

    p_bsrmv_info[0] = nullptr;

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        return rocsparse_status_success;
    }

    if(descr->storage_mode == rocsparse_storage_mode_sorted)
    {
        if(block_dim == 1)
        {
            p_bsrmv_info[0]                 = new _rocsparse_bsrmv_info();
            rocsparse_bsrmv_info bsrmv_info = p_bsrmv_info[0];

            rocsparse_csrmv_info csrmv_info;
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csrmv_analysis_template<I, J, A>(handle,
                                                             trans,
                                                             rocsparse::csrmv_alg_adaptive,
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             &csrmv_info)));
            bsrmv_info->set_csrmv_info(csrmv_info);
            return rocsparse_status_success;
        }
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename J, typename A>
    static rocsparse_status bsrmv_analysis_impl(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans,
                                                J                         mb,
                                                J                         nb,
                                                I                         nnzb,
                                                const rocsparse_mat_descr descr,
                                                const A*                  bsr_val,
                                                const I*                  bsr_row_ptr,
                                                const J*                  bsr_col_ind,
                                                J                         block_dim,
                                                rocsparse_mat_info        info)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        // Check for valid matrix descriptor and info struct
        ROCSPARSE_CHECKARG_POINTER(6, descr);
        ROCSPARSE_CHECKARG_POINTER(11, info);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<A>("rocsparse_Xbsrmv_analysis"),
                             dir,
                             trans,
                             mb,
                             nb,
                             nnzb,
                             (const void*&)descr,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             block_dim,
                             (const void*&)info);

        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans);

        ROCSPARSE_CHECKARG(
            2, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

        // Check matrix type
        ROCSPARSE_CHECKARG(6,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        // Check sizes

        ROCSPARSE_CHECKARG_SIZE(3, mb);
        ROCSPARSE_CHECKARG_SIZE(4, nb);
        ROCSPARSE_CHECKARG_SIZE(5, nnzb);
        ROCSPARSE_CHECKARG_SIZE(10, block_dim);
        ROCSPARSE_CHECKARG(10, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        // Quick return if possible
        if(mb == 0 || nb == 0 || nnzb == 0)
        {
            return rocsparse_status_success;
        }

        // Check the rest of pointer arguments
        ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(8, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_col_ind);

        rocsparse_bsrmv_info bsrmv_info = (info != nullptr) ? info->get_bsrmv_info() : nullptr;

        if(bsrmv_info != nullptr)
        {
            delete bsrmv_info;
            bsrmv_info = nullptr;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrmv_analysis_template<I, J, A>(handle,
                                                                               dir,
                                                                               trans,
                                                                               mb,
                                                                               nb,
                                                                               nnzb,
                                                                               descr,
                                                                               bsr_val,
                                                                               bsr_row_ptr,
                                                                               bsr_col_ind,
                                                                               block_dim,
                                                                               &bsrmv_info)));

        if(info != nullptr)
        {
            info->set_bsrmv_info(bsrmv_info);
        }
        else if(bsrmv_info != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }

        return rocsparse_status_success;
    }

}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::bsrmv_template_dispatch(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans,
                                                    J                         mb,
                                                    J                         nb,
                                                    I                         nnzb,
                                                    const T*                  alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const A*                  bsr_val,
                                                    const I*                  bsr_row_ptr,
                                                    const J*                  bsr_col_ind,
                                                    J                         block_dim,
                                                    const X*                  x,
                                                    const T*                  beta_device_host,
                                                    Y*                        y)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    //
    // block_dim == 1 is the CSR case
    //
    if(block_dim == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_rowsplit_template_dispatch(handle,
                                                                              trans,
                                                                              mb,
                                                                              nb,
                                                                              nnzb,
                                                                              alpha_device_host,
                                                                              descr,
                                                                              bsr_val,
                                                                              bsr_row_ptr,
                                                                              bsr_row_ptr + 1,
                                                                              bsr_col_ind,
                                                                              x,
                                                                              beta_device_host,
                                                                              y,
                                                                              false));
        return rocsparse_status_success;
    }

    // LCOV_EXCL_START
    // Run different bsrmv kernels
    if(handle->wavefront_size == 32)
    {

        rocsparse::bsrxmvn_general<T, I, J>(handle,
                                            dir,
                                            mb,
                                            alpha_device_host,
                                            0,
                                            nullptr,
                                            bsr_row_ptr,
                                            nullptr,
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
        rocsparse::bsrxmvn_2x2<T, I, J>(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        alpha_device_host,
                                        0,
                                        nullptr,
                                        bsr_row_ptr,
                                        nullptr,

                                        bsr_col_ind,
                                        bsr_val,
                                        x,
                                        beta_device_host,
                                        y,
                                        descr->base);
    }
    else if(block_dim == 3)
    {
        rocsparse::bsrxmvn_3x3<T, I, J>(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        alpha_device_host,
                                        0,
                                        nullptr,
                                        bsr_row_ptr,
                                        nullptr,

                                        bsr_col_ind,
                                        bsr_val,
                                        x,
                                        beta_device_host,
                                        y,
                                        descr->base);
    }
    else if(block_dim == 4)
    {
        rocsparse::bsrxmvn_4x4<T, I, J>(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        alpha_device_host,
                                        0,
                                        nullptr,
                                        bsr_row_ptr,
                                        nullptr,

                                        bsr_col_ind,
                                        bsr_val,
                                        x,
                                        beta_device_host,
                                        y,
                                        descr->base);
    }
    else if(block_dim == 5)
    {
        rocsparse::bsrxmvn_5x5<T, I, J>(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        alpha_device_host,
                                        0,
                                        nullptr,
                                        bsr_row_ptr,
                                        nullptr,

                                        bsr_col_ind,
                                        bsr_val,
                                        x,
                                        beta_device_host,
                                        y,
                                        descr->base);
    }
    else if(block_dim == 8)
    {
        rocsparse::bsrxmvn_8x8<T, I, J>(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        alpha_device_host,
                                        0,
                                        nullptr,
                                        bsr_row_ptr,
                                        nullptr,

                                        bsr_col_ind,
                                        bsr_val,
                                        x,
                                        beta_device_host,
                                        y,
                                        descr->base);
    }
    else if(block_dim == 16)
    {
        rocsparse::bsrxmvn_16x16<T, I, J>(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          alpha_device_host,
                                          0,
                                          nullptr,
                                          bsr_row_ptr,
                                          nullptr,

                                          bsr_col_ind,
                                          bsr_val,
                                          x,
                                          beta_device_host,
                                          y,
                                          descr->base);
    }
    else if(block_dim > 16 && block_dim <= 32)
    {

        rocsparse::bsrxmvn_17_32<T, I, J>(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          alpha_device_host,
                                          0,
                                          nullptr,
                                          bsr_row_ptr,
                                          nullptr,

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
        rocsparse::bsrxmvn_general<T, I, J>(handle,
                                            dir,
                                            mb,
                                            alpha_device_host,
                                            0,
                                            nullptr,
                                            bsr_row_ptr,
                                            nullptr,
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

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::bsrmv_adaptive_template_dispatch(rocsparse_handle    handle,
                                                             rocsparse_direction dir,
                                                             rocsparse_operation trans,
                                                             J                   mb,
                                                             J                   nb,
                                                             I                   nnzb,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  bsr_val,
                                                             const I*                  bsr_row_ptr,
                                                             const J*                  bsr_col_ind,
                                                             J                         block_dim,
                                                             rocsparse_bsrmv_info      bsrmv_info,
                                                             const X*                  x,
                                                             const T* beta_device_host,
                                                             Y*       y)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(trans != rocsparse_operation_none)
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }

    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // block_dim == 1 is the CSR case
    if(block_dim == 1)
    {
        rocsparse_csrmv_info csrmv_info
            = (bsrmv_info != nullptr) ? bsrmv_info->get_csrmv_info() : nullptr;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_adaptive_template_dispatch(handle,
                                                                              trans,
                                                                              mb,
                                                                              nb,
                                                                              nnzb,
                                                                              alpha_device_host,
                                                                              descr,
                                                                              bsr_val,
                                                                              bsr_row_ptr,
                                                                              bsr_col_ind,
                                                                              csrmv_info,
                                                                              x,
                                                                              beta_device_host,
                                                                              y,
                                                                              false));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template_dispatch(handle,
                                                                 dir,
                                                                 trans,
                                                                 mb,
                                                                 nb,
                                                                 nnzb,
                                                                 alpha_device_host,
                                                                 descr,
                                                                 bsr_val,
                                                                 bsr_row_ptr,
                                                                 bsr_col_ind,
                                                                 block_dim,
                                                                 x,
                                                                 beta_device_host,
                                                                 y));
    return rocsparse_status_success;
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::bsrmv_template(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           int64_t                   mb_,
                                           int64_t                   nb_,
                                           int64_t                   nnzb_,
                                           const void*               alpha_device_host_,
                                           const rocsparse_mat_descr descr,
                                           const void*               bsr_val_,
                                           const void*               bsr_row_ptr_,
                                           const void*               bsr_col_ind_,
                                           int64_t                   block_dim_,
                                           rocsparse_bsrmv_info      bsrmv_info,
                                           const void*               x_,
                                           const void*               beta_device_host_,
                                           void*                     y_)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  mb                = static_cast<J>(mb_);
    const J  nb                = static_cast<J>(nb_);
    const I  nnzb              = static_cast<I>(nnzb_);
    const A* bsr_val           = reinterpret_cast<const A*>(bsr_val_);
    const I* bsr_row_ptr       = reinterpret_cast<const I*>(bsr_row_ptr_);
    const J* bsr_col_ind       = reinterpret_cast<const J*>(bsr_col_ind_);
    const J  block_dim         = static_cast<J>(block_dim_);
    const X* x                 = reinterpret_cast<const X*>(x_);
    Y*       y                 = reinterpret_cast<Y*>(y_);
    const T* alpha_device_host = reinterpret_cast<const T*>(alpha_device_host_);
    const T* beta_device_host  = reinterpret_cast<const T*>(beta_device_host_);
    //
    // Quick return if possible
    //
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize = (trans == rocsparse_operation_none) ? block_dim * mb : block_dim * nb;
        if(ysize > 0)
        {
            if(y == nullptr && beta_device_host == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));
        }

        return rocsparse_status_success;
    }

    //
    // Another quick return.
    //
    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    if(bsrmv_info == nullptr || trans != rocsparse_operation_none
       || descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        // If bsrmv info is not available, call bsrmv general
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template_dispatch(handle,
                                                                     dir,
                                                                     trans,
                                                                     mb,
                                                                     nb,
                                                                     nnzb,
                                                                     alpha_device_host,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     block_dim,
                                                                     x,
                                                                     beta_device_host,
                                                                     y));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_adaptive_template_dispatch(handle,
                                                                              dir,
                                                                              trans,
                                                                              mb,
                                                                              nb,
                                                                              nnzb,
                                                                              alpha_device_host,
                                                                              descr,
                                                                              bsr_val,
                                                                              bsr_row_ptr,
                                                                              bsr_col_ind,
                                                                              block_dim,
                                                                              bsrmv_info,
                                                                              x,
                                                                              beta_device_host,
                                                                              y));
        return rocsparse_status_success;
    }
}

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    static rocsparse_status bsrmv_impl(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_operation       trans,
                                       J                         mb,
                                       J                         nb,
                                       I                         nnzb,
                                       const T*                  alpha_device_host,
                                       const rocsparse_mat_descr descr,
                                       const A*                  bsr_val,
                                       const I*                  bsr_row_ptr,
                                       const J*                  bsr_col_ind,
                                       J                         block_dim,
                                       rocsparse_mat_info        info,
                                       const X*                  x,
                                       const T*                  beta_device_host,
                                       Y*                        y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(7, descr);

        //
        // Logging
        //
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrmv"),
                             dir,
                             trans,
                             mb,
                             nb,
                             nnzb,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                             descr,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             block_dim,
                             info,
                             x,
                             LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                             y);

        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans);

        ROCSPARSE_CHECKARG(
            2, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

        // Check matrix type
        ROCSPARSE_CHECKARG(6,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        //
        // Check sizes
        //
        ROCSPARSE_CHECKARG_SIZE(3, mb);
        ROCSPARSE_CHECKARG_SIZE(4, nb);
        ROCSPARSE_CHECKARG_SIZE(5, nnzb);
        ROCSPARSE_CHECKARG_SIZE(11, block_dim);
        ROCSPARSE_CHECKARG(11, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        //
        // Quick return if possible
        //
        if(mb == 0 || nb == 0 || nnzb == 0)
        {
            // matrix never accessed however still need to update y vector
            rocsparse_int ysize
                = (trans == rocsparse_operation_none) ? block_dim * mb : block_dim * nb;
            if(ysize > 0)
            {
                if(y == nullptr && beta_device_host == nullptr)
                {
                    return rocsparse_status_invalid_pointer;
                }

                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::scale_array(handle, ysize, beta_device_host, y));
            }

            return rocsparse_status_success;
        }

        //
        // Check pointer arguments
        //
        ROCSPARSE_CHECKARG_POINTER(6, alpha_device_host);
        ROCSPARSE_CHECKARG_POINTER(14, beta_device_host);

        //
        // Another quick return.
        //
        if(handle->pointer_mode == rocsparse_pointer_mode_host
           && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
        {
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(8, int64_t(nnzb) * block_dim * block_dim, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(13, x);
        ROCSPARSE_CHECKARG_POINTER(15, y);

        rocsparse_bsrmv_info bsrmv_info = (info != nullptr) ? info->get_bsrmv_info() : nullptr;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrmv_template<T, I, J, A, X, Y>(handle,
                                                                               dir,
                                                                               trans,
                                                                               mb,
                                                                               nb,
                                                                               nnzb,
                                                                               alpha_device_host,
                                                                               descr,
                                                                               bsr_val,
                                                                               bsr_row_ptr,
                                                                               bsr_col_ind,
                                                                               block_dim,
                                                                               bsrmv_info,
                                                                               x,
                                                                               beta_device_host,
                                                                               y)));

        return rocsparse_status_success;
    }

}

#define INSTANTIATE(T, I, J)                                               \
    template rocsparse_status rocsparse::bsrmv_analysis_template<I, J, T>( \
        rocsparse_handle          handle,                                  \
        rocsparse_direction       dir,                                     \
        rocsparse_operation       trans,                                   \
        int64_t                   mb,                                      \
        int64_t                   nb,                                      \
        int64_t                   nnzb,                                    \
        const rocsparse_mat_descr descr,                                   \
        const void*               bsr_val,                                 \
        const void*               bsr_row_ptr,                             \
        const void*               bsr_col_ind,                             \
        int64_t                   block_dim,                               \
        rocsparse_bsrmv_info*     p_bsrmv_info);                               \
    template rocsparse_status rocsparse::bsrmv_template<T, I, J, T, T, T>( \
        rocsparse_handle          handle,                                  \
        rocsparse_direction       dir,                                     \
        rocsparse_operation       trans,                                   \
        int64_t                   mb,                                      \
        int64_t                   nb,                                      \
        int64_t                   nnzb,                                    \
        const void*               alpha_device_host,                       \
        const rocsparse_mat_descr descr,                                   \
        const void*               bsr_val,                                 \
        const void*               bsr_row_ptr,                             \
        const void*               bsr_col_ind,                             \
        int64_t                   block_dim,                               \
        rocsparse_bsrmv_info      bsrmv_info,                              \
        const void*               x,                                       \
        const void*               beta_device_host,                        \
        void*                     y)

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED_ANALYSIS(I, J, A)                                \
    template rocsparse_status rocsparse::bsrmv_analysis_template<I, J, A>( \
        rocsparse_handle          handle,                                  \
        rocsparse_direction       dir,                                     \
        rocsparse_operation       trans,                                   \
        int64_t                   mb,                                      \
        int64_t                   nb,                                      \
        int64_t                   nnzb,                                    \
        const rocsparse_mat_descr descr,                                   \
        const void*               bsr_val,                                 \
        const void*               bsr_row_ptr,                             \
        const void*               bsr_col_ind,                             \
        int64_t                   block_dim,                               \
        rocsparse_bsrmv_info*     p_bsrmv_info)

INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, int8_t);
INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, _Float16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, _Float16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, _Float16);
INSTANTIATE_MIXED_ANALYSIS(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_MIXED_ANALYSIS(int64_t, int64_t, rocsparse_bfloat16);
#undef INSTANTIATE_MIXED_ANALYSIS

#define INSTANTIATE_MIXED(T, I, J, A, X, Y)                                \
    template rocsparse_status rocsparse::bsrmv_template<T, I, J, A, X, Y>( \
        rocsparse_handle          handle,                                  \
        rocsparse_direction       dir,                                     \
        rocsparse_operation       trans,                                   \
        int64_t                   mb,                                      \
        int64_t                   nb,                                      \
        int64_t                   nnzb,                                    \
        const void*               alpha_device_host,                       \
        const rocsparse_mat_descr descr,                                   \
        const void*               bsr_val,                                 \
        const void*               bsr_row_ptr,                             \
        const void*               bsr_col_ind,                             \
        int64_t                   block_dim,                               \
        rocsparse_bsrmv_info      bsrmv_info,                              \
        const void*               x,                                       \
        const void*               beta_device_host,                        \
        void*                     y)

INSTANTIATE_MIXED(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int32_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int64_t,
                  int32_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_float_complex,
                  int64_t,
                  int64_t,
                  float,
                  rocsparse_float_complex,
                  rocsparse_float_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int32_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int64_t,
                  double,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

INSTANTIATE_MIXED(double, int32_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int64_t, float, double, double);

INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

#undef INSTANTIATE_MIXED

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

// rocsparse_xbsrmv_analysis
#define C_IMPL(NAME, T)                                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                    \
                                     rocsparse_direction       dir,                       \
                                     rocsparse_operation       trans,                     \
                                     rocsparse_int             mb,                        \
                                     rocsparse_int             nb,                        \
                                     rocsparse_int             nnzb,                      \
                                     const rocsparse_mat_descr descr,                     \
                                     const T*                  bsr_val,                   \
                                     const rocsparse_int*      bsr_row_ptr,               \
                                     const rocsparse_int*      bsr_col_ind,               \
                                     rocsparse_int             block_dim,                 \
                                     rocsparse_mat_info        info)                      \
    try                                                                                   \
    {                                                                                     \
        ROCSPARSE_ROUTINE_TRACE;                                                          \
        RETURN_IF_ROCSPARSE_ERROR(                                                        \
            (rocsparse::bsrmv_analysis_impl<rocsparse_int, rocsparse_int, T>(handle,      \
                                                                             dir,         \
                                                                             trans,       \
                                                                             mb,          \
                                                                             nb,          \
                                                                             nnzb,        \
                                                                             descr,       \
                                                                             bsr_val,     \
                                                                             bsr_row_ptr, \
                                                                             bsr_col_ind, \
                                                                             block_dim,   \
                                                                             info)));     \
        return rocsparse_status_success;                                                  \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        RETURN_ROCSPARSE_EXCEPTION();                                                     \
    }

C_IMPL(rocsparse_sbsrmv_analysis, float);
C_IMPL(rocsparse_dbsrmv_analysis, double);
C_IMPL(rocsparse_cbsrmv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv_analysis, rocsparse_double_complex);

#undef C_IMPL

// rocsparse_xbsrmv
#define C_IMPL(NAME, T)                                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                    \
                                     rocsparse_direction       dir,                       \
                                     rocsparse_operation       trans,                     \
                                     rocsparse_int             mb,                        \
                                     rocsparse_int             nb,                        \
                                     rocsparse_int             nnzb,                      \
                                     const T*                  alpha,                     \
                                     const rocsparse_mat_descr descr,                     \
                                     const T*                  bsr_val,                   \
                                     const rocsparse_int*      bsr_row_ptr,               \
                                     const rocsparse_int*      bsr_col_ind,               \
                                     rocsparse_int             block_dim,                 \
                                     rocsparse_mat_info        info,                      \
                                     const T*                  x,                         \
                                     const T*                  beta,                      \
                                     T*                        y)                         \
    try                                                                                   \
    {                                                                                     \
        ROCSPARSE_ROUTINE_TRACE;                                                          \
        RETURN_IF_ROCSPARSE_ERROR(                                                        \
            (rocsparse::bsrmv_impl<T, rocsparse_int, rocsparse_int, T, T, T>(handle,      \
                                                                             dir,         \
                                                                             trans,       \
                                                                             mb,          \
                                                                             nb,          \
                                                                             nnzb,        \
                                                                             alpha,       \
                                                                             descr,       \
                                                                             bsr_val,     \
                                                                             bsr_row_ptr, \
                                                                             bsr_col_ind, \
                                                                             block_dim,   \
                                                                             info,        \
                                                                             x,           \
                                                                             beta,        \
                                                                             y)));        \
        return rocsparse_status_success;                                                  \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        RETURN_ROCSPARSE_EXCEPTION();                                                     \
    }

C_IMPL(rocsparse_sbsrmv, float);
C_IMPL(rocsparse_dbsrmv, double);
C_IMPL(rocsparse_cbsrmv, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv, rocsparse_double_complex);

#undef C_IMPL

extern "C" rocsparse_status rocsparse_bsrmv_clear(rocsparse_handle handle, rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
