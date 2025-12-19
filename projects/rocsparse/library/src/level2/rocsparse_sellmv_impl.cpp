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

#include "rocsparse_common.h"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_sellmv.hpp"

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"
#include "sellmv_device.h"

namespace rocsparse
{
    template <uint32_t THREADS_PER_ROW,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(128 * THREADS_PER_ROW)
    void sellmvn_kernel(J m,
                        J n,
                        I nnz,
                        J sell_slice_size,
                        I sell_colval_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                        const I* __restrict__ sell_slice_offsets,
                        const J* __restrict__ sell_col_ind,
                        const A* __restrict__ sell_val,
                        const X* __restrict__ x,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                        Y* __restrict__ y,
                        rocsparse_index_base idx_base,
                        bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::sellmvn_device<THREADS_PER_ROW>(m,
                                                       n,
                                                       nnz,
                                                       sell_slice_size,
                                                       sell_colval_size,
                                                       alpha,
                                                       sell_slice_offsets,
                                                       sell_col_ind,
                                                       sell_val,
                                                       x,
                                                       beta,
                                                       y,
                                                       idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sellmvn_large_slice_kernel(J m,
                                    J n,
                                    I nnz,
                                    J sell_slice_size,
                                    I sell_colval_size,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                    const I* __restrict__ sell_slice_offsets,
                                    const J* __restrict__ sell_col_ind,
                                    const A* __restrict__ sell_val,
                                    const X* __restrict__ x,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                    Y* __restrict__ y,
                                    rocsparse_index_base idx_base,
                                    bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::sellmvn_large_slice_device<BLOCKSIZE>(m,
                                                             n,
                                                             nnz,
                                                             sell_slice_size,
                                                             sell_colval_size,
                                                             alpha,
                                                             sell_slice_offsets,
                                                             sell_col_ind,
                                                             sell_val,
                                                             x,
                                                             beta,
                                                             y,
                                                             idx_base);
        }
    }

    template <uint32_t THREADS_PER_ROW,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(128 * THREADS_PER_ROW)
    void sellmvt_kernel(rocsparse_operation trans,
                        J                   m,
                        J                   n,
                        I                   nnz,
                        J                   sell_slice_size,
                        I                   sell_colval_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                        const I* __restrict__ sell_slice_offsets,
                        const J* __restrict__ sell_col_ind,
                        const A* __restrict__ sell_val,
                        const X* __restrict__ x,
                        Y* __restrict__ y,
                        rocsparse_index_base idx_base,
                        bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::sellmvt_device<THREADS_PER_ROW>(trans,
                                                       m,
                                                       n,
                                                       nnz,
                                                       sell_slice_size,
                                                       sell_colval_size,
                                                       alpha,
                                                       sell_slice_offsets,
                                                       sell_col_ind,
                                                       sell_val,
                                                       x,
                                                       y,
                                                       idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sellmvt_large_slice_kernel(rocsparse_operation trans,
                                    J                   m,
                                    J                   n,
                                    I                   nnz,
                                    J                   sell_slice_size,
                                    I                   sell_colval_size,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                    const I* __restrict__ sell_slice_offsets,
                                    const J* __restrict__ sell_col_ind,
                                    const A* __restrict__ sell_val,
                                    const X* __restrict__ x,
                                    Y* __restrict__ y,
                                    rocsparse_index_base idx_base,
                                    bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::sellmvt_large_slice_device<BLOCKSIZE>(trans,
                                                             m,
                                                             n,
                                                             nnz,
                                                             sell_slice_size,
                                                             sell_colval_size,
                                                             alpha,
                                                             sell_slice_offsets,
                                                             sell_col_ind,
                                                             sell_val,
                                                             x,
                                                             y,
                                                             idx_base);
        }
    }

    template <typename I, typename J, typename A, typename X, typename Y, typename T>
    rocsparse_status sellmv_dispatch(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     J                         m,
                                     J                         n,
                                     I                         nnz,
                                     J                         sell_slice_size,
                                     I                         sell_colval_size,
                                     const T*                  alpha_device_host,
                                     const rocsparse_mat_descr descr,
                                     const A*                  sell_val,
                                     const I*                  sell_slice_offsets,
                                     const J*                  sell_col_ind,
                                     const X*                  x,
                                     const T*                  beta_device_host,
                                     Y*                        y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        const I nslices = (m - 1) / sell_slice_size + 1;

        const int32_t blocks_x = std::sqrt(nslices);
        const int32_t blocks_y = (nslices - 1) / blocks_x + 1;

        if(trans == rocsparse_operation_none)
        {
            if(sell_slice_size <= 128)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::sellmvn_kernel<8>),
                    dim3(blocks_x, blocks_y, 1),
                    dim3(sell_slice_size, 8),
                    sell_slice_size * 8 * sizeof(T),
                    stream,
                    m,
                    n,
                    nnz,
                    sell_slice_size,
                    sell_colval_size,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    sell_slice_offsets,
                    sell_col_ind,
                    sell_val,
                    x,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    y,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::sellmvn_large_slice_kernel<256>),
                    dim3(nslices),
                    dim3(256),
                    0,
                    stream,
                    m,
                    n,
                    nnz,
                    sell_slice_size,
                    sell_colval_size,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    sell_slice_offsets,
                    sell_col_ind,
                    sell_val,
                    x,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    y,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
        }
        else
        {
            // Scale y with beta
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, n, beta_device_host, y));

            if(sell_slice_size <= 128)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::sellmvt_kernel<8>),
                    dim3(blocks_x, blocks_y, 1),
                    dim3(sell_slice_size, 8),
                    0,
                    stream,
                    trans,
                    m,
                    n,
                    nnz,
                    sell_slice_size,
                    sell_colval_size,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    sell_slice_offsets,
                    sell_col_ind,
                    sell_val,
                    x,
                    y,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::sellmvt_large_slice_kernel<256>),
                    dim3(nslices),
                    dim3(256),
                    0,
                    stream,
                    trans,
                    m,
                    n,
                    nnz,
                    sell_slice_size,
                    sell_colval_size,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    sell_slice_offsets,
                    sell_col_ind,
                    sell_val,
                    x,
                    y,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
        }

        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::sellmv_template(rocsparse_handle          handle, // 0
                                            rocsparse_operation       trans, //1
                                            int64_t                   m_, //2
                                            int64_t                   n_, //3
                                            int64_t                   nnz_, //4
                                            int64_t                   sell_slice_size_, //5
                                            int64_t                   sell_colval_size_, //6
                                            const void*               alpha_device_host_, //7
                                            const rocsparse_mat_descr descr, //8
                                            const void*               sell_val_, //9
                                            const void*               sell_slice_offsets_, //10
                                            const void*               sell_col_ind_, //11
                                            const void*               x_, //12
                                            const void*               beta_device_host_, //13
                                            void*                     y_) //14
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m                  = static_cast<J>(m_);
    const J  n                  = static_cast<J>(n_);
    const I  nnz                = static_cast<I>(nnz_);
    const J  sell_slice_size    = static_cast<J>(sell_slice_size_);
    const I  sell_colval_size   = static_cast<I>(sell_colval_size_);
    const T* alpha_device_host  = reinterpret_cast<const T*>(alpha_device_host_);
    const A* sell_val           = reinterpret_cast<const A*>(sell_val_);
    const I* sell_slice_offsets = reinterpret_cast<const I*>(sell_slice_offsets_);
    const J* sell_col_ind       = reinterpret_cast<const J*>(sell_col_ind_);
    const X* x                  = reinterpret_cast<const X*>(x_);
    const T* beta_device_host   = reinterpret_cast<const T*>(beta_device_host_);
    Y*       y                  = reinterpret_cast<Y*>(y_);

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(8, descr);

    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        8, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(8,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);
    ROCSPARSE_CHECKARG_SIZE(5, sell_slice_size);
    ROCSPARSE_CHECKARG_SIZE(6, sell_colval_size);

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;
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

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(7, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(13, beta_device_host);

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0))
    {
        if(*beta_device_host != static_cast<T>(1))
        {
            rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;
            if(ysize > 0)
            {
                if(y == nullptr)
                {
                    return rocsparse_status_invalid_pointer;
                }

                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::scale_array(handle, ysize, beta_device_host, y));
            }
        }
        return rocsparse_status_success;
    }

    // Check the rest of the pointer arguments
    ROCSPARSE_CHECKARG_POINTER(9, sell_val);
    ROCSPARSE_CHECKARG_POINTER(10, sell_slice_offsets);
    ROCSPARSE_CHECKARG_POINTER(11, sell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(12, x);
    ROCSPARSE_CHECKARG_POINTER(14, y);

    if(sell_slice_size == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_template<T, I, J, A, X, Y>(handle,
                                                         trans,
                                                         rocsparse::csrmv_alg_rowsplit,
                                                         m,
                                                         n,
                                                         nnz,
                                                         alpha_device_host,
                                                         descr,
                                                         sell_val,
                                                         sell_slice_offsets,
                                                         sell_slice_offsets + 1,
                                                         sell_col_ind,
                                                         nullptr,
                                                         x,
                                                         beta_device_host,
                                                         y,
                                                         false,
                                                         false)));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sellmv_dispatch(handle,
                                                         trans,
                                                         m,
                                                         n,
                                                         nnz,
                                                         sell_slice_size,
                                                         sell_colval_size,
                                                         alpha_device_host,
                                                         descr,
                                                         sell_val,
                                                         sell_slice_offsets,
                                                         sell_col_ind,
                                                         x,
                                                         beta_device_host,
                                                         y));
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, J)                                                \
    template rocsparse_status rocsparse::sellmv_template<T, I, J, T, T, T>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans,                                    \
        int64_t                   m,                                        \
        int64_t                   n,                                        \
        int64_t                   nnz,                                      \
        int64_t                   sell_slice_size,                          \
        int64_t                   sell_colval_size,                         \
        const void*               alpha,                                    \
        const rocsparse_mat_descr descr,                                    \
        const void*               sell_val,                                 \
        const void*               sell_slice_offsets,                       \
        const void*               sell_col_ind,                             \
        const void*               x,                                        \
        const void*               beta,                                     \
        void*                     y);

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

#define INSTANTIATE_MIXED(T, I, J, A, X, Y)                                 \
    template rocsparse_status rocsparse::sellmv_template<T, I, J, A, X, Y>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans,                                    \
        int64_t                   m,                                        \
        int64_t                   n,                                        \
        int64_t                   nnz,                                      \
        int64_t                   sell_slice_size,                          \
        int64_t                   sell_colval_size,                         \
        const void*               alpha,                                    \
        const rocsparse_mat_descr descr,                                    \
        const void*               sell_val,                                 \
        const void*               sell_slice_offsets,                       \
        const void*               sell_col_ind,                             \
        const void*               x,                                        \
        const void*               beta,                                     \
        void*                     y);

INSTANTIATE_MIXED(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int32_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE_MIXED(float, int64_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE_MIXED(float, int64_t, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE_MIXED(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(
    float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE_MIXED(
    float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE_MIXED(
    float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
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
