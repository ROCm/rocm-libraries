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

#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_coomv_aos.hpp"
#include "rocsparse_utility.hpp"

#include "coomv_device.h"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_coomv_aos_alg value_)
{
    switch(value_)
    {
    case rocsparse_coomv_aos_alg_default:
    case rocsparse_coomv_aos_alg_segmented:
    case rocsparse_coomv_aos_alg_atomic:
    {
        return false;
    }
    }
    return true;
};

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coomvn_aos_segmented_loops(int64_t nnz,
                                    I       nloops,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                    const I* __restrict__ coo_ind,
                                    const A* __restrict__ coo_val,
                                    const X* __restrict__ x,
                                    Y* __restrict__ y,
                                    I* __restrict__ row_block_red,
                                    T* __restrict__ val_block_red,
                                    rocsparse_index_base idx_base,
                                    bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::coomvn_aos_segmented_loops_device<BLOCKSIZE>(
                nnz, nloops, alpha, coo_ind, coo_val, x, y, row_block_red, val_block_red, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename Y, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coomvn_segmented_loops_reduce(I nblocks,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                       const I* __restrict__ row_block_red,
                                       const T* __restrict__ val_block_red,
                                       Y* __restrict__ y,
                                       bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::coomvn_segmented_loops_reduce_device<BLOCKSIZE>(
                nblocks, row_block_red, val_block_red, y);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t LOOPS,
              typename I,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coomvn_aos_atomic_loops(int64_t nnz,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 const I* __restrict__ coo_ind,
                                 const A* __restrict__ coo_val,
                                 const X* __restrict__ x,
                                 Y* __restrict__ y,
                                 rocsparse_index_base idx_base,
                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::coomvn_aos_atomic_loops_device<BLOCKSIZE, LOOPS>(
                nnz, alpha, coo_ind, coo_val, x, y, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void coomvt_aos_kernel(rocsparse_operation trans,
                           int64_t             nnz,
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                           const I* __restrict__ coo_ind,
                           const A* __restrict__ coo_val,
                           const X* __restrict__ x,
                           Y* __restrict__ y,
                           rocsparse_index_base idx_base,
                           bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::coomvt_aos_device(trans, nnz, alpha, coo_ind, coo_val, x, y, idx_base);
        }
    }

    template <typename I, typename A, typename X, typename Y, typename T>
    rocsparse_status coomv_aos_atomic_dispatch(rocsparse_handle          handle,
                                               rocsparse_operation       trans,
                                               I                         m,
                                               I                         n,
                                               int64_t                   nnz,
                                               const T*                  alpha_device_host,
                                               const rocsparse_mat_descr descr,
                                               const A*                  coo_val,
                                               const I*                  coo_ind,
                                               const X*                  x,
                                               const T*                  beta_device_host,
                                               Y*                        y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        const I ysize = (trans == rocsparse_operation_none) ? m : n;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));

        // Run different coomv kernels
        switch(trans)
        {
        case rocsparse_operation_none:
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::coomvn_aos_atomic_loops<256, 1>),
                dim3((nnz - 1) / 256 + 1),
                dim3(256),
                0,
                stream,
                nnz,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                coo_ind,
                coo_val,
                x,
                y,
                descr->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::coomvt_aos_kernel<1024>),
                dim3((nnz - 1) / 1024 + 1),
                dim3(1024),
                0,
                handle->stream,
                trans,
                nnz,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                coo_ind,
                coo_val,
                x,
                y,
                descr->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
            break;
        }
        }

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename A, typename X, typename Y>
    rocsparse_status coomv_aos_segmented_dispatch(rocsparse_handle          handle,
                                                  rocsparse_operation       trans,
                                                  I                         m,
                                                  I                         n,
                                                  int64_t                   nnz,
                                                  const T*                  alpha_device_host,
                                                  const rocsparse_mat_descr descr,
                                                  const A*                  coo_val,
                                                  const I*                  coo_ind,
                                                  const X*                  x,
                                                  const T*                  beta_device_host,
                                                  Y*                        y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        const I ysize = (trans == rocsparse_operation_none) ? m : n;

        // Scale y with beta
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));

        // Run different coomv kernels
        switch(trans)
        {
        case rocsparse_operation_none:
        {
#define COOMVN_DIM 256
            int maxthreads = handle->properties.maxThreadsPerBlock;
            int nprocs     = 2 * handle->properties.multiProcessorCount;
            int maxblocks  = (nprocs * maxthreads - 1) / COOMVN_DIM + 1;

            I minblocks = (nnz - 1) / COOMVN_DIM + 1;
            I nblocks   = maxblocks < minblocks ? maxblocks : minblocks;
            I nloops    = (nnz - 1) / (COOMVN_DIM * nblocks) + 1;

            // Buffer
            char* ptr = reinterpret_cast<char*>(handle->buffer);
            ptr += 256;

            // row block reduction buffer
            I* row_block_red = reinterpret_cast<I*>(ptr);
            ptr += ((sizeof(I) * nblocks - 1) / 256 + 1) * 256;

            // val block reduction buffer
            T* val_block_red = reinterpret_cast<T*>(ptr);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::coomvn_aos_segmented_loops<COOMVN_DIM>),
                dim3(nblocks),
                dim3(COOMVN_DIM),
                0,
                stream,
                nnz,
                nloops,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                coo_ind,
                coo_val,
                x,
                y,
                row_block_red,
                val_block_red,
                descr->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::coomvn_segmented_loops_reduce<COOMVN_DIM>),
                dim3(1),
                dim3(COOMVN_DIM),
                0,
                stream,
                nblocks,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                row_block_red,
                val_block_red,
                y,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef COOMVN_DIM
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::coomvt_aos_kernel<1024>),
                dim3((nnz - 1) / 1024 + 1),
                dim3(1024),
                0,
                handle->stream,
                trans,
                nnz,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                coo_ind,
                coo_val,
                x,
                y,
                descr->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
            break;
        }
        }

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename A, typename X, typename Y>
    rocsparse_status coomv_aos_dispatch(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        rocsparse_coomv_aos_alg   alg,
                                        I                         m,
                                        I                         n,
                                        int64_t                   nnz,
                                        const T*                  alpha_device_host,
                                        const rocsparse_mat_descr descr,
                                        const A*                  coo_val,
                                        const I*                  coo_ind,
                                        const X*                  x,
                                        const T*                  beta_device_host,
                                        Y*                        y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(alg)
        {
        case rocsparse_coomv_aos_alg_default:
        case rocsparse_coomv_aos_alg_atomic:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_aos_atomic_dispatch(handle,
                                                                           trans,
                                                                           m,
                                                                           n,
                                                                           nnz,
                                                                           alpha_device_host,
                                                                           descr,
                                                                           coo_val,
                                                                           coo_ind,
                                                                           x,
                                                                           beta_device_host,
                                                                           y));
            return rocsparse_status_success;
        }

        case rocsparse_coomv_aos_alg_segmented:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_aos_segmented_dispatch(handle,
                                                                              trans,
                                                                              m,
                                                                              n,
                                                                              nnz,
                                                                              alpha_device_host,
                                                                              descr,
                                                                              coo_val,
                                                                              coo_ind,
                                                                              x,
                                                                              beta_device_host,
                                                                              y));
            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

template <typename T, typename I, typename A, typename X, typename Y>
rocsparse_status rocsparse::coomv_aos_template(rocsparse_handle          handle,
                                               rocsparse_operation       trans,
                                               rocsparse_coomv_aos_alg   alg,
                                               int64_t                   m_,
                                               int64_t                   n_,
                                               int64_t                   nnz,
                                               const void*               alpha_device_host_,
                                               const rocsparse_mat_descr descr,
                                               const void*               coo_val_,
                                               const void*               coo_ind_,
                                               const void*               x_,
                                               const void*               beta_device_host_,
                                               void*                     y_,
                                               bool                      fallback_algorithm)
{
    ROCSPARSE_ROUTINE_TRACE;

    const I  m                 = static_cast<I>(m_);
    const I  n                 = static_cast<I>(n_);
    const T* alpha_device_host = reinterpret_cast<const T*>(alpha_device_host_);
    const A* coo_val           = reinterpret_cast<const A*>(coo_val_);
    const I* coo_ind           = reinterpret_cast<const I*>(coo_ind_);
    const X* x                 = reinterpret_cast<const X*>(x_);
    const T* beta_device_host  = reinterpret_cast<const T*>(beta_device_host_);
    Y*       y                 = reinterpret_cast<Y*>(y_);

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(7, descr);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcoomv_aos"),
                         trans,
                         m,
                         n,
                         nnz,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)coo_val,
                         (const void*&)coo_ind,
                         (const void*&)x,
                         LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                         (const void*&)y);

    // Check index base
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_ENUM(2, alg);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        7, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(3, m);
    ROCSPARSE_CHECKARG_SIZE(4, n);
    ROCSPARSE_CHECKARG_SIZE(5, nnz);

    const rocsparse_int xsize = (trans == rocsparse_operation_none) ? n : m;
    const rocsparse_int ysize = (trans == rocsparse_operation_none) ? m : n;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        // matrix never accessed however still need to update y vector
        if(ysize > 0)
        {
            ROCSPARSE_CHECKARG_ARRAY(12, ysize, y);
            ROCSPARSE_CHECKARG_POINTER(11, beta_device_host);

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(6, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(11, beta_device_host);

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    // Check the rest of the pointer arguments
    ROCSPARSE_CHECKARG_ARRAY(8, nnz, coo_val);
    ROCSPARSE_CHECKARG_ARRAY(9, nnz, coo_ind);
    ROCSPARSE_CHECKARG_ARRAY(10, xsize, x);
    ROCSPARSE_CHECKARG_ARRAY(12, ysize, y);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_aos_dispatch(handle,
                                                            trans,
                                                            alg,
                                                            m,
                                                            n,
                                                            nnz,
                                                            alpha_device_host,
                                                            descr,
                                                            coo_val,
                                                            coo_ind,
                                                            x,
                                                            beta_device_host,
                                                            y));
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I)                                                   \
    template rocsparse_status rocsparse::coomv_aos_template<T, I, T, T, T>( \
        rocsparse_handle,                                                   \
        rocsparse_operation,                                                \
        rocsparse_coomv_aos_alg,                                            \
        int64_t,                                                            \
        int64_t,                                                            \
        int64_t,                                                            \
        const void*,                                                        \
        const rocsparse_mat_descr,                                          \
        const void*,                                                        \
        const void*,                                                        \
        const void*,                                                        \
        const void*,                                                        \
        void*,                                                              \
        bool)

INSTANTIATE(float, int32_t);
INSTANTIATE(float, int64_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(double, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED(T, I, A, X, Y)                                    \
    template rocsparse_status rocsparse::coomv_aos_template<T, I, A, X, Y>( \
        rocsparse_handle,                                                   \
        rocsparse_operation,                                                \
        rocsparse_coomv_aos_alg,                                            \
        int64_t,                                                            \
        int64_t,                                                            \
        int64_t,                                                            \
        const void*,                                                        \
        const rocsparse_mat_descr,                                          \
        const void*,                                                        \
        const void*,                                                        \
        const void*,                                                        \
        const void*,                                                        \
        void*,                                                              \
        bool)

INSTANTIATE_MIXED(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(float, int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int64_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(float, int32_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int64_t, _Float16, _Float16, float);
INSTANTIATE_MIXED(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE_MIXED(
    rocsparse_float_complex, int32_t, float, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_MIXED(
    rocsparse_float_complex, int64_t, float, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE_MIXED(
    rocsparse_double_complex, int32_t, double, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE_MIXED(
    rocsparse_double_complex, int64_t, double, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE_MIXED(double, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, float, double, double);

INSTANTIATE_MIXED(rocsparse_double_complex,
                  int32_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);
INSTANTIATE_MIXED(rocsparse_double_complex,
                  int64_t,
                  rocsparse_float_complex,
                  rocsparse_double_complex,
                  rocsparse_double_complex);

#undef INSTANTIATE_MIXED
