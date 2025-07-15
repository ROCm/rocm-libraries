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

#include "rocsparse_bsrxmv_spzl.hpp"

namespace rocsparse
{
    // BSRXMV kernel for BSR block dimension of 17 to 32
    template <uint32_t BSRDIM,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_DEVICE_ILF void bsrxmvn_17_32_device(J                   mb,
                                                   rocsparse_direction dir,
                                                   T                   alpha,
                                                   J                   size_of_mask,
                                                   const J* __restrict__ bsr_mask_ptr,
                                                   const I* __restrict__ bsr_row_ptr,
                                                   const I* __restrict__ bsr_end_ptr,
                                                   const J* __restrict__ bsr_col_ind,
                                                   const A* __restrict__ bsr_val,
                                                   const X* __restrict__ x,
                                                   T beta,
                                                   Y* __restrict__ y,
                                                   rocsparse_index_base idx_base)
    {
        static constexpr int SQBSRDIM = BSRDIM * BSRDIM;

        // BSR block lane id
        J lid = hipThreadIdx_x % BSRDIM;

        // Offset into x vector
        J idx = (dir == rocsparse_direction_column) ? ((hipThreadIdx_x / BSRDIM) % BSRDIM) : lid;

        // Each thread block processes a single BSR row
        J row = hipBlockIdx_x;

        if(bsr_mask_ptr != nullptr)
        {
            row = bsr_mask_ptr[row] - idx_base;
        }

        // BSR row entry and exit point
        I row_begin = bsr_row_ptr[row] - idx_base;
        I row_end   = (bsr_end_ptr == nullptr) ? (bsr_row_ptr[row + 1] - idx_base)
                                               : (bsr_end_ptr[row] - idx_base);

        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block value
        bsr_val += size_t(row_begin) * SQBSRDIM + hipThreadIdx_x;
        I j = row_begin;

        for(; j < row_end; ++j)
        {
            I k = j + hipThreadIdx_x / SQBSRDIM;

            // Do not exceed the row
            if(k < row_end)
            {
                // Column index into x vector
                J col = (bsr_col_ind[k] - idx_base) * BSRDIM;

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                sum = rocsparse::fma<T>(*bsr_val, x[col + idx], sum);
                bsr_val += SQBSRDIM;
            }
        }

        // Accumulate each row sum of the BSR block
        __shared__ T sdata[SQBSRDIM];

        sdata[hipThreadIdx_x] = sum;

        __syncthreads();

        if(dir == rocsparse_direction_column)
        {
            if(hipThreadIdx_x < SQBSRDIM - BSRDIM * 16)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 16];
            __syncthreads();
            if(hipThreadIdx_x < BSRDIM * 8)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
            __syncthreads();
            if(hipThreadIdx_x < BSRDIM * 4)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
            __syncthreads();
            if(hipThreadIdx_x < BSRDIM * 2)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
            __threadfence_block();
            if(hipThreadIdx_x < BSRDIM * 1)
                sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
        }
        else
        {
            // Reduce the intra block row sum
            if(lid < BSRDIM - 16)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 16];
            __syncthreads();
            if(lid < 8)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 8];
            __syncthreads();
            if(lid < 4)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
            __syncthreads();
            if(lid < 2)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
            __syncthreads();

            // Final reduction
            if(hipThreadIdx_x < BSRDIM)
                sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
        }

        // First bunch of threads write row sums to global memory
        if(hipThreadIdx_x < BSRDIM)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * BSRDIM + hipThreadIdx_x]
                    = rocsparse::fma<T>(beta, y[row * BSRDIM + hipThreadIdx_x], alpha * sum);
            }
            else
            {
                y[row * BSRDIM + hipThreadIdx_x] = alpha * sum;
            }
        }
    }

    // Kernels for BSR block dimensions of 17 to 32
    template <uint32_t BSRDIM,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_KERNEL(BSRDIM* BSRDIM)
    void bsrxmvn_17_32_kernel(J                   mb,
                              rocsparse_direction dir,
                              ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                              J size_of_mask,
                              const J* __restrict__ bsr_mask_ptr,
                              const I* __restrict__ bsr_row_ptr,
                              const I* __restrict__ bsr_end_ptr,
                              const J* __restrict__ bsr_col_ind,
                              const A* __restrict__ bsr_val,
                              const X* __restrict__ x,
                              ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                              Y* __restrict__ y,
                              rocsparse_index_base idx_base,
                              bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::bsrxmvn_17_32_device<BSRDIM>(mb,
                                                    dir,
                                                    alpha,
                                                    size_of_mask,
                                                    bsr_mask_ptr,
                                                    bsr_row_ptr,
                                                    bsr_end_ptr,
                                                    bsr_col_ind,
                                                    bsr_val,
                                                    x,
                                                    beta,
                                                    y,
                                                    idx_base);
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void rocsparse::bsrxmvn_17_32(rocsparse_handle     handle,
                              rocsparse_direction  dir,
                              J                    mb,
                              I                    nnzb,
                              const T*             alpha_device_host,
                              J                    size_of_mask,
                              const J*             bsr_mask_ptr,
                              const I*             bsr_row_ptr,
                              const I*             bsr_end_ptr,
                              const J*             bsr_col_ind,
                              const A*             bsr_val,
                              J                    block_dim,
                              const X*             x,
                              const T*             beta_device_host,
                              Y*                   y,
                              rocsparse_index_base base)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J size = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;
    if(block_dim == 17)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<17>),
            dim3(size),
            dim3(17 * 17),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 18)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<18>),
            dim3(size),
            dim3(18 * 18),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 19)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<19>),
            dim3(size),
            dim3(19 * 19),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 20)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<20>),
            dim3(size),
            dim3(20 * 20),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 21)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<21>),
            dim3(size),
            dim3(21 * 21),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 22)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<22>),
            dim3(size),
            dim3(22 * 22),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 23)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<23>),
            dim3(size),
            dim3(23 * 23),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 24)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<24>),
            dim3(size),
            dim3(24 * 24),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 25)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<25>),
            dim3(size),
            dim3(25 * 25),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 26)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<26>),
            dim3(size),
            dim3(26 * 26),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 27)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<27>),
            dim3(size),
            dim3(27 * 27),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 28)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<28>),
            dim3(size),
            dim3(28 * 28),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 29)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<29>),
            dim3(size),
            dim3(29 * 29),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 30)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<30>),
            dim3(size),
            dim3(30 * 30),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 31)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<31>),
            dim3(size),
            dim3(31 * 31),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim == 32)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_17_32_kernel<32>),
            dim3(size),
            dim3(32 * 32),
            0,
            handle->stream,
            mb,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
}

//
// INSTANTIATE.
//
#define INSTANTIATE(T, I, J)                                                       \
    template void rocsparse::bsrxmvn_17_32(rocsparse_handle     handle,            \
                                           rocsparse_direction  dir,               \
                                           J                    mb,                \
                                           I                    nnzb,              \
                                           const T*             alpha_device_host, \
                                           J                    size_of_mask,      \
                                           const J*             bsr_mask_ptr,      \
                                           const I*             bsr_row_ptr,       \
                                           const I*             bsr_end_ptr,       \
                                           const J*             bsr_col_ind,       \
                                           const T*             bsr_val,           \
                                           J                    block_dim,         \
                                           const T*             x,                 \
                                           const T*             beta_device_host,  \
                                           T*                   y,                 \
                                           rocsparse_index_base base)

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);

INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);

INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE_MIXED(T, I, J, A, X, Y)                                        \
    template void rocsparse::bsrxmvn_17_32(rocsparse_handle     handle,            \
                                           rocsparse_direction  dir,               \
                                           J                    mb,                \
                                           I                    nnzb,              \
                                           const T*             alpha_device_host, \
                                           J                    size_of_mask,      \
                                           const J*             bsr_mask_ptr,      \
                                           const I*             bsr_row_ptr,       \
                                           const I*             bsr_end_ptr,       \
                                           const J*             bsr_col_ind,       \
                                           const A*             bsr_val,           \
                                           J                    block_dim,         \
                                           const X*             x,                 \
                                           const T*             beta_device_host,  \
                                           Y*                   y,                 \
                                           rocsparse_index_base base)

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
