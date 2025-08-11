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
    // BSRXMV kernel for BSR block dimension of 2
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_DEVICE_ILF void bsrxmvn_2x2_device(J                   mb,
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
        // BSR block dimension
        static constexpr int BSRDIM = 2;

        // Lane id
        const J lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        const J wid = hipThreadIdx_x / WFSIZE;

        // Each thread block processes (BLOCKSIZE / WFSIZE) BSR rows
        J row = hipBlockIdx_x * (BLOCKSIZE / WFSIZE) + wid;

        // Do not run out of bounds
        if(bsr_mask_ptr == nullptr)
        {
            if(row >= mb)
            {
                return;
            }
        }
        else
        {
            if(row >= size_of_mask)
            {
                return;
            }
            row = bsr_mask_ptr[row] - idx_base;
        }

        // BSR row entry and exit point
        const I row_begin = bsr_row_ptr[row] - idx_base;
        const I row_end   = (bsr_end_ptr == nullptr) ? (bsr_row_ptr[row + 1] - idx_base)
                                                     : (bsr_end_ptr[row] - idx_base);

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);

        static constexpr uint32_t VALOFFSET    = BSRDIM * BSRDIM * WFSIZE;
        static constexpr size_t   longSQBSRDIM = BSRDIM * BSRDIM;

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block
        {
            I        j    = row_begin + lid;
            const A* bval = bsr_val + longSQBSRDIM * j;
            if(dir == rocsparse_direction_column)
            {
                for(; j < row_end; j += WFSIZE)
                {
                    // Column index into x vector
                    const J col = (bsr_col_ind[j] - idx_base) * BSRDIM;

                    // Compute the sum of the two rows within the BSR blocks of the current
                    // BSR row
                    sum0 = rocsparse::fma<T>(bval[0], x[col + 0], sum0);
                    sum1 = rocsparse::fma<T>(bval[1], x[col + 0], sum1);

                    sum0 = rocsparse::fma<T>(bval[2], x[col + 1], sum0);
                    sum1 = rocsparse::fma<T>(bval[3], x[col + 1], sum1);

                    bval += VALOFFSET;
                }
            }
            else
            {
                for(; j < row_end; j += WFSIZE)
                {
                    // Do not exceed the row
                    // Column index into x vector
                    const J col = (bsr_col_ind[j] - idx_base) * BSRDIM;

                    // Compute the sum of the two rows within the BSR blocks of the current
                    // BSR row
                    sum0 = rocsparse::fma<T>(bval[0], x[col + 0], sum0);
                    sum0 = rocsparse::fma<T>(bval[1], x[col + 1], sum0);

                    sum1 = rocsparse::fma<T>(bval[2], x[col + 0], sum1);
                    sum1 = rocsparse::fma<T>(bval[3], x[col + 1], sum1);
                    bval += VALOFFSET;
                }
            }
        }

        // Each wavefront accumulates its BSR block row sum
        sum0 = rocsparse::wfreduce_sum<WFSIZE>(sum0);
        sum1 = rocsparse::wfreduce_sum<WFSIZE>(sum1);

        // Last lane of each wavefront writes the two row sums to global memory
        if(lid == WFSIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                y[row * BSRDIM + 0] = rocsparse::fma<T>(beta, y[row * BSRDIM + 0], alpha * sum0);
                y[row * BSRDIM + 1] = rocsparse::fma<T>(beta, y[row * BSRDIM + 1], alpha * sum1);
            }
            else
            {
                y[row * BSRDIM + 0] = alpha * sum0;
                y[row * BSRDIM + 1] = alpha * sum1;
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrxmvn_2x2_kernel(J                   mb,
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
            rocsparse::bsrxmvn_2x2_device<BLOCKSIZE, WFSIZE>(mb,
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
void rocsparse::bsrxmvn_2x2(rocsparse_handle     handle,
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
                            const X*             x,
                            const T*             beta_device_host,
                            Y*                   y,
                            rocsparse_index_base base)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J blocks_per_row = nnzb / mb;
    const J size           = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;

#define BSRXMVN_DIM 128
    if(blocks_per_row < 8)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_2x2_kernel<BSRXMVN_DIM, 4, T>),
            dim3((size - 1) / (BSRXMVN_DIM / 4) + 1),
            dim3(BSRXMVN_DIM),
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
    else if(blocks_per_row < 16)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_2x2_kernel<BSRXMVN_DIM, 8, T>),
            dim3((size - 1) / (BSRXMVN_DIM / 8) + 1),
            dim3(BSRXMVN_DIM),
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
    else if(blocks_per_row < 32)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_2x2_kernel<BSRXMVN_DIM, 16, T>),
            dim3((size - 1) / (BSRXMVN_DIM / 16) + 1),
            dim3(BSRXMVN_DIM),
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
    else if(blocks_per_row < 64)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_2x2_kernel<BSRXMVN_DIM, 32, T>),
            dim3((size - 1) / (BSRXMVN_DIM / 32) + 1),
            dim3(BSRXMVN_DIM),
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
    else
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_2x2_kernel<BSRXMVN_DIM, 64, T>),
            dim3((size - 1) / (BSRXMVN_DIM / 64) + 1),
            dim3(BSRXMVN_DIM),
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

#undef BSRXMVN_DIM
}

//
// INSTANTIATE.
//
#define INSTANTIATE(T, I, J)                                                     \
    template void rocsparse::bsrxmvn_2x2(rocsparse_handle     handle,            \
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

#define INSTANTIATE_MIXED(T, I, J, A, X, Y)                                      \
    template void rocsparse::bsrxmvn_2x2(rocsparse_handle     handle,            \
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
