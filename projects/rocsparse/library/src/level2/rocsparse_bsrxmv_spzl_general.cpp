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
    // General BSRXMV that works for any BSR block dimensions
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_DEVICE_ILF void bsrxmvn_general_device(rocsparse_direction dir,
                                                     T                   alpha,
                                                     J                   size_of_mask,
                                                     const J* __restrict__ bsr_mask_ptr,
                                                     const I* __restrict__ bsr_row_ptr,
                                                     const I* __restrict__ bsr_end_ptr,
                                                     const J* __restrict__ bsr_col_ind,
                                                     const A* __restrict__ bsr_val,
                                                     J block_dim,
                                                     const X* __restrict__ x,
                                                     T beta,
                                                     Y* __restrict__ y,
                                                     rocsparse_index_base idx_base)
    {
        // Lane id
        J lid = hipThreadIdx_x & (WFSIZE - 1);

        // Wavefront id
        J wid = hipThreadIdx_x / WFSIZE;

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
        // Each wavefront processes a row of the BSR block.
        // If the number of BSR block rows exceed the number of wavefronts, each wavefront
        // processes multiple rows. 'bi' is the row index into the BSR block and 'bj' is
        // the column index.
        // BLOCKSIZE must be the square of WFSIZE.

        // Loop over the rows of the BSR block in chunks of WFSIZE, such that each
        // wavefront will process a row
        for(J bi = wid; bi < block_dim; bi += WFSIZE)
        {
            // BSR block row accumulator
            T sum = static_cast<T>(0);

            // Loop over all BSR blocks in the current row
            for(I j = row_begin; j < row_end; ++j)
            {
                // BSR column index
                J col = bsr_col_ind[j] - idx_base;

                // Loop over the columns of the BSR block in chunks of WFSIZE, such that
                // each lane will process a single value of the BSR block
                for(J bj = lid; bj < block_dim; bj += WFSIZE)
                {
                    // Each lane computes the sum of a specific entry over all BSR blocks in
                    // the current row

#define LBSR_IND(j, bi, bj, dir) \
    ((dir == rocsparse_direction_row) ? LBSR_IND_R(j, bi, bj) : LBSR_IND_C(j, bi, bj))
#define LBSR_IND_R(j, bi, bj) (size_t(block_dim) * block_dim * (j) + (bi)*block_dim + (bj))
#define LBSR_IND_C(j, bi, bj) (size_t(block_dim) * block_dim * (j) + (bi) + (bj)*block_dim)

                    sum = rocsparse::fma<T>(
                        bsr_val[LBSR_IND(j, bi, bj, dir)], x[block_dim * col + bj], sum);
                }
            }

            // Each wavefront accumulates its BSR block row sum
            sum = rocsparse::wfreduce_sum<WFSIZE>(sum);

            // Last lane of each wavefront writes its result to global memory
            if(lid == WFSIZE - 1)
            {
                if(beta != static_cast<T>(0))
                {
                    y[row * block_dim + bi]
                        = rocsparse::fma<T>(beta, y[row * block_dim + bi], alpha * sum);
                }
                else
                {
                    y[row * block_dim + bi] = alpha * sum;
                }
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
    void bsrxmvn_general_kernel(rocsparse_direction dir,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                J size_of_mask,
                                const J* __restrict__ bsr_mask_ptr,
                                const I* __restrict__ bsr_row_ptr,
                                const I* __restrict__ bsr_end_ptr,
                                const J* __restrict__ bsr_col_ind,
                                const A* __restrict__ bsr_val,
                                J block_dim,
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
            rocsparse::bsrxmvn_general_device<BLOCKSIZE, WFSIZE>(dir,
                                                                 alpha,
                                                                 size_of_mask,
                                                                 bsr_mask_ptr,
                                                                 bsr_row_ptr,
                                                                 bsr_end_ptr,
                                                                 bsr_col_ind,
                                                                 bsr_val,
                                                                 block_dim,
                                                                 x,
                                                                 beta,
                                                                 y,
                                                                 idx_base);
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void rocsparse::bsrxmvn_general(rocsparse_handle     handle,
                                rocsparse_direction  dir,
                                J                    mb,
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
    // Differentiate BSR block dimensions
    if(block_dim <= 8)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_general_kernel<64, 8>),
            dim3(size),
            dim3(8 * 8),
            0,
            handle->stream,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            block_dim,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(block_dim <= 16)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_general_kernel<256, 16>),
            dim3(size),
            dim3(16 * 16),
            0,
            handle->stream,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            block_dim,
            x,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            base,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrxmvn_general_kernel<1024, 32>),
            dim3(size),
            dim3(32 * 32),
            0,
            handle->stream,
            dir,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            size_of_mask,
            bsr_mask_ptr,
            bsr_row_ptr,
            bsr_end_ptr,
            bsr_col_ind,
            bsr_val,
            block_dim,
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
#define INSTANTIATE(T, I, J)                                                         \
    template void rocsparse::bsrxmvn_general(rocsparse_handle     handle,            \
                                             rocsparse_direction  dir,               \
                                             J                    mb,                \
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

#define INSTANTIATE_MIXED(T, I, J, A, X, Y)                                          \
    template void rocsparse::bsrxmvn_general(rocsparse_handle     handle,            \
                                             rocsparse_direction  dir,               \
                                             J                    mb,                \
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

INSTANTIATE_MIXED(double, int32_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int32_t, float, double, double);
INSTANTIATE_MIXED(double, int64_t, int64_t, float, double, double);

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
