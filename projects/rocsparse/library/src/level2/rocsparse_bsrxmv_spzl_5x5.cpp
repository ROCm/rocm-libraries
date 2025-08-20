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
    template <uint32_t BLOCKSIZE, rocsparse_direction DIR, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void sbsrxmvn_5x5_device(J     mb,
                                                  float alpha,
                                                  J     size_of_mask,
                                                  const J* __restrict__ bsr_mask_ptr,
                                                  const I* __restrict__ bsr_row_ptr,
                                                  const I* __restrict__ bsr_end_ptr,
                                                  const J* __restrict__ bsr_col_ind,
                                                  const float* __restrict__ bsr_val,
                                                  const float* __restrict__ x,
                                                  float beta,
                                                  float* __restrict__ y,
                                                  rocsparse_index_base idx_base)

    {
        static constexpr int block_size    = 5;
        static constexpr int sq_block_size = block_size * block_size;

        int const local_i = threadIdx.x % block_size;
        int const local_j = threadIdx.x / block_size;

        int row = blockIdx.x * blockDim.y + threadIdx.y;
        if(bsr_mask_ptr != nullptr)
        {
            row = bsr_mask_ptr[row] - idx_base;
        }

        if((row < mb) && (local_j < block_size))
        {
            const int global_row = local_i + row * block_size;

            const I offset_begin = bsr_row_ptr[row] - idx_base;
            const I offset_end   = (bsr_end_ptr == nullptr) ? (bsr_row_ptr[row + 1] - idx_base)
                                                            : (bsr_end_ptr[row] - idx_base);

            float fk, f1;
            fk = static_cast<float>(0);
            if(DIR == rocsparse_direction_row)
            {
                I offset = offset_begin;
                bsr_val += local_j + local_i * block_size;
                bsr_val += int64_t(sq_block_size) * offset;
#pragma unroll 4
                for(; offset < offset_end; offset++)
                {
                    const J jam0 = bsr_col_ind[offset] - idx_base;
                    fk           = rocsparse::fma(*bsr_val, x[local_j + jam0 * block_size], fk);
                    bsr_val += sq_block_size;
                }
            }
            else
            {
                I offset = offset_begin;
                bsr_val += local_i + local_j * block_size;
                bsr_val += int64_t(sq_block_size) * offset;
#pragma unroll 4
                for(; offset < offset_end; offset++)
                {
                    const J jam0 = bsr_col_ind[offset] - idx_base;
                    fk           = rocsparse::fma(*bsr_val, x[local_j + jam0 * block_size], fk);
                    bsr_val += sq_block_size;
                }
            }

            // Reduction along the subcolumns, threads with l=0 hold the complete sum
            f1 = fk;
#pragma unroll 4
            for(int s = 1; s < block_size; ++s)
                f1 = f1 + rocsparse::shfl(fk, local_i + s * block_size, 32);

            f1 = beta * y[global_row] + alpha * f1;
            if(local_j == 0)
            {
                y[global_row] = f1;
            }
        }
    }

    // BSRXMV kernel for BSR block dimension of 5
    template <uint32_t BLOCKSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_DEVICE_ILF void bsrxmvn_5x5_device(J                   mb,
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
        static constexpr int BSRDIM = 5;

        // BSR block lane id
        J lid = hipThreadIdx_x % BSRDIM;

        // Number of BSR blocks processed at the same time
        const uint32_t NBLOCKS = BLOCKSIZE / (BSRDIM * BSRDIM);

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

#if 0
    // Each thread block processes a single BSR row
    J row = bsr_mask_ptr[hipBlockIdx_x] - idx_base;



    // BSR row entry and exit point
    I row_begin = bsr_row_ptr[row] - idx_base;
    I row_end   = bsr_end_ptr[row] - idx_base;
#endif

        // BSR block row accumulator
        T sum = static_cast<T>(0);

        // Loop over all BSR blocks in the current row where each lane
        // processes a BSR block value
        for(I j = row_begin; j < row_end; j += NBLOCKS)
        {
            I k = j + hipThreadIdx_x / (BSRDIM * BSRDIM);

            // Do not exceed the row
            if(k < row_end)
            {
                // Column index into x vector
                J col = (bsr_col_ind[k] - idx_base) * BSRDIM;

                // Compute the sum of the two rows within the BSR blocks of the current
                // BSR row
                sum = rocsparse::fma<T>(
                    bsr_val[j * BSRDIM * BSRDIM + hipThreadIdx_x], x[col + idx], sum);
            }
        }

        // Accumulate each row sum of the BSR block
        __shared__ T sdata[BSRDIM * BSRDIM * NBLOCKS];

        sdata[hipThreadIdx_x] = sum;

        __threadfence_block();

        if(dir == rocsparse_direction_column)
        {
            if(hipThreadIdx_x < BLOCKSIZE - BSRDIM * 8)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 8];
            __threadfence_block();
            if(hipThreadIdx_x < BSRDIM * 4)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 4];
            __threadfence_block();
            if(hipThreadIdx_x < BSRDIM * 2)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * 2];
            __threadfence_block();
            if(hipThreadIdx_x < BSRDIM * 1)
                sum = sdata[hipThreadIdx_x] + sdata[hipThreadIdx_x + BSRDIM * 1];
        }
        else
        {
            // Accumulate the row sum for different blocks
            if(hipThreadIdx_x < BSRDIM * BSRDIM)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + BSRDIM * BSRDIM];
            __threadfence_block();

            // Reduce the intra block row sum
            if(lid < 1)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 4];
            __threadfence_block();
            if(lid < 2)
                sdata[hipThreadIdx_x] += sdata[hipThreadIdx_x + 2];
            __threadfence_block();

            // Final reduction
            if(hipThreadIdx_x < BSRDIM)
                sum = sdata[hipThreadIdx_x * BSRDIM] + sdata[hipThreadIdx_x * BSRDIM + 1];
        }

        // First 5 threads write row sums to global memory
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

    template <uint32_t BLOCKSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrxmvn_5x5_kernel(J                   mb,
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
            rocsparse::bsrxmvn_5x5_device<BLOCKSIZE>(mb,
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

    template <uint32_t BLOCKSIZE, rocsparse_direction DIR, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void sbsrxmvn_5x5_kernel(J mb,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, alpha),
                             J size_of_mask,
                             const J* __restrict__ bsr_mask_ptr,
                             const I* __restrict__ bsr_row_ptr,
                             const I* __restrict__ bsr_end_ptr,
                             const J* __restrict__ bsr_col_ind,
                             const float* __restrict__ bsr_val,
                             const float* __restrict__ x,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, beta),
                             float* __restrict__ y,
                             rocsparse_index_base idx_base,
                             bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != static_cast<float>(0) || beta != static_cast<float>(1))
        {
            rocsparse::sbsrxmvn_5x5_device<BLOCKSIZE, DIR>(mb,
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

    template <typename T, typename A, typename X, typename Y>
    struct kernels_type_dispatch
    {
        template <typename I, typename J>
        static void bsrxmvn_5x5(rocsparse_handle     handle,
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

            const J size = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;
            THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrxmvn_5x5_kernel<50>),
                dim3(size),
                dim3(50),
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
    };

    //
    // Specialization for floats.
    //
    template <>
    struct kernels_type_dispatch<float, float, float, float>
    {
        using T = float;
        template <typename I, typename J>
        static void bsrxmvn_5x5(rocsparse_handle     handle,
                                rocsparse_direction  dir,
                                J                    mb,
                                I                    nnzb,
                                const float*         alpha_device_host,
                                J                    size_of_mask,
                                const J*             bsr_mask_ptr,
                                const I*             bsr_row_ptr,
                                const I*             bsr_end_ptr,
                                const J*             bsr_col_ind,
                                const float*         bsr_val,
                                const float*         x,
                                const float*         beta_device_host,
                                float*               y,
                                rocsparse_index_base base)
        {
            ROCSPARSE_ROUTINE_TRACE;

            const int wsize = handle->wavefront_size;
            if(wsize == 32)
            {
                const J size = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;
                THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::bsrxmvn_5x5_kernel<50, float>),
                    dim3(size),
                    dim3(50),
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
                static constexpr int nhalfwarps_per_block  = 8;
                static constexpr int nthreads_per_halfwarp = 32;
                const J              size = (bsr_mask_ptr == nullptr) ? mb : size_of_mask;
                dim3 const nThreads_solver(nthreads_per_halfwarp, nhalfwarps_per_block, 1);
                dim3 const nBlocks_solver((size - 1) / nhalfwarps_per_block + 1, 1, 1);

                if(rocsparse_direction_row == dir)
                {
                    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::sbsrxmvn_5x5_kernel<nthreads_per_halfwarp
                                                            * nhalfwarps_per_block,
                                                        rocsparse_direction_row>),
                        nBlocks_solver,
                        nThreads_solver,
                        0,
                        handle->stream,
                        mb,
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
                        (rocsparse::sbsrxmvn_5x5_kernel<nthreads_per_halfwarp
                                                            * nhalfwarps_per_block,
                                                        rocsparse_direction_column>),
                        nBlocks_solver,
                        nThreads_solver,
                        0,
                        handle->stream,
                        mb,
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
        }
    };
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void rocsparse::bsrxmvn_5x5(rocsparse_handle     handle,
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

    kernels_type_dispatch<T, A, X, Y>::template bsrxmvn_5x5<I, J>(handle,
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
                                                                  base);
}

//
// INSTANTIATE.
//
#define INSTANTIATE(T, I, J)                                                     \
    template void rocsparse::bsrxmvn_5x5(rocsparse_handle     handle,            \
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
    template void rocsparse::bsrxmvn_5x5(rocsparse_handle     handle,            \
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
