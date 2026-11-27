/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse_common.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_scalar.hpp"

#include <cstdint>
#include <limits>

// 64-bit bsrsv solve kernels. The host launches these when the matrix index
// types are not rocsparse_int or when an offset of the solve does not fit in
// rocsparse_int. Each thread block walks a contiguous chunk of the row-map
// slots, so a clamped grid still covers every block row and the done-flag
// spins stay in topological order.
//
// The narrow kernels in bsrsv_device.h keep their rocsparse_int signatures.
// Their 32/32 instantiation is the legacy launch, and they walk a clamped
// grid the same way, so a clamped grid alone does not need these kernels.

namespace rocsparse
{
    // True when the wide kernels run instead of the narrow (rocsparse_int)
    // ones: mb itself does not fit in rocsparse_int, or the in-block offset
    // bi * block_dim, which BSR_IND forms in 32 bits, does not fit. Both paths
    // form the vector offsets row * block_dim and the block value offset
    // block_dim * block_dim * j in int64_t, so neither mb * block_dim nor nnzb
    // decides.
    __host__ __forceinline__ bool bsrsv_solve_needs_wide(int64_t mb, int64_t block_dim)
    {
        if(mb <= 0 || block_dim <= 0)
        {
            return false;
        }

        const int64_t narrow_max = static_cast<int64_t>(std::numeric_limits<rocsparse_int>::max());
        if(mb > narrow_max)
        {
            return true;
        }

        // block_dim * block_dim <= narrow_max is tested without forming the product.
        return block_dim > narrow_max / block_dim;
    }

    // Offset of entry (bi, bj) of block j in bsr_val, formed entirely in
    // int64_t. BSR_IND widens only block_dim * block_dim * j, and its
    // bi * block_dim and bj * block_dim overflow int from block_dim 46341,
    // which bsrsv_solve_needs_wide sends to the general wide kernels.
    __host__ __device__ __forceinline__ int64_t bsrsv_wide_ind(
        int64_t j, int64_t bi, int64_t bj, int64_t block_dim, rocsparse_direction dir)
    {
        return block_dim * block_dim * j
               + ((dir == rocsparse_direction_row) ? bi * block_dim + bj : bi + bj * block_dim);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsrsv_lower_general_wide_device(int64_t idx,
                                                              J       mb,
                                                              T       alpha,
                                                              const I* __restrict__ bsr_row_ptr,
                                                              const J* __restrict__ bsr_col_ind,
                                                              const T* __restrict__ bsr_val,
                                                              rocsparse_int block_dim,
                                                              const T* __restrict__ x,
                                                              T* y,
                                                              int* __restrict__ done_array,
                                                              const J* __restrict__ map,
                                                              J*                   zero_pivot,
                                                              rocsparse_index_base idx_base,
                                                              rocsparse_diag_type  diag_type,
                                                              rocsparse_direction  dir)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");

        const int lid = hipThreadIdx_x & (WFSIZE - 1);

        const J       row_j = map[idx];
        const int64_t row   = static_cast<int64_t>(row_j);
        const int64_t y_row = row * static_cast<int64_t>(block_dim);

        const I row_begin = bsr_row_ptr[row_j] - idx_base;
        const I row_end   = bsr_row_ptr[row_j + 1] - idx_base;

        int64_t local_col = static_cast<int64_t>(mb);

        for(int bi = lid; bi < block_dim; bi += WFSIZE)
        {
            y[y_row + bi] = alpha * x[y_row + bi];
        }

        I j;
        for(j = row_begin; j < row_end; ++j)
        {
            local_col = static_cast<int64_t>(bsr_col_ind[j] - idx_base);

            if(local_col >= row)
            {
                break;
            }

            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __MEMORY_SCOPE_DEVICE);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            const int64_t y_col = local_col * static_cast<int64_t>(block_dim);
            for(int bi = lid; bi < block_dim; bi += WFSIZE)
            {
                T local_sum = static_cast<T>(0);
                for(int bj = 0; bj < block_dim; ++bj)
                {
                    local_sum = rocsparse::fma(
                        bsr_val[rocsparse::bsrsv_wide_ind(j, bi, bj, block_dim, dir)],
                        y[y_col + bj],
                        local_sum);
                }
                y[y_row + bi] -= local_sum;
            }
        }

        bool pivot = false;

        if(local_col == row)
        {
            for(int bi = 0; bi < block_dim; ++bi)
            {
                const int64_t diag_off = rocsparse::bsrsv_wide_ind(j, bi, bi, block_dim, dir);
                const T       diag = (diag_type == rocsparse_diag_type_non_unit) ? bsr_val[diag_off]
                                                                                 : static_cast<T>(1);

                T val = y[y_row + bi];
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    y[y_row + bi] = val /= diag;
                }

                for(int bj = bi + lid + 1; bj < block_dim; bj += WFSIZE)
                {
                    y[y_row + bj]
                        -= val * bsr_val[rocsparse::bsrsv_wide_ind(j, bj, bi, block_dim, dir)];
                }
            }
        }

        if(lid == 0)
        {
            rocsparse::atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, static_cast<J>(row + idx_base));
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsrsv_upper_general_wide_device(int64_t idx,
                                                              J       mb,
                                                              T       alpha,
                                                              const I* __restrict__ bsr_row_ptr,
                                                              const J* __restrict__ bsr_col_ind,
                                                              const T* __restrict__ bsr_val,
                                                              rocsparse_int block_dim,
                                                              const T* __restrict__ x,
                                                              T* y,
                                                              int* __restrict__ done_array,
                                                              const J* __restrict__ map,
                                                              J*                   zero_pivot,
                                                              rocsparse_index_base idx_base,
                                                              rocsparse_diag_type  diag_type,
                                                              rocsparse_direction  dir)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");

        const int lid = hipThreadIdx_x & (WFSIZE - 1);

        const J       row_j = map[idx];
        const int64_t row   = static_cast<int64_t>(row_j);
        const int64_t y_row = row * static_cast<int64_t>(block_dim);

        const I row_begin = bsr_row_ptr[row_j] - idx_base;
        const I row_end   = bsr_row_ptr[row_j + 1] - idx_base;

        int64_t local_col = static_cast<int64_t>(mb);

        for(int bi = lid; bi < block_dim; bi += WFSIZE)
        {
            y[y_row + bi] = alpha * x[y_row + bi];
        }

        I j;
        for(j = row_end - 1; j >= row_begin; --j)
        {
            local_col = static_cast<int64_t>(bsr_col_ind[j] - idx_base);

            if(local_col <= row)
            {
                break;
            }

            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __MEMORY_SCOPE_DEVICE);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            const int64_t y_col = local_col * static_cast<int64_t>(block_dim);
            for(int bi = lid; bi < block_dim; bi += WFSIZE)
            {
                T local_sum = static_cast<T>(0);
                for(int bj = 0; bj < block_dim; ++bj)
                {
                    local_sum = rocsparse::fma(
                        bsr_val[rocsparse::bsrsv_wide_ind(j, bi, bj, block_dim, dir)],
                        y[y_col + bj],
                        local_sum);
                }
                y[y_row + bi] -= local_sum;
            }
        }

        bool pivot = false;

        if(local_col == row)
        {
            for(int bi = block_dim - 1; bi >= 0; --bi)
            {
                const int64_t diag_off = rocsparse::bsrsv_wide_ind(j, bi, bi, block_dim, dir);
                const T       diag = (diag_type == rocsparse_diag_type_non_unit) ? bsr_val[diag_off]
                                                                                 : static_cast<T>(1);

                T val = y[y_row + bi];
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    y[y_row + bi] = val /= diag;
                }

                for(int bj = lid; bj < bi; bj += WFSIZE)
                {
                    y[y_row + bj]
                        -= val * bsr_val[rocsparse::bsrsv_wide_ind(j, bj, bi, block_dim, dir)];
                }
            }
        }

        if(lid == 0)
        {
            rocsparse::atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, static_cast<J>(row + idx_base));
            }
        }
    }

    template <uint32_t      BLOCKSIZE,
              uint32_t      WFSIZE,
              rocsparse_int BSRDIM,
              bool          SLEEP,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void bsrsv_lower_shared_wide_device(int64_t idx,
                                                             J       mb,
                                                             T       alpha,
                                                             const I* __restrict__ bsr_row_ptr,
                                                             const J* __restrict__ bsr_col_ind,
                                                             const T* __restrict__ bsr_val,
                                                             rocsparse_int block_dim,
                                                             const T* __restrict__ x,
                                                             T* y,
                                                             int* __restrict__ done_array,
                                                             const J* __restrict__ map,
                                                             J*                   zero_pivot,
                                                             rocsparse_index_base idx_base,
                                                             rocsparse_diag_type  diag_type,
                                                             rocsparse_direction  dir)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        static_assert(WFSIZE % BSRDIM == 0, "WFSIZE must be a multiple of BSRDIM.");

        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        const J       row_j = map[idx];
        const int64_t row   = static_cast<int64_t>(row_j);
        const int64_t y_row = row * static_cast<int64_t>(block_dim);

        const I row_begin = bsr_row_ptr[row_j] - idx_base;
        const I row_end   = bsr_row_ptr[row_j + 1] - idx_base;

        int64_t local_col = static_cast<int64_t>(mb);

        T local_sum = alpha * ((lid < block_dim) ? x[y_row + lid] : static_cast<T>(0));

        __shared__ T sdata1[BLOCKSIZE / WFSIZE * BSRDIM * BSRDIM];
        __shared__ T sdata2[BLOCKSIZE / WFSIZE * BSRDIM];

        T* bsr_values  = &sdata1[wid * BSRDIM * BSRDIM];
        T* bsr_updates = &sdata2[wid * BSRDIM];

        I j;
        for(j = row_begin; j < row_end; ++j)
        {
            local_col = static_cast<int64_t>(bsr_col_ind[j] - idx_base);

            const int bi = lid & (BSRDIM - 1);
            const int bj = lid / BSRDIM;

            for(int k = bj; k < BSRDIM; k += WFSIZE / BSRDIM)
            {
                bsr_values[bi + k * BSRDIM] = (bi < block_dim && k < block_dim)
                                                  ? bsr_val[BSR_IND(j, bi, k, dir)]
                                                  : static_cast<T>(0);
            }

            if(local_col >= row)
            {
                break;
            }

            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __MEMORY_SCOPE_DEVICE);
            rocsparse::acquire_done_rows();

            const int64_t y_col = local_col * static_cast<int64_t>(block_dim);
            if(lid < BSRDIM)
            {
                bsr_updates[lid] = (lid < block_dim)
                                       ? rocsparse::load_done_row_data(&y[y_col + lid])
                                       : static_cast<T>(0);
            }

            __threadfence_block();

            if(lid < block_dim)
            {
                for(int l = 0; l < BSRDIM; ++l)
                {
                    local_sum
                        = rocsparse::fma(-bsr_values[lid + l * BSRDIM], bsr_updates[l], local_sum);
                }
            }
        }

        bool pivot = false;

        if(local_col == row)
        {
            for(int bi = 0; bi < block_dim; ++bi)
            {
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_values[bi + bi * BSRDIM]
                                   : static_cast<T>(1);

                T val = rocsparse::shfl(local_sum, bi);

                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    val /= diag;
                }

                if(lid < block_dim)
                {
                    if(bi < lid)
                    {
                        local_sum = rocsparse::fma(-val, bsr_values[lid + bi * BSRDIM], local_sum);
                    }
                    else if(lid == bi)
                    {
                        local_sum = val;
                    }
                }
            }
        }

        if(lid < block_dim)
        {
            rocsparse::store_done_row_data(&y[y_row + lid], local_sum);
        }

        if(lid == 0)
        {
            rocsparse::set_done_flag(&done_array[row], 1);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, static_cast<J>(row + idx_base));
            }
        }
    }

    template <uint32_t      BLOCKSIZE,
              uint32_t      WFSIZE,
              rocsparse_int BSRDIM,
              bool          SLEEP,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void bsrsv_upper_shared_wide_device(int64_t idx,
                                                             J       mb,
                                                             T       alpha,
                                                             const I* __restrict__ bsr_row_ptr,
                                                             const J* __restrict__ bsr_col_ind,
                                                             const T* __restrict__ bsr_val,
                                                             rocsparse_int block_dim,
                                                             const T* __restrict__ x,
                                                             T* y,
                                                             int* __restrict__ done_array,
                                                             const J* __restrict__ map,
                                                             J*                   zero_pivot,
                                                             rocsparse_index_base idx_base,
                                                             rocsparse_diag_type  diag_type,
                                                             rocsparse_direction  dir)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        static_assert(WFSIZE % BSRDIM == 0, "WFSIZE must be a multiple of BSRDIM.");

        const int lid = hipThreadIdx_x & (WFSIZE - 1);
        const int wid = hipThreadIdx_x / WFSIZE;

        const J       row_j = map[idx];
        const int64_t row   = static_cast<int64_t>(row_j);
        const int64_t y_row = row * static_cast<int64_t>(block_dim);

        const I row_begin = bsr_row_ptr[row_j] - idx_base;
        const I row_end   = bsr_row_ptr[row_j + 1] - idx_base;

        int64_t local_col = static_cast<int64_t>(mb);

        T local_sum = alpha * ((lid < block_dim) ? x[y_row + lid] : static_cast<T>(0));

        __shared__ T sdata1[BLOCKSIZE / WFSIZE * BSRDIM * BSRDIM];
        __shared__ T sdata2[BLOCKSIZE / WFSIZE * BSRDIM];

        T* bsr_values  = &sdata1[wid * BSRDIM * BSRDIM];
        T* bsr_updates = &sdata2[wid * BSRDIM];

        I j;
        for(j = row_end - 1; j >= row_begin; --j)
        {
            local_col = static_cast<int64_t>(bsr_col_ind[j] - idx_base);

            const int bi = lid & (BSRDIM - 1);
            const int bj = lid / BSRDIM;

            for(int k = bj; k < BSRDIM; k += WFSIZE / BSRDIM)
            {
                bsr_values[bi + k * BSRDIM] = (bi < block_dim && k < block_dim)
                                                  ? bsr_val[BSR_IND(j, bi, k, dir)]
                                                  : static_cast<T>(0);
            }

            if(local_col <= row)
            {
                break;
            }

            rocsparse::spin_loop<SLEEP>(&done_array[local_col], __MEMORY_SCOPE_DEVICE);
            rocsparse::acquire_done_rows();

            const int64_t y_col = local_col * static_cast<int64_t>(block_dim);
            if(lid < BSRDIM)
            {
                bsr_updates[lid] = (lid < block_dim)
                                       ? rocsparse::load_done_row_data(&y[y_col + lid])
                                       : static_cast<T>(0);
            }

            __threadfence_block();

            if(lid < block_dim)
            {
                for(int l = 0; l < BSRDIM; ++l)
                {
                    local_sum
                        = rocsparse::fma(-bsr_values[lid + l * BSRDIM], bsr_updates[l], local_sum);
                }
            }
        }

        bool pivot = false;

        if(local_col == row)
        {
            for(int bi = block_dim - 1; bi >= 0; --bi)
            {
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_values[bi + bi * BSRDIM]
                                   : static_cast<T>(1);

                T val = rocsparse::shfl(local_sum, bi);

                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    val /= diag;
                }

                if(lid < block_dim)
                {
                    if(bi > lid)
                    {
                        local_sum = rocsparse::fma(-val, bsr_values[lid + bi * BSRDIM], local_sum);
                    }
                    else if(lid == bi)
                    {
                        local_sum = val;
                    }
                }
            }
        }

        if(lid < block_dim)
        {
            rocsparse::store_done_row_data(&y[y_row + lid], local_sum);
        }

        if(lid == 0)
        {
            rocsparse::set_done_flag(&done_array[row], 1);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, static_cast<J>(row + idx_base));
            }
        }
    }

    // One virtual block covers BLOCKSIZE / WFSIZE block rows. A clamped grid
    // walks its chunk; an exact grid runs a single virtual block, the same
    // slot the narrow kernel takes from hipBlockIdx_x.
#define BSRSV_WIDE_FOR_EACH(BLOCKSIZE, WFSIZE, MB, BODY)                                         \
    do                                                                                           \
    {                                                                                            \
        constexpr uint32_t bsrsv_nwarps = (BLOCKSIZE) / (WFSIZE);                                \
        const int          bsrsv_wid    = hipThreadIdx_x / (WFSIZE);                             \
        const int64_t      bsrsv_count                                                           \
            = (static_cast<int64_t>(MB) - 1) / static_cast<int64_t>(bsrsv_nwarps) + 1;           \
        int64_t bsrsv_first;                                                                     \
        int64_t bsrsv_last;                                                                      \
        rocsparse::grid_x_chunk(                                                                 \
            bsrsv_count, hipGridDim_x, hipBlockIdx_x, bsrsv_first, bsrsv_last);                  \
        for(int64_t bsrsv_block_id = bsrsv_first; bsrsv_block_id < bsrsv_last; ++bsrsv_block_id) \
        {                                                                                        \
            const int64_t idx = bsrsv_block_id * static_cast<int64_t>(bsrsv_nwarps) + bsrsv_wid; \
            if(idx < static_cast<int64_t>(MB))                                                   \
            {                                                                                    \
                BODY;                                                                            \
            }                                                                                    \
        }                                                                                        \
    } while(0)

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_lower_general_wide(J mb,
                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                  const I* __restrict__ bsr_row_ptr,
                                  const J* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  rocsparse_int block_dim,
                                  const T* __restrict__ x,
                                  T* y,
                                  int* __restrict__ done_array,
                                  const J* __restrict__ map,
                                  J*                   zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir,
                                  bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        BSRSV_WIDE_FOR_EACH(
            BLOCKSIZE,
            WFSIZE,
            mb,
            (rocsparse::bsrsv_lower_general_wide_device<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>(
                idx,
                mb,
                alpha,
                bsr_row_ptr,
                bsr_col_ind,
                bsr_val,
                block_dim,
                x,
                y,
                done_array,
                map,
                zero_pivot,
                idx_base,
                diag_type,
                dir)));
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_upper_general_wide(J mb,
                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                  const I* __restrict__ bsr_row_ptr,
                                  const J* __restrict__ bsr_col_ind,
                                  const T* __restrict__ bsr_val,
                                  rocsparse_int block_dim,
                                  const T* __restrict__ x,
                                  T* y,
                                  int* __restrict__ done_array,
                                  const J* __restrict__ map,
                                  J*                   zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir,
                                  bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        BSRSV_WIDE_FOR_EACH(
            BLOCKSIZE,
            WFSIZE,
            mb,
            (rocsparse::bsrsv_upper_general_wide_device<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>(
                idx,
                mb,
                alpha,
                bsr_row_ptr,
                bsr_col_ind,
                bsr_val,
                block_dim,
                x,
                y,
                done_array,
                map,
                zero_pivot,
                idx_base,
                diag_type,
                dir)));
    }

    template <uint32_t      BLOCKSIZE,
              uint32_t      WFSIZE,
              rocsparse_int BSRDIM,
              bool          SLEEP,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_lower_shared_wide(J mb,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 const I* __restrict__ bsr_row_ptr,
                                 const J* __restrict__ bsr_col_ind,
                                 const T* __restrict__ bsr_val,
                                 rocsparse_int block_dim,
                                 const T* __restrict__ x,
                                 T* y,
                                 int* __restrict__ done_array,
                                 const J* __restrict__ map,
                                 J*                   zero_pivot,
                                 rocsparse_index_base idx_base,
                                 rocsparse_diag_type  diag_type,
                                 rocsparse_direction  dir,
                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        BSRSV_WIDE_FOR_EACH(
            BLOCKSIZE,
            WFSIZE,
            mb,
            (rocsparse::bsrsv_lower_shared_wide_device<BLOCKSIZE, WFSIZE, BSRDIM, SLEEP, T, I, J>(
                idx,
                mb,
                alpha,
                bsr_row_ptr,
                bsr_col_ind,
                bsr_val,
                block_dim,
                x,
                y,
                done_array,
                map,
                zero_pivot,
                idx_base,
                diag_type,
                dir)));
    }

    template <uint32_t      BLOCKSIZE,
              uint32_t      WFSIZE,
              rocsparse_int BSRDIM,
              bool          SLEEP,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_upper_shared_wide(J mb,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 const I* __restrict__ bsr_row_ptr,
                                 const J* __restrict__ bsr_col_ind,
                                 const T* __restrict__ bsr_val,
                                 rocsparse_int block_dim,
                                 const T* __restrict__ x,
                                 T* y,
                                 int* __restrict__ done_array,
                                 const J* __restrict__ map,
                                 J*                   zero_pivot,
                                 rocsparse_index_base idx_base,
                                 rocsparse_diag_type  diag_type,
                                 rocsparse_direction  dir,
                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        BSRSV_WIDE_FOR_EACH(
            BLOCKSIZE,
            WFSIZE,
            mb,
            (rocsparse::bsrsv_upper_shared_wide_device<BLOCKSIZE, WFSIZE, BSRDIM, SLEEP, T, I, J>(
                idx,
                mb,
                alpha,
                bsr_row_ptr,
                bsr_col_ind,
                bsr_val,
                block_dim,
                x,
                y,
                done_array,
                map,
                zero_pivot,
                idx_base,
                diag_type,
                dir)));
    }
}

#undef BSRSV_WIDE_FOR_EACH
