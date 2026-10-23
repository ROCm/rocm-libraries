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

// Transpose-free solve of op(A) * y = alpha * x for op(A) = A^T or A^H with A
// stored in CSR. Row i of A holds the coefficients that y_i contributes to the
// other unknowns, so once y_i is known its group pushes op(a_ij) * y_i into
// sum[j] and decrements rem[j]. Unknown j is ready when rem[j] reaches zero.
namespace rocsparse
{
    template <uint32_t L, typename T>
    __device__ __forceinline__ T group_broadcast(T v)
    {
        return __shfl(v, 0, L);
    }

    template <uint32_t L>
    __device__ __forceinline__ int group_any(int v)
    {
        for(uint32_t o = L / 2; o > 0; o >>= 1)
        {
            v |= __shfl_xor(v, o, L);
        }
        return v;
    }

    template <uint32_t BLOCKSIZE, uint32_t L, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsv_transpose_free_analysis_kernel(J m,
                                              const I* __restrict__ csr_row_ptr,
                                              const J* __restrict__ csr_col_ind,
                                              I* __restrict__ csr_diag_ind,
                                              J* __restrict__ in_degree,
                                              I* __restrict__ max_nnz,
                                              J* __restrict__ zero_pivot,
                                              rocsparse_index_base idx_base,
                                              rocsparse_diag_type  diag_type,
                                              rocsparse_fill_mode  fill_mode)
    {
        const uint32_t lid = hipThreadIdx_x & (L - 1);
        const J        row = (hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x) / L;

        if(row >= m)
        {
            return;
        }

        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        int has_diag = 0;
        for(I j = row_begin + lid; j < row_end; j += L)
        {
            const J col = csr_col_ind[j] - idx_base;
            if(col == row)
            {
                csr_diag_ind[row] = j;
                has_diag          = 1;
            }
            else if((fill_mode == rocsparse_fill_mode_lower) ? (col < row) : (col > row))
            {
                rocsparse::atomic_add(&in_degree[col], static_cast<J>(1));
            }
        }

        has_diag = rocsparse::group_any<L>(has_diag);
        if(lid == 0)
        {
            rocsparse::atomic_max(max_nnz, row_end - row_begin);
            if(!has_diag && diag_type == rocsparse_diag_type_non_unit)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsv_transpose_free_init_kernel(J       m,
                                          int64_t batch_count,
                                          const J* __restrict__ in_degree,
                                          J* __restrict__ rem)
    {
        const int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(i >= m * batch_count)
        {
            return;
        }
        rem[i] = in_degree[i % m];
    }

    // One group of L lanes per unknown. Blocks take their position in the schedule
    // from an atomic ticket, so a group only waits on groups that are already running.
    // The wait and the push live in the same loop: several groups may share a
    // wavefront, and a ready group must not wait for a spinning neighbour.
    template <uint32_t BLOCKSIZE, uint32_t L, typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void csrsv_transpose_free_device(J m,
                                                          T alpha,
                                                          const I* __restrict__ csr_row_ptr,
                                                          const J* __restrict__ csr_col_ind,
                                                          const T* __restrict__ csr_val,
                                                          int64_t csr_val_inc,
                                                          const I* __restrict__ csr_diag_ind,
                                                          const T* __restrict__ x,
                                                          int64_t x_inc,
                                                          T*      y,
                                                          int64_t y_inc,
                                                          J* __restrict__ rem,
                                                          T* __restrict__ sum,
                                                          int32_t* __restrict__ ticket,
                                                          J*                   zero_pivot,
                                                          rocsparse_index_base idx_base,
                                                          rocsparse_fill_mode  fill_mode,
                                                          rocsparse_diag_type  diag_type,
                                                          bool                 conj)
    {
        __shared__ int32_t block_id;
        if(hipThreadIdx_x == 0)
        {
            block_id = rocsparse::atomic_add(ticket, 1);
        }
        __syncthreads();

        const uint32_t lid = hipThreadIdx_x & (L - 1);
        const int64_t  w64 = static_cast<int64_t>(block_id) * (BLOCKSIZE / L) + hipThreadIdx_x / L;

        if(w64 >= m)
        {
            return;
        }

        const J w = static_cast<J>(w64);

        const bool lower = (fill_mode == rocsparse_fill_mode_lower);
        const J    row   = lower ? (m - 1 - w) : w;

        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Off the critical path: everything that does not depend on other unknowns.
        T inv_diag = static_cast<T>(1);
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            const I d = csr_diag_ind[row];
            if(d >= 0)
            {
                T diag = rocsparse::conj_val(csr_val[d * csr_val_inc], conj);
                if(diag == static_cast<T>(0))
                {
                    if(lid == 0)
                    {
                        rocsparse::atomic_min(zero_pivot, row + idx_base);
                    }
                    diag = static_cast<T>(1);
                }
                inv_diag = static_cast<T>(1) / diag;
            }
        }
        const T ax = alpha * x[row * x_inc];

        bool done = false;
        while(!done)
        {
            J r = (lid == 0)
                      ? rocsparse::atomic_load(&rem[row], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      : static_cast<J>(0);
            r   = rocsparse::group_broadcast<L>(r);

            if(r == 0)
            {
                rocsparse::acquire_done_rows();

                const T s
                    = rocsparse::atomic_load(&sum[row], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                T yi = ax - s;
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    yi = yi * inv_diag;
                }

                if(lid == 0)
                {
                    y[row * y_inc] = yi;
                }

                for(I j = row_begin + lid; j < row_end; j += L)
                {
                    const J col = csr_col_ind[j] - idx_base;
                    if(lower ? (col < row) : (col > row))
                    {
                        const T val = rocsparse::conj_val(csr_val[j * csr_val_inc], conj);
                        rocsparse::atomic_add(&sum[col], val * yi);
                    }
                }

                // One release for all the additions of this lane, without the L2 writeback that
                // an agent scope release costs on GPUs with non-coherent XCD L2s.
                rocsparse::release_done_row_data();

                for(I j = row_begin + lid; j < row_end; j += L)
                {
                    const J col = csr_col_ind[j] - idx_base;
                    if(lower ? (col < row) : (col > row))
                    {
                        __scoped_atomic_fetch_sub(
                            &rem[col], static_cast<J>(1), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                    }
                }
                done = true;
            }
            else
            {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
}
