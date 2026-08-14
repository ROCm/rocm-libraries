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

extern "C" void __builtin_amdgcn_s_sleep(int);

namespace rocsparse
{
    // ELL storage is column-major with leading dimension m: entry (row, slot p)
    // lives at ell_col_ind[p * m + row] / ell_val[p * m + row]. Padding entries
    // carry an out-of-range column index. The kernels below scan every slot and
    // skip padded / non-contributing entries, so they do not depend on the
    // entries being sorted within a row (which lets the transposed ELL matrix be
    // produced by an unordered atomic scatter).

    // Analysis kernel for a lower triangular ELL matrix. Each wavefront processes
    // a single row and computes its dependency depth (level) by spin-waiting on
    // the depths of the rows it depends on. The depths are written into
    // done_array and later sorted to obtain the row execution order.
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_analysis_lower_kernel(I m,
                                     I n,
                                     int64_t ell_width,
                                     const I* __restrict__ ell_col_ind,
                                     int* __restrict__ done_array,
                                     rocsparse_index_base idx_base)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const int lid = hipThreadIdx_x & (WF_SIZE - 1);
        const int wid = hipThreadIdx_x / WF_SIZE;

        const I row = static_cast<I>(hipBlockIdx_x) * (BLOCKSIZE / WF_SIZE) + wid;

        if(row >= m)
        {
            return;
        }

        // Local dependency depth.
        int local_max = 0;

        for(int64_t p = lid; p < ell_width; p += WF_SIZE)
        {
            const int64_t idx = p * static_cast<int64_t>(m) + row;
            const I       col = rocsparse::nontemporal_load(ell_col_ind + idx) - idx_base;

            // Skip padded (out-of-range) entries.
            if(col < 0 || col >= n)
            {
                continue;
            }

            // Only strictly-lower entries are dependencies.
            if(col < row)
            {
                const int local_done
                    = rocsparse::spin_loop<SLEEP>(&done_array[col], __HIP_MEMORY_SCOPE_AGENT);
                local_max = rocsparse::max(local_done, local_max);
            }
        }

        rocsparse::wfreduce_max<WF_SIZE>(&local_max);

        __threadfence_block();

        if(lid == WF_SIZE - 1)
        {
            __hip_atomic_store(
                &done_array[row], local_max + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Analysis kernel for an upper triangular ELL matrix. Rows are processed from
    // the last to the first; the dependencies of a row are the strictly-upper
    // entries.
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_analysis_upper_kernel(I m,
                                     I n,
                                     int64_t ell_width,
                                     const I* __restrict__ ell_col_ind,
                                     int* __restrict__ done_array,
                                     rocsparse_index_base idx_base)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const int lid = hipThreadIdx_x & (WF_SIZE - 1);
        const int wid = hipThreadIdx_x / WF_SIZE;

        const I row = (m - 1) - (static_cast<I>(hipBlockIdx_x) * (BLOCKSIZE / WF_SIZE) + wid);

        if(row < 0)
        {
            return;
        }

        int local_max = 0;

        for(int64_t p = lid; p < ell_width; p += WF_SIZE)
        {
            const int64_t idx = p * static_cast<int64_t>(m) + row;
            const I       col = rocsparse::nontemporal_load(ell_col_ind + idx) - idx_base;

            // Skip padded (out-of-range) entries.
            if(col < 0 || col >= n)
            {
                continue;
            }

            // Only strictly-upper entries are dependencies.
            if(col > row)
            {
                const int local_done
                    = rocsparse::spin_loop<SLEEP>(&done_array[col], __HIP_MEMORY_SCOPE_AGENT);
                local_max = rocsparse::max(local_done, local_max);
            }
        }

        rocsparse::wfreduce_max<WF_SIZE>(&local_max);

        __threadfence_block();

        if(lid == WF_SIZE - 1)
        {
            __hip_atomic_store(
                &done_array[row], local_max + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Native ELL triangular solve. Each wavefront solves one row (taken from the
    // analysis row map so that dependencies are scheduled first) and spin-waits on
    // the rows it depends on, exactly like the CSR solve but reading directly from
    // the ELL storage.
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void ellsv_device(I m,
                                           I n,
                                           int64_t ell_width,
                                           T       alpha,
                                           const I* __restrict__ ell_col_ind,
                                           const T* __restrict__ ell_val,
                                           const T* __restrict__ x,
                                           int64_t x_inc,
                                           T*      y,
                                           int64_t y_inc,
                                           int* __restrict__ done_array,
                                           const I* __restrict__ map,
                                           rocsparse_index_base idx_base,
                                           rocsparse_fill_mode  fill_mode,
                                           rocsparse_diag_type  diag_type)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t lid = hipThreadIdx_x & (WF_SIZE - 1);
        int            wid = hipThreadIdx_x / WF_SIZE;

        wid = rocsparse::read_first_lane(wid);

        const I idx = static_cast<I>(hipBlockIdx_x) * (BLOCKSIZE / WF_SIZE) + wid;

        // Shared memory to hold the (reciprocal) diagonal entry of each row.
        __shared__ T diagonal[BLOCKSIZE / WF_SIZE];

        if(idx >= m)
        {
            return;
        }

        // The row this wavefront operates on.
        const I row = map[idx];

        // Default diagonal factor (used for unit diagonal or a missing diagonal).
        if(lid == 0)
        {
            diagonal[wid] = static_cast<T>(1);
        }

        T local_sum = static_cast<T>(0);
        if(lid == 0)
        {
            local_sum = alpha * rocsparse::nontemporal_load(x + x_inc * row);
        }

        for(int64_t p = lid; p < ell_width; p += WF_SIZE)
        {
            const int64_t eidx = p * static_cast<int64_t>(m) + row;
            const I       col  = rocsparse::nontemporal_load(ell_col_ind + eidx) - idx_base;

            // Skip padded (out-of-range) entries.
            if(col < 0 || col >= n)
            {
                continue;
            }

            T local_val = rocsparse::nontemporal_load(ell_val + eidx);

            // Diagonal entry.
            if(col == row)
            {
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    if(local_val == static_cast<T>(0))
                    {
                        local_val = static_cast<T>(1);
                    }
                    diagonal[wid] = static_cast<T>(1) / local_val;
                }

                continue;
            }

            // Only entries on the solved triangular side contribute.
            if(fill_mode == rocsparse_fill_mode_upper)
            {
                if(col < row)
                {
                    continue;
                }
            }
            else
            {
                if(col > row)
                {
                    continue;
                }
            }

            // Spin until the dependency row has been solved.
            (void)rocsparse::spin_loop<SLEEP>(&done_array[col], __HIP_MEMORY_SCOPE_AGENT);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            local_sum = rocsparse::fma(-local_val, y[col * y_inc], local_sum);
        }

        local_sum = rocsparse::wfreduce_sum<WF_SIZE>(local_sum);

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            __threadfence_block();
            local_sum = local_sum * diagonal[wid];
        }

        if(lid == WF_SIZE - 1)
        {
            rocsparse::nontemporal_store(local_sum, &y[row * y_inc]);

            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // Transpose helpers -----------------------------------------------------
    //
    // The transposed operations are handled by materializing the transpose of
    // the ELL matrix as another ELL matrix and solving it with a flipped fill
    // mode. The three kernels below count the entries per column of A (= per row
    // of A^T), then scatter the entries into the transposed storage.

    // Count the number of valid entries per column (one thread per row).
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_transpose_count_kernel(I m,
                                      I n,
                                      int64_t ell_width,
                                      const I* __restrict__ ell_col_ind,
                                      rocsparse_index_base idx_base,
                                      unsigned long long* __restrict__ counts)
    {
        const I row = static_cast<I>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(row >= m)
        {
            return;
        }

        for(int64_t p = 0; p < ell_width; ++p)
        {
            const int64_t idx = p * static_cast<int64_t>(m) + row;
            const I       col = ell_col_ind[idx] - idx_base;
            if(col >= 0 && col < n)
            {
                atomicAdd(&counts[col], 1ull);
            }
        }
    }

    // Initialize an ELL column index array with a padding sentinel.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_fill_col_ind_kernel(int64_t total, I* __restrict__ col_ind, I pad_value)
    {
        const int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(i >= total)
        {
            return;
        }
        col_ind[i] = pad_value;
    }

    // Scatter the entries of A into the transposed ELL storage (one thread per
    // row of A). Entry (row, col) of A becomes entry (col, row) of A^T.
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_transpose_scatter_kernel(I m,
                                        I n,
                                        int64_t ell_width,
                                        const I* __restrict__ ell_col_ind,
                                        const T* __restrict__ ell_val,
                                        rocsparse_index_base idx_base,
                                        bool                 conj,
                                        unsigned long long* __restrict__ positions,
                                        I* __restrict__ t_col_ind,
                                        T* __restrict__ t_val)
    {
        const I row = static_cast<I>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
        if(row >= m)
        {
            return;
        }

        for(int64_t p = 0; p < ell_width; ++p)
        {
            const int64_t idx = p * static_cast<int64_t>(m) + row;
            const I       col = ell_col_ind[idx] - idx_base;
            if(col < 0 || col >= n)
            {
                continue;
            }

            const unsigned long long pos = atomicAdd(&positions[col], 1ull);

            // The transposed matrix keeps the same leading dimension (m).
            const int64_t dest = static_cast<int64_t>(pos) * static_cast<int64_t>(m) + col;

            t_col_ind[dest] = row + idx_base;

            const T v    = ell_val[idx];
            t_val[dest]  = conj ? rocsparse::conj(v) : v;
        }
    }
}
