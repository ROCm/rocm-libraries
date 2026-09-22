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

namespace rocsparse
{
    // Both ell_col_ind and ell_val hold the m x ell_width slot array with a
    // leading dimension of m, so entry (row, slot p) lives at
    // ELL_IND(row, p, m, ell_width). Padding entries carry an out-of-range column
    // index. The kernels below scan every slot and skip padded / non-contributing
    // entries, so they do not depend on the entries being sorted within a row.

    // Analysis kernel for a lower triangular ELL matrix. Each wavefront processes
    // a single row and computes its dependency depth (level) by spin-waiting on
    // the depths of the rows it depends on. The depths are written into
    // done_array and later sorted to obtain the row execution order.
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void ellsv_analysis_lower_kernel(I m,
                                     I n,
                                     I ell_width,
                                     const I* __restrict__ ell_col_ind,
                                     int* __restrict__ done_array,
                                     I* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base,
                                     rocsparse_diag_type  diag_type)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t lid = hipThreadIdx_x & (WF_SIZE - 1);
        const uint32_t wid = hipThreadIdx_x / WF_SIZE;

        const int64_t wavefront = static_cast<int64_t>(hipBlockIdx_x) * (BLOCKSIZE / WF_SIZE) + wid;

        if(wavefront >= m)
        {
            return;
        }

        const I row = static_cast<I>(wavefront);

        // Local dependency depth.
        int32_t local_max      = 0;
        int32_t local_has_diag = 0;

        for(I p = lid; p < ell_width; p += WF_SIZE)
        {
            const int64_t idx = ELL_IND(row, static_cast<int64_t>(p), m, ell_width);
            const I       col = rocsparse::nontemporal_load(ell_col_ind + idx) - idx_base;

            // Skip padded (out-of-range) entries.
            if(col < 0 || col >= n)
            {
                continue;
            }

            if(col == row)
            {
                local_has_diag = 1;
            }

            // Only strictly-lower entries are dependencies.
            if(col < row)
            {
                const int32_t local_done
                    = rocsparse::spin_loop<SLEEP>(&done_array[col], __HIP_MEMORY_SCOPE_AGENT);
                local_max = rocsparse::max(local_done, local_max);
            }
        }

        rocsparse::wfreduce_max<WF_SIZE>(&local_max);
        rocsparse::wfreduce_max<WF_SIZE>(&local_has_diag);

        __threadfence_block();

        if(lid == WF_SIZE - 1)
        {
            if(local_has_diag == 0 && diag_type == rocsparse_diag_type_non_unit)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }

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
                                     I ell_width,
                                     const I* __restrict__ ell_col_ind,
                                     int* __restrict__ done_array,
                                     I* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base,
                                     rocsparse_diag_type  diag_type)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t lid = hipThreadIdx_x & (WF_SIZE - 1);
        const uint32_t wid = hipThreadIdx_x / WF_SIZE;

        const int64_t wavefront = static_cast<int64_t>(hipBlockIdx_x) * (BLOCKSIZE / WF_SIZE) + wid;

        if(wavefront >= m)
        {
            return;
        }

        const I row = static_cast<I>(m - 1 - wavefront);

        int32_t local_max      = 0;
        int32_t local_has_diag = 0;

        for(I p = lid; p < ell_width; p += WF_SIZE)
        {
            const int64_t idx = ELL_IND(row, static_cast<int64_t>(p), m, ell_width);
            const I       col = rocsparse::nontemporal_load(ell_col_ind + idx) - idx_base;

            // Skip padded (out-of-range) entries.
            if(col < 0 || col >= n)
            {
                continue;
            }

            if(col == row)
            {
                local_has_diag = 1;
            }

            // Only strictly-upper entries are dependencies.
            if(col > row)
            {
                const int32_t local_done
                    = rocsparse::spin_loop<SLEEP>(&done_array[col], __HIP_MEMORY_SCOPE_AGENT);
                local_max = rocsparse::max(local_done, local_max);
            }
        }

        rocsparse::wfreduce_max<WF_SIZE>(&local_max);
        rocsparse::wfreduce_max<WF_SIZE>(&local_has_diag);

        __threadfence_block();

        if(lid == WF_SIZE - 1)
        {
            if(local_has_diag == 0 && diag_type == rocsparse_diag_type_non_unit)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }

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
                                           I ell_width,
                                           T alpha,
                                           const I* __restrict__ ell_col_ind,
                                           const T* __restrict__ ell_val,
                                           const T* __restrict__ x,
                                           int64_t x_inc,
                                           T*      y,
                                           int64_t y_inc,
                                           int* __restrict__ done_array,
                                           const I* __restrict__ map,
                                           I* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base,
                                           rocsparse_fill_mode  fill_mode,
                                           rocsparse_diag_type  diag_type)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t lid = hipThreadIdx_x & (WF_SIZE - 1);
        const uint32_t wid = rocsparse::read_first_lane(hipThreadIdx_x / WF_SIZE);

        const int64_t wavefront = static_cast<int64_t>(hipBlockIdx_x) * (BLOCKSIZE / WF_SIZE) + wid;

        // Shared memory to hold the (reciprocal) diagonal entry of each row.
        __shared__ T diagonal[BLOCKSIZE / WF_SIZE];

        if(wavefront >= m)
        {
            return;
        }

        // The row this wavefront operates on.
        const I row = map[wavefront];

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

        for(I p = lid; p < ell_width; p += WF_SIZE)
        {
            const int64_t eidx = ELL_IND(row, static_cast<int64_t>(p), m, ell_width);
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
                        rocsparse::atomic_min(zero_pivot, row + idx_base);
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

            // Spin until the dependency row has been solved. Its done flag only
            // ever goes from 0 to 1 here, so the returned value carries no
            // information beyond the wait itself and is discarded.
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
}
