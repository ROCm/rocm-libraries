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
    // The tiled PCR writes each tile's two interface equations into a tridiagonal
    // spike system by swapping neighboring columns. That swap parks the tile-edge
    // off-diagonals on the diagonal of each interior pair and parks the real pivots
    // on the off-diagonals. A diagonal or one-sided bidiagonal system then has exact
    // zeros on the spike diagonal, and the spike PCR divides by them.
    //
    // Swapping the two equations of such a pair puts the pivots back on the diagonal.
    // The swap stays tridiagonal when neither equation reaches outside the pair, and
    // it does not change the unknown order the backward substitution reads. A zero
    // that still has its pivot on the subdiagonal is eliminated against the previous
    // row only when that row has no subdiagonal of its own. Otherwise the subtraction
    // would write a coefficient two columns to the left, and the tridiagonal storage
    // would drop it. The elimination and the swap that follows it run in one
    // ascending sweep, so a cleared subdiagonal unlocks the next elimination in
    // that same sweep. A single zero off-diagonal inside an otherwise tridiagonal
    // system can still leave a spike diagonal at zero; that case is unchanged.
    // Systems whose spike diagonal is already nonzero are left untouched.
    //
    // B holds nrhs right-hand sides, each of length num_spikes, at stride rhs_stride.
    template <typename T>
    ROCSPARSE_DEVICE_ILF void gtsv_nopivot_swap_spike_rows(rocsparse_int i,
                                                           rocsparse_int j,
                                                           T*            dl,
                                                           T*            d,
                                                           T*            du,
                                                           T*            B,
                                                           rocsparse_int nrhs,
                                                           int64_t       rhs_stride)
    {
        const T d_i  = d[i];
        const T du_i = du[i];
        const T dl_j = dl[j];
        const T d_j  = d[j];

        dl[i] = static_cast<T>(0);
        d[i]  = dl_j;
        du[i] = d_j;

        dl[j] = d_i;
        d[j]  = du_i;
        du[j] = static_cast<T>(0);

        for(rocsparse_int rhs = 0; rhs < nrhs; ++rhs)
        {
            T* const column = B + static_cast<int64_t>(rhs) * rhs_stride;
            const T  tmp    = column[i];
            column[i]       = column[j];
            column[j]       = tmp;
        }
    }

    template <typename T>
    ROCSPARSE_DEVICE_ILF void gtsv_nopivot_repair_spike_system(
        rocsparse_int num_spikes, T* dl, T* d, T* du, T* B, rocsparse_int nrhs, int64_t rhs_stride)
    {
        const T zero = static_cast<T>(0);

        bool any_zero = false;
        for(rocsparse_int row = 0; row < num_spikes; ++row)
        {
            if(d[row] == zero)
            {
                any_zero = true;
                break;
            }
        }

        if(!any_zero)
        {
            return;
        }

        // Each step is exact. Swaps need both equations to stay inside the pair,
        // and an elimination needs the previous row to have no subdiagonal, so
        // nothing is written outside the three stored bands. Elimination and the
        // swap after it share one ascending sweep: the swap clears the next
        // row's subdiagonal before that row is visited. A swap puts nonzero
        // values on both diagonals, so the passes cannot cycle.
        for(rocsparse_int pass = 0; pass < num_spikes; ++pass)
        {
            bool progress = false;

            for(rocsparse_int i = 1; i < num_spikes;)
            {
                const rocsparse_int j = i + 1;

                // Interior pairs (1, 2), (3, 4), ... The last spike row keeps the
                // real pivot on its diagonal and is not the left side of a pair.
                if((i % 2) == 1 && j <= num_spikes - 2 && dl[i] == zero && du[j] == zero
                   && (d[i] == zero || d[j] == zero) && dl[j] != zero && du[i] != zero)
                {
                    gtsv_nopivot_swap_spike_rows<T>(i, j, dl, d, du, B, nrhs, rhs_stride);
                    progress = true;
                }

                if(d[i] == zero && dl[i] != zero && d[i - 1] != zero && dl[i - 1] == zero)
                {
                    const T mult = dl[i] / d[i - 1];
                    d[i]         = d[i] - mult * du[i - 1];
                    dl[i]        = zero;
                    for(rocsparse_int rhs = 0; rhs < nrhs; ++rhs)
                    {
                        T* const column = B + static_cast<int64_t>(rhs) * rhs_stride;
                        column[i]       = column[i] - mult * column[i - 1];
                    }
                    progress = true;
                }

                if(j < num_spikes && d[i] == zero && dl[i] == zero && du[i] != zero && du[j] == zero
                   && dl[j] != zero)
                {
                    gtsv_nopivot_swap_spike_rows<T>(i, j, dl, d, du, B, nrhs, rhs_stride);
                    progress = true;
                    i += 2;
                }
                else
                {
                    ++i;
                }
            }

            if(!progress)
            {
                break;
            }

            bool still_zero = false;
            for(rocsparse_int row = 0; row < num_spikes; ++row)
            {
                if(d[row] == zero)
                {
                    still_zero = true;
                    break;
                }
            }
            if(!still_zero)
            {
                break;
            }
        }
    }

    template <typename T>
    ROCSPARSE_KERNEL(1)
    void gtsv_nopivot_spike_repair_kernel(
        rocsparse_int num_spikes, rocsparse_int n, T* dl, T* d, T* du, T* B)
    {
        rocsparse::gtsv_nopivot_repair_spike_system<T>(num_spikes, dl, d, du, B, n, num_spikes);
    }

    // One spike system per batch, stored at stride num_spikes.
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_spike_repair_kernel(
        rocsparse_int num_spikes, rocsparse_int batch_count, T* dl, T* d, T* du, T* B)
    {
        const int64_t stride = num_spikes;
        const int64_t step   = static_cast<int64_t>(hipGridDim_x) * hipBlockDim_x;

        for(int64_t batch = static_cast<int64_t>(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;
            batch < batch_count;
            batch += step)
        {
            const int64_t offset = batch * stride;
            rocsparse::gtsv_nopivot_repair_spike_system<T>(
                num_spikes, dl + offset, d + offset, du + offset, B + offset, 1, 1);
        }
    }
}
