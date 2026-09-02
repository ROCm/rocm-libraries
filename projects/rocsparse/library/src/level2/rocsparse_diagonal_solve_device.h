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
    // Solves D y = alpha x for a single batch entry: val, x, y and zero_pivot have
    // already been shifted to that entry by the caller. One thread owns one row,
    // and walks the right-hand sides along the y dimension of the grid.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void diagonal_solve_device(J m,
                                                    J nrhs,
                                                    T alpha,
                                                    const I* __restrict__ diag_ind,
                                                    const I* __restrict__ transposed_perm,
                                                    const T* __restrict__ val,
                                                    const T* __restrict__ x,
                                                    int64_t x_row_inc,
                                                    int64_t x_rhs_inc,
                                                    T* __restrict__ y,
                                                    int64_t y_row_inc,
                                                    int64_t y_rhs_inc,
                                                    J* __restrict__ zero_pivot,
                                                    rocsparse_index_base        base,
                                                    rocsparse_diagonal_modifier modifier,
                                                    bool                        conj,
                                                    bool                        conj_x)
    {
        const auto row = static_cast<J>(hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x);
        if(row >= m)
        {
            return;
        }

        // Where A[row,row] sits, as recorded by the analysis stage. It is negative
        // when the row has no diagonal entry at all.
        const I diag_pos = diag_ind[row];

        // A row with a missing or zero diagonal is reported as a pivot and left
        // unscaled, which is what keeping the neutral divisor below amounts to.
        bool pivot = (diag_pos < 0);
        T    denom = static_cast<T>(1);
        if(!pivot)
        {
            // A CSC matrix is analysed through its transpose, so the position in the
            // value array is reached through the permutation the analysis left behind.
            const I val_pos = (transposed_perm != nullptr) ? transposed_perm[diag_pos] : diag_pos;
            const T diag_value = val[val_pos];

            pivot = (diag_value == static_cast<T>(0));
            if(!pivot)
            {
                denom = (modifier == rocsparse_diagonal_modifier_absolute)
                            ? static_cast<T>(rocsparse::abs(diag_value))
                            : (conj ? rocsparse::conj(diag_value) : diag_value);
            }
        }

        // Every block along the right-hand sides sees the same pivot, so only the
        // first one reports it. atomic_min would be idempotent otherwise, but a
        // singular matrix with many right-hand sides would contend for nothing.
        if(pivot && hipBlockIdx_y == 0)
        {
            rocsparse::atomic_min(zero_pivot, static_cast<J>(row + base));
        }

        // Only the diagonal of the matrix was needed, and it has been read above.
        // What is left is dividing every right-hand side of this row by it.
        for(J rhs = hipBlockIdx_y; rhs < nrhs; rhs += hipGridDim_y)
        {
            const T xval = x[row * x_row_inc + rhs * x_rhs_inc];

            y[row * y_row_inc + rhs * y_rhs_inc]
                = (alpha * (conj_x ? rocsparse::conj(xval) : xval)) / denom;
        }
    }
}
