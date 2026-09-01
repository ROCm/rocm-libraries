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
    // Applies the inverse of the diagonal to one row of one batch entry: val, x, y
    // and zero_pivot have already been shifted to that batch entry by the caller.
    // The right-hand sides are walked from col_first with a stride of col_inc so
    // that the caller decides how they map onto the grid.
    template <typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void diagonal_solve_device(J row,
                                                    J nrhs,
                                                    J col_first,
                                                    J col_inc,
                                                    T alpha,
                                                    const I* __restrict__ diag_ind,
                                                    const I* __restrict__ transposed_perm,
                                                    const T* __restrict__ val,
                                                    const T* __restrict__ x,
                                                    int64_t x_row_stride,
                                                    int64_t x_col_stride,
                                                    T* __restrict__ y,
                                                    int64_t y_row_stride,
                                                    int64_t y_col_stride,
                                                    J* __restrict__ zero_pivot,
                                                    rocsparse_index_base        base,
                                                    rocsparse_diagonal_modifier modifier,
                                                    bool                        conj,
                                                    bool                        conj_x)
    {
        const I p = diag_ind[row];

        // A row with a missing or zero diagonal is reported as a pivot and left
        // unscaled, which is what keeping the neutral divisor below amounts to.
        bool pivot = (p < 0);
        T    denom = static_cast<T>(1);
        if(!pivot)
        {
            const I q = (transposed_perm != nullptr) ? transposed_perm[p] : p;
            const T d = val[q];

            pivot = (d == static_cast<T>(0));
            if(!pivot)
            {
                denom = (modifier == rocsparse_diagonal_modifier_absolute)
                            ? static_cast<T>(rocsparse::abs(d))
                            : (conj ? rocsparse::conj(d) : d);
            }
        }

        if(pivot)
        {
            rocsparse::atomic_min(zero_pivot, static_cast<J>(row + base));
        }

        for(J col = col_first; col < nrhs; col += col_inc)
        {
            const T xval = x[row * x_row_stride + col * x_col_stride];

            y[row * y_row_stride + col * y_col_stride]
                = (alpha * (conj_x ? rocsparse::conj(xval) : xval)) / denom;
        }
    }
}
