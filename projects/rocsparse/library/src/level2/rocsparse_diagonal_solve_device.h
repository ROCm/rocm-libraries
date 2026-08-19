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
#include "rocsparse_scalar.hpp"

namespace rocsparse
{
    // Unified diagonal backsolve: Y = alpha * op(X) / D (or / |D|) for one or many
    // right-hand sides, in a single launch and independent of the matrix format.
    //
    // The diagonal is seen as a sparse vector: \p diag_ind gives the per-row offset of
    // the diagonal entry into the shared \p val array (produced by the triangular-solve
    // analysis, rocsparse::trm_info_t::diag_ind), so d_row = val[diag_ind[row]].
    //
    // The RHS is a dense block of \p nrhs columns addressed by generic strides
    // (row/col/batch), so a plain vector (sptrsv) is just nrhs == 1 with a col stride
    // of 0, and a dense matrix (sptrsm) uses its leading dimension / order (and the
    // conjugation of op(X) via \p conj_x). One thread owns a matrix row and walks its
    // columns with a grid-stride loop on the y grid dimension.
    //
    // rocsparse_diagonal_mode_absolute divides by |d| so that L|D|Lᴴ is HPD. The
    // diagonal is its own transpose, so only the conjugate transpose (\p conj)
    // conjugates d. A missing (offset < 0) or numerically zero diagonal is reported
    // once per row through \p zero_pivot.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void diagonal_solve_kernel(J m,
                               J nrhs,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                               const I* __restrict__ diag_ind,
                               const I* __restrict__ transposed_perm,
                               const T* __restrict__ val,
                               int64_t val_batch_stride,
                               const T* __restrict__ x,
                               int64_t x_row_stride,
                               int64_t x_col_stride,
                               int64_t x_batch_stride,
                               T* __restrict__ y,
                               int64_t y_row_stride,
                               int64_t y_col_stride,
                               int64_t y_batch_stride,
                               J* __restrict__ zero_pivot,
                               int64_t                 zero_pivot_stride,
                               rocsparse_index_base    base,
                               rocsparse_diagonal_mode diagonal_mode,
                               bool                    conj,
                               bool                    conj_x,
                               bool                    is_host_mode)
    {
        const J row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(row >= m)
        {
            return;
        }

        const uint32_t batch = hipBlockIdx_z;
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        // Resolve the diagonal value once per row (shared across all columns).
        const I p     = diag_ind[row];
        bool    pivot = (p < 0);
        T       d     = static_cast<T>(1);
        if(!pivot)
        {
            // When the analysis ran on a transposed structure (CSR transpose /
            // conjugate-transpose, or any CSC), diag_ind is an offset into that
            // transposed layout. Map it back to the shared val array through the
            // analysis permutation; a null perm means diag_ind already indexes val.
            const I q = (transposed_perm != nullptr) ? transposed_perm[p] : p;
            d         = val[batch * val_batch_stride + q];
            if(d == static_cast<T>(0))
            {
                pivot = true;
                d     = static_cast<T>(1);
            }
        }

        // Report a missing / numerically zero diagonal once per row.
        if(hipBlockIdx_y == 0 && pivot)
        {
            rocsparse::atomic_min(zero_pivot + batch * zero_pivot_stride,
                                  static_cast<J>(row + base));
        }

        const T denom = pivot ? static_cast<T>(1)
                              : ((diagonal_mode == rocsparse_diagonal_mode_absolute)
                                     ? static_cast<T>(rocsparse::abs(d))
                                     : (conj ? rocsparse::conj(d) : d));

        for(J col = hipBlockIdx_y; col < nrhs; col += hipGridDim_y)
        {
            T xval = x[batch * x_batch_stride + row * x_row_stride + col * x_col_stride];
            // For a conjugate-transpose RHS (sptrsm), op(X) conjugates the entries.
            if(conj_x)
            {
                xval = rocsparse::conj(xval);
            }
            const T xv = alpha * xval;

            y[batch * y_batch_stride + row * y_row_stride + col * y_col_stride]
                = pivot ? xv : (xv / denom);
        }
    }
}
