/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

namespace rocsparse
{
    // ELL SpMV for general, non-transposed matrices
    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void ellmvn_device(I                    m,
                                            I                    n,
                                            I                    ell_width,
                                            T                    alpha,
                                            const I*             ell_col_ind,
                                            const A*             ell_val,
                                            const X*             x,
                                            T                    beta,
                                            Y*                   y,
                                            rocsparse_index_base idx_base)
    {
        const I ai = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        T sum = static_cast<T>(0);
        for(I p = 0; p < ell_width; ++p)
        {

            const int64_t idx = ELL_IND(ai, (int64_t)p, m, ell_width);
            const I       col = rocsparse::nontemporal_load(ell_col_ind + idx) - idx_base;
            if(col >= 0 && col < n)
            {
                sum = rocsparse::fma<T>(
                    rocsparse::nontemporal_load(ell_val + idx), rocsparse::ldg(x + col), sum);
            }
            else
            {
                break;
            }
        }

        if(beta != static_cast<T>(0))
        {
            const Y yv = rocsparse::nontemporal_load(y + ai);
            rocsparse::nontemporal_store(rocsparse::fma<T>(beta, yv, alpha * sum), y + ai);
        }
        else
        {
            rocsparse::nontemporal_store(alpha * sum, y + ai);
        }
    }

    // ELL SpMV for general, (conjugate) transposed matrices
    template <uint32_t BLOCKSIZE, typename I, typename A, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void ellmvt_device(rocsparse_operation  trans,
                                            I                    m,
                                            I                    n,
                                            I                    ell_width,
                                            T                    alpha,
                                            const I*             ell_col_ind,
                                            const A*             ell_val,
                                            const X*             x,
                                            Y*                   y,
                                            rocsparse_index_base idx_base)
    {
        const I ai = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

        if(ai >= m)
        {
            return;
        }

        const T row_val = alpha * rocsparse::ldg(x + ai);

        for(I p = 0; p < ell_width; ++p)
        {
            const int64_t idx = ELL_IND(ai, (int64_t)p, m, ell_width);
            const I       col = rocsparse::nontemporal_load(ell_col_ind + idx) - idx_base;

            if(col >= 0 && col < n)
            {
                A val = rocsparse::nontemporal_load(ell_val + idx);

                if(trans == rocsparse_operation_conjugate_transpose)
                {
                    val = rocsparse::conj(val);
                }

                rocsparse::atomic_add(&y[col], row_val * val);
            }
            else
            {
                break;
            }
        }
    }
}
