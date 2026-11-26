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
#ifndef ROCSPARSE_CHECK_CHOLESKY_HPP
#define ROCSPARSE_CHECK_CHOLESKY_HPP

#include "rocsparse_math.hpp"
#include "rocsparse_traits.hpp"

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Check that the product of the IC factors L*L^H matches the original matrix A on the strictly
// lower part of the pattern of A. The diagonal is skipped: it is sqrt(|d|), which breaks the
// identity when A is not positive definite. `label` names the factor in the error message.
template <typename T, typename I, typename J>
void check_csr_cholesky(J                    M,
                        const I*             csr_row_ptr,
                        const J*             csr_col_ind,
                        const T*             A,
                        const T*             L,
                        rocsparse_index_base base,
                        const char*          label)
{
    using R          = floating_data_t<T>;
    const R eps      = std::numeric_limits<R>::epsilon();
    int64_t failures = 0;

    std::vector<I> diag(M, -1);
    for(J i = 0; i < M; ++i)
        for(I p = csr_row_ptr[i] - base; p < csr_row_ptr[i + 1] - base; ++p)
            if(csr_col_ind[p] - base == i)
                diag[i] = p;

    std::vector<I> pos(M, -1);
    for(J i = 0; i < M; ++i)
    {
        const I begin = csr_row_ptr[i] - base;
        const I end   = csr_row_ptr[i + 1] - base;
        for(I p = begin; p < end; ++p)
            pos[csr_col_ind[p] - base] = p;

        for(I p = begin; p < end; ++p)
        {
            const J j = csr_col_ind[p] - base;
            if(j >= i || diag[j] == -1)
                continue;

            T       s     = L[p] * rocsparse_conj(L[diag[j]]);
            R       bound = rocsparse_abs(A[p]) + rocsparse_abs(s);
            int64_t terms = 1;
            // Columns are sorted, so the entries of row j before its diagonal are its k < j.
            for(I q = csr_row_ptr[j] - base; q < diag[j]; ++q)
            {
                const J k = csr_col_ind[q] - base;
                if(pos[k] != -1)
                {
                    const T t = L[pos[k]] * rocsparse_conj(L[q]);
                    s         = s + t;
                    bound += rocsparse_abs(t);
                    ++terms;
                }
            }

            // Backward error bound of a sum of `terms` products, (terms + 1) * eps * sum |t|,
            // with a margin of 10.
            const R err = rocsparse_abs(s - A[p]);
            const R tol = 10 * (terms + 1) * eps * bound;
            if(!(err <= tol))
            {
                if(failures == 0)
                {
                    std::cerr << "IC(0) (" << label << "): (L * L^H)(" << i << ", " << j
                              << ") = " << s << " differs from A(" << i << ", " << j
                              << ") = " << A[p] << " by " << err << " > " << tol << " (" << terms
                              << " terms, bound " << bound << ")" << std::endl;
                }
                ++failures;
            }
        }

        for(I p = begin; p < end; ++p)
            pos[csr_col_ind[p] - base] = -1;
    }

    if(failures != 0)
    {
        std::cerr << "IC(0) (" << label << "): " << failures << " entries of L * L^H differ from A"
                  << std::endl;
#ifdef GOOGLE_TEST
        FAIL();
#else
        exit(EXIT_FAILURE);
#endif
    }
}

#endif // ROCSPARSE_CHECK_CHOLESKY_HPP
