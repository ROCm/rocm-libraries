/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/generate.hpp"
#include "common/misc/rocsolver_arguments.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// Bad-argument tests: gerand should throw rocblas_status_invalid_size when
// m < 0, n < 0, or lda < m, and rocblas_status_invalid_pointer when A is null.
// ---------------------------------------------------------------------------
template <typename T>
void gerand_checkBadArgs()
{
    const rocblas_int m = 4;
    const rocblas_int n = 4;
    const rocblas_int lda = 4;
    std::vector<T> A(lda * n);

    // m < 0
    EXPECT_THROW(gerand(-1, n, A.data(), lda), rocblas_status);

    // n < 0
    EXPECT_THROW(gerand(m, -1, A.data(), lda), rocblas_status);

    // lda < m
    EXPECT_THROW(gerand(m, n, A.data(), m - 1), rocblas_status);

    // null pointer
    EXPECT_THROW(gerand(m, n, (T*)nullptr, lda), rocblas_status);
}

// ---------------------------------------------------------------------------
// Correctness test: verify all entries of A lie in (-1, 1) for real types,
// and that both real and imaginary parts lie in (-1, 1) for complex types.
// Also verify the leading-dimension stride: entries A[i + j*lda] for
// i in [0, m) and j in [0, n) must be filled, while the padding rows
// A[m .. lda-1 + j*lda] are untouched (remain zero).
// ---------------------------------------------------------------------------
template <typename T>
void testing_gerand(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);

    // skip invalid sizes (those are exercised in bad-arg tests)
    if(m < 0 || n < 0 || lda < m)
        return;

    // zero-initialize so padding rows are detectable
    std::vector<T> A(static_cast<size_t>(lda) * n, T(0));

    gerand(m, n, A.data(), lda);

    // quick return: nothing to check when m == 0 or n == 0
    if(m == 0 || n == 0)
        return;

    for(rocblas_int j = 0; j < n; ++j)
    {
        for(rocblas_int i = 0; i < m; ++i)
        {
            T val = A[i + j * lda];
            S re = std::real(val);
            S im = std::imag(val);

            EXPECT_GT(re, S(-1)) << "re out of range at (" << i << "," << j << ")";
            EXPECT_LT(re, S( 1)) << "re out of range at (" << i << "," << j << ")";

            if constexpr(rocblas_is_complex<T>)
            {
                EXPECT_GT(im, S(-1)) << "im out of range at (" << i << "," << j << ")";
                EXPECT_LT(im, S( 1)) << "im out of range at (" << i << "," << j << ")";
            }
        }

        // padding rows (lda > m) must be untouched
        for(rocblas_int i = m; i < lda; ++i)
        {
            EXPECT_EQ(A[i + j * lda], T(0)) << "padding modified at (" << i << "," << j << ")";
        }
    }

    argus.validate_consumed();
}
