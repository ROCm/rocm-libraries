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

#include "common/misc/clientcommon.hpp"
#include "common/misc/generate.hpp"
#include "common/misc/rocblas_test.hpp"
#include "common/misc/rocsolver_arguments.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

//------------------------------------------------------------------------------
// Bad-argument tests: syrand should throw rocblas_status_invalid_pointer when
// A is null (with n > 0), but not throw for quick return (n == 0).
//------------------------------------------------------------------------------
template <typename T>
void testing_syrand_bad_arg()
{
    const rocblas_int n = 4;
    const rocblas_int lda = 4;

    // null pointer with n > 0
    EXPECT_THROW_VALUE(syrand(rocblas_fill_lower, n, (T*)nullptr, lda),
                       rocblas_status,
                       rocblas_status_invalid_pointer);
    EXPECT_THROW_VALUE(syrand(rocblas_fill_upper, n, (T*)nullptr, lda),
                       rocblas_status,
                       rocblas_status_invalid_pointer);
    EXPECT_THROW_VALUE(syrand(rocblas_fill_full, n, (T*)nullptr, lda),
                       rocblas_status,
                       rocblas_status_invalid_pointer);

    // quick return: n == 0 with null pointer should not throw
    EXPECT_NO_THROW(syrand(rocblas_fill_lower, 0, (T*)nullptr, lda));
    EXPECT_NO_THROW(syrand(rocblas_fill_upper, 0, (T*)nullptr, lda));
    EXPECT_NO_THROW(syrand(rocblas_fill_full,  0, (T*)nullptr, lda));
}

//------------------------------------------------------------------------------
// Correctness test for syrand.
//
// Checks:
//   - Filled entries lie in (-1, 1) for real and imaginary parts.
//   - For uplo == lower: only lower triangle is written; upper (excl. diag)
//     and padding are untouched (== flag).
//   - For uplo == upper: only upper triangle is written; lower (excl. diag)
//     and padding are untouched (== flag).
//   - For uplo == full: upper triangle equals (non-conjugated) transpose of
//     lower, i.e., A[i,j] == A[j,i] for i < j; padding untouched.
//------------------------------------------------------------------------------
template <typename T>
void testing_syrand(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    rocblas_int n     = argus.get<rocblas_int>("n");
    rocblas_int pad   = 2;
    rocblas_int lda   = argus.get<rocblas_int>("lda", n + pad);
    char        uploC = argus.get<char>("uplo");

    rocblas_fill uplo = char2rocblas_fill(uploC);

    // check invalid sizes
    if(n < 0 || lda < n)
    {
        EXPECT_THROW_VALUE(syrand(uplo, n, (T*)nullptr, lda),
                           rocblas_status,
                           rocblas_status_invalid_size);
        return;
    }

    // Allocate with extra padding columns. Initialize all to flag so we can
    // detect which entries were written.
    rocblas_int n_padded = n + pad;
    T const flag = T(-1234);
    std::vector<T> A(static_cast<size_t>(lda) * n_padded, flag);

    syrand(uplo, n, A.data(), lda);

    // Verify filled entries and structure.
    for(rocblas_int j = 0; j < n; ++j)
    {
        for(rocblas_int i = 0; i < n; ++i)
        {
            T val = A[i + j * lda];
            S re  = std::real(val);
            S im  = std::imag(val);

            auto expect_in_range = [&](S re_, S im_, const char* tri) {
                EXPECT_GT(re_, S(-1)) << tri << " re out of range at (" << i << "," << j << ")";
                EXPECT_LT(re_, S( 1)) << tri << " re out of range at (" << i << "," << j << ")";
                if constexpr(rocblas_is_complex<T>)
                {
                    EXPECT_GT(im_, S(-1)) << tri << " im out of range at (" << i << "," << j << ")";
                    EXPECT_LT(im_, S( 1)) << tri << " im out of range at (" << i << "," << j << ")";
                }
            };

            if(i == j) // diagonal: always filled by all uplo variants
            {
                expect_in_range(re, im, "diag");
            }
            else if(i > j) // strictly lower triangle
            {
                if(uplo == rocblas_fill_lower || uplo == rocblas_fill_full)
                    expect_in_range(re, im, "lower");
                else // uplo == upper: strictly lower untouched
                    EXPECT_EQ(val, flag) << "lower modified at (" << i << "," << j << ")";
            }
            else // i < j: strictly upper triangle
            {
                if(uplo == rocblas_fill_upper)
                {
                    expect_in_range(re, im, "upper");
                }
                else if(uplo == rocblas_fill_full)
                {
                    // Upper must equal (non-conjugate) transpose of lower: A[i,j] == A[j,i].
                    EXPECT_EQ(val, A[j + i * lda])
                        << "upper not transpose of lower at (" << i << "," << j << ")";
                }
                else // uplo == lower: strictly upper untouched
                {
                    EXPECT_EQ(val, flag) << "upper modified at (" << i << "," << j << ")";
                }
            }
        }

        // padding rows must be untouched
        for(rocblas_int i = n; i < lda; ++i)
        {
            EXPECT_EQ(A[i + j * lda], flag) << "padding row modified at (" << i << "," << j << ")";
        }
    }

    // padding cols must be untouched
    for(rocblas_int j = n; j < n_padded; ++j)
    {
        for(rocblas_int i = 0; i < lda; ++i)
        {
            EXPECT_EQ(A[i + j * lda], flag) << "padding col modified at (" << i << "," << j << ")";
        }
    }

    argus.validate_consumed();
}
