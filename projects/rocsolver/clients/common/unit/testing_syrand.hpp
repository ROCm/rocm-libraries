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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/generate.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

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

    // pointers
    EXPECT_THROW_VALUE(syrand(rocblas_fill_lower, n, (T*)nullptr, lda), rocblas_status,
                       rocblas_status_invalid_pointer);
    EXPECT_THROW_VALUE(syrand(rocblas_fill_upper, n, (T*)nullptr, lda), rocblas_status,
                       rocblas_status_invalid_pointer);
    EXPECT_THROW_VALUE(syrand(rocblas_fill_full, n, (T*)nullptr, lda), rocblas_status,
                       rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_NO_THROW(syrand(rocblas_fill_lower, 0, (T*)nullptr, lda));
    EXPECT_NO_THROW(syrand(rocblas_fill_upper, 0, (T*)nullptr, lda));
    EXPECT_NO_THROW(syrand(rocblas_fill_full, 0, (T*)nullptr, lda));
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
//   - For uplo == full: upper triangle equals transpose of lower,
//     i.e., A[i,j] == A[j,i] for i < j; padding untouched.
//   - Few, if any, zero entries.
//------------------------------------------------------------------------------
template <typename T>
void testing_syrand(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int pad = 2;
    rocblas_int lda = argus.get<rocblas_int>("lda", n + pad);
    rocblas_int n_padded = n + pad;
    char uploC = argus.get<char>("uplo");
    rocblas_fill uplo = char2rocblas_fill(uploC);

    // determine sizes
    size_t size_A = size_t(lda) * n_padded;
    double cpu_time_used = 0;

    // check invalid sizes
    if(n < 0 || lda < n)
    {
        EXPECT_THROW_VALUE(syrand(uplo, n, (T*)nullptr, lda), rocblas_status,
                           rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // no memory size query

    // memory allocations
    // Initialize entries to flag to detect which entries were written.
    T const flag = T(-1234);
    std::vector<T> A(size_A, flag);

    // check computations
    // collect CPU performance data
    // CPU-only, no GPU performance data to collect
    cpu_time_used = get_time_us_no_sync();
    syrand(uplo, n, A.data(), lda);
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    // validate results for rocsolver-test
    // Count zeros to check randomness: re and im zeros should each be < 1% of
    // the n*(n+1)/2 independently-filled entries (lower/diag triangle).
    int64_t nfilled = int64_t(n) * (n + 1) / 2;
    int64_t nzero_re = 0, nzero_im = 0;

    // Lambda to check that re and im are in range [-1, 1] and count zero entries.
    auto expect_in_range = [&](S re_, S im_, rocblas_int i_, rocblas_int j_, const char* loc) {
        EXPECT_GE(re_, S(-1)) << loc << " re out of range at (" << i_ << "," << j_ << ")";
        EXPECT_LE(re_, S(1)) << loc << " re out of range at (" << i_ << "," << j_ << ")";
        if(re_ == S(0))
            ++nzero_re;
        if constexpr(rocblas_is_complex<T>)
        {
            EXPECT_GE(im_, S(-1)) << loc << " im out of range at (" << i_ << "," << j_ << ")";
            EXPECT_LE(im_, S(1)) << loc << " im out of range at (" << i_ << "," << j_ << ")";
            if(im_ == S(0))
                ++nzero_im;
        }
    };

    for(rocblas_int j = 0; j < n; ++j)
    {
        for(rocblas_int i = 0; i < n; ++i)
        {
            T val = A[i + j * lda];
            S re = std::real(val);
            S im = std::imag(val);

            if(i == j) // diagonal: always filled by all uplo variants
            {
                expect_in_range(re, im, i, j, "diag");
            }
            else if(i > j) // strictly lower triangle
            {
                if(uplo == rocblas_fill_lower || uplo == rocblas_fill_full)
                    expect_in_range(re, im, i, j, "lower");
                else // uplo == upper: strictly lower untouched
                    EXPECT_EQ(val, flag) << "lower modified at (" << i << "," << j << ")";
            }
            else // i < j: strictly upper triangle
            {
                if(uplo == rocblas_fill_upper)
                {
                    expect_in_range(re, im, i, j, "upper");
                }
                else if(uplo == rocblas_fill_full)
                {
                    // Upper must equal transpose of lower: A[i,j] == A[j,i].
                    EXPECT_EQ(val, A[j + i * lda])
                        << "upper not transpose of lower at (" << i << "," << j << ")";
                }
                else // uplo == lower: strictly upper untouched
                {
                    EXPECT_EQ(val, flag) << "upper modified at (" << i << "," << j << ")";
                }
            }
        }

        // padding rows (i >= n) must be untouched
        for(rocblas_int i = n; i < lda; ++i)
        {
            EXPECT_EQ(A[i + j * lda], flag) << "padding row modified at (" << i << "," << j << ")";
        }
    }

    // padding cols (j >= n) must be untouched
    for(rocblas_int j = n; j < n_padded; ++j)
    {
        for(rocblas_int i = 0; i < lda; ++i)
        {
            EXPECT_EQ(A[i + j * lda], flag) << "padding col modified at (" << i << "," << j << ")";
        }
    }

    // If any, number of zeros should be << 1%.
    EXPECT_LE(nzero_re, int64_t(0.01 * nfilled));
    EXPECT_LE(nzero_im, int64_t(0.01 * nfilled));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        rocsolver_bench_header("Arguments:");
        rocsolver_bench_output("uplo", "n", "lda");
        rocsolver_bench_output(uploC, n, lda);

        rocsolver_bench_header("Results:");
        rocsolver_bench_output("cpu_time_us");
        rocsolver_bench_output(cpu_time_used);
        rocsolver_bench_endl();
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_SYRAND(...) extern template void testing_syrand<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYRAND, FOREACH_SCALAR_TYPE, APPLY_STAMP)
