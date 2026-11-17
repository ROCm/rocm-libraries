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

#include <cmath>
#include <complex>
#include <vector>

//------------------------------------------------------------------------------
// Bad-argument tests: hbrand should throw rocblas_status_invalid_pointer when
// Aband is null (with n > 0), but not throw for quick return (n == 0).
//------------------------------------------------------------------------------
template <typename T>
void testing_hbrand_bad_arg()
{
    const rocblas_int n = 4;
    const rocblas_int kl = 2;
    const rocblas_int ku = 2;
    const rocblas_int ldab = kl + ku + 1;

    // pointers
    EXPECT_THROW_VALUE(hbrand(n, kl, ku, (T*)nullptr, ldab), rocblas_status,
                       rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_NO_THROW(hbrand(0, kl, ku, (T*)nullptr, ldab));
}

//------------------------------------------------------------------------------
// Correctness test for hbrand.
//
// hbrand stores a Hermitian band matrix in band storage Aband[ldab x n],
// where idiag = ku is the row index of the main diagonal.
// Element A[i,j] of the full matrix maps to Aband[ku + (i-j) + j*ldab].
//
// Checks:
//   - Invalid sizes throw rocblas_status_invalid_size.
//   - Diagonal entries (Aband[ku + j*ldab]) are real and in (-1, 1).
//   - Off-diagonal band entries in (-1, 1) (real and imaginary parts).
//   - Copied entries satisfy Aband[ku-k + (j+k)*ldab] == sconj(Aband[ku+k + j*ldab])
//     for the min(kl,ku)-1 diagonals that are symmetrized.
//   - Out-of-band entries in Aband are nan.
//   - Zero count for independently-filled entries is < 1%.
//------------------------------------------------------------------------------
template <typename T>
void testing_hbrand(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int kl = argus.get<rocblas_int>("kl");
    rocblas_int ku = argus.get<rocblas_int>("ku");
    rocblas_int pad = 2;
    rocblas_int ldab = argus.get<rocblas_int>("ldab", kl + ku + 1 + pad);
    rocblas_int n_padded = n + pad;

    // determine sizes
    size_t size_Aband = size_t(ldab) * n_padded;
    double cpu_time_used = 0;

    // check invalid sizes
    if(n < 0 || kl < 0 || ku < 0 || ldab < kl + ku + 1)
    {
        EXPECT_THROW_VALUE(hbrand(n, kl, ku, (T*)nullptr, ldab), rocblas_status,
                           rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // no memory size query

    // memory allocations
    // Initialize entries to flag to detect which entries were written.
    T const flag = T(-1234);
    std::vector<T> Aband(size_Aband, flag);

    // check computations
    // collect CPU performance data
    // CPU-only, no GPU performance data to collect
    cpu_time_used = get_time_us_no_sync();
    hbrand(n, kl, ku, Aband.data(), ldab);
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    // Main diagonal row index in band storage.
    rocblas_int idiag = ku;

    // validate results for rocsolver-test
    // Count zeros to check randomness: re and im zeros should each be < 1% of
    // the n*(n+1)/2 independently-filled entries (lower/diag triangle).
    // For herand the diagonal im is forced real (always 0 for complex), so only
    // off-diagonal entries contribute to nzero_im.
    int64_t nfilled = 0;
    int64_t nzero_re = 0, nzero_im = 0;

    // Lambda to check that re and im are in range [-1, 1] and count zero entries.
    auto expect_in_range = [&](S re_, S im_, const char* loc) {
        EXPECT_GE(re_, S(-1)) << loc << " re out of range";
        EXPECT_LE(re_, S(1)) << loc << " re out of range";
        ++nfilled;
        if(re_ == S(0))
            ++nzero_re;
        if constexpr(rocblas_is_complex<T>)
        {
            EXPECT_GE(im_, S(-1)) << loc << " im out of range";
            EXPECT_LE(im_, S(1)) << loc << " im out of range";
            if(im_ == S(0))
                ++nzero_im;
        }
    };

    for(rocblas_int j = 0; j < n; ++j)
    {
        // --- diagonal entry: must be real and in (-1, 1) ---
        {
            T val = Aband[idiag + j * ldab];
            S re = std::real(val);
            S im = std::imag(val);

            EXPECT_GE(re, S(-1)) << "diag re out of range at j=" << j;
            EXPECT_LE(re, S(1)) << "diag re out of range at j=" << j;
            ++nfilled;
            if(re == S(0))
                ++nzero_re;
            if constexpr(rocblas_is_complex<T>)
            {
                // Diagonal is forced real.
                EXPECT_EQ(im, S(0)) << "diag not real at j=" << j;
            }
        }

        // --- off-diagonal band entries ---
        // When kl >= ku: lower band (k=1..kl) is randomly filled; upper band
        //   (k=1..ku-1) is conjugate-copied from lower; k=ku..kl lower only.
        // When kl < ku: upper band (k=1..ku) is randomly filled; lower band
        //   (k=1..kl-1) is conjugate-copied from upper; k=kl..ku upper only.
        //
        // k > 0 denotes a subdiagonal; element A[j+k, j] is in lower band,
        // stored at Aband[idiag+k + j*ldab].
        // Element A[j, j+k] is in upper band,
        // stored at Aband[idiag-k + (j+k)*ldab].

        rocblas_int nsub = std::min(kl, n - 1 - j); // available subdiagonals from col j
        rocblas_int nsup = std::min(ku, n - 1 - j); // available superdiagonals from col j

        if(kl >= ku)
        {
            // Lower band is the source; verify it as randomly filled.
            for(rocblas_int k = 1; k <= nsub; ++k)
            {
                T val = Aband[idiag + k + j * ldab];
                S re = std::real(val);
                S im = std::imag(val);
                expect_in_range(re, im, "lower band");
            }

            // Upper band (k=1..min(ku-1, nsup)) is a conjugate copy of lower.
            for(rocblas_int k = 1; k < ku && k <= nsup; ++k)
            {
                T lower = Aband[idiag + k + j * ldab];
                T upper = Aband[idiag - k + (j + k) * ldab];
                EXPECT_EQ(upper, sconj(lower))
                    << "upper not conj of lower at subdiag k=" << k << " col j=" << j;
            }
        }
        else // kl < ku
        {
            // Upper band is the source; verify it as randomly filled.
            for(rocblas_int k = 1; k <= nsup; ++k)
            {
                T val = Aband[idiag - k + (j + k) * ldab];
                S re = std::real(val);
                S im = std::imag(val);
                expect_in_range(re, im, "upper band");
            }

            // Lower band (k=1..min(kl-1, nsub)) is a conjugate copy of upper.
            for(rocblas_int k = 1; k < kl && k <= nsub; ++k)
            {
                T upper = Aband[idiag - k + (j + k) * ldab];
                T lower = Aband[idiag + k + j * ldab];
                EXPECT_EQ(lower, sconj(upper))
                    << "lower not conj of upper at subdiag k=" << k << " col j=" << j;
            }
        }

        // --- out-of-band entries must be nan ---
        // Upper out-of-band: rows 0..(ku-j-1) of column j when j < ku.
        if(j < ku)
        {
            for(rocblas_int row = 0; row < ku - j; ++row)
            {
                EXPECT_TRUE(std::isnan(std::real(Aband[row + j * ldab])))
                    << "upper out-of-band not nan at row=" << row << " col j=" << j;
            }
        }
        // Lower out-of-band: rows (idiag+kl-j+1)..(ldab-1) ... handled by the
        // lower nan-fill loop which works from the right. Check via the
        // right-most columns.
        if(j > n - 1 - kl)
        {
            rocblas_int j_from_end = n - 1 - j; // 0-based distance from last col
            for(rocblas_int k = j_from_end; k < kl; ++k)
            {
                EXPECT_TRUE(std::isnan(std::real(Aband[idiag + 1 + k + j * ldab])))
                    << "lower out-of-band not nan at subdiag k+1=" << k + 1 << " col j=" << j;
            }
        }
    }

    // padding cols (j >= n) must be untouched
    for(rocblas_int j = n; j < n_padded; ++j)
    {
        for(rocblas_int i = 0; i < ldab; ++i)
        {
            EXPECT_EQ(Aband[i + j * ldab], flag)
                << "padding col modified at (" << i << "," << j << ")";
        }
    }

    // If any, number of zeros should be << 1%.
    EXPECT_LE(nzero_re, int64_t(0.01 * nfilled));
    EXPECT_LE(nzero_im, int64_t(0.01 * nfilled));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        rocsolver_bench_header("Arguments:");
        rocsolver_bench_output("n", "kl", "ku", "ldab");
        rocsolver_bench_output(n, kl, ku, ldab);

        rocsolver_bench_header("Results:");
        rocsolver_bench_output("cpu_time_us");
        rocsolver_bench_output(cpu_time_used);
        rocsolver_bench_endl();
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_HBRAND(...) extern template void testing_hbrand<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_HBRAND, FOREACH_SCALAR_TYPE, APPLY_STAMP)
