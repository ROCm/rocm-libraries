/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, bool SYTRF, typename T, typename U>
void sytf2_sytrf_checkBadArgs(const rocblas_handle handle,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              U dinfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, nullptr, uplo, n, dA, lda, stA,
                                                dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, rocblas_fill_full, n, dA,
                                                lda, stA, dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                    dIpiv, stP, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T) nullptr, lda,
                                                stA, dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                (U) nullptr, stP, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                dIpiv, stP, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, 0, (T) nullptr, lda,
                                                stA, (U) nullptr, stP, dinfo, bc),
                          rocblas_status_success);
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                    dIpiv, stP, (U) nullptr, 0),
                              rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                    dIpiv, stP, dinfo, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool SYTRF, typename T>
void testing_sytf2_sytrf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sytf2_sytrf_checkBadArgs<STRIDED, SYTRF>(handle, uplo, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sytf2_sytrf_checkBadArgs<STRIDED, SYTRF>(handle, uplo, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
}

/** SYTF2_SYTRF_NAN_PIVOT returns the diagonal entry of matrix b of the batch that is set to NaN
    with singular = 2, or -1: an interior entry (n / 2) in the matrix bc / 2, and, in batches,
    the entry factored last (the first one with uplo = upper, the last one with uplo = lower,
    where the search for an interchange is empty) in the matrix 0. The matrices are diagonally
    dominant (without interchanges), so that the NaN is the pivot of its column. **/
inline rocblas_int sytf2_sytrf_nan_pivot(const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int bc,
                                         const rocblas_int b,
                                         const int singular)
{
    if(singular != 2 || n <= 0)
        return -1;
    if(b == bc / 2)
        return n / 2;
    if(b == 0)
        return (uplo == rocblas_fill_upper) ? 0 : n - 1;
    return -1;
}

/** SYTF2_SYTRF_RESIDUAL returns ||A - X D X^T|| / ||A|| (Frobenius norm), where A is the
    symmetric matrix given by its triangle uplo, and X D X^T is rebuilt from the factorization
    computed by SYTF2/SYTRF (factors F and pivots ipiv, in the format of LAPACK's xSYTRF):
    X = P(n) U(n) ... P(1) U(1) if uplo = upper, X = P(1) L(1) ... P(n) L(n) if uplo = lower,
    and D is block diagonal with 1-by-1 and 2-by-2 blocks. Returns 1 if ipiv is not valid. **/
template <typename T>
double sytf2_sytrf_residual(const rocblas_fill uplo,
                            const rocblas_int n,
                            const T* A,
                            const rocblas_int lda,
                            const T* F,
                            const rocblas_int* ipiv)
{
    // (in double precision, so that the rounding errors are those of the factorization)
    using H = std::conditional_t<rocblas_is_complex<T>, rocblas_double_complex, double>;
    const bool upper = (uplo == rocblas_fill_upper);
    auto f = [&](const rocblas_int i, const rocblas_int j) { return H(F[i + size_t(j) * lda]); };
    std::vector<H> X(size_t(n) * n, H(0)), D(size_t(n) * n, H(0));
    auto x = [&](const rocblas_int i, const rocblas_int j) -> H& { return X[i + size_t(j) * n]; };
    auto d = [&](const rocblas_int i, const rocblas_int j) -> H& { return D[i + size_t(j) * n]; };
    for(rocblas_int i = 0; i < n; i++)
        x(i, i) = H(1);

    // X is built by multiplying I by the factors P(k) and U(k) (or L(k)) on the right
    rocblas_int k = upper ? n - 1 : 0;
    while(upper ? k >= 0 : k < n)
    {
        // a 2-by-2 block is the pair (k-1, k) (upper) or (k, k+1) (lower), with the same negative
        // pivot -kp: the row k-1 (upper) or k+1 (lower) is interchanged with the row kp
        const bool two = ipiv[k] < 0;
        const rocblas_int s = two ? 2 : 1;
        const rocblas_int kb = upper ? k - s + 1 : k;
        if(two && (kb < 0 || kb + 1 >= n || ipiv[kb] != ipiv[kb + 1]))
            return 1;
        const rocblas_int kx = upper ? kb : kb + s - 1;
        const rocblas_int kp = (two ? -ipiv[k] : ipiv[k]) - 1;
        if(kp < 0 || kp >= n)
            return 1;

        // X = X P(k)
        for(rocblas_int i = 0; i < n; i++)
            std::swap(x(i, kx), x(i, kp));
        // X = X U(k) (or L(k)): the columns of the block gain X times their vectors, in the rows
        // above the block (upper) or below it (lower)
        for(rocblas_int c = kb; c < kb + s; c++)
        {
            const rocblas_int r0 = upper ? 0 : kb + s;
            const rocblas_int r1 = upper ? kb : n;
            for(rocblas_int r = r0; r < r1; r++)
            {
                const H v = f(r, c);
                for(rocblas_int i = 0; i < n; i++)
                    x(i, c) += x(i, r) * v;
            }
        }
        // D (its block is in the triangle uplo of F)
        for(rocblas_int c = kb; c < kb + s; c++)
            for(rocblas_int r = kb; r < kb + s; r++)
                d(r, c) = (upper == (r <= c)) ? f(r, c) : f(c, r);

        k = upper ? k - s : k + s;
    }

    // R = X D X^T (transposed, not conjugated: the complex matrices are symmetric)
    std::vector<H> Y(size_t(n) * n, H(0)), R(size_t(n) * n, H(0)), S(size_t(n) * n);
    for(rocblas_int j = 0; j < n; j++)
        for(rocblas_int ell = std::max(0, j - 1); ell <= std::min(n - 1, j + 1); ell++)
            for(rocblas_int i = 0; i < n; i++)
                Y[i + size_t(j) * n] += x(i, ell) * d(ell, j);
    for(rocblas_int j = 0; j < n; j++)
        for(rocblas_int ell = 0; ell < n; ell++)
        {
            const H v = x(j, ell);
            for(rocblas_int i = 0; i < n; i++)
                R[i + size_t(j) * n] += Y[i + size_t(ell) * n] * v;
        }
    for(rocblas_int j = 0; j < n; j++)
        for(rocblas_int i = 0; i < n; i++)
            S[i + size_t(j) * n]
                = (upper == (i <= j)) ? H(A[i + size_t(j) * lda]) : H(A[j + size_t(i) * lda]);
    return norm_error('F', n, n, n, S.data(), R.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sytf2_sytrf_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv,
                          Uh& hInfo,
                          const int singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            if(singular == 3)
            {
                // a symmetric indefinite matrix with random entries, whose factorization has
                // 1-by-1 and 2-by-2 blocks and interchanges in no particular pattern
                for(rocblas_int j = 0; j < n; j++)
                {
                    for(rocblas_int i = 0; i <= j; i++)
                    {
                        const size_t ij = i + size_t(j) * lda;
                        const size_t ji = j + size_t(i) * lda;
                        tmp = hA[b][ij] - T(5.5) + T(0.001 * ((i * 31 + j * 17) % 97));
                        hA[b][ij] = tmp;
                        hA[b][ji] = tmp;
                    }
                }
            }
            else
            {
                // scale A to avoid singularities
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        if(i == j)
                            hA[b][i + size_t(j) * lda] += 400;
                        else
                            hA[b][i + size_t(j) * lda] -= 4;
                    }
                }

                // shuffle rows to test pivoting
                // always the same permuation for debugging purposes
                // (not with a NaN pivot, which must not be moved by interchanges)
                for(rocblas_int i = 0; i < (singular == 2 || singular == 4 ? 0 : n / 2); i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        tmp = hA[b][i + size_t(j) * lda];
                        hA[b][i + size_t(j) * lda] = hA[b][n - 1 - i + size_t(j) * lda];
                        hA[b][n - 1 - i + size_t(j) * lda] = tmp;
                    }
                }
            }

            // singular = 2: a NaN pivot (see sytf2_sytrf_nan_pivot), which the unblocked algorithm
            // treats as singular, as LAPACK's xSYTF2 (DISNAN(ABSAKK)); info must point to it
            const rocblas_int knan = sytf2_sytrf_nan_pivot(uplo, n, bc, b, singular);
            if(knan >= 0)
                hA[b][knan + size_t(knan) * lda]
                    = T(std::numeric_limits<decltype(std::real(T{}))>::quiet_NaN());

            // singular = 4: the column factored second (k = 1 for lower, n - 2 for upper) becomes
            // exactly zero after the 1-by-1 pivot of the column factored first (k = 0, or n - 1),
            // although it is not zero in A: its entries are those of the first column times 1/2,
            // with a pivot of 512, so that all the products of the update are exact. The
            // factorization finds it singular, info points to it, and the updated (zero) column
            // is stored in A, so that D has a zero there (as the reference LAPACK's xLASYF does
            // since Reference-LAPACK/lapack#1433, but not LAPACK 3.12 and earlier: these cases
            // are checked by the residual of the factorization)
            if(singular == 4 && n >= 2)
            {
                const bool lower = (uplo == rocblas_fill_lower);
                const rocblas_int k0 = lower ? 0 : n - 1;
                const rocblas_int k1 = lower ? 1 : n - 2;
                hA[b][k0 + size_t(k0) * lda] = 512;
                hA[b][k1 + size_t(k0) * lda] = hA[b][k0 + size_t(k1) * lda] = 256;
                hA[b][k1 + size_t(k1) * lda] = 128;
                for(rocblas_int i = 0; i < n; i++)
                {
                    if(i == k0 || i == k1)
                        continue;
                    const T aik0 = hA[b][i + size_t(k0) * lda];
                    hA[b][i + size_t(k0) * lda] = hA[b][k0 + size_t(i) * lda] = aik0;
                    hA[b][i + size_t(k1) * lda] = hA[b][k1 + size_t(i) * lda] = aik0 / T(2);
                }
            }

            if(singular == 1 && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // add some singularities
                // always the same elements for debugging purposes
                // the algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool SYTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sytf2_sytrf_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          Uh& hIpivRes,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const int singular)
{
    int lwork = (SYTRF ? 64 * n : 0);
    std::vector<T> work(lwork);

    // input data initialization
    sytf2_sytrf_initData<true, true, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                        hIpiv, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // (the random indefinite matrices, and those with a column that becomes zero, are kept to
    // check the residual)
    std::vector<std::vector<T>> hA0;
    if(singular == 3 || singular == 4)
        for(rocblas_int b = 0; b < bc; ++b)
            hA0.emplace_back(hA[b], hA[b] + size_t(lda) * n);

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        SYTRF ? cpu_sytrf(uplo, n, hA[b], lda, hIpiv[b], work.data(), lwork, hInfo[b])
              : cpu_sytf2(uplo, n, hA[b], lda, hIpiv[b], hInfo[b]);
    }

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    // (for the random indefinite matrices, the pivots are chosen by comparisons that rounding
    // can tip either way, so that the GPU and the host LAPACK can choose different and equally
    // valid pivots; the error is then the residual of the factorization. It is also the residual
    // for the matrices with a column that becomes zero, whose factors in LAPACK 3.12 and earlier
    // keep the original column in A)
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(singular == 3 || singular == 4)
        {
            err = sytf2_sytrf_residual(uplo, n, hA0[b].data(), lda, hARes[b], hIpivRes[b]);
            *max_err = rocblas_max_nan(*max_err, err);
            continue;
        }

        // a NaN pivot is singular: the factorization skips it and info points to it (the NaN
        // stays in A, and is left out of the error, which must not be NaN)
        const rocblas_int knan = sytf2_sytrf_nan_pivot(uplo, n, bc, b, singular);
        if(knan >= 0)
        {
            const size_t kk = knan + size_t(knan) * lda;
            const bool isnan_kk = std::isnan(std::real(hARes[b][kk]));
            EXPECT_TRUE(isnan_kk) << "where b = " << b;
            EXPECT_EQ(hInfoRes[b][0], knan + 1) << "where b = " << b;
            if(!isnan_kk || hInfoRes[b][0] != knan + 1)
                *max_err += 1;
            hA[b][kk] = hARes[b][kk] = 0;
        }

        err = norm_error('F', n, n, lda, hA[b], hARes[b]);
        *max_err = rocblas_max_nan(err, *max_err);

        // also check pivoting (count the number of incorrect pivots)
        err = 0;
        for(rocblas_int i = 0; i < n; ++i)
        {
            EXPECT_EQ(hIpiv[b][i], hIpivRes[b][i]) << "where b = " << b << ", i = " << i;
            if(hIpiv[b][i] != hIpivRes[b][i])
                err++;
        }
        *max_err = rocblas_max_nan(err, *max_err);
    }

    // also check info
    err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <bool STRIDED, bool SYTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sytf2_sytrf_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             Ud& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             Uh& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const int singular)
{
    int lwork = (SYTRF ? 64 * n : 0);
    std::vector<T> work(lwork);

    if(!perf)
    {
        sytf2_sytrf_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc,
                                             hA, hIpiv, hInfo, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            SYTRF ? cpu_sytrf(uplo, n, hA[b], lda, hIpiv[b], work.data(), lwork, hInfo[b])
                  : cpu_sytf2(uplo, n, hA[b], lda, hIpiv[b], hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sytf2_sytrf_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                         hIpiv, hInfo, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sytf2_sytrf_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc,
                                             hA, hIpiv, hInfo, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(), lda,
                                                  stA, dIpiv.data(), stP, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sytf2_sytrf_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc,
                                             hA, hIpiv, hInfo, singular);

        timer.start(stream);
        rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(), lda, stA, dIpiv.data(),
                              stP, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, bool SYTRF, typename T>
void testing_sytf2_sytrf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", n);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // check non-supported values
    if(uplo == rocblas_fill_full)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, stP,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, stP,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(
                rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytf2_sytrf_getError<STRIDED, SYTRF, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP,
                                                    dInfo, bc, hA, hARes, hIpiv, hIpivRes, hInfo,
                                                    hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            sytf2_sytrf_getPerfData<STRIDED, SYTRF, T>(
                handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytf2_sytrf_getError<STRIDED, SYTRF, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP,
                                                    dInfo, bc, hA, hARes, hIpiv, hIpivRes, hInfo,
                                                    hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            sytf2_sytrf_getPerfData<STRIDED, SYTRF, T>(
                handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                argus.perf, argus.singular);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "lda");
                rocsolver_bench_output(uploC, n, lda);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_SYTF2_SYTRF(...) \
    extern template void testing_sytf2_sytrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYTF2_SYTRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
