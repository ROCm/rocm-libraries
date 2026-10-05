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

#include <random>

#include "common/lapack/testing_gebal.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

/** HSEQR_GENMATRIX generates an n-by-n matrix A (with leading dimension n)
    according to mtype, and sets ilo and ihi:
    - mtype = 0: random matrix with normally distributed entries.
    - mtype = 1: A = X * diag(lambda) * X^(-1), with eigenvalues lambda in
                 three tight clusters.
    - mtype = 2: A = X * diag(lambda) * X^(-1), with about 8% of the eigenvalues
                 of size about 1e-14 (a near-zero cluster).
    - mtype = 3: random upper triangular matrix.
    - mtype = 4: matrix with isolated eigenvalues and bad scaling, balanced with GEBAL,
                 so that ilo > 1 and ihi < n.
    - mtype = 5: random upper triangular matrix, balanced with GEBAL, so that all the
                 eigenvalues are isolated (ilo = ihi).
    - mtype = 6: random matrix with n/3 leading columns and n/3 trailing rows made
                 triangular, permuted with GEBAL, so that the active block is small
                 (ihi - ilo + 1 ~ n/3, few shifts in the multishift sweeps).
    - mtype >= 8: as mtype = 4; a NaN or an infinite entry is set after the reduction to
                 Hessenberg form (see hseqr_nonfinite).
    For mtype < 4, ilo = 1 and ihi = n.
    Classes 0, 3, 4, 5 and 6 have well-conditioned eigenvalues (in practice). **/
template <typename T>
void hseqr_genMatrix(const rocblas_int n,
                     std::vector<T>& A,
                     const rocblas_int mtype,
                     const rocblas_int seed,
                     rocblas_int& ilo,
                     rocblas_int& ihi)
{
    using S = decltype(std::real(T{}));

    std::mt19937_64 rng(1000003ull * n + 7919ull * seed + 104729ull * mtype);
    std::normal_distribution<double> gauss(0.0, 1.0);
    auto rnd = [&]() { return T(S(gauss(rng)), S(gauss(rng))); };

    ilo = 1;
    ihi = n;
    A.assign(size_t(n) * n, T(0));
    if(n == 0)
        return;

    if(mtype == 0 || mtype == 4 || mtype == 6 || mtype >= 8)
    {
        for(auto& a : A)
            a = rnd();
    }
    else if(mtype == 3 || mtype == 5)
    {
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i <= j; i++)
                A[i + size_t(j) * n] = rnd();
    }
    else
    {
        // A = X * diag(lambda) * X^(-1)
        std::vector<T> lambda(n);
        const T centers[3] = {T(1, 0), T(-1, 1), T(0, 2)};
        for(rocblas_int j = 0; j < n; j++)
        {
            if(mtype == 1)
                lambda[j] = centers[j % 3] + S(1e-4) * rnd();
            else
                lambda[j] = (j % 12 == 5) ? S(1e-14) * rnd() : rnd();
        }

        std::vector<T> X(size_t(n) * n), Xinv(size_t(n) * n), XL(size_t(n) * n);
        for(auto& x : X)
            x = rnd();
        Xinv = X;
        std::vector<rocblas_int> ipiv(n);
        rocblas_int info;
        cpu_getrf(n, n, Xinv.data(), n, ipiv.data(), &info);
        std::vector<T> work(size_t(n) * 64);
        cpu_getri(n, Xinv.data(), n, ipiv.data(), work.data(), (rocblas_int)work.size(), &info);
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
                XL[i + size_t(j) * n] = X[i + size_t(j) * n] * lambda[j];
        cpu_gemm(rocblas_operation_none, rocblas_operation_none, n, n, n, T(1), XL.data(), n,
                 Xinv.data(), n, T(0), A.data(), n);
    }

    if(mtype == 6)
    {
        const rocblas_int third = n / 3;
        for(rocblas_int j = 0; j < third; j++)
            for(rocblas_int i = j + 1; i < n; i++)
                A[i + size_t(j) * n] = T(0);
        for(rocblas_int i = n - third; i < n; i++)
            for(rocblas_int j = 0; j < i; j++)
                A[i + size_t(j) * n] = T(0);
        std::vector<S> scale(n);
        rocblas_int info;
        cpu_gebal(rocsolver_balance_permute, n, A.data(), n, &ilo, &ihi, scale.data(), &info);
    }

    if(mtype == 5)
    {
        std::vector<S> scale(n);
        rocblas_int info;
        cpu_gebal(rocsolver_balance_both, n, A.data(), n, &ilo, &ihi, scale.data(), &info);
    }

    if(mtype == 4 || mtype >= 8)
    {
        // isolated eigenvalues and bad scaling (see gebal_genMatrix), then balance
        gebal_genMatrix(n, A.data(), n, 2, seed);
        std::vector<S> scale(n);
        rocblas_int info;
        cpu_gebal(rocsolver_balance_both, n, A.data(), n, &ilo, &ihi, scale.data(), &info);
    }
}

/** HSEQR_MEMBER_CLASS returns the class of the matrix b of a batch of bc matrices of the
    test class mtype (mtype = 14 is a mixed batch: class 4 for even b, class 12 for odd b,
    or class 12 if the batch has a single matrix). **/
inline rocblas_int
    hseqr_member_class(const rocblas_int mtype, const rocblas_int b, const rocblas_int bc)
{
    if(mtype == 14)
        return (b % 2 == 0 && bc > 1) ? 4 : 12;
    return mtype;
}

/** HSEQR_NONFINITE sets, for the classes mtype >= 8, a NaN or an infinite entry in the upper
    Hessenberg matrix H (with the active block ilo:ihi, 1-based, ilo < ihi), after its
    reduction to Hessenberg form (so that its position is controlled), with m = (ilo+ihi)/2:
    - mtype = 8:  NaN in H(ilo, ilo),
    - mtype = 9:  NaN in H(ihi, ihi-1),
    - mtype = 10: +Inf in the real part only of H(m, m),
    - mtype = 11: +Inf in the imaginary part only of H(m, m+1),
    - mtype = 12: H(m+1, m) = 0 (two irreducible blocks), and a NaN in H(ilo, m), in the top
                  block: the bottom block m+1:ihi must be processed as usual,
    - mtype = 13: H(m+1, m) = 0, and a NaN in H(ilo, ihi), outside both diagonal blocks: it
                  must not affect the computation,
    - mtype = 15: all the subdiagonal entries of the active block set to zero (upper
                  triangular), with (1, -Inf) in H(ilo, ilo), +Inf in H(m, m) and a NaN in
                  H(ihi, ihi): 1x1 blocks are deflated at once, W = diag(H) and info = 0,
    - mtype = 16: H(m, m-1) = H(m+1, m) = 0 and (+Inf, 1) in H(m, m): a 1x1 block with an
                  infinite eigenvalue between two irreducible blocks; info = 0,
    - mtype = 17: H(ihi, ihi-1) = 0 and a NaN in H(m, m), in the block ilo:ihi-1: nothing is
                  left to iterate on, info = ihi-1 and W(ihi) = H(ihi, ihi).
    It returns the expected info (the last row of the lowest diagonal block of size > 1 with
    a NaN or an infinite entry, or 0), and sets splits to the rows k (increasing) for which
    H(k+1, k) is set to zero. If H is null, it only returns these values. **/
template <typename T>
rocblas_int hseqr_nonfinite(const rocblas_int mtype,
                            const rocblas_int ilo,
                            const rocblas_int ihi,
                            T* H,
                            const rocblas_int ldh,
                            std::vector<rocblas_int>& splits)
{
    using S = decltype(std::real(T{}));
    const S nan = std::numeric_limits<S>::quiet_NaN();
    const S inf = std::numeric_limits<S>::infinity();
    auto set = [&](const rocblas_int i, const rocblas_int j, const T v) {
        if(H)
            H[(i - 1) + size_t(j - 1) * ldh] = v;
    };
    auto split = [&](const rocblas_int k) {
        splits.push_back(k);
        set(k + 1, k, T(0));
    };
    const rocblas_int m = (ilo + ihi) / 2;
    splits.clear();
    if(mtype < 8 || ilo >= ihi)
        return 0;
    switch(mtype)
    {
    case 8: set(ilo, ilo, T(nan, S(0))); return ihi;
    case 9: set(ihi, ihi - 1, T(nan, S(0))); return ihi;
    case 10: set(m, m, T(inf, S(0))); return ihi;
    case 11: set(m, m + 1, T(S(1), inf)); return ihi;
    case 12:
        split(m);
        set(ilo, m, T(nan, S(0)));
        return m;
    case 13:
        split(m);
        set(ilo, ihi, T(nan, S(0)));
        return 0;
    case 15:
        for(rocblas_int k = ilo; k < ihi; k++)
            split(k);
        set(ilo, ilo, T(S(1), -inf));
        set(m, m, T(inf, S(0)));
        set(ihi, ihi, T(nan, S(0)));
        return 0;
    case 16:
        if(m > ilo)
            split(m - 1);
        split(m);
        set(m, m, T(inf, S(1)));
        return 0;
    case 17:
        split(ihi - 1);
        set(m, m, T(nan, S(0)));
        return (ihi - 1 > ilo) ? ihi - 1 : 0;
    default: return 0;
    }
}

template <bool STRIDED, typename T, typename U>
void hseqr_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_schur_job job,
                        const rocsolver_schur_vectors compz,
                        const rocblas_int n,
                        rocblas_int* dIlo,
                        rocblas_int* dIhi,
                        U dH,
                        const rocblas_int ldh,
                        const rocblas_stride stH,
                        T* dW,
                        const rocblas_stride stW,
                        U dZ,
                        const rocblas_int ldz,
                        const rocblas_stride stZ,
                        rocblas_int* dInfo,
                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, nullptr, job, compz, n, dIlo, dIhi, dH, ldh, stH,
                                          dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, rocsolver_schur_job(0), compz, n, dIlo,
                                          dIhi, dH, ldh, stH, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, rocsolver_schur_vectors(0), n, dIlo,
                                          dIhi, dH, ldh, stH, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, dIhi, dH, ldh,
                                              stH, dW, stW, dZ, ldz, stZ, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, (rocblas_int*)nullptr,
                                          dIhi, dH, ldh, stH, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, (rocblas_int*)nullptr,
                                          dH, ldh, stH, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, dIhi, (U) nullptr,
                                          ldh, stH, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, dIhi, dH, ldh, stH,
                                          (T*)nullptr, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, dIhi, dH, ldh, stH,
                                          dW, stW, (U) nullptr, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, dIhi, dH, ldh, stH,
                                          dW, stW, dZ, ldz, stZ, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, 0, (rocblas_int*)nullptr,
                                          (rocblas_int*)nullptr, (U) nullptr, ldh, stH, (T*)nullptr,
                                          stW, (U) nullptr, ldz, stZ, dInfo, bc),
                          rocblas_status_success);
    // Z is not referenced when compz = none
    EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, rocsolver_schur_vectors_none, n,
                                          dIlo, dIhi, dH, ldh, stH, dW, stW, (U) nullptr, ldz, stZ,
                                          dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo, dIhi, dH, ldh,
                                              stH, dW, stW, dZ, ldz, stZ, (rocblas_int*)nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_hseqr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocsolver_schur_job job = rocsolver_schur_form;
    rocsolver_schur_vectors compz = rocsolver_schur_vectors_initialize;
    rocblas_int n = 1;
    rocblas_int ldh = 1;
    rocblas_int ldz = 1;
    rocblas_stride stH = 1;
    rocblas_stride stW = 1;
    rocblas_stride stZ = 1;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_HSEQR
    // memory allocations
    device_strided_batch_vector<rocblas_int> dIlo(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIhi(1, 1, 1, 1);
    device_strided_batch_vector<T> dW(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dIlo.memcheck());
    CHECK_HIP_ERROR(dIhi.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    rocblas_int one = 1;
    CHECK_HIP_ERROR(hipMemcpy(dIlo.data(), &one, sizeof(rocblas_int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIhi.data(), &one, sizeof(rocblas_int), hipMemcpyHostToDevice));

    if(BATCHED)
    {
        device_batch_vector<T> dH(1, 1, 1);
        device_batch_vector<T> dZ(1, 1, 1);
        CHECK_HIP_ERROR(dH.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());

        // check bad arguments
        hseqr_checkBadArgs<STRIDED>(handle, job, compz, n, dIlo.data(), dIhi.data(), dH.data(), ldh,
                                    stH, dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc);
    }
    else
    {
        device_strided_batch_vector<T> dH(1, 1, 1, 1);
        device_strided_batch_vector<T> dZ(1, 1, 1, 1);
        CHECK_HIP_ERROR(dH.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());

        // check bad arguments
        hseqr_checkBadArgs<STRIDED>(handle, job, compz, n, dIlo.data(), dIhi.data(), dH.data(), ldh,
                                    stH, dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc);
    }
#endif
}

/** HSEQR_INITDATA generates, for each matrix of the batch, an upper Hessenberg matrix H
    (by reducing the matrix given by hseqr_genMatrix with the host GEHRD), the values of
    ilo and ihi, and, for compz = update, the unitary matrix Q of the reduction (from the
    host UNGHR) as the initial Z. The host copies H0 and Z0 keep the input data. **/
template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void hseqr_initData(const rocblas_handle handle,
                    const rocsolver_schur_vectors compz,
                    const rocblas_int n,
                    Ud& dIlo,
                    Ud& dIhi,
                    Td& dH,
                    const rocblas_int ldh,
                    Td& dZ,
                    const rocblas_int ldz,
                    const rocblas_int bc,
                    Uh& hIlo,
                    Uh& hIhi,
                    Th& hH0,
                    Th& hZ0,
                    const rocblas_int mtype)
{
    if(CPU)
    {
        std::vector<T> A;
        std::vector<T> tau(std::max(n, 1));
        std::vector<T> work(size_t(std::max(n, 1)) * 64);
        for(rocblas_int b = 0; b < bc; ++b)
        {
            rocblas_int ilo, ihi;
            const rocblas_int mt = hseqr_member_class(mtype, b, bc);
            hseqr_genMatrix(n, A, mt, b, ilo, ihi);
            hIlo[b][0] = ilo;
            hIhi[b][0] = ihi;

            // reduce to Hessenberg form
            cpu_gehrd(n, ilo, ihi, A.data(), n, tau.data(), work.data(), (rocblas_int)work.size());
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    hH0[b][i + size_t(j) * ldh] = (i <= j + 1) ? A[i + size_t(j) * n] : T(0);

            // NaN or infinite entry, if any
            std::vector<rocblas_int> splits;
            hseqr_nonfinite(mt, ilo, ihi, hH0[b], ldh, splits);

            // initial Z
            if(compz == rocsolver_schur_vectors_update)
            {
                cpu_orghr_unghr(n, ilo, ihi, A.data(), n, tau.data(), work.data(),
                                (rocblas_int)work.size());
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        hZ0[b][i + size_t(j) * ldz] = A[i + size_t(j) * n];
            }
            else if(compz == rocsolver_schur_vectors_initialize)
            {
                // garbage, to check that Z is initialized
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        hZ0[b][i + size_t(j) * ldz] = T(3, -5);
            }
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dIlo.transfer_from(hIlo));
        CHECK_HIP_ERROR(dIhi.transfer_from(hIhi));
        CHECK_HIP_ERROR(dH.transfer_from(hH0));
        if(compz != rocsolver_schur_vectors_none)
            CHECK_HIP_ERROR(dZ.transfer_from(hZ0));
    }
}

/** HSEQR_CHECKNONFINITE checks the results for a matrix of the classes mtype >= 8 (see
    hseqr_nonfinite), with ibad the expected info and lo = max(ilo, ibad+1); the rows lo:ihi
    are split into diagonal blocks at the zero subdiagonal entries set by hseqr_nonfinite:
    - info = ibad,
    - the eigenvalues isolated by GEBAL and those of the 1x1 blocks of the rows lo:ihi are
      copied from the diagonal (exactly: NaN if it is NaN, infinite if it is infinite),
      W(ilo:ibad) is NaN, and the other eigenvalues of the rows lo:ihi match the host LAPACK
      eigenvalues (computed with the non-finite entries replaced by zero, which does not
      change them; the 1x1 blocks are left out of the matching),
    - with the Schur form, T(:, lo:ihi) is upper triangular with W(lo:ihi) on its diagonal,
      and with the Schur vectors, each diagonal block B = p:q of the rows lo:ihi (except
      the 1x1 blocks with a non-finite entry) satisfies H0(B,B) = Q(B,B) T(B,B) Q(B,B)^H
      with Q(B,B) unitary, where Q = Z (or Z0^H Z if Z is updated),
    - if nothing is left to iterate on (the rows lo:ihi are all in 1x1 blocks), Z is not
      changed (or is the identity) and, with the Schur form, neither is H.
    The errors are added to max_err (and max_err_eig for the eigenvalues). **/
template <typename T>
void hseqr_checkNonfinite(const rocsolver_schur_job job,
                          const rocsolver_schur_vectors compz,
                          const rocblas_int n,
                          const rocblas_int ilo,
                          const rocblas_int ihi,
                          const T* H0,
                          const rocblas_int ldh,
                          const T* Z0,
                          const rocblas_int ldz,
                          const T* HRes,
                          const T* WRes,
                          const T* ZRes,
                          const rocblas_int info,
                          const rocblas_int mt,
                          const rocblas_int b,
                          double* max_err,
                          double* max_err_eig)
{
    const bool wantt = (job == rocsolver_schur_form);
    const bool wantz = (compz != rocsolver_schur_vectors_none);
    auto h0 = [&](const rocblas_int i, const rocblas_int j) {
        return H0[(i - 1) + size_t(j - 1) * ldh];
    };
    auto hr = [&](const rocblas_int i, const rocblas_int j) {
        return HRes[(i - 1) + size_t(j - 1) * ldh];
    };
    auto finite
        = [](const T z) { return std::isfinite(std::real(z)) && std::isfinite(std::imag(z)); };
    // equality, with NaN equal to NaN (in the real and imaginary parts separately)
    auto same = [](const T x, const T y) {
        auto eq = [](const auto u, const auto v) {
            return (u == v) || (std::isnan(u) && std::isnan(v));
        };
        return eq(std::real(x), std::real(y)) && eq(std::imag(x), std::imag(y));
    };

    std::vector<rocblas_int> splits;
    const rocblas_int ibad = hseqr_nonfinite<T>(mt, ilo, ihi, nullptr, ldh, splits);
    const rocblas_int lo = std::max(ilo, ibad + 1);

    // diagonal blocks of the rows lo:ihi, and the rows in 1x1 blocks
    std::vector<std::pair<rocblas_int, rocblas_int>> blocks;
    std::vector<bool> single(n + 1, false);
    {
        rocblas_int p = lo;
        for(const rocblas_int k : splits)
            if(k >= lo && k < ihi)
            {
                blocks.push_back({p, k});
                p = k + 1;
            }
        blocks.push_back({p, ihi});
        for(const auto& blk : blocks)
            if(blk.first == blk.second)
                single[blk.first] = true;
    }
    bool nothing_left = true;
    for(rocblas_int j = lo; j <= ihi; j++)
        nothing_left = nothing_left && single[j];

    EXPECT_EQ(info, ibad) << "where b = " << b;
    if(info != ibad)
        *max_err += 1;

    // eigenvalues: isolated and 1x1 blocks (exact), NaN (ilo:ibad), and finite (lo:ihi)
    for(rocblas_int j = 1; j <= n; j++)
    {
        const T w = WRes[j - 1];
        bool ok;
        if(j < ilo || j > ihi || (j >= lo && single[j]))
            ok = same(w, h0(j, j));
        else if(j <= ibad)
            ok = std::isnan(std::real(w)) && std::isnan(std::imag(w));
        else
            ok = finite(w);
        EXPECT_TRUE(ok) << "where b = " << b << ", j = " << j;
        if(!ok)
            *max_err += 1;
    }

    // eigenvalues lo:ihi (except the 1x1 blocks) compared with the host LAPACK
    std::vector<T> Hc(size_t(n) * n), Wc(n), Zc(1), work(size_t(n) * 64);
    double hnorm = 0;
    for(rocblas_int j = 1; j <= n; j++)
        for(rocblas_int i = 1; i <= n; i++)
        {
            const T a = finite(h0(i, j)) ? h0(i, j) : T(0);
            Hc[(i - 1) + size_t(j - 1) * n] = a;
            hnorm += std::norm(a);
        }
    hnorm = std::max(std::sqrt(hnorm), 1e-300);
    if(lo <= ihi && !nothing_left)
    {
        rocblas_int hinfo;
        cpu_hseqr(rocsolver_schur_eigenvalues, rocsolver_schur_vectors_none, n, lo, ihi, Hc.data(),
                  n, Wc.data(), Zc.data(), 1, work.data(), (rocblas_int)work.size(), &hinfo);
        EXPECT_EQ(hinfo, 0) << "(host LAPACK) where b = " << b;
        std::vector<bool> used(n, false);
        for(rocblas_int j = lo - 1; j < ihi; j++)
        {
            if(single[j + 1])
                continue;
            double best = std::numeric_limits<double>::infinity();
            rocblas_int kbest = -1;
            for(rocblas_int k = lo - 1; k < ihi; k++)
            {
                if(used[k] || single[k + 1])
                    continue;
                const double d = std::abs(Wc[j] - WRes[k]);
                if(d < best)
                {
                    best = d;
                    kbest = k;
                }
            }
            if(kbest < 0)
            {
                *max_err_eig = std::max(*max_err_eig, 1.0);
                continue;
            }
            used[kbest] = true;
            *max_err_eig = std::max(*max_err_eig, best / hnorm);
        }
    }

    // T(:, lo:ihi) upper triangular, with W on its diagonal
    if(wantt)
    {
        rocblas_int nonzeros = 0, mismatches = 0;
        for(rocblas_int j = lo; j <= ihi; j++)
        {
            for(rocblas_int i = j + 1; i <= n; i++)
                nonzeros += !(hr(i, j) == T(0));
            mismatches += !same(hr(j, j), WRes[j - 1]);
        }
        EXPECT_EQ(nonzeros, 0) << "where b = " << b;
        EXPECT_EQ(mismatches, 0) << "where b = " << b;
        *max_err += nonzeros + mismatches;
    }

    // residual and orthogonality of the diagonal blocks of the rows lo:ihi
    if(wantt && wantz && lo <= ihi)
    {
        std::vector<T> Z(size_t(n) * n), Q(size_t(n) * n);
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
            {
                Z[i + size_t(j) * n] = ZRes[i + size_t(j) * ldz];
                Q[i + size_t(j) * n] = Z0[i + size_t(j) * ldz];
            }
        if(compz == rocsolver_schur_vectors_update)
        {
            std::vector<T> Z0c = Q;
            cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n, n, T(1),
                     Z0c.data(), n, Z.data(), n, T(0), Q.data(), n);
        }
        else
            Q = Z;

        for(const auto& blk : blocks)
        {
            const rocblas_int p = blk.first;
            const rocblas_int nb = blk.second - p + 1;
            if(nb == 1 && !finite(h0(p, p)))
                continue;
            std::vector<T> Hb(size_t(nb) * nb), Tb(size_t(nb) * nb), Qb(size_t(nb) * nb),
                QT(size_t(nb) * nb);
            for(rocblas_int j = 0; j < nb; j++)
                for(rocblas_int i = 0; i < nb; i++)
                {
                    Hb[i + size_t(j) * nb] = h0(p + i, p + j);
                    Tb[i + size_t(j) * nb] = (i <= j) ? hr(p + i, p + j) : T(0);
                    Qb[i + size_t(j) * nb] = Q[(p - 1 + i) + size_t(p - 1 + j) * n];
                }
            const double bnorm = std::max(double(snorm('F', nb, nb, Hb.data(), nb)), 1e-300);
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, nb, nb, nb, T(1), Qb.data(),
                     nb, Tb.data(), nb, T(0), QT.data(), nb);
            cpu_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, nb, nb, nb,
                     T(-1), QT.data(), nb, Qb.data(), nb, T(1), Hb.data(), nb);
            double err = snorm('F', nb, nb, Hb.data(), nb) / bnorm;
            *max_err = (err > *max_err || std::isnan(err)) ? (std::isnan(err) ? 1.0 : err) : *max_err;

            for(rocblas_int j = 0; j < nb; j++)
                for(rocblas_int i = 0; i < nb; i++)
                    QT[i + size_t(j) * nb] = (i == j) ? T(1) : T(0);
            cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, nb, nb, nb,
                     T(-1), Qb.data(), nb, Qb.data(), nb, T(1), QT.data(), nb);
            err = snorm('F', nb, nb, QT.data(), nb) / std::sqrt(double(nb));
            *max_err = (err > *max_err || std::isnan(err)) ? (std::isnan(err) ? 1.0 : err) : *max_err;
        }
    }

    // nothing left to iterate on: Z (and, with the Schur form, H) is not changed
    if(nothing_left)
    {
        rocblas_int changed = 0;
        if(wantz)
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                {
                    const T z0 = (compz == rocsolver_schur_vectors_update) ? Z0[i + size_t(j) * ldz]
                                                                           : T(i == j ? 1 : 0);
                    changed += !(ZRes[i + size_t(j) * ldz] == z0);
                }
        if(wantt)
            for(rocblas_int j = 1; j <= n; j++)
                for(rocblas_int i = 1; i <= n; i++)
                    changed += !same(hr(i, j), h0(i, j));
        EXPECT_EQ(changed, 0) << "where b = " << b;
        *max_err += changed;
    }
}

template <bool STRIDED, typename T, typename Td, typename Wd, typename Ud, typename Th, typename Wh, typename Uh>
void hseqr_getError(const rocblas_handle handle,
                    const rocsolver_schur_job job,
                    const rocsolver_schur_vectors compz,
                    const rocblas_int n,
                    Ud& dIlo,
                    Ud& dIhi,
                    Td& dH,
                    const rocblas_int ldh,
                    const rocblas_stride stH,
                    Wd& dW,
                    const rocblas_stride stW,
                    Td& dZ,
                    const rocblas_int ldz,
                    const rocblas_stride stZ,
                    Ud& dInfo,
                    const rocblas_int bc,
                    Uh& hIlo,
                    Uh& hIhi,
                    Th& hH0,
                    Th& hZ0,
                    Th& hHRes,
                    Wh& hWRes,
                    Th& hZRes,
                    Uh& hInfoRes,
                    const rocblas_int mtype,
                    double* max_err,
                    double* max_err_eig)
{
    using S = decltype(std::real(T{}));
    const bool wantt = (job == rocsolver_schur_form);
    const bool wantz = (compz != rocsolver_schur_vectors_none);

    // input data initialization
    hseqr_initData<true, true, T>(handle, compz, n, dIlo, dIhi, dH, ldh, dZ, ldz, bc, hIlo, hIhi,
                                  hH0, hZ0, mtype);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo.data(), dIhi.data(),
                                        dH.data(), ldh, stH, dW.data(), stW, dZ.data(), ldz, stZ,
                                        dInfo.data(), bc));
    CHECK_HIP_ERROR(hHRes.transfer_from(dH));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(wantz)
        CHECK_HIP_ERROR(hZRes.transfer_from(dZ));

    *max_err = 0;
    *max_err_eig = 0;
    std::vector<T> Hc(size_t(n) * n), Zc(size_t(n) * n), Wc(n), work(size_t(n) * 64);
    std::vector<T> M(size_t(n) * n), ZT(size_t(n) * n), R(size_t(n) * n);
    for(rocblas_int b = 0; b < bc; ++b)
    {
        const rocblas_int ilo = hIlo[b][0];
        const rocblas_int ihi = hIhi[b][0];
        const rocblas_int mt = hseqr_member_class(mtype, b, bc);

        // eigenvalues with well-conditioned eigenvalues are compared with the host LAPACK
        const bool compare_eigenvalues = (mt == 0 || mt == 3 || mt == 4 || mt == 5 || mt == 6);

        if(mt >= 8 && ilo < ihi)
        {
            hseqr_checkNonfinite(job, compz, n, ilo, ihi, hH0[b], ldh, hZ0[b], ldz, hHRes[b],
                                 hWRes[b], hZRes[b], hInfoRes[b][0], mt, b, max_err, max_err_eig);
            continue;
        }

        EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
        if(hInfoRes[b][0] != 0)
            *max_err += 1;

        // eigenvalues isolated by GEBAL are copied from the diagonal
        for(rocblas_int j = 0; j < n; j++)
            if(j + 1 < ilo || j + 1 > ihi)
            {
                EXPECT_EQ(hWRes[b][j], hH0[b][j + size_t(j) * ldh])
                    << "where b = " << b << ", j = " << j;
                if(hWRes[b][j] != hH0[b][j + size_t(j) * ldh])
                    *max_err += 1;
            }

        if(wantt)
        {
            // T must be upper triangular, with the eigenvalues on its diagonal
            rocblas_int nonzeros = 0, mismatches = 0;
            for(rocblas_int j = 0; j < n; j++)
            {
                for(rocblas_int i = j + 1; i < n; i++)
                    nonzeros += (hHRes[b][i + size_t(j) * ldh] != T(0));
                mismatches += (hHRes[b][j + size_t(j) * ldh] != hWRes[b][j]);
            }
            EXPECT_EQ(nonzeros, 0) << "where b = " << b;
            EXPECT_EQ(mismatches, 0) << "where b = " << b;
            *max_err += nonzeros + mismatches;
        }

        // the sum of the eigenvalues is the trace of H0 (checked for all the classes, also
        // those whose eigenvalues are too ill-conditioned to be compared one by one):
        // error |sum(W) - trace(H0)| / (sqrt(n) * ||H0||)
        {
            T sumw = 0, trace = 0;
            double hnorm2 = 0;
            for(rocblas_int j = 0; j < n; j++)
            {
                sumw += hWRes[b][j];
                trace += hH0[b][j + size_t(j) * ldh];
                for(rocblas_int i = 0; i < n; i++)
                    hnorm2 += std::norm(hH0[b][i + size_t(j) * ldh]);
            }
            const double err
                = std::abs(sumw - trace) / std::max(std::sqrt(double(n) * hnorm2), 1e-300);
            *max_err_eig = std::max(*max_err_eig, err);
        }

        if(wantt && wantz)
        {
            // M = Z0 * H0 * Z0^H (or H0 if Z0 = I)
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    M[i + size_t(j) * n] = hH0[b][i + size_t(j) * ldh];
            if(compz == rocsolver_schur_vectors_update)
            {
                std::vector<T> Z0(size_t(n) * n);
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        Z0[i + size_t(j) * n] = hZ0[b][i + size_t(j) * ldz];
                cpu_gemm(rocblas_operation_none, rocblas_operation_none, n, n, n, T(1), Z0.data(),
                         n, M.data(), n, T(0), R.data(), n);
                cpu_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, n, n, n,
                         T(1), R.data(), n, Z0.data(), n, T(0), M.data(), n);
            }

            // residual ||M - Z * T * Z^H|| / ||M||
            std::vector<T> Z(size_t(n) * n), Tm(size_t(n) * n);
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                {
                    Z[i + size_t(j) * n] = hZRes[b][i + size_t(j) * ldz];
                    Tm[i + size_t(j) * n] = hHRes[b][i + size_t(j) * ldh];
                }
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, n, n, n, T(1), Z.data(), n,
                     Tm.data(), n, T(0), ZT.data(), n);
            R = M;
            cpu_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, n, n, n, T(-1),
                     ZT.data(), n, Z.data(), n, T(1), R.data(), n);
            double err = snorm('F', n, n, R.data(), n)
                / std::max(double(snorm('F', n, n, M.data(), n)), 1e-300);
            *max_err = err > *max_err ? err : *max_err;

            // orthogonality ||Z^H * Z - I||
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    R[i + size_t(j) * n] = (i == j) ? T(1) : T(0);
            cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n, n, T(-1),
                     Z.data(), n, Z.data(), n, T(1), R.data(), n);
            err = snorm('F', n, n, R.data(), n) / std::sqrt(double(n));
            *max_err = err > *max_err ? err : *max_err;
        }
        else if(wantz)
        {
            // without the Schur form, only the orthogonality of Z can be checked:
            // ||Z^H * Z - I|| / sqrt(n)
            std::vector<T> Z(size_t(n) * n);
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                {
                    Z[i + size_t(j) * n] = hZRes[b][i + size_t(j) * ldz];
                    R[i + size_t(j) * n] = (i == j) ? T(1) : T(0);
                }
            cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n, n, T(-1),
                     Z.data(), n, Z.data(), n, T(1), R.data(), n);
            const double err = snorm('F', n, n, R.data(), n) / std::sqrt(double(n));
            *max_err = err > *max_err ? err : *max_err;
        }

        if(compare_eigenvalues)
        {
            // host LAPACK eigenvalues
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    Hc[i + size_t(j) * n] = hH0[b][i + size_t(j) * ldh];
            rocblas_int info;
            cpu_hseqr(rocsolver_schur_eigenvalues, rocsolver_schur_vectors_none, n, ilo, ihi,
                      Hc.data(), n, Wc.data(), Zc.data(), 1, work.data(), (rocblas_int)work.size(),
                      &info);
            EXPECT_EQ(info, 0) << "(host LAPACK) where b = " << b;

            // match each host eigenvalue with the closest unmatched device eigenvalue;
            // error is max |lambda - lambda_ref| / ||H0||
            double hnorm = 0;
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    hnorm += std::norm(hH0[b][i + size_t(j) * ldh]);
            hnorm = std::max(std::sqrt(hnorm), 1e-300);
            std::vector<bool> used(n, false);
            for(rocblas_int j = 0; j < n; j++)
            {
                double best = std::numeric_limits<double>::infinity();
                rocblas_int kbest = -1;
                for(rocblas_int k = 0; k < n; k++)
                {
                    if(used[k])
                        continue;
                    double d = std::abs(Wc[j] - hWRes[b][k]);
                    if(d < best)
                    {
                        best = d;
                        kbest = k;
                    }
                }
                if(kbest < 0)
                {
                    *max_err_eig = std::max(*max_err_eig, 1.0);
                    continue;
                }
                used[kbest] = true;
                *max_err_eig = std::max(*max_err_eig, best / hnorm);
            }
        }
    }
}

template <bool STRIDED, typename T, typename Td, typename Wd, typename Ud, typename Th, typename Uh>
void hseqr_getPerfData(const rocblas_handle handle,
                       const rocsolver_schur_job job,
                       const rocsolver_schur_vectors compz,
                       const rocblas_int n,
                       Ud& dIlo,
                       Ud& dIhi,
                       Td& dH,
                       const rocblas_int ldh,
                       const rocblas_stride stH,
                       Wd& dW,
                       const rocblas_stride stW,
                       Td& dZ,
                       const rocblas_int ldz,
                       const rocblas_stride stZ,
                       Ud& dInfo,
                       const rocblas_int bc,
                       Uh& hIlo,
                       Uh& hIhi,
                       Th& hH0,
                       Th& hZ0,
                       const rocblas_int mtype,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    hseqr_initData<true, false, T>(handle, compz, n, dIlo, dIhi, dH, ldh, dZ, ldz, bc, hIlo, hIhi,
                                   hH0, hZ0, mtype);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        std::vector<T> Hc(size_t(n) * n), Zc(size_t(n) * n), Wc(n), work(size_t(n) * 64);
        rocblas_int info;
        *cpu_time_used = 0;
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                {
                    Hc[i + size_t(j) * n] = hH0[b][i + size_t(j) * ldh];
                    if(compz != rocsolver_schur_vectors_none)
                        Zc[i + size_t(j) * n] = hZ0[b][i + size_t(j) * ldz];
                }
            double t0 = get_time_us_no_sync();
            cpu_hseqr(job, compz, n, hIlo[b][0], hIhi[b][0], Hc.data(), n, Wc.data(), Zc.data(), n,
                      work.data(), (rocblas_int)work.size(), &info);
            *cpu_time_used += get_time_us_no_sync() - t0;
        }
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        hseqr_initData<false, true, T>(handle, compz, n, dIlo, dIhi, dH, ldh, dZ, ldz, bc, hIlo,
                                       hIhi, hH0, hZ0, mtype);

        CHECK_ROCBLAS_ERROR(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo.data(),
                                            dIhi.data(), dH.data(), ldh, stH, dW.data(), stW,
                                            dZ.data(), ldz, stZ, dInfo.data(), bc));
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
        hseqr_initData<false, true, T>(handle, compz, n, dIlo, dIhi, dH, ldh, dZ, ldz, bc, hIlo,
                                       hIhi, hH0, hZ0, mtype);

        timer.start(stream);
        rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo.data(), dIhi.data(), dH.data(), ldh,
                        stH, dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_hseqr(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char jobC = argus.get<char>("schur_job");
    char compzC = argus.get<char>("compz");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldh = argus.get<rocblas_int>("ldh", n);
    rocblas_int ldz = argus.get<rocblas_int>("ldz", n);
    rocblas_stride stH = argus.get<rocblas_stride>("strideH", rocblas_stride(ldh) * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", n);
    rocblas_stride stZ = argus.get<rocblas_stride>("strideZ", rocblas_stride(ldz) * n);
    rocblas_int mtype = argus.get<rocblas_int>("mtype", 0);

    rocsolver_schur_job job = char2rocsolver_schur_job(jobC);
    rocsolver_schur_vectors compz = char2rocsolver_schur_vectors(compzC);

    if(argus.alg_mode == 1)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_set_alg_mode(handle, rocsolver_function_hseqr, rocsolver_alg_mode_hybrid),
            rocblas_status_success);

        rocsolver_alg_mode alg_mode;
        EXPECT_ROCBLAS_STATUS(rocsolver_get_alg_mode(handle, rocsolver_function_hseqr, &alg_mode),
                              rocblas_status_success);

        EXPECT_EQ(alg_mode, rocsolver_alg_mode_hybrid);
    }
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    const bool wantz = (compz != rocsolver_schur_vectors_none);

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_H = size_t(ldh) * n;
    size_t size_W = size_t(n);
    size_t size_Z = wantz ? size_t(ldz) * n : 1;
    stZ = wantz ? stZ : 1;
    double max_error = 0, max_error_eig = 0, gpu_time_used = 0, cpu_time_used = 0;

    bool check = (argus.unit_check || argus.norm_check);
    size_t size_HRes = check ? size_H : 0;
    size_t size_WRes = check ? size_W : 0;
    size_t size_ZRes = check ? size_Z : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_HSEQR
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_hseqr(STRIDED, handle, job, compz, n, (rocblas_int*)nullptr,
                                (rocblas_int*)nullptr, (T* const*)nullptr, ldh, stH, (T*)nullptr,
                                stW, (T* const*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc),
                rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (T*)nullptr, ldh, stH, (T*)nullptr, stW,
                                                  (T*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || ldh < n || ldh < 1 || ldz < 1 || (wantz && ldz < n) || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_hseqr(STRIDED, handle, job, compz, n, (rocblas_int*)nullptr,
                                (rocblas_int*)nullptr, (T* const*)nullptr, ldh, stH, (T*)nullptr,
                                stW, (T* const*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (T*)nullptr, ldh, stH, (T*)nullptr, stW,
                                                  (T*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_hseqr(STRIDED, handle, job, compz, n, (rocblas_int*)nullptr,
                                              (rocblas_int*)nullptr, (T* const*)nullptr, ldh, stH,
                                              (T*)nullptr, stW, (T* const*)nullptr, ldz, stZ,
                                              (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_hseqr(STRIDED, handle, job, compz, n, (rocblas_int*)nullptr,
                                              (rocblas_int*)nullptr, (T*)nullptr, ldh, stH,
                                              (T*)nullptr, stW, (T*)nullptr, ldz, stZ,
                                              (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    host_strided_batch_vector<rocblas_int> hIlo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hIhi(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    host_strided_batch_vector<T> hWRes(size_WRes, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dIlo(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dIhi(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    device_strided_batch_vector<T> dW(size_W, 1, stW, bc);
    if(bc)
    {
        CHECK_HIP_ERROR(dIlo.memcheck());
        CHECK_HIP_ERROR(dIhi.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
    }
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hH0(size_H, 1, bc);
        host_batch_vector<T> hHRes(size_HRes, 1, bc);
        host_batch_vector<T> hZ0(size_Z, 1, bc);
        host_batch_vector<T> hZRes(size_ZRes, 1, bc);
        device_batch_vector<T> dH(size_H, 1, bc);
        device_batch_vector<T> dZ(size_Z, 1, bc);
        if(size_H)
            CHECK_HIP_ERROR(dH.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo.data(),
                                                  dIhi.data(), dH.data(), ldh, stH, dW.data(), stW,
                                                  dZ.data(), ldz, stZ, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(check)
            hseqr_getError<STRIDED, T>(handle, job, compz, n, dIlo, dIhi, dH, ldh, stH, dW, stW, dZ,
                                       ldz, stZ, dInfo, bc, hIlo, hIhi, hH0, hZ0, hHRes, hWRes,
                                       hZRes, hInfoRes, mtype, &max_error, &max_error_eig);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            hseqr_getPerfData<STRIDED, T>(handle, job, compz, n, dIlo, dIhi, dH, ldh, stH, dW, stW,
                                          dZ, ldz, stZ, dInfo, bc, hIlo, hIhi, hH0, hZ0, mtype,
                                          &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                          argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hH0(size_H, 1, stH, bc);
        host_strided_batch_vector<T> hHRes(size_HRes, 1, stH, bc);
        host_strided_batch_vector<T> hZ0(size_Z, 1, stZ, bc);
        host_strided_batch_vector<T> hZRes(size_ZRes, 1, stZ, bc);
        device_strided_batch_vector<T> dH(size_H, 1, stH, bc);
        device_strided_batch_vector<T> dZ(size_Z, 1, stZ, bc);
        if(size_H)
            CHECK_HIP_ERROR(dH.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_hseqr(STRIDED, handle, job, compz, n, dIlo.data(),
                                                  dIhi.data(), dH.data(), ldh, stH, dW.data(), stW,
                                                  dZ.data(), ldz, stZ, dInfo.data(), bc),
                                  rocblas_status_success);
            if(n == 0 && bc > 0)
            {
                CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
                for(rocblas_int b = 0; b < bc; ++b)
                    EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
            }
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(check)
            hseqr_getError<STRIDED, T>(handle, job, compz, n, dIlo, dIhi, dH, ldh, stH, dW, stW, dZ,
                                       ldz, stZ, dInfo, bc, hIlo, hIhi, hH0, hZ0, hHRes, hWRes,
                                       hZRes, hInfoRes, mtype, &max_error, &max_error_eig);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            hseqr_getPerfData<STRIDED, T>(handle, job, compz, n, dIlo, dIhi, dH, ldh, stH, dW, stW,
                                          dZ, ldz, stZ, dInfo, bc, hIlo, hIhi, hH0, hZ0, mtype,
                                          &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                          argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // - backward error and orthogonality,
    // - eigenvalues (well-conditioned classes only), relative to ||H||:
    // using max(n, 32) * machine_precision as tolerance (for small n, the rounding
    // errors of the individual reflections, which do not depend on n, dominate)
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, std::max(n, 32));
        ROCSOLVER_TEST_CHECK(T, max_error_eig, std::max(n, 32));
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        max_error = std::max(max_error, max_error_eig);
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("schur_job", "compz", "n", "ldh", "strideW", "ldz", "batch_c");
                rocsolver_bench_output(jobC, compzC, n, ldh, stW, ldz, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("schur_job", "compz", "n", "ldh", "strideH", "strideW",
                                       "ldz", "strideZ", "batch_c");
                rocsolver_bench_output(jobC, compzC, n, ldh, stH, stW, ldz, stZ, bc);
            }
            else
            {
                rocsolver_bench_output("schur_job", "compz", "n", "ldh", "ldz");
                rocsolver_bench_output(jobC, compzC, n, ldh, ldz);
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

#define EXTERN_TESTING_HSEQR(...) extern template void testing_hseqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_HSEQR, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
