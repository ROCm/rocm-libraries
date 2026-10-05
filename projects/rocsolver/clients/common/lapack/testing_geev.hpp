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

/** GEEV_GENMATRIX generates an n-by-n matrix A (with leading dimension n) according to mtype:
    - mtype = 0: random matrix with normally distributed entries (G).
    - mtype = 1: 1e-3 * G + I (clustered eigenvalues).
    - mtype = 2: D^(-1) * G * D with D = diag(10^k), k uniform in [-4, 4] ([-2, 2] in single
                 precision) (bad scaling).
    - mtype = 3: random upper Hessenberg matrix (ill-conditioned eigenvalues).
    - mtype = 4: random upper triangular matrix (GEBAL isolates all the eigenvalues).
    - mtype = 5: G * 1e300 (1e35 in single precision) (the matrix is scaled down).
    - mtype = 6: G * 1e-300 (1e-35 in single precision) (the matrix is scaled up).
    - mtype = 7: G with isolated eigenvalues and bad scaling, as gebal_genMatrix with
                 mtype = 2 (GEBAL gives ilo > 1 and ihi < n).
    - mtype = 8: G whose first k = geev_nan_k(n) columns are zero below the diagonal and
                 whose last k rows are zero left of the diagonal (so that GEBAL gives
                 ilo = k+1 and ihi = n-k), with a NaN in the active block: the QR algorithm
                 fails at once (info = n-k), and only the isolated eigenvalues (the diagonal
                 entries of those columns and rows) are computed.
    - mtype = 10: G whose last row is zero left of the diagonal (GEBAL gives ilo = 1 and
                  ihi = n-1), with a NaN in A(1, n-1) (0-based), outside the active block:
                  the eigenvalues are those of the NaN-free matrix and info = 0 (the
                  eigenvectors are not checked, as the NaN is in the coupling with the
                  isolated eigenvalue).
    Classes 0, 2, 4, 5, 6 and 7 have well-conditioned eigenvalues (in practice).
    The batch classes (see geev_member_class) are not generated here:
    - mtype = 9: mixed batch; the matrix b uses the class 0, 7 or 8 for b % 3 = 0, 1 or 2
                 (different ranges ilo:ihi in one batch, and a NaN next to clean matrices).
    - mtype = 11: mixed batch; the matrix b uses the class 10 if b is odd (or if the batch
                  has a single matrix) and the class 0 otherwise. **/
inline rocblas_int geev_nan_k(const rocblas_int n)
{
    return std::max(1, n / 10);
}

/** GEEV_MEMBER_CLASS returns the class of the matrix b of a batch of bc matrices of the
    test class mtype. **/
inline rocblas_int geev_member_class(const rocblas_int mtype, const rocblas_int b, const rocblas_int bc)
{
    if(mtype == 9)
        return (b % 3 == 0) ? 0 : (b % 3 == 1) ? 7 : 8;
    if(mtype == 11)
        return (b % 2 == 1 || bc == 1) ? 10 : 0;
    return mtype;
}

template <typename T>
void geev_genMatrix(const rocblas_int n,
                    std::vector<T>& A,
                    const rocblas_int mtype,
                    const rocblas_int seed)
{
    using S = decltype(std::real(T{}));
    constexpr bool single = std::is_same<S, float>::value;

    std::mt19937_64 rng(1000003ull * n + 7919ull * seed + 104729ull * mtype);
    std::normal_distribution<double> gauss(0.0, 1.0);
    auto rnd = [&](double s) {
        const double re = gauss(rng);
        const double im = gauss(rng);
        return T(S(s * re), S(s * im));
    };

    A.assign(size_t(n) * n, T(0));
    if(n == 0)
        return;

    double scale = 1.0;
    if(mtype == 5)
        scale = single ? 1e35 : 1e300;
    else if(mtype == 6)
        scale = single ? 1e-35 : 1e-300;

    for(rocblas_int j = 0; j < n; j++)
        for(rocblas_int i = 0; i < n; i++)
        {
            if((mtype == 3 && i > j + 1) || (mtype == 4 && i > j))
                continue;
            if(mtype == 8 && ((j < geev_nan_k(n) && i > j) || (i >= n - geev_nan_k(n) && j < i)))
                continue;
            if(mtype == 10 && i == n - 1 && j < i)
                continue;
            A[i + size_t(j) * n] = rnd(scale);
        }

    if(mtype == 1)
    {
        for(auto& a : A)
            a = a * T(S(1e-3));
        for(rocblas_int j = 0; j < n; j++)
            A[j + size_t(j) * n] = A[j + size_t(j) * n] + T(1);
    }
    else if(mtype == 2)
    {
        const double kmax = single ? 2.0 : 4.0;
        std::uniform_real_distribution<double> unif(-kmax, kmax);
        std::vector<double> d(n);
        for(rocblas_int i = 0; i < n; i++)
            d[i] = std::pow(10.0, unif(rng));
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
                A[i + size_t(j) * n] = A[i + size_t(j) * n] * T(S(d[j] / d[i]));
    }
    else if(mtype == 7)
    {
        gebal_genMatrix(n, A.data(), n, 2, seed);
    }
    else if(mtype == 8)
    {
        const S nan = std::numeric_limits<S>::quiet_NaN();
        A[n / 2 + size_t(n / 2) * n] = T(nan, S(0));
    }
    else if(mtype == 10 && n >= 3)
    {
        const S nan = std::numeric_limits<S>::quiet_NaN();
        A[1 + size_t(n - 1) * n] = T(nan, S(0));
    }
}

template <bool STRIDED, typename T, typename U>
void geev_checkBadArgs(const rocblas_handle handle,
                       const rocblas_evect jobvl,
                       const rocblas_evect jobvr,
                       const rocblas_int n,
                       U dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       T* dW,
                       const rocblas_stride stW,
                       U dVL,
                       const rocblas_int ldvl,
                       const rocblas_stride stVL,
                       U dVR,
                       const rocblas_int ldvr,
                       const rocblas_stride stVR,
                       rocblas_int* dInfo,
                       const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, nullptr, jobvl, jobvr, n, dA, lda, stA, dW, stW,
                                         dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, rocblas_evect(0), jobvr, n, dA, lda, stA,
                                         dW, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, rocblas_evect(0), n, dA, lda, stA,
                                         dW, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, rocblas_evect_tridiagonal, jobvr, n, dA, lda,
                                         stA, dW, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, rocblas_evect_tridiagonal, n, dA, lda,
                                         stA, dW, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dW,
                                             stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (U) nullptr, lda, stA,
                                         dW, stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, (T*)nullptr,
                                         stW, dVL, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dW, stW,
                                         (U) nullptr, ldvl, stVL, dVR, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dW, stW,
                                         dVL, ldvl, stVL, (U) nullptr, ldvr, stVR, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dW, stW,
                                         dVL, ldvl, stVL, dVR, ldvr, stVR, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, 0, (U) nullptr, lda, stA,
                                         (T*)nullptr, stW, (U) nullptr, ldvl, stVL, (U) nullptr,
                                         ldvr, stVR, dInfo, bc),
                          rocblas_status_success);
    // VL is not referenced when jobvl = none, and VR is not referenced when jobvr = none
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, rocblas_evect_none, jobvr, n, dA, lda,
                                         stA, dW, stW, (U) nullptr, ldvl, stVL, dVR, ldvr, stVR,
                                         dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, rocblas_evect_none, n, dA, lda,
                                         stA, dW, stW, dVL, ldvl, stVL, (U) nullptr, ldvr, stVR,
                                         dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, rocblas_evect_none, rocblas_evect_none, n,
                                         dA, lda, stA, dW, stW, (U) nullptr, ldvl, stVL,
                                         (U) nullptr, ldvr, stVR, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA, lda, stA, dW,
                                             stW, dVL, ldvl, stVL, dVR, ldvr, stVR,
                                             (rocblas_int*)nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geev_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect jobvl = rocblas_evect_original;
    rocblas_evect jobvr = rocblas_evect_original;
    rocblas_int n = 2;
    rocblas_int lda = 2;
    rocblas_int ldvl = 2;
    rocblas_int ldvr = 2;
    rocblas_stride stA = 4;
    rocblas_stride stW = 2;
    rocblas_stride stVL = 4;
    rocblas_stride stVR = 4;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_GEEV
    // some calls run the computation, so A is initialized
    const T a0[4] = {T(1, 1), T(3), T(2), T(4, -1)};
    device_strided_batch_vector<T> dW(2, 1, 2, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    if(BATCHED)
    {
        host_batch_vector<T> hA(4, 1, 1);
        for(rocblas_int k = 0; k < 4; k++)
            hA[0][k] = a0[k];
        device_batch_vector<T> dA(4, 1, 1);
        device_batch_vector<T> dVL(4, 1, 1);
        device_batch_vector<T> dVR(4, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        // check bad arguments
        geev_checkBadArgs<STRIDED>(handle, jobvl, jobvr, n, dA.data(), lda, stA, dW.data(), stW,
                                   dVL.data(), ldvl, stVL, dVR.data(), ldvr, stVR, dInfo.data(), bc);
    }
    else
    {
        host_strided_batch_vector<T> hA(4, 1, 4, 1);
        for(rocblas_int k = 0; k < 4; k++)
            hA[0][k] = a0[k];
        device_strided_batch_vector<T> dA(4, 1, 4, 1);
        device_strided_batch_vector<T> dVL(4, 1, 4, 1);
        device_strided_batch_vector<T> dVR(4, 1, 4, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        // check bad arguments
        geev_checkBadArgs<STRIDED>(handle, jobvl, jobvr, n, dA.data(), lda, stA, dW.data(), stW,
                                   dVL.data(), ldvl, stVL, dVR.data(), ldvr, stVR, dInfo.data(), bc);
    }
#endif
}

/** GEEV_GAPVALUE is the garbage set in the gaps between the members of a strided batch. **/
template <typename T>
T geev_gapValue(const size_t k, const int64_t b)
{
    using S = decltype(std::real(T{}));
    return T(S(k % 13) - S(6.5), S(b) + S(0.25));
}

/** GEEV_FILLGAPS sets, for a strided batch, the gaps used:stride-1 between the members of h
    (the entries after the used part of each member) to garbage, which must not be changed
    (nothing is done for a batch of pointers). **/
template <typename T, typename H>
void geev_fillGaps(H& h, const size_t used)
{
    if constexpr(std::is_same_v<H, host_strided_batch_vector<T>>)
        for(int64_t b = 0; b < h.batch_count(); b++)
            for(size_t k = used; k < size_t(h.stride()); k++)
                h[b][k] = geev_gapValue<T>(k, b);
}

/** GEEV_GAPSCHANGED returns the number of entries in the gaps between the members of the
    strided batch h that differ from the garbage set by geev_fillGaps. **/
template <typename T, typename H>
rocblas_int geev_gapsChanged(H& h, const size_t used)
{
    rocblas_int changed = 0;
    if constexpr(std::is_same_v<H, host_strided_batch_vector<T>>)
        for(int64_t b = 0; b < h.batch_count(); b++)
            for(size_t k = used; k < size_t(h.stride()); k++)
                changed += !(h[b][k] == geev_gapValue<T>(k, b));
    return changed;
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geev_initData(const rocblas_handle handle,
                   const rocblas_int n,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_int bc,
                   Th& hA,
                   const rocblas_int mtype)
{
    if(CPU)
    {
        std::vector<T> A;
        for(rocblas_int b = 0; b < bc; ++b)
        {
            geev_genMatrix(n, A, geev_member_class(mtype, b, bc), b);
            // (the padding, which must not be referenced, is set to garbage)
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < lda; i++)
                    hA[b][i + size_t(j) * lda] = (i < n) ? A[i + size_t(j) * n] : T(7, -3);
        }
        // (and so are the gaps between the members of a strided batch)
        geev_fillGaps<T>(hA, size_t(lda) * n);
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

// err = max(err, e), with any NaN or Inf counted as an error of 1
inline void geev_updateError(double& err, const double e)
{
    err = std::isfinite(e) ? std::max(err, e) : std::max(err, 1.0);
}

/** GEEV_VECTORERROR returns the error of the computed (right or left) eigenvectors in V.
    As = A / amax and Ws = W / amax (in double precision, with leading dimension n), where
    amax = max |a_ij|, so that the check neither overflows nor underflows. It is the maximum of
    - the residuals ||As v_j - Ws_j v_j|| / (||As||_F ||v_j||) for right vectors, or
      ||u_j^H As - Ws_j u_j^H|| / (||As||_F ||u_j||) for left vectors,
    - the normalization errors | ||v_j|| - 1 |,
    - 1 if the entry of largest modulus of v_j (up to rounding) is not real (imaginary part
      exactly zero), or if v_j is zero or not finite. **/
template <typename T>
double geev_vectorError(const bool left,
                        const rocblas_int n,
                        std::vector<rocblas_double_complex>& As,
                        const double normA,
                        std::vector<rocblas_double_complex>& Ws,
                        T* V,
                        const rocblas_int ldv)
{
    using S = decltype(std::real(T{}));
    using Z = rocblas_double_complex;
    const double eps = std::numeric_limits<S>::epsilon();
    const size_t nn = size_t(n) * n;
    double err = 0;

    // normalization
    std::vector<Z> X(nn), R(nn);
    for(rocblas_int j = 0; j < n; j++)
    {
        double vnorm = 0, vmax = 0;
        for(rocblas_int i = 0; i < n; i++)
        {
            const T v = V[i + size_t(j) * ldv];
            const double re = double(std::real(v)), im = double(std::imag(v));
            X[i + size_t(j) * n] = Z(re, im);
            vnorm += re * re + im * im;
            vmax = std::max(vmax, re * re + im * im);
        }
        if(!std::isfinite(vnorm) || vnorm == 0)
        {
            err = 1;
            continue;
        }
        geev_updateError(err, std::abs(std::sqrt(vnorm) - 1));

        bool real_max = false;
        for(rocblas_int i = 0; i < n; i++)
        {
            const T v = V[i + size_t(j) * ldv];
            const double re = double(std::real(v)), im = double(std::imag(v));
            if(std::imag(v) == S(0) && re * re + im * im >= vmax * (1 - 8 * eps))
                real_max = true;
        }
        if(!real_max)
            err = 1;
    }

    // R = As X - X diag(Ws), or R = As^H X - X conj(diag(Ws))
    cpu_gemm(left ? rocblas_operation_conjugate_transpose : rocblas_operation_none,
             rocblas_operation_none, n, n, n, Z(1), As.data(), n, X.data(), n, Z(0), R.data(), n);
    const double anorm = normA > 0 ? normA : 1.0;
    for(rocblas_int j = 0; j < n; j++)
    {
        const Z lambda = left ? std::conj(Ws[j]) : Ws[j];
        double rnorm = 0, xnorm = 0;
        for(rocblas_int i = 0; i < n; i++)
        {
            const Z x = X[i + size_t(j) * n];
            const Z r = R[i + size_t(j) * n] - lambda * x;
            rnorm += std::real(r) * std::real(r) + std::imag(r) * std::imag(r);
            xnorm += std::real(x) * std::real(x) + std::imag(x) * std::imag(x);
        }
        if(xnorm == 0 || !std::isfinite(xnorm) || !std::isfinite(rnorm))
            err = 1;
        else
            geev_updateError(err, std::sqrt(rnorm) / (anorm * std::sqrt(xnorm)));
    }

    return err;
}

template <bool STRIDED, typename T, typename Td, typename Wd, typename Ud, typename Th, typename Wh, typename Uh>
void geev_getError(const rocblas_handle handle,
                   const rocblas_evect jobvl,
                   const rocblas_evect jobvr,
                   const rocblas_int n,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Wd& dW,
                   const rocblas_stride stW,
                   Td& dVL,
                   const rocblas_int ldvl,
                   const rocblas_stride stVL,
                   Td& dVR,
                   const rocblas_int ldvr,
                   const rocblas_stride stVR,
                   Ud& dInfo,
                   const rocblas_int bc,
                   Th& hA,
                   Wh& hWRes,
                   Th& hVLRes,
                   Th& hVRRes,
                   Uh& hInfoRes,
                   const rocblas_int mtype,
                   double* max_err,
                   double* max_err_eig)
{
    using Z = rocblas_double_complex;
    const bool leftv = (jobvl == rocblas_evect_original);
    const bool rightv = (jobvr == rocblas_evect_original);

    // input data initialization
    geev_initData<true, true, T>(handle, n, dA, lda, bc, hA, mtype);

    // garbage in the gaps between the members of the strided batches W, VL and VR
    // (strides larger than n or ld*n), which must not be changed
    geev_fillGaps<T>(hWRes, size_t(n));
    CHECK_HIP_ERROR(dW.transfer_from(hWRes));
    if(leftv)
    {
        geev_fillGaps<T>(hVLRes, size_t(ldvl) * n);
        CHECK_HIP_ERROR(dVL.transfer_from(hVLRes));
    }
    if(rightv)
    {
        geev_fillGaps<T>(hVRRes, size_t(ldvr) * n);
        CHECK_HIP_ERROR(dVR.transfer_from(hVRRes));
    }

    // execute computations
    // GPU lapack
    // (hA keeps the original matrices; A is destroyed on the device and is not checked,
    // except for the gaps between the members of a strided batch)
    CHECK_ROCBLAS_ERROR(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                       dW.data(), stW, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                                       stVR, dInfo.data(), bc));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(leftv)
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
    if(rightv)
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));

    rocblas_int gaps = geev_gapsChanged<T>(hWRes, size_t(n));
    if(leftv)
        gaps += geev_gapsChanged<T>(hVLRes, size_t(ldvl) * n);
    if(rightv)
        gaps += geev_gapsChanged<T>(hVRRes, size_t(ldvr) * n);
    if constexpr(std::is_same_v<Th, host_strided_batch_vector<T>>)
    {
        host_strided_batch_vector<T> hARes(hA.n(), 1, hA.stride(), bc);
        CHECK_HIP_ERROR(hARes.transfer_from(dA));
        gaps += geev_gapsChanged<T>(hARes, size_t(lda) * n);
    }
    EXPECT_EQ(gaps, 0);

    // the eigenvectors are not compared with the host results (they can be ill-conditioned and
    // are only defined up to a unimodular factor); the error is based on the residuals instead
    // (see geev_vectorError). The eigenvalues are compared with the host LAPACK only for the
    // classes with well-conditioned eigenvalues; for all classes, their sum must match the trace.
    // In a mixed batch, each matrix is checked according to its own class.
    const size_t nn = size_t(n) * n;
    std::vector<Z> As(nn), Ws(n);
    std::vector<T> Ac(nn), Wc(n), VLc(1), VRc(1);
    *max_err = gaps;
    *max_err_eig = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        const rocblas_int mt = geev_member_class(mtype, b, bc);
        const bool nan_coupling = (mt == 10 && n >= 3);
        const bool compare_eigenvalues
            = (mt == 0 || mt == 2 || mt == 4 || mt == 5 || mt == 6 || mt == 7 || nan_coupling);

        if(mt == 8)
        {
            // NaN in the active block: info = ihi = n-k, and the isolated eigenvalues W(1:k)
            // and W(n-k+1:n) are the diagonal entries of the isolated columns and rows (in
            // some order); the eigenvectors are undefined
            const rocblas_int k = geev_nan_k(n);
            EXPECT_EQ(hInfoRes[b][0], n - k) << "where b = " << b;
            if(hInfoRes[b][0] != n - k)
                *max_err += 1;
            std::vector<T> ref, got;
            for(rocblas_int i = 0; i < n; i++)
                if(i < k || i >= n - k)
                {
                    ref.push_back(hA[b][i + size_t(i) * lda]);
                    got.push_back(hWRes[b][i]);
                }
            for(const T& w : got)
            {
                auto it = std::find(ref.begin(), ref.end(), w);
                if(it == ref.end())
                    *max_err += 1;
                else
                    ref.erase(it);
            }
            continue;
        }

        EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
        if(hInfoRes[b][0] != 0)
        {
            // the eigenvectors are undefined
            *max_err += 1;
            continue;
        }

        // As = A / amax, Ws = W / amax
        // (with a NaN outside the active block, A is the NaN-free matrix with the NaN
        // replaced by zero, which has the same eigenvalues)
        auto entry = [&](const rocblas_int i, const rocblas_int j) -> T {
            const T a = hA[b][i + size_t(j) * lda];
            return (nan_coupling && std::isnan(std::real(a))) ? T(0) : a;
        };
        double amax = 0;
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
            {
                const T a = entry(i, j);
                amax = std::max(amax, std::hypot(double(std::real(a)), double(std::imag(a))));
            }
        if(!(amax > 0))
            amax = 1;
        double normA = 0;
        Z trace = Z(0);
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
            {
                const T a = entry(i, j);
                const Z as = Z(double(std::real(a)) / amax, double(std::imag(a)) / amax);
                As[i + size_t(j) * n] = as;
                normA += std::real(as) * std::real(as) + std::imag(as) * std::imag(as);
                if(i == j)
                    trace = trace + as;
            }
        normA = std::sqrt(normA);
        const double anorm = normA > 0 ? normA : 1.0;
        Z wsum = Z(0);
        for(rocblas_int j = 0; j < n; j++)
        {
            const T w = hWRes[b][j];
            Ws[j] = Z(double(std::real(w)) / amax, double(std::imag(w)) / amax);
            wsum = wsum + Ws[j];
        }

        // sum of the eigenvalues: |sum_j Ws_j - trace(As)| / (sqrt(n) ||As||_F)
        // (well-conditioned for all classes: |trace(E)| <= sqrt(n) ||E||_F)
        geev_updateError(*max_err_eig, std::abs(wsum - trace) / (std::sqrt(double(n)) * anorm));

        // (with a NaN outside the active block, the eigenvectors are not checked)
        if(rightv && !nan_coupling)
            geev_updateError(*max_err, geev_vectorError(false, n, As, normA, Ws, hVRRes[b], ldvr));
        if(leftv && !nan_coupling)
            geev_updateError(*max_err, geev_vectorError(true, n, As, normA, Ws, hVLRes[b], ldvl));

        if(compare_eigenvalues)
        {
            // host LAPACK eigenvalues
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    Ac[i + size_t(j) * n] = entry(i, j);
            rocblas_int info;
            cpu_geev(rocblas_evect_none, rocblas_evect_none, n, Ac.data(), n, Wc.data(), VLc.data(),
                     1, VRc.data(), 1, &info);
            EXPECT_EQ(info, 0) << "(host LAPACK) where b = " << b;

            // for the badly scaled classes, the eigenvalues are only determined to about
            // eps * ||B||_F, with B = D^(-1) * A * D balanced by GEBAL, which can be much
            // smaller than ||A||_F: the errors are relative to ||B||_F
            double enorm = anorm;
            if(mt == 2 || mt == 7)
            {
                std::vector<T> Bc(size_t(n) * n);
                std::vector<decltype(std::real(T{}))> scale(n);
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        Bc[i + size_t(j) * n] = entry(i, j);
                rocblas_int bilo, bihi, binfo;
                cpu_gebal(rocsolver_balance_both, n, Bc.data(), n, &bilo, &bihi, scale.data(),
                          &binfo);
                double bnorm = 0;
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        bnorm += std::norm(Bc[i + size_t(j) * n]);
                bnorm = std::sqrt(bnorm) / amax;
                if(binfo == 0 && bnorm > 0)
                    enorm = std::min(enorm, bnorm);
            }

            // match each host eigenvalue with the closest unmatched device eigenvalue;
            // error is max |lambda - lambda_ref| / ||A||_F (or ||B||_F, see above; computed
            // with the scaled values)
            std::vector<bool> used(n, false);
            for(rocblas_int j = 0; j < n; j++)
            {
                const Z wref = Z(double(std::real(Wc[j])) / amax, double(std::imag(Wc[j])) / amax);
                double best = std::numeric_limits<double>::infinity();
                rocblas_int kbest = -1;
                for(rocblas_int k = 0; k < n; k++)
                {
                    if(used[k])
                        continue;
                    double d = std::abs(wref - Ws[k]);
                    if(d < best)
                    {
                        best = d;
                        kbest = k;
                    }
                }
                if(kbest < 0)
                {
                    // (NaN eigenvalues)
                    *max_err_eig = std::max(*max_err_eig, 1.0);
                    continue;
                }
                used[kbest] = true;
                geev_updateError(*max_err_eig, best / enorm);
            }
        }
    }
}

template <bool STRIDED, typename T, typename Td, typename Wd, typename Ud, typename Th>
void geev_getPerfData(const rocblas_handle handle,
                      const rocblas_evect jobvl,
                      const rocblas_evect jobvr,
                      const rocblas_int n,
                      Td& dA,
                      const rocblas_int lda,
                      const rocblas_stride stA,
                      Wd& dW,
                      const rocblas_stride stW,
                      Td& dVL,
                      const rocblas_int ldvl,
                      const rocblas_stride stVL,
                      Td& dVR,
                      const rocblas_int ldvr,
                      const rocblas_stride stVR,
                      Ud& dInfo,
                      const rocblas_int bc,
                      Th& hA,
                      const rocblas_int mtype,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const int profile,
                      const bool profile_kernels,
                      const bool perf)
{
    const bool leftv = (jobvl == rocblas_evect_original);
    const bool rightv = (jobvr == rocblas_evect_original);
    rocblas_int info;

    if(!perf)
    {
        geev_initData<true, false, T>(handle, n, dA, lda, bc, hA, mtype);

        std::vector<T> W(std::max(n, 1));
        std::vector<T> VL(leftv ? size_t(ldvl) * n : 1);
        std::vector<T> VR(rightv ? size_t(ldvr) * n : 1);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_geev(jobvl, jobvr, n, hA[b], lda, W.data(), VL.data(), ldvl, VR.data(), ldvr, &info);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    geev_initData<true, false, T>(handle, n, dA, lda, bc, hA, mtype);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geev_initData<false, true, T>(handle, n, dA, lda, bc, hA, mtype);

        CHECK_ROCBLAS_ERROR(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA,
                                           dW.data(), stW, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                                           stVR, dInfo.data(), bc));
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
        geev_initData<false, true, T>(handle, n, dA, lda, bc, hA, mtype);

        timer.start(stream);
        rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda, stA, dW.data(), stW,
                       dVL.data(), ldvl, stVL, dVR.data(), ldvr, stVR, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geev(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char jobvlC = argus.get<char>("jobvl");
    char jobvrC = argus.get<char>("jobvr");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", n);
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", rocblas_stride(lda) * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", n);
    rocblas_stride stVL = argus.get<rocblas_stride>("strideVL", rocblas_stride(ldvl) * n);
    rocblas_stride stVR = argus.get<rocblas_stride>("strideVR", rocblas_stride(ldvr) * n);
    rocblas_int mtype = argus.get<rocblas_int>("mtype", 0);

    rocblas_evect jobvl = char2rocblas_evect(jobvlC);
    rocblas_evect jobvr = char2rocblas_evect(jobvrC);

    if(argus.alg_mode == 1)
    {
        // GEEV follows the algorithm mode of HSEQR
        EXPECT_ROCBLAS_STATUS(
            rocsolver_set_alg_mode(handle, rocsolver_function_geev, rocsolver_alg_mode_hybrid),
            rocblas_status_success);

        rocsolver_alg_mode alg_mode;
        EXPECT_ROCBLAS_STATUS(rocsolver_get_alg_mode(handle, rocsolver_function_geev, &alg_mode),
                              rocblas_status_success);
        EXPECT_EQ(alg_mode, rocsolver_alg_mode_hybrid);

        EXPECT_ROCBLAS_STATUS(rocsolver_get_alg_mode(handle, rocsolver_function_hseqr, &alg_mode),
                              rocblas_status_success);
        EXPECT_EQ(alg_mode, rocsolver_alg_mode_hybrid);
    }
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    const bool leftv = (jobvl == rocblas_evect_original);
    const bool rightv = (jobvr == rocblas_evect_original);

    // check non-supported values
    if((jobvl != rocblas_evect_original && jobvl != rocblas_evect_none)
       || (jobvr != rocblas_evect_original && jobvr != rocblas_evect_none))
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n,
                                                 (T* const*)nullptr, lda, stA, (T*)nullptr, stW,
                                                 (T* const*)nullptr, ldvl, stVL, (T* const*)nullptr,
                                                 ldvr, stVR, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda,
                                                 stA, (T*)nullptr, stW, (T*)nullptr, ldvl, stVL,
                                                 (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_W = size_t(n);
    size_t size_VL = leftv ? size_t(ldvl) * n : 1;
    size_t size_VR = rightv ? size_t(ldvr) * n : 1;
    stVL = leftv ? stVL : 1;
    stVR = rightv ? stVR : 1;
    double max_error = 0, max_error_eig = 0, gpu_time_used = 0, cpu_time_used = 0;

    bool check = (argus.unit_check || argus.norm_check);
    size_t size_WRes = check ? size_W : 0;
    size_t size_VLRes = check ? size_VL : 0;
    size_t size_VRRes = check ? size_VR : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_GEEV
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n,
                                                 (T* const*)nullptr, lda, stA, (T*)nullptr, stW,
                                                 (T* const*)nullptr, ldvl, stVL, (T* const*)nullptr,
                                                 ldvr, stVR, (rocblas_int*)nullptr, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda,
                                                 stA, (T*)nullptr, stW, (T*)nullptr, ldvl, stVL,
                                                 (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || lda < 1 || ldvl < 1 || ldvr < 1 || (leftv && ldvl < n)
                         || (rightv && ldvr < n) || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n,
                                                 (T* const*)nullptr, lda, stA, (T*)nullptr, stW,
                                                 (T* const*)nullptr, ldvl, stVL, (T* const*)nullptr,
                                                 ldvr, stVR, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda,
                                                 stA, (T*)nullptr, stW, (T*)nullptr, ldvl, stVL,
                                                 (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T* const*)nullptr,
                                             lda, stA, (T*)nullptr, stW, (T* const*)nullptr, ldvl,
                                             stVL, (T* const*)nullptr, ldvr, stVR,
                                             (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, (T*)nullptr, lda,
                                             stA, (T*)nullptr, stW, (T*)nullptr, ldvl, stVL,
                                             (T*)nullptr, ldvr, stVR, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    host_strided_batch_vector<T> hWRes(size_WRes, 1, stW, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    device_strided_batch_vector<T> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(bc)
        CHECK_HIP_ERROR(dInfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hVLRes(size_VLRes, 1, bc);
        host_batch_vector<T> hVRRes(size_VRRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dVL(size_VL, 1, bc);
        device_batch_vector<T> dVR(size_VR, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda,
                                                 stA, dW.data(), stW, dVL.data(), ldvl, stVL,
                                                 dVR.data(), ldvr, stVR, dInfo.data(), bc),
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
            geev_getError<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dW, stW, dVL, ldvl,
                                      stVL, dVR, ldvr, stVR, dInfo, bc, hA, hWRes, hVLRes, hVRRes,
                                      hInfoRes, mtype, &max_error, &max_error_eig);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            geev_getPerfData<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dW, stW, dVL, ldvl,
                                         stVL, dVR, ldvr, stVR, dInfo, bc, hA, mtype,
                                         &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                         argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hVLRes(size_VLRes, 1, stVL, bc);
        host_strided_batch_vector<T> hVRRes(size_VRRes, 1, stVR, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dVL(size_VL, 1, stVL, bc);
        device_strided_batch_vector<T> dVR(size_VR, 1, stVR, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_geev(STRIDED, handle, jobvl, jobvr, n, dA.data(), lda,
                                                 stA, dW.data(), stW, dVL.data(), ldvl, stVL,
                                                 dVR.data(), ldvr, stVR, dInfo.data(), bc),
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
            geev_getError<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dW, stW, dVL, ldvl,
                                      stVL, dVR, ldvr, stVR, dInfo, bc, hA, hWRes, hVLRes, hVRRes,
                                      hInfoRes, mtype, &max_error, &max_error_eig);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            geev_getPerfData<STRIDED, T>(handle, jobvl, jobvr, n, dA, lda, stA, dW, stW, dVL, ldvl,
                                         stVL, dVR, ldvr, stVR, dInfo, bc, hA, mtype,
                                         &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                         argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // - residuals and normalization of the eigenvectors (and info = 0),
    // - eigenvalues (sum for all classes, host comparison for well-conditioned classes only),
    //   relative to ||A||_F:
    // using max(n, 32) * machine_precision as tolerance
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
                rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "strideW", "ldvl", "ldvr",
                                       "batch_c");
                rocsolver_bench_output(jobvlC, jobvrC, n, lda, stW, ldvl, ldvr, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "strideA", "strideW", "ldvl",
                                       "strideVL", "ldvr", "strideVR", "batch_c");
                rocsolver_bench_output(jobvlC, jobvrC, n, lda, stA, stW, ldvl, stVL, ldvr, stVR, bc);
            }
            else
            {
                rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "ldvl", "ldvr");
                rocsolver_bench_output(jobvlC, jobvrC, n, lda, ldvl, ldvr);
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

#define EXTERN_TESTING_GEEV(...) extern template void testing_geev<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEEV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
