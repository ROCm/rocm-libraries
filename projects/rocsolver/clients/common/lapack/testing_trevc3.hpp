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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, typename U>
void trevc3_checkBadArgs(const rocblas_handle handle,
                         const rocblas_side side,
                         const rocsolver_eigenvectors howmny,
                         const rocblas_int n,
                         U dT,
                         const rocblas_int ldt,
                         const rocblas_stride stT,
                         U dVL,
                         const rocblas_int ldvl,
                         const rocblas_stride stVL,
                         U dVR,
                         const rocblas_int ldvr,
                         const rocblas_stride stVR,
                         const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, nullptr, side, howmny, n, dT, ldt, stT, dVL,
                                           ldvl, stVL, dVR, ldvr, stVR, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, rocblas_side(0), howmny, n, dT, ldt,
                                           stT, dVL, ldvl, stVL, dVR, ldvr, stVR, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, rocsolver_eigenvectors(0), n, dT,
                                           ldt, stT, dVL, ldvl, stVL, dVR, ldvr, stVR, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT, ldt, stT, dVL,
                                               ldvl, stVL, dVR, ldvr, stVR, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, (U) nullptr, ldt, stT,
                                           dVL, ldvl, stVL, dVR, ldvr, stVR, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT, ldt, stT,
                                           (U) nullptr, ldvl, stVL, dVR, ldvr, stVR, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT, ldt, stT, dVL,
                                           ldvl, stVL, (U) nullptr, ldvr, stVR, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, 0, (U) nullptr, ldt, stT,
                                           (U) nullptr, ldvl, stVL, (U) nullptr, ldvr, stVR, bc),
                          rocblas_status_success);
    // VL is not referenced when side = right, and VR is not referenced when side = left
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, rocblas_side_right, howmny, n, dT, ldt,
                                           stT, (U) nullptr, ldvl, stVL, dVR, ldvr, stVR, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, rocblas_side_left, howmny, n, dT, ldt,
                                           stT, dVL, ldvl, stVL, (U) nullptr, ldvr, stVR, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT, ldt, stT, dVL,
                                               ldvl, stVL, dVR, ldvr, stVR, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_trevc3_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_both;
    rocsolver_eigenvectors howmny = rocsolver_eigenvectors_backtransform;
    rocblas_int n = 2;
    rocblas_int ldt = 2;
    rocblas_int ldvl = 2;
    rocblas_int ldvr = 2;
    rocblas_stride stT = 4;
    rocblas_stride stVL = 4;
    rocblas_stride stVR = 4;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_TREVC
    if(BATCHED)
    {
        device_batch_vector<T> dT(4, 1, 1);
        device_batch_vector<T> dVL(4, 1, 1);
        device_batch_vector<T> dVR(4, 1, 1);
        CHECK_HIP_ERROR(dT.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());

        // check bad arguments
        trevc3_checkBadArgs<STRIDED>(handle, side, howmny, n, dT.data(), ldt, stT, dVL.data(), ldvl,
                                     stVL, dVR.data(), ldvr, stVR, bc);
    }
    else
    {
        device_strided_batch_vector<T> dT(4, 1, 4, 1);
        device_strided_batch_vector<T> dVL(4, 1, 4, 1);
        device_strided_batch_vector<T> dVR(4, 1, 4, 1);
        CHECK_HIP_ERROR(dT.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());

        // check bad arguments
        trevc3_checkBadArgs<STRIDED>(handle, side, howmny, n, dT.data(), ldt, stT, dVL.data(), ldvl,
                                     stVL, dVR.data(), ldvr, stVR, bc);
    }
#endif
}

/** TREVC3_FILLGAPS sets, for a strided batch, the gaps used:stride-1 between the members of
    h (the entries after the used part of each member) to garbage, which must not be changed
    (nothing is done for a batch of pointers). **/
template <typename T, typename Th>
void trevc3_fillGaps(Th& h, const size_t used)
{
    using S = decltype(std::real(T{}));
    if constexpr(std::is_same_v<Th, host_strided_batch_vector<T>>)
        for(int64_t b = 0; b < h.batch_count(); b++)
            for(size_t k = used; k < size_t(h.stride()); k++)
                h[b][k] = T(S(k % 13) - S(6.5), S(b) + S(0.25));
}

/** TREVC3_GAPSCHANGED returns the number of entries in the gaps between the members of the
    strided batch hRes (see trevc3_fillGaps) that differ from those in h. **/
template <typename T, typename Th>
rocblas_int trevc3_gapsChanged(Th& hRes, Th& h, const size_t used)
{
    rocblas_int changed = 0;
    if constexpr(std::is_same_v<Th, host_strided_batch_vector<T>>)
        for(int64_t b = 0; b < h.batch_count(); b++)
            for(size_t k = used; k < size_t(h.stride()); k++)
                changed += !(hRes[b][k] == h[b][k]);
    return changed;
}

/** TREVC3_INITDATA generates an n-by-n upper triangular matrix T according to mtype:
    - mtype = 0: random T (normally distributed entries).
    - mtype = 1: clustered diagonal, T(j,j) = 1 + 1e-3 * random (strong growth in the
                 back substitution; exercises the overflow control).
    - mtype = 2: repeated and zero eigenvalues, T(j,j) in {0, 1, i} (exercises the
                 perturbation of small pivots).
    - mtype = 3: Schur form of a random upper Hessenberg matrix (computed with the host HSEQR).
    - mtype = 4: bidiagonal T with superdiagonal 1 and diagonal (-1, ..., -1, 0, 0, 0), and Q
                 32 times a reflector whose first row is (1, ..., 1) / sqrt(n): the vector of
                 the last eigenvalue reaches the bound of the solve, and Q x overflows unless
                 the vectors are scaled before the back-transformation.
    - mtype = 5: off-diagonal entries of size about huge/8, and T(0,1) = (0.75, 0.75) * huge
                 (|Re| + |Im| of T(0,1), and the row and column sums, overflow).
    The entries below the diagonal of T (and in the padding) are random garbage, as they must
    not be referenced; so are the gaps between the members of a strided batch, for T, VL and
    VR (strides larger than ld*n). If howmny = backtransform, the input VL and/or VR is a
    unitary matrix Q (the Schur vectors for mtype = 3; 32 times a reflector for mtype = 4; a
    random unitary matrix otherwise), which is also kept in hQ (with leading dimension n); the
    residuals are invariant under the scaling of Q. If howmny = all, VL and VR are initialized
    with garbage. **/
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void trevc3_initData(const rocblas_handle handle,
                     const rocblas_side side,
                     const rocsolver_eigenvectors howmny,
                     const rocblas_int n,
                     Td& dT,
                     const rocblas_int ldt,
                     Td& dVL,
                     const rocblas_int ldvl,
                     Td& dVR,
                     const rocblas_int ldvr,
                     const rocblas_int bc,
                     Th& hT,
                     Th& hVL,
                     Th& hVR,
                     std::vector<T>& hQ,
                     const rocblas_int mtype)
{
    using S = decltype(std::real(T{}));
    const bool leftv = (side == rocblas_side_left || side == rocblas_side_both);
    const bool rightv = (side == rocblas_side_right || side == rocblas_side_both);
    const bool over = (howmny == rocsolver_eigenvectors_backtransform);

    if(CPU)
    {
        const size_t nn = size_t(n) * n;
        hQ.assign(over ? nn * bc : 0, T(0));
        std::vector<T> A(nn), W(std::max(n, 1)), tau(std::max(n, 1));
        std::vector<T> work(size_t(std::max(n, 1)) * 64);
        rocblas_int info;

        for(rocblas_int b = 0; b < bc; ++b)
        {
            std::mt19937_64 rng(1000003ull * n + 7919ull * b + 104729ull * mtype);
            std::normal_distribution<double> gauss(0.0, 1.0);
            auto rnd = [&]() { return T(S(gauss(rng)), S(gauss(rng))); };
            T* Q = over ? hQ.data() + nn * b : nullptr;

            // matrix T (with garbage below the diagonal and in the padding)
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < ldt; i++)
                    hT[b][i + size_t(j) * ldt] = rnd();

            if(mtype == 1)
            {
                for(rocblas_int j = 0; j < n; j++)
                    hT[b][j + size_t(j) * ldt] = T(1) + S(1e-3) * rnd();
            }
            else if(mtype == 2)
            {
                const T values[3] = {T(0), T(1), T(0, 1)};
                for(rocblas_int j = 0; j < n; j++)
                {
                    rocblas_int k = rng() % 4;
                    hT[b][j + size_t(j) * ldt] = values[k < 2 ? 0 : k - 1];
                }
            }
            else if(mtype == 4)
            {
                // bidiagonal: superdiagonal 1, diagonal -1 except the last three entries 0
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i <= j; i++)
                        hT[b][i + size_t(j) * ldt] = (i == j) ? T(j >= n - 3 ? 0 : -1)
                            : (i == j - 1)                    ? T(1)
                                                              : T(0);
            }
            else if(mtype == 5)
            {
                // huge off-diagonal entries, with T(0,1) = (0.75, 0.75) * huge: the row and
                // column sums of |Re| + |Im| (even |Re| + |Im| of T(0,1)) overflow
                const S huge = std::numeric_limits<S>::max();
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < j; i++)
                    {
                        T t = rnd();
                        const S re = std::max(S(-4), std::min(S(4), std::real(t)));
                        const S im = std::max(S(-4), std::min(S(4), std::imag(t)));
                        hT[b][i + size_t(j) * ldt] = T(re * (huge / 8), im * (huge / 8));
                    }
                if(n > 1)
                    hT[b][0 + size_t(1) * ldt] = T(S(0.75) * huge, S(0.75) * huge);
            }
            else if(mtype == 3)
            {
                // random upper Hessenberg matrix reduced to Schur form
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        A[i + size_t(j) * n] = (i <= j + 1) ? rnd() : T(0);
                std::vector<T> Z(std::max(nn, size_t(1)));
                cpu_hseqr(rocsolver_schur_form, rocsolver_schur_vectors_initialize, n, 1, n, A.data(),
                          n, W.data(), Z.data(), n, work.data(), (rocblas_int)work.size(), &info);
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i <= j; i++)
                        hT[b][i + size_t(j) * ldt] = A[i + size_t(j) * n];
                if(over)
                    for(size_t k = 0; k < nn; k++)
                        Q[k] = Z[k];
            }

            if(over && mtype == 4)
            {
                // 32 times the reflector I - 2 v v^H / (v^H v), v = e_1 - u, whose first row
                // is u^H = (1, ..., 1) / sqrt(n): the entries of the vector of the last
                // eigenvalue reach the solve's bound xbig and add up in (Q x)_1
                std::vector<double> v(n, -1.0 / std::sqrt(double(n)));
                v[0] += 1.0;
                double vv = 0;
                for(rocblas_int i = 0; i < n; i++)
                    vv += v[i] * v[i];
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                    {
                        const double hij = (i == j ? 1.0 : 0.0) - (vv > 0 ? 2 * v[i] * v[j] / vv : 0);
                        Q[i + size_t(j) * n] = T(S(32 * hij));
                    }
            }

            // random unitary Q
            if(over && mtype != 3 && mtype != 4)
            {
                for(size_t k = 0; k < nn; k++)
                    A[k] = rnd();
                cpu_geqrf(n, n, A.data(), n, tau.data(), work.data(), (rocblas_int)work.size());
                cpu_orgqr_ungqr(n, n, n, A.data(), n, tau.data(), work.data(),
                                (rocblas_int)work.size());
                for(size_t k = 0; k < nn; k++)
                    Q[k] = A[k];
            }

            // input VL and VR (Q, or garbage if howmny = all)
            if(leftv)
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < ldvl; i++)
                        hVL[b][i + size_t(j) * ldvl] = (over && i < n) ? Q[i + size_t(j) * n] : rnd();
            if(rightv)
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < ldvr; i++)
                        hVR[b][i + size_t(j) * ldvr] = (over && i < n) ? Q[i + size_t(j) * n] : rnd();
        }

        // garbage in the gaps between the members of a strided batch
        trevc3_fillGaps<T>(hT, size_t(ldt) * n);
        if(leftv)
            trevc3_fillGaps<T>(hVL, size_t(ldvl) * n);
        if(rightv)
            trevc3_fillGaps<T>(hVR, size_t(ldvr) * n);
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dT.transfer_from(hT));
        if(leftv)
            CHECK_HIP_ERROR(dVL.transfer_from(hVL));
        if(rightv)
            CHECK_HIP_ERROR(dVR.transfer_from(hVR));
    }
}

/** TREVC3_VECTORERROR returns the error of the computed (right or left) eigenvectors
    in V. With X = V (or X = Q^H V if back-transformed), it is the maximum of
    - the residuals ||T x_j - T(j,j) x_j|| / (||T||_F ||x_j||) for right vectors, or
      ||x_j^H T - T(j,j) x_j^H|| / (||T||_F ||x_j||) for left vectors,
    - the normalization errors |max_i (|Re v_ij| + |Im v_ij|) - 1|,
    - 1 if howmny = all and the entries of x_j below (right) or above (left) row j
      are not zero, or if x_j is zero or not finite.
    Tu is the upper triangular part of T, in double precision, with leading dimension n. **/
template <typename T>
double trevc3_vectorError(const bool left,
                          const bool over,
                          const rocblas_int n,
                          std::vector<rocblas_double_complex>& Tu,
                          const double normT,
                          T* Q,
                          T* V,
                          const rocblas_int ldv)
{
    using Z = rocblas_double_complex;
    auto dbl = [](T z) { return Z(double(std::real(z)), double(std::imag(z))); };
    const size_t nn = size_t(n) * n;
    double err = 0;

    // normalization and zero structure
    for(rocblas_int j = 0; j < n; j++)
    {
        double vmax = 0;
        for(rocblas_int i = 0; i < n; i++)
        {
            const T v = V[i + size_t(j) * ldv];
            // (NaN-aware: a NaN entry makes vmax NaN, and it stays NaN)
            const double a = std::abs(double(std::real(v))) + std::abs(double(std::imag(v)));
            if(std::isnan(a) || a > vmax)
                vmax = a;
            if(!over && (left ? i < j : i > j) && !(v == T(0)))
                err = 1;
        }
        if(!std::isfinite(vmax))
            err = 1;
        else
            err = std::max(err, std::abs(vmax - 1));
    }

    // X = V, or X = Q^H V
    std::vector<Z> X(nn), R(nn);
    for(rocblas_int j = 0; j < n; j++)
        for(rocblas_int i = 0; i < n; i++)
            R[i + size_t(j) * n] = dbl(V[i + size_t(j) * ldv]);
    if(over)
    {
        std::vector<Z> Qd(nn);
        for(size_t k = 0; k < nn; k++)
            Qd[k] = dbl(Q[k]);
        cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n, n, Z(1),
                 Qd.data(), n, R.data(), n, Z(0), X.data(), n);
    }
    else
        X = R;

    // R = T X - X diag(T), or R = T^H X - X conj(diag(T))
    cpu_gemm(left ? rocblas_operation_conjugate_transpose : rocblas_operation_none,
             rocblas_operation_none, n, n, n, Z(1), Tu.data(), n, X.data(), n, Z(0), R.data(), n);
    const double tnorm = std::max(1.0, normT);
    for(rocblas_int j = 0; j < n; j++)
    {
        const Z lambda = left ? std::conj(Tu[j + size_t(j) * n]) : Tu[j + size_t(j) * n];
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
            err = std::max(err, std::sqrt(rnorm) / (tnorm * std::sqrt(xnorm)));
    }

    return err;
}

template <bool STRIDED, typename T, typename Td, typename Th>
void trevc3_getError(const rocblas_handle handle,
                     const rocblas_side side,
                     const rocsolver_eigenvectors howmny,
                     const rocblas_int n,
                     Td& dT,
                     const rocblas_int ldt,
                     const rocblas_stride stT,
                     Td& dVL,
                     const rocblas_int ldvl,
                     const rocblas_stride stVL,
                     Td& dVR,
                     const rocblas_int ldvr,
                     const rocblas_stride stVR,
                     const rocblas_int bc,
                     Th& hT,
                     Th& hVL,
                     Th& hVR,
                     Th& hTRes,
                     Th& hVLRes,
                     Th& hVRRes,
                     std::vector<T>& hQ,
                     const rocblas_int mtype,
                     double* max_err)
{
    const bool leftv = (side == rocblas_side_left || side == rocblas_side_both);
    const bool rightv = (side == rocblas_side_right || side == rocblas_side_both);
    const bool over = (howmny == rocsolver_eigenvectors_backtransform);

    // input data initialization
    trevc3_initData<true, true, T>(handle, side, howmny, n, dT, ldt, dVL, ldvl, dVR, ldvr, bc, hT,
                                   hVL, hVR, hQ, mtype);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT.data(), ldt, stT,
                                         dVL.data(), ldvl, stVL, dVR.data(), ldvr, stVR, bc));
    CHECK_HIP_ERROR(hTRes.transfer_from(dT));
    if(leftv)
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
    if(rightv)
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));

    // the eigenvectors of triangular matrices can be extremely ill-conditioned, so they are
    // not compared with the host results; the error is based on the residuals instead
    // (see trevc3_vectorError). T must not be modified.
    const size_t nn = size_t(n) * n;
    std::vector<rocblas_double_complex> Tu(nn);
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        for(size_t k = 0; k < size_t(ldt) * n; k++)
            if(!(hTRes[b][k] == hT[b][k]))
                *max_err = 1;

        // Tu = T / 2^e (exact), with 2^e >= max(|Re|, |Im|) of the entries of T, so that the
        // check neither overflows nor underflows (the residuals are relative)
        double amax = 0;
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i <= j; i++)
            {
                const T t = hT[b][i + size_t(j) * ldt];
                amax = std::max(
                    {amax, std::abs(double(std::real(t))), std::abs(double(std::imag(t)))});
            }
        int e = 0;
        if(amax > 0 && std::isfinite(amax))
            std::frexp(amax, &e);
        double normT = 0;
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
            {
                const T t = (i <= j) ? hT[b][i + size_t(j) * ldt] : T(0);
                const double re = std::ldexp(double(std::real(t)), -e);
                const double im = std::ldexp(double(std::imag(t)), -e);
                Tu[i + size_t(j) * n] = rocblas_double_complex(re, im);
                normT += re * re + im * im;
            }
        normT = std::sqrt(normT);

        // the padding rows n:ldv-1 of VL and VR must not be changed
        rocblas_int padding = 0;
        for(rocblas_int j = 0; j < n; j++)
        {
            if(rightv)
                for(rocblas_int i = n; i < ldvr; i++)
                    padding += !(hVRRes[b][i + size_t(j) * ldvr] == hVR[b][i + size_t(j) * ldvr]);
            if(leftv)
                for(rocblas_int i = n; i < ldvl; i++)
                    padding += !(hVLRes[b][i + size_t(j) * ldvl] == hVL[b][i + size_t(j) * ldvl]);
        }
        EXPECT_EQ(padding, 0) << "where b = " << b;
        if(padding)
            *max_err = 1;

        // nor the gaps between the members of a strided batch (checked once)
        if(b == 0)
        {
            rocblas_int gaps = trevc3_gapsChanged<T>(hTRes, hT, size_t(ldt) * n);
            if(leftv)
                gaps += trevc3_gapsChanged<T>(hVLRes, hVL, size_t(ldvl) * n);
            if(rightv)
                gaps += trevc3_gapsChanged<T>(hVRRes, hVR, size_t(ldvr) * n);
            EXPECT_EQ(gaps, 0);
            if(gaps)
                *max_err = 1;
        }

        T* Q = over ? hQ.data() + nn * b : nullptr;
        if(rightv)
        {
            err = trevc3_vectorError(false, over, n, Tu, normT, Q, hVRRes[b], ldvr);
            *max_err = err > *max_err ? err : *max_err;
        }
        if(leftv)
        {
            err = trevc3_vectorError(true, over, n, Tu, normT, Q, hVLRes[b], ldvl);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void trevc3_getPerfData(const rocblas_handle handle,
                        const rocblas_side side,
                        const rocsolver_eigenvectors howmny,
                        const rocblas_int n,
                        Td& dT,
                        const rocblas_int ldt,
                        const rocblas_stride stT,
                        Td& dVL,
                        const rocblas_int ldvl,
                        const rocblas_stride stVL,
                        Td& dVR,
                        const rocblas_int ldvr,
                        const rocblas_stride stVR,
                        const rocblas_int bc,
                        Th& hT,
                        Th& hVL,
                        Th& hVR,
                        std::vector<T>& hQ,
                        const rocblas_int mtype,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf)
{
    rocblas_int info;

    if(!perf)
    {
        trevc3_initData<true, false, T>(handle, side, howmny, n, dT, ldt, dVL, ldvl, dVR, ldvr, bc,
                                        hT, hVL, hVR, hQ, mtype);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_trevc3(side, howmny, n, hT[b], ldt, hVL[b], ldvl, hVR[b], ldvr, &info);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trevc3_initData<true, false, T>(handle, side, howmny, n, dT, ldt, dVL, ldvl, dVR, ldvr, bc, hT,
                                    hVL, hVR, hQ, mtype);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trevc3_initData<false, true, T>(handle, side, howmny, n, dT, ldt, dVL, ldvl, dVR, ldvr, bc,
                                        hT, hVL, hVR, hQ, mtype);

        CHECK_ROCBLAS_ERROR(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT.data(), ldt, stT,
                                             dVL.data(), ldvl, stVL, dVR.data(), ldvr, stVR, bc));
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
        trevc3_initData<false, true, T>(handle, side, howmny, n, dT, ldt, dVL, ldvl, dVR, ldvr, bc,
                                        hT, hVL, hVR, hQ, mtype);

        timer.start(stream);
        rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT.data(), ldt, stT, dVL.data(), ldvl,
                         stVL, dVR.data(), ldvr, stVR, bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_trevc3(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char howmnyC = argus.get<char>("howmny");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldt = argus.get<rocblas_int>("ldt", n);
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", n);
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", n);
    rocblas_stride stT = argus.get<rocblas_stride>("strideT", rocblas_stride(ldt) * n);
    rocblas_stride stVL = argus.get<rocblas_stride>("strideVL", rocblas_stride(ldvl) * n);
    rocblas_stride stVR = argus.get<rocblas_stride>("strideVR", rocblas_stride(ldvr) * n);
    rocblas_int mtype = argus.get<rocblas_int>("mtype", 0);

    rocblas_side side = char2rocblas_side(sideC);
    rocsolver_eigenvectors howmny = char2rocsolver_eigenvectors(howmnyC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    const bool leftv = (side == rocblas_side_left || side == rocblas_side_both);
    const bool rightv = (side == rocblas_side_right || side == rocblas_side_both);

    // check non-supported values
    if((side != rocblas_side_left && side != rocblas_side_right && side != rocblas_side_both)
       || (howmny != rocsolver_eigenvectors_all && howmny != rocsolver_eigenvectors_backtransform))
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n,
                                                   (T* const*)nullptr, ldt, stT, (T* const*)nullptr,
                                                   ldvl, stVL, (T* const*)nullptr, ldvr, stVR, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, (T*)nullptr,
                                                   ldt, stT, (T*)nullptr, ldvl, stVL, (T*)nullptr,
                                                   ldvr, stVR, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_T = size_t(ldt) * n;
    size_t size_VL = leftv ? size_t(ldvl) * n : 1;
    size_t size_VR = rightv ? size_t(ldvr) * n : 1;
    stVL = leftv ? stVL : 1;
    stVR = rightv ? stVR : 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    bool check = (argus.unit_check || argus.norm_check);
    size_t size_TRes = check ? size_T : 0;
    size_t size_VLRes = check ? size_VL : 0;
    size_t size_VRRes = check ? size_VR : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_TREVC
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n,
                                                   (T* const*)nullptr, ldt, stT, (T* const*)nullptr,
                                                   ldvl, stVL, (T* const*)nullptr, ldvr, stVR, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, (T*)nullptr,
                                                   ldt, stT, (T*)nullptr, ldvl, stVL, (T*)nullptr,
                                                   ldvr, stVR, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || ldt < n || ldt < 1 || ldvl < 1 || ldvr < 1 || (leftv && ldvl < n)
                         || (rightv && ldvr < n) || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n,
                                                   (T* const*)nullptr, ldt, stT, (T* const*)nullptr,
                                                   ldvl, stVL, (T* const*)nullptr, ldvr, stVR, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, (T*)nullptr,
                                                   ldt, stT, (T*)nullptr, ldvl, stVL, (T*)nullptr,
                                                   ldvr, stVR, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_trevc3(STRIDED, handle, side, howmny, n, (T* const*)nullptr,
                                               ldt, stT, (T* const*)nullptr, ldvl, stVL,
                                               (T* const*)nullptr, ldvr, stVR, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_trevc3(STRIDED, handle, side, howmny, n, (T*)nullptr, ldt,
                                               stT, (T*)nullptr, ldvl, stVL, (T*)nullptr, ldvr,
                                               stVR, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // host copies of the input unitary matrices (if back-transformed)
    std::vector<T> hQ;

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hT(size_T, 1, bc);
        host_batch_vector<T> hTRes(size_TRes, 1, bc);
        host_batch_vector<T> hVL(size_VL, 1, bc);
        host_batch_vector<T> hVLRes(size_VLRes, 1, bc);
        host_batch_vector<T> hVR(size_VR, 1, bc);
        host_batch_vector<T> hVRRes(size_VRRes, 1, bc);
        device_batch_vector<T> dT(size_T, 1, bc);
        device_batch_vector<T> dVL(size_VL, 1, bc);
        device_batch_vector<T> dVR(size_VR, 1, bc);
        if(size_T)
            CHECK_HIP_ERROR(dT.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT.data(), ldt,
                                                   stT, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                                                   stVR, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(check)
            trevc3_getError<STRIDED, T>(handle, side, howmny, n, dT, ldt, stT, dVL, ldvl, stVL, dVR,
                                        ldvr, stVR, bc, hT, hVL, hVR, hTRes, hVLRes, hVRRes, hQ,
                                        mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            trevc3_getPerfData<STRIDED, T>(handle, side, howmny, n, dT, ldt, stT, dVL, ldvl, stVL,
                                           dVR, ldvr, stVR, bc, hT, hVL, hVR, hQ, mtype,
                                           &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hT(size_T, 1, stT, bc);
        host_strided_batch_vector<T> hTRes(size_TRes, 1, stT, bc);
        host_strided_batch_vector<T> hVL(size_VL, 1, stVL, bc);
        host_strided_batch_vector<T> hVLRes(size_VLRes, 1, stVL, bc);
        host_strided_batch_vector<T> hVR(size_VR, 1, stVR, bc);
        host_strided_batch_vector<T> hVRRes(size_VRRes, 1, stVR, bc);
        device_strided_batch_vector<T> dT(size_T, 1, stT, bc);
        device_strided_batch_vector<T> dVL(size_VL, 1, stVL, bc);
        device_strided_batch_vector<T> dVR(size_VR, 1, stVR, bc);
        if(size_T)
            CHECK_HIP_ERROR(dT.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_trevc3(STRIDED, handle, side, howmny, n, dT.data(), ldt,
                                                   stT, dVL.data(), ldvl, stVL, dVR.data(), ldvr,
                                                   stVR, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(check)
            trevc3_getError<STRIDED, T>(handle, side, howmny, n, dT, ldt, stT, dVL, ldvl, stVL, dVR,
                                        ldvr, stVR, bc, hT, hVL, hVR, hTRes, hVLRes, hVRRes, hQ,
                                        mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            trevc3_getPerfData<STRIDED, T>(handle, side, howmny, n, dT, ldt, stT, dVL, ldvl, stVL,
                                           dVR, ldvr, stVR, bc, hT, hVL, hVR, hQ, mtype,
                                           &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using max(n, 32) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, std::max(n, 32));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("side", "howmny", "n", "ldt", "ldvl", "ldvr", "batch_c");
                rocsolver_bench_output(sideC, howmnyC, n, ldt, ldvl, ldvr, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("side", "howmny", "n", "ldt", "strideT", "ldvl", "strideVL",
                                       "ldvr", "strideVR", "batch_c");
                rocsolver_bench_output(sideC, howmnyC, n, ldt, stT, ldvl, stVL, ldvr, stVR, bc);
            }
            else
            {
                rocsolver_bench_output("side", "howmny", "n", "ldt", "ldvl", "ldvr");
                rocsolver_bench_output(sideC, howmnyC, n, ldt, ldvl, ldvr);
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

#define EXTERN_TESTING_TREVC3(...) extern template void testing_trevc3<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_TREVC3, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
