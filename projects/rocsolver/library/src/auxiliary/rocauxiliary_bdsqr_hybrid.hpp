
/****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_host_helpers.hpp"
#include "lapack_host_functions.hpp"
#include "rocauxiliary_lasr.hpp"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/************************************************************************************/
/***************** Kernel launchers *************************************************/
/************************************************************************************/

template <typename T, typename I>
static void swap_template(rocblas_handle handle,
                          I const n,
                          T* x,
                          I const incx,
                          T* y,
                          I const incy,
                          hipStream_t stream)
{
    auto nthreads = get_device_warp_size() * 2;
    auto nblocks = (n - 1) / nthreads + 1;

    ROCSOLVER_LAUNCH_KERNEL((swap_kernel<T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0,
                            stream, n, x, incx, y, incy);
}

template <typename S, typename T, typename I>
static void rot_template(rocblas_handle handle,
                         I const n,
                         T* x,
                         I const incx,
                         T* y,
                         I const incy,
                         S const c,
                         S const s,
                         hipStream_t stream)
{
    auto nthreads = get_device_warp_size() * 2;
    auto nblocks = (n - 1) / nthreads + 1;

    ROCSOLVER_LAUNCH_KERNEL((rot_kernel<S, T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0,
                            stream, n, x, incx, y, incy, c, s);
}

template <typename S, typename T, typename I>
static void scal_template(rocblas_handle handle,
                          I const n,
                          S const da,
                          T* const x,
                          I const incx,
                          hipStream_t stream)
{
    auto nthreads = get_device_warp_size() * 2;
    auto nblocks = (n - 1) / nthreads + 1;

    ROCSOLVER_LAUNCH_KERNEL((scal_kernel<S, T, I>), dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0,
                            stream, n, da, x, incx);
}

/** Call to lasr functionality.
    run_lasr can be executed as a host or device function **/
template <typename S, typename T, typename I>
static void call_lasr(rocblas_side& side,
                      rocblas_pivot& pivot,
                      rocblas_direct& direct,
                      I& m,
                      I& n,
                      S& c,
                      S& s,
                      T& A,
                      I& lda)
{
    I const tid = 0;
    I const i_inc = 1;

    run_lasr<T, S, I>(side, pivot, direct, m, n, &c, &s, &A, lda, tid, i_inc);
}

/************************************************************************************/
/***************** Main template functions ******************************************/
/************************************************************************************/

template <typename S, typename T, typename I>
static void bdsqr_single_template(rocblas_handle handle,
                                  char uplo,
                                  I n,
                                  I ncvt,
                                  I nru,
                                  I ncc,
                                  S* d_,
                                  S* e_,
                                  T* vt_,
                                  I ldvt,
                                  T* u_,
                                  I ldu,
                                  T* c_,
                                  I ldc,
                                  S* work_,
                                  I& info,
                                  S* dwork_ = nullptr,
                                  hipStream_t stream = 0)
{
    // -------------------------------------
    // Lambda expressions used as helpers
    // -------------------------------------
    auto call_swap_gpu = [=](I n, T& x, I incx, T& y, I incy) {
        swap_template<T, I>(handle, n, &x, incx, &y, incy, stream);
    };

    auto call_rot_gpu = [=](I n, T& x, I incx, T& y, I incy, S cosl, S sinl) {
        rot_template<S, T, I>(handle, n, &x, incx, &y, incy, cosl, sinl, stream);
    };

    auto call_scal_gpu = [=](I n, auto da, T& x, I incx) {
        scal_template<S, T, I>(handle, n, da, &x, incx, stream);
    };

    auto call_lasr_gpu_nocopy
        = [=](rocblas_side const side, rocblas_pivot const pivot, rocblas_direct const direct,
              I const m, I const n, S& dc, S& ds, T& A, I const lda, hipStream_t stream) {
              bool const is_left_side = (side == rocblas_side_left);
              auto const mn = (is_left_side) ? m : n;
              auto const mn_m1 = (mn - 1);

              rocsolver_lasr_template<T, S>(handle, side, pivot, direct, m, n, &dc, 0, &ds, 0, &A,
                                            0, lda, 0, I(1));
          };

    auto call_lasr_gpu = [=](rocblas_side const side, rocblas_pivot const pivot,
                             rocblas_direct const direct, I const m, I const n, S& c, S& s, T& A,
                             I const lda, S* const dwork_, hipStream_t stream) {
        bool const is_left_side = (side == rocblas_side_left);
        auto const mn = (is_left_side) ? m : n;
        auto const mn_m1 = (mn - 1);
        S* const dc = dwork_;
        S* const ds = dwork_ + mn_m1;
        CHECK_HIP(hipStreamSynchronize(stream));

        CHECK_HIP(hipMemcpyAsync(dc, &c, sizeof(S) * mn_m1, hipMemcpyHostToDevice, stream));
        CHECK_HIP(hipMemcpyAsync(ds, &s, sizeof(S) * mn_m1, hipMemcpyHostToDevice, stream));

        rocsolver_lasr_template<T, S>(handle, side, pivot, direct, m, n, dc, 0, ds, 0, &A, 0, lda,
                                      0, I(1));
        CHECK_HIP(hipStreamSynchronize(stream));
    };

    auto d = [=](auto i) -> S& { return (d_[i - 1]); };

    auto e = [=](auto i) -> S& { return (e_[i - 1]); };

    auto work = [=](auto i) -> S& { return (work_[i - 1]); };

    auto dwork = [=](auto i) -> S& { return (dwork_[i - 1]); };

    auto c = [=](auto i, auto j) -> T& { return (c_[idx2D(i - 1, j - 1, ldc)]); };

    auto u = [=](auto i, auto j) -> T& { return (u_[idx2D(i - 1, j - 1, ldu)]); };

    auto vt = [=](auto i, auto j) -> T& { return (vt_[idx2D(i - 1, j - 1, ldvt)]); };

    auto sign = [](auto a, auto b) {
        auto const abs_a = std::abs(a);
        return ((b >= 0) ? abs_a : -abs_a);
    };

    auto dble = [](auto x) { return (static_cast<double>(x)); };
    // -------------------------------

    // ----------------
    // Initialization
    // ----------------
    bool const use_gpu = (dwork_ != nullptr);

    // Lapack code used O(n^2) algorithm for sorting
    // Consider turning off this and rely on
    // bdsqr_sort() to perform sorting
    bool constexpr need_sort = false;

    S const zero = 0;
    S const one = 1;
    S negone = -1;
    S const hndrd = 100;
    S const hndrth = one / hndrd;
    S const ten = 10;
    S const eight = 8;
    S const meight = -one / eight;
    I const maxitr = 6;
    I ione = 1;

    bool const lower = (uplo == 'L') || (uplo == 'l');
    bool const upper = (uplo == 'U') || (uplo == 'u');

    //rotate is true if any singular vectors desired, false otherwise
    bool const rotate = (ncvt > 0) || (nru > 0) || (ncc > 0);

    I i = 0, idir = 0, isub = 0, iter = 0, iterdivn = 0, j = 0, ll = 0, lll = 0, m = 0,
      maxitdivn = 0, nm1 = 0, nm12 = 0, nm13 = 0, oldll = 0, oldm = 0;

    I const nrc = n; // number of rows in C matrix
    I const nrvt = n; // number of rows in VT matrix
    I const ncu = n; // number of columns in U matrix

    S abse = 0, abss = 0, cosl = 0, cosr = 0, cs = 0, eps = 0, f = 0, g = 0, h = 0, mu = 0,
      oldcs = 0, oldsn = 0, r = 0, shift = 0, sigmn = 0, sigmx = 0, sinl = 0, sinr = 0, sll = 0,
      smax = 0, smin = 0, sminl = 0, sminoa = 0, sn = 0, thresh = 0, tol = 0, tolmul = 0, unfl = 0;

    bool const need_update_singular_vectors = (nru > 0) || (ncc > 0);
    bool constexpr use_lasr_gpu_nocopy = false;

    nm1 = n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;

    //get machine constants
    {
        char cmach_eps = 'E';
        char cmach_unfl = 'S';
        call_lamch(cmach_eps, eps);
        call_lamch(cmach_unfl, unfl);
    }

    // -----------------------------------------
    // rotate to upper bidiagonal if necesarry
    // -----------------------------------------
    if(lower)
    {
        for(i = 1; i <= (n - 1); i++)
        {
            call_lartg(d(i), e(i), cs, sn, r);
            d(i) = r;
            e(i) = sn * d(i + 1);
            d(i + 1) = cs * d(i + 1);
            work(i) = cs;
            work(nm1 + i) = sn;
        }

        // update singular vectors if desired
        if(use_lasr_gpu_nocopy)
        {
            CHECK_HIP(hipStreamSynchronize(stream));

            if(need_update_singular_vectors)
            {
                // copy rotations
                size_t const nbytes = sizeof(S) * (n - 1);
                hipMemcpyKind const kind = hipMemcpyHostToDevice;

                {
                    void* const src = (void*)&(work(1));
                    void* const dst = (void*)&(dwork(1));
                    CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                }

                {
                    void* const src = (void*)&(work(n));
                    void* const dst = (void*)&(dwork(n));
                    CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                }
            }
            CHECK_HIP(hipStreamSynchronize(stream));
        }

        if(nru > 0)
        {
            // call_lasr( 'r', 'v', 'f', nru, n, work( 1 ), work( n ), u, ldu );
            rocblas_side side = rocblas_side_right;
            rocblas_pivot pivot = rocblas_pivot_variable;
            rocblas_direct direct = rocblas_forward_direction;
            if(use_gpu)
            {
                if(use_lasr_gpu_nocopy)
                {
                    call_lasr_gpu_nocopy(side, pivot, direct, nru, n, dwork(1), dwork(n), u(1, 1),
                                         ldu, stream);
                }
                else
                {
                    call_lasr_gpu(side, pivot, direct, nru, n, work(1), work(n), u(1, 1), ldu,
                                  dwork_, stream);
                }
            }
            else
            {
                call_lasr(side, pivot, direct, nru, n, work(1), work(n), u(1, 1), ldu);
            }
        }
        if(ncc > 0)
        {
            // call_lasr( 'l', 'v', 'f', n, ncc, work( 1 ), work( n ), c, ldc );
            rocblas_side side = rocblas_side_left;
            rocblas_pivot pivot = rocblas_pivot_variable;
            rocblas_direct direct = rocblas_forward_direction;
            if(use_gpu)
            {
                if(use_lasr_gpu_nocopy)
                {
                    call_lasr_gpu_nocopy(side, pivot, direct, n, ncc, dwork(1), dwork(n), c(1, 1),
                                         ldc, stream);
                }
                else
                {
                    call_lasr_gpu(side, pivot, direct, n, ncc, work(1), work(n), c(1, 1), ldc,
                                  dwork_, stream);
                }
            }
            else
            {
                call_lasr(side, pivot, direct, n, ncc, work(1), work(n), c(1, 1), ldc);
            }
        }
    }

    // -------------------------------------------------------------
    // Compute singular values and vector to relative accuracy tol
    // -------------------------------------------------------------
    tolmul = std::max(ten, std::min(hndrd, pow(eps, meight)));
    tol = tolmul * eps;

    // compute approximate maximum, minimum singular values
    smax = zero;
    for(i = 1; i <= n; i++)
    {
        smax = std::max(smax, std::abs(d(i)));
    }
L20:
    for(i = 1; i <= (n - 1); i++)
    {
        smax = std::max(smax, std::abs(e(i)));
    }

    // compute tolerance
L30:
    sminl = zero;
    if(tol >= zero)
    {
        // relative accuracy desired
        sminoa = std::abs(d(1));
        if(sminoa == zero)
            goto L50;
        mu = sminoa;
        for(i = 2; i <= n; i++)
        {
            mu = std::abs(d(i)) * (mu / (mu + std::abs(e(i - 1))));
            sminoa = std::min(sminoa, mu);
            if(sminoa == zero)
                goto L50;
        }
    L40:
    L50:
        sminoa = sminoa / std::sqrt(dble(n));
        thresh = std::max(tol * sminoa, ((unfl * n) * n) * maxitr);
    }
    else
    {
        //absolute accuracy desired
        thresh = std::max(std::abs(tol) * smax, ((unfl * n) * n) * maxitr);
    }

    /** prepare for main iteration loop for the singular values
        (maxit is the maximum number of passes through the inner
        loop permitted before nonconvergence signalled.) **/
    maxitdivn = maxitr * n;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;
    m = n;

    ///////////////////////////
    /// MAIN ITERATION LOOP ///
    ///////////////////////////
L60:
    // check for convergence or exceeding iteration count
    if(m <= 1)
        goto L160;

    if(iter >= n)
    {
        iter = iter - n;
        iterdivn = iterdivn + 1;
        if(iterdivn >= maxitdivn)
            goto L200;
    }

    // find diagonal block of matrix to work on
    if(tol < zero && std::abs(d(m)) <= thresh)
        d(m) = zero;

    smax = std::abs(d(m));
    smin = smax;
    for(lll = 1; lll <= (m - 1); lll++)
    {
        ll = m - lll;
        abss = std::abs(d(ll));
        abse = std::abs(e(ll));
        if(tol < zero && abss <= thresh)
            d(ll) = zero;
        if(abse <= thresh)
            goto L80;
        smin = std::min(smin, abss);
        smax = std::max(smax, std::max(abss, abse));
    }

L70:
    ll = 0;
    goto L90;

L80:
    e(ll) = zero;
    // matrix splits since e(ll) = 0
    if(ll == m - 1)
    {
        // convergence of bottom singular value, return to top of loop
        m = m - 1;
        goto L60;
    }

L90:
    ll = ll + 1;
    // e(ll) through e(m-1) are nonzero, e(ll-1) is zero
    if(ll == m - 1)
    {
        // 2 by 2 block, handle separately
        call_lasv2(d(m - 1), e(m - 1), d(m), sigmn, sigmx, sinr, cosr, sinl, cosl);
        d(m - 1) = sigmx;
        e(m - 1) = zero;
        d(m) = sigmn;

        // compute singular vectors, if desired
        if(ncvt > 0)
        {
            if(use_gpu)
            {
                call_rot_gpu(ncvt, vt(m - 1, 1), ldvt, vt(m, 1), ldvt, cosr, sinr);
            }
            else
            {
                call_rot(ncvt, vt(m - 1, 1), ldvt, vt(m, 1), ldvt, cosr, sinr);
            }
        }
        if(nru > 0)
        {
            if(use_gpu)
            {
                call_rot_gpu(nru, u(1, m - 1), ione, u(1, m), ione, cosl, sinl);
            }
            else
            {
                call_rot(nru, u(1, m - 1), ione, u(1, m), ione, cosl, sinl);
            }
        }
        if(ncc > 0)
        {
            if(use_gpu)
            {
                call_rot_gpu(ncc, c(m - 1, 1), ldc, c(m, 1), ldc, cosl, sinl);
            }
            else
            {
                call_rot(ncc, c(m - 1, 1), ldc, c(m, 1), ldc, cosl, sinl);
            }
        }
        m = m - 2;
        goto L60;
    }

    /** if working on new submatrix, choose shift direction
        (from larger end diagonal element towards smaller) **/
    if(ll > oldm || m < oldll)
    {
        if(std::abs(d(ll)) >= std::abs(d(m)))
        {
            // chase bulge from top (big end) to bottom (small end)
            idir = 1;
        }
        else
        {
            // chase bulge from bottom (big end) to top (small end)
            idir = 2;
        }
    }

    // apply convergence test
    if(idir == 1)
    {
        // run convergence test in forward direction
        // first apply standard test to bottom of matrix
        if(std::abs(e(m - 1)) <= std::abs(tol) * std::abs(d(m))
           || (tol < zero && std::abs(e(m - 1)) <= thresh))
        {
            e(m - 1) = zero;
            goto L60;
        }

        if(tol >= zero)
        {
            // if relative accuracy desired,
            // apply convergence criterion forward
            mu = std::abs(d(ll));
            sminl = mu;
            for(lll = ll; lll <= (m - 1); lll++)
            {
                if(std::abs(e(lll)) <= tol * mu)
                {
                    e(lll) = zero;
                    goto L60;
                }
                mu = std::abs(d(lll + 1)) * (mu / (mu + std::abs(e(lll))));
                sminl = std::min(sminl, mu);
            }
        }
    }
    else
    {
        // run convergence test in backward direction
        // first apply standard test to top of matrix
        if(std::abs(e(ll)) <= std::abs(tol) * std::abs(d(ll))
           || (tol < zero && std::abs(e(ll)) <= thresh))
        {
            e(ll) = zero;
            goto L60;
        }

        if(tol >= zero)
        {
            // if relative accuracy desired,
            // apply convergence criterion backward
            mu = std::abs(d(m));
            sminl = mu;
            for(lll = (m - 1); lll >= ll; lll--)
            {
                if(std::abs(e(lll)) <= tol * mu)
                {
                    e(lll) = zero;
                    goto L60;
                }
                mu = std::abs(d(lll)) * (mu / (mu + std::abs(e(lll))));
                sminl = std::min(sminl, mu);
            }
        }
    }

    /** compute shift.  first, test if shifting would ruin relative
        accuracy, and if so set the shift to zero **/
    oldll = ll;
    oldm = m;
    if(tol >= zero && n * tol * (sminl / smax) <= std::max(eps, hndrth * tol))
    {
        //use a zero shift to avoid loss of relative accuracy
        shift = zero;
    }
    else
    {
        // compute the shift from 2-by-2 block at end of matrix
        if(idir == 1)
        {
            sll = std::abs(d(ll));
            call_las2(d(m - 1), e(m - 1), d(m), shift, r);
        }
        else
        {
            sll = std::abs(d(m));
            call_las2(d(ll), e(ll), d(ll + 1), shift, r);
        }
        // test if shift negligible, and if so set to zero
        if(sll > zero)
        {
            if((shift / sll) * (shift / sll) < eps)
                shift = zero;
        }
    }

    // increment iteration count
    iter = iter + m - ll;

    // if shift = 0, do simplified qr iteration
    if(shift == zero)
    {
        if(idir == 1)
        {
            // chase bulge from top to bottom
            // save cosines and sines for later singular vector updates
            cs = one;
            oldcs = one;
            for(i = ll; i <= (m - 1); i++)
            {
                auto di_cs = d(i) * cs;
                call_lartg(di_cs, e(i), cs, sn, r);
                if(i > ll)
                    e(i - 1) = oldsn * r;
                auto oldcs_r = oldcs * r;
                auto dip1_sn = d(i + 1) * sn;
                call_lartg(oldcs_r, dip1_sn, oldcs, oldsn, d(i));
                work(i - ll + 1) = cs;
                work(i - ll + 1 + nm1) = sn;
                work(i - ll + 1 + nm12) = oldcs;
                work(i - ll + 1 + nm13) = oldsn;
            }

        L120:
            h = d(m) * cs;
            d(m) = h * oldcs;
            e(m - 1) = h * oldsn;

            //   update singular vectors
            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // copy rotations
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncvt, work( 1 ), work( n ), vt(ll, 1 ), ldvt )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_forward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(1), dwork(n),
                                             vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1),
                                      ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1), ldvt);
                }
            }
            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'f', nru, m-ll+1, work( nm12+1 ), work( nm13+1 ), u( 1, ll ), ldu )
                rocblas_side side = rocblas_side_right;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_forward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(nm12 + 1),
                                             dwork(nm13 + 1), u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                                      u(1, ll), ldu, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                              u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncc, work( nm12+1 ), work( nm13+1 ), c( ll, 1 ), ldc )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_forward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(nm12 + 1),
                                             dwork(nm13 + 1), c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                                      c(ll, 1), ldc, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                              c(ll, 1), ldc);
                }
            }

            // test convergence
            if(std::abs(e(m - 1)) <= thresh)
                e(m - 1) = zero;
        }
        else
        {
            // chase bulge from bottom to top
            // save cosines and sines for later singular vector updates
            cs = one;
            oldcs = one;
            for(i = m; i >= (ll + 1); i--)
            {
                auto di_cs = d(i) * cs;
                call_lartg(di_cs, e(i - 1), cs, sn, r);

                if(i < m)
                    e(i) = oldsn * r;

                auto oldcs_r = oldcs * r;
                auto dim1_sn = d(i - 1) * sn;
                call_lartg(oldcs_r, dim1_sn, oldcs, oldsn, d(i));

                work(i - ll) = cs;
                work(i - ll + nm1) = -sn;
                work(i - ll + nm12) = oldcs;
                work(i - ll + nm13) = -oldsn;
            }

        L130:
            h = d(ll) * cs;
            d(ll) = h * oldcs;
            e(ll) = h * oldsn;

            // update singular vectors
            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // copy rotations
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncvt, work( nm12+1 ), work( nm13+1 ), vt( ll, 1 ), ldvt );
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_backward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(nm12 + 1),
                                             dwork(nm13 + 1), vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                                      vt(ll, 1), ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                              vt(ll, 1), ldvt);
                }
            }
            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'b', nru, m-ll+1, work( 1 ), work( n ), u( 1, ll ), ldu )
                rocblas_side side = rocblas_side_right;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_backward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(1), dwork(n),
                                             u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncc, work( 1 ), work( n ), c( ll, 1 ), ldc )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_backward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(1), dwork(n),
                                             c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc);
                }
            }

            // test convergence
            if(std::abs(e(ll)) <= thresh)
                e(ll) = zero;
        }
    }

    // otherwise use nonzero shift
    else
    {
        if(idir == 1)
        {
            // chase bulge from top to bottom
            // save cosines and sines for later singular vector updates
            f = (std::abs(d(ll)) - shift) * (sign(one, d(ll)) + shift / d(ll));
            g = e(ll);
            for(i = ll; i <= (m - 1); i++)
            {
                call_lartg(f, g, cosr, sinr, r);
                if(i > ll)
                    e(i - 1) = r;
                f = cosr * d(i) + sinr * e(i);
                e(i) = cosr * e(i) - sinr * d(i);
                g = sinr * d(i + 1);
                d(i + 1) = cosr * d(i + 1);
                call_lartg(f, g, cosl, sinl, r);
                d(i) = r;
                f = cosl * e(i) + sinl * d(i + 1);
                d(i + 1) = cosl * d(i + 1) - sinl * e(i);
                if(i < m - 1)
                {
                    g = sinl * e(i + 1);
                    e(i + 1) = cosl * e(i + 1);
                }
                work(i - ll + 1) = cosr;
                work(i - ll + 1 + nm1) = sinr;
                work(i - ll + 1 + nm12) = cosl;
                work(i - ll + 1 + nm13) = sinl;
            }

        L140:
            e(m - 1) = f;

            // update singular vectors
            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // copy rotations
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncvt, work( 1 ), work( n ), vt(ll, 1 ), ldvt )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_forward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(1), dwork(n),
                                             vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1),
                                      ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(1), work(n), vt(ll, 1), ldvt);
                }
            }

            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'f', nru, m-ll+1, work( nm12+1 ), work( nm13+1 ), u( 1, ll ), ldu )
                rocblas_side side = rocblas_side_right;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_forward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(nm12 + 1),
                                             dwork(nm13 + 1), u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                                      u(1, ll), ldu, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(nm12 + 1), work(nm13 + 1),
                              u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'f', m-ll+1, ncc, work( nm12+1 ), work( nm13+1 ), c( ll, 1 ), ldc )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_forward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(nm12 + 1),
                                             dwork(nm13 + 1), c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                                      c(ll, 1), ldc, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(nm12 + 1), work(nm13 + 1),
                              c(ll, 1), ldc);
                }
            }

            // test convergence
            if(std::abs(e(m - 1)) <= thresh)
                e(m - 1) = zero;
        }
        else
        {
            // chase bulge from bottom to top
            // save cosines and sines for later singular vector updates
            f = (std::abs(d(m)) - shift) * (sign(one, d(m)) + shift / d(m));
            g = e(m - 1);
            for(i = m; i >= (ll + 1); i--)
            {
                call_lartg(f, g, cosr, sinr, r);
                if(i < m)
                    e(i) = r;
                f = cosr * d(i) + sinr * e(i - 1);
                e(i - 1) = cosr * e(i - 1) - sinr * d(i);
                g = sinr * d(i - 1);
                d(i - 1) = cosr * d(i - 1);
                call_lartg(f, g, cosl, sinl, r);
                d(i) = r;
                f = cosl * e(i - 1) + sinl * d(i - 1);
                d(i - 1) = cosl * d(i - 1) - sinl * e(i - 1);
                if(i > ll + 1)
                {
                    g = sinl * e(i - 2);
                    e(i - 2) = cosl * e(i - 2);
                }
                work(i - ll) = cosr;
                work(i - ll + nm1) = -sinr;
                work(i - ll + nm12) = cosl;
                work(i - ll + nm13) = -sinl;
            }

        L150:
            e(ll) = f;

            // test convergence
            if(std::abs(e(ll)) <= thresh)
                e(ll) = zero;

            // update singular vectors
            if(use_lasr_gpu_nocopy)
            {
                CHECK_HIP(hipStreamSynchronize(stream));

                if(rotate)
                {
                    // copy rotations
                    size_t const nbytes = sizeof(S) * (n - 1);
                    hipMemcpyKind const kind = hipMemcpyHostToDevice;

                    if((nru > 0) || (ncc > 0))
                    {
                        {
                            void* const src = (void*)&(work(1));
                            void* const dst = (void*)&(dwork(1));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(n));
                            void* const dst = (void*)&(dwork(n));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }

                    if(ncvt > 0)
                    {
                        {
                            void* const src = (void*)&(work(nm12));
                            void* const dst = (void*)&(dwork(nm12));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }

                        {
                            void* const src = (void*)&(work(nm13));
                            void* const dst = (void*)&(dwork(nm13));
                            CHECK_HIP(hipMemcpyAsync(dst, src, nbytes, kind, stream));
                        }
                    }
                }

                CHECK_HIP(hipStreamSynchronize(stream));
            }

            if(ncvt > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncvt, work( nm12+1 ), work(nm13+1), vt( ll, 1 ), ldvt )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_backward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncvt, dwork(nm12 + 1),
                                             dwork(nm13 + 1), vt(ll, 1), ldvt, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                                      vt(ll, 1), ldvt, dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncvt, work(nm12 + 1), work(nm13 + 1),
                              vt(ll, 1), ldvt);
                }
            }
            if(nru > 0)
            {
                // call_lasr( 'r', 'v', 'b', nru, m-ll+1, work( 1 ), work( n ), u( 1, ll ), ldu )
                rocblas_side side = rocblas_side_right;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_backward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, nru, mm, dwork(1), dwork(n),
                                             u(1, ll), ldu, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, nru, mm, work(1), work(n), u(1, ll), ldu);
                }
            }
            if(ncc > 0)
            {
                // call_lasr( 'l', 'v', 'b', m-ll+1, ncc, work( 1 ), work( n ), c( ll, 1 ), ldc )
                rocblas_side side = rocblas_side_left;
                rocblas_pivot pivot = rocblas_pivot_variable;
                rocblas_direct direct = rocblas_backward_direction;
                auto mm = m - ll + 1;
                if(use_gpu)
                {
                    if(use_lasr_gpu_nocopy)
                    {
                        call_lasr_gpu_nocopy(side, pivot, direct, mm, ncc, dwork(1), dwork(n),
                                             c(ll, 1), ldc, stream);
                    }
                    else
                    {
                        call_lasr_gpu(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc,
                                      dwork_, stream);
                    }
                }
                else
                {
                    call_lasr(side, pivot, direct, mm, ncc, work(1), work(n), c(ll, 1), ldc);
                }
            }
        }
    }
    CHECK_HIP(hipStreamSynchronize(stream));

    // qr iteration finished, go back and check convergence
    goto L60;

L160:
    // all singular values converged, so make them positive
    for(i = 1; i <= n; i++)
    {
        if(d(i) < zero)
        {
            d(i) = -d(i);
            if(ncvt > 0)
            {
                if(use_gpu)
                {
                    call_scal_gpu(ncvt, negone, vt(i, 1), ldvt);
                }
                else
                {
                    call_scal(ncvt, negone, vt(i, 1), ldvt);
                }
            }
        }
    }

L170:
    // sort the singular values into decreasing order (insertion sort on
    // singular values, but only one transposition per singular vector)
    if(need_sort)
    {
        for(i = 1; i <= (n - 1); i++)
        {
            // scan for smallest d(i)
            isub = 1;
            smin = d(1);
            for(j = 2; j <= (n + 1 - i); j++)
            {
                if(d(j) <= smin)
                {
                    isub = j;
                    smin = d(j);
                }
            }
        L180:
            if(isub != n + 1 - i)
            {
                // swap singular values and vectors
                d(isub) = d(n + 1 - i);
                d(n + 1 - i) = smin;
                if(ncvt > 0)
                {
                    if(use_gpu)
                    {
                        call_swap_gpu(ncvt, vt(isub, 1), ldvt, vt(n + 1 - i, 1), ldvt);
                    }
                    else
                    {
                        call_swap(ncvt, vt(isub, 1), ldvt, vt(n + 1 - i, 1), ldvt);
                    }
                }
                if(nru > 0)
                {
                    if(use_gpu)
                    {
                        call_swap_gpu(nru, u(1, isub), ione, u(1, n + 1 - i), ione);
                    }
                    else
                    {
                        call_swap(nru, u(1, isub), ione, u(1, n + 1 - i), ione);
                    }
                }
                if(ncc > 0)
                {
                    if(use_gpu)
                    {
                        call_swap_gpu(ncc, c(isub, 1), ldc, c(n + 1 - i, 1), ldc);
                    }
                    else
                    {
                        call_swap(ncc, c(isub, 1), ldc, c(n + 1 - i, 1), ldc);
                    }
                }
            }
        }
    }

L190:
    goto L220;

// maximum number of iterations exceeded, failure to converge
L200:
    info = 0;
    for(i = 1; i <= (n - 1); i++)
    {
        if(e(i) != zero)
            info = info + 1;
    }

L210:
L220:
    return;
}

template <typename T, typename S, typename W1, typename W2, typename W3, typename I = rocblas_int>
rocblas_status rocsolver_bdsqr_host_batch_template(rocblas_handle handle,
                                                   const rocblas_fill uplo_in,
                                                   const I n,
                                                   const I nv,
                                                   const I nu,
                                                   const I nc,
                                                   S* D,
                                                   const rocblas_stride strideD,
                                                   S* E,
                                                   const rocblas_stride strideE,
                                                   W1 V,
                                                   const I shiftV,
                                                   const I ldv,
                                                   const rocblas_stride strideV,
                                                   W2 U,
                                                   const I shiftU,
                                                   const I ldu,
                                                   const rocblas_stride strideU,
                                                   W3 C,
                                                   const I shiftC,
                                                   const I ldc,
                                                   const rocblas_stride strideC,
                                                   I* info_array,
                                                   const I batch_count,
                                                   I* splits_map,
                                                   S* work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // -----------------------------------
    // transfer arrays from device to host
    // -----------------------------------
    rocsolver_hybrid_storage<S, I, S*> hD;
    rocsolver_hybrid_storage<S, I, S*> hE;
    rocsolver_hybrid_storage<T, I, W1> hV;
    rocsolver_hybrid_storage<T, I, W2> hU;
    rocsolver_hybrid_storage<T, I, W3> hC;
    rocsolver_hybrid_storage<I, I, I*> hInfo;

    ROCBLAS_CHECK(hD.init_async(n, D, 0, strideD, batch_count, stream));
    ROCBLAS_CHECK(hE.init_async(n - 1, E, 0, strideE, batch_count, stream));
    if(nv > 0)
        ROCBLAS_CHECK(hV.init_pointers_only(V, shiftV, strideV, batch_count, stream));
    if(nu > 0)
        ROCBLAS_CHECK(hU.init_pointers_only(U, shiftU, strideU, batch_count, stream));
    if(nc > 0)
        ROCBLAS_CHECK(hC.init_pointers_only(C, shiftC, strideC, batch_count, stream));
    ROCBLAS_CHECK(hInfo.init_async(1, info_array, 0, 1, batch_count, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    S* hwork = nullptr;
    HIP_CHECK(hipHostMalloc(&hwork, sizeof(S) * (4 * n)));
    S* dwork = nullptr;
    HIP_CHECK(hipMalloc(&dwork, sizeof(S) * (4 * n)));

    // --------------------------------------
    // Execute for each instance in the batch
    // --------------------------------------
    for(I bid = 0; bid < batch_count; bid++)
    {
        if(hInfo[bid][0] != 0)
            continue;

        char uplo = (uplo_in == rocblas_fill_lower) ? 'L' : 'U';
        I info = 0;

        bdsqr_single_template<S, T, I>(handle, uplo, n, nv, nu, nc, hD[bid], hE[bid], hV[bid], ldv,
                                       hU[bid], ldu, hC[bid], ldc, hwork, info, dwork, stream);

        if(info == 0)
        {
            // explicitly zero out "E" array
            // to be compatible with rocsolver bdsqr
            S const zero = S(0);
            for(I i = 0; i < (n - 1); i++)
            {
                hE[bid][i] = zero;
            }
        }

        if(hInfo[bid][0] == 0)
        {
            hInfo[bid][0] = info;
        }
    } // end for bid

    // -----------------------------------
    // transfer arrays from host to device
    // -----------------------------------
    ROCBLAS_CHECK(hD.write_to_device_async(stream));
    ROCBLAS_CHECK(hE.write_to_device_async(stream));
    ROCBLAS_CHECK(hInfo.write_to_device_async(stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // ----------------------
    // free allocated storage
    // ----------------------
    HIP_CHECK(hipHostFree(hwork));
    HIP_CHECK(hipFree(dwork));

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
#undef LASR_MAX_NTHREADS
#undef CHECK_HIP
