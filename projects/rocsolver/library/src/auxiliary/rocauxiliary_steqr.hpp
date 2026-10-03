/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "lapack_device_functions.hpp"
#include "rocauxiliary_bdsqr_rotlog.hpp"
#include "rocauxiliary_lasr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** STEQR_KERNEL/RUN_STEQR implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T, typename S, typename U, typename I>
rocblas_status run_steqr_hybrid(rocblas_handle handle,
                                I n,
                                S* dD,
                                const rocblas_stride strideD,
                                S* dE,
                                const rocblas_stride strideE,
                                U dC,
                                const rocblas_stride shiftC,
                                const I ldc,
                                const rocblas_stride strideC,
                                const I batch_count,
                                I* dInfo,
                                S* dWork,
                                const I max_iters,
                                const S eps,
                                const S ssfmin,
                                const S ssfmax,
                                const bool ordered = true)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I m, el, lsv, lend, lendsv, lsv_scaling, lendsv_scaling;
    I l1;
    I iters;
    S anorm, p;

    rocblas_stride strideW = 2 * n;

    rocsolver_hybrid_storage<S, I, S*> hD;
    rocsolver_hybrid_storage<S, I, S*> hE;
    rocsolver_hybrid_storage<I, I, I*> hInfo;
    rocsolver_hybrid_storage<S, I, S*> hWork;
    rocsolver_hybrid_storage<T, I, U> hC;

    ROCBLAS_CHECK(hD.init_async(n, dD, 0, strideD, batch_count, stream));
    ROCBLAS_CHECK(hE.init_async(n - 1, dE, 0, strideE, batch_count, stream));
    ROCBLAS_CHECK(hInfo.init_async(1, dInfo, 0, 1, batch_count, stream));
    ROCBLAS_CHECK(hWork.init_async(2 * n, dWork, 0, strideW, 1, stream));
    ROCBLAS_CHECK(hC.init_pointers_only(dC, shiftC, strideC, batch_count, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    I blocks = (n - 1) / BS1 + 1;

    // the rotations of the eigenvectors (columns of C) are recorded and applied in blocks
    bdsqr_rotlog<S, T, I> rlog(handle, stream, n);

    for(I b = 0; b < batch_count; b++)
    {
        S* D = hD[b];
        S* E = hE[b];
        I* info = hInfo[b];
        S* work = hWork[0];
        T* C = hC[b] + shiftC;
        rlog.set_matrices(n, nullptr, 0, 0, C, ldc, n, nullptr, 0, 0);

        l1 = 0;
        iters = 0;

        while(l1 < n && iters < max_iters)
        {
            // Determine submatrix indices
            if(l1 > 0)
                E[l1 - 1] = 0;
            for(m = l1; m < n - 1; m++)
            {
                if(abs(E[m]) <= sqrt(abs(D[m])) * sqrt(abs(D[m + 1])) * eps)
                {
                    E[m] = 0;
                    break;
                }
            }

            lsv = el = l1;
            lendsv = lend = m;
            l1 = m + 1;

            // Choose iteration type (QL or QR)
            if(abs(D[lend]) < abs(D[el]))
            {
                lend = lsv;
                el = lendsv;
            }

            // Get scaling factor
            anorm = find_max_tridiag(lsv, lendsv, D, E);

            if(lend == el)
                continue;

            lsv_scaling = lsv;
            lendsv_scaling = lendsv;
            // Scale submatrix
            if(anorm == 0)
                continue;
            else if(anorm > ssfmax)
                scale_tridiag(lsv_scaling, lendsv_scaling, D, E, ssfmax / anorm);
            else if(anorm < ssfmin)
                scale_tridiag(lsv_scaling, lendsv_scaling, D, E, ssfmin / anorm);

            if(lend >= el)
            {
                // QL iteration
                while(el <= lend && iters < max_iters)
                {
                    // Find small subdiagonal element
                    for(m = el; m <= lend - 1; m++)
                        if(abs(E[m] * E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                            break;

                    lsv = el;

                    if(m < lend)
                        E[m] = 0;
                    p = D[el];
                    if(m == el)
                    {
                        D[el] = p;
                        el++;
                    }
                    else if(m == el + 1)
                    {
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[el], E[el], D[el + 1], rt1, rt2, c, s);
                        work[el] = c;
                        work[n - 1 + el] = s;

                        D[el] = rt1;
                        D[el + 1] = rt2;
                        E[el] = 0;
                        el = el + 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[el + 1] - p) / (2 * E[el]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[el] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(I i = m - 1; i >= el; i--)
                        {
                            f = s * E[i];
                            b = c * E[i];
                            lartg(g, f, c, s, r);
                            s = -s; //get the transpose of the rotation
                            if(i != m - 1)
                                E[i + 1] = r;

                            g = D[i + 1] - p;
                            r = (D[i] - g) * s + 2 * c * b;
                            p = s * r;
                            D[i + 1] = g + p;
                            g = c * r - b;

                            // Save rotations
                            work[i] = c;
                            work[n - 1 + i] = -s;
                        }

                        D[el] -= p;
                        E[el] = g;
                    }

                    // Apply saved rotations
                    if(m != el)
                    {
                        rlog.lasr(rocblas_side_right, rocblas_backward_direction, n, m - lsv + 1,
                                  work + lsv, work + n - 1 + lsv, C[idx2D(0, lsv, ldc)]);
                    }
                }
            }

            else
            {
                // QR iteration
                while(el >= lend && iters < max_iters)
                {
                    // Find small subdiagonal element
                    for(m = el; m >= lend + 1; m--)
                        if(abs(E[m - 1] * E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                            break;

                    lsv = el;

                    if(m > lend)
                        E[m - 1] = 0;
                    p = D[el];
                    if(m == el)
                    {
                        D[el] = p;
                        el--;
                    }
                    else if(m == el - 1)
                    {
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[el - 1], E[el - 1], D[el], rt1, rt2, c, s);
                        work[m] = c;
                        work[n - 1 + m] = s;

                        D[el - 1] = rt1;
                        D[el] = rt2;
                        E[el - 1] = 0;
                        el = el - 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[el - 1] - p) / (2 * E[el - 1]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[el - 1] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(I i = m; i <= el - 1; i++)
                        {
                            f = s * E[i];
                            b = c * E[i];
                            lartg(g, f, c, s, r);
                            s = -s; //get the transpose of the rotation
                            if(i != m)
                                E[i - 1] = r;

                            g = D[i] - p;
                            r = (D[i + 1] - g) * s + 2 * c * b;
                            p = s * r;
                            D[i] = g + p;
                            g = c * r - b;

                            // Save rotations
                            work[i] = c;
                            work[n - 1 + i] = s;
                        }

                        D[el] -= p;
                        E[el - 1] = g;
                    }

                    // Apply saved rotations
                    if(m != el)
                    {
                        rlog.lasr(rocblas_side_right, rocblas_forward_direction, n, lsv - m + 1,
                                  work + m, work + n - 1 + m, C[idx2D(0, m, ldc)]);
                    }
                }
            }

            // Undo scaling
            if(anorm > ssfmax)
                scale_tridiag(lsv_scaling, lendsv_scaling, D, E, anorm / ssfmax);
            if(anorm < ssfmin)
                scale_tridiag(lsv_scaling, lendsv_scaling, D, E, anorm / ssfmin);
        }

        // Check for convergence
        for(I i = 0; i < n - 1; i++)
            if(E[i] != 0)
                *info = *info + 1;

        // Sort eigenvalues and eigenvectors by selection sort
        if(ordered)
        {
            for(I ii = 1; ii < n; ii++)
            {
                el = ii - 1;
                m = el;
                p = D[el];
                for(I j = ii; j < n; j++)
                {
                    if(D[j] < p)
                    {
                        m = j;
                        p = D[j];
                    }
                }
                if(m != el)
                {
                    D[m] = D[el];
                    D[el] = p;
                }

                if(m != el)
                {
                    rlog.swap(C[idx2D(0, el, ldc)], C[idx2D(0, m, ldc)], I(1));
                }
            }
        }
        rlog.finish();
    }

    ROCBLAS_CHECK(hD.write_to_device_async(stream));
    ROCBLAS_CHECK(hE.write_to_device_async(stream));
    ROCBLAS_CHECK(hInfo.write_to_device_async(stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    return rocblas_status_success;
}

/** STEQR_KERNEL/RUN_STEQR implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T, typename S, typename I>
__device__ void run_steqr(const I tid,
                          const I tid_inc,
                          const I n,
                          S* D,
                          S* E,
                          T* C,
                          const I ldc,
                          I* info,
                          S* work,
                          const I max_iters,
                          const S eps,
                          const S ssfmin,
                          const S ssfmax,
                          const bool ordered = true)
{
    __shared__ I m, el, lsv, lend, lendsv, lsv_scaling, lendsv_scaling;
    __shared__ I l1;
    __shared__ I iters;
    __shared__ S anorm, p;

    if(tid == 0)
    {
        l1 = 0;
        iters = 0;
    }
    __syncthreads();

    while(l1 < n && iters < max_iters)
    {
        if(tid == 0)
        {
            // Determine submatrix indices
            if(l1 > 0)
                E[l1 - 1] = 0;
            for(m = l1; m < n - 1; m++)
            {
                if(abs(E[m]) <= sqrt(abs(D[m])) * sqrt(abs(D[m + 1])) * eps)
                {
                    E[m] = 0;
                    break;
                }
            }

            lsv = el = l1;
            lendsv = lend = m;
            l1 = m + 1;

            // Choose iteration type (QL or QR)
            if(abs(D[lend]) < abs(D[el]))
            {
                lend = lsv;
                el = lendsv;
            }

            // Get scaling factor
            anorm = find_max_tridiag(lsv, lendsv, D, E);

            // Save for the case we have to undo the scaling later
            lsv_scaling = lsv;
            lendsv_scaling = lendsv;
        }
        __syncthreads();

        if(lend == el)
            continue;

        // Scale submatrix
        if(anorm == 0)
            continue;
        else if(anorm > ssfmax)
            scale_tridiag(lsv_scaling, lendsv_scaling, D, E, ssfmax / anorm, tid, tid_inc);
        else if(anorm < ssfmin)
            scale_tridiag(lsv_scaling, lendsv_scaling, D, E, ssfmin / anorm, tid, tid_inc);
        __syncthreads();

        if(lend >= el)
        {
            // QL iteration
            while(el <= lend && iters < max_iters)
            {
                if(tid == 0)
                {
                    // Find small subdiagonal element
                    for(m = el; m <= lend - 1; m++)
                        if(abs(E[m] * E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                            break;

                    lsv = el;

                    if(m < lend)
                        E[m] = 0;
                    p = D[el];
                    if(m == el)
                    {
                        D[el] = p;
                        el++;
                    }
                    else if(m == el + 1)
                    {
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[el], E[el], D[el + 1], rt1, rt2, c, s);
                        work[el] = c;
                        work[n - 1 + el] = s;

                        D[el] = rt1;
                        D[el + 1] = rt2;
                        E[el] = 0;
                        el = el + 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[el + 1] - p) / (2 * E[el]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[el] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(I i = m - 1; i >= el; i--)
                        {
                            f = s * E[i];
                            b = c * E[i];
                            lartg(g, f, c, s, r);
                            s = -s; //get the transpose of the rotation
                            if(i != m - 1)
                                E[i + 1] = r;

                            g = D[i + 1] - p;
                            r = (D[i] - g) * s + 2 * c * b;
                            p = s * r;
                            D[i + 1] = g + p;
                            g = c * r - b;

                            // Save rotations
                            work[i] = c;
                            work[n - 1 + i] = -s;
                        }

                        D[el] -= p;
                        E[el] = g;
                    }
                }
                __syncthreads();

                // Apply saved rotations
                if(m != el)
                {
                    run_lasr(rocblas_side_right, rocblas_pivot_variable, rocblas_backward_direction,
                             n, m - lsv + 1, work + lsv, work + n - 1 + lsv, C + 0 + lsv * ldc, ldc,
                             tid, tid_inc);
                    __syncthreads();
                }
            }
        }

        else
        {
            // QR iteration
            while(el >= lend && iters < max_iters)
            {
                if(tid == 0)
                {
                    // Find small subdiagonal element
                    for(m = el; m >= lend + 1; m--)
                        if(abs(E[m - 1] * E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                            break;

                    lsv = el;

                    if(m > lend)
                        E[m - 1] = 0;
                    p = D[el];
                    if(m == el)
                    {
                        D[el] = p;
                        el--;
                    }
                    else if(m == el - 1)
                    {
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[el - 1], E[el - 1], D[el], rt1, rt2, c, s);
                        work[m] = c;
                        work[n - 1 + m] = s;

                        D[el - 1] = rt1;
                        D[el] = rt2;
                        E[el - 1] = 0;
                        el = el - 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[el - 1] - p) / (2 * E[el - 1]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[el - 1] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(I i = m; i <= el - 1; i++)
                        {
                            f = s * E[i];
                            b = c * E[i];
                            lartg(g, f, c, s, r);
                            s = -s; //get the transpose of the rotation
                            if(i != m)
                                E[i - 1] = r;

                            g = D[i] - p;
                            r = (D[i + 1] - g) * s + 2 * c * b;
                            p = s * r;
                            D[i] = g + p;
                            g = c * r - b;

                            // Save rotations
                            work[i] = c;
                            work[n - 1 + i] = s;
                        }

                        D[el] -= p;
                        E[el - 1] = g;
                    }
                }
                __syncthreads();

                // Apply saved rotations
                if(m != el)
                {
                    run_lasr(rocblas_side_right, rocblas_pivot_variable, rocblas_forward_direction,
                             n, lsv - m + 1, work + m, work + n - 1 + m, C + 0 + m * ldc, ldc, tid,
                             tid_inc);
                    __syncthreads();
                }
            }
        }
        __syncthreads();

        // Undo scaling
        if(anorm > ssfmax)
            scale_tridiag(lsv_scaling, lendsv_scaling, D, E, anorm / ssfmax, tid, tid_inc);
        if(anorm < ssfmin)
            scale_tridiag(lsv_scaling, lendsv_scaling, D, E, anorm / ssfmin, tid, tid_inc);
        __syncthreads();
    }

    // Check for convergence
    for(I i = tid; i < n - 1; i += tid_inc)
        if(E[i] != 0)
            atomicAdd(
                reinterpret_cast<std::conditional_t<sizeof(I) == 4, unsigned int, unsigned long long>*>(
                    info),
                1u);

    // Sort eigenvalues and eigenvectors by selection sort
    if(ordered)
    {
        for(I ii = 1; ii < n; ii++)
        {
            if(tid == 0)
            {
                el = ii - 1;
                m = el;
                p = D[el];
                for(I j = ii; j < n; j++)
                {
                    if(D[j] < p)
                    {
                        m = j;
                        p = D[j];
                    }
                }
                if(m != el)
                {
                    D[m] = D[el];
                    D[el] = p;
                }
            }
            __syncthreads();

            if(m != el)
            {
                for(I j = 0; j < n; j++)
                    swap(C[j + el * ldc], C[j + m * ldc]);
            }
            __syncthreads();
        }
    }
}

// GPU STEQR applies the rotations in accumulated blocks for one problem of at least this order
#ifndef STEQR_BLOCKED_MIN
#define STEQR_BLOCKED_MIN 256
#endif

/** STEQR_CHASE_STATE is the state of the QL/QR iteration of STEQR_CHASE_KERNEL between launches **/
template <typename S, typename I>
struct steqr_chase_state
{
    I l1, iters, phase, ql, el, lend, lsv_s, lendsv_s, done;
    S anorm;
};

/** STEQR_CHASE_KERNEL runs the iteration of RUN_STEQR (one thread, one problem) for at most ns
    sweeps from its saved state, without updating the eigenvectors: the rotations of each sweep go
    to a slot of the log of BDSQR_GPULOG (lc/ls, indexed by pair, with the convention of LASR) with
    a descriptor, and are applied when the kernel returns. **/
template <typename S, typename I>
ROCSOLVER_KERNEL void steqr_chase_kernel(const I n,
                                         S* D,
                                         S* E,
                                         steqr_chase_state<S, I>* st,
                                         const I max_iters,
                                         const S eps,
                                         const S ssfmin,
                                         const S ssfmax,
                                         const int ns,
                                         S* lc,
                                         S* ls,
                                         bdsqr_rot_desc* desc,
                                         int* ndesc)
{
    I l1 = st->l1, iters = st->iters, phase = st->phase, ql = st->ql, el = st->el, lend = st->lend;
    I lsv_s = st->lsv_s, lendsv_s = st->lendsv_s;
    S anorm = st->anorm;
    int nslot = 0;
    I m;

    while(true)
    {
        if(phase == 0)
        {
            if(l1 >= n || iters >= max_iters)
            {
                st->done = 1;
                break;
            }

            // Determine submatrix indices
            if(l1 > 0)
                E[l1 - 1] = 0;
            for(m = l1; m < n - 1; m++)
            {
                if(abs(E[m]) <= sqrt(abs(D[m])) * sqrt(abs(D[m + 1])) * eps)
                {
                    E[m] = 0;
                    break;
                }
            }
            el = l1;
            lend = m;
            lsv_s = l1;
            lendsv_s = m;
            l1 = m + 1;

            // Choose iteration type (QL or QR)
            if(abs(D[lend]) < abs(D[el]))
            {
                lend = lsv_s;
                el = lendsv_s;
            }

            // Get scaling factor
            anorm = find_max_tridiag(lsv_s, lendsv_s, D, E);
            if(lend == el || anorm == 0)
                continue;

            // Scale submatrix
            if(anorm > ssfmax)
                scale_tridiag(lsv_s, lendsv_s, D, E, ssfmax / anorm);
            else if(anorm < ssfmin)
                scale_tridiag(lsv_s, lendsv_s, D, E, ssfmin / anorm);
            ql = (lend >= el);
            phase = 1;
        }

        if(!((ql ? el <= lend : el >= lend) && iters < max_iters))
        {
            // Undo scaling
            if(anorm > ssfmax)
                scale_tridiag(lsv_s, lendsv_s, D, E, anorm / ssfmax);
            if(anorm < ssfmin)
                scale_tridiag(lsv_s, lendsv_s, D, E, anorm / ssfmin);
            phase = 0;
            continue;
        }
        if(nslot == ns)
            break;

        S* wc = lc + size_t(nslot) * n;
        S* ws = ls + size_t(nslot) * n;
        I lsv = el;
        if(ql)
        {
            // Find small subdiagonal element
            for(m = el; m <= lend - 1; m++)
                if(abs(E[m] * E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                    break;

            if(m < lend)
                E[m] = 0;
            S p = D[el];
            if(m == el)
            {
                el++;
                continue;
            }
            else if(m == el + 1)
            {
                // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                S rt1, rt2, c, s;
                laev2(D[el], E[el], D[el + 1], rt1, rt2, c, s);
                wc[el] = c;
                ws[el] = s;

                D[el] = rt1;
                D[el + 1] = rt2;
                E[el] = 0;
                el = el + 2;
            }
            else
            {
                iters++;

                S f, g, c, s, b, r;

                // Form shift
                g = (D[el + 1] - p) / (2 * E[el]);
                if(g >= 0)
                    r = abs(sqrt(1 + g * g));
                else
                    r = -abs(sqrt(1 + g * g));
                g = D[m] - p + (E[el] / (g + r));

                c = 1;
                s = 1;
                p = 0;

                for(I i = m - 1; i >= el; i--)
                {
                    f = s * E[i];
                    b = c * E[i];
                    lartg(g, f, c, s, r);
                    s = -s; //get the transpose of the rotation
                    if(i != m - 1)
                        E[i + 1] = r;

                    g = D[i + 1] - p;
                    r = (D[i] - g) * s + 2 * c * b;
                    p = s * r;
                    D[i + 1] = g + p;
                    g = c * r - b;

                    // Save rotations
                    wc[i] = c;
                    ws[i] = -s;
                }

                D[el] -= p;
                E[el] = g;
            }
            // (the rotations of the columns lsv..m, backward)
            const int d = (*ndesc)++;
            desc[d] = {nslot, -1, int(lsv), int(m)};
        }
        else
        {
            // Find small subdiagonal element
            for(m = el; m >= lend + 1; m--)
                if(abs(E[m - 1] * E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                    break;

            if(m > lend)
                E[m - 1] = 0;
            S p = D[el];
            if(m == el)
            {
                el--;
                continue;
            }
            else if(m == el - 1)
            {
                // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                S rt1, rt2, c, s;
                laev2(D[el - 1], E[el - 1], D[el], rt1, rt2, c, s);
                wc[m] = c;
                ws[m] = s;

                D[el - 1] = rt1;
                D[el] = rt2;
                E[el - 1] = 0;
                el = el - 2;
            }
            else
            {
                iters++;

                S f, g, c, s, b, r;

                // Form shift
                g = (D[el - 1] - p) / (2 * E[el - 1]);
                if(g >= 0)
                    r = abs(sqrt(1 + g * g));
                else
                    r = -abs(sqrt(1 + g * g));
                g = D[m] - p + (E[el - 1] / (g + r));

                c = 1;
                s = 1;
                p = 0;

                for(I i = m; i <= el - 1; i++)
                {
                    f = s * E[i];
                    b = c * E[i];
                    lartg(g, f, c, s, r);
                    s = -s; //get the transpose of the rotation
                    if(i != m)
                        E[i - 1] = r;

                    g = D[i] - p;
                    r = (D[i + 1] - g) * s + 2 * c * b;
                    p = s * r;
                    D[i] = g + p;
                    g = c * r - b;

                    // Save rotations
                    wc[i] = c;
                    ws[i] = s;
                }

                D[el] -= p;
                E[el - 1] = g;
            }
            // (the rotations of the columns m..lsv, forward)
            const int d = (*ndesc)++;
            desc[d] = {nslot, 1, int(m), int(lsv)};
        }
        nslot++;
    }

    st->l1 = l1;
    st->iters = iters;
    st->phase = phase;
    st->ql = ql;
    st->el = el;
    st->lend = lend;
    st->lsv_s = lsv_s;
    st->lendsv_s = lendsv_s;
    st->anorm = anorm;
}

/** STEQR_SORT_PERM computes the swaps of the selection sort of RUN_STEQR (sw[i] = index swapped
    with i), and sorts D; STEQR_APPLY_SWAPS applies them to the columns of C (a thread per row).
    STEQR_COUNT_INFO sets info to the number of nonzero entries of E. **/
template <typename S, typename I>
ROCSOLVER_KERNEL void steqr_sort_perm(const I n, S* D, I* sw)
{
    for(I ii = 1; ii < n; ii++)
    {
        I el = ii - 1, m = el;
        S p = D[el];
        for(I j = ii; j < n; j++)
        {
            if(D[j] < p)
            {
                m = j;
                p = D[j];
            }
        }
        if(m != el)
        {
            D[m] = D[el];
            D[el] = p;
        }
        sw[el] = m;
    }
}

template <typename T, typename I>
ROCSOLVER_KERNEL void steqr_apply_swaps(const I n, T* C, const I ldc, const I* sw)
{
    const I r = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(r >= n)
        return;
    for(I el = 0; el < n - 1; el++)
    {
        const I m = sw[el];
        if(m != el)
        {
            const T t = C[r + el * ldc];
            C[r + el * ldc] = C[r + m * ldc];
            C[r + m * ldc] = t;
        }
    }
}

template <typename S, typename I>
ROCSOLVER_KERNEL void steqr_count_info(const I n, const S* E, I* info)
{
    for(I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < n - 1;
        i += hipGridDim_x * hipBlockDim_x)
        if(E[i] != 0)
            atomicAdd(
                reinterpret_cast<std::conditional_t<sizeof(I) == 4, unsigned int, unsigned long long>*>(
                    info),
                1u);
}

/** RUN_STEQR_BLOCKED runs STEQR on the device for one problem, with the rotations of each
    BDSQR_ROT_SWEEPS sweeps applied in accumulated blocks (see BDSQR_GPULOG) **/
template <typename T, typename S, typename U, typename I>
rocblas_status run_steqr_blocked(rocblas_handle handle,
                                 const I n,
                                 S* D,
                                 S* E,
                                 U C,
                                 const rocblas_stride shiftC,
                                 const I ldc,
                                 I* info,
                                 I* sw,
                                 const I max_iters,
                                 const S eps,
                                 const S ssfmin,
                                 const S ssfmax)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    T* Ch = bdsqr_host_ptr<T>(C, shiftC, stream);
    bdsqr_gpulog<S, T> glog(handle, int(n), nullptr, 0, 0, Ch, int(ldc), int(n), nullptr, 0, 0);
    steqr_chase_state<S, I>* st;
    HIP_CHECK(hipMalloc(&st, sizeof(*st)));
    // (freed on any return or exception)
    std::unique_ptr<steqr_chase_state<S, I>, void (*)(steqr_chase_state<S, I>*)> st_guard(
        st, [](steqr_chase_state<S, I>* p) { (void)hipFree(p); });
    HIP_CHECK(hipMemsetAsync(st, 0, sizeof(*st), stream));
    I done = 0;
    while(!done)
    {
        ROCSOLVER_LAUNCH_KERNEL((steqr_chase_kernel<S, I>), dim3(1), dim3(1), 0, stream, n, D, E,
                                st, max_iters, eps, ssfmin, ssfmax, glog.sweeps(), glog.log_c(),
                                glog.log_s(), glog.desc(), glog.ndesc());
        HIP_CHECK(hipMemcpyAsync(&done, &st->done, sizeof(I), hipMemcpyDeviceToHost, stream));
        glog.flush();
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    ROCSOLVER_LAUNCH_KERNEL((steqr_count_info<S, I>), dim3((n - 1) / 256 + 1), dim3(256), 0, stream,
                            n, E, info);
    ROCSOLVER_LAUNCH_KERNEL((steqr_sort_perm<S, I>), dim3(1), dim3(1), 0, stream, n, D, sw);
    ROCSOLVER_LAUNCH_KERNEL((steqr_apply_swaps<T, I>), dim3((n - 1) / 64 + 1), dim3(64), 0, stream,
                            n, Ch, ldc, sw);
    return rocblas_status_success;
}

template <typename T, typename S, typename U, typename I>
ROCSOLVER_KERNEL void steqr_kernel(const I n,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   U CC,
                                   const rocblas_stride shiftC,
                                   const I ldc,
                                   const rocblas_stride strideC,
                                   I* iinfo,
                                   S* WW,
                                   const I max_iters,
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax)
{
    // select bacth instance
    I tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    I tid_inc = hipGridDim_x * hipBlockDim_x;
    I bid = hipBlockIdx_y;
    rocblas_stride strideW = 2 * n;

    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    T* C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* work = WW + (bid * strideW);
    I* info = iinfo + bid;

    // execute
    run_steqr(tid, tid_inc, n, D, E, C, ldc, info, work, max_iters, eps, ssfmin, ssfmax);
}

template <typename T, typename S, typename I>
void rocsolver_steqr_getMemorySize(const rocblas_evect evect,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_work_stack)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work_stack = 0;
        return;
    }

    // size of stack (for lasrt); indexed as I* in sterf_kernel
    if(evect == rocblas_evect_none)
        *size_work_stack = sizeof(I) * (2 * 32) * batch_count;
    else
        *size_work_stack = sizeof(S) * (2 * n) * batch_count;
}

template <typename T, typename S, typename I>
rocblas_status rocsolver_steqr_argCheck(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const I n,
                                        S D,
                                        S E,
                                        T C,
                                        const I ldc,
                                        I* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_none && evect != rocblas_evect_tridiagonal
       && evect != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(evect != rocblas_evect_none && ldc < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || (evect != rocblas_evect_none && n && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U, typename I>
rocblas_status rocsolver_steqr_template(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const I n,
                                        S* D,
                                        const rocblas_stride shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride shiftE,
                                        const rocblas_stride strideE,
                                        U C,
                                        const rocblas_stride shiftC,
                                        const I ldc,
                                        const rocblas_stride strideC,
                                        I* info,
                                        const I batch_count,
                                        void* work_stack)
{
    ROCSOLVER_ENTER("steqr", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocsolver_alg_mode alg_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_steqr, &alg_mode));

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 1 && evect != rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0, stream, C,
                                strideC, n, 1);
    if(n <= 1)
        return rocblas_status_success;

    // Initialize identity matrix
    if(evect == rocblas_evect_tridiagonal)
    {
        I blocks = (n - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                                stream, n, n, C, shiftC, ldc, strideC);
    }

    S eps = get_epsilon<S>();
    S ssfmin = get_safemin<S>();
    S ssfmax = S(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / S(3.0);

    if(evect == rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(sterf_kernel<S>, dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, E + shiftE, strideE, info, (I*)work_stack,
                                30 * n, eps, ssfmin, ssfmax);
    else
    {
        if(alg_mode == rocsolver_alg_mode_hybrid)
        {
            ROCBLAS_CHECK(run_steqr_hybrid<T>(handle, n, D + shiftD, strideD, E + shiftE, strideE,
                                              C, shiftC, ldc, strideC, batch_count, info,
                                              (S*)work_stack, 30 * n, eps, ssfmin, ssfmax));
        }
        else if(batch_count == 1 && n >= STEQR_BLOCKED_MIN)
        {
            // (sw: the swaps of the final sort, in the workspace)
            ROCBLAS_CHECK(run_steqr_blocked<T>(handle, n, D + shiftD, E + shiftE, C, shiftC, ldc,
                                               info, (I*)work_stack, 30 * n, eps, ssfmin, ssfmax));
        }
        else
        {
            const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

            ROCSOLVER_LAUNCH_KERNEL((steqr_kernel<T>), dim3(1, batch_count), dim3(props->warpSize),
                                    0, stream, n, D + shiftD, strideD, E + shiftE, strideE, C, shiftC,
                                    ldc, strideC, info, (S*)work_stack, 30 * n, eps, ssfmin, ssfmax);
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
