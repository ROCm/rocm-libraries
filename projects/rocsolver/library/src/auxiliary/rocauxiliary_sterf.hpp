/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/****************************************************************************
(TODO:THIS IS BASIC IMPLEMENTATION. THE ONLY PARALLELISM INTRODUCED HERE IS
  FOR THE BATCHED VERSIONS (A DIFFERENT THREAD WORKS ON EACH INSTANCE OF THE
  BATCH).)
***************************************************************************/

/** STERF_SQ_E squares the elements of E **/
template <typename T>
__device__ __host__ void sterf_sq_e(const rocblas_int start, const rocblas_int end, T* E)
{
    for(int i = start; i < end; i++)
        E[i] = E[i] * E[i];
}

template <typename T>
__device__ __host__ void run_sterf(const rocblas_int n,
                                   T* D,
                                   T* E,
                                   rocblas_int* info,
                                   rocblas_int* stack,
                                   const rocblas_int max_iters,
                                   const T eps,
                                   const T ssfmin,
                                   const T ssfmax)
{
    rocblas_int m, l, lsv, lend, lendsv;
    rocblas_int l1 = 0;
    rocblas_int iters = 0;
    T anorm, p;

    while(l1 < n && iters < max_iters)
    {
        // Determine submatrix indices
        if(l1 > 0)
            E[l1 - 1] = 0;
        for(m = l1; m < n - 1; m++)
        {
            if(std::abs(E[m]) <= std::sqrt(std::abs(D[m])) * std::sqrt(std::abs(D[m + 1])) * eps)
            {
                E[m] = 0;
                break;
            }
        }

        lsv = l = l1;
        lendsv = lend = m;
        l1 = m + 1;
        if(lend == l)
            continue;

        // Scale submatrix
        anorm = find_max_tridiag(l, lend, D, E);
        if(anorm == 0)
            continue;
        else if(anorm > ssfmax)
            scale_tridiag(l, lend, D, E, anorm / ssfmax);
        else if(anorm < ssfmin)
            scale_tridiag(l, lend, D, E, anorm / ssfmin);
        sterf_sq_e(l, lend, E);

        // Choose iteration type (QL or QR)
        if(std::abs(D[lend]) < std::abs(D[l]))
        {
            lend = lsv;
            l = lendsv;
        }

        if(lend >= l)
        {
            // QL iteration
            while(l <= lend && iters < max_iters)
            {
                // Find small subdiagonal element
                for(m = l; m <= lend - 1; m++)
                    if(std::abs(E[m]) <= eps * eps * std::abs(D[m] * D[m + 1]))
                        break;

                if(m < lend)
                    E[m] = 0;
                p = D[l];
                if(m == l)
                {
                    D[l] = p;
                    l++;
                }
                else if(m == l + 1)
                {
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    T rte, rt1, rt2;
                    rte = std::sqrt(E[l]);
                    lae2(D[l], rte, D[l + 1], rt1, rt2);
                    D[l] = rt1;
                    D[l + 1] = rt2;
                    E[l] = 0;
                    l = l + 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    T sigma, gamma, r, rte, c, s;

                    // Form shift
                    rte = std::sqrt(E[l]);
                    sigma = (D[l + 1] - p) / (2 * rte);
                    if(sigma >= 0)
                        r = std::abs(sqrt(1 + sigma * sigma));
                    else
                        r = -std::abs(sqrt(1 + sigma * sigma));
                    sigma = p - (rte / (sigma + r));

                    c = 1;
                    s = 0;
                    gamma = D[m] - sigma;
                    p = gamma * gamma;

                    for(int i = m - 1; i >= l; i--)
                    {
                        T bb = E[i];
                        r = p + bb;
                        if(i != m - 1)
                            E[i + 1] = s * r;

                        T oldc = c;
                        c = p / r;
                        s = bb / r;
                        T oldgam = gamma;
                        gamma = c * (D[i] - sigma) - s * oldgam;
                        D[i + 1] = oldgam + (D[i] - gamma);
                        if(c != 0)
                            p = (gamma * gamma) / c;
                        else
                            p = oldc * bb;
                    }

                    E[l] = s * p;
                    D[l] = sigma + gamma;
                }
            }
        }

        else
        {
            // QR iteration
            while(l >= lend && iters < max_iters)
            {
                // Find small subdiagonal element
                for(m = l; m >= lend + 1; m--)
                    if(std::abs(E[m - 1]) <= eps * eps * std::abs(D[m] * D[m - 1]))
                        break;

                if(m > lend)
                    E[m - 1] = 0;
                p = D[l];
                if(m == l)
                {
                    D[l] = p;
                    l--;
                }
                else if(m == l - 1)
                {
                    // Use lae2 to compute 2x2 eigenvalues. Using rte, rt1, rt2.
                    T rte, rt1, rt2;
                    rte = std::sqrt(E[l - 1]);
                    lae2(D[l], rte, D[l - 1], rt1, rt2);
                    D[l] = rt1;
                    D[l - 1] = rt2;
                    E[l - 1] = 0;
                    l = l - 2;
                }
                else
                {
                    if(iters == max_iters)
                        break;
                    iters++;

                    T sigma, gamma, r, rte, c, s;

                    // Form shift. Using rte, r, c, s.
                    rte = std::sqrt(E[l - 1]);
                    sigma = (D[l - 1] - p) / (2 * rte);
                    if(sigma >= 0)
                        r = std::abs(std::sqrt(1 + sigma * sigma));
                    else
                        r = -std::abs(std::sqrt(1 + sigma * sigma));
                    sigma = p - (rte / (sigma + r));

                    c = 1;
                    s = 0;
                    gamma = D[m] - sigma;
                    p = gamma * gamma;

                    for(int i = m; i <= l - 1; i++)
                    {
                        T bb = E[i];
                        r = p + bb;
                        if(i != m)
                            E[i - 1] = s * r;

                        T oldc = c;
                        c = p / r;
                        s = bb / r;
                        T oldgam = gamma;
                        gamma = c * (D[i + 1] - sigma) - s * oldgam;
                        D[i] = oldgam + (D[i + 1] - gamma);
                        if(c != 0)
                            p = (gamma * gamma) / c;
                        else
                            p = oldc * bb;
                    }

                    E[l - 1] = s * p;
                    D[l] = sigma + gamma;
                }
            }
        }

        // Undo scaling
        if(anorm > ssfmax)
            scale_tridiag(lsv, lendsv, D, E, ssfmax / anorm);
        if(anorm < ssfmin)
            scale_tridiag(lsv, lendsv, D, E, ssfmin / anorm);
    }

    // Check for convergence
    for(int i = 0; i < n - 1; i++)
        if(E[i] != 0)
            (*info)++;

    // Sort eigenvalues
    /** (TODO: the quick-sort method implemented in lasrt_increasing fails for some cases.
        Substituting it here with a simple sorting algorithm. If more performance is required in
        the future, lasrt_increasing should be debugged or another quick-sort method
        could be implemented) **/
    //lasrt_increasing(n, D, stack);

    for(int ii = 1; ii < n; ii++)
    {
        l = ii - 1;
        m = l;
        p = D[l];
        for(int j = ii; j < n; j++)
        {
            if(D[j] < p)
            {
                m = j;
                p = D[j];
            }
        }
        if(m != l)
        {
            D[m] = D[l];
            D[l] = p;
        }
    }
}

/** STERF_KERNEL implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T>
ROCSOLVER_KERNEL void sterf_kernel(const rocblas_int n,
                                   T* DD,
                                   const rocblas_stride strideD,
                                   T* EE,
                                   const rocblas_stride strideE,
                                   rocblas_int* infoA,
                                   rocblas_int* stackA,
                                   const rocblas_int max_iters,
                                   const T eps,
                                   const T ssfmin,
                                   const T ssfmax)
{
    rocblas_int bid = hipBlockIdx_x;

    T* D = DD + (bid * strideD);
    T* E = EE + (bid * strideE);
    rocblas_int* info = infoA + bid;
    rocblas_int* stack = stackA + bid * (2 * 32);

    run_sterf<T>(n, D, E, info, stack, max_iters, eps, ssfmin, ssfmax);
}

template <typename T>
void rocsolver_sterf_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_stack)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_stack = 0;
        return;
    }

    // size of stack (for lasrt)
    *size_stack = sizeof(rocblas_int) * (2 * 32) * batch_count;
}

template <typename T>
rocblas_status
    rocsolver_sterf_argCheck(rocblas_handle handle, const rocblas_int n, T D, T E, rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_sterf_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        U D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        U E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        rocblas_int* stack)
{
    ROCSOLVER_ENTER("sterf", "n:", n, "shiftD:", shiftD, "shiftE:", shiftE, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocsolver_alg_mode alg_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &alg_mode));

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n <= 1)
        return rocblas_status_success;

    T eps = get_epsilon<T>();
    T ssfmin = get_safemin<T>();
    T ssfmax = T(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / T(3.0);

    if(alg_mode == rocsolver_alg_mode_hybrid)
    {
        rocsolver_hybrid_storage<T, rocblas_int, U> hD;
        rocsolver_hybrid_storage<T, rocblas_int, U> hE;
        rocsolver_hybrid_storage<rocblas_int, rocblas_int, rocblas_int*> hInfo;

        ROCBLAS_CHECK(hD.init_async(n, D, shiftD, strideD, batch_count, stream));
        ROCBLAS_CHECK(hE.init_async(n - 1, E, shiftE, strideE, batch_count, stream));
        ROCBLAS_CHECK(hInfo.init_async(1, info, 0, 1, batch_count, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        for(rocblas_int b = 0; b < batch_count; b++)
        {
            run_sterf<T>(n, hD[b], hE[b], hInfo[b], nullptr, 30 * n, eps, ssfmin, ssfmax);
        }

        ROCBLAS_CHECK(hD.write_to_device_async(stream));
        ROCBLAS_CHECK(hE.write_to_device_async(stream));
        ROCBLAS_CHECK(hInfo.write_to_device_async(stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL(sterf_kernel<T>, dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, E + shiftE, strideE, info, stack, 30 * n, eps,
                                ssfmin, ssfmax);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
