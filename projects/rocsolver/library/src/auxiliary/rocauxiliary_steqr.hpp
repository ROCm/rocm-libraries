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
#include "rocauxiliary_lasr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** STEQR_KERNEL/RUN_STEQR implements the main loop of the sterf algorithm
    to compute the eigenvalues of a symmetric tridiagonal matrix given by D
    and E **/
template <typename T, typename S, typename U>
rocblas_status run_steqr_hybrid(rocblas_handle handle,
                                rocblas_int n,
                                S* dD,
                                const rocblas_stride strideD,
                                S* dE,
                                const rocblas_stride strideE,
                                U dC,
                                const rocblas_stride shiftC,
                                const rocblas_int ldc,
                                const rocblas_stride strideC,
                                const rocblas_int batch_count,
                                rocblas_int* dInfo,
                                S* dWork,
                                const rocblas_int max_iters,
                                const S eps,
                                const S ssfmin,
                                const S ssfmax,
                                const bool ordered = true)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int m, l, lsv, lend, lendsv;
    rocblas_int l1;
    rocblas_int iters;
    S anorm, p;

    rocblas_stride strideW = 2 * n;

    rocsolver_hybrid_storage<S, rocblas_int, S*> hD;
    rocsolver_hybrid_storage<S, rocblas_int, S*> hE;
    rocsolver_hybrid_storage<rocblas_int, rocblas_int, rocblas_int*> hInfo;
    rocsolver_hybrid_storage<S, rocblas_int, S*> hWork;
    rocsolver_hybrid_storage<T, rocblas_int, U> hC;

    ROCBLAS_CHECK(hD.init_async(n, dD, 0, strideD, batch_count, stream));
    ROCBLAS_CHECK(hE.init_async(n - 1, dE, 0, strideE, batch_count, stream));
    ROCBLAS_CHECK(hInfo.init_async(1, dInfo, 0, 1, batch_count, stream));
    ROCBLAS_CHECK(hWork.init_async(2 * n, dWork, 0, strideW, 1, stream));
    ROCBLAS_CHECK(hC.init_pointers_only(dC, shiftC, strideC, batch_count, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    rocblas_int blocks = (n - 1) / BS1 + 1;

    for(rocblas_int b = 0; b < batch_count; b++)
    {
        S* D = hD[b];
        S* E = hE[b];
        rocblas_int* info = hInfo[b];
        S* work = hWork[0];
        T* C = hC[b] + shiftC;

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

            lsv = l = l1;
            lendsv = lend = m;
            l1 = m + 1;

            // Choose iteration type (QL or QR)
            if(abs(D[lend]) < abs(D[l]))
            {
                lend = lsv;
                l = lendsv;
            }

            // Get scaling factor
            anorm = find_max_tridiag(lsv, lendsv, D, E);

            if(lend == l)
                continue;

            // Scale submatrix
            if(anorm == 0)
                continue;
            else if(anorm > ssfmax)
                scale_tridiag(lsv, lendsv, D, E, anorm / ssfmax);
            else if(anorm < ssfmin)
                scale_tridiag(lsv, lendsv, D, E, anorm / ssfmin);

            if(lend >= l)
            {
                // QL iteration
                while(l <= lend && iters < max_iters)
                {
                    // Find small subdiagonal element
                    for(m = l; m <= lend - 1; m++)
                        if(abs(E[m] * E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                            break;

                    lsv = l;

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
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[l], E[l], D[l + 1], rt1, rt2, c, s);
                        work[l] = c;
                        work[n - 1 + l] = s;

                        D[l] = rt1;
                        D[l + 1] = rt2;
                        E[l] = 0;
                        l = l + 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[l + 1] - p) / (2 * E[l]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[l] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(int i = m - 1; i >= l; i--)
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

                        D[l] -= p;
                        E[l] = g;
                    }

                    // Apply saved rotations
                    if(m != l)
                    {
                        ROCBLAS_CHECK(hWork.write_to_device_async(stream));
                        ROCBLAS_CHECK(rocsolver_lasr_template<T, S>(
                            handle, rocblas_side_right, rocblas_pivot_variable,
                            rocblas_backward_direction, n, m - lsv + 1, dWork + lsv, strideW,
                            dWork + n - 1 + lsv, strideW, C, lsv * ldc, ldc, strideC, 1));
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
                        if(abs(E[m - 1] * E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                            break;

                    lsv = l;

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
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[l - 1], E[l - 1], D[l], rt1, rt2, c, s);
                        work[m] = c;
                        work[n - 1 + m] = s;

                        D[l - 1] = rt1;
                        D[l] = rt2;
                        E[l - 1] = 0;
                        l = l - 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[l - 1] - p) / (2 * E[l - 1]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[l - 1] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(int i = m; i <= l - 1; i++)
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

                        D[l] -= p;
                        E[l - 1] = g;
                    }

                    // Apply saved rotations
                    if(m != l)
                    {
                        ROCBLAS_CHECK(hWork.write_to_device_async(stream));
                        ROCBLAS_CHECK(rocsolver_lasr_template<T, S>(
                            handle, rocblas_side_right, rocblas_pivot_variable,
                            rocblas_forward_direction, n, lsv - m + 1, dWork + m, strideW,
                            dWork + n - 1 + m, strideW, C, m * ldc, ldc, strideC, 1));
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
                *info = *info + 1;

        // Sort eigenvalues and eigenvectors by selection sort
        if(ordered)
        {
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

                if(m != l)
                {
                    ROCSOLVER_LAUNCH_KERNEL(swap_kernel<T>, dim3(blocks), dim3(BS1), 0, stream, n,
                                            C + l * ldc, 1, C + m * ldc, 1);
                }
            }
        }
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
template <typename T, typename S>
__device__ void run_steqr(const rocblas_int tid,
                          const rocblas_int tid_inc,
                          const rocblas_int n,
                          S* D,
                          S* E,
                          T* C,
                          const rocblas_int ldc,
                          rocblas_int* info,
                          S* work,
                          const rocblas_int max_iters,
                          const S eps,
                          const S ssfmin,
                          const S ssfmax,
                          const bool ordered = true)
{
    __shared__ rocblas_int m, l, lsv, lend, lendsv;
    __shared__ rocblas_int l1;
    __shared__ rocblas_int iters;
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

            lsv = l = l1;
            lendsv = lend = m;
            l1 = m + 1;

            // Choose iteration type (QL or QR)
            if(abs(D[lend]) < abs(D[l]))
            {
                lend = lsv;
                l = lendsv;
            }

            // Get scaling factor
            anorm = find_max_tridiag(lsv, lendsv, D, E);
        }
        __syncthreads();

        if(lend == l)
            continue;

        // Scale submatrix
        if(anorm == 0)
            continue;
        else if(anorm > ssfmax)
            scale_tridiag(lsv, lendsv, D, E, anorm / ssfmax, tid, tid_inc);
        else if(anorm < ssfmin)
            scale_tridiag(lsv, lendsv, D, E, anorm / ssfmin, tid, tid_inc);
        __syncthreads();

        if(lend >= l)
        {
            // QL iteration
            while(l <= lend && iters < max_iters)
            {
                if(tid == 0)
                {
                    // Find small subdiagonal element
                    for(m = l; m <= lend - 1; m++)
                        if(abs(E[m] * E[m]) <= eps * eps * abs(D[m] * D[m + 1]))
                            break;

                    lsv = l;

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
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[l], E[l], D[l + 1], rt1, rt2, c, s);
                        work[l] = c;
                        work[n - 1 + l] = s;

                        D[l] = rt1;
                        D[l + 1] = rt2;
                        E[l] = 0;
                        l = l + 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[l + 1] - p) / (2 * E[l]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[l] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(int i = m - 1; i >= l; i--)
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

                        D[l] -= p;
                        E[l] = g;
                    }
                }
                __syncthreads();

                // Apply saved rotations
                if(m != l)
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
            while(l >= lend && iters < max_iters)
            {
                if(tid == 0)
                {
                    // Find small subdiagonal element
                    for(m = l; m >= lend + 1; m--)
                        if(abs(E[m - 1] * E[m - 1]) <= eps * eps * abs(D[m] * D[m - 1]))
                            break;

                    lsv = l;

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
                        // Use laev2 to compute 2x2 eigenvalues and eigenvectors
                        S rt1, rt2, c, s;
                        laev2(D[l - 1], E[l - 1], D[l], rt1, rt2, c, s);
                        work[m] = c;
                        work[n - 1 + m] = s;

                        D[l - 1] = rt1;
                        D[l] = rt2;
                        E[l - 1] = 0;
                        l = l - 2;
                    }
                    else
                    {
                        iters++;

                        S f, g, c, s, b, r;

                        // Form shift
                        g = (D[l - 1] - p) / (2 * E[l - 1]);
                        if(g >= 0)
                            r = abs(sqrt(1 + g * g));
                        else
                            r = -abs(sqrt(1 + g * g));
                        g = D[m] - p + (E[l - 1] / (g + r));

                        c = 1;
                        s = 1;
                        p = 0;

                        for(int i = m; i <= l - 1; i++)
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

                        D[l] -= p;
                        E[l - 1] = g;
                    }
                }
                __syncthreads();

                // Apply saved rotations
                if(m != l)
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
            scale_tridiag(lsv, lendsv, D, E, ssfmax / anorm, tid, tid_inc);
        if(anorm < ssfmin)
            scale_tridiag(lsv, lendsv, D, E, ssfmin / anorm, tid, tid_inc);
        __syncthreads();
    }

    // Check for convergence
    for(int i = tid; i < n - 1; i += tid_inc)
        if(E[i] != 0)
            atomicAdd(info, 1);

    // Sort eigenvalues and eigenvectors by selection sort
    if(ordered)
    {
        for(int ii = 1; ii < n; ii++)
        {
            if(tid == 0)
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
            __syncthreads();

            if(m != l)
            {
                for(int j = 0; j < n; j++)
                    swap(C[j + l * ldc], C[j + m * ldc]);
            }
            __syncthreads();
        }
    }
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void steqr_kernel(const rocblas_int n,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* EE,
                                   const rocblas_stride strideE,
                                   U CC,
                                   const rocblas_int shiftC,
                                   const rocblas_int ldc,
                                   const rocblas_stride strideC,
                                   rocblas_int* iinfo,
                                   S* WW,
                                   const rocblas_int max_iters,
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax)
{
    // select bacth instance
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tid_inc = hipGridDim_x * hipBlockDim_x;
    rocblas_int bid = hipBlockIdx_y;
    rocblas_stride strideW = 2 * n;

    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    T* C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
    S* work = WW + (bid * strideW);
    rocblas_int* info = iinfo + bid;

    // execute
    run_steqr(tid, tid_inc, n, D, E, C, ldc, info, work, max_iters, eps, ssfmin, ssfmax);
}

template <typename T, typename S>
void rocsolver_steqr_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work_stack = 0;
        return;
    }

    // size of stack (for lasrt)
    if(evect == rocblas_evect_none)
        *size_work_stack = sizeof(rocblas_int) * (2 * 32) * batch_count;
    else
        *size_work_stack = sizeof(S) * (2 * n) * batch_count;
}

template <typename T, typename S>
rocblas_status rocsolver_steqr_argCheck(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S D,
                                        S E,
                                        T C,
                                        const rocblas_int ldc,
                                        rocblas_int* info)
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

template <typename T, typename S, typename U>
rocblas_status rocsolver_steqr_template(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        U C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
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

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
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
        rocblas_int blocks = (n - 1) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                                stream, n, n, C, shiftC, ldc, strideC);
    }

    S eps = get_epsilon<S>();
    S ssfmin = get_safemin<S>();
    S ssfmax = S(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / S(3.0);

    if(evect == rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(sterf_kernel<S>, dim3(batch_count), dim3(1), 0, stream, n,
                                D + shiftD, strideD, E + shiftE, strideE, info,
                                (rocblas_int*)work_stack, 30 * n, eps, ssfmin, ssfmax);
    else
    {
        if(alg_mode == rocsolver_alg_mode_hybrid)
        {
            ROCBLAS_CHECK(run_steqr_hybrid<T>(handle, n, D + shiftD, strideD, E + shiftE, strideE,
                                              C, shiftC, ldc, strideC, batch_count, info,
                                              (S*)work_stack, 30 * n, eps, ssfmin, ssfmax));
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
