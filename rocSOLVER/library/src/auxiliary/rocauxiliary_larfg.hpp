/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.8.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2017
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void run_set_taubeta(T* tau, T* norms, T* alpha, T* beta)
{
    const auto ignore_beta = (beta == nullptr);
    if(norms[0] > 0)
    {
        T n = sqrt(norms[0] + alpha[0] * alpha[0]);
        n = alpha[0] >= 0 ? -n : n;

        // scaling factor:
        norms[0] = 1.0 / (alpha[0] - n);

        // tau:
        tau[0] = (n - alpha[0]) / n;

        // beta:
        if(ignore_beta)
        {
            alpha[0] = n;
        }
        else
        {
            beta[0] = n;
            alpha[0] = 1;
        }
    }
    else
    {
        norms[0] = 1;
        tau[0] = 0;

        // beta:
        if(!ignore_beta)
        {
            beta[0] = alpha[0];
            alpha[0] = 1;
        }
    }
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ void run_set_taubeta(T* tau, T* norms, T* alpha, T* beta)
{
    using S = decltype(std::real(T{}));
    S r, rr, ri, ar, ai;

    ar = alpha[0].real();
    ai = alpha[0].imag();
    S m = ai * ai;

    const auto ignore_beta = (beta == nullptr);
    if(norms[0].real() > 0 || m > 0)
    {
        m += ar * ar;
        S n = sqrt(norms[0].real() + m);
        n = ar >= 0 ? -n : n;

        // scaling factor:
        //    norms[0] = 1.0 / (alpha[0] - n);
        r = (ar - n) * (ar - n) + ai * ai;
        rr = (ar - n) / r;
        ri = -ai / r;
        norms[0] = rocblas_complex_num<S>(rr, ri);

        // tau:
        //    tau[0] = (n - alpha[0]) / n;
        rr = (n - ar) / n;
        ri = -ai / n;
        tau[0] = rocblas_complex_num<S>(rr, ri);

        // beta:
        if(ignore_beta)
        {
            alpha[0] = n;
        }
        else
        {
            beta[0] = n;
            alpha[0] = 1;
        }
    }
    else
    {
        norms[0] = 1;
        tau[0] = 0;

        // beta:
        if(!ignore_beta)
        {
            beta[0] = alpha[0];
            alpha[0] = 1;
        }
    }
}

template <typename T, typename I, typename U, typename UB>
ROCSOLVER_KERNEL void set_taubeta(T* tauA,
                                  const rocblas_stride strideP,
                                  T* norms,
                                  U alphaA,
                                  const rocblas_stride shiftA,
                                  const rocblas_stride strideA,
                                  UB betaA,
                                  const rocblas_stride shiftb,
                                  const rocblas_stride strideb)
{
    I bid = hipBlockIdx_x;

    // select batch instance
    T* alpha = load_ptr_batch<T>(alphaA, bid, shiftA, strideA);
    T* beta = betaA ? load_ptr_batch<T>(betaA, bid, shiftb, strideb) : nullptr;
    T* tau = tauA + bid * strideP;

    run_set_taubeta<T>(tau, norms + bid, alpha, beta);
}

template <typename T, typename I>
rocblas_status rocsolver_larfg_getMemorySize(const I n,
                                             const I batch_count,
                                             size_t* size_work,
                                             size_t* size_norms)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_norms = 0;
        *size_work = 0;
        return rocblas_status_success;
    }

    // if small size no workspace needed
    if(n <= LARFG_SSKER_MAX_N)
    {
        *size_norms = 0;
        *size_work = 0;

        // TODO: Some architectures have failures in sygvx with small-size kernels enabled, more investigation needed
        int device;
        HIP_CHECK(hipGetDevice(&device));
        hipDeviceProp_t deviceProperties;
        HIP_CHECK(hipGetDeviceProperties(&deviceProperties, device));
        if(deviceProperties.warpSize >= 64)
            return rocblas_status_success;
    }

    // size of space to store norms
    *size_norms = sizeof(T) * batch_count;

    // size of re-usable workspace
    // TODO: replace with rocBLAS call
    constexpr I ROCBLAS_DOT_NB = 512;
    *size_work = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    *size_work *= sizeof(T) * batch_count;

    return rocblas_status_success;
}

template <typename T, typename I, typename U>
rocblas_status
    rocsolver_larfg_argCheck(rocblas_handle handle, const I n, const I incx, T alpha, T x, U tau)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || incx < 1)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n > 1 && !x) || (n && !alpha) || (n && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larfg_template(rocblas_handle handle,
                                        const I n,
                                        U alpha,
                                        const rocblas_stride shifta,
                                        T* beta,
                                        const rocblas_stride shiftb,
                                        const rocblas_stride strideb,
                                        U x,
                                        const rocblas_stride shiftx,
                                        const I incx,
                                        const rocblas_stride stridex,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* work,
                                        T* norms)
{
    // TODO: How to get alpha for trace logging
    ROCSOLVER_ENTER("larfg", "n:", n, "shiftA:", shifta, "shiftX:", shiftx, "incx:", incx,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if n==1 return tau=0
    dim3 gridReset(1, batch_count, 1);
    dim3 setDiag(batch_count, 1, 1);
    dim3 threads(1, 1, 1);
    if(n == 1 && !COMPLEX)
    {
        ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, gridReset, threads, 0, stream, tau, strideP, 1,
                                0);
        if(beta != nullptr)
        {
            ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), setDiag, threads, 0, stream, beta, shiftb,
                                    strideb, alpha, shifta, n, stridex, (I)1, true);
        }
        return rocblas_status_success;
    }

    // if n is small, use small-size kernel
    if(n <= LARFG_SSKER_MAX_N)
    {
        // TODO: Some architectures have failures in sygvx with small-size kernels enabled, more investigation needed
        int device;
        HIP_CHECK(hipGetDevice(&device));
        hipDeviceProp_t deviceProperties;
        HIP_CHECK(hipGetDeviceProperties(&deviceProperties, device));
        if(deviceProperties.warpSize >= 64)
        {
            return larfg_run_small(handle, n, alpha, shifta, stridex, beta, shiftb, strideb, x,
                                   shiftx, incx, stridex, tau, strideP, batch_count);
        }
    }

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // compute squared norm of x
    rocblasCall_dot<COMPLEX, T>(handle, n - 1, x, shiftx, incx, stridex, x, shiftx, incx, stridex,
                                batch_count, norms, work);

    // set value of tau and beta and scalling factor for vector x
    // alpha <- beta, norms <- scaling
    ROCSOLVER_LAUNCH_KERNEL((set_taubeta<T, I>), dim3(batch_count), dim3(1), 0, stream, tau,
                            strideP, norms, alpha, shifta, stridex, beta, shiftb, strideb);

    // compute vector v=x*norms
    rocblasCall_scal<T>(handle, n - 1, norms, 1, x, shiftx, incx, stridex, batch_count);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larfg_template(rocblas_handle handle,
                                        const I n,
                                        U alpha,
                                        const rocblas_stride shifta,
                                        U x,
                                        const rocblas_stride shiftx,
                                        const I incx,
                                        const rocblas_stride stridex,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* work,
                                        T* norms)
{
    return rocsolver_larfg_template(handle, n, alpha, shifta, (T*)nullptr, 0, 0, x, shiftx, incx,
                                    stridex, tau, strideP, batch_count, work, norms);
}

ROCSOLVER_END_NAMESPACE
