/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_host_functions.hpp"
#include "lib_device_helpers.hpp"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S>
void run_sb2st_hb2st(rocblas_int n, rocblas_int nb, T* A, rocblas_int lda, S* D, S* E, T* work)
{
    for(rocblas_int s = 0; s < n - 1; s++)
    {
        rocblas_int sm_i = s + 1;
        rocblas_int sm_e = std::min(s + 1 + nb, n);
        rocblas_int su_i = sm_e;
        rocblas_int su_e = std::min(sm_e + nb, n);

        // generate Householder reflector
        rocblas_int mm = sm_e - sm_i;
        rocblas_int incx = 1;
        T tau = 0;
        call_larfg(mm, A[sm_i + s * lda], A + (sm_i + 1) + s * lda, incx, tau);
        E[s] = std::real(A[sm_i + s * lda]);
        A[sm_i + s * lda] = 1;

        // apply Householder reflector
        rocblas_int nn = su_e - sm_i;
        call_larf(rocblas_side_left, mm, nn, A + sm_i + s * lda, incx, conj(tau),
                  A + sm_i + sm_i * lda, lda, work);
        call_larf(rocblas_side_right, mm, mm, A + sm_i + s * lda, incx, tau, A + sm_i + sm_i * lda,
                  lda, work);
        for(rocblas_int i = su_i; i < su_e; i++)
            for(rocblas_int j = sm_i; j < sm_e; j++)
                A[i + j * lda] = conj(A[j + i * lda]);

        // save tau
        A[sm_i + s * lda] = tau;

        sm_i = su_i;
        sm_e = su_e;

        // complete the sweep
        while(sm_i < n)
        {
            su_i = sm_e;
            su_e = std::min(sm_e + nb, n);
            rocblas_int sd_i = std::max(sm_i - nb, 1);
            rocblas_int sd_e = sm_i;

            // generate Householder reflector
            mm = sm_e - sm_i;
            call_larfg(mm, A[sm_i + sd_i * lda], A + (sm_i + 1) + sd_i * lda, incx, tau);

            // copy Householder vector to column s
            A[sm_i + s * lda] = 1;
            for(rocblas_int i = sm_i + 1; i < sm_e; i++)
            {
                A[i + s * lda] = A[i + sd_i * lda];
                A[i + sd_i * lda] = 0;
            }

            // apply Householder reflector
            nn = su_e - sd_i - 1;
            call_larf(rocblas_side_left, mm, nn, A + sm_i + s * lda, incx, conj(tau),
                      A + sm_i + (sd_i + 1) * lda, lda, work);
            call_larf(rocblas_side_right, mm, mm, A + sm_i + s * lda, incx, tau,
                      A + sm_i + sm_i * lda, lda, work);
            for(rocblas_int i = su_i; i < su_e; i++)
                for(rocblas_int j = sm_i; j < sm_e; j++)
                    A[i + j * lda] = conj(A[j + i * lda]);
            for(rocblas_int i = sd_i; i < sd_e; i++)
                for(rocblas_int j = sm_i; j < sm_e; j++)
                    A[i + j * lda] = conj(A[j + i * lda]);

            // save tau
            A[sm_i + s * lda] = tau;

            sm_i = su_i;
            sm_e = su_e;
        }
    }

    for(rocblas_int i = 0; i < n; i++)
        D[i] = std::real(A[i + i * lda]);
}

template <bool BATCHED, typename T, typename S>
void rocsolver_sb2st_hb2st_getMemorySize(const rocblas_int n,
                                         const rocblas_int nb,
                                         const rocblas_int batch_count)
{
}

template <typename T, typename S>
rocblas_status rocsolver_sb2st_hb2st_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              const rocblas_int lda,
                                              T A,
                                              S* D,
                                              S* E,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nb < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n > 0 && !A) || (n > 0 && !D) || (n > 1 && !E))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("sb2st_hb2st", "n:", n, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream,
                                rocblas_evect_none, A, strideA, D, strideD, batch_count);
        return rocblas_status_success;
    }

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocsolver_hybrid_storage<T, rocblas_int, U> hA;
    rocsolver_hybrid_storage<S, rocblas_int, S*> hD;
    rocsolver_hybrid_storage<S, rocblas_int, S*> hE;

    ROCBLAS_CHECK(hA.init_async(n * lda, A, shiftA, strideA, batch_count, stream));
    ROCBLAS_CHECK(hD.init_async(n, D, 0, strideD, batch_count, stream));
    ROCBLAS_CHECK(hE.init_async(n - 1, E, 0, strideE, batch_count, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    T* hwork = nullptr;
    HIP_CHECK(hipHostMalloc(&hwork, sizeof(T) * (3 * nb)));

    for(rocblas_int bid = 0; bid < batch_count; bid++)
    {
        run_sb2st_hb2st<T, S>(n, nb, hA[bid], lda, hD[bid], hE[bid], hwork);
    }

    ROCBLAS_CHECK(hA.write_to_device_async(stream));
    ROCBLAS_CHECK(hD.write_to_device_async(stream));
    ROCBLAS_CHECK(hE.write_to_device_async(stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipHostFree(hwork));

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
