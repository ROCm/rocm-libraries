/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
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

#include "ideal_sizes.hpp"
#include "lapack/roclapack_sytrs.hpp"
#include "lapack_device_functions.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Sets *flag to 1 if D has a zero 1-by-1 diagonal block (ipiv[i] > 0 and A[i,i] = 0). Launch with
    a single block of BS1 threads. **/
template <typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS1)
    sycon_check_singularity(const I n, const T* A, const I lda, const I* ipiv, rocblas_int* flag)
{
    for(I i = threadIdx.x; i < n; i += BS1)
    {
        if(ipiv[i] > 0 && A[idx2D(i, i, lda)] == T(0))
            *flag = 1;
    }
}

/** Sets *flag to 1 if x has an element that is not finite. Launch with a single block of BS1
    threads. **/
template <typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS1)
    sycon_check_finite(const I n, const T* x, rocblas_int* flag)
{
    for(I i = threadIdx.x; i < n; i += BS1)
    {
        const T xi = x[i];
        bool finite;
        if constexpr(rocblas_is_complex<T>)
            finite = std::isfinite(xi.real()) && std::isfinite(xi.imag());
        else
            finite = std::isfinite(xi);
        if(!finite)
            *flag = 1;
    }
}

/** rcond = (1 / ainvnm) / anorm, or zero if ainvnm is zero or a solve overflowed (flag set). **/
template <typename S>
ROCSOLVER_KERNEL void
    sycon_compute_rcond(S* rcond, const S* ainvnm, const S* anorm, const rocblas_int* flag)
{
    if(threadIdx.x == 0)
    {
        const S est = *ainvnm;
        if(*flag || est == 0 || !std::isfinite(est))
            *rcond = 0;
        else
            *rcond = (S(1) / est) / *anorm;
    }
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_sycon_getMemorySize(rocblas_handle handle,
                                             const I n,
                                             const I lda,
                                             const I batch_count,
                                             size_t* size_work_v,
                                             size_t* size_work_x,
                                             size_t* size_work_isgn,
                                             size_t* size_scalars,
                                             size_t* size_work_sytrs)
{
    *size_work_v = 0;
    *size_work_x = 0;
    *size_work_isgn = 0;
    *size_scalars = 0;
    *size_work_sytrs = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    // vectors v, x and isgn of LACN2 (only one matrix is processed at a time)
    *size_work_v = sizeof(T) * n;
    *size_work_x = sizeof(T) * n;
    *size_work_isgn = rocblas_is_complex<T> ? 0 : sizeof(I) * n;

    // scalars of LACN2 (estimate, index, kase, jump), the singularity/overflow flag and the norm
    // of A on the host side, in 8-byte slots
    *size_scalars = 5 * sizeof(int64_t);

    // workspace for the solves (one right-hand side)
    return rocsolver_sytrs_getMemorySize<T, I>(handle, n, I(1), I(1), lda, n, size_work_sytrs);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_sycon_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        const I lda,
                                        T A,
                                        I* ipiv,
                                        const S* anorm,
                                        S* rcond)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(n && (!A || !ipiv || !anorm || !rcond))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_sycon_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        I* ipiv,
                                        const rocblas_stride strideP,
                                        const S* anorm,
                                        S* rcond,
                                        const I batch_count,
                                        T* work_v,
                                        T* work_x,
                                        I* work_isgn,
                                        void* scalars,
                                        void* work_sytrs,
                                        const size_t size_work_sytrs,
                                        const I max_iter)
{
    ROCSOLVER_ENTER("sycon", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return: rcond = 1 if n = 0
    if(n == 0)
    {
        if(rcond)
        {
            const I blocks = (batch_count - 1) / BS1 + 1;
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocks), dim3(BS1), 0, stream,
                                    rcond, batch_count, S(1));
        }
        return rocblas_status_success;
    }

    // scalars in 8-byte slots (see getMemorySize)
    int64_t* slots = (int64_t*)scalars;
    S* d_est = (S*)slots;
    I* d_max_idx = (I*)(slots + 1);
    rocblas_int* d_kase = (rocblas_int*)(slots + 2);
    rocblas_int* d_jump = (rocblas_int*)(slots + 3);
    rocblas_int* d_flag = (rocblas_int*)(slots + 4);

    rocsolver_hybrid_storage<T, I, U> hA;
    ROCBLAS_CHECK(hA.init_pointers_only(A, shiftA, strideA, batch_count, stream));

    std::vector<S> h_anorm(batch_count);
    HIP_CHECK(hipMemcpyAsync(h_anorm.data(), anorm, sizeof(S) * batch_count, hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // the estimation is driven from the host, one matrix at a time
    for(I b = 0; b < batch_count; b++)
    {
        T* Ab = (T*)hA[b];
        I* ipivb = ipiv + b * strideP;

        // as in LAPACK, rcond = 0 if anorm <= 0 or if D is singular
        rocblas_int h_flag = 0;
        if(h_anorm[b] > 0)
        {
            HIP_CHECK(hipMemsetAsync(d_flag, 0, sizeof(rocblas_int), stream));
            ROCSOLVER_LAUNCH_KERNEL((sycon_check_singularity<T, I>), dim3(1), dim3(BS1), 0, stream,
                                    n, Ab, lda, ipivb, d_flag);
            HIP_CHECK(hipMemcpyAsync(&h_flag, d_flag, sizeof(rocblas_int), hipMemcpyDeviceToHost,
                                     stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        if(!(h_anorm[b] > 0) || h_flag)
        {
            HIP_CHECK(hipMemsetAsync(rcond + b, 0, sizeof(S), stream));
            continue;
        }

        T* x = work_x;
        T* v = work_v;
        rocblas_int h_kase = 0;
        rocblas_int h_jump = 0;
        I h_iters = 1;

        do
        {
            ROCBLAS_CHECK(rocsolver_lacn2_template<T, I, S>(handle, n, &x, &v, work_isgn, d_est,
                                                            d_max_idx, d_kase, d_jump, &h_kase,
                                                            &h_jump, &h_iters, max_iter, stream));
            if(h_kase == 0)
                break;

            // as in LAPACK, both kase values solve A x = b (A is symmetric)
            ROCBLAS_CHECK(rocsolver_sytrs_template<T>(
                handle, uplo, n, I(1), Ab, rocblas_stride(0), lda, rocblas_stride(0), ipivb,
                rocblas_stride(0), x, rocblas_stride(0), n, rocblas_stride(n), I(1), work_sytrs,
                size_work_sytrs));

            // the solves are not scaled (unlike LAPACK's): an overflow means that A is
            // numerically singular, and rcond is set to zero
            ROCSOLVER_LAUNCH_KERNEL((sycon_check_finite<T, I>), dim3(1), dim3(BS1), 0, stream, n, x,
                                    d_flag);
        } while(h_kase != 0);

        ROCSOLVER_LAUNCH_KERNEL((sycon_compute_rcond<S>), dim3(1), dim3(1), 0, stream, rcond + b,
                                d_est, anorm + b, d_flag);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
