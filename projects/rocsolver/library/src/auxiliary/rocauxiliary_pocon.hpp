/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routines (version 3.12.0) --
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

#include "rocauxiliary_trcon.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
void rocsolver_pocon_getMemorySize(const I n,
                                   const I batch_count,
                                   size_t* size_work_v,
                                   size_t* size_work_x,
                                   size_t* size_work_isgn,
                                   size_t* size_scalars,
                                   size_t* size_work_trsm_1,
                                   size_t* size_work_trsm_2,
                                   size_t* size_work_trsm_3,
                                   size_t* size_work_trsm_4,
                                   bool* optim_mem)
{
    // the workspace is that of TRCON without the norm of A
    size_t size_work_norm, size_anorm;
    rocsolver_trcon_getMemorySize<T, I, S>(rocsolver_norm_type_one, n, batch_count, &size_work_norm,
                                           &size_anorm, size_work_v, size_work_x, size_work_isgn,
                                           size_scalars, size_work_trsm_1, size_work_trsm_2,
                                           size_work_trsm_3, size_work_trsm_4, optim_mem);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_pocon_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        const I lda,
                                        T A,
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
    if((n && !A) || (n && !anorm) || (n && !rcond))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_pocon_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const S* anorm,
                                        S* rcond,
                                        const I batch_count,
                                        T* work_v,
                                        T* work_x,
                                        I* work_isgn,
                                        void* scalars,
                                        const bool optim_mem,
                                        void* work_trsm_1,
                                        void* work_trsm_2,
                                        void* work_trsm_3,
                                        void* work_trsm_4,
                                        const I max_iter)
{
    ROCSOLVER_ENTER("pocon", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
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

    std::vector<S> h_anorm(batch_count);
    HIP_CHECK(hipMemcpyAsync(h_anorm.data(), anorm, sizeof(S) * batch_count, hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    rocsolver_hybrid_storage<T, I, U> hA;
    ROCBLAS_CHECK(hA.init_pointers_only(A, shiftA, strideA, batch_count, stream));

    // the estimation is driven from the host, one matrix at a time
    for(I b = 0; b < batch_count; b++)
    {
        // as in LAPACK, rcond = 0 if anorm = 0
        if(h_anorm[b] == 0)
        {
            HIP_CHECK(hipMemsetAsync(rcond + b, 0, sizeof(S), stream));
            continue;
        }

        // inv(A) is Hermitian: its 1-norm and infinity-norm are equal
        ROCBLAS_CHECK(rocsolver_con_estimate<false, T, I, S>(
            handle, rocsolver_norm_type_one, uplo, rocblas_diagonal_non_unit, n, (T*)hA[b], lda,
            anorm + b, rcond + b, work_v, work_x, work_isgn, scalars, optim_mem, work_trsm_1,
            work_trsm_2, work_trsm_3, work_trsm_4, max_iter, stream));
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
