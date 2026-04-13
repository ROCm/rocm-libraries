/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T, typename I>
void rocsolver_gehd2_getMemorySize(const I n,
                                   const I ilo,
                                   const I ihi,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    using S = decltype(std::real(T{}));

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    const I dim = ihi - ilo;

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, s3, w1, w2, w3;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_right, ihi, dim, batch_count,
                                             size_scalars, &s1, &w1);
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_left, dim, n - ilo, batch_count,
                                             size_scalars, &s2, &w2);
    rocsolver_larfg_getMemorySize<T>(dim, batch_count, &w3, &s3);
    *size_work_workArr = std::max({w1, w2, w3});
    *size_Abyx_norms = std::max({s1, s2, s3});

    // size of array to store temporary diagonal values
    *size_diag = sizeof(S) * dim * batch_count;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_gehd2_argCheck(rocblas_handle handle,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        const I lda,
                                        T A,
                                        U ipiv,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0 || n && (ilo < 1 || ihi < ilo || ihi > n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n > 1 && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gehd2_template(rocblas_handle handle,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms,
                                        void* diag)
{
    ROCSOLVER_ENTER("gehd2", "n:", n, "ilo:", ilo, "ihi:", ihi, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);
    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const I dim = ihi - ilo;
    for(I i1 = ilo; i1 < ihi; ++i1)
    {
        const I i = i1 - 1;

        // generate Householder reflector to work on column i
        rocsolver_larfg_template<T>(handle, ihi - i1, A, shiftA + idx2D(i + 1, i, lda), (S*)diag,
                                    i1 - ilo, dim, A,
                                    shiftA + idx2D(std::min(i + 2, n - 1), i, lda), (I)1, strideA,
                                    (ipiv + i), strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // apply reflector from the right to A(0:ihi-1,i+1:ihi-1)
        rocsolver_larf_template(handle, rocblas_side_right, ihi, ihi - i1, A,
                                shiftA + idx2D(i + 1, i, lda), (I)1, strideA, (ipiv + i), strideP,
                                A, shiftA + idx2D(0, i + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);

        // conjugate tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, (I)1, ipiv, i, (I)1, strideP, batch_count);

        // apply reflector from the left to A(i+1:ihi-1,i+1:n-1)
        rocsolver_larf_template(handle, rocblas_side_left, ihi - i1, n - i1, A,
                                shiftA + idx2D(i + 1, i, lda), (I)1, strideA, (ipiv + i), strideP,
                                A, shiftA + idx2D(i + 1, i + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);

        // restore tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, (I)1, ipiv, i, (I)1, strideP, batch_count);
    }

    // restore subdiagonal values of A
    constexpr int DIAG_NTHREADS = 64;
    I blocks = (dim - 1) / DIAG_NTHREADS + 1;
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>), dim3(batch_count, blocks, 1),
                            dim3(1, DIAG_NTHREADS, 1), 0, stream, (S*)diag, 0, dim, A,
                            shiftA + idx2D(ilo, ilo - 1, lda), lda, strideA, dim);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
