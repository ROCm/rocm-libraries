/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_gelq2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_right, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(T) * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_gelq2_gelqf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gelq2_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms,
                                        T* diag)
{
    ROCSOLVER_ENTER("gelq2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = std::min(m, n); // total number of pivots

    for(rocblas_int j = 0; j < dim; ++j)
    {
        // conjugate the jth row of A
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        batch_count);

        // generate Householder reflector to work on row j
        rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                                 (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // insert one in A(j,j) tobuild/apply the householder matrix
        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                                0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                1, true);

        // Apply Householder reflector to the rest of matrix from the right
        if(j < m - 1)
        {
            rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                    shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                    scalars, Abyx_norms, (T**)work_workArr);
        }

        // restore original value of A(j,j)
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                dim3(1, 1, 1), 0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda),
                                lda, strideA, 1);

        // restore the jth row of A
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        batch_count);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
