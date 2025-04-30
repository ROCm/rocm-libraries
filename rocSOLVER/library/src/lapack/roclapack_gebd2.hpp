/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
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
void rocsolver_gebd2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_both, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(std::max(m, n), batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_gebd2_gebrd_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              S D,
                                              S E,
                                              U tauq,
                                              U taup,
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
    if((m && n && (!A || !D || !tauq || !taup)) || (std::min(m, n) > 1 && !E))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gebd2_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tauq,
                                        const rocblas_stride strideQ,
                                        T* taup,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms)
{
    ROCSOLVER_ENTER("gebd2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = std::min(m, n); // total number of pivots

    if(m >= n)
    {
        // generate upper bidiagonal form
        for(rocblas_int j = 0; j < n; j++)
        {
            // generate Householder reflector H(j)
            rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                                     shiftA + idx2D(std::min(j + 1, m - 1), j, lda), 1, strideA,
                                     (tauq + j), strideQ, batch_count, (T*)work_workArr, Abyx_norms);

            // copy A(j,j) to D and insert one to build/apply the householder matrix
            ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

            // Apply Householder reflector H(j)
            if(j < n - 1)
            {
                // conjugate tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                        shiftA + idx2D(j, j, lda), 1, strideA, (tauq + j), strideQ,
                                        A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                        scalars, Abyx_norms, (T**)work_workArr);

                // restore tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);
            }

            // restore original value of A(j,j)
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, 1);

            if(j < n - 1)
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                                lda, strideA, batch_count);

                // generate Householder reflector G(j)
                rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda), A,
                                         shiftA + idx2D(j, std::min(j + 2, n - 1), lda), lda,
                                         strideA, (taup + j), strideP, batch_count,
                                         (T*)work_workArr, Abyx_norms);

                // copy A(j,j+1) to E and insert one to build/apply the householder
                // matrix
                ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                        dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, 1, true);

                // Apply Householder reflector G(j)
                rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j - 1, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, (taup + j),
                                        strideP, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                        batch_count, scalars, Abyx_norms, (T**)work_workArr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                                lda, strideA, batch_count);

                // restore original value of A(j,j+1)
                ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                        dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, 1);
            }
            else
            {
                // zero taup(j)
                ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0,
                                        stream, taup + j, strideP, 1, 0);
            }
        }
    }
    else
    {
        // generate lower bidiagonal form
        for(rocblas_int j = 0; j < m; j++)
        {
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                            strideA, batch_count);

            // generate Householder reflector G(j)
            rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                                     shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                                     (taup + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

            // copy A(j,j) to D and insert one to build/apply the householder matrix
            ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

            // Apply Householder reflector G(j)
            if(j < m - 1)
            {
                rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                        shiftA + idx2D(j, j, lda), lda, strideA, (taup + j),
                                        strideP, A, shiftA + idx2D(j + 1, j, lda), lda, strideA,
                                        batch_count, scalars, Abyx_norms, (T**)work_workArr);
            }

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                            strideA, batch_count);

            // restore original value of A(j,j)
            ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, 1);

            if(j < m - 1)
            {
                // generate Householder reflector H(j)
                rocsolver_larfg_template(handle, m - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                         shiftA + idx2D(std::min(j + 2, m - 1), j, lda), 1, strideA,
                                         (tauq + j), strideQ, batch_count, (T*)work_workArr,
                                         Abyx_norms);

                // copy A(j+1,j) to D and insert one to build/apply the householder
                // matrix
                ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                        dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                        shiftA + idx2D(j + 1, j, lda), lda, strideA, 1, true);

                // conjugate tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                // Apply Householder reflector H(j)
                rocsolver_larf_template(handle, rocblas_side_left, m - j - 1, n - j - 1, A,
                                        shiftA + idx2D(j + 1, j, lda), 1, strideA, (tauq + j),
                                        strideQ, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                        batch_count, scalars, Abyx_norms, (T**)work_workArr);

                // restore tauq
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                // restore original value of A(j,j+1)
                ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                        dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                        shiftA + idx2D(j + 1, j, lda), lda, strideA, 1);
            }
            else
            {
                // zero tauq(j)
                ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0,
                                        stream, tauq + j, strideQ, 1, 0);
            }
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
