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

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_labrd_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_norms)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_norms = 0;
        return;
    }

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    size_t s1, s2;

    // size of array of pointers (batched cases)
    if(BATCHED)
        s1 = 2 * sizeof(T*) * batch_count;
    else
        s1 = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(std::max(m, n), batch_count, &s2, size_norms);

    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    *size_work_workArr = std::max(s1, s2);
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_labrd_argCheck(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        const rocblas_int lda,
                                        const rocblas_int ldx,
                                        const rocblas_int ldy,
                                        T A,
                                        S D,
                                        S E,
                                        U tauq,
                                        U taup,
                                        T X,
                                        T Y,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || nb < 0 || nb > std::min(m, n) || lda < m || ldx < m || ldy < n
       || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (nb && !D) || (nb && !E) || (nb && !tauq) || (nb && !taup)
       || (m && nb && !X) || (n && nb && !Y))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_labrd_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
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
                                        T* X,
                                        const rocblas_int shiftX,
                                        const rocblas_int ldx,
                                        const rocblas_stride strideX,
                                        T* Y,
                                        const rocblas_int shiftY,
                                        const rocblas_int ldy,
                                        const rocblas_stride strideY,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* norms)
{
    ROCSOLVER_ENTER("labrd", "m:", m, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "shiftX:", shiftX, "ldx:", ldx, "shiftY:", shiftY, "ldy:", ldy,
                    "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    if(m >= n)
    {
        // generate upper bidiagonal form
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, Y, shiftY + idx2D(j, 0, ldy), ldy, strideY,
                                            batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, m - j, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, Y, shiftY + idx2D(j, 0, ldy), ldy, strideY,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, (T**)work_workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, Y, shiftY + idx2D(j, 0, ldy), ldy, strideY,
                                            batch_count);
            rocblasCall_gemv<T>(handle, rocblas_operation_none, m - j, j,
                                cast2constType<T>(scalars), 0, X, shiftX + idx2D(j, 0, lda), ldx,
                                strideX, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, (T**)work_workArr);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(
                handle,
                m - j, // order of reflector
                A, shiftA + idx2D(j, j, lda), // value of alpha
                A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), // vector x to work on
                1, strideA, // inc of x
                (tauq + j), strideQ, // tau
                batch_count, (T*)work_workArr, norms);

            ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, 1, j < n - 1);

            if(j < n - 1)
            {
                // compute column j of Y
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, m - j, n - j - 1,
                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
                    A, shiftA + idx2D(j, j, lda), 1, strideA, cast2constType<T>(scalars + 1), 0, Y,
                    shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, m - j, j,
                                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, 0, lda),
                                    lda, strideA, A, shiftA + idx2D(j, j, lda), 1, strideA,
                                    cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(0, j, ldy),
                                    1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, n - j - 1, j, cast2constType<T>(scalars), 0, Y,
                    shiftY + idx2D(j + 1, 0, ldy), ldy, strideY, Y, shiftY + idx2D(0, j, ldy), 1,
                    strideY, cast2constType<T>(scalars + 2), 0, Y, shiftY + idx2D(j + 1, j, ldy), 1,
                    strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, m - j, j,
                                    cast2constType<T>(scalars + 2), 0, X, shiftX + idx2D(j, 0, ldx),
                                    ldx, strideX, A, shiftA + idx2D(j, j, lda), 1, strideA,
                                    cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(0, j, ldy),
                                    1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, j, n - j - 1,
                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA,
                    Y, shiftY + idx2D(0, j, ldy), 1, strideY, cast2constType<T>(scalars + 2), 0, Y,
                    shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_scal<T>(handle, n - j - 1, (tauq + j), strideQ, Y,
                                    shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count);

                // update row j of A
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n, A, shiftA + idx2D(j, 0, lda), lda,
                                                strideA, batch_count);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, n - j - 1, j + 1, cast2constType<T>(scalars), 0,
                    Y, shiftY + idx2D(j + 1, 0, ldy), ldy, strideY, A, shiftA + idx2D(j, 0, lda),
                    lda, strideA, cast2constType<T>(scalars + 2), 0, A,
                    shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count, (T**)work_workArr);

                if(COMPLEX)
                {
                    rocsolver_lacgv_template<T>(handle, j + 1, A, shiftA + idx2D(j, 0, lda), lda,
                                                strideA, batch_count);
                    rocsolver_lacgv_template<T>(handle, j, X, shiftX + idx2D(j, 0, ldx), ldx,
                                                strideX, batch_count);
                }

                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, j, n - j - 1,
                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA,
                    X, shiftX + idx2D(j, 0, ldx), ldx, strideX, cast2constType<T>(scalars + 2), 0,
                    A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count, (T**)work_workArr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, j, X, shiftX + idx2D(j, 0, ldx), ldx,
                                                strideX, batch_count);

                // generate Householder reflector to work on row j
                rocsolver_larfg_template(
                    handle,
                    n - j - 1, // order of reflector
                    A, shiftA + idx2D(j, j + 1, lda), // value of alpha
                    A, shiftA + idx2D(j, std::min(j + 2, n - 1), lda), // vector x to work on
                    lda, strideA, // inc of x
                    (taup + j), strideP, // tau
                    batch_count, (T*)work_workArr, norms);

                ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                        dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                        shiftA + idx2D(j, j + 1, lda), lda, strideA, 1, true);

                // compute column j of X
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, n - j - 1,
                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda), lda,
                    strideA, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
                    cast2constType<T>(scalars + 1), 0, X, shiftX + idx2D(j + 1, j, ldx), 1, strideX,
                    batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j + 1,
                                    cast2constType<T>(scalars + 2), 0, Y,
                                    shiftY + idx2D(j + 1, 0, ldy), ldy, strideY, A,
                                    shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                    cast2constType<T>(scalars + 1), 0, X, shiftX + idx2D(0, j, ldx),
                                    1, strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, j + 1, cast2constType<T>(scalars), 0,
                    A, shiftA + idx2D(j + 1, 0, lda), lda, strideA, X, shiftX + idx2D(0, j, ldx), 1,
                    strideX, cast2constType<T>(scalars + 2), 0, X, shiftX + idx2D(j + 1, j, ldx), 1,
                    strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, j, n - j - 1, cast2constType<T>(scalars + 2), 0,
                    A, shiftA + idx2D(0, j + 1, lda), lda, strideA, A,
                    shiftA + idx2D(j, j + 1, lda), lda, strideA, cast2constType<T>(scalars + 1), 0,
                    X, shiftX + idx2D(0, j, ldx), 1, strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, j, cast2constType<T>(scalars), 0, X,
                    shiftX + idx2D(j + 1, 0, ldx), ldx, strideX, X, shiftX + idx2D(0, j, ldx), 1,
                    strideX, cast2constType<T>(scalars + 2), 0, X, shiftX + idx2D(j + 1, j, ldx), 1,
                    strideX, batch_count, (T**)work_workArr);
                rocblasCall_scal<T>(handle, m - j - 1, (taup + j), strideP, X,
                                    shiftX + idx2D(j + 1, j, ldx), 1, strideX, batch_count);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                                lda, strideA, batch_count);
            }
        }
    }

    else
    {
        // generate lower bidiagonal form
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update row j of A
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, Y, shiftY + idx2D(j, 0, ldy), ldy,
                                strideY, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda),
                                lda, strideA, batch_count, (T**)work_workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);
                rocsolver_lacgv_template<T>(handle, j, X, shiftX + idx2D(j, 0, ldx), ldx, strideX,
                                            batch_count);
            }
            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j, lda), lda,
                                strideA, X, shiftX + idx2D(j, 0, ldx), ldx, strideX,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda),
                                lda, strideA, batch_count, (T**)work_workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, X, shiftX + idx2D(j, 0, ldx), ldx, strideX,
                                            batch_count);

            // generate Householder reflector to work on row j
            rocsolver_larfg_template(
                handle,
                n - j, // order of reflector
                A, shiftA + idx2D(j, j, lda), // value of alpha
                A, shiftA + idx2D(j, std::min(j + 1, n - 1), lda), // vector x to work on
                lda, strideA, // inc of x
                (taup + j), strideP, // tau
                batch_count, (T*)work_workArr, norms);

            ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, 1, j < m - 1);

            if(j < m - 1)
            {
                // compute column j of X
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, n - j, cast2constType<T>(scalars + 2),
                    0, A, shiftA + idx2D(j + 1, j, lda), lda, strideA, A, shiftA + idx2D(j, j, lda),
                    lda, strideA, cast2constType<T>(scalars + 1), 0, X,
                    shiftX + idx2D(j + 1, j, ldx), 1, strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j, j,
                                    cast2constType<T>(scalars + 2), 0, Y, shiftY + idx2D(j, 0, ldy),
                                    ldy, strideY, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    cast2constType<T>(scalars + 1), 0, X, shiftX + idx2D(0, j, ldx),
                                    1, strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, j, cast2constType<T>(scalars), 0, A,
                    shiftA + idx2D(j + 1, 0, lda), lda, strideA, X, shiftX + idx2D(0, j, ldx), 1,
                    strideX, cast2constType<T>(scalars + 2), 0, X, shiftX + idx2D(j + 1, j, ldx), 1,
                    strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - j,
                                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda),
                                    lda, strideA, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    cast2constType<T>(scalars + 1), 0, X, shiftX + idx2D(0, j, ldx),
                                    1, strideX, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, j, cast2constType<T>(scalars), 0, X,
                    shiftX + idx2D(j + 1, 0, ldx), ldx, strideX, X, shiftX + idx2D(0, j, ldx), 1,
                    strideX, cast2constType<T>(scalars + 2), 0, X, shiftX + idx2D(j + 1, j, ldx), 1,
                    strideX, batch_count, (T**)work_workArr);
                rocblasCall_scal<T>(handle, m - j - 1, (taup + j), strideP, X,
                                    shiftX + idx2D(j + 1, j, ldx), 1, strideX, batch_count);

                if(COMPLEX)
                {
                    rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                                strideA, batch_count);
                    // update column j of A
                    rocsolver_lacgv_template<T>(handle, j, Y, shiftY + idx2D(j, 0, ldy), ldy,
                                                strideY, batch_count);
                }

                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, j, cast2constType<T>(scalars), 0, A,
                    shiftA + idx2D(j + 1, 0, lda), lda, strideA, Y, shiftY + idx2D(j, 0, ldy), ldy,
                    strideY, cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, j, lda), 1,
                    strideA, batch_count, (T**)work_workArr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, j, Y, shiftY + idx2D(j, 0, ldy), ldy,
                                                strideY, batch_count);

                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, m - j - 1, j + 1, cast2constType<T>(scalars), 0,
                    X, shiftX + idx2D(j + 1, 0, lda), ldx, strideX, A, shiftA + idx2D(0, j, lda), 1,
                    strideA, cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, j, lda), 1,
                    strideA, batch_count, (T**)work_workArr);

                // generate Householder reflector to work on column j
                rocsolver_larfg_template(
                    handle,
                    m - j - 1, // order of reflector
                    A, shiftA + idx2D(j + 1, j, lda), // value of alpha
                    A, shiftA + idx2D(std::min(j + 2, m - 1), j, lda), // vector x to work on
                    1, strideA, // inc of x
                    (tauq + j), strideQ, // tau
                    batch_count, (T*)work_workArr, norms);

                ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                        dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                        shiftA + idx2D(j + 1, j, lda), lda, strideA, 1, true);

                // compute column j of Y
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, m - j - 1, n - j - 1,
                    cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda), lda,
                    strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                    cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(j + 1, j, ldy), 1, strideY,
                    batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, m - j - 1, j,
                                    cast2constType<T>(scalars + 2), 0, A,
                                    shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                                    shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                    cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(0, j, ldy),
                                    1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_none, n - j - 1, j, cast2constType<T>(scalars), 0, Y,
                    shiftY + idx2D(j + 1, 0, ldy), ldy, strideY, Y, shiftY + idx2D(0, j, ldy), 1,
                    strideY, cast2constType<T>(scalars + 2), 0, Y, shiftY + idx2D(j + 1, j, ldy), 1,
                    strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, m - j - 1, j + 1,
                                    cast2constType<T>(scalars + 2), 0, X,
                                    shiftX + idx2D(j + 1, 0, ldx), ldx, strideX, A,
                                    shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                    cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(0, j, ldy),
                                    1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_gemv<T>(
                    handle, rocblas_operation_conjugate_transpose, j + 1, n - j - 1,
                    cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA,
                    Y, shiftY + idx2D(0, j, ldy), 1, strideY, cast2constType<T>(scalars + 2), 0, Y,
                    shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count, (T**)work_workArr);
                rocblasCall_scal<T>(handle, n - j - 1, (tauq + j), strideQ, Y,
                                    shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count);
            }
            else
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                                strideA, batch_count);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
