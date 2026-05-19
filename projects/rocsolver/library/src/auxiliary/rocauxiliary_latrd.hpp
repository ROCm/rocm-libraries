/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <sstream>

#include <hip/hip_cooperative_groups.h>

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include "rocsolver_device_workspace.hpp"
#include <rocprofiler-sdk-roctx/roctx.h>

static bool print_debug_messages_latrd_forsytrd
    = std::getenv("PRINT_DEBUG") != nullptr ? true : false;

static bool latrd_forsytrd_multi_kernel = std::getenv("LATRD_MULTI_KERNEL") != nullptr ? true : false;

static bool force_coop_launch = std::getenv("COOP_LAUNCH") != nullptr ? true : false;

#define HIP_TRACE(call)                                                                      \
    do                                                                                       \
    {                                                                                        \
        hipError_t _err = (call);                                                            \
        hipError_t _last = hipGetLastError(); /* resets after read */                        \
        if(print_debug_messages_latrd_forsytrd)                                              \
            std::fprintf(stderr, "[HIP_TRACE] %s:%d  call=%-60s  ret=%s(%d)  last=%s(%d)\n", \
                         __FILE__, __LINE__, #call, hipGetErrorName(_err), (int)_err,        \
                         hipGetErrorName(_last), (int)_last);                                \
        if(_err != hipSuccess)                                                               \
        {                                                                                    \
            std::fprintf(stderr, "[HIP_TRACE] %s:%d  call=%-60s  ret=%s(%d)  last=%s(%d)\n", \
                         __FILE__, __LINE__, #call, hipGetErrorName(_err), (int)_err,        \
                         hipGetErrorName(_last), (int)_last);                                \
        }                                                                                    \
    } while(0)

ROCSOLVER_BEGIN_NAMESPACE

template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) latrd_dot_scale_axpy(const I n,
                                                                       U AA,
                                                                       const rocblas_stride shiftA,
                                                                       const rocblas_stride strideA,
                                                                       T* WW,
                                                                       const rocblas_stride shiftW,
                                                                       const rocblas_stride strideW,
                                                                       T* tauA,
                                                                       const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WW, bid, shiftW, strideW);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // shared variables
    __shared__ T sval[MAX_THDS / WarpSize];
    __shared__ T sh_A[MAX_THDS];
    __shared__ T sh_W[MAX_THDS];

    // dot
    T norm2 = 0;
    for(I i = tid; i < n; i += MAX_THDS)
    {
        T tempA = A[i];
        T tempW = W[i];
        if(i < MAX_THDS)
        {
            sh_A[i] = tempA;
            sh_W[i] = tempW;
        }

        norm2 += tempA * conj(tempW);
    }

    // reduce squared entries to find squared norm of x
    norm2 += shift_left(norm2, 1);
    norm2 += shift_left(norm2, 2);
    norm2 += shift_left(norm2, 4);
    norm2 += shift_left(norm2, 8);
    norm2 += shift_left(norm2, 16);
    if(warpSize > 32)
        norm2 += shift_left(norm2, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm2;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
            norm2 += sval[k];
        sval[0] = -0.5 * tau[0] * norm2;
    }
    __syncthreads();

    // axpy
    for(I i = tid; i < n; i += MAX_THDS)
    {
        if(i < MAX_THDS)
            W[i] = sh_W[i] + sval[0] * sh_A[i];
        else
            W[i] = W[i] + sval[0] * A[i];
    }
}

/********************************************************************************/
/******************* Host functions for latrd api *******************************/
/********************************************************************************/
template <typename T, typename S, typename U>
rocblas_status rocsolver_latrd_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int lda,
                                        const rocblas_int ldw,
                                        T A,
                                        S E,
                                        U tau,
                                        U W,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || k < 0 || k > n || lda < n || ldw < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !E) || (n && !tau) || (n && k && !W))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T>
void rocsolver_latrd_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_norms,
                                   size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0;

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // size of re-usable workspace
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    *size_norms = std::max(n1, n2);
    *size_work = std::max({w1, w2, w3});
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        T* W,
                                        const rocblas_int shiftW,
                                        const rocblas_int ldw,
                                        const rocblas_stride strideW,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T* norms,
                                        T** workArr)
{
    ROCSOLVER_ENTER("latrd", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);
    roctxRangePush("rocsolver_latrd");

    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    std::stringstream ss;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        ss = std::stringstream();
        print_device_matrix(ss, "Scalars", 3, 1, scalars, 3);
        print_device_matrix(ss, "Input matrix A", n, n, A, lda);
        std::cout << ss.str();
        for(rocblas_int j = 0; j < k; ++j)
        {
            printf("::: Iteration: %d\n", j);
            if(j > 0)
            {
                ss = std::stringstream();
                print_device_matrix(ss, "Matrix A with new reflector", n, n, A, lda);
                std::cout << ss.str();

                ss = std::stringstream();
                print_device_matrix(ss, "Updated matrix W", n, k, W, ldw);
                std::cout << ss.str();
            }

            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, j, W, shiftW + idx2D(j, 0, ldw), ldw, strideW,
                                            batch_count);
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j, 0, ldw), ldw,
                                strideW, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                                strideA, batch_count, workArr);

            if(j > 0)
            {
                ss = std::stringstream();
                print_device_matrix(ss, "Updated matrix A with reflector", n, n, A, lda);
                std::cout << ss.str();
            }

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                                            batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                     strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1,
                                     strideA, (tau + j), strideP, batch_count, work, norms);

            // compute/update column j of W
            rocblasCall_symv_hemv<T>(
                handle, uplo, n - 1 - j, (scalars + 2), 0, A, shiftA + idx2D(j + 1, j + 1, lda),
                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA, (scalars + 1), 0, W,
                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count, work, workArr);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W 1/6 ", n, k, W, ldw);
            std::cout << ss.str();

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W 2/6 ", n, k, W, ldw);
            std::cout << ss.str();

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W 3/6 ", n, k, W, ldw);
            std::cout << ss.str();

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, n - j - 1, j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j + 1, 0, lda),
                                lda, strideA, A, shiftA + idx2D(j + 1, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W, shiftW + idx2D(0, j, ldw), 1,
                                strideW, batch_count, workArr);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W 4/6 ", n, k, W, ldw);
            std::cout << ss.str();

            rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(j + 1, 0, ldw),
                                ldw, strideW, W, shiftW + idx2D(0, j, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(j + 1, j, ldw),
                                1, strideW, batch_count, workArr);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W 5/6 ", n, k, W, ldw);
            std::cout << ss.str();

            rocblasCall_scal<T>(handle, n - j - 1, (tau + j), strideP, W,
                                shiftW + idx2D(j + 1, j, ldw), 1, strideW, batch_count);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W 6/6 ", n, k, W, ldw);
            std::cout << ss.str();

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, n - 1 - j, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, W,
                                    shiftW + idx2D(j + 1, j, ldw), strideW, tau + j, strideP);

            ss = std::stringstream();
            print_device_matrix(ss, "Matrix W with new entry", n, k, W, ldw);
            std::cout << ss.str();
        }
    }

    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;
        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;
            // update column j of A with reflector computed in step j-1
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j, jw + 1, ldw), ldw, strideW,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
            {
                rocsolver_lacgv_template<T>(handle, n - 1 - j, W, shiftW + idx2D(j, jw + 1, ldw),
                                            ldw, strideW, batch_count);
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);
            }

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j + 1, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j, lda), 1,
                                strideA, batch_count, workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);

            // generate Householder reflector to work on column j
            rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1, strideE,
                                     A, shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1),
                                     strideP, batch_count, work, norms);

            // compute/update column j of W
            rocblasCall_symv_hemv<T>(handle, uplo, j, (scalars + 2), 0, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, (scalars + 1), 0, W,
                                     shiftW + idx2D(0, jw, ldw), 1, strideW, batch_count, work,
                                     workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, j, n - 1 - j,
                                cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(0, j + 1, lda),
                                lda, strideA, A, shiftA + idx2D(0, j, lda), 1, strideA,
                                cast2constType<T>(scalars + 1), 0, W,
                                shiftW + idx2D(j + 1, jw, ldw), 1, strideW, batch_count, workArr);

            rocblasCall_gemv<T>(handle, rocblas_operation_none, j, n - 1 - j,
                                cast2constType<T>(scalars), 0, W, shiftW + idx2D(0, jw + 1, ldw),
                                ldw, strideW, W, shiftW + idx2D(j + 1, jw, ldw), 1, strideW,
                                cast2constType<T>(scalars + 2), 0, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count, workArr);

            rocblasCall_scal<T>(handle, j, (tau + j - 1), strideP, W, shiftW + idx2D(0, jw, ldw), 1,
                                strideW, batch_count);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, j, A, shiftA + idx2D(0, j, lda),
                                    strideA, W, shiftW + idx2D(0, jw, ldw), strideW, tau + j - 1,
                                    strideP);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    roctxRangePop();
    return rocblas_status_success;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
auto rocsolver_latrd_getWorkItems(rocblas_handle handle,
                                  const rocblas_fill uplo,
                                  const rocblas_int n,
                                  const rocblas_int k,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* E,
                                  const rocblas_stride strideE,
                                  T* tau,
                                  const rocblas_stride strideP,
                                  T* W,
                                  const rocblas_int shiftW,
                                  const rocblas_int ldw,
                                  const rocblas_stride strideW,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements for calling LARFG
    size_t size_work, size_norms;
    rocsolver_latrd_getMemorySize<false, T>(n, k, batch_count, &size_scalars, &size_work,
                                            &size_norms, &size_workArr);

    auto work_items = create_work_item({"latrd_scalars", size_scalars})
        + create_work_item({"latrd_workArr", size_workArr})
        + create_work_item({"latrd_work", size_work}) + create_work_item({"latrd_norms", size_norms});

    return work_items;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tau,
                                        const rocblas_stride strideP,
                                        T* W,
                                        const rocblas_int shiftW,
                                        const rocblas_int ldw,
                                        const rocblas_stride strideW,
                                        const rocblas_int batch_count,
                                        rocsolver_device_workspace_ptr_t dwptr)
{
    ROCSOLVER_INIT_DEVICE_WORKSPACE(
        dwptr,
        rocsolver_latrd_getWorkItems(handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE, tau,
                                     strideP, W, shiftW, ldw, strideW, batch_count));

    T* scalars = (T*)dwptr->work("latrd_scalars");
    T* work = (T*)dwptr->work("latrd_work");
    T* norms = (T*)dwptr->work("latrd_norms");
    T** workArr = (T**)dwptr->work("latrd_workArr");

    if(dwptr->size("latrd_scalars") > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_latrd_template<T>(handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE, tau,
                                       strideP, W, shiftW, ldw, strideW, batch_count, (T*)scalars,
                                       (T*)work, (T*)norms, (T**)workArr);
}

/**************************************************************************************/
/***************** Kernels/Device functions *******************************************/
/**************************************************************************************/

/***** Kernel to reduce results inter-groups *****/
/*************************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_reduce_kernel(const rocblas_fill uplo,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int c,
                                          T* dacA,
                                          const rocblas_int ldd,
                                          const rocblas_stride strideD,
                                          U yA,
                                          const rocblas_int shiftY,
                                          const rocblas_int ldy,
                                          const rocblas_stride strideY,
                                          T* workA,
                                          const rocblas_stride strideblk)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idr = bidr * threadsr + tidr;
    int idc = bidc * threadsc + tidc;

    // select batch instance
    bool upper = (uplo == rocblas_fill_upper);
    T* y1 = upper ? load_ptr_batch<T>(yA, bid, shiftY, strideY) : workA + bid * strideblk;
    T* y2 = upper ? workA + bid * strideblk : load_ptr_batch<T>(yA, bid, shiftY, strideY);
    T* dac = dacA + bid * strideD;

    // rpgr is the number of rounds a group should run
    // to cover all the rows
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    int i, it;

    // Registers/LDS:
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* tmp = reinterpret_cast<T*>(smem);
    T val;
    T* y;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        val = 0;

        it = (i < c) ? i : i - c;
        y = (i < c) ? y1 : y2;

        // read groups results
        if(i < m)
        {
            for(int j = idc; j < n; j += totalthsc)
                val += dac[i + j * ldd];
        }
        tmp[tidr + tidc * threadsr] = val;
        __syncthreads();

        // reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                val += tmp[tidr + (tidc + r) * threadsr];
                tmp[tidr + tidc * threadsr] = val;
            }
            __syncthreads();
        }

        // write results
        if(tidc == 0 && i < m)
            y[it] = val;
    }
}

/***** Kernels to update column of A *****/
/*****************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_updateA_kernel(const rocblas_int mm,
                                                 const rocblas_int k,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* WA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);

    /* ------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            y = A(0:c, c)
            A1 = A(0:c, c+1:mm-1)
            A2 = W(0:c, cw+1:k-1)
            x1 = W(c, cw+1:k-1)
            x2 = A(c, c+1:mm-1)

        operation:
            y = y - A1 * x1' - A2 * x2'
    ------------------------ */
    int n = mm - c - 1;
    int m = c + 1;
    int cw = c - mm + k;
    T* y = A + idx2D(0, c, lda);
    T* A1 = A + idx2D(0, c + 1, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda2 = ldw;
    T* x1 = W + idx2D(c, cw + 1, ldw);
    int incx1 = ldw;
    T* x2 = A + idx2D(c, c + 1, lda);
    int incx2 = lda;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? conj(x1[j * incx1]) : 0;
            sx2 = (j < n) ? conj(x2[j * incx2]) : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write results
        if(tidc == 0 && i < m)
            y[i] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_updateA_kernel(const rocblas_int mm,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* WA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);

    /* ------------------------
    formulate gemv problem:

        components:
            y = A(c:mm-1, c)
            A1 = A(c:mm-1, 0:c-1)
            A2 = W(c:mm-1, 0:c-1)
            x1 = W(c, 0:c-1)
            x2 = A(c, 0:c-1)

        operation:
            y = y - A1 * x1' - A2 * x2'
    ------------------------ */
    int m = mm - c;
    int n = c;
    T* y = A + idx2D(c, c, lda);
    T* A1 = A + idx2D(c, 0, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(c, 0, ldw);
    int lda2 = ldw;
    T* x1 = W + idx2D(c, 0, ldw);
    int incx1 = ldw;
    T* x2 = A + idx2D(c, 0, lda);
    int incx2 = lda;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? conj(x1[j * incx1]) : 0;
            sx2 = (j < n) ? conj(x2[j * incx2]) : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write results
        if(tidc == 0 && i < m)
            y[i] = ac;
    }
}

/***** Kernels to compute column of W *****/
/******************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_symv_kernel(const rocblas_int mm,
                                                       const rocblas_int k,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* WA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            A1 = A(0:c-1, :)
            A2 = W(0:c-1, cw+1:k-1)
            x = A(0:c-1, c)
            dac = temporary buffer

        operation:
                  [   A1(:, 0:c-1)   ]
                  [        0         ]
            dac = [ A1(:, c+1:mm-1)' ] * x
                  [        A2'       ]

        Notes:
            1. Here A1(:, 0:c-1) is symmetric (data referenced only above diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ 0  ]
                  [ y2 ] <- reduce(dac)
                  [ y3 ]
              where
                  [ y1 ]
                  [ 0  ] = W(:, cw)
                  [ y2 ]
                    y3   = work (temp buffer)
    ------------------------------ */
    int n = c;
    int cc = mm - c - 1;
    int m = mm + cc;
    int cw = c - mm + k;
    T* A1 = A;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T const* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < mm) ? i : i - mm;
        a = (i < mm) ? A1 : A2;
        ld = (i < mm) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (i > c) ? conj(a[j + it * ld]) * sx
                    : (j < i) ? conj(a[j + i * ld]) * sx
                              : a[i + j * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_symv_kernel(const rocblas_int mm,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* WA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            A1 = W(c+1:mm-1, 0:c-1)
            A2 = A(c+1:mm-1, :)
            x = A(c+1,mm-1, c)
            dac = temporary buffer

        operation:
                  [       A1'       ]
            dac = [  A2(:, 0:c-1)'  ] * x
                  [        0        ]
                  [ A2(:, c+1:mm-1) ]

        Notes:
            1. Here A2(:, c+1:mm-1) is symmetric (data referenced only below diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
                  [ 0  ]
                  [ y3 ]
              where
                    y1   = work (temp buffer)
                  [ y2 ]
                  [ 0  ] = W(:, c)
                  [ y3 ]
    ------------------------------ */
    int n = mm - c - 1;
    int m = mm + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < c) ? i : i - c;
        a = (i < c) ? A1 : A2;
        ld = (i < c) ? lda1 : lda2;
        it2 = it - c - 1;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (it < c)  ? conj(a[j + it * ld]) * sx
                    : (j > it2) ? conj(a[j + (it2 + c + 1) * ld]) * sx
                                : a[it2 + (j + c + 1) * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_gemv_kernel(const rocblas_int mm,
                                                       const rocblas_int k,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* WA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            A1 = A(0:c-1, :)
            A2 = W(0:c-1, cw+1:k-1)
            x = A(0:c-1, c)
            dac = temporary buffer

        operation:
                  [   A1(:, 0:c-1)   ]
                  [        0         ]
            dac = [ A1(:, c+1:mm-1)' ] * x
                  [        A2'       ]

        Notes:
            1. Here A1(:, 0:c-1) is full/general matrix (data below and above diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ 0  ]
                  [ y2 ] <- reduce(dac)
                  [ y3 ]
              where
                  [ y1 ]
                  [ 0  ] = W(:, cw)
                  [ y2 ]
                    y3   = work (temp buffer)
    ------------------------------ */
    int n = c;
    int cc = mm - c - 1;
    int m = mm + cc;
    int cw = c - mm + k;
    T* A1 = A;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T const* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < mm) ? i : i - mm;
        a = (i < mm) ? A1 : A2;
        ld = (i < mm) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (i > c) ? conj(a[j + it * ld]) * sx : a[i + j * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <int NB_X, typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_gemvt_kernel(const rocblas_int mm,
                                                        const rocblas_int k,
                                                        const rocblas_int c,
                                                        U AA,
                                                        const rocblas_int shiftA,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        T* WA,
                                                        const rocblas_int shiftW,
                                                        const rocblas_int ldw,
                                                        const rocblas_stride strideW,
                                                        T* yA,
                                                        const rocblas_int shiftY,
                                                        const rocblas_int ldy,
                                                        const rocblas_stride strideY,
                                                        T* workA,
                                                        const rocblas_stride strideblk)
{
    rocblas_int bid = blockIdx.z;
    rocblas_int tx = threadIdx.x;
    rocblas_int i = blockIdx.x;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* y1 = load_ptr_batch<T>(yA, bid, shiftY, strideY);
    T* y2 = workA + bid * strideblk;

    int n = c;
    int cc = mm - c - 1;
    // int m = mm + cc;
    int cw = c - mm + k;
    T* A1 = A;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    int it = (i < mm) ? i : i - mm;
    T* a = (i < mm) ? A1 : A2;
    int ld = (i < mm) ? lda1 : lda2;
    T* y = (i < mm) ? y1 : y2;

    if(tx < n)
        a += tx;

    a += it * size_t(ld);

    T res = 0;

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int n_full = (n / NB_X) * NB_X;

    if(i != c)
    {
        for(rocblas_int j = 0; j < n_full; j += NB_X)
            res += conj(a[j]) * x[tx + j];

        if(tx + n_full < n)
            res += conj(a[n_full]) * x[tx + n_full];

        // reduction of partial sums
        res += shift_left(res, 1);
        res += shift_left(res, 2);
        res += shift_left(res, 4);
        res += shift_left(res, 8);
        res += shift_left(res, 16);
        if(warpSize > 32)
            res += shift_left(res, 32);
        if(tx % warpSize == 0)
            sdata[tx / warpSize] = res;
        __syncthreads();
        if(tx == 0)
        {
            for(rocblas_int k = 1; k < NB_X / warpSize; k++)
                res += sdata[k];
        }
    }

    if(tx == 0)
    {
        y[it] = res;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_gemv_kernel(const rocblas_int mm,
                                                       const rocblas_int c,
                                                       U AA,
                                                       const rocblas_int shiftA,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,
                                                       T* WA,
                                                       const rocblas_int shiftW,
                                                       const rocblas_int ldw,
                                                       const rocblas_stride strideW,
                                                       T* dacA,
                                                       const rocblas_int ldd,
                                                       const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            A1 = W(c+1:mm-1, 0:c-1)
            A2 = A(c+1:mm-1, :)
            x = A(c+1,mm-1, c)
            dac = temporary buffer

        operation:
                  [       A1'       ]
            dac = [  A2(:, 0:c-1)'  ] * x
                  [        0        ]
                  [ A2(:, c+1:mm-1) ]

        Notes:
            1. Here A2(:, c+1:mm-1) is full/general matrix (data below and above diagonal)
            2. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
                  [ 0  ]
                  [ y3 ]
              where
                    y1   = work (temp buffer)
                  [ y2 ]
                  [ 0  ] = W(:, c)
                  [ y3 ]
    ------------------------------ */
    int n = mm - c - 1;
    int m = mm + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it, it2;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T const* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < c) ? i : i - c;
        a = (i < c) ? A1 : A2;
        ld = (i < c) ? lda1 : lda2;
        it2 = it - c - 1;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n && it != c)
                ac += (it < c) ? conj(a[j + it * ld]) * sx : a[it2 + (j + c + 1) * ld] * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <int NB_X, typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_gemvt_kernel(const rocblas_int mm,
                                                        const rocblas_int c,
                                                        U AA,
                                                        const rocblas_int shiftA,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        T* WA,
                                                        const rocblas_int shiftW,
                                                        const rocblas_int ldw,
                                                        const rocblas_stride strideW,
                                                        T* yA,
                                                        const rocblas_int shiftY,
                                                        const rocblas_int ldy,
                                                        const rocblas_stride strideY,
                                                        T* workA,
                                                        const rocblas_stride strideblk)
{
    rocblas_int bid = blockIdx.z;
    rocblas_int tx = threadIdx.x;
    rocblas_int i = blockIdx.x;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* y1 = workA + bid * strideblk;
    T* y2 = load_ptr_batch<T>(yA, bid, shiftY, strideY);

    int n = mm - c - 1;
    // int m = mm + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    int it = (i < c) ? i : i - c;
    T* a = (i < c) ? A1 : A2;
    int ld = (i < c) ? lda1 : lda2;
    int it2 = it - c - 1;
    T* y = (i < c) ? y1 : y2;

    if(tx < n)
        a += tx;

    a += it * size_t(ld);

    T res = 0;

    __shared__ T sdata[NB_X];

    // partial sums
    rocblas_int n_full = (n / NB_X) * NB_X;

    if(it != c)
    {
        for(rocblas_int j = 0; j < n_full; j += NB_X)
            res += conj(a[j]) * x[tx + j];

        if(tx + n_full < n)
            res += conj(a[n_full]) * x[tx + n_full];

        // reduction of partial sums
        res += shift_left(res, 1);
        res += shift_left(res, 2);
        res += shift_left(res, 4);
        res += shift_left(res, 8);
        res += shift_left(res, 16);
        if(warpSize > 32)
            res += shift_left(res, 32);
        if(tx % warpSize == 0)
            sdata[tx / warpSize] = res;
        __syncthreads();
        if(tx == 0)
        {
            for(rocblas_int k = 1; k < NB_X / warpSize; k++)
                res += sdata[k];
        }
    }

    if(tx == 0)
    {
        y[it] = res;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_computeW_kernel(const rocblas_int mm,
                                                  const rocblas_int k,
                                                  const rocblas_int c,
                                                  U AA,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  T* WA,
                                                  const rocblas_int shiftW,
                                                  const rocblas_int ldw,
                                                  const rocblas_stride strideW,
                                                  T* dacA,
                                                  const rocblas_int ldd,
                                                  const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            A1 = A(0:c-1, c+1:mm-1)
            A2 = W(0:c-1, cw+1:k-1)
            x = A(0:c-1, c)
            dac = temporary buffer

        operation:
                  [ A1' ]
            dac = [ A2' ] * x

        Notes:
            1. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
              where
                  y1 = W(c+1:mm-1, cw)
                  y2 = work (temp buffer)
    ------------------------------ */
    int n = c;
    int cc = mm - c - 1;
    int m = cc + cc;
    int cw = c - mm + k;
    T* A1 = A + idx2D(0, c + 1, lda);
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda1 = lda;
    int lda2 = ldw;
    T* x = A + idx2D(0, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < cc) ? i : i - cc;
        a = (i < cc) ? A1 : A2;
        ld = (i < cc) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac += conj(a[j + it * ld]) * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_computeW_kernel(const rocblas_int mm,
                                                  const rocblas_int c,
                                                  U AA,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  T* WA,
                                                  const rocblas_int shiftW,
                                                  const rocblas_int ldw,
                                                  const rocblas_stride strideW,
                                                  T* dacA,
                                                  const rocblas_int ldd,
                                                  const rocblas_stride strideD)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* dac = dacA + bid * strideD;

    /* -----------------------------
    formulate gemv problem:

        components:
            A1 = W(c+1:mm-1, 0:c-1)
            A2 = A(c+1:mm-1, 0:c-1)
            x = A(c+1,mm-1, c)
            dac = temporary buffer

        operation:
                  [ A1' ]
            dac = [ A2' ] * x

        Notes:
            1. dac is further reduced by reduce_kernel; results stored in
                  [ y1 ]
                  [ y2 ] <- reduce(dac)
              where
                  y1 = work (temp buffer)
                  y2 = W(0:c-1, c)
    ------------------------------ */
    int n = mm - c - 1;
    int m = c + c;
    T* A1 = W + idx2D(c + 1, 0, ldw);
    T* A2 = A + idx2D(c + 1, 0, lda);
    int lda1 = ldw;
    int lda2 = lda;
    T* x = A + idx2D(c + 1, c, lda);

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j, it;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac, sx;
    T* a;
    int ld;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;
        ac = 0;

        it = (i < c) ? i : i - c;
        a = (i < c) ? A1 : A2;
        ld = (i < c) ? lda1 : lda2;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx = (j < n) ? x[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac += conj(a[j + it * ld]) * sx;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            dac[i + bidc * ldd] = ac;
    }
}

/***** Kernels to update column of W *****/
/*****************************************/
template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_upper_updateW_kernel(const rocblas_int mm,
                                                 const rocblas_int k,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* WA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 T* workA,
                                                 const rocblas_stride strideblk,
                                                 T* tauA,
                                                 const rocblas_stride strideP)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* work = workA + bid * strideblk;
    T* tau = tauA + bid * strideP;

    /* ------------------------
    formulate gemv problem:

        components:
            cw = c - mm + k
            y = W(0:c-1, cw)
            A1 = A(0:c-1, c+1:mm-1)
            A2 = W(0:c-1, cw+1:k-1)
            x1 = work (temp buffer)
            x2 = W(c+1:mm-1, cw)
            t = tau(c-1)

        operation:
            y = t * (y - A1 * x1 - A2 * x2)
    ------------------------ */
    int n = mm - c - 1;
    int m = c;
    int cw = c - mm + k;
    T* y = W + idx2D(0, cw, ldw);
    T* A1 = A + idx2D(0, c + 1, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(0, cw + 1, ldw);
    int lda2 = ldw;
    T* x1 = work;
    T* x2 = W + idx2D(c + 1, cw, ldw);
    T* t = tau + c - 1;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? x1[j] : 0;
            sx2 = (j < n) ? x2[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            y[i] = ac * t[0];
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void latrd_lower_updateW_kernel(const rocblas_int mm,
                                                 const rocblas_int c,
                                                 U AA,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 T* WA,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 T* workA,
                                                 const rocblas_stride strideblk,
                                                 T* tauA,
                                                 const rocblas_stride strideP)
{
    int bid = hipBlockIdx_z;
    int bidr = hipBlockIdx_x;
    int bidc = hipBlockIdx_y;
    int tidr = hipThreadIdx_x;
    int tidc = hipThreadIdx_y;
    int threadsr = hipBlockDim_x;
    int threadsc = hipBlockDim_y;
    int groupsr = hipGridDim_x;
    int groupsc = hipGridDim_y;
    int totalthsr = groupsr * threadsr;
    int totalthsc = groupsc * threadsc;
    int idc = bidc * threadsc + tidc;
    int idr = bidr * threadsr + tidr;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WA, bid, shiftW, strideW);
    T* work = workA + bid * strideblk;
    T* tau = tauA + bid * strideP;

    /* ------------------------
    formulate gemv problem:

        components:
            y = W(c+1:mm-1, c)
            A1 = A(c+1:mm-1, 0:c-1)
            A2 = W(c+1:mm-1, 0:c-1)
            x1 = work (temp buffer)
            x2 = W(0:c-1, c)
            t = tau(c)

        operation:
            y = t * (y - A1 * x1 - A2 * x2)
    ------------------------ */
    int m = mm - c - 1;
    int n = c;
    T* y = W + idx2D(c + 1, c, ldw);
    T* A1 = A + idx2D(c + 1, 0, lda);
    int lda1 = lda;
    T* A2 = W + idx2D(c + 1, 0, ldw);
    int lda2 = ldw;
    T* x1 = work;
    T* x2 = W + idx2D(0, c, ldw);
    T* t = tau + c;

    // rpgr and rpgc are the number of rounds a group should run
    // to cover all the rows and columns, respectively
    int ngrp = (m - 1) / threadsr + 1;
    int rpgr = (ngrp - 1) / groupsr + 1;
    ngrp = (n - 1) / threadsc + 1;
    int rpgc = (ngrp - 1) / groupsc + 1;
    int i, j;

    // Registers/LDS:
    // ac, acs -> accumulator
    // sx -> hold the elements of 'x'
    extern __shared__ double smem[]; //min size should be threadsr x threadsc
    T* acs = reinterpret_cast<T*>(smem);
    T ac;
    T sx1, sx2;

    for(int ii = 0; ii < rpgr; ++ii)
    {
        i = ii * totalthsr + idr;

        // read y
        ac = (idc == 0 && i < m) ? y[i] : 0;

        for(int jj = 0; jj < rpgc; ++jj)
        {
            // read x
            j = jj * totalthsc + idc;
            sx1 = (j < n) ? x1[j] : 0;
            sx2 = (j < n) ? x2[j] : 0;

            // operation for all rows
            if(i < m && j < n)
                ac -= A1[i + j * lda1] * sx1 + A2[i + j * lda2] * sx2;
        }
        acs[tidr + tidc * threadsr] = ac;
        __syncthreads();

        // group reduction
        for(int r = threadsc / 2; r > 0; r /= 2)
        {
            if(tidc < r)
            {
                ac += acs[tidr + (tidc + r) * threadsr];
                acs[tidr + tidc * threadsr] = ac;
            }
            __syncthreads();
        }

        // write groups results in temp array for further reduction
        if(tidc == 0 && i < m)
            y[i] = ac * t[0];
    }
}

/******************* Host functions for latrd aux of sytrd **********************/
/********************************************************************************/

// Method to determine configuration for update kernels depending on n and k
// TODO: fine tuning may be required
template <typename T>
void latrd_get_config_for_updates(const rocblas_int n,
                                  const rocblas_int k,
                                  rocblas_int* dr,
                                  rocblas_int* thr,
                                  rocblas_int* dc,
                                  rocblas_int* thc)
{
    if(n <= 256)
    {
        *thr = 8;
        *thc = 16;
    }
    else if(n <= 3584)
    {
        *thr = 16;
        *thc = 16;
    }
    else if(n <= 7168)
    {
        *thr = 32;
        *thc = 8;
    }
    else
    {
        *thr = 64;
        *thc = 8;
    }

    *dr = 4;
    *dc = 0;
}

template <bool BATCHED, typename T>
void rocsolver_latrd_forsytrd_getMemorySize(const rocblas_int n,
                                            const rocblas_int k,
                                            const rocblas_int batch_count,
                                            size_t* size_scalars,
                                            size_t* size_work,
                                            size_t* size_norms,
                                            size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || k == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0, n3 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0, w4 = 0;

    // size of scalars (constants) for rocblas calls
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers (batched cases)
    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements for calling larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // extra requirements for calling dotp
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    // arrays for temporary values
    // TODO: smaller quotes could be considered if we know the latrd_mode and
    // the configuration of the computeW kernels in advance. For now, taking
    // worst case.
    w4 = sizeof(T) * k * batch_count;
    rocblas_int gr = (n - 1) / 4 + 1;
    n3 = sizeof(T) * (n + k) * gr * batch_count;

    *size_norms = std::max({n1, n2, n3});
    *size_work = std::max({w1, w2, w3, w4});
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_forsytrd_template(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int k,
                                                 U A,
                                                 const rocblas_int shiftA,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 S* E,
                                                 const rocblas_stride strideE,
                                                 T* tau,
                                                 const rocblas_stride strideP,
                                                 T* W,
                                                 const rocblas_int shiftW,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 const rocblas_int batch_count,
                                                 T* scalars,
                                                 T* work,
                                                 T* norms,
                                                 T** workArr)
{
    ROCSOLVER_ENTER("latrd_forsytrd", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA,
                    "lda:", lda, "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);
    roctxRangePush("rocsolver_latrd_forsytrd");

    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // configure updateA and updateW kernels:
    rocblas_int dr, dc;
    rocblas_int thr_updates, thc_updates;
    latrd_get_config_for_updates<T>(n, k, &dr, &thr_updates, &dc, &thc_updates);
    size_t lmemsize_updates = sizeof(T) * (thr_updates * thc_updates);
    rocblas_int grr_updates = (n * dr / 4 - 1) / thr_updates + 1;
    rocblas_int grc_updates = (k * dc / 4 - 1) / thc_updates + 1;

    rocblas_stride strideblk = k;

    if(uplo == rocblas_fill_lower)
    {
        // reduce the first k columns of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < k; ++j)
        {
            // update column j of A with reflector computed in step j-1
            //----------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(latrd_lower_updateA_kernel<T>,
                                    dim3(grr_updates, grc_updates, batch_count),
                                    dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n,
                                    j, A, shiftA, lda, strideA, W, shiftW, ldw, strideW);
            //-------------------------------------------------------------

            // reduce column j of A with new reflector, then copy off-diagonal element
            // to E(j) and set off-diagonal to 1
            //----------------------------------------------------------
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                     strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda), 1,
                                     strideA, (tau + j), strideP, batch_count, work, norms);
            //-----------------------------------------------------------

            // compute column j of W
            //--------------------------------------------------------------
            static constexpr int NB = 256;
            dim3 gemvt_grid(n + j, 1, batch_count);
            dim3 gemvt_threads(NB);
            ROCSOLVER_LAUNCH_KERNEL((latrd_lower_computeW_gemvt_kernel<NB, T>), gemvt_grid,
                                    gemvt_threads, 0, stream, n, j, A, shiftA, lda, strideA, W,
                                    shiftW, ldw, strideW, W, shiftW + idx2D(0, j, ldw), ldw,
                                    strideW, work, strideblk);

            // update column j of W
            //--------------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(
                latrd_lower_updateW_kernel<T>, dim3(grr_updates, grc_updates, batch_count),
                dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n, j, A, shiftA, lda,
                strideA, W, shiftW, ldw, strideW, work, strideblk, tau, strideP);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, n - 1 - j, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, W,
                                    shiftW + idx2D(j + 1, j, ldw), strideW, tau + j, strideP);
            //--------------------------------------------------------------
        }
    }

    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;
        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;

            // update column j of A with reflector computed in step j-1
            //----------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(latrd_upper_updateA_kernel<T>,
                                    dim3(grr_updates, grc_updates, batch_count),
                                    dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n,
                                    k, j, A, shiftA, lda, strideA, W, shiftW, ldw, strideW);
            //-------------------------------------------------------------

            // reduce column j of A with new reflector, then copy off-diagonal element
            // to E(j) and set off-diagonal to 1
            //----------------------------------------------------------
            rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1, strideE,
                                     A, shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1),
                                     strideP, batch_count, work, norms);
            //----------------------------------------------------------

            // compute column j of W
            //--------------------------------------------------------------
            static constexpr int NB = 256;
            dim3 gemvt_grid(n + n - j - 1, 1, batch_count);
            dim3 gemvt_threads(NB);
            ROCSOLVER_LAUNCH_KERNEL((latrd_upper_computeW_gemvt_kernel<NB, T>), gemvt_grid,
                                    gemvt_threads, 0, stream, n, k, j, A, shiftA, lda, strideA, W,
                                    shiftW, ldw, strideW, W, shiftW + idx2D(0, jw, ldw), ldw,
                                    strideW, work, strideblk);

            // update column j of W
            //--------------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(
                latrd_upper_updateW_kernel<T>, dim3(grr_updates, grc_updates, batch_count),
                dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n, k, j, A, shiftA,
                lda, strideA, W, shiftW, ldw, strideW, work, strideblk, tau, strideP);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, j, A, shiftA + idx2D(0, j, lda),
                                    strideA, W, shiftW + idx2D(0, jw, ldw), strideW, tau + j - 1,
                                    strideP);
        }
    }

    roctxRangePop();
    return rocblas_status_success;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
auto rocsolver_latrd_forsytrd_getWorkItems(rocblas_handle handle,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int k,
                                           U A,
                                           const rocblas_int shiftA,
                                           const rocblas_int lda,
                                           const rocblas_stride strideA,
                                           S* E,
                                           const rocblas_stride strideE,
                                           T* tau,
                                           const rocblas_stride strideP,
                                           T* W,
                                           const rocblas_int shiftW,
                                           const rocblas_int ldw,
                                           const rocblas_stride strideW,
                                           const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements for calling LARFG
    size_t size_work, size_norms;
    rocsolver_latrd_forsytrd_getMemorySize<false, T>(n, k, batch_count, &size_scalars, &size_work,
                                                     &size_norms, &size_workArr);

    std::size_t size_A = n * lda;
    std::size_t size_W = k * ldw;
    std::size_t buffer = 3 * n * n;
    size_work = std::max(size_work, size_A + size_W + buffer);
    auto work_items = create_work_item({"latrd_scalars", size_scalars})
        + create_work_item({"latrd_workArr", size_workArr})
        + create_work_item({"latrd_work", size_work}) + create_work_item({"latrd_norms", size_norms});

    return work_items;
}

template <std::int32_t WDIM = 0, typename S>
__device__ inline void reduce_wave_sum(S& val)
{
    /* assert(WDIM == warpSize); */

#pragma unroll
    for(rocblas_int r = warpSize / 2; r >= 1; r /= 2)
    {
        val += shift_left(val, r);
    }

    /* val = __shfl(val, 0); */
}

template <std::int32_t BDIM = 0, typename S>
__device__ inline void reduce_block_sum(S& val, S* smem)
{
    /* assert(BDIM == blockDim.x); */

    /* if(blockDim.x > warpSize) */
    if constexpr(true)
    {
        /* __shared__ S smem[BDIM]; */
        rocblas_int tid = threadIdx.x;

        smem[tid] = val;
        __syncthreads();

        /* #pragma unroll */
        for(rocblas_int r = blockDim.x / 2; r >= warpSize; r /= 2)
        {
            if(tid < r)
            {
                smem[tid] += smem[tid + r];
            }
            __syncthreads();
        }

        val = smem[tid];
        __syncthreads();

        /* #pragma unroll */
        /* for(rocblas_int r = warpSize / 2; r >= 1; r /= 2) */
        /* { */
        /*     val += shift_left(val, r); */
        /* } */
        reduce_wave_sum(val);

        if(threadIdx.x == 0)
        {
            smem[0] = val;
        }
        __syncthreads();

        /* val = smem[0]; */
        /* __syncthreads(); */
    }
    else
    {
        reduce_wave_sum(val);
    }
}

template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    latrd_lower_kernel_small(const I n,
                             const rocblas_int nb,
                             U AA,
                             const rocblas_stride shiftA,
                             const I lda,
                             const rocblas_stride strideA,
                             S* EE,
                             const rocblas_stride strideE,
                             T* tauA,
                             const rocblas_stride strideP,
                             T* WW,
                             const rocblas_int shiftW,
                             const rocblas_int ldw,
                             const rocblas_stride strideW)
{
    constexpr bool is_complex_t = rocblas_is_complex<T>;

    I batch_id = blockIdx.z;
    I bid = blockIdx.x;
    I tid = threadIdx.x;

    // Select batch instance
    T* A = load_ptr_batch<T>(AA, batch_id, shiftA, strideA);
    S* E = load_ptr_batch<S>(EE, batch_id, 0, strideE);
    T* tau = load_ptr_batch<T>(tauA, batch_id, 0, strideP);
    T* W = load_ptr_batch<T>(WW, batch_id, 0, strideW);
    T* Atmp = nullptr;
    T* Wtmp = nullptr;

    // Shared variables
    extern __shared__ double lmem[];
    T* tau_j = reinterpret_cast<T*>(lmem);
    T* As = reinterpret_cast<T*>(tau_j + 1);
    T* Ws = reinterpret_cast<T*>(As + n * n);
    T* v = reinterpret_cast<T*>(Ws + n * nb);
    T* w = reinterpret_cast<T*>(v + n); // this piece of LDS is left unused for the time being
    T* smem = reinterpret_cast<T*>(w + n);

    // Load A into LDS
    for(I ii = tid % (MAX_THDS / 2); ii < n; ii += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I jj = tidy; jj < n; jj += 2)
        {
            As[ii + jj * n] = A[ii + jj * lda];
        }
    }
    __syncthreads();

    // Remove later if not necessary
    for(I ii = tid % (MAX_THDS / 2); ii < n; ii += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I jj = tidy; jj < n; jj += 2)
        {
            // Ignore imaginary part of the diagonal
            if(ii == jj)
            {
                As[ii + jj * n] = std::real(As[ii + jj * n]);
            }
            // Copy lower triangular part to upper triangle
            if(ii < jj)
            {
                As[ii + jj * n] = conj(As[jj + ii * n]);
            }
        }
    }

    // Zero W
    for(I ii = tid % (MAX_THDS / 2); ii < n; ii += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I jj = tidy; jj < nb; jj += 2)
        {
            Ws[ii + jj * n] = T(0);
        }
    }
    __syncthreads();

    // Reduce the lower part of A: main loop running forwards (for each column)
    I nj{};
    T temp{};
    for(rocblas_int j = 0; j < nb; ++j)
    {
        nj = n - j - 1;
        w = Ws + j * n;

        //
        // Update A(j:n-1, j) with previously computed reflectors and Ws.
        // (Notice that the triangle below the diagonal of A(:, 0:j-1) holds
        // previously computed Householder reflectors.)
        //
        if(j > 0)
        {
            // Step 1: A(j:n-1, j) = -A(j:n-1, 0:j-1) * W(j, 0:1-j)^H + A(j:n-1, j)
            //
            Atmp = As + j + j * n;
            for(I ii = tid; ii < nj + 1; ii += MAX_THDS)
            {
                temp = T(0);
                for(I jj = 0; jj < j; jj++)
                {
                    temp += As[j + ii + jj * n] * Ws[j + jj * n];
                }
                Atmp[ii] -= temp;
            }
            __syncthreads();

            // Step 2: A(j:n-1, j) = -W(j:n-1, 0:j-1) * A(j, 0:j-1)^H + A(j:n-1, j)
            //
            Atmp = As + j + j * n;
            for(I ii = tid; ii < nj + 1; ii += MAX_THDS)
            {
                temp = T(0);
                for(I jj = 0; jj < j; jj++)
                {
                    temp += Ws[j + ii + jj * n] * As[j + jj * n];
                }
                Atmp[ii] -= temp;
            }
            __syncthreads();

            // grid.sync()
            //
            // Note: since
            //
            //     z1 = A(j:n-1, 0:j-1) * W(j, 0:1-j)^H (computed in Step 1), and
            //     z2 = W(j:n-1, 0:j-1) * A(j, 0:j-1)^H (computed in Step 2)
            //
            // are independent, these two GEMVs above can be fused to compute:
            //
            //     A(j:n-1, j) -= z1 + z2
            //
            // in a single pass.
            //
            // Work has to be synchronized here because A(j:n-1, j) is used to compute a
            // Householder reflector in Step 3.
        }

        //
        // Step 3: Generate Householder reflector to annihilate A(j+2:n-1,j)
        // and copy off-diagonal element to E[j]
        //

        // Load A(j+1:n-1,j) into v
        v = As + (j + 1) + j * n;

        // LARFG
        temp = T(0);
        for(I i = tid; i < nj - 1; i += MAX_THDS)
        {
            temp += v[i + 1] * conj(v[i + 1]);
        }
        reduce_block_sum(temp, smem);

        if(tid == 0)
        {
            // set tau, beta, and put scaling factor into smem[0]
            run_set_taubeta<T>(tau_j, &temp, v, E + j);

            tau[j] = tau_j[0];
            smem[0] = temp;
        }
        __syncthreads();

        // Scale v
        T scal = smem[0];
        for(I i = tid; i < nj - 1; i += MAX_THDS)
        {
            v[i + 1] *= scal;
        }
        __syncthreads();

        // grid.sync()
        //
        // Note: both v and tau_j are required for the next steps.

        // Copy v back to A(j+1:n-1,j)
        // This data will only be used on the next iteration,
        // provided that j < nb - 1.
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            As[(ii + j + 1) + j * n] = v[ii];
        }

        //
        // Compute w = tau_j*A*v - 1/2*tau_j^2*(v'*A*v)*v
        //

        // SYMV
        //
        // Step 4: w_0 = A(j+1:n-1, j+1:n-1) * v(0:n-1-j)
        //
        Atmp = As + (j + 1) + (j + 1) * n;
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            temp = T(0);
            for(I jj = 0; jj < nj; jj++)
            {
                temp += Atmp[ii + jj * n] * v[jj];
            }
            w[ii + j + 1] = temp;
        }
        __syncthreads();

        // Step 5: w(0:j-1) = W(j+1:n-1, 0:j-1)^H * v(0:n-1-j)
        //
        Wtmp = Ws + (j + 1);
        for(I jj = tid; jj < j; jj += MAX_THDS)
        {
            temp = T(0);
            for(I ii = 0; ii < nj; ++ii)
            {
                temp += conj(Wtmp[ii + jj * n]) * v[ii];
            }
            /* reduce_block_sum(temp, smem); */
            /* Ws[jj + j * n] = smem[0]; */
            w[jj] = temp;
        }
        __syncthreads();

        // Step 6: w(j+1:n-1) = -A(j+1:n-1, 0:j-1) * w(0:j-1) + w(j+1:n-1)
        //
        Atmp = As + (j + 1);
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            temp = T(0);
            for(I jj = 0; jj < j; ++jj)
            {
                temp -= Atmp[ii + jj * n] * w[jj];
            }
            w[j + 1 + ii] += temp;
        }
        __syncthreads();

        // grid.sync()
        //
        // Note: notice that Steps 4, 5 and 7 are functionally independent
        // and can be computed without synchronization.

        // Step 7: w(0:j-1) = A(j+1:n-1, 0:j-1)^H * v(0:n - 1 -j);
        //
        Atmp = As + (j + 1);
        for(I jj = tid; jj < j; jj += MAX_THDS)
        {
            temp = T(0);
            for(I ii = 0; ii < nj; ++ii)
            {
                temp += conj(Atmp[ii + jj * n]) * v[ii];
            }
            /* reduce_block_sum(temp, smem); */
            /* Ws[jj + j * n] = smem[0]; */
            w[jj] = temp;
        }
        __syncthreads();

        // Step 8: w(j+1:n-1) = -W(j+1:n, 0:j-1) * w(0:j-1) + W(j+1:n-1)
        //
        Wtmp = Ws + (j + 1);
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            temp = T(0);
            for(I jj = 0; jj < j; ++jj)
            {
                temp -= Wtmp[ii + jj * n] * w[jj];
            }
            w[j + 1 + ii] += temp;
        }
        __syncthreads();

        // grid.sync()
        //
        // Note: Steps 6 and 8 can be fused.

        // Step 9: w(j+1:n-1) = alpha * v(0:n-j-1) + tauj * w(j+1:n-1)
        //
        // alpha = -0.5 * tauj^2 * <v, w>
        //
        // Dot product <v, w>
        temp = 0;
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            temp += v[ii] * conj(w[ii + j + 1]);
        }
        reduce_block_sum(temp, smem);

        if(tid == 0)
        {
            // alpha = - 1/2 * tauj^2 * <v, w>
            smem[0] = -0.5 * tau_j[0] * tau_j[0] * temp;
        }
        __syncthreads();

        // AXPY
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            w[ii + j + 1] = smem[0] * v[ii] + tau_j[0] * w[ii + j + 1];
        }
        __syncthreads();

        // grid.sync()
        //
        // Note: the result of the AXPY is required for Steps 1 and 2.
    }

    // Write LDS back to A
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            if(i >= j)
            {
                A[i + j * lda] = As[i + j * n];
            }
        }
    }

    // Write LDS back to W
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < nb; j += 2)
        {
            W[i + j * ldw] = Ws[i + j * n];
        }
    }
}

template <int MAX_THDS, typename T, typename I, typename S, typename U>
__global__ void __launch_bounds__(MAX_THDS) latrd_lower_kernel_naive(const I n,
                                                                     const rocblas_int nb,
                                                                     U AA,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     S* EE,
                                                                     const rocblas_stride strideE,
                                                                     T* tauA,
                                                                     const rocblas_stride strideP,
                                                                     T* WW,
                                                                     const rocblas_int shiftW,
                                                                     const rocblas_int ldw,
                                                                     const rocblas_stride strideW,
                                                                     T* work)
{
    constexpr bool is_complex_t = rocblas_is_complex<T>;
    auto grid = cooperative_groups::this_grid();

    I batch_id = blockIdx.z;
    I bid = blockIdx.x;
    I tid = threadIdx.x;

    // Select batch instance
    T* A = load_ptr_batch<T>(AA, batch_id, shiftA, strideA);
    S* E = load_ptr_batch<S>(EE, batch_id, 0, strideE);
    T* tau = load_ptr_batch<T>(tauA, batch_id, 0, strideP);
    T* W = load_ptr_batch<T>(WW, batch_id, 0, strideW);
    T* pw = nullptr;
    T* pv = nullptr;
    T* Atmp = nullptr;
    T* pSAtmp = nullptr;
    T* Wtmp = nullptr;
    T* pSWtmp = nullptr;

    // Shared variables
    extern __shared__ double lmem[];
    T* tau_j = reinterpret_cast<T*>(lmem);
    /* T* As = reinterpret_cast<T*>(tau_j + 1); */
    T* pSA = A;
    /* T* Ws = reinterpret_cast<T*>(pSA + n * n); */
    T* pSW = W;
    /* T* v = reinterpret_cast<T*>(pSW + n * nb); */
    T* v = reinterpret_cast<T*>(lmem + 1);
    T* pSv = v;
    T* w = reinterpret_cast<T*>(v + n);
    T* pSw = w;
    T* pSz1 = reinterpret_cast<T*>(w + n);
    T* pSz2 = reinterpret_cast<T*>(pSz1 + nb);
    T* pSmem = reinterpret_cast<T*>(pSz2 + nb);

    // Workspace
    // work is aligned at 64 bytes
    T* pA = A;
    T* pW = W;
    T* pz1 = work;
    T* pz2 = work + n;

    T tauj;
    T alpha;
    I ldSA = lda;
    I ldSW = ldw;

    // Reduce the lower part of A: main loop running forwards (for each column)
    I nj{};
    T temp{};
    for(rocblas_int j = 0; j < nb; ++j)
    {
        nj = n - j - 1;
        /* pv = A + j * lda; */
        /* pw = W + j * ldw; */

        // Zero W
        for(I ii = tid; ii < n; ii += MAX_THDS)
        {
            w[ii] = T(0);
        }

        // Part A:
        //
        // Update A(j:n-1, j) with previously computed reflectors and W.
        // (Notice that the triangle below the diagonal of A holds
        // previously computed Householder reflectors.)
        //
        if(j > 0)
        {
            // Step 1: A(j:n-1, j) = -A(j:n-1, 0:j-1) * W(j, 0:1-j)^H + A(j:n-1, j)
            // Step 2: A(j:n-1, j) = -W(j:n-1, 0:j-1) * A(j, 0:j-1)^H + A(j:n-1, j)
            for(I ii = tid; ii < nj + 1; ii += MAX_THDS)
            {
                temp = T(0);
                for(I jj = 0; jj < j; jj++)
                {
                    temp += pA[j + ii + jj * ldSA] * pW[j + jj * ldSW];
                    temp += pW[j + ii + jj * ldSW] * pA[j + jj * ldSA];
                }
                pA[ii + j + j * ldSA] -= temp;
            }
            grid.sync();
            // Work has to be synchronized here because A(j:n-1, j) is used to compute a
            // Householder reflector in Step 3.
        }

        // Part B:
        //
        // Step 3: Generate Householder reflector to annihilate A(j+2:n-1,j)
        // and copy off-diagonal element to E[j]
        //
        // Note: both v and tau_j are required for the next steps.
        if(bid == 0)
        {
            // Load A(j+1:n-1,j) into v
            for(I ii = tid; ii < nj; ii += MAX_THDS)
            {
                v[ii] = pA[ii + (j + 1) + j * ldSA];
            }
            __syncthreads();

            // LARFG
            temp = T(0);
            for(I ii = tid; ii < nj - 1; ii += MAX_THDS)
            {
                temp += v[ii + 1] * conj(v[ii + 1]);
            }
            reduce_block_sum(temp, pSmem);

            if(tid == 0)
            {
                // set tau, beta, and put scaling factor into pSmem[0]
                run_set_taubeta<T>(tau_j, &temp, v, E + j);

                tau[j] = tau_j[0];
                pSmem[0] = temp;
            }
            __syncthreads();

            // Scale v
            T scal = pSmem[0];
            for(I ii = tid; ii < nj - 1; ii += MAX_THDS)
            {
                v[ii + 1] *= scal;
            }

            // Copy v back to A(j+1:n-1,j)
            for(I ii = tid; ii < nj; ii += MAX_THDS)
            {
                pA[(ii + j + 1) + j * ldSA] = v[ii];
            }
        }
        grid.sync();

        //
        // Compute w = tau_j*A*v - 1/2*tau_j^2*(v'*A*v)*v
        //

        // Part C:
        //
        // SYMV
        //
        // Step 4: w_0 = A(j+1:n-1, j+1:n-1) * v(0:n-1-j)
        //
        Atmp = pA + (j + 1) + (j + 1) * ldSA;
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            temp = T(0);
            for(I jj = 0; jj < nj; jj++)
            {
                temp += Atmp[ii + jj * ldSA] * v[jj];
            }
            w[ii + j + 1] = temp;
        }

        // Step 5: z1(0:j-1) = W(j+1:n-1, 0:j-1)^H * v(0:n-1-j)
        //
        Wtmp = pW + (j + 1);
        for(I jj = tid; jj < j; jj += MAX_THDS)
        {
            temp = T(0);
            for(I ii = 0; ii < nj; ++ii)
            {
                temp += conj(Wtmp[ii + jj * ldSW]) * v[ii];
            }
            /* reduce_block_sum(temp, pSmem); */
            /* pSW[jj + j * ldSW] = pSmem[0]; */
            pSz1[jj] = temp;
        }

        // Step 7: z2(0:j-1) = A(j+1:n-1, 0:j-1)^H * v(0:n - 1 -j);
        //
        Atmp = pA + (j + 1);
        for(I jj = tid; jj < j; jj += MAX_THDS)
        {
            temp = T(0);
            for(I ii = 0; ii < nj; ++ii)
            {
                temp += conj(Atmp[ii + jj * ldSA]) * v[ii];
            }
            /* reduce_block_sum(temp, pSmem); */
            /* pSW[jj + j * ldSW] = pSmem[0]; */
            pSz2[jj] = temp;
        }
        grid.sync();

        // Part D:
        //
        // Step 6: w(j+1:n-1) = -A(j+1:n-1, 0:j-1) * z1(0:j-1) + w(j+1:n-1)
        // Step 8: w(j+1:n-1) = -W(j+1:n-1, 0:j-1) * z2(0:j-1) + w(j+1:n-1)
        //
        // Note: Steps 6 and 8 can be fused.
        Atmp = pA + (j + 1);
        Wtmp = pW + (j + 1);
        for(I ii = tid; ii < nj; ii += MAX_THDS)
        {
            temp = T(0);
            for(I jj = 0; jj < j; ++jj)
            {
                temp -= Atmp[ii + jj * ldSA] * pSz1[jj];
                temp -= Wtmp[ii + jj * ldSW] * pSz2[jj];
            }
            w[j + 1 + ii] += temp;
        }
        grid.sync();

        // Part E:
        //
        // Step 9: w(j+1:n-1) = alpha * v(0:n-j-1) + tauj * w(j+1:n-1)
        //
        // alpha = -0.5 * tauj^2 * <v, w>
        //
        // Dot product <v, w>
        if(bid == 0)
        {
            temp = 0;
            for(I ii = tid; ii < nj; ii += MAX_THDS)
            {
                temp += v[ii] * conj(w[ii + j + 1]);
            }
            reduce_block_sum(temp, pSmem);

            if(tid == 0)
            {
                // alpha = - 1/2 * tauj^2 * <v, w>
                pSmem[0] = -0.5 * tau_j[0] * tau_j[0] * temp;
            }
            __syncthreads();

            // AXPY
            for(I ii = tid; ii < nj; ii += MAX_THDS)
            {
                w[ii + j + 1] = pSmem[0] * v[ii] + tau_j[0] * w[ii + j + 1];
            }
            __syncthreads();

            for(I ii = tid; ii < n; ii += MAX_THDS)
            {
                pSW[ii + j * ldSW] = w[ii];
            }
        }
        grid.sync();

        // grid.sync()
        //
        // Note: the result of the AXPY is required for Steps 1 and 2.
    }
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_latrd_forsytrd_template(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 rocblas_int n,
                                                 rocblas_int k,
                                                 U A,
                                                 rocblas_int shiftA,
                                                 rocblas_int lda,
                                                 rocblas_stride strideA,
                                                 S* E,
                                                 rocblas_stride strideE,
                                                 T* tau,
                                                 rocblas_stride strideP,
                                                 T* W,
                                                 rocblas_int shiftW,
                                                 rocblas_int ldw,
                                                 rocblas_stride strideW,
                                                 rocblas_int batch_count,
                                                 rocsolver_device_workspace_ptr_t dwptr)
{
    ROCSOLVER_ENTER("latrd_forsytrd_alt", "uplo:", uplo, "n:", n, "k:", k, "shiftA:", shiftA,
                    "lda:", lda, "shiftW:", shiftW, "ldw:", ldw, "bc:", batch_count);
    ROCSOLVER_INIT_DEVICE_WORKSPACE(dwptr,
                                    rocsolver_latrd_forsytrd_getWorkItems(
                                        handle, uplo, n, k, A, shiftA, lda, strideA, E, strideE,
                                        tau, strideP, W, shiftW, ldw, strideW, batch_count));

    T* scalars = (T*)dwptr->work("latrd_scalars");
    T* work = (T*)dwptr->work("latrd_work");
    T* norms = (T*)dwptr->work("latrd_norms");
    T** workArr = (T**)dwptr->work("latrd_workArr");

    if(dwptr->size("latrd_scalars") > 0)
        init_scalars(handle, (T*)scalars);

    if(print_debug_messages_latrd_forsytrd)
    {
        std::cout << "Using latrd_forsytrd entry point." << std::endl;
    }

    roctxRangePush("rocsolver_latrd_forsytrd");

    // quick return
    if(n == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // configure updateA and updateW kernels:
    rocblas_int dr, dc;
    rocblas_int thr_updates, thc_updates;
    latrd_get_config_for_updates<T>(n, k, &dr, &thr_updates, &dc, &thc_updates);
    size_t lmemsize_updates = sizeof(T) * (thr_updates * thc_updates);
    rocblas_int grr_updates = (n * dr / 4 - 1) / thr_updates + 1;
    rocblas_int grc_updates = (k * dc / 4 - 1) / thc_updates + 1;

    rocblas_stride strideblk = k;

    if(uplo == rocblas_fill_lower)
    {
        if(print_debug_messages_latrd_forsytrd)
        {
            std::cout << "Using latrd's lower path" << std::endl;
        }

        bool select_coop_launch = true || force_coop_launch;

        // reduce the first k columns of A
        // main loop running forwards (for each column)
        const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
        const rocblas_int nn = n;
        std::size_t size_W = sizeof(T) * ldw * k * batch_count;

        const size_t lmemsize_small
            = ((256 / props->warpSize) + 2 * nn + 1 + nn * nn + nn * k) * sizeof(T);
        constexpr size_t small_switch_size = 128;
        bool use_small_kernel = !select_coop_launch && (n < small_switch_size)
            && (lmemsize_small <= props->sharedMemPerBlock);

        const size_t lmemsize_fused = ((256 / props->warpSize) + 1 + 5 * n + 5 * k) * sizeof(T);
        /* std::cout << "lmemsize = " << std::to_string(lmemsize_fused / 1024.0) << "KB" << std::endl; */
        bool use_fused_kernel = select_coop_launch
            && !rocblas_is_complex<T> && (lmemsize_fused <= props->sharedMemPerBlock);

        if(!latrd_forsytrd_multi_kernel && use_small_kernel)
        {
            if(print_debug_messages_latrd_forsytrd)
            {
                std::cout << "Using latrd's small kernel, lmemsize = "
                          << std::to_string(lmemsize_small / 1024.0) << "KB" << std::endl;
            }

            HIP_TRACE(hipMemsetAsync((void*)W, 0, size_W, stream));
            rocblas_int j = 0;
            ROCSOLVER_LAUNCH_KERNEL((latrd_lower_kernel_small<256, T>), dim3(1, 1, batch_count),
                                    dim3(256), lmemsize_small, stream, n, k, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, E + j, strideE,
                                    tau + j, strideP, W, shiftW, ldw, strideW);
        }
        else if(!latrd_forsytrd_multi_kernel && use_fused_kernel)
        {
            if(print_debug_messages_latrd_forsytrd)
            {
                std::cout << "Using latrd's fused kernel, lmemsize = "
                          << std::to_string(lmemsize_fused / 1024.0) << "KB" << std::endl;
            }

            HIP_TRACE(hipMemsetAsync((void*)W, 0, size_W, stream));
            rocblas_int j = 0;

            /* ROCSOLVER_LAUNCH_KERNEL((latrd_lower_kernel_naive<256, T>), dim3(1, 1, batch_count), */
            /*                         dim3(256), lmemsize_small, stream, n, k, A, */
            /*                         shiftA + idx2D(j, j, lda), lda, strideA, E + j, strideE, */
            /*                         tau + j, strideP, W, shiftW, ldw, strideW, work); */

            rocblas_int device_id = 0;
            HIP_TRACE(hipGetDevice(&device_id));
            rocblas_int supports_coop_launch = 0;
            HIP_TRACE(hipDeviceGetAttribute(&supports_coop_launch,
                                            hipDeviceAttributeCooperativeLaunch, device_id));
            if(!supports_coop_launch)
            {
                std::cout << "::: Device does not support cooperative launch" << std::endl;
                abort();
            }

            rocblas_stride shiftA_ = shiftA + idx2D(j, j, lda);
            T* tau_j_ = tau + j;
            S* E_j_ = E + j;
            void* kernelArgs[] = {(void*)&n,      (void*)&k,       (void*)&A,    (void*)&shiftA_,
                                  (void*)&lda,    (void*)&strideA, (void*)&E_j_, (void*)&strideE,
                                  (void*)&tau_j_, (void*)&strideP, (void*)&W,    (void*)&shiftW,
                                  (void*)&ldw,    (void*)&strideW, (void*)&work};

            HIP_TRACE(hipLaunchCooperativeKernel(
                (void*)(latrd_lower_kernel_naive<256, T, rocblas_int, S, U>),
                dim3(1, 1, batch_count), dim3(256), kernelArgs, lmemsize_fused, stream));
            HIP_TRACE(hipDeviceSynchronize());
        }
        else
        {
            for(rocblas_int j = 0; j < k; ++j)
            {
                // update column j of A with reflector computed in step j-1
                //----------------------------------------------------------
                ROCSOLVER_LAUNCH_KERNEL(latrd_lower_updateA_kernel<T>,
                                        dim3(grr_updates, grc_updates, batch_count),
                                        dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream,
                                        n, j, A, shiftA, lda, strideA, W, shiftW, ldw, strideW);
                //-------------------------------------------------------------

                // reduce column j of A with new reflector, then copy off-diagonal element
                // to E(j) and set off-diagonal to 1
                //----------------------------------------------------------
                rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                         strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda),
                                         1, strideA, (tau + j), strideP, batch_count, work, norms);
                //-----------------------------------------------------------

                // compute column j of W
                //--------------------------------------------------------------
                static constexpr int NB = 256;
                dim3 gemvt_grid(n + j, 1, batch_count);
                dim3 gemvt_threads(NB);
                ROCSOLVER_LAUNCH_KERNEL((latrd_lower_computeW_gemvt_kernel<NB, T>), gemvt_grid,
                                        gemvt_threads, 0, stream, n, j, A, shiftA, lda, strideA, W,
                                        shiftW, ldw, strideW, W, shiftW + idx2D(0, j, ldw), ldw,
                                        strideW, work, strideblk);

                // update column j of W
                //--------------------------------------------------------------
                ROCSOLVER_LAUNCH_KERNEL(
                    latrd_lower_updateW_kernel<T>, dim3(grr_updates, grc_updates, batch_count),
                    dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n, j, A, shiftA,
                    lda, strideA, W, shiftW, ldw, strideW, work, strideblk, tau, strideP);

                ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                        dim3(1024, 1, 1), 0, stream, n - 1 - j, A,
                                        shiftA + idx2D(j + 1, j, lda), strideA, W,
                                        shiftW + idx2D(j + 1, j, ldw), strideW, tau + j, strideP);
                //--------------------------------------------------------------
            }
        }
    }

    else
    {
        // reduce the last k columns of A
        // main loop running forwards (for each column)
        rocblas_int jw;
        for(rocblas_int j = n - 1; j >= n - k; --j)
        {
            jw = j - n + k;

            // update column j of A with reflector computed in step j-1
            //----------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(latrd_upper_updateA_kernel<T>,
                                    dim3(grr_updates, grc_updates, batch_count),
                                    dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n,
                                    k, j, A, shiftA, lda, strideA, W, shiftW, ldw, strideW);
            //-------------------------------------------------------------

            // reduce column j of A with new reflector, then copy off-diagonal element
            // to E(j) and set off-diagonal to 1
            //----------------------------------------------------------
            rocsolver_larfg_template(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1, strideE,
                                     A, shiftA + idx2D(0, j, lda), 1, strideA, (tau + j - 1),
                                     strideP, batch_count, work, norms);
            //----------------------------------------------------------

            // compute column j of W
            //--------------------------------------------------------------
            static constexpr int NB = 256;
            dim3 gemvt_grid(n + n - j - 1, 1, batch_count);
            dim3 gemvt_threads(NB);
            ROCSOLVER_LAUNCH_KERNEL((latrd_upper_computeW_gemvt_kernel<NB, T>), gemvt_grid,
                                    gemvt_threads, 0, stream, n, k, j, A, shiftA, lda, strideA, W,
                                    shiftW, ldw, strideW, W, shiftW + idx2D(0, jw, ldw), ldw,
                                    strideW, work, strideblk);

            // update column j of W
            //--------------------------------------------------------------
            ROCSOLVER_LAUNCH_KERNEL(
                latrd_upper_updateW_kernel<T>, dim3(grr_updates, grc_updates, batch_count),
                dim3(thr_updates, thc_updates, 1), lmemsize_updates, stream, n, k, j, A, shiftA,
                lda, strideA, W, shiftW, ldw, strideW, work, strideblk, tau, strideP);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>), dim3(1, 1, batch_count),
                                    dim3(1024, 1, 1), 0, stream, j, A, shiftA + idx2D(0, j, lda),
                                    strideA, W, shiftW + idx2D(0, jw, ldw), strideW, tau + j - 1,
                                    strideP);
        }
    }

    roctxRangePop();
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
