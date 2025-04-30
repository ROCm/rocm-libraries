/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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

#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <rocblas_int DIM, typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(TRTRI_MAX_COLS)
    getri_kernel_small(U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_int shiftP,
                       const rocblas_stride strideP,
                       rocblas_int* info,
                       const bool complete,
                       const bool pivot)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if(i >= DIM)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
    rocblas_int* ipiv;
    if(pivot)
        ipiv = load_ptr_batch<rocblas_int>(ipivA, b, shiftP, strideP);

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    T temp;
    rocblas_int jp;

    // read corresponding row from global memory in local array
    T rA[DIM];
#pragma unroll
    for(int j = 0; j < DIM; ++j)
        rA[j] = A[i + j * lda];

    if(complete)
    {
        __shared__ T diag[DIM];
        __shared__ rocblas_int _info;

        // compute info
        if(i == 0)
            _info = 0;
        __syncthreads();
        if(rA[i] == 0)
        {
            rocblas_int _info_temp = _info;
            while(_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
        __syncthreads();

        if(i == 0)
            info[b] = _info;
        if(_info != 0)
            return;

        //--- TRTRI ---
        // diagonal element
        rA[i] = 1.0 / rA[i];
        diag[i] = -rA[i];

        // compute element i of each column j
#pragma unroll
        for(rocblas_int j = 1; j < DIM; j++)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i < j)
            {
                temp = 0;

                for(rocblas_int ii = i; ii < j; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
            __syncthreads();
        }
    }

    if(info[b] != 0)
        return;

        //--- GETRI ---
#pragma unroll
    for(rocblas_int j = DIM - 2; j >= 0; j--)
    {
        // extract lower triangular column (copy_and_zero)
        if(i > j)
        {
            common[i] = rA[j];
            rA[j] = 0;
        }
        __syncthreads();

        // update column j (gemv)
        temp = 0;

        for(rocblas_int ii = j + 1; ii < DIM; ii++)
            temp += rA[ii] * common[ii];

        rA[j] -= temp;
    }

    // apply pivots (getri_pivot)
    if(pivot)
    {
#pragma unroll
        for(rocblas_int j = DIM - 2; j >= 0; j--)
        {
            jp = ipiv[j] - 1;
            if(jp != j)
                swap(rA[j], rA[jp]);
        }
    }

// write results to global memory from local array
#pragma unroll
    for(int j = 0; j < DIM; j++)
        A[i + j * lda] = rA[j];
}

/*************************************************************
    Launchers of specilized  kernels
*************************************************************/

template <typename T, typename U>
rocblas_status getri_run_small(rocblas_handle handle,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const bool complete,
                               const bool pivot)
{
#define RUN_GETRI_SMALL(DIM)                                                                      \
    ROCSOLVER_LAUNCH_KERNEL((getri_kernel_small<DIM, T>), grid, block, 0, stream, A, shiftA, lda, \
                            strideA, ipiv, shiftP, strideP, info, complete, pivot)

    dim3 grid(batch_count, 1, 1);
    dim3 block(TRTRI_MAX_COLS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch(n)
    {
    case 1: RUN_GETRI_SMALL(1); break;
    case 2: RUN_GETRI_SMALL(2); break;
    case 3: RUN_GETRI_SMALL(3); break;
    case 4: RUN_GETRI_SMALL(4); break;
    case 5: RUN_GETRI_SMALL(5); break;
    case 6: RUN_GETRI_SMALL(6); break;
    case 7: RUN_GETRI_SMALL(7); break;
    case 8: RUN_GETRI_SMALL(8); break;
    case 9: RUN_GETRI_SMALL(9); break;
    case 10: RUN_GETRI_SMALL(10); break;
    case 11: RUN_GETRI_SMALL(11); break;
    case 12: RUN_GETRI_SMALL(12); break;
    case 13: RUN_GETRI_SMALL(13); break;
    case 14: RUN_GETRI_SMALL(14); break;
    case 15: RUN_GETRI_SMALL(15); break;
    case 16: RUN_GETRI_SMALL(16); break;
    case 17: RUN_GETRI_SMALL(17); break;
    case 18: RUN_GETRI_SMALL(18); break;
    case 19: RUN_GETRI_SMALL(19); break;
    case 20: RUN_GETRI_SMALL(20); break;
    case 21: RUN_GETRI_SMALL(21); break;
    case 22: RUN_GETRI_SMALL(22); break;
    case 23: RUN_GETRI_SMALL(23); break;
    case 24: RUN_GETRI_SMALL(24); break;
    case 25: RUN_GETRI_SMALL(25); break;
    case 26: RUN_GETRI_SMALL(26); break;
    case 27: RUN_GETRI_SMALL(27); break;
    case 28: RUN_GETRI_SMALL(28); break;
    case 29: RUN_GETRI_SMALL(29); break;
    case 30: RUN_GETRI_SMALL(30); break;
    case 31: RUN_GETRI_SMALL(31); break;
    case 32: RUN_GETRI_SMALL(32); break;
    case 33: RUN_GETRI_SMALL(33); break;
    case 34: RUN_GETRI_SMALL(34); break;
    case 35: RUN_GETRI_SMALL(35); break;
    case 36: RUN_GETRI_SMALL(36); break;
    case 37: RUN_GETRI_SMALL(37); break;
    case 38: RUN_GETRI_SMALL(38); break;
    case 39: RUN_GETRI_SMALL(39); break;
    case 40: RUN_GETRI_SMALL(40); break;
    case 41: RUN_GETRI_SMALL(41); break;
    case 42: RUN_GETRI_SMALL(42); break;
    case 43: RUN_GETRI_SMALL(43); break;
    case 44: RUN_GETRI_SMALL(44); break;
    case 45: RUN_GETRI_SMALL(45); break;
    case 46: RUN_GETRI_SMALL(46); break;
    case 47: RUN_GETRI_SMALL(47); break;
    case 48: RUN_GETRI_SMALL(48); break;
    case 49: RUN_GETRI_SMALL(49); break;
    case 50: RUN_GETRI_SMALL(50); break;
    case 51: RUN_GETRI_SMALL(51); break;
    case 52: RUN_GETRI_SMALL(52); break;
    case 53: RUN_GETRI_SMALL(53); break;
    case 54: RUN_GETRI_SMALL(54); break;
    case 55: RUN_GETRI_SMALL(55); break;
    case 56: RUN_GETRI_SMALL(56); break;
    case 57: RUN_GETRI_SMALL(57); break;
    case 58: RUN_GETRI_SMALL(58); break;
    case 59: RUN_GETRI_SMALL(59); break;
    case 60: RUN_GETRI_SMALL(60); break;
    case 61: RUN_GETRI_SMALL(61); break;
    case 62: RUN_GETRI_SMALL(62); break;
    case 63: RUN_GETRI_SMALL(63); break;
    case 64: RUN_GETRI_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GETRI_SMALL(T, U)                                              \
    template rocblas_status getri_run_small<T, U>(                                 \
        rocblas_handle handle, const rocblas_int n, U A, const rocblas_int shiftA, \
        const rocblas_int lda, const rocblas_stride strideA, rocblas_int* ipiv,    \
        const rocblas_int shiftP, const rocblas_stride strideP, rocblas_int* info, \
        const rocblas_int batch_count, const bool complete, const bool pivot)

ROCSOLVER_END_NAMESPACE
