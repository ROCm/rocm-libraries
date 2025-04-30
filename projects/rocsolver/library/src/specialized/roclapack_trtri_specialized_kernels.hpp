/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
    trti2_kernel_small(const rocblas_fill uplo,
                       const rocblas_diagonal diagtype,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA)
{
    int b = hipBlockIdx_x;
    int i = hipThreadIdx_x;

    if(i >= DIM)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);

    // read corresponding row from global memory in local array
    T rA[DIM];
#pragma unroll
    for(int j = 0; j < DIM; ++j)
        rA[j] = A[i + j * lda];

    // shared memory (for communication between threads in group)
    __shared__ T common[DIM];
    __shared__ T diag[DIM];
    T temp;

    // diagonal element
    const bool unit = (diagtype == rocblas_diagonal_unit);
    if(unit)
    {
        diag[i] = -1.0;
    }
    else
    {
        rA[i] = 1.0 / rA[i];
        diag[i] = -rA[i];
    }

    // compute element i of each column j
    if(uplo == rocblas_fill_upper)
    {
#pragma unroll
        for(rocblas_int j = 1; j < DIM; j++)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i < j)
            {
                temp = unit ? common[i] : rA[i] * common[i];

                for(rocblas_int ii = i + 1; ii < j; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
        }
    }
    else
    {
#pragma unroll
        for(rocblas_int j = DIM - 2; j >= 0; j--)
        {
            // share current column and diagonal
            common[i] = rA[j];
            __syncthreads();

            if(i > j)
            {
                temp = unit ? common[i] : rA[i] * common[i];

                for(rocblas_int ii = j + 1; ii < i; ii++)
                    temp += rA[ii] * common[ii];

                rA[j] = diag[j] * temp;
            }
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
void trti2_run_small(rocblas_handle handle,
                     const rocblas_fill uplo,
                     const rocblas_diagonal diag,
                     const rocblas_int n,
                     U A,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     const rocblas_int batch_count)
{
#define RUN_TRTI2_SMALL(DIM)                                                                     \
    ROCSOLVER_LAUNCH_KERNEL((trti2_kernel_small<DIM, T>), grid, block, 0, stream, uplo, diag, A, \
                            shiftA, lda, strideA)

    dim3 grid(batch_count, 1, 1);
    dim3 block(TRTRI_MAX_COLS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    switch(n)
    {
    case 1: RUN_TRTI2_SMALL(1); break;
    case 2: RUN_TRTI2_SMALL(2); break;
    case 3: RUN_TRTI2_SMALL(3); break;
    case 4: RUN_TRTI2_SMALL(4); break;
    case 5: RUN_TRTI2_SMALL(5); break;
    case 6: RUN_TRTI2_SMALL(6); break;
    case 7: RUN_TRTI2_SMALL(7); break;
    case 8: RUN_TRTI2_SMALL(8); break;
    case 9: RUN_TRTI2_SMALL(9); break;
    case 10: RUN_TRTI2_SMALL(10); break;
    case 11: RUN_TRTI2_SMALL(11); break;
    case 12: RUN_TRTI2_SMALL(12); break;
    case 13: RUN_TRTI2_SMALL(13); break;
    case 14: RUN_TRTI2_SMALL(14); break;
    case 15: RUN_TRTI2_SMALL(15); break;
    case 16: RUN_TRTI2_SMALL(16); break;
    case 17: RUN_TRTI2_SMALL(17); break;
    case 18: RUN_TRTI2_SMALL(18); break;
    case 19: RUN_TRTI2_SMALL(19); break;
    case 20: RUN_TRTI2_SMALL(20); break;
    case 21: RUN_TRTI2_SMALL(21); break;
    case 22: RUN_TRTI2_SMALL(22); break;
    case 23: RUN_TRTI2_SMALL(23); break;
    case 24: RUN_TRTI2_SMALL(24); break;
    case 25: RUN_TRTI2_SMALL(25); break;
    case 26: RUN_TRTI2_SMALL(26); break;
    case 27: RUN_TRTI2_SMALL(27); break;
    case 28: RUN_TRTI2_SMALL(28); break;
    case 29: RUN_TRTI2_SMALL(29); break;
    case 30: RUN_TRTI2_SMALL(30); break;
    case 31: RUN_TRTI2_SMALL(31); break;
    case 32: RUN_TRTI2_SMALL(32); break;
    case 33: RUN_TRTI2_SMALL(33); break;
    case 34: RUN_TRTI2_SMALL(34); break;
    case 35: RUN_TRTI2_SMALL(35); break;
    case 36: RUN_TRTI2_SMALL(36); break;
    case 37: RUN_TRTI2_SMALL(37); break;
    case 38: RUN_TRTI2_SMALL(38); break;
    case 39: RUN_TRTI2_SMALL(39); break;
    case 40: RUN_TRTI2_SMALL(40); break;
    case 41: RUN_TRTI2_SMALL(41); break;
    case 42: RUN_TRTI2_SMALL(42); break;
    case 43: RUN_TRTI2_SMALL(43); break;
    case 44: RUN_TRTI2_SMALL(44); break;
    case 45: RUN_TRTI2_SMALL(45); break;
    case 46: RUN_TRTI2_SMALL(46); break;
    case 47: RUN_TRTI2_SMALL(47); break;
    case 48: RUN_TRTI2_SMALL(48); break;
    case 49: RUN_TRTI2_SMALL(49); break;
    case 50: RUN_TRTI2_SMALL(50); break;
    case 51: RUN_TRTI2_SMALL(51); break;
    case 52: RUN_TRTI2_SMALL(52); break;
    case 53: RUN_TRTI2_SMALL(53); break;
    case 54: RUN_TRTI2_SMALL(54); break;
    case 55: RUN_TRTI2_SMALL(55); break;
    case 56: RUN_TRTI2_SMALL(56); break;
    case 57: RUN_TRTI2_SMALL(57); break;
    case 58: RUN_TRTI2_SMALL(58); break;
    case 59: RUN_TRTI2_SMALL(59); break;
    case 60: RUN_TRTI2_SMALL(60); break;
    case 61: RUN_TRTI2_SMALL(61); break;
    case 62: RUN_TRTI2_SMALL(62); break;
    case 63: RUN_TRTI2_SMALL(63); break;
    case 64: RUN_TRTI2_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_TRTI2_SMALL(T, U)                                                          \
    template void trti2_run_small<T, U>(rocblas_handle handle, const rocblas_fill uplo,        \
                                        const rocblas_diagonal diag, const rocblas_int n, U A, \
                                        const rocblas_int shiftA, const rocblas_int lda,       \
                                        const rocblas_stride strideA, const rocblas_int batch_count)

ROCSOLVER_END_NAMESPACE
