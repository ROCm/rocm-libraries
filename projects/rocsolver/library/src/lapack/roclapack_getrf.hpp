/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
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

#include "rocblas.hpp"
#include "roclapack_getf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Constants for inner block size of getrf **/
// clang-format off
#define GETRF_NUMROWS_REAL 20
#define GETRF_NUMCOLS_REAL 13
#define GETRF_INTERVALSROW_REAL                             \
    64, 128, 160, 256, 512, 768, 1024, 1152, 1408, 1792,    \
    1856, 2048, 2560, 2944, 2304, 3584, 5376, 6400, 9216
#define GETRF_INTERVALSCOL_REAL                             \
    20, 28, 40, 56, 80, 112, 144, 208, 240, 288, 416, 480
#define GETRF_INNBLKSIZES_REAL                              \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},    \
    {1,  1,  1,  1, 32, 32, 32, 32, 32, 32, 32, 32, 32},    \
    {1,  1,  1, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32},    \
    {1,  1,  1, 16, 32, 24, 16, 16, 16, 32, 32, 32, 32},    \
    {1,  1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},    \
    {1,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8, 16, 16},    \
    {1,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8},    \
    {1,  1,  1,  1,  1,  1, 16, 16, 16, 16, 16,  8,  8},    \
    {1,  1,  1,  1,  1,  1, 16, 16, 16, 16, 16, 16, 16},    \
    {1,  1,  1,  1,  1,  1, 16, 32, 32, 24, 16, 16, 16},    \
    {1,  1,  1,  1,  1, 24, 16, 32, 32, 24, 16, 24, 16},    \
    {1,  1,  1,  1,  1, 24, 16, 32, 16, 32, 24, 24, 32},    \
    {1,  1,  1,  1, 24, 24, 16, 32, 16, 32, 24, 24, 32},    \
    {1,  1,  1,  1, 24, 24, 16, 32, 16, 32, 24, 32, 32},    \
    {1,  1,  1,  1, 24, 24, 16, 16, 16, 32, 24, 32, 32},    \
    {1,  1,  1,  1, 24, 24, 16, 16, 16, 32, 32, 32, 32},    \
    {1,  1,  1, 24, 24, 24, 16, 16, 16, 32, 32, 32, 32},    \
    {1, 16, 16, 24, 24, 24, 16, 16, 16, 32, 32, 32, 40},    \
    {1, 16, 16, 24, 24, 24, 16, 16, 24, 32, 32, 32, 40},    \
    {1,  8, 16, 24, 24, 24, 16, 16, 24, 32, 32, 32, 40}

#define GETRF_BATCH_NUMROWS_REAL 16
#define GETRF_BATCH_NUMCOLS_REAL 13
#define GETRF_BATCH_INTERVALSROW_REAL                       \
    38, 48, 54, 64, 128, 144, 152, 216, 240, 256, 304, 432, \
    480, 608, 1024
#define GETRF_BATCH_INTERVALSCOL_REAL                       \
    20, 28, 40, 56, 80, 112, 144, 176, 288, 352, 416, 480
#define GETRF_BATCH_INNBLKSIZES_REAL                        \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},    \
    {1,  1,  1, 24, 24, 24, 24,  1,  1,  1,  1,  1,  1},    \
    {1,  1,  1, 24, 32, 32, 32, 32, 32,  1,  1,  1,  1},    \
    {1,  1,  1, 24, 32, 32, 32, 32, 32, 32,  1,  1,  1},    \
    {1,  1, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24},    \
    {1,  1, 16, 16, 16, 24, 24, 24, 24, 24, 24, 24, 24},    \
    {1, 16, 16, 16, 16, 24, 24, 24, 24, 24, 24, 24, 24},    \
    {1, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 24},    \
    {1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24},    \
    {8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24},    \
    {8,  8,  8,  8, 16, 16, 16, 16, 16, 16, 16, 16, 16},    \
    {8,  8,  8,  8,  8, 16, 16, 16, 16, 16, 16, 16, 16},    \
    {8,  8,  8,  8,  8,  8, 16, 16, 16, 16, 16, 16, 16},    \
    {8,  8,  8,  8,  8,  8, 16, 16, 16, 16, 16, 16, 24},    \
    {8,  8,  8,  8,  8,  8,  8, 16, 16, 16, 16, 16, 24},    \
    {8,  8,  8,  8, 16, 16, 16, 16, 16, 16, 16, 24, 24}

#define GETRF_NPVT_NUMROWS_REAL 4
#define GETRF_NPVT_NUMCOLS_REAL 3
#define GETRF_NPVT_INTERVALSROW_REAL                        \
    64, 1536, 3072
#define GETRF_NPVT_INTERVALSCOL_REAL                        \
    40, 56
#define GETRF_NPVT_INNBLKSIZES_REAL                         \
    {1, 1, 1},                                              \
    {1, 1, 16},                                             \
    {1, 24, 16},                                            \
    {1, 1, 16}

#define GETRF_NPVT_BATCH_NUMROWS_REAL 3
#define GETRF_NPVT_BATCH_NUMCOLS_REAL 3
#define GETRF_NPVT_BATCH_INTERVALSROW_REAL                  \
    40, 46
#define GETRF_NPVT_BATCH_INTERVALSCOL_REAL                  \
    40, 56
#define GETRF_NPVT_BATCH_INNBLKSIZES_REAL                   \
    {1, 1, 1},                                              \
    {1, 1, 32},                                             \
    {1, 16, 32}

#define GETRF_NUMROWS_COMPLEX 21
#define GETRF_NUMCOLS_COMPLEX 10
#define GETRF_INTERVALSROW_COMPLEX                          \
    64, 128, 160, 192, 256, 320, 512, 768, 896, 1024, 1216, \
    1536, 1728, 1984, 2560, 2944, 3712, 5632, 7424, 9216
#define GETRF_INTERVALSCOL_COMPLEX                          \
    20, 28, 40, 56, 80, 144, 208, 288, 416
#define GETRF_INNBLKSIZES_COMPLEX                           \
    {1,  1,  1,  1,  1,  1,  1,  1,  1,  1},                \
    {1,  1,  1,  1, 16, 16, 32, 32, 32, 32},                \
    {1,  1,  1, 16, 16, 16, 32, 32, 32, 32},                \
    {1,  1,  1, 16, 16, 16, 16, 16, 32, 32},                \
    {1,  1, 16, 16, 16, 16, 16, 16, 32, 32},                \
    {1,  1, 16, 16, 16, 16, 16, 16, 16, 16},                \
    {1,  8, 16, 16, 16, 16, 16, 16, 16, 16},                \
    {1,  8,  8,  8,  8,  8,  8,  8, 16, 16},                \
    {8,  8,  8,  8,  8,  8,  8,  8, 16, 16},                \
    {8,  8,  8,  8,  8,  8,  8,  8,  8,  8},                \
    {1,  1,  1,  1,  1, 24, 16, 16,  8,  8},                \
    {1,  1,  1,  1,  1, 24, 16, 16, 24, 16},                \
    {1,  1,  1,  1,  1, 24, 16, 16, 24, 24},                \
    {1,  1,  1,  1, 16, 24, 32, 32, 24, 24},                \
    {1,  1,  1,  1, 16, 24, 32, 32, 32, 32},                \
    {1,  1,  1,  1, 16, 24, 16, 32, 32, 32},                \
    {1,  1,  1, 16, 16, 16, 16, 32, 32, 32},                \
    {1,  1, 16, 16, 16, 16, 16, 32, 32, 32},                \
    {1,  1, 16, 16, 16, 16, 16, 16, 32, 32},                \
    {1,  1, 16, 16, 16, 16, 24, 16, 32, 32},                \
    {1,  1, 16, 16, 16, 16, 24, 24, 32, 32}

#define GETRF_BATCH_NUMROWS_COMPLEX 9
#define GETRF_BATCH_NUMCOLS_COMPLEX 6
#define GETRF_BATCH_INTERVALSROW_COMPLEX                    \
    24, 26, 32, 128, 208, 256, 304, 432
#define GETRF_BATCH_INTERVALSCOL_COMPLEX                    \
    20, 28, 40, 80, 144
#define GETRF_BATCH_INNBLKSIZES_COMPLEX                     \
    {1,  1,  1,  1,  1,  1},                                \
    {1,  1, 16, 16,  1,  1},                                \
    {1,  1, 16, 16, 16,  1},                                \
    {1,  1, 16, 16, 16, 16},                                \
    {1, 16, 16, 16, 16, 16},                                \
    {1,  8, 16, 16, 16, 16},                                \
    {8,  8,  8, 16, 16, 16},                                \
    {8,  8,  8,  8, 16, 16},                                \
    {8,  8,  8,  8,  8, 16}

#define GETRF_NPVT_NUMROWS_COMPLEX 4
#define GETRF_NPVT_NUMCOLS_COMPLEX 4
#define GETRF_NPVT_INTERVALSROW_COMPLEX                     \
    64, 384, 5376
#define GETRF_NPVT_INTERVALSCOL_COMPLEX                     \
    56, 80, 288
#define GETRF_NPVT_INNBLKSIZES_COMPLEX                      \
    {1,  1,  1,  1},                                        \
    {1,  1,  8,  8},                                        \
    {1,  1,  8, 16},                                        \
    {1, 32,  8, 16}

#define GETRF_NPVT_BATCH_NUMROWS_COMPLEX 5
#define GETRF_NPVT_BATCH_NUMCOLS_COMPLEX 4
#define GETRF_NPVT_BATCH_INTERVALSROW_COMPLEX               \
    24, 256, 640, 1024
#define GETRF_NPVT_BATCH_INTERVALSCOL_COMPLEX               \
    20, 28, 288
#define GETRF_NPVT_BATCH_INNBLKSIZES_COMPLEX                \
    {1, 1, 1, 1},                                           \
    {1, 1, 16, 16},                                         \
    {1, 1, 16, 32},                                         \
    {1, 8, 16, 32},                                         \
    {1, 8, 16, 16}
// clang-format on

/** Execute all permutations dictated by the panel factorization
    in parallel (concurrency by rows and columns) **/
template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void getrf_row_permutate(const I n,
                                          const I offset,
                                          const I blk,
                                          U AA,
                                          const rocblas_stride shiftA,
                                          const I inca,
                                          const I lda,
                                          const rocblas_stride strideA,
                                          I* pividx,
                                          const rocblas_stride stridePI)
{
    I id = hipBlockIdx_z;
    I tx = hipThreadIdx_x;
    I ty = hipThreadIdx_y;
    I bdx = hipBlockDim_x;
    I j = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + ty;
    if(j >= offset)
        j += blk;

    if(j < n)
    {
        // batch instance
        T* A = load_ptr_batch(AA, id, shiftA, strideA);
        I* piv = pividx + id * stridePI;

        // shared mem for temporary values
        extern __shared__ double lmem[];
        T* temp = reinterpret_cast<T*>(lmem);

        // do permutations in parallel (each tx perform a row swap)
        I idx1 = piv[tx];
        I idx2 = piv[idx1];
        temp[tx + ty * bdx] = A[idx1 * inca + j * lda];
        A[idx1 * inca + j * lda] = A[idx2 * inca + j * lda];
        __syncthreads();

        // copy temp results back to A
        A[tx * inca + j * lda] = temp[tx + ty * bdx];
    }
}

/** This function returns the outer block size based on defined variables
    tunable by the user (defined in ideal_sizes.hpp) **/
template <bool ISBATCHED, typename T, typename I, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
I getrf_get_blksize(I dim, const bool pivot)
{
    I blk;

    if(ISBATCHED)
    {
        if(pivot)
        {
            I size[] = {GETRF_BATCH_BLKSIZES_REAL};
            I intervals[] = {GETRF_BATCH_INTERVALS_REAL};
            I max = GETRF_BATCH_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            I size[] = {GETRF_NPVT_BATCH_BLKSIZES_REAL};
            I intervals[] = {GETRF_NPVT_BATCH_INTERVALS_REAL};
            I max = GETRF_NPVT_BATCH_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
    }
    else
    {
        if(pivot)
        {
            I size[] = {GETRF_BLKSIZES_REAL};
            I intervals[] = {GETRF_INTERVALS_REAL};
            I max = GETRF_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            I size[] = {GETRF_NPVT_BLKSIZES_REAL};
            I intervals[] = {GETRF_NPVT_INTERVALS_REAL};
            I max = GETRF_NPVT_NUM_INTERVALS_REAL;
            blk = size[get_index(intervals, max, dim)];
        }
    }

    if(blk == 1 || blk == -1)
        blk *= dim;

    return blk;
}

/** Complex type version **/
template <bool ISBATCHED, typename T, typename I, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
I getrf_get_blksize(I dim, const bool pivot)
{
    I blk;

    if(ISBATCHED)
    {
        if(pivot)
        {
            I size[] = {GETRF_BATCH_BLKSIZES_COMPLEX};
            I intervals[] = {GETRF_BATCH_INTERVALS_COMPLEX};
            I max = GETRF_BATCH_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            I size[] = {GETRF_NPVT_BATCH_BLKSIZES_COMPLEX};
            I intervals[] = {GETRF_NPVT_BATCH_INTERVALS_COMPLEX};
            I max = GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
    }
    else
    {
        if(pivot)
        {
            I size[] = {GETRF_BLKSIZES_COMPLEX};
            I intervals[] = {GETRF_INTERVALS_COMPLEX};
            I max = GETRF_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
        else
        {
            I size[] = {GETRF_NPVT_BLKSIZES_COMPLEX};
            I intervals[] = {GETRF_NPVT_INTERVALS_COMPLEX};
            I max = GETRF_NPVT_NUM_INTERVALS_COMPLEX;
            blk = size[get_index(intervals, max, dim)];
        }
    }

    if(blk == 1 || blk == -1)
        blk *= dim;

    return blk;
}

/** This function returns the inner block size. This has been tuned based on
    experiments with panel matrices; it is not expected to change a lot.
    (not tunable by the user for now) **/
template <bool ISBATCHED, typename T, typename I, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
I getrf_get_innerBlkSize(I m, I n, const bool pivot)
{
    I blk;

    if(ISBATCHED)
    {
        if(pivot)
        {
            I M = GETRF_BATCH_NUMROWS_REAL - 1;
            I N = GETRF_BATCH_NUMCOLS_REAL - 1;
            I intervalsM[] = {GETRF_BATCH_INTERVALSROW_REAL};
            I intervalsN[] = {GETRF_BATCH_INTERVALSCOL_REAL};
            I size[][GETRF_BATCH_NUMCOLS_REAL] = {GETRF_BATCH_INNBLKSIZES_REAL};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
        else
        {
            I M = GETRF_NPVT_BATCH_NUMROWS_REAL - 1;
            I N = GETRF_NPVT_BATCH_NUMCOLS_REAL - 1;
            I intervalsM[] = {GETRF_NPVT_BATCH_INTERVALSROW_REAL};
            I intervalsN[] = {GETRF_NPVT_BATCH_INTERVALSCOL_REAL};
            I size[][GETRF_NPVT_BATCH_NUMCOLS_REAL] = {GETRF_NPVT_BATCH_INNBLKSIZES_REAL};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
    }
    else
    {
        if(pivot)
        {
            I M = GETRF_NUMROWS_REAL - 1;
            I N = GETRF_NUMCOLS_REAL - 1;
            I intervalsM[] = {GETRF_INTERVALSROW_REAL};
            I intervalsN[] = {GETRF_INTERVALSCOL_REAL};
            I size[][GETRF_NUMCOLS_REAL] = {GETRF_INNBLKSIZES_REAL};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
        else
        {
            I M = GETRF_NPVT_NUMROWS_REAL - 1;
            I N = GETRF_NPVT_NUMCOLS_REAL - 1;
            I intervalsM[] = {GETRF_NPVT_INTERVALSROW_REAL};
            I intervalsN[] = {GETRF_NPVT_INTERVALSCOL_REAL};
            I size[][GETRF_NPVT_NUMCOLS_REAL] = {GETRF_NPVT_INNBLKSIZES_REAL};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
    }

    if(blk == 1)
        blk = n;

    return blk;
}

/** complex type version **/
template <bool ISBATCHED, typename T, typename I, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
I getrf_get_innerBlkSize(I m, I n, const bool pivot)
{
    I blk;

    if(ISBATCHED)
    {
        if(pivot)
        {
            I M = GETRF_BATCH_NUMROWS_COMPLEX - 1;
            I N = GETRF_BATCH_NUMCOLS_COMPLEX - 1;
            I intervalsM[] = {GETRF_BATCH_INTERVALSROW_COMPLEX};
            I intervalsN[] = {GETRF_BATCH_INTERVALSCOL_COMPLEX};
            I size[][GETRF_BATCH_NUMCOLS_COMPLEX] = {GETRF_BATCH_INNBLKSIZES_COMPLEX};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
        else
        {
            I M = GETRF_NPVT_BATCH_NUMROWS_COMPLEX - 1;
            I N = GETRF_NPVT_BATCH_NUMCOLS_COMPLEX - 1;
            I intervalsM[] = {GETRF_NPVT_BATCH_INTERVALSROW_COMPLEX};
            I intervalsN[] = {GETRF_NPVT_BATCH_INTERVALSCOL_COMPLEX};
            I size[][GETRF_NPVT_BATCH_NUMCOLS_COMPLEX] = {GETRF_NPVT_BATCH_INNBLKSIZES_COMPLEX};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
    }
    else
    {
        if(pivot)
        {
            I M = GETRF_NUMROWS_COMPLEX - 1;
            I N = GETRF_NUMCOLS_COMPLEX - 1;
            I intervalsM[] = {GETRF_INTERVALSROW_COMPLEX};
            I intervalsN[] = {GETRF_INTERVALSCOL_COMPLEX};
            I size[][GETRF_NUMCOLS_COMPLEX] = {GETRF_INNBLKSIZES_COMPLEX};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
        else
        {
            I M = GETRF_NPVT_NUMROWS_COMPLEX - 1;
            I N = GETRF_NPVT_NUMCOLS_COMPLEX - 1;
            I intervalsM[] = {GETRF_NPVT_INTERVALSROW_COMPLEX};
            I intervalsN[] = {GETRF_NPVT_INTERVALSCOL_COMPLEX};
            I size[][GETRF_NPVT_NUMCOLS_COMPLEX] = {GETRF_NPVT_INNBLKSIZES_COMPLEX};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
    }

    if(blk == 1)
        blk = n;

    return blk;
}

/** This is the implementation of the factorization of the
    panel blocks in getrf **/
template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename U>
rocblas_status getrf_panelLU(rocblas_handle handle,
                             const I mm,
                             const I nn,
                             const I n,
                             U A,
                             const rocblas_stride r_shiftA,
                             const I inca,
                             const I lda,
                             const rocblas_stride strideA,
                             I* ipiv,
                             const rocblas_stride shiftP,
                             const rocblas_stride strideP,
                             INFO* info,
                             const I batch_count,
                             const bool pivot,
                             T* scalars,
                             void* work1,
                             void* work2,
                             void* work3,
                             void* work4,
                             const bool optim_mem,
                             T* pivotval,
                             I* pivotidx,
                             const I offset,
                             I* permut_idx,
                             const rocblas_stride stridePI)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    // r_shiftA is the row where the panel-block starts,
    // the actual position of the panel-block in the matrix is:
    rocblas_stride shiftA = r_shiftA + idx2D(0, offset, inca, lda);

    I blk = getrf_get_innerBlkSize<ISBATCHED, T>(mm, nn, pivot);
    I jb;
    I dimx, dimy, blocks, blocksy;
    dim3 grid, threads;
    size_t lmemsize;

    // Main loop
    for(I k = 0; k < nn; k += blk)
    {
        jb = std::min(nn - k, blk); // number of columns/pivots in the inner block

        // factorize inner panel block
        rocsolver_getf2_template<ISBATCHED, T>(handle, mm - k, jb, A, shiftA + idx2D(k, k, inca, lda),
                                               inca, lda, strideA, ipiv, shiftP + k, strideP, info,
                                               batch_count, scalars, pivotval, pivotidx, pivot,
                                               offset + k, permut_idx, stridePI);
        if(pivot)
        {
            dimx = jb;
            dimy = I(1024) / dimx;
            blocks = (n - jb - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimx * dimy * sizeof(T);

            // swap rows
            ROCSOLVER_LAUNCH_KERNEL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, n,
                                    offset + k, jb, A, r_shiftA + k * inca, inca, lda, strideA,
                                    permut_idx, stridePI);
        }

        // update trailing sub-block
        if(k + jb < nn)
        {
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb,
                nn - k - jb, A, shiftA + idx2D(k, k, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(k, k + jb, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);

            if(k + jb < mm)
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mm - k - jb,
                               nn - k - jb, jb, &minone, A, shiftA + idx2D(k + jb, k, inca, lda),
                               inca, lda, strideA, A, shiftA + idx2D(k, k + jb, inca, lda), inca,
                               lda, strideA, &one, A, shiftA + idx2D(k + jb, k + jb, inca, lda),
                               inca, lda, strideA, batch_count, (T**)nullptr);
        }
    }

    return rocblas_status_success;
}

/** Return the sizes of the different workspace arrays **/
template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_getrf_getMemorySize(const I m,
                                   const I n,
                                   const bool pivot,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx,
                                   size_t* size_iipiv,
                                   size_t* size_iinfo,
                                   bool* optim_mem,
                                   const I lda = 1,
                                   const I inca = 1)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    I dim = std::min(m, n);
    I blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);

    if(blk == 0)
    {
        // requirements for one single GETF2
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, n, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx, false, inca);
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }
    else
    {
        // largest block panel dimension is 512
        dim = min(dim, I(512));

        // requirements for largest possible GETF2 for the sub blocks
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, dim, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx, true, inca);

        // extra workspace to store info about singularity and pivots of sub blocks
        *size_iinfo = sizeof(I) * batch_count;
        *size_iipiv = pivot ? m * sizeof(I) * batch_count : 0;

        // extra workspace for calling largest possible TRSM
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none, dim, n,
                                                batch_count, size_work1, size_work2, size_work3,
                                                size_work4, optim_mem, true, lda, lda, inca, inca);
        if(!pivot)
        {
            size_t w1, w2, w3, w4;
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none, m,
                                                    dim, batch_count, &w1, &w2, &w3, &w4, optim_mem,
                                                    true, lda, lda, inca, inca);
            *size_work1 = std::max(*size_work1, w1);
            *size_work2 = std::max(*size_work2, w2);
            *size_work3 = std::max(*size_work3, w3);
            *size_work4 = std::max(*size_work4, w4);
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename INFO, typename U>
rocblas_status rocsolver_getrf_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I inca,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        I* ipiv,
                                        const rocblas_stride shiftP,
                                        const rocblas_stride strideP,
                                        INFO* info,
                                        const I batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivotval,
                                        I* pivotidx,
                                        I* iipiv,
                                        INFO* iinfo,
                                        const bool optim_mem,
                                        const bool pivot)
{
    ROCSOLVER_ENTER("getrf", "m:", m, "n:", n, "shiftA:", shiftA, "inca:", inca, "lda:", lda,
                    "shiftP:", shiftP, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;
    I dim = std::min(m, n);
    I blocks, blocksy;
    dim3 grid, threads;

    // quick return if no dimensions
    if(m == 0 || n == 0)
    {
        blocks = (batch_count - 1) / BS1 + 1;
        grid = dim3(blocks, 1, 1);
        threads = dim3(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, info, batch_count, 0);
        return rocblas_status_success;
    }

    // size of outer blocks
    I blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);

    if(blk == 0)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, inca, lda, strideA,
                                                      ipiv, shiftP, strideP, info, batch_count,
                                                      scalars, pivotval, pivotidx, pivot);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T minone = -1;

    I jb, dimx, dimy;
    I nextpiv, mm, nn;
    size_t lmemsize;
    I j = 0;

    // in the npvt cases, panel determines whether the whole block-panel or only the
    // diagonal block is factorized
    bool panel = false;
    if(blk < 0)
    {
        panel = true;
        blk = -blk;
    }

    // MAIN LOOP
    for(I j = 0; j < dim; j += blk)
    {
        jb = std::min(dim - j, blk);

        if(pivot || panel)
        {
            // factorize outer block panel
            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j * inca, inca,
                                               lda, strideA, ipiv, shiftP + j, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j, iipiv, m);
        }
        else
        {
            // factorize only outer diagonal block
            getrf_panelLU<BATCHED, STRIDED, T>(handle, jb, jb, n, A, shiftA + j * inca, inca, lda,
                                               strideA, ipiv, shiftP + j, strideP, info,
                                               batch_count, pivot, scalars, work1, work2, work3,
                                               work4, optim_mem, pivotval, pivotidx, j, iipiv, m);

            // update remaining rows in outer panel
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
                m - j - jb, jb, A, shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(jb + j, j, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);
        }

        // update trailing matrix
        nextpiv = j + jb; //position for the matrix update
        mm = m - nextpiv; //size for the matrix update
        nn = n - nextpiv; //size for the matrix update
        if(nextpiv < n)
        {
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb, nn, A,
                shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
                shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);

            if(nextpiv < m)
            {
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb,
                               &minone, A, shiftA + idx2D(nextpiv, j, inca, lda), inca, lda,
                               strideA, A, shiftA + idx2D(j, nextpiv, inca, lda), inca, lda,
                               strideA, &one, A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), inca,
                               lda, strideA, batch_count, (T**)nullptr);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
