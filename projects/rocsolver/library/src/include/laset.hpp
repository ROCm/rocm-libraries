// todo: borrowed from Ed's CholQR branch.

/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once

#include <algorithm>
#include <cassert>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "lib_macros.hpp"
#include "rocsolver_logger.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// -----------------------------
// Initialize matrix
// motivated by xLASET in LAPACK
//
// matrix A is m by n
//
// uplo_c == 'U' : assign to upper triangular matrix
// uplo_c == 'L' : assign to lower triangular matrix
// uplo_c == 'G' : assign to entire matrix
//
//
// assign beta to diagonal
// assign alpha to off-diagonal
// -----------------------------

#ifndef LASET_MAX_THREADS
#define LASET_MAX_THREADS 256
#endif

template <typename T, typename I, typename Istride, typename UA>
__global__ static void __launch_bounds__(LASET_MAX_THREADS) laset_kernel(char const uplo_c,
                                                                         I const m,
                                                                         I const n,
                                                                         T const alpha,
                                                                         T const beta,

                                                                         UA A_,
                                                                         Istride const shiftA,
                                                                         I const lda,
                                                                         Istride const strideA,

                                                                         I const batch_count)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }
    bool const use_upper = (uplo_c == 'U') || (uplo_c == 'u');
    bool const use_lower = (uplo_c == 'L') || (uplo_c == 'l');
    bool const use_full = (uplo_c == 'G') || (uplo_c == 'g'); // "GE" general full matrix

    {
        bool const isvalid_uplo = (use_upper || use_lower || use_full);
        assert(isvalid_uplo);
    }

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A = load_ptr_batch<T>(A_, bid, shiftA, strideA);

        if(use_lower)
        {
            // ---------------------------------
            // assign to lower triangular matrix
            // ---------------------------------
            for(I j = 0 + j_start; j < n; j += j_inc)
            {
                for(I i = j + i_start; i < m; i += i_inc)
                {
                    bool const is_diagonal = (i == j);
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = (is_diagonal) ? beta : alpha;

                    A[ij] = aij;
                }
            }
        }
        else if(use_upper)
        {
            // ---------------------------------
            // assign to upper triangular matrix
            // ---------------------------------

            for(I j = 0 + j_start; j < n; j += j_inc)
            {
                for(I i = 0 + i_start; i < std::min(m, j + 1); i += i_inc)
                {
                    bool const is_diagonal = (i == j);
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = (is_diagonal) ? beta : alpha;

                    A[ij] = aij;
                }
            }
        }
        else if(use_full)
        {
            // ------------------------
            // assign to entire matrix
            // ------------------------

            for(I j = 0 + j_start; j < n; j += j_inc)
            {
                for(I i = 0 + i_start; i < m; i += i_inc)
                {
                    bool const is_diagonal = (i == j);
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = (is_diagonal) ? beta : alpha;

                    A[ij] = aij;
                }
            }
        }

        __syncthreads();
    } // end for bid
}

// Initializes matrix on the device.
template <typename T, typename I, typename Istride, typename UA>
inline void laset(rocblas_handle handle,
                  char const uplo_c,
                  I const m,
                  I const n,
                  T const alpha,
                  T const beta,

                  UA A_,
                  Istride const shiftA,
                  I const lda,
                  Istride const strideA,

                  I const batch_count)
{
    // quick return
    if(m <= 0 || n <= 0 || batch_count <= 0)
        return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_threads = LASET_MAX_THREADS;
    I const nx = (m <= 32) ? 32 : 64;
    I const ny = max_threads / nx;

    I const max_blocks = 1024;
    I const nbx = std::min(max_blocks, ceildiv(m, nx));
    I const nby = std::min(max_blocks, ceildiv(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    laset_kernel<T, I, Istride, UA>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(uplo_c, m, n, alpha, beta,

                                                              A_, shiftA, lda, strideA,

                                                              batch_count);
}

ROCSOLVER_END_NAMESPACE
