// todo: borrowed from Ed's CholQR branch.

/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once

#include <algorithm>
#include <cassert>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "lib_host_helpers.hpp"
#include "lib_macros.hpp"
#include "rocblas_utility.hpp"
#include "rocsolver_logger.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// -----------------------------
// Initialize matrix
// motivated by xLASET in LAPACK
//
// matrix A is m by n
//
// uplo == rocblas_fill_upper : assign to upper triangular matrix
// uplo == rocblas_fill_lower : assign to lower triangular matrix
// uplo == rocblas_fill_full : assign to entire matrix
//
// Assign offdiag to off-diagonal elements.
// Assign diag to diagonal elements.
//
// Thread block is (dimX, dimY), which can be arbitrary.
// Grid is (ceil( m / dimX ), ceil( n / dimY ), batch_count).
// -----------------------------

template <typename T, typename I, typename UA>
__global__ static void laset_kernel(const rocblas_fill uplo,
                                    const I m,
                                    const I n,
                                    const T offdiag,
                                    const T diag,
                                    UA AA,
                                    const rocblas_stride shiftA,
                                    const I lda,
                                    const rocblas_stride strideA,
                                    const I batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A = load_ptr_batch<T>(AA, bid, shiftA, strideA);

        if(uplo == rocblas_fill_lower)
        {
            // ---------------------------------
            // assign to lower triangular matrix
            // ---------------------------------
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = j + i_start; i < m; i += i_inc)
                {
                    bool const is_diagonal = (i == j);
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = is_diagonal ? diag : offdiag;

                    A[ij] = aij;
                }
            }
        }
        else if(uplo == rocblas_fill_upper)
        {
            // ---------------------------------
            // assign to upper triangular matrix
            // ---------------------------------
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < std::min(m, j + 1); i += i_inc)
                {
                    bool const is_diagonal = (i == j);
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = is_diagonal ? diag : offdiag;

                    A[ij] = aij;
                }
            }
        }
        else
        {
            // ------------------------
            // assign to entire matrix
            // ------------------------
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    bool const is_diagonal = (i == j);
                    auto const ij = idx2D(i, j, lda);
                    auto const aij = is_diagonal ? diag : offdiag;

                    A[ij] = aij;
                }
            }
        }
    }
}

template <typename T, typename I, typename UA>
void rocsolver_laset(rocblas_handle handle,
                     const rocblas_fill uplo,
                     const I m,
                     const I n,
                     const T offdiag,
                     const T diag,
                     UA AA,
                     const rocblas_stride shiftA,
                     const I lda,
                     const rocblas_stride strideA,
                     const I batch_count)
{
    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    dim3 blocks(ceildiv(m, BS2), ceildiv(n, BS2), batch_count);
    dim3 threads(BS2, BS2);
    ROCSOLVER_LAUNCH_KERNEL(laset_kernel<T>, blocks, threads, 0, stream, // kernel
                            uplo, m, n, offdiag, diag, // opts
                            AA, shiftA, lda, strideA, // A
                            batch_count);
}

ROCSOLVER_END_NAMESPACE
