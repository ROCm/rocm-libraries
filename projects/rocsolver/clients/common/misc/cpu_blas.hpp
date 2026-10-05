/* **************************************************************************
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

#include "clients_utility.hpp"
#include "rocsolver_test.hpp"

// Implements BLAS-like routines on CPU needed for rocSolver testers.

//------------------------------------------------------------------------------
// A simple symmetric tridiagonal matrix-matrix multiply,
//      C := A B
// where B is an n-by-n real tridiagonal matrix,
// represented by diagonal D and sub/super-diagonal E,
// and A and C are real or complex m-by-n general matrices.
//
template <typename T, typename S = decltype(std::real(T{}))>
void cpu_stmm(rocblas_int m,
              rocblas_int n,
              T const* A,
              rocblas_int lda,
              S const* D,
              S const* E,
              T* C,
              rocblas_int ldc)
{
    assert(m >= 0);
    assert(n >= 0);
    assert(lda >= m);
    assert(ldc >= m);

    // Quick return
    if(m == 0 || n == 0)
        return;

    // Single vector
    if(n == 1)
    {
        for(rocblas_int i = 0; i < m; ++i)
        {
            C[i] = A[i] * D[0];
        }
        return;
    }

    // General case, n >= 2.
    for(rocblas_int j = 0; j < n; ++j)
    {
        if(j == 0)
        {
            // C[:,0] = A[:,0]*D[0] + A[:,1]*E[0]
            for(rocblas_int i = 0; i < m; ++i)
            {
                C[i + j * ldc] = A[i + j * lda] * D[j] + A[i + (j + 1) * lda] * E[j];
            }
        }
        else if(j == n - 1)
        {
            // C[:,n-1] = A[:,n-2]*E[n-2] + A[:,n-1]*D[n-1]
            for(rocblas_int i = 0; i < m; ++i)
            {
                C[i + j * ldc] = A[i + (j - 1) * lda] * E[j - 1] + A[i + j * lda] * D[j];
            }
        }
        else
        {
            // C[:,j] = A[:,j-1]*E[j-1] + A[:,j]*D[j] + A[:,j+1]*E[j]
            for(rocblas_int i = 0; i < m; ++i)
            {
                C[i + j * ldc] = A[i + (j - 1) * lda] * E[j - 1] + A[i + j * lda] * D[j]
                    + A[i + (j + 1) * lda] * E[j];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Add Hermitian band matrix A to general matrix C:
//      C = alpha A + beta C.
// A is an n-by-n Hermitian band matrix with bandwidth kd.
// C is an n-by-n general matrix.
//
// If lower:
//      [ d d d d d ]   main diagonal
//      [ 1 1 1 1 . ]   sub-diag 1
//      [ 2 2 2 . . ]   sub-diag 2
//
// If upper:
//      [     2 2 2 ]   super-diag 2
//      [   1 1 1 1 ]   super-diag 1
//      [ d d d d d ]   main diagonal
//
template <typename T>
void cpu_hbadd(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int kd,
               T alpha,
               T const* Aband,
               rocblas_int ldab,
               T beta,
               T* C,
               rocblas_int ldc)
{
    using std::imag, std::min, std::max;

    assert(n >= 0);
    assert(kd >= 0);
    assert(ldab >= kd + 1);
    assert(ldc >= n);
    assert(uplo == rocblas_fill_lower || uplo == rocblas_fill_upper);

    // Quick return
    if(n == 0)
        return;

    if(uplo == rocblas_fill_lower)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = j; i < min(j + kd + 1, n); ++i)
            {
                T Aij = Aband[i - j + j * ldab];
                if(i == j)
                    assert(imag(Aij) == 0);
                C[i + j * ldc] = alpha * Aij + beta * C[i + j * ldc];
                if(i != j)
                    C[j + i * ldc] = alpha * sconj(Aij) + beta * C[j + i * ldc];
            }
        }
    }
    else // upper
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = max(j - kd, 0); i <= j; ++i)
            {
                T Aij = Aband[kd - (i - j) + j * ldab];
                if(i == j)
                    assert(imag(Aij) == 0);
                C[i + j * ldc] = alpha * Aij + beta * C[i + j * ldc];
                if(i != j)
                    C[j + i * ldc] = alpha * sconj(Aij) + beta * C[j + i * ldc];
            }
        }
    }
}
