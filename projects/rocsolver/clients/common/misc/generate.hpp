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

#include "common/misc/rocblas_random.hpp"
#include "common/misc/rocsolver_test.hpp"

#include <limits>
#include <random>

//------------------------------------------------------------------------------
// returns random real value in distribution dist,
// or complex value in dist x dist*i.
//
template <typename T, typename DistType>
T rand_value(DistType& dist)
{
    if constexpr(rocblas_is_complex<T>)
    {
        return T(dist(rocblas_rng), dist(rocblas_rng));
    }
    else
    {
        return dist(rocblas_rng);
    }
}

//------------------------------------------------------------------------------
// Fill in general m-by-n matrix A with random values drawn from dist.
// Default distribution is uniform on [-1, 1).
//
template <typename T,
          typename S = decltype(std::real(T{})),
          typename DistType = std::uniform_real_distribution<S>>
void gerand(rocblas_int m,
            rocblas_int n,
            T* A,
            rocblas_int lda,
            bool seed_reset = false,
            DistType dist = DistType(S(-1), S(1)))
{
    if(m < 0 || n < 0 || lda < m)
        throw rocblas_status_invalid_size;
    if(m && n && !A)
        throw rocblas_status_invalid_pointer;

    if(seed_reset)
        rocblas_seedrand();

    for(rocblas_int j = 0; j < n; ++j)
    {
        for(rocblas_int i = 0; i < m; ++i)
        {
            A[i + j * lda] = rand_value<T>(dist);
        }
    }
}

//------------------------------------------------------------------------------
// Fill in n-by-n Hermitian matrix A with random uniform values on unit square.
// In complex, the diagonal is real.
// If uplo == rocblas_fill_lower, only lower triangle is set;
// if uplo == rocblas_fill_upper, only upper triangle is set;
// if uplo == rocblas_fill_full, both lower and upper triangles are set,
// with upper being conjugate-transpose of lower.
//
template <typename T,
          typename S = decltype(std::real(T{})),
          typename DistType = std::uniform_real_distribution<S>>
void herand(rocblas_fill uplo,
            rocblas_int n,
            T* A,
            rocblas_int lda,
            bool seed_reset = false,
            DistType dist = DistType(S(-1), S(1)))
{
    if(n < 0 || lda < n)
        throw rocblas_status_invalid_size;
    if(n && !A)
        throw rocblas_status_invalid_pointer;

    if(seed_reset)
        rocblas_seedrand();

    if(uplo == rocblas_fill_lower)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            A[j + j * lda] = rand_value<S>(dist); // diagonal real
            for(rocblas_int i = j + 1; i < n; ++i) // strictly lower
            {
                A[i + j * lda] = rand_value<T>(dist);
            }
        }
    }
    else if(uplo == rocblas_fill_upper)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = 0; i < j; ++i) // strictly upper
            {
                A[i + j * lda] = rand_value<T>(dist);
            }
            A[j + j * lda] = rand_value<S>(dist); // diagonal real
        }
    }
    else if(uplo == rocblas_fill_full)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            A[j + j * lda] = rand_value<S>(dist); // diagonal real
            for(rocblas_int i = j + 1; i < n; ++i) // strictly lower
            {
                A[i + j * lda] = rand_value<T>(dist);
                A[j + i * lda] = sconj(A[i + j * lda]);
            }
        }
    }
    else
    {
        throw rocblas_status_invalid_value;
    }
}

//------------------------------------------------------------------------------
// Fill in n-by-n symmetric matrix A with random uniform values on unit square.
// In complex, this makes a complex-symmetric matrix with complex diagonal,
// not a Hermitian matrix.
// If uplo == rocblas_fill_lower, only lower triangle is set;
// if uplo == rocblas_fill_upper, only upper triangle is set;
// if uplo == rocblas_fill_full, both lower and upper triangles are set.
//
template <typename T,
          typename S = decltype(std::real(T{})),
          typename DistType = std::uniform_real_distribution<S>>
void syrand(rocblas_fill uplo,
            rocblas_int n,
            T* A,
            rocblas_int lda,
            bool seed_reset = false,
            DistType dist = DistType(S(-1), S(1)))
{
    if(n < 0 || lda < n)
        throw rocblas_status_invalid_size;
    if(n && !A)
        throw rocblas_status_invalid_pointer;

    if(seed_reset)
        rocblas_seedrand();

    if(uplo == rocblas_fill_lower)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = j; i < n; ++i) // lower
            {
                A[i + j * lda] = rand_value<T>(dist);
            }
        }
    }
    else if(uplo == rocblas_fill_upper)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = 0; i <= j; ++i) // upper
            {
                A[i + j * lda] = rand_value<T>(dist);
            }
        }
    }
    else if(uplo == rocblas_fill_full)
    {
        for(rocblas_int j = 0; j < n; ++j)
        {
            A[j + j * lda] = rand_value<T>(dist); // diagonal
            for(rocblas_int i = j + 1; i < n; ++i) // strictly lower
            {
                A[i + j * lda] = rand_value<T>(dist);
                A[j + i * lda] = A[i + j * lda];
            }
        }
    }
    else
    {
        throw rocblas_status_invalid_value;
    }
}

//------------------------------------------------------------------------------
// Fill in Hermitian band matrix Aband with random uniform values on unit
// square, and optionally symmetrize by copying min( kl, ku ) diagonals between
// lower and upper band. To make a fully explicitly symmetrized matrix,
// set kl = ku. To make an implicitly Hermitian matrix, as in LAPACK,
// set kl = 0 or ku = 0. The diagonal is real. Entries outside the matrix are
// set to nan. Example with n = 6, ku = 2, kl = 3, set `.` entries to nan:
//
//      [ . . b b b b ]  }  ku = 2
//      [ . a a a a a ]  }
//      [ d d d d d d ]  <= main diag
//      [ a a a a a . ]  }  kl = 3
//      [ b b b b . . ]  }
//      [ c c c . . . ]  }  Because ku < kl, this diagonal is not copied to upper band.
//
template <typename T,
          typename S = decltype(std::real(T{})),
          typename DistType = std::uniform_real_distribution<S>>
void hbrand(rocblas_int n,
            rocblas_int kl,
            rocblas_int ku,
            T* Aband,
            rocblas_int ldab,
            bool seed_reset = false,
            DistType dist = DistType(S(-1), S(1)))
{
    T const nan = T(std::numeric_limits<S>::quiet_NaN());

    if(n < 0 || kl < 0 || ku < 0 || ldab < kl + ku + 1)
        throw rocblas_status_invalid_size;
    if(n && !Aband)
        throw rocblas_status_invalid_pointer;

    if(seed_reset)
        rocblas_seedrand();

    // Index of main diagonal.
    rocblas_int idiag = ku;

    for(rocblas_int j = 0; j < n; ++j)
    {
        // Diagonal is real.
        // Random on [-1, 1].
        Aband[idiag + j * ldab] = rand_value<S>(dist);

        if(kl >= ku)
        {
            // Fill in lower band and copy conjugate to upper band.
            // k is index of diagonal, k > 0 is a subdiagonal, k < 0 is a superdiagonal.
            // idiag+k is row index in Aband array. A[ i, j ] is Aband[ idiag+j+k, j ].
            for(rocblas_int k = 1; k < kl + 1 && k + j < n; ++k)
            {
                // Random on complex [-1, 1] x [-1, 1]i or real [-1, 1].
                Aband[idiag + k + j * ldab] = rand_value<T>(dist);

                // Within the requested ku bandwidth, copy conj of lower band
                // to upper band, A{j, j+k} = conj( A{j+k, j} ).
                if(k <= ku)
                {
                    Aband[idiag - k + (j + k) * ldab] = sconj(Aband[idiag + k + j * ldab]);
                }
            }
        }
        else // kl < ku
        {
            // Fill in upper band and copy conjugate to lower band.
            for(rocblas_int k = 1; k < ku + 1 && k + j < n; ++k)
            {
                // Random on complex [-1, 1] x [-1, 1]i or real [-1, 1].
                Aband[idiag - k + (j + k) * ldab] = rand_value<T>(dist);

                // Within the requested kl bandwidth, copy conj of upper band
                // to lower band, A{j, j+k} = conj( A{j+k, j} ).
                if(k <= kl)
                {
                    Aband[idiag + k + j * ldab] = sconj(Aband[idiag - k + (j + k) * ldab]);
                }
            }
        }

        // Mark entries outside the band as nan, to ensure we don't use them.
        // Upper band
        if(j < ku)
        {
            for(rocblas_int k = 0; k < ku - j; ++k)
            {
                Aband[k + j * ldab] = nan;
            }
        }

        // Lower band
        if(j > n - 1 - kl)
        {
            for(rocblas_int k = n - 1 - j; k < kl; ++k)
            {
                Aband[idiag + 1 + k + j * ldab] = nan;
            }
        }
    }
}
