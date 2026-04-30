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

#include <random>

//------------------------------------------------------------------------------
// returns random real value in distribution dist,
// or complex value in dist x dist*i.
//
template <typename T, typename DistType>
T rand_value( DistType& dist )
{
    if constexpr (rocblas_is_complex<T>)
    {
        return T( dist( rocblas_rng ),
                  dist( rocblas_rng ) );
    }
    else
    {
        return dist( rocblas_rng );
    }
}

//------------------------------------------------------------------------------
// Fill in general m-by-n matrix A with random uniform values on unit square.
//
// todo: pass dist into routine, with default = uniform [-1, 1]?
//
template <typename T>
void gerand( rocblas_int m, rocblas_int n, T* A, rocblas_int lda )
{
    using S = decltype( std::real( T{} ) );
    std::uniform_real_distribution<S> dist( S(-1), S(1) );

    assert( m >= 0 );
    assert( n >= 0 );
    assert( lda >= m );

    for (rocblas_int j = 0; j < n; ++j)
    {
        for (rocblas_int i = 0; i < m; ++i)
        {
            A[i + j*lda] = rand_value<T>( dist );
        }
    }
}

//------------------------------------------------------------------------------
// Fill in n-by-n Hermitian matrix A with random uniform values on unit square.
// In complex, the diagonal is real.
// If uplo == rocblas_fill_lower, only lower triangle is set;
// if uplo == rocblas_fill_upper, only upper triangle is set;
// if uplo == rocblas_fill_full, both lower and upper triangles are set.
//
// todo: pass dist into routine, with default = uniform [-1, 1]?
//
template <typename T>
void herand( rocblas_fill uplo, rocblas_int n, T* A, rocblas_int lda )
{
    using S = decltype( std::real( T{} ) );
    std::uniform_real_distribution<S> dist( S(-1), S(1) );

    using foo::conjugate;  // todo

    assert( n >= 0 );
    assert( lda >= n );

    if (uplo == rocblas_fill_lower)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            A[ j + j*lda ] = rand_value<S>( dist );  // diagonal real
            for (rocblas_int i = j+1; i < n; ++i)  // strictly lower
            {
                A[i + j*lda] = rand_value<T>( dist );
            }
        }
    }
    else if (uplo == rocblas_fill_upper)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            for (rocblas_int i = 0; i < j; ++i)  // strictly upper
            {
                A[i + j*lda] = rand_value<T>( dist );
            }
            A[ j + j*lda ] = rand_value<S>( dist );  // diagonal real
        }
    }
    else if (uplo == rocblas_fill_full)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            A[ j + j*lda ] = rand_value<S>( dist );  // diagonal real
            for (rocblas_int i = j+1; i < n; ++i)  // strictly lower
            {
                A[i + j*lda] = rand_value<T>( dist );
                A[j + i*lda] = conjugate( A[i + j*lda] );
            }
        }
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
// todo: pass dist into routine, with default = uniform [-1, 1]?
//
template <typename T>
void syrand( rocblas_fill uplo, rocblas_int n, T* A, rocblas_int lda )
{
    using S = decltype( std::real( T{} ) );
    std::uniform_real_distribution<S> dist( S(-1), S(1) );

    using foo::conjugate;  // todo

    assert( n >= 0 );
    assert( lda >= n );

    if (uplo == rocblas_fill_lower)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            for (rocblas_int i = j; i < n; ++i)  // lower
            {
                A[i + j*lda] = rand_value<T>( dist );
            }
        }
    }
    else if (uplo == rocblas_fill_upper)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            for (rocblas_int i = 0; i <= j; ++i)  // upper
            {
                A[i + j*lda] = rand_value<T>( dist );
            }
        }
    }
    else if (uplo == rocblas_fill_full)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            A[ j + j*lda ] = rand_value<T>( dist );  // diagonal
            for (rocblas_int i = j+1; i < n; ++i)  // strictly lower
            {
                A[i + j*lda] = rand_value<T>( dist );
                A[j + i*lda] = A[i + j*lda];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Fill in Hermitian band matrix Aband with random uniform values on unit square.
// The diagonal is real. Entries outside the band are set to nan.
// As a special case for hb2st, this stores the full lower band and copies
// conjugate of part of it to the upper band.
//
// todo: pass dist into routine, with default = uniform [-1, 1]?
//
template <typename T>
void hbrand( rocblas_int n, rocblas_int kd,
             T* Aband, rocblas_int ldab )
{
    using S = decltype( std::real( T{} ) );
    std::uniform_real_distribution<S> dist( S(-1), S(1) );

    using foo::conjugate;  // todo

    assert( n >= 0 );
    assert( kd >= 0 );

    // For bandwidth kd, need ku = kd-1 superdiagonals to cover the diagonal
    // blocks. (ku superdiagonals needed if we update diag blocks using
    // gemv/ger, but none needed if we used hemv/her2.)
    // Need kl = 2*kd-1 subdiagonals to cover the off-diagonal blocks and bulges.
    rocblas_int ku = kd - 1;
    rocblas_int kl = 2*kd - 1;
    assert( ldab >= ku + kl + 1 );

    // Index of main diagonal.
    rocblas_int idiag = ku;

    for (rocblas_int j = 0; j < n; ++j)
    {
        // Diagonal is real.
        // Random on [-1, 1].
        // todo: better random number generator.
        Aband[idiag + j*ldab] = rand_value<S>( dist );

        // Fill in lower band and copy conjugate to upper band.
        for (rocblas_int i = 1; i < kd + 1 && i + j < n; ++i)
        {
            // Random on complex [-1, 1] x [-1, 1]i or real [-1, 1].
            Aband[idiag + i + j*ldab] = rand_value<T>( dist );

            // Copy conj of lower band to upper band. The kd-th diagonal is
            // not needed, as it is outside the kd x kd diagonal blocks.
            if (i < kd)
            {
                Aband[idiag - i + (j + i)*ldab]
                    = conjugate( Aband[idiag + i + j*ldab] );
            }
        }
        // Zero out entries outside band where bulges will fill in.
        for (rocblas_int i = kd+1; i < 2*kd; ++i)
        {
            Aband[idiag + i + j*ldab] = 0;
        }
    }

    // Mark entries outside the band structure as nan,
    // to ensure we don't use them.
    // Example with n = 6, ku = 2, kl = 2, set x = nan:
    // [ x x . . . . ] }
    // [ x . . . . . ] } ku = 2
    // [ . . . . . . ] <= main diag
    // [ . . . . . x ] } kl = 2
    // [ . . . . x x ] }
    for (rocblas_int j = 0; j < ku; ++j)
    {
        for (rocblas_int i = 0; i < ku - j; ++i)
        {
            Aband[i + j*ldab] = nan( "" );
        }
    }
    // For lower band, work from right-most column (n-1) to left.
    for (rocblas_int j = 0; j < kl; ++j)
    {
        for (rocblas_int i = j; i < kl; ++i)
        {
            Aband[idiag + 1 + i + (n - 1 - j)*ldab] = nan( "" );
        }
    }
}
