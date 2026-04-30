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
            // Random on complex [-1, 1] x [-1, 1]i or real [-1, 1].
            A[i + j*lda] = rand_value<T>( dist );
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
