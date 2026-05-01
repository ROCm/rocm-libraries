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

#include "util.hpp"
#include "get_type_name.hpp"

#include <algorithm>
#include <complex>
#include <cassert>
#include <cmath>
#include <cstdio>

//==============================================================================
#define HIP
#ifdef HIP

#include <hip/hip_runtime.h>

//------------------------------------------------------------------------------
#define hip_call_( err, file, line ) \
    do { \
        hipError_t error = (err); \
        if (error) { \
            fprintf( stderr, "Failed: %s (error %d) at %s:%d\n", \
                     hipGetErrorString( error ), error, file, line ); \
            throw std::exception(); \
        } \
    } while(0)

#define hip_call( err ) \
        hip_call_( err, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
template <typename T, typename I>
void copy_matrix( I m, I n, T* src, I ld_src, T* dst, I ld_dst, hipStream_t stream )
{
    using llong = long long;

    printf( "# %s( m=%lld, n=%lld, src=%p, ld=%lld, dst=%p, ld=%lld, stream=%p )\n",
            __func__, llong(m), llong(n), src, llong(ld_src),
            dst, llong(ld_dst), stream );
    hip_call(
        hipMemcpy2DAsync(
            dst, ld_dst*sizeof(T),
            src, ld_src*sizeof(T),
            m*sizeof(T), n, hipMemcpyDefault, stream ) );
    hip_call(
        hipStreamSynchronize( stream ) );
}

//------------------------------------------------------------------------------
// todo: this ignores unified and managed memory types.
inline bool is_devptr( void* ptr )
{
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes( &attr, ptr );
    return err == hipSuccess && attr.type == hipMemoryTypeDevice;
}

//==============================================================================
#else  // not HIP

typedef void* hipStream_t;

#endif // not HIP
//==============================================================================


//------------------------------------------------------------------------------
class PrintOptions
{
public:
    PrintOptions( int w_in, int p_in ):
        w( std::max( w_in, p_in + 6 ) ),
        p( p_in )
    {
        f_hi = pow( 10, w - p - 2 ) - 0.5*pow( 10, -p );
    }

    int w, p;
    double f_hi;
};

//------------------------------------------------------------------------------
// Prints value as int, fixed, or exponent, depending on its value.
template <typename T>
void print_value( PrintOptions const& opts, T value )
{
    using std::real, std::imag;
    using S = decltype( real( T() ) );

    const int w = opts.w;
    const int p = opts.p;

    if constexpr (rocblas_is_complex<T>) {
        S re = std::abs( real( value ) );
        S im = std::abs( imag( value ) );
        if (! std::isnan(re) && re == int64_t(re) && im == 0) {
            // Medium integers print as int, padded to align with decimal point.
            printf( "%#*.0f%*s",
                    w-p, real( value ), w+p+4, "" );
        }
        else {
            // All other values print real and imag parts separately.
            print_value( opts, real( value ) );
            printf( " + " );
            print_value( opts, imag( value ) );
            printf( "i" );
        }
    }
    else {
        S re = std::abs( value );
        if (! (re < opts.f_hi)) {
            // Large numbers and NaN print as %e.
            // Note ! (x < y) matches NaN that x >= y would not.
            printf( "%*.*e",
                    w, p-1, value );
        }
        else if (re == int64_t(re)) {
            // Medium integers print as int, padded to align with decimal point.
            printf( "%#*.0f%*s",
                    w-p, value, p, "" );
        }
        else if (re > 1.) {
            // Medium values print with %f.
            printf( "%*.*f",
                    w, p, value );
        }
        else {
            // Small values print with %g.
            printf( "%#*.*g",
                    w, p, value );
        }
    }
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void print_matrix(
    std::string const& label, I m, I n, T* A, I lda, int p=3,
    hipStream_t stream=nullptr )
{
    using llong = long long;

    PrintOptions opts( p+6, p );

    T* hA;
    I ldha;
#ifdef HIP
    if (is_devptr( A )) {
        printf( "# copying device %s => host\n", label.c_str() );
        ldha = m;
        hA = new T[ ldha*n ];
        copy_matrix( m, n, A, lda, hA, ldha, stream );
    }
    else
#endif
    {
        ldha = lda;
        hA = A;
    }

    printf( "# %s %lld x %lld, ld %lld, %s\n"
            "%s = numpy.array([\n",
            label.c_str(), llong(m), llong(n), llong(lda),
            get_type_name<T>().c_str(), label.c_str() );
    for (I i = 0; i < m; ++i) {
        printf( "  [  " );
        for (I j = 0; j < n; ++j) {
            //printf( "  " );
            print_value( opts, hA[ i + j*ldha ] );
            printf( ",  " );
        }
        printf( "],\n" );
    }
    printf( "]);\n" );

    if (hA != A) {
        delete[] hA;
    }
}
