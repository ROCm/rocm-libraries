/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "print_matrix.hpp"

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename Ud, typename Uh>
void sb2st_hb2st_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          const rocblas_int kd,
                          Ud& dAband,
                          const rocblas_int ldab,
                          Uh& hAband)
{
    // TODO: how to handle uplo? Easiest would be to convert upper to lower.

    // For bandwidth kd, need ku = kd-1 superdiagonals to cover the diagonal
    // blocks. (ku superdiagonals needed if we update diag blocks using
    // gemv/ger, but none needed if we used hemv/her2.)
    // Need kl = 2*kd-1 subdiagonals to cover the off-diagonal blocks and bulges.
    rocblas_int ku = kd - 1;
    rocblas_int kl = 2*kd - 1;
    rocblas_int m  = ku + kl + 1;
    assert( ldab >= m );

    // Index of main diagonal.
    rocblas_int idiag = ku;

    if (CPU)
    {
        for (rocblas_int j = 0; j < n; ++j)
        {
            #if 0
                printf( "skipping init\n" );
                for (rocblas_int i = 0; i < ldab; ++i)
                {
                    hAband[0][i + j*ldab] = i + j/100.;
                }
            #else
                // Diagonal is real.
                // Random on [-1, 1].
                // todo: better random number generator.
                hAband[0][idiag + j*ldab] = 2*(rand() / S( RAND_MAX )) - 1;
                // j/100.;

                // Fill in lower band and copy conjugate to upper band.
                for (rocblas_int i = 1; i < kd + 1 && i + j < n; ++i)
                {
                    // Random on complex [-1, 1] x [-1, 1]i or real [-1, 1].
                    if constexpr (rocblas_is_complex<T>)
                    {
                        hAband[0][idiag + i + j*ldab]
                            = T( 2*(rand() / S( RAND_MAX )) - 1,
                                 2*(rand() / S( RAND_MAX )) - 1 );
                    }
                    else
                    {
                        hAband[0][idiag + i + j*ldab]
                            //= i + j/100.;
                            = 2*(rand() / S( RAND_MAX )) - 1;
                    }

                    #if 1
                        // Copy conj of lower band to upper band. The kd-th diagonal is
                        // not needed, as it is outside the kd x kd diagonal blocks.
                        if (i < kd)
                        {
                            hAband[0][idiag - i + (j + i)*ldab]
                                = conjugate( hAband[0][idiag + i + j*ldab] );
                        }
                    #endif
                }
                // Zero out entries outside band where bulges will fill in.
                for (rocblas_int i = kd+1; i < 2*kd; ++i)
                {
                    hAband[0][idiag + i + j*ldab] = 0;
                }
            #endif
        }

        #if 1
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
                    hAband[0][i + j*ldab] = nan( "" );
                }
            }
            // For lower band, work from right-most column (n-1) to left.
            for (rocblas_int j = 0; j < kl; ++j)
            {
                for (rocblas_int i = j; i < kl; ++i)
                {
                    hAband[0][idiag + 1 + i + (n - 1 - j)*ldab] = nan( "" );
                }
            }
        #endif
    }

    if (GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR( dAband.transfer_from( hAband ) );
    }
}

//------------------------------------------------------------------------------
template <typename T, typename Ud, typename Td, typename Uh, typename Th>
void sb2st_hb2st_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          const rocblas_int kd,
                          Ud& dAband,
                          const rocblas_int ldab,
                          Td& dD,
                          Td& dE,
                          Ud& dV,
                          const rocblas_int ldv,
                          Uh& hAband,
                          Uh& hAbandRes,
                          Th& hDRes,
                          Th& hERes,
                          Th& hW,
                          double* max_err)
{
    using S = decltype( std::real( T{} ) );
    using std::abs, std::imag, std::real, std::max;

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(
        rocblas_get_stream( handle, &stream ) );

    rocblas_int idiag = kd - 1;

    // input data initialization
    sb2st_hb2st_initData<true, true, T>(
        handle, uplo, n, kd, dAband, ldab, hAband );

    // execute computations
    // GPU lapack
    double start, time;
    start = get_time_us_sync(stream);
    CHECK_ROCBLAS_ERROR(
        rocsolver_sb2st_hb2st(
            handle, uplo, n, kd,
            dAband.data(), ldab,
            dD.data(), dE.data(),
            dV.data(), ldv ) );
    time = get_time_us_sync(stream) - start;
        CHECK_HIP_ERROR( hAbandRes.transfer_from( dAband ) );
    CHECK_HIP_ERROR( hDRes.transfer_from( dD ) );
    CHECK_HIP_ERROR( hERes.transfer_from( dE ) );

    S err = 0;
    S max_err_ = 0;
    if constexpr (rocblas_is_complex<T>)
    {
        // Check that diag is real.
        for (rocblas_int j = 0; j < n; ++j)
        {
            err = max( err, abs( imag( hAbandRes[0][idiag + j*ldab] ) ) );
        }
        printf( "imag( D ) error %9.3g\n", err );
        max_err_ = max( err, max_err_ );

        // Check that subdiag is real.
        err = 0;
        for (rocblas_int j = 0; j < n-1; ++j)
        {
            err = max( err, abs( imag( hAbandRes[0][idiag + 1 + j*ldab] ) ) );
        }
        max_err_ = max( err, max_err_ );
    }

    // Check that diag( A ) == D.
    err = 0;
    for (rocblas_int j = 0; j < n; ++j)
    {
        err += hDRes[0][j] != real( hAbandRes[0][idiag + j*ldab] );
    }
    max_err_ = max( err, max_err_ );

    // Check that diag( A, -1 ) == E.
    err = 0;
    for (rocblas_int j = 0; j < n-1; ++j)
    {
        err += hERes[0][j] != real( hAbandRes[0][idiag + 1 + j*ldab] );
    }
    printf( "diag( A, -1 ) == E error %9.3g\n", err );
    max_err_ = max( err, max_err_ );

    #if 1
        // Compute eigenvalues of tridiagonal matrix
        start = get_time_us_sync(stream);
        cpu_sterf( n, hDRes.data(), hERes.data() );
        time = get_time_us_sync(stream) - start;
        printf( "n %d, CPU sterf time %.4f\n", n, time );

        printf( "eig_rocsol = [" );
        for (int i = 0; i < std::min( 5, n ); ++i)
        {
            printf( "  %11.5g", hDRes[0][i] );
        }
        printf( "  ..." );
        for (int i = std::max( n-5, 5 ); i < n; ++i)
        {
            printf( "  %11.5g", hDRes[0][i] );
        }
        printf( " ];\n" );
    #endif

    #if 1
        // CPU lapack
        // Compute eigenvalues of banded matrix
        int info;
        int worksize = rocblas_is_complex<T> ? n : std::max( 1, 3*n - 2 );
        std::vector<T> work( worksize, T( 0 ) );
        int worksize_real = rocblas_is_complex<T> ? std::max( 1, 3*n - 2 ) : 0;
        std::vector<S> work_real( worksize_real, S( 0 ) );
        T dummy;  // for Z
        // todo: assuming lower
        start = get_time_us_sync(stream);
        cpu_sbev_hbev( rocblas_evect_none, uplo, n, kd, &hAband[0][idiag], ldab,
                       hW.data(), &dummy, 1,
                       work.data(), work_real.data(), &info );
        time = get_time_us_sync(stream) - start;
        printf( "n %d, kd %d, CPU hbev time %.4f\n", n, kd, time );

        printf( "eig_lapack = [" );
        for (int i = 0; i < std::min( 5, n ); ++i)
        {
            printf( "  %11.5g", hW[0][i] );
        }
        printf( "  ..." );
        for (int i = std::max( n-5, 5 ); i < n; ++i)
        {
            printf( "  %11.5g", hW[0][i] );
        }
        printf( " ];\n" );
    #endif

    // Compare CPU and GPU eigval results in hW and hDRes, respectively.
    err = norm_error( 'F', 1, n, 1, hW.data(), hDRes.data() );
    max_err_ = max( max_err_, err );
    if (std::isnan( err ))
        max_err_ = err;

    *max_err = max_err_;

    // todo: check orthogonality of V?
    // todo: check backwards error Q^H Aband - Atridiag or Aband - Q Atridiag?
    // cf. LAWN 41.
}

template <typename T, typename Td, typename Ud, typename Uh>
void sb2st_hb2st_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             const rocblas_int kd,
                             Ud& dAband,
                             const rocblas_int ldab,
                             Td& dD,
                             Td& dE,
                             Ud& dV,
                             const rocblas_int ldv,
                             Uh& hAband,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(
        rocblas_get_stream( handle, &stream ) );

    if (! perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = nan( "" );
    }

    sb2st_hb2st_initData<true, false, T>(
        handle, uplo, n, kd, dAband, ldab, hAband);

    // cold calls
    double start, time;
    for (int iter = 0; iter < 2; iter++)
    {
        sb2st_hb2st_initData<false, true, T>(
            handle, uplo, n, kd, dAband, ldab, hAband );

        start = get_time_us_sync(stream);
        CHECK_ROCBLAS_ERROR(
            rocsolver_sb2st_hb2st(
                handle, uplo, n, kd,
                dAband.data(), ldab,
                dD.data(), dE.data(),
                dV.data(), ldv ) );
        time = get_time_us_sync(stream) - start;
        printf( "n %d, kd %d, cold iter %d, time %.4f\n", n, kd, iter, time );
    }

    // gpu-lapack performance
    if (profile > 0)
    {
        if (profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode( rocblas_layer_mode_log_profile );
        rocsolver_log_set_max_levels( profile );
    }

    for (rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sb2st_hb2st_initData<false, true, T>(
            handle, uplo, n, kd, dAband, ldab, hAband);

        start = get_time_us_sync( stream );
        CHECK_ROCBLAS_ERROR(
            rocsolver_sb2st_hb2st(
                handle, uplo, n, kd,
                dAband.data(), ldab,
                dD.data(), dE.data(),
                dV.data(), ldv ) );
        time = get_time_us_sync(stream) - start;
        *gpu_time_used += time;
        printf( "n %d, kd %d, hot  iter %d, time %.4f\n", n, kd, iter, time );
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_sb2st_hb2st( Arguments& argus )
{
    using S = decltype( std::real( T{} ) );

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int kd = argus.get<rocblas_int>("kd");
    rocblas_int ldab = argus.get<rocblas_int>("ldab", 3*kd - 1);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", 3*kd);

    // V is ldv x nv
    // todo: if eigval only, don't need T, and only need V vectors and tau
    // for 2 rounds, so maybe 2(n+1) or so is enough.
    rocblas_int nt = ceildiv( n, kd );
    rocblas_int nv_blocks = nt*(nt + 1)/2;
    rocblas_int nv = nv_blocks*kd;

    rocblas_fill uplo = char2rocblas_fill( uploC );
    rocblas_int hot_calls = argus.iters;

    // determine sizes
    size_t size_Aband = ldab * n;
    size_t size_V = ldv * nv;
    size_t size_D = n;
    size_t size_E = n - 1;
    size_t size_W = size_D;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ares = (argus.unit_check || argus.norm_check) ? size_Aband : 0;
    size_t size_Dres = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_Eres = (argus.unit_check || argus.norm_check) ? size_E : 0;
    size_t size_Vres = 0;  // todo: not yet checked

    // check invalid sizes
    bool invalid_size = (n < 0 || kd < 0 || ldab < 3*kd - 1 || ldv < 3*kd);
    if (invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_sb2st_hb2st(
                handle, uplo, n, kd,
                (T*)nullptr, ldab,
                (S*)nullptr, (S*)nullptr,
                (T*)nullptr, ldv ),
            rocblas_status_invalid_size );

        if (argus.timing)
            rocsolver_bench_inform( inform_invalid_size );

        return;
    }

    // memory size query is necessary
    if (argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(
            rocblas_start_device_memory_size_query( handle ) );

        CHECK_ALLOC_QUERY(
            rocsolver_sb2st_hb2st(
                handle, uplo, n, kd,
                (T*)nullptr, ldab,
                (S*)nullptr, (S*)nullptr,
                (T*)nullptr, ldv ) );

        size_t size;
        CHECK_ROCBLAS_ERROR(
            rocblas_stop_device_memory_size_query( handle, &size ) );

        rocsolver_bench_inform( inform_mem_query, size );
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hAband( size_Aband, 1, size_Aband, 1 );
    host_strided_batch_vector<S> hW( size_W, 1, size_W, 1 );
    host_strided_batch_vector<T> hAbandRes( size_Ares, 1, size_Ares, 1 );
    host_strided_batch_vector<S> hDRes( size_Dres, 1, size_Dres, 1 );
    host_strided_batch_vector<S> hERes( size_Eres, 1, size_Eres, 1 );

    device_strided_batch_vector<T> dAband( size_Aband, 1, size_Aband, 1 );
    device_strided_batch_vector<T> dV( size_V, 1, size_V, 1 );
    device_strided_batch_vector<S> dD( size_D, 1, size_D, 1 );
    device_strided_batch_vector<S> dE( size_E, 1, size_E, 1 );

    if (size_Aband)
        CHECK_HIP_ERROR( dAband.memcheck() );
    if (size_V)
        CHECK_HIP_ERROR( dV.memcheck() );
    if (size_D)
        CHECK_HIP_ERROR( dD.memcheck() );
    if (size_E)
        CHECK_HIP_ERROR( dE.memcheck() );

    // check quick return
    if (kd == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_sb2st_hb2st(
                handle, uplo, n, kd,
                dAband.data(), ldab,
                dD.data(), dE.data(),
                dV.data(), ldv ),
            rocblas_status_success );
        if (argus.timing)
            rocsolver_bench_inform( inform_quick_return );
        return;
    }

    // check computations
    if (argus.unit_check || argus.norm_check)
    {
        sb2st_hb2st_getError<T>(
            handle, uplo, n, kd,
            dAband, ldab,
            dD, dE,
            dV, ldv,
            hAband, hAbandRes, hDRes, hERes, hW,
            &max_error );
    }

    // collect performance data
    if (argus.timing && hot_calls > 0)
    {
        sb2st_hb2st_getPerfData<T>(
            handle, uplo, n, kd,
            dAband, ldab,
            dD, dE,
            dV, ldv,
            hAband,
            &gpu_time_used, &cpu_time_used,
            hot_calls, argus.profile, argus.profile_kernels, argus.perf );
    }

    // validate results for rocsolver-test
    if (argus.unit_check)
        ROCSOLVER_TEST_CHECK( T, max_error, n );

    // output results for rocsolver-bench
    if (argus.timing)
    {
        if (! argus.perf)
        {
            rocsolver_bench_header( "Arguments:" );
            rocsolver_bench_output( "uplo", "n", "kd", "ldab", "ldv" );
            rocsolver_bench_output( uplo, n, kd, ldab, ldv );
            rocsolver_bench_header( "Results:" );
            if (argus.norm_check)
            {
                rocsolver_bench_output( "cpu_time_us", "gpu_time_us", "error" );
                rocsolver_bench_output( cpu_time_used, gpu_time_used, max_error );
            }
            else
            {
                rocsolver_bench_output( "cpu_time_us", "gpu_time_us" );
                rocsolver_bench_output( cpu_time_used, gpu_time_used );
            }
            rocsolver_bench_endl();
        }
        else
        {
            if (argus.norm_check)
                rocsolver_bench_output( gpu_time_used, max_error );
            else
                rocsolver_bench_output( gpu_time_used );
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_SB2ST_HB2ST( ... ) \
    extern template void testing_sb2st_hb2st<__VA_ARGS__>( Arguments& );

INSTANTIATE( EXTERN_TESTING_SB2ST_HB2ST, FOREACH_SCALAR_TYPE, APPLY_STAMP )
