/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/misc/generate.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocblas.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"
#include "rocblas_utility.hpp"

//------------------------------------------------------------------------------
// Hermitianize A, on host, copying conjugate of strictly lower triangle to
// upper, or vice-versa.
// A is n-by-n.
template <typename T, typename I, typename Th>
void cpu_hermitianize(const rocblas_fill uplo,
                      const I n,
                      Th hA,
                      const I lda)
{
    if (uplo == rocblas_fill_lower)
    {
        for (I j = 0; j < n; ++j)
        {
            for (I i = 0; i < j; ++i) // strictly upper
            {
                hA[ i + j*lda ] = sconj( hA[ j + i*lda ] );
            }
        }
    }
    else if (uplo == rocblas_fill_upper)
    {
        for (I j = 0; j < n; ++j)
        {
            for (I i = 0; i < j; ++i) // strictly upper
            {
                hA[ j + i*lda ] = sconj( hA[ i + j*lda ] );
            }
        }
    }
    else
    {
        throw std::runtime_error( "unknown uplo" );
    }
}

//------------------------------------------------------------------------------
// Copy Aband => A, on host.
// Aband is n-by-n, bandwidth kd; currently assumes lower band.
// A is n-by-n.
template <typename T, typename I, typename Th>
void cpu_band2full(const rocblas_fill uplo,
                   const I n,
                   const I kd,
                   Th hAband,
                   const I ldab,
                   Th hA,
                   const I lda)
{
    if (uplo == rocblas_fill_lower)
    {
        // Set A = 0.
        T zero = 0;
        cpu_laset( 'f', n, n, zero, zero, hA, lda );

        // Example with n = 5, kd = 2:
        // Aband = [ 1  4  7 | a  c ]  =>  A = [ 1       |      ]
        //         [ 2  5  8 | b  * ]          [ 2  4    |      ]
        //         [ 3  6  9 | *  * ]          [ 3  5  7 |      ]
        //           \_____/   \__/            [    6  8 | a    ]
        //              R       T              [       9 | b  c ]
        //
        // Copy (kd + 1)-by-(n - kd) rectangle R, using lda + 1 to shift columns.
        cpu_lacpy( rocblas_fill_full, kd + 1, n - kd, hAband, ldab, hA, lda + 1 );

        // Copy kd-by-kd lower triangle T, using ldab - 1 to shift columns.
        cpu_lacpy( rocblas_fill_lower, kd, kd,  // opts
                   &hAband[ ldab*(n - kd) ], ldab - 1,  // src
                   &hA[ (lda + 1)*(n - kd) ], lda ); // dst

        // Hermitianize matrix A, copying conjugate of strictly lower to upper.
        // todo: restrict to band.
        for (I j = 0; j < n; ++j)
        {
            for (I i = 0; i < j; ++i) // strictly upper
            {
                hA[ i + j*lda ] = sconj( hA[ j + i*lda ] );
            }
        }
    }
    else
    {
        throw std::runtime_error( "upper not yet implemented" );
    }
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void sy2sb_he2hb_checkBadArgs(const rocblas_handle handle,
                              const rocblas_fill uplo,
                              const I n,
                              const I kd,
                              const I nb,
                              T dA,
                              const I lda,
                              T dAband,
                              const I ldab,
                              T dTau)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(nullptr, uplo, n, kd, nb, dA, lda, dAband, ldab, dTau),
                          rocblas_status_invalid_handle);

    // invalid value
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, static_cast<rocblas_fill>(0), n, kd, nb, dA,
                                                lda, dAband, ldab, dTau),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, (T) nullptr, lda, dAband, ldab, dTau),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, dA, lda, (T) nullptr, ldab, dTau),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, dA, lda, dAband, ldab, (T) nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, I(0), kd, nb, (T) nullptr, lda,
                                                (T) nullptr, ldab, (T) nullptr),
                          rocblas_status_success);
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_sy2sb_he2hb_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_lower;
    I n = 1;
    I kd = 1;
    I nb = 1;
    I lda = 1;
    I ldab = 3;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dAband(ldab, 1, ldab, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dAband.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());

#ifndef ROCSOLVER_ENABLE_EIG_2STAGE
    // he2hb is gated behind ROCSOLVER_ENABLE_EIG_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, dA.data(), lda,
                                                dAband.data(), ldab, dTau.data()),
                          rocblas_status_not_implemented);
    return;
#endif

    // check bad arguments
    sy2sb_he2hb_checkBadArgs(handle, uplo, n, kd, nb, dA.data(), lda, dAband.data(), ldab,
                             dTau.data());
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const I n,
                          Td& dA,
                          const I lda,
                          Th& hA)
{
    if(CPU)
    {
        herand(uplo, n, hA[0], lda, true);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

//------------------------------------------------------------------------------
// n        -- matrix dimension
// kd       -- desired bandwidth
// nb       -- outer blocksize to use
//
// dA       -- matrix on GPU, lda-by-n, lda >= n
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 3 kd - 1 (size needed for 2nd stage)
// dTau     -- output vector on GPU, length n-kd
//
// hARes    -- output matrix on CPU to copy GPU result, lda-by-n, lda >= n
// hAbandRes-- output band matrix on CPU to copy GPU result, ldab-by-n
// hTauRes  -- output vector on CPU to copy GPU result, length n-kd
//
// hA       -- matrix on CPU, lda-by-n, lda >= n
// hAband   -- output band matrix on CPU, ldab-by-n, ldab >= kd+1
// hTau     -- output vector on CPU, length n-kd
//
template <typename T, typename I, typename Td, typename Th>
void sy2sb_he2hb_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const I n,
                          const I kd,
                          const I nb,

                          Td& dA,
                          const I lda,
                          Td& dAband,
                          const I ldab,
                          Td& dTau,

                          Th& hARes,
                          Th& hAbandRes,
                          Th& hTauRes,

                          Th& hA,
                          Th& hAband,
                          Th& hTau,
                          double max_errors[2])
{
    using S = decltype(std::real(T{}));

    T const one = 1;
    T const neg_one = -1;
    T const zero = 0;
    I const idiag = kd - 1; // row of Aband storing diagonal

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    size_t size_A = lda * n;
    device_strided_batch_vector<T> dV(size_A, 1, size_A, 1); // Householder vectors
    device_strided_batch_vector<T> dW(size_A, 1, size_A, 1); // work
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());

    // Scalar for lange output, shared by all checks below.
    host_strided_batch_vector<S> hnorm(1, 1, 1, 1);
    device_strided_batch_vector<S> dnorm(1, 1, 1, 1);
    CHECK_HIP_ERROR(dnorm.memcheck());

    // input data initialization
    sy2sb_he2hb_initData<true, true, T, I>(handle, uplo, n, dV, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                              dV.data(), lda, // A input, V output
                                              dAband.data(), ldab, // Aband
                                              dTau.data())); // tau

    //--------------------
    // Check 0: Backward error: norm( Q*Aband*Q^H - A ) / (n * norm( A )), using 1-norm.

    // Hermitianize A, since we have only rocsolver_lange, not lanhe.
    cpu_hermitianize<T>( uplo, n, hA[0], lda );
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    // W := full( Aband )
    // Currently, he2hb always outputs lower band, regardless of uplo.
    // todo: Conversion currently on CPU. Need GPU lacpy and hermitianize routines.
    CHECK_HIP_ERROR(hAbandRes.transfer_from(dAband));
    cpu_band2full<T>( rocblas_fill_lower, n, kd, &hAbandRes[0][idiag], ldab, hARes[0], lda );
    CHECK_HIP_ERROR(dW.transfer_from(hARes));

    // W := Q W = Q Aband. Updates rows kd:n.
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormxr_unmxr(
            true, handle, rocblas_side_left, rocblas_operation_none, // opts
            n - kd, n, n - kd, // dims
            &dV[0][kd], lda, dTau[0], // Householder V vectors below bandwidth kd
            &dW[0][kd], lda));

    // W := W Q^H = (Q Aband) Q^H. Updates cols kd:n.
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormxr_unmxr(
            true, handle, rocblas_side_right, rocblas_operation_conjugate_transpose, // opts
            n, n - kd, n - kd, // dims
            &dV[0][kd], lda, dTau[0], // Householder V vectors below bandwidth kd
            &dW[0][kd*lda], lda));

    // W -= A.
    // todo: geadd or tzadd. Using axpy assumes dA and dW have the same lda.f
    CHECK_ROCBLAS_ERROR(
        rocblas_axpy( handle, lda*n, &neg_one, dA[0], I(1), dW[0], I(1) ));

    // err = norm( W ) / (n * norm( A )
    //     = norm( Q Aband Q^H - A ) / (n * norm( A ))
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange( handle, rocsolver_norm_type_one, n, n, dA[0], lda, dnorm ));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    S Anorm = hnorm[0][0];

    CHECK_ROCBLAS_ERROR(
        rocsolver_lange( handle, rocsolver_norm_type_one, n, n, dW[0], lda, dnorm ));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    max_errors[0] = hnorm[0][0] / n;
    if (Anorm != 0)
        max_errors[0] /= Anorm;

    //--------------------
    // Check 1: Orthogonality: norm( I - Q^H * Q ) / (n - kd), using 1-norm.
    // Q is generated explicitly via ungqr from the GPU factored output.
    // Q is (n - kd)-by-(n - kd), below bandwidth kd.

    // Generate Q, overwriting V.
    CHECK_ROCBLAS_ERROR(
        rocsolver_orgxr_ungxr(
            true, handle, n - kd, n - kd, n - kd, &dV[0][kd], lda, dTau[0] ));

    // W = I.
    CHECK_ROCBLAS_ERROR(
        rocsolver_laset( handle, rocblas_fill_full, n - kd, n - kd, zero, one, dW[0], lda ));

    // W = I - Q^H Q, with Q in V.
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm(
            handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, // opts
            n - kd, n - kd, n - kd, // dims
            &neg_one, &dV[0][kd], lda, // Q^H
            &dV[0][kd], lda, // Q
            &one, dW[0], lda ));

    CHECK_ROCBLAS_ERROR(
        rocsolver_lange( handle, rocsolver_norm_type_one, n - kd, n - kd, dW[0], lda, dnorm ));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    max_errors[1] = hnorm[0][0] / (n - kd);
}

//------------------------------------------------------------------------------
// n        -- matrix dimension
// kd       -- desired bandwidth
// nb       -- outer blocksize to use
// dA       -- matrix on GPU, lda-by-n, lda >= n
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 3 kd - 1 (size needed for 2nd stage)
// hA       -- matrix on CPU, lda-by-n, lda >= n
template <typename T, typename I, typename Td, typename Th>
void sy2sb_he2hb_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const I n,
                             const I kd,
                             const I nb,
                             Td& dA,
                             const I lda,
                             Td& dAband,
                             const I ldab,
                             Td& dTau,
                             Th& hA,
                             Th& hAband,
                             Th& hTau,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    rocsolver_timer timer;
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    sy2sb_he2hb_initData<true, false, T, I>(handle, uplo, n, dA, lda, hA);

    if(!perf)
    {
        // lwork for LAPACK hetrd_he2hb
        size_t lwork = n * kd + n * std::max<I>(kd, 128) + 2 * kd * kd;
        std::vector<T> hwork(lwork);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_sy2sb_he2hb(rocblas_fill_lower, n, kd, hA[0], lda, hAband[0], ldab, hTau[0],
                        hwork.data(), lwork);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sy2sb_he2hb_initData<false, true, T, I>(handle, uplo, n, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                                  dA.data(), lda, // A
                                                  dAband.data(), ldab, // Aband
                                                  dTau.data())); // tau
    }

    // gpu-lapack performance
    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sy2sb_he2hb_initData<false, true, T, I>(handle, uplo, n, dA, lda, hA);

        timer.start(stream);
        rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                              dA.data(), lda, // A
                              dAband.data(), ldab, // Aband
                              dTau.data()); // tau
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_sy2sb_he2hb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_fill uplo = char2rocblas_fill(uploC);
    I n = argus.get<I>("n");
    I kd = argus.get<I>("kd", 1);
    I nb = argus.get<I>("nb", kd);
    I lda = argus.get<I>("lda", n);
    // rocSolver 2nd stage hb2st needs 3*kd - 1. LAPACK needs kd+1.
    I ldab = 3 * kd - 1;

    rocblas_int hot_calls = argus.iters;

#ifndef ROCSOLVER_ENABLE_EIG_2STAGE
    // he2hb is gated behind ROCSOLVER_ENABLE_EIG_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                                (T*)nullptr, lda, // A
                                                (T*)nullptr, ldab, // Aband
                                                (T*)nullptr), // tau
                          rocblas_status_not_implemented);
    return;
#endif

    // check non-supported values
    bool invalid_uplo
        = (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper && uplo != rocblas_fill_full);
    if(invalid_uplo)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                                    (T*)nullptr, lda, // A
                                                    (T*)nullptr, ldab, // Aband
                                                    (T*)nullptr), // tau
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = lda * n;
    size_t size_Aband = ldab * n;
    size_t size_tau = std::max<I>(n - kd, 0);
    double max_errors[2] = { 0, 0 };
    double gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_AbandRes = (argus.unit_check || argus.norm_check) ? size_Aband : 0;
    size_t size_tauRes = (argus.unit_check || argus.norm_check) ? size_tau : 0;

    // check invalid sizes
    bool invalid_size
        = (n < 0 || kd < 1 || nb < 1 || nb < kd || nb % kd != 0 || lda < n || ldab < 3 * kd - 1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                                    (T*)nullptr, lda, // A
                                                    (T*)nullptr, ldab, // Aband
                                                    (T*)nullptr), // tau
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                                (T*)nullptr, lda, // A
                                                (T*)nullptr, ldab, // Aband
                                                (T*)nullptr)); // tau

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAband(size_Aband, 1, size_Aband, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);

    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hAbandRes(size_AbandRes, 1, size_AbandRes, 1);
    host_strided_batch_vector<T> hTauRes(size_tauRes, 1, size_tauRes, 1);

    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dAband(size_Aband, 1, size_Aband, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_Aband)
        CHECK_HIP_ERROR(dAband.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, kd, nb, // opts
                                                    dA.data(), lda, // A
                                                    dAband.data(), ldab, // Aband
                                                    dTau.data()), // tau
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        sy2sb_he2hb_getError<T, I>(handle, uplo, n, kd, nb, dA, lda, dAband, ldab, dTau, hARes,
                                   hAbandRes, hTauRes, hA, hAband, hTau, max_errors);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        sy2sb_he2hb_getPerfData<T, I>(handle, uplo, n, kd, nb, dA, lda, dAband, ldab, dTau, hA,
                                      hAband, hTau, &gpu_time_used, &cpu_time_used, hot_calls,
                                      argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 10*machine_precision as tolerance
    // max_errors is already normalized, e.g., by n.
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_errors[0], 10);
        ROCSOLVER_TEST_CHECK(T, max_errors[1], 10);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "kd", "nb", "lda");
            rocsolver_bench_output(uploC, n, kd, nb, lda);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "berror", "orthogonality");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_errors[0], max_errors[1]);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_errors[0], max_errors[1]);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_SY2SB_HE2HB(...) \
    extern template void testing_sy2sb_he2hb<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SY2SB_HE2HB, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
