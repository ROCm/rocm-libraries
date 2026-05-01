/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/misc/rocsolver_timer.hpp"
#include "common/misc/generate.hpp"
#include "common/misc/cpu_blas.hpp"

//------------------------------------------------------------------------------
// todo: is COMPLEX needed?
template <bool COMPLEX, typename T>
void ormtr_unmtr_hb2st_checkBadArgs(const rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int kd,
                                    T dV,
                                    const rocblas_int ldv,
                                    T dTau,
                                    T dC,
                                    const rocblas_int ldc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            nullptr, side, trans, m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, rocblas_side(0), trans, m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, rocblas_operation(0), m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd, (T) nullptr, ldv, dTau, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd, dV, ldv, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd, dV, ldv, dTau, (T) nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, 0, n, kd,
            (T) nullptr, ldv, (T) nullptr, (T) nullptr, ldc),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, 0, kd,
            (T) nullptr, ldv, (T) nullptr, (T) nullptr, ldc),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, 0,
            (T) nullptr, ldv, (T) nullptr, (T) nullptr, ldc),
        rocblas_status_success);
}

//------------------------------------------------------------------------------
// todo: is COMPLEX needed?
// todo: why not merge this with ormtr_unmtr_hb2st_checkBadArgs?
template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormtr_unmtr_hb2st_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_conjugate_transpose;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int kd = 1;
    rocblas_int ldv = 2;
    rocblas_int ldc = 2;

    // memory allocation
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    // check bad arguments
    ormtr_unmtr_hb2st_checkBadArgs<COMPLEX>(
        handle, side, trans, m, n, kd,
        dV.data(), ldv, dTau.data(), dC.data(), ldc);
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormtr_unmtr_hb2st_initData(const rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int kd,
                                Td& dAband,
                                const rocblas_int ldab,
                                Td& dV,
                                const rocblas_int ldv,
                                Td& dTau,
                                Td& dC,
                                const rocblas_int ldc,
                                Th& hAband,
                                Th& hV,
                                Th& hTau,
                                Th& hC)
{
    // TODO: how to handle uplo? Easiest would be to convert upper to lower.

    if(CPU)
    {
        // Matrix A is m-by-m on left, or n-by-n on right.
        rocblas_int nA = (side == rocblas_side_left ? m : n);

        // For bandwidth kd, need ku = kd-1 superdiagonals to cover the diagonal
        // blocks. (ku superdiagonals needed if we update diag blocks using
        // gemv/ger, but none needed if we used hemv/her2.)
        // Need kl = 2*kd-1 subdiagonals to cover the off-diagonal blocks and bulges.
        rocblas_int ku = kd - 1;
        rocblas_int kl = 2*kd - 1;
        assert( ldab >= ku + kl + 1 );

        gerand( m, n, C, ldc );
        hbrand( nA, ku, kl, Aband, ldab );

        // Call hb2st on GPU to compute dV. (dTau stored in dV.)
        CHECK_HIP_ERROR(dAband.transfer_from(hAband));
        CHECK_ROCBLAS_ERROR(
            rocsolver_sb2st_hb2st(
                handle, uplo, nA, kd,
                dAband.data(), ldab,
                dD.data(), dE.data(),
                dV.data(), ldv ) );

        // copy data from device to CPU
        CHECK_HIP_ERROR(hV.transfer_from(dV));
        CHECK_HIP_ERROR(hD.transfer_from(dD));
        CHECK_HIP_ERROR(hE.transfer_from(dE));
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
    }
}

//------------------------------------------------------------------------------
// Fills entries of errors with results of 3 tests:
//
// 0. Generate explicit Q and check orthogonality of Q:
//    || I - Q^H Q ||_1 / n
//    Q is na-by-na, R is na-by-na.
//
// 1. Backwards error, where B is tridiagonal output of hb2st:
//    || Q B Q^H - A ||_1 / (n || A ||_1)
//    Q is na-by-na, R is na-by-na.
//
// 2. Compare multiplying random C with explicitly generated Q with implicit Q#:
//    || op(Q#) C - op(Q) C ||_1 / (m || C ||_1)  for side=left, or
//    || C op(Q#) - C op(Q) ||_1 / (m || C ||_1)  for side=right.
//    C is m-by-n, Q is na-by-na, R is m-by-n.
//
// C is m-by-n, A and Q are na-by-na, where na = m for side = left,
// and na = n for side = right. R for residuals is max(m,n)-by-max(m,n).
// cf. LAPACK LAWN 41
//
template <typename T, typename Td, typename Th>
void ormtr_unmtr_hb2st_getError(const rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int kd,
                                Td& dAband,
                                const rocblas_int ldab,
                                Td& dV,
                                const rocblas_int ldv,
                                Td& dTau,
                                Td& dC,
                                const rocblas_int ldc,
                                Td& dR,
                                const rocblas_int ldr,
                                Sd& dnorm,
                                Th& hAband,
                                Th& hV,
                                Th& hTau,
                                Th& hC,
                                Th& hR,
                                Sh& hnorm,
                                double errors[3])
{
    rocblas_int nA = (side == rocblas_side_left ? m : n);

    rocsolver_norm_type norm = rocblas_norm_one;

    // langb, lanhb need n. gemm needs n^2.
    // todo: should we pass hW in? should we allocate dR and hR here?
    // What things get passed and what get allocated locally?
    size_t size_W = nA * nA;
    std::vector<T> hW( size_W );

    // initialize data
    ormtr_unmtr_hb2st_initData<true, true, T>(
        handle, side, trans, m, n, kd,
        dAband, ldab, dV, ldv, dTau, dC, ldc,
        hAband, hV, hTau, hC);

    // execute computations
    // Set Q = Identity, then generate Q = Q*I or I*Q. (Works for either side.)
    CHECK_ROCBLAS_ERROR(
        laset(
            handle, rocblas_fill_general, nA, nA, zero, one, dQ.data(), ldq ));
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, rocblas_operation_none, m, n, kd,
            dV.data(), ldv, dTau.data(), dQ.data(), ldq));

    // Check 0: || I - Q^H Q ||_1 / n
    // Set R = Identity, then generate residual R = I - Q^H Q.
    CHECK_ROCBLAS_ERROR(
        laset(
            handle, rocblas_fill_general, nA, nA, zero, one, dR.data(), ldr ));
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm(
            handle, rocblas_operation_conjugate_transpose, rocblas_operation_none,
            nA, nA, nA,
            -one, dQ.data(), ldq, dQ.data(), ldq,
            one, dR.data(), ldr ));
    // norm( R )
    CHECK_ROCBLAS_ERROR(
        rocblas_lange(
            handle, norm, nA, nA, dR.data(), ldr, dnorm.data() ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    errors[0] = hnorm[0][0] / n;

    // Check 1: || Q B Q^H - A ||_1 / (n || A ||_1)
    // Transfer Q to CPU. (We don't have needed band or tridiag kernels on GPU.)
    // Multiply C = Q B, then add C -= A.
    // Use only the diag and lower band of A; ignore that some part of the upper
    // band is also stored for hb2st.
    CHECK_HIP_ERROR(
        hQ.transfer_from( dQ ) );
    cpu_stmm( nA, nA, hQ.data(), ldq, hD.data(), hE.data(), hR.data(), ldr );
    cpu_gemm( rocblas_operation_none, rocblas_operation_conjugate_transpose,
              nA, nA, nA,
              one, hR.data(), ldr, hQ.data(), ldq,
              zero, hW.data(), ldw );
    cpu_hbadd( rocblas_fill_lower, nA, kd,
               -one, &hAband[0][idiag], ldab,
               one, hW.data(), ldw );
    errors[1] = cpu_lange( norm, nA, nA, hR.data(), ldr, hW ) / nA;
    S Anorm = cpu_lanhb( norm, rocblas_fill_lower, nA, kd, &hAband[0][idiag], ldab, hW );
    if (Anorm != 0)
        errors[1] /= Anorm;

    // Check 2:
    // || op(Q#) C - op(Q) C ||_1 / (m || C ||_1) for left
    // || C op(Q#) - C op(Q) ||_1 / (m || C ||_1) for right
    // todo: normalize with m or n? LAWN 41 sec 7.1.3 has m in all 4 cases.
    //
    // R = op(Q#) C  or  C op(Q#)  using implicit Q# via unmtr.
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd,
            dV.data(), ldv, dTau.data(), dR.data(), ldr));
    if (side == left)
    {
        // R -= op(Q) C
        assert( nA == m );
        CHECK_ROCBLAS_ERROR(
            rocblas_gemm(
                handle, trans, rocblas_operation_none, m, n, nA,
                -one, dQ.data(), ldq, dC.data(), ldc,
                one, dR.data(), ldr));
    }
    else // right
    {
        // R -= C op(Q)
        assert( nA == n );
        CHECK_ROCBLAS_ERROR(
            rocblas_gemm(
                handle, rocblas_operation_none, trans, m, n, nA,
                -one, dC.data(), ldc, dQ.data(), ldq,
                one, dR.data(), ldr));
    }
    // norm( R )
    CHECK_ROCBLAS_ERROR(
        rocblas_lange(
            handle, norm, m, n, dR.data(), ldr, dnorm ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    errors[2] = hnorm[0][0] / m;

    // norm( C )
    CHECK_ROCBLAS_ERROR(
        rocblas_lange(
            handle, norm, m, n, dC.data(), ldc, dnorm ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    if (hnorm[0][0] != 0)
        errors[2] /= hnorm[0][0];
}

//------------------------------------------------------------------------------
template <typename T, typename Td, typename Th>
void ormtr_unmtr_hb2st_getPerfData(const rocblas_handle handle,
                                   const rocblas_side side,
                                   const rocblas_operation trans,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int kd,
                                   Td& dAband,
                                   const rocblas_int ldab,
                                   Td& dV,
                                   const rocblas_int ldv,
                                   Td& dTau,
                                   Td& dC,
                                   const rocblas_int ldc,
                                   Th& hAband,
                                   Th& hV,
                                   Th& hTau,
                                   Th& hC,
                                   double* gpu_time_used,
                                   double* cpu_time_used,
                                   const rocblas_int hot_calls,
                                   const int profile,
                                   const bool profile_kernels,
                                   const bool perf)
{
    // todo: hW needed?
    size_t size_W = (side == rocblas_side_left ? m : n) * 32;
    std::vector<T> hW(size_W);

    // todo: No CPU implementation readily available.
    // unmtr_hb2st is in PLASMA but not LAPACK.

    // Initialize CPU data.
    ormtr_unmtr_hb2st_initData<true, false, T>(
        handle, side, trans, m, n, kd,
        dAband, ldab, dV, ldv, dTau, dC, ldc,
        hAband, hV, hTau, hC);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormtr_unmtr_hb2st_initData<false, true, T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc,
            hAband, hV, hTau, hC);

        CHECK_ROCBLAS_ERROR(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(int iter = 0; iter < hot_calls; iter++)
    {
        ormtr_unmtr_hb2st_initData<false, true, T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc,
            hAband, hV, hTau, hC);

        timer.start(stream);
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd,
            dV.data(), ldv, dTau.data(), dC.data(), ldc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormtr_unmtr_hb2st(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char transC = argus.get<char>("trans");
    rocblas_int m, n, nA;
    if(sideC == 'L')
    {
        m = argus.get<rocblas_int>("m");
        n = argus.get<rocblas_int>("n", m);
        nA = m;  // A is m-by-m on left.
    }
    else
    {
        n = argus.get<rocblas_int>("n");
        m = argus.get<rocblas_int>("m", n);
        nA = n;  // A is n-by-n on right.
    }
    rocblas_int kd = argus.get<rocblas_int>("kd", 1);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", 3*kd);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", m);
    rocblas_int ldq = nA;
    rocblas_int ldr = std::max( m, n );
    rocblas_int ldab = 3*kd - 1;

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value = (side == rocblas_side_both
                          || (COMPLEX && trans == rocblas_operation_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc),
            rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // V is ldv x nv
    rocblas_int nt = ceildiv( nA-1, kd );
    rocblas_int nv_blocks = nt*(nt + 1)/2;
    rocblas_int nv = nv_blocks*kd;

    // determine sizes
    size_t size_Aband = ldab * n;
    size_t size_V = ldv * nv;
    size_t size_D = n;
    size_t size_E = n - 1;
    size_t size_tau = size_t(kd);
    size_t size_C = size_t(ldc) * n;
    size_t size_Q = size_t(ldq) * nA;
    size_t size_R = size_t(ldr) * std::max( m, n );
    double errors[3] = { 0, 0, 0 }, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || kd < 0 || ldv < 3*kd || ldc < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hAband( size_Aband, 1, size_Aband, 1 );
    host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<S> hD( size_Dres, 1, size_Dres, 1 );
    host_strided_batch_vector<S> hE( size_Eres, 1, size_Eres, 1 );
    host_strided_batch_vector<T> hQ(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hR(size_C, 1, size_C, 1);

    device_strided_batch_vector<T> dAband( size_Aband, 1, size_Aband, 1 );
    device_strided_batch_vector<T> dV( size_V, 1, size_V, 1 );
    device_strided_batch_vector<T> dC( size_C, 1, size_C, 1 );
    device_strided_batch_vector<S> dD( size_D, 1, size_D, 1 );
    device_strided_batch_vector<S> dE( size_E, 1, size_E, 1 );
    device_strided_batch_vector<S> dQ( size_Q, 1, size_Q, 1 );
    device_strided_batch_vector<S> dR( size_R, 1, size_R, 1 );

    if (size_Aband)
        CHECK_HIP_ERROR( dAband.memcheck() );
    if (size_V)
        CHECK_HIP_ERROR( dV.memcheck() );
    if (size_C)
        CHECK_HIP_ERROR( dC.memcheck() );
    if (size_D)
        CHECK_HIP_ERROR( dD.memcheck() );
    if (size_E)
        CHECK_HIP_ERROR( dE.memcheck() );
    if (size_Q)
        CHECK_HIP_ERROR( dQ.memcheck() );
    if (size_R)
        CHECK_HIP_ERROR( dR.memcheck() );

    // check quick return
    if(m == 0 || n == 0 || kd == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        ormtr_unmtr_hb2st_getError<T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc,
            hAband, hV, hTau, hC, hCr, errors);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        ormtr_unmtr_hb2st_getPerfData<T>(
            handle, side, trans, m, n, kd,
            dAband, dV, ldv, dTau, dC, ldc,
            hAband, hV, hTau, hC,
            &gpu_time_used, &cpu_time_used, hot_calls,
            argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using machine_precision as tolerance
    // Normalization by m or n already baked into error.
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, errors[0], 1);
        ROCSOLVER_TEST_CHECK(T, errors[1], 1);
        ROCSOLVER_TEST_CHECK(T, errors[2], 1);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("side", "trans", "m", "n", "kd", "ldv", "ldc");
            rocsolver_bench_output(sideC, transC, m, n, kd, ldv, ldc);

            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "ortho", "berror", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, errors[0], errors[1], errors[2]);
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
                rocsolver_bench_output(gpu_time_used, errors[0], errors[1], errors[2]);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_ORMTR_UNMTR_HB2ST(...) \
    extern template void testing_ormtr_unmtr_hb2st<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORMTR_UNMTR_HB2ST, FOREACH_SCALAR_TYPE, APPLY_STAMP)
