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
#include "print_matrix.hpp"

//------------------------------------------------------------------------------
// todo: where should these go?
inline rocblas_status rocblas_copy(
    rocblas_handle handle, rocblas_int n,
    float const* x, rocblas_int incx,
    float* y, rocblas_int incy )
{
    return rocblas_scopy( handle, n, x, incx, y, incy );
}

inline rocblas_status rocblas_copy(
    rocblas_handle handle, rocblas_int n,
    double const* x, rocblas_int incx,
    double* y, rocblas_int incy )
{
    return rocblas_dcopy( handle, n, x, incx, y, incy );
}

inline rocblas_status rocblas_copy(
    rocblas_handle handle, rocblas_int n,
    rocblas_float_complex const* x, rocblas_int incx,
    rocblas_float_complex* y, rocblas_int incy )
{
    return rocblas_ccopy( handle, n, x, incx, y, incy );
}

inline rocblas_status rocblas_copy(
    rocblas_handle handle, rocblas_int n,
    rocblas_double_complex const* x, rocblas_int incx,
    rocblas_double_complex* y, rocblas_int incy )
{
    return rocblas_zcopy( handle, n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
template <typename T, typename I = rocblas_int>
void ormtr_unmtr_hb2st_checkBadArgs(const rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const I m,
                                    const I n,
                                    const I kd,
                                    T dV,
                                    const I ldv,
                                    T dTau,
                                    T dC,
                                    const I ldc)
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
// todo: why not merge this with ormtr_unmtr_hb2st_checkBadArgs?
template <typename T, typename I = rocblas_int>
void testing_ormtr_unmtr_hb2st_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_conjugate_transpose;
    I m = 2;
    I n = 2;
    I kd = 1;
    I ldv = 2;
    I ldc = 2;

    // memory allocation
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    // check bad arguments
    ormtr_unmtr_hb2st_checkBadArgs<decltype(dV.data()), I>(
        handle, side, trans, m, n, kd,
        dV.data(), ldv, dTau.data(), dC.data(), ldc);
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_initData(
    const rocblas_handle handle,
    const rocblas_side side,
    const rocblas_operation trans,
    const I m,
    const I n,
    const I kd,

    // todo: dAband is not actually used, but hAband with ldab is used.
    Td& dAband,
    const I ldab,
    Td& dV,
    const I ldv,
    Td& dTau,
    Td& dC,
    const I ldc,
    Sd& dD,
    Sd& dE,

    Th& hAband,
    Th& hV,
    Th& hTau,
    Th& hC,
    Sh& hD,
    Sh& hE)
{
    // TODO: how to handle uplo? Easiest would be to convert upper to lower.

    if(CPU)
    {
        // Matrix Aband and Q are m-by-m on left, or n-by-n on right.
        I nq = (side == rocblas_side_left ? m : n);

        gerand( m, n, hC[0], ldc );
        hbrand( nq, kd, hAband[0], ldab );

        // Call hb2st on GPU to compute dV. (dTau stored in dV.)
        CHECK_HIP_ERROR(dAband.transfer_from(hAband));
        CHECK_ROCBLAS_ERROR(
            rocsolver_sb2st_hb2st(
                handle, rocblas_fill_lower, nq, kd,
                dAband.data(), ldab,
                dD.data(), dE.data(),
                dV.data(), ldv ) );

        // copy data from device to CPU
        CHECK_HIP_ERROR(hV.transfer_from(dV));
        CHECK_HIP_ERROR(hD.transfer_from(dD));
        CHECK_HIP_ERROR(hE.transfer_from(dE));

        I nt = ceildiv( nq - 1, kd );
        I nv_blocks = nt*(nt + 1)/2;
        I nv = nv_blocks*kd;

        // todo: remove Tau from V in hb2st
        CHECK_ROCBLAS_ERROR(
            rocblas_copy(
                handle, nv, &dV[0][ldv-1], ldv, dTau[0], 1 ) );
        CHECK_HIP_ERROR(hTau.transfer_from(dTau));
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
        CHECK_HIP_ERROR(dV.transfer_from(hV));
    }
}

//------------------------------------------------------------------------------
// Fills entries of errors with results of 3 tests.
// cf. LAPACK LAWN 41, section 7.1.3.
//
// 0. Generate explicit Q and check orthogonality of Q:
//    || I - Q^H Q ||_1 / n
//    I, Q, R are nq-by-nq.
//
// 1. Backwards error, where S is tridiagonal output of hb2st:
//    || Q S Q^H - Aband ||_1 / (n || Aband ||_1)
//    Aband, S, Q, R are nq-by-nq; Aband is banded, S is real symmetric tridiagonal.
//
// 2. Compare multiplying random C by explicitly generated Q1 and by implicit Q2:
//    || op(Q2) C - op(Q1) C ||_1 / (m || C ||_1)  for side=left  (nq = m), or
//    || C op(Q2) - C op(Q1) ||_1 / (m || C ||_1)  for side=right (nq = n).
//    C, R are m-by-n; Q is nq-by-nq.
//
// Allocate R as max( m, nq )-by-max( n, nq ) for use in all 3 tests.
//
template <typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_getError(
    const rocblas_handle handle,
    const rocblas_side side,
    const rocblas_operation trans,
    const I m,
    const I n,
    const I kd,

    Td& dAband,
    const I ldab,
    Td& dV,
    const I ldv,
    Td& dTau,  // unused
    Td& dC,
    const I ldc,
    Sd& dD,
    Sd& dE,
    Td& dQ,
    const I ldq,
    Td& dR,
    const I ldr,
    Sd& dnorm,

    Th& hAband,
    Th& hV,
    Th& hTau,  // unused
    Th& hC,
    Sh& hD,
    Sh& hE,
    Th& hQ,
    Th& hR,
    Sh& hnorm,

    double errors[3])
{
    using S = decltype( std::real( T{} ) );

    const T one = 1;
    const T negone = -1;
    const T zero = 0;

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;
    timer.start( stream );

    I idiag = kd - 1;

    I nq = (side == rocblas_side_left ? m : n);
    rocblas_stride shift = 0;
    rocblas_stride stride = 0;

    // todo: can we remove "_type" from these names. Doesn't seem to add anything.
    rocsolver_norm_type norm = rocsolver_norm_type_one;

    // cpu_lange, cpu_lanhb need rwork size n.
    std::vector<S> hrwork( nq );

    // cpu_gemm needs hW size nq*nq.
    // todo: Should this be passed in?
    I ldw = nq;
    std::vector<T> hW( ldw * nq );

    // initialize data
    ormtr_unmtr_hb2st_initData<true, true, T, I>(
        handle, side, trans, m, n, kd,
        dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
        hAband, hV, hTau, hC, hD, hE);

    // execute computations
    // Set Q = Identity, then generate Q = Q*I or I*Q. (Works for either side.)
    // ungtr would be more efficient, but that isn't implemented.
    CHECK_ROCBLAS_ERROR(
        rocsolver_laset(
            handle, 'g' /*rocblas_fill_full*/, nq, nq, zero, one,
            dQ.data(), shift, ldq, stride, 1 ));

    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, rocblas_operation_none, nq, nq, kd,
            dV.data(), ldv, dTau.data(), dQ.data(), ldq));

    //--------------------
    // Check 0: || I - Q^H Q ||_1 / nq
    // Set R = Identity.
    CHECK_ROCBLAS_ERROR(
        rocsolver_laset(
            handle, 'g' /*rocblas_fill_full*/, nq, nq, zero, one,
            dR.data(), shift, ldr, stride, 1 ));

    // Residual R = I - Q^H Q.
    CHECK_ROCBLAS_ERROR(
        rocsolver_gemm(
            false, handle,
            rocblas_operation_conjugate_transpose, rocblas_operation_none,
            nq, nq, nq,
            &negone, dQ.data(), ldq, stride,
                     dQ.data(), ldq, stride,
            &one,    dR.data(), ldr, stride, 1 ));

    // norm( R )
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange(
            handle, norm, nq, nq, dR.data(), ldr, dnorm.data() ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    errors[0] = hnorm[0][0] / nq;

    //--------------------
    // TODO: use S instead of S?
    // Check 1: || Q S Q^H - Aband ||_1 / (nq || Aband ||_1)
    // Transfer Q to CPU. (We don't have needed band or tridiag kernels on GPU.)
    // Multiply C = Q S, then add C -= Aband.
    // Use only the diag and lower band of Aband; ignore that some part of the upper
    // band is also stored for hb2st.
    CHECK_HIP_ERROR(
        hQ.transfer_from( dQ ) );

    // R = Q S
    cpu_stmm( nq, nq, hQ.data(), ldq, hD.data(), hE.data(), hR.data(), ldr );

    // W = (Q S) Q^H
    cpu_gemm( rocblas_operation_none, rocblas_operation_conjugate_transpose,
              nq, nq, nq,
              one, hR.data(), ldr, hQ.data(), ldq,
              zero, hW.data(), ldw );

    // W -= Aband
    cpu_hbadd( rocblas_fill_lower, nq, kd,
               negone, &hAband[0][idiag], ldab,
               one, hW.data(), ldw );

    // Error = norm( W ) / (nq * norm( Aband ))
    errors[1] = cpu_lange( '1', nq, nq, hW.data(), ldw, hrwork.data() )
              / nq;
    S Anorm = cpu_lanhb( '1', 'L', nq, kd, &hAband[0][idiag], ldab, hrwork.data() );
    if (Anorm != 0)
        errors[1] /= Anorm;

    //--------------------
    // Check 2:
    // || op(Q#) C - op(Q) C ||_1 / (m || C ||_1) for left
    // || C op(Q#) - C op(Q) ||_1 / (m || C ||_1) for right
    // todo: normalize with m or n? LAWN 41 sec 7.1.3 has m in all 4 cases.
    CHECK_HIP_ERROR(
        hipMemcpy2DAsync(
            dR[0], ldr*sizeof(T),
            dC[0], ldc*sizeof(T),
            m*sizeof(T), n, hipMemcpyDefault, stream ) );

    // R = op(Q#) C  or  C op(Q#)  using implicit Q# via unmtr.
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd,
            dV.data(), ldv, dTau.data(), dR.data(), ldr));

    if (side == rocblas_side_left)
    {
        // R -= op(Q) C
        assert( nq == m );
        CHECK_ROCBLAS_ERROR(
            rocsolver_gemm(
                false, handle, trans, rocblas_operation_none, m, n, nq,
                &negone, dQ.data(), ldq, stride,
                         dC.data(), ldc, stride,
                &one,    dR.data(), ldr, stride, 1 ));
    }
    else // right
    {
        // R -= C op(Q)
        assert( nq == n );
        CHECK_ROCBLAS_ERROR(
            rocsolver_gemm(
                false, handle, rocblas_operation_none, trans, m, n, nq,
                &negone, dC.data(), ldc, stride,
                         dQ.data(), ldq, stride,
                &one,    dR.data(), ldr, stride, 1 ));
    }

    // norm( R )
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange(
            handle, norm, m, n, dR.data(), ldr, dnorm ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    errors[2] = hnorm[0][0] / m;

    // norm( C )
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange(
            handle, norm, m, n, dC.data(), ldc, dnorm ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    if (hnorm[0][0] != 0)
        errors[2] /= hnorm[0][0];

    timer.end( stream );
    S eps = get_epsilon<T>();
    std::cout << "# getError"
            << ": m " << m << ", n " << n << ", kd " << kd
            << ", side " << rocblas2char_side( side )
            << ", trans " << rocblas2char_operation( trans )
            << std::scientific << std::setprecision( 3 )
            << ", ortho "   << errors[0]
            << ", A-QBQ^h " << errors[1]
            << ", QC-QC "   << errors[2]
            << ", time " << timer.get_combined()
            << (errors[0] < eps && errors[1] < eps && errors[2] < eps
                ? " ok" : " FAILED")
            << "\n\n";
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_getPerfData(
    const rocblas_handle handle,
    const rocblas_side side,
    const rocblas_operation trans,
    const I m,
    const I n,
    const I kd,

    Td& dAband,
    const I ldab,
    Td& dV,
    const I ldv,
    Td& dTau,
    Td& dC,
    const I ldc,
    Sd& dD,
    Sd& dE,

    Th& hAband,
    Th& hV,
    Th& hTau,
    Th& hC,
    Sh& hD,
    Sh& hE,

    double* gpu_time_used,
    double* cpu_time_used,
    const rocblas_int hot_calls,
    const int profile,
    const bool profile_kernels,
    const bool perf)
{
    using S = decltype( std::real( T{} ) );

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    // todo: No CPU implementation readily available.
    // unmtr_hb2st is in PLASMA but not LAPACK.

    // Initialize CPU data.
    ormtr_unmtr_hb2st_initData<true, false, T, I>(
        handle, side, trans, m, n, kd,
        dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
        hAband, hV, hTau, hC, hD, hE);

    // cold calls
    double start, time;
    for(int iter = 0; iter < 2; iter++)
    {
        ormtr_unmtr_hb2st_initData<false, true, T, I>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
            hAband, hV, hTau, hC, hD, hE);

        start = get_time_us_sync( stream );
        CHECK_ROCBLAS_ERROR(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc));
        time = get_time_us_sync(stream) - start;
        printf( "m %d, n %d, kd %d, cold iter %d, time %.4f\n", m, n, kd, iter, time );
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

    for(int iter = 0; iter < hot_calls; iter++)
    {
        ormtr_unmtr_hb2st_initData<false, true, T, I>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
            hAband, hV, hTau, hC, hD, hE);

        timer.start(stream);
        start = get_time_us_sync( stream );
        CHECK_ROCBLAS_ERROR(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc));
        time = get_time_us_sync(stream) - start;
        timer.end(stream);
        printf( "m %d, n %d, kd %d, hot  iter %d, time %.4f\n", m, n, kd, iter, time );
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, typename I = rocblas_int>
void testing_ormtr_unmtr_hb2st(Arguments& argus)
{
    using S = decltype( std::real( T{} ) );

    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char transC = argus.get<char>("trans");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I nq = (sideC == 'L' ? m : n);
    I kd = argus.get<I>("kd", 1);
    I ldv = argus.get<I>("ldv", 3*kd);
    I ldc = argus.get<I>("ldc", m);
    I ldq = nq;
    I ldr = std::max( m, nq );
    I ldab = 3*kd - 1;

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value = (side == rocblas_side_both
                          || (rocblas_is_complex<T> && trans == rocblas_operation_transpose));
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
    I nt = ceildiv( nq-1, kd );
    I nv_blocks = nt*(nt + 1)/2;
    I nv = nv_blocks*kd;

    // determine sizes
    size_t size_Aband = size_t(ldab) * nq;
    size_t size_V = size_t(ldv) * nv;
    size_t size_D = nq;
    size_t size_E = nq - 1;
    size_t size_tau = nv;
    size_t size_C = size_t(ldc) * n;
    size_t size_Q = size_t(ldq) * nq;
    size_t size_R = size_t(ldr) * std::max( n, nq );
    size_t size_norm = 1;
    double errors[3] = { 0, 0, 0 }, gpu_time_used = 0, cpu_time_used = 0;

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
    host_strided_batch_vector<T> hV( size_V, 1, size_V, 1 );
    host_strided_batch_vector<T> hTau( size_tau, 1, size_tau, 1 );  // unused
    host_strided_batch_vector<T> hC( size_C, 1, size_C, 1 );
    host_strided_batch_vector<S> hD( size_D, 1, size_D, 1 );
    host_strided_batch_vector<S> hE( size_E, 1, size_E, 1 );
    host_strided_batch_vector<T> hQ( size_Q, 1, size_Q, 1 );
    host_strided_batch_vector<T> hR( size_R, 1, size_R, 1 );
    host_strided_batch_vector<S> hnorm(size_norm, 1, size_norm, 1);

    device_strided_batch_vector<T> dAband( size_Aband, 1, size_Aband, 1 );
    device_strided_batch_vector<T> dV( size_V, 1, size_V, 1 );
    device_strided_batch_vector<T> dTau( size_tau, 1, size_tau, 1 );  // unused
    device_strided_batch_vector<T> dC( size_C, 1, size_C, 1 );
    device_strided_batch_vector<S> dD( size_D, 1, size_D, 1 );
    device_strided_batch_vector<S> dE( size_E, 1, size_E, 1 );
    device_strided_batch_vector<T> dQ( size_Q, 1, size_Q, 1 );
    device_strided_batch_vector<T> dR( size_R, 1, size_R, 1 );
    device_strided_batch_vector<S> dnorm(size_norm, 1, size_norm, 1);

    if (size_Aband)
        CHECK_HIP_ERROR( dAband.memcheck() );
    if (size_V)
        CHECK_HIP_ERROR( dV.memcheck() );
    if (size_tau)
        CHECK_HIP_ERROR( dTau.memcheck() );
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
    if (size_norm)
        CHECK_HIP_ERROR( dnorm.memcheck() );

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
        ormtr_unmtr_hb2st_getError<T, I>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE, dQ, ldq, dR, ldr, dnorm,
            hAband, hV, hTau, hC, hD, hE, hQ, hR, hnorm, errors);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        ormtr_unmtr_hb2st_getPerfData<T, I>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
            hAband, hV, hTau, hC, hD, hE,
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
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "ortho I-Q^HQ", "berror A-QBQ^H", "ferror QC-Q#C");
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

INSTANTIATE(EXTERN_TESTING_ORMTR_UNMTR_HB2ST, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
