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
template <typename T>
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
std::cout << "\n# " << __func__ << "\n";

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

std::cout << "# " << __func__ << " done\n\n";
}

//------------------------------------------------------------------------------
// todo: why not merge this with ormtr_unmtr_hb2st_checkBadArgs?
template <typename T>
void testing_ormtr_unmtr_hb2st_bad_arg()
{
std::cout << "# " << __func__ << "\n";

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
    ormtr_unmtr_hb2st_checkBadArgs(
        handle, side, trans, m, n, kd,
        dV.data(), ldv, dTau.data(), dC.data(), ldc);

std::cout << "# " << __func__ << " done\n";
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_initData(
    const rocblas_handle handle,
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
    Sd& dD,
    Sd& dE,

    Th& hAband,
    Th& hV,
    Th& hTau,
    Th& hC,
    Sh& hD,
    Sh& hE)
{
std::cout << "# " << __func__ << "\n";
    // TODO: how to handle uplo? Easiest would be to convert upper to lower.

    if(CPU)
    {
        // Matrix A is m-by-m on left, or n-by-n on right.
        rocblas_int nq = (side == rocblas_side_left ? m : n);

        gerand( m, n, hC[0], ldc );
        hbrand( nq, kd, hAband[0], ldab );

        print_matrix( "C", m, n, hC[0], ldc, 6 );
        print_matrix( "Aband", ldab, nq, hAband[0], ldab, 6 );
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

        rocblas_int nt = ceildiv( nq - 1, kd );
        rocblas_int nv_blocks = nt*(nt + 1)/2;
        rocblas_int nv = nv_blocks*kd;
        print_matrix( "V", 3*kd, nv, hV[0], ldv, 6 );
        print_matrix( "D", 1, n,   hD[0], 1, 6 );
        print_matrix( "E", 1, n-1, hE[0], 1, 6 );

        CHECK_ROCBLAS_ERROR(
            rocblas_copy(
                handle, nv, &dV[0][ldv-1], ldv, dTau[0], 1 ) );
        print_matrix( "Tau", 1, nv, dTau[0], 1, 6 );
        CHECK_HIP_ERROR(hTau.transfer_from(dTau));
        print_matrix( "hTau", 1, nv, hTau[0], 1, 6 );
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
    }
    std::cout << "# " << __func__ << " done\n";
}

//------------------------------------------------------------------------------
// Fills entries of errors with results of 3 tests.
// cf. LAPACK LAWN 41, section 7.1.3.
//
// 0. Generate explicit Q and check orthogonality of Q:
//    || I - Q^H Q ||_1 / n
//    I, Q, R are nq-by-nq.
//
// 1. Backwards error, where B is tridiagonal output of hb2st:
//    || Q B Q^H - A ||_1 / (n || A ||_1)
//    A, B, Q, R are nq-by-nq; A is banded, B is real tridiagonal.
//
// 2. Compare multiplying random C by explicitly generated Q1 and by implicit Q2:
//    || op(Q2) C - op(Q1) C ||_1 / (m || C ||_1)  for side=left  (nq = m), or
//    || C op(Q2) - C op(Q1) ||_1 / (m || C ||_1)  for side=right (nq = n).
//    C, R are m-by-n; Q is nq-by-nq.
//
// Allocate R as max( m, nq )-by-max( n, nq ) for use in all 3 tests.
//
template <typename T, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_getError(
    const rocblas_handle handle,
    const rocblas_side side,
    const rocblas_operation trans,
    const rocblas_int m,
    const rocblas_int n,
    const rocblas_int kd,

    Td& dAband,
    const rocblas_int ldab,
    Td& dV,
    const rocblas_int ldv,
    Td& dTau,  // unused
    Td& dC,
    const rocblas_int ldc,
    Sd& dD,
    Sd& dE,
    Td& dQ,
    const rocblas_int ldq,
    Td& dR,
    const rocblas_int ldr,
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
    std::cout << "\n# " << __func__ << "\n";
    using S = decltype( std::real( T{} ) );

    const T one = 1;
    const T negone = -1;
    const T zero = 0;

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;
    timer.start( stream );

    rocblas_int idiag = kd - 1;

    rocblas_int nq = (side == rocblas_side_left ? m : n);
    rocblas_stride shift = 0;
    rocblas_stride stride = 0;
    std::cout << "# side = " << side
            << ", trans = " << trans
            << ", m = " << m
            << ", n = " << n
            << ", nq = " << nq
            << ", kd = " << kd
            << ", ldab = " << ldab
            << ", ldv = " << ldv
            << ", ldq = " << ldq
            << ", ldr = " << ldr
            << "\n";

    // todo: can we remove "_type" from these names. Doesn't seem to add anything.
    rocsolver_norm_type norm = rocsolver_norm_type_one;

    // cpu_lange, cpu_lanhb need rwork size n.
    std::vector<S> hrwork( nq );

    // cpu_gemm needs hW size n*n.
    // todo: Should this be passed in?
    rocblas_int ldw = nq;
    std::vector<T> hW( ldw * nq );

    // initialize data
    ormtr_unmtr_hb2st_initData<true, true, T>(
        handle, side, trans, m, n, kd,
        dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
        hAband, hV, hTau, hC, hD, hE);

    // execute computations
    // Set Q = Identity, then generate Q = Q*I or I*Q. (Works for either side.)
    std::cout << "# Q = I\n";
    CHECK_ROCBLAS_ERROR(
        rocsolver_laset(
            handle, 'g' /*rocblas_fill_full*/, nq, nq, zero, one,
            dQ.data(), shift, ldq, stride, 1 ));
    print_matrix( "Q_identity", nq, nq, dQ[0], ldq, 6 );

    std::cout << "# unmtr( V, Q )\n";
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, rocblas_operation_none, nq, nq, kd,
            dV.data(), ldv, dTau.data(), dQ.data(), ldq));
    print_matrix( "Q", nq, nq, dQ[0], ldq, 6 );

    // Check 0: || I - Q^H Q ||_1 / n
    // Set R = Identity.
    std::cout << "# R = I\n";
    CHECK_ROCBLAS_ERROR(
        rocsolver_laset(
            handle, 'g' /*rocblas_fill_full*/, nq, nq, zero, one,
            dR.data(), shift, ldr, stride, 1 ));
    print_matrix( "R_identity", nq, nq, dR[0], ldr, 6 );

    // Residual R = I - Q^H Q.
    std::cout << "# R = I - Q^H Q\n";
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm(
            false, handle,
            rocblas_operation_conjugate_transpose, rocblas_operation_none,
            nq, nq, nq,
            &negone, dQ.data(), ldq, stride,
                     dQ.data(), ldq, stride,
            &one,    dR.data(), ldr, stride, 1 ));
    print_matrix( "R_ortho", nq, nq, dR[0], ldr, 6 );

    // norm( R )
    std::cout << "norm( R )\n";
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange(
            handle, norm, nq, nq, dR.data(), ldr, dnorm.data() ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    std::cout << "hnorm( R ) = " << hnorm[0][0] << "\n";
    errors[0] = hnorm[0][0] / n;

    // Check 1: || Q B Q^H - A ||_1 / (n || A ||_1)
    // Transfer Q to CPU. (We don't have needed band or tridiag kernels on GPU.)
    // Multiply C = Q B, then add C -= A.
    // Use only the diag and lower band of A; ignore that some part of the upper
    // band is also stored for hb2st.
    CHECK_HIP_ERROR(
        hQ.transfer_from( dQ ) );
    print_matrix( "hQ", nq, nq, hQ[0], ldq, 6 );

    // R = Q B
    std::cout << "# R = Q*B, B is tridiagonal matrix\n";
    cpu_stmm( nq, nq, hQ.data(), ldq, hD.data(), hE.data(), hR.data(), ldr );
    print_matrix( "R_QB", nq, nq, hR.data(), ldr, 6 );

    // W = (Q B) Q^H
    std::cout << "# W = (Q B) Q^H\n";
    cpu_gemm( rocblas_operation_none, rocblas_operation_conjugate_transpose,
              nq, nq, nq,
              one, hR.data(), ldr, hQ.data(), ldq,
              zero, hW.data(), ldw );
    print_matrix( "W_QBQh", nq, nq, hW.data(), ldw, 6 );

    // W -= Aband
    std::cout << "# W -= Aband\n";
    cpu_hbadd( rocblas_fill_lower, nq, kd,
               negone, &hAband[0][idiag], ldab,
               one, hW.data(), ldw );
    print_matrix( "W_QBQh_A", nq, nq, hW.data(), ldw, 6 );

    // Error = norm( W ) / (nq * norm( A ))
    errors[1] = cpu_lange( '1', nq, nq, hW.data(), ldw, hrwork.data() ) / nq;
    S Anorm = cpu_lanhb( '1', 'L', nq, kd, &hAband[0][idiag], ldab, hrwork.data() );
    std::cout << "# norm( W ) = " << errors[1] << "\n";
    std::cout << "# norm( A ) = " << Anorm << "\n";
    if (Anorm != 0)
        errors[1] /= Anorm;

    // Check 2:
    // || op(Q#) C - op(Q) C ||_1 / (m || C ||_1) for left
    // || C op(Q#) - C op(Q) ||_1 / (m || C ||_1) for right
    // todo: normalize with m or n? LAWN 41 sec 7.1.3 has m in all 4 cases.
    //
    // R = op(Q#) C  or  C op(Q#)  using implicit Q# via unmtr.
    std::cout << "# R = op(Q) C (left) or C op(Q) (right)\n";
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd,
            dV.data(), ldv, dTau.data(), dR.data(), ldr));

    if (side == rocblas_side_left)
    {
        // R -= op(Q) C
        print_matrix( "R_QC", m, n, dR[0], ldr, 6 );
        std::cout << "# R -= op(Q) C (left)\n";
        assert( nq == m );
        CHECK_ROCBLAS_ERROR(
            rocblas_gemm(
                false, handle, trans, rocblas_operation_none, m, n, nq,
                &negone, dQ.data(), ldq, stride,
                         dC.data(), ldc, stride,
                &one,    dR.data(), ldr, stride, 1 ));
        print_matrix( "R_QC_QC", m, n, dR[0], ldr, 6 );
    }
    else // right
    {
        // R -= C op(Q)
        print_matrix( "R_CQ", m, n, dR[0], ldr, 6 );
        std::cout << "# R -= C op(Q) (right)\n";
        assert( nq == n );
        CHECK_ROCBLAS_ERROR(
            rocblas_gemm(
                false, handle, rocblas_operation_none, trans, m, n, nq,
                &negone, dC.data(), ldc, stride,
                         dQ.data(), ldq, stride,
                &one,    dR.data(), ldr, stride, 1 ));
        print_matrix( "R_CQ_CQ", m, n, dR.data(), ldr, 6 );
    }

    // norm( R )
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange(
            handle, norm, m, n, dR.data(), ldr, dnorm ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    std::cout << "# norm( R ) = " << hnorm[0][0] << "\n";
    errors[2] = hnorm[0][0] / m;

    // norm( C )
    CHECK_ROCBLAS_ERROR(
        rocsolver_lange(
            handle, norm, m, n, dC.data(), ldc, dnorm ));
    CHECK_HIP_ERROR(
        hnorm.transfer_from( dnorm ) );
    std::cout << "# norm( C ) = " << hnorm[0][0] << "\n";
    if (hnorm[0][0] != 0)
        errors[2] /= hnorm[0][0];
    timer.end( stream );
    std::cout << "# " << __func__ << " done, errors "
            << std::scientific << std::setprecision( 3 )
            << errors[0] << ", " << errors[1] << ", " << errors[2]
            << ", time " << timer.get_combined() << "\n\n";
}

//------------------------------------------------------------------------------
template <typename T, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_getPerfData(
    const rocblas_handle handle,
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
std::cout << "\n" << __func__ << "\n";
    using S = decltype( std::real( T{} ) );

    // todo: hW needed?
    size_t size_W = (side == rocblas_side_left ? m : n) * 32;
    std::vector<T> hW(size_W);

    // todo: No CPU implementation readily available.
    // unmtr_hb2st is in PLASMA but not LAPACK.

std::cout << __func__ << ":" << __LINE__ << "\n";
    // Initialize CPU data.
    ormtr_unmtr_hb2st_initData<true, false, T>(
        handle, side, trans, m, n, kd,
        dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
        hAband, hV, hTau, hC, hD, hE);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
std::cout << __func__ << ":" << __LINE__ << "\n";
        ormtr_unmtr_hb2st_initData<false, true, T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
            hAband, hV, hTau, hC, hD, hE);

std::cout << __func__ << ":" << __LINE__ << " rocsolver_ormtr_unmtr_hb2st\n";
        CHECK_ROCBLAS_ERROR(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc));
std::cout << __func__ << ":" << __LINE__ << "\n";
    }

std::cout << __func__ << ":" << __LINE__ << "\n";
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
std::cout << __func__ << ":" << __LINE__ << "\n";
        ormtr_unmtr_hb2st_initData<false, true, T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
            hAband, hV, hTau, hC, hD, hE);

std::cout << __func__ << ":" << __LINE__ << " rocsolver_ormtr_unmtr_hb2st\n";
        timer.start(stream);
        CHECK_ROCBLAS_ERROR(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc));
        timer.end(stream);
std::cout << __func__ << ":" << __LINE__ << "\n";
    }
    *gpu_time_used = timer.get_combined();

std::cout << __func__ << " done\n\n";
}

//------------------------------------------------------------------------------
template <typename T>
void testing_ormtr_unmtr_hb2st(Arguments& argus)
{
std::cout << "\n" << __func__ << "\n";
    using S = decltype( std::real( T{} ) );

    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char transC = argus.get<char>("trans");
    rocblas_int m, n, nq;
    if(sideC == 'L')
    {
        m = argus.get<rocblas_int>("m");
        n = argus.get<rocblas_int>("n", m);
        nq = m;  // A is m-by-m on left.
    }
    else
    {
        n = argus.get<rocblas_int>("n");
        m = argus.get<rocblas_int>("m", n);
        nq = n;  // A is n-by-n on right.
    }
    rocblas_int kd = argus.get<rocblas_int>("kd", 1);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", 3*kd);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", m);
    rocblas_int ldq = nq;
    rocblas_int ldr = std::max( m, nq );
    rocblas_int ldab = 3*kd - 1;

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value = (side == rocblas_side_both
                          || (rocblas_is_complex<T> && trans == rocblas_operation_transpose));
    if(invalid_value)
    {
std::cout << __func__ << ":" << __LINE__ << "\n";
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
    rocblas_int nt = ceildiv( nq-1, kd );
    rocblas_int nv_blocks = nt*(nt + 1)/2;
    rocblas_int nv = nv_blocks*kd;

    // determine sizes
    size_t size_Aband = size_t(ldab) * n;
    size_t size_V = size_t(ldv) * nv;
    size_t size_D = n;
    size_t size_E = n - 1;
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
std::cout << __func__ << ":" << __LINE__ << " rocsolver_ormtr_unmtr_hb2st invalid\n";
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc),
            rocblas_status_invalid_size);
std::cout << __func__ << ":" << __LINE__ << "\n";

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
std::cout << __func__ << ":" << __LINE__ << " rocsolver_ormtr_unmtr_hb2st query\n";
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc));
std::cout << __func__ << ":" << __LINE__ << "\n";

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

std::cout << __func__ << ":" << __LINE__ << " memory\n";
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

std::cout << __func__ << ":" << __LINE__ << "\n";
    // check quick return
    if(m == 0 || n == 0 || kd == 0)
    {
std::cout << __func__ << ":" << __LINE__ << " rocsolver_ormtr_unmtr_hb2st quick\n";
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
std::cout << __func__ << ":" << __LINE__ << "\n";
        ormtr_unmtr_hb2st_getError<T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE, dQ, ldq, dR, ldr, dnorm,
            hAband, hV, hTau, hC, hD, hE, hQ, hR, hnorm, errors);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
std::cout << __func__ << ":" << __LINE__ << "\n";
        ormtr_unmtr_hb2st_getPerfData<T>(
            handle, side, trans, m, n, kd,
            dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE,
            hAband, hV, hTau, hC, hD, hE,
            &gpu_time_used, &cpu_time_used, hot_calls,
            argus.profile, argus.profile_kernels, argus.perf);
    }

std::cout << __func__ << ":" << __LINE__ << "\n";
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
std::cout << __func__ << " done\n\n";
}

#define EXTERN_TESTING_ORMTR_UNMTR_HB2ST(...) \
    extern template void testing_ormtr_unmtr_hb2st<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORMTR_UNMTR_HB2ST, FOREACH_SCALAR_TYPE, APPLY_STAMP)
