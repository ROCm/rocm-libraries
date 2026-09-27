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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/cpu_blas.hpp"
#include "common/misc/generate.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

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
        rocsolver_ormtr_unmtr_hb2st(nullptr, side, trans, m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, rocblas_side(0), trans, m, n, kd, dV,
                                                      ldv, dTau, dC, ldc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, rocblas_operation(0), m, n, kd,
                                                      dV, ldv, dTau, dC, ldc),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, (T) nullptr, ldv, dTau, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV, ldv, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV, ldv, dTau, (T) nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, I(0), n, kd, (T) nullptr,
                                                      ldv, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, I(0), kd, (T) nullptr,
                                                      ldv, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
}

//------------------------------------------------------------------------------
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

#ifndef ROCSOLVER_ENABLE_EIG_2STAGE
    // unmtr_hb2st is gated behind ROCSOLVER_ENABLE_EIG_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV.data(), ldv,
                                                      dTau.data(), dC.data(), ldc),
                          rocblas_status_not_implemented);
    return;
#endif

    // check bad arguments
    ormtr_unmtr_hb2st_checkBadArgs<decltype(dV.data()), I>(handle, side, trans, m, n, kd, dV.data(),
                                                           ldv, dTau.data(), dC.data(), ldc);
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_initData(const rocblas_handle handle,
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
                                Sh& hE)
{
    if(CPU)
    {
        // Matrix Aband and Q are m-by-m on left, or n-by-n on right.
        I nq = (side == rocblas_side_left ? m : n);

        gerand(m, n, hC[0], ldc);
        hbrand(nq, kd, kd - 1, hAband[0], ldab);

        // Call hb2st on GPU to compute dV and dTau.
        CHECK_HIP_ERROR(dAband.transfer_from(hAband));
        CHECK_ROCBLAS_ERROR(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, nq, kd, dAband.data(),
                                                  ldab, dD.data(), dE.data(), dV.data(), ldv,
                                                  dTau.data()));

        // copy data from device to CPU
        CHECK_HIP_ERROR(hV.transfer_from(dV));
        CHECK_HIP_ERROR(hD.transfer_from(dD));
        CHECK_HIP_ERROR(hE.transfer_from(dE));
        CHECK_HIP_ERROR(hTau.transfer_from(dTau));
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dTau.transfer_from(hTau));
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
// 1. Backwards error, where Atri is tridiagonal output of hb2st:
//    || Q Atri Q^H - Aband ||_1 / (n || Aband ||_1)
//    Aband, Atri, Q, R are nq-by-nq; Aband is banded, Atri is real symmetric tridiagonal.
//
// 2. Compare multiplying random C by explicitly generated Q1 and by implicit Q2:
//    || op(Q2) C - op(Q1) C ||_1 / (nq || C ||_1)  for side=left  (nq = m), or
//    || C op(Q2) - C op(Q1) ||_1 / (nq || C ||_1)  for side=right (nq = n).
//    C, R are m-by-n; Q is nq-by-nq.
//
// Allocate R as max( m, nq )-by-max( n, nq ) for use in all 3 tests.
//
template <typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_getError(const rocblas_handle handle,
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
                                Td& dQ,
                                const I ldq,
                                Td& dR,
                                const I ldr,
                                Sd& dnorm,

                                Th& hAband,
                                Th& hV,
                                Th& hTau,
                                Th& hC,
                                Sh& hD,
                                Sh& hE,
                                Th& hQ,
                                Th& hR,
                                Sh& hnorm,

                                double errors[3])
{
    using S = decltype(std::real(T{}));

    const T one = 1;
    const T negone = -1;
    const T zero = 0;

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    I idiag = kd - 1;

    I nq = (side == rocblas_side_left ? m : n);
    rocblas_stride shift = 0;
    rocblas_stride stride = 0;

    rocsolver_norm_type norm = rocsolver_norm_type_one;

    // cpu_lange, cpu_lanhb need rwork size n.
    std::vector<S> hrwork(nq);

    // cpu_gemm needs hW size nq*nq.
    I ldw = nq;
    std::vector<T> hW(ldw * nq);

    // initialize data
    ormtr_unmtr_hb2st_initData<true, true, T, I>(handle, side, trans, m, n, kd, // opts
                                                 dAband, ldab, dV, ldv, dTau, dC, ldc, dD, dE, // dev
                                                 hAband, hV, hTau, hC, hD, hE); // host

    // execute computations
    // Set Q = Identity, then generate Q = Q*I or I*Q. (Works for either side.)
    // ungtr would be more efficient, but that isn't implemented.
    CHECK_ROCBLAS_ERROR(rocsolver_laset(handle, rocblas_fill_full, nq, nq, zero, one, dQ.data(), ldq));

    CHECK_ROCBLAS_ERROR(rocsolver_ormtr_unmtr_hb2st(handle, side, rocblas_operation_none, nq, nq, kd,
                                                    dV.data(), ldv, dTau.data(), dQ.data(), ldq));

    //--------------------
    // Check 0: || I - Q^H Q ||_1 / nq
    // Set R = Identity.
    CHECK_ROCBLAS_ERROR(rocsolver_laset(handle, rocblas_fill_full, nq, nq, zero, one, dR.data(), ldr));

    // Residual R = I - Q^H Q.
    CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle, rocblas_operation_conjugate_transpose,
                                       rocblas_operation_none, nq, nq, nq, // opts
                                       &negone, dQ.data(), ldq, stride, // Q^H
                                       dQ.data(), ldq, stride, // Q
                                       &one, dR.data(), ldr, stride, 1)); // R

    // norm( R )
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, norm, nq, nq, dR.data(), ldr, dnorm.data()));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    errors[0] = hnorm[0][0] / nq;

    //--------------------
    // Check 1: || Q Atri Q^H - Aband ||_1 / (nq || Aband ||_1)
    // Transfer Q to CPU. (We don't have needed band or tridiag kernels on GPU.)
    // Multiply C = Q Atri, then add C -= Aband.
    // Use only the diag and lower band of Aband; ignore that some part of the upper
    // band is also stored for hb2st.
    CHECK_HIP_ERROR(hQ.transfer_from(dQ));

    // R = Q Atri
    cpu_stmm(nq, nq, hQ.data(), ldq, // Q
             hD.data(), hE.data(), // Atri in {D, E}
             hR.data(), ldr); // R

    // W = (Q Atri) Q^H
    cpu_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, nq, nq, nq, // opts
             one, hR.data(), ldr, // R
             hQ.data(), ldq, // Q^H
             zero, hW.data(), ldw); // W

    // W -= Aband
    cpu_hbadd(rocblas_fill_lower, nq, kd, negone, &hAband[0][idiag], ldab, one, hW.data(), ldw);

    // Error = norm( W ) / (nq * norm( Aband ))
    errors[1] = cpu_lange('1', nq, nq, hW.data(), ldw, hrwork.data()) / nq;
    S Anorm = cpu_lanhb('1', 'L', nq, kd, &hAband[0][idiag], ldab, hrwork.data());
    if(Anorm != 0)
        errors[1] /= Anorm;

    //--------------------
    // Check 2:
    // || op(Q#) C - op(Q) C ||_1 / (nq || C ||_1) for left
    // || C op(Q#) - C op(Q) ||_1 / (nq || C ||_1) for right
    // Normalize by nq. LAWN 41 sec 7.1.3 normalizes m in all 4 cases.
    CHECK_HIP_ERROR(hipMemcpy2DAsync(dR[0], ldr * sizeof(T), // R
                                     dC[0], ldc * sizeof(T), // C
                                     m * sizeof(T), n, hipMemcpyDefault, stream));

    // R = op(Q#) C  or  C op(Q#)  using implicit Q# via unmtr.
    CHECK_ROCBLAS_ERROR(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV.data(), ldv,
                                                    dTau.data(), dR.data(), ldr));

    if(side == rocblas_side_left)
    {
        // R -= op(Q) C
        assert(nq == m);
        CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle, trans, rocblas_operation_none, m, n, nq, // opts
                                           &negone, dQ.data(), ldq, stride, // op(Q)
                                           dC.data(), ldc, stride, // C
                                           &one, dR.data(), ldr, stride, 1)); // R
    }
    else // right
    {
        // R -= C op(Q)
        assert(nq == n);
        CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle, rocblas_operation_none, trans, m, n, nq, // opts
                                           &negone, dC.data(), ldc, stride, // C
                                           dQ.data(), ldq, stride, // op(Q)
                                           &one, dR.data(), ldr, stride, 1)); // R
    }

    // norm( R ) / nq
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, norm, m, n, dR.data(), ldr, dnorm));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    errors[2] = hnorm[0][0] / nq;

    // norm( C )
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, norm, m, n, dC.data(), ldc, dnorm));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    if(hnorm[0][0] != 0)
        errors[2] /= hnorm[0][0];
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ormtr_unmtr_hb2st_getPerfData(const rocblas_handle handle,
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
    using S = decltype(std::real(T{}));

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    // todo: No CPU implementation readily available.
    // unmtr_hb2st is in PLASMA but not LAPACK.

    // Initialize CPU data.
    ormtr_unmtr_hb2st_initData<true, false, T, I>(handle, side, trans, m, n, kd, dAband, ldab, dV,
                                                  ldv, dTau, dC, ldc, dD, dE, hAband, hV, hTau, hC,
                                                  hD, hE);

    // cold calls
    double start, time;
    for(int iter = 0; iter < 2; iter++)
    {
        ormtr_unmtr_hb2st_initData<false, true, T, I>(handle, side, trans, m, n, kd, dAband, ldab,
                                                      dV, ldv, dTau, dC, ldc, dD, dE, hAband, hV,
                                                      hTau, hC, hD, hE);

        CHECK_ROCBLAS_ERROR(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV.data(),
                                                        ldv, dTau.data(), dC.data(), ldc));
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
        ormtr_unmtr_hb2st_initData<false, true, T, I>(handle, side, trans, m, n, kd, dAband, ldab,
                                                      dV, ldv, dTau, dC, ldc, dD, dE, hAband, hV,
                                                      hTau, hC, hD, hE);

        timer.start(stream);
        CHECK_ROCBLAS_ERROR(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV.data(),
                                                        ldv, dTau.data(), dC.data(), ldc));
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, typename I = rocblas_int>
void testing_ormtr_unmtr_hb2st(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char transC = argus.get<char>("trans");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I nq = (sideC == 'L' ? m : n);
    I kd = argus.get<I>("kd", 1);
    I ldv = argus.get<I>("ldv", 2 * kd - 1);
    I ldc = argus.get<I>("ldc", m);
    I ldq = nq;
    I ldr = std::max(m, nq);
    I ldab = 3 * kd - 1;

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

#ifndef ROCSOLVER_ENABLE_EIG_2STAGE
    // unmtr_hb2st is gated behind ROCSOLVER_ENABLE_EIG_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, // opts
                                                      (T*)nullptr, ldv, // V
                                                      (T*)nullptr, // tau
                                                      (T*)nullptr, ldc), // C
                          rocblas_status_not_implemented);
    return;
#endif

    // check non-supported values
    bool invalid_value = (side == rocblas_side_both
                          || (rocblas_is_complex<T> && trans == rocblas_operation_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, (T*)nullptr,
                                                          ldv, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // V is ldv x nv
    I nt = 0;
    if(kd > 0)
        nt = ceildiv(nq - 1, kd);
    I nv_blocks = nt * (nt + 1) / 2;
    I nv = nv_blocks * kd;

    // determine sizes
    size_t size_Aband = size_t(ldab) * nq;
    size_t size_V = size_t(ldv) * nv;
    size_t size_D = nq;
    size_t size_E = nq - 1;
    size_t size_tau = nv;
    size_t size_C = size_t(ldc) * n;
    size_t size_Q = size_t(ldq) * nq;
    size_t size_R = size_t(ldr) * std::max(n, nq);
    size_t size_norm = 1;
    double errors[3] = {0, 0, 0}, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || kd < 1 || ldv < 2 * kd - 1 || ldc < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, (T*)nullptr,
                                                          ldv, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, (T*)nullptr,
                                                      ldv, (T*)nullptr, (T*)nullptr, ldc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hAband(size_Aband, 1, size_Aband, 1);
    host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1); // unused
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hQ(size_Q, 1, size_Q, 1);
    host_strided_batch_vector<T> hR(size_R, 1, size_R, 1);
    host_strided_batch_vector<S> hnorm(size_norm, 1, size_norm, 1);

    device_strided_batch_vector<T> dAband(size_Aband, 1, size_Aband, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1); // unused
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dQ(size_Q, 1, size_Q, 1);
    device_strided_batch_vector<T> dR(size_R, 1, size_R, 1);
    device_strided_batch_vector<S> dnorm(size_norm, 1, size_norm, 1);

    if(size_Aband)
        CHECK_HIP_ERROR(dAband.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_Q)
        CHECK_HIP_ERROR(dQ.memcheck());
    if(size_R)
        CHECK_HIP_ERROR(dR.memcheck());
    if(size_norm)
        CHECK_HIP_ERROR(dnorm.memcheck());

    // check quick return
    if(m == 0 || n == 0 || nq == 1)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr_hb2st(handle, side, trans, m, n, kd, dV.data(),
                                                          ldv, dTau.data(), dC.data(), ldc),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        ormtr_unmtr_hb2st_getError<T, I>(handle, side, trans, m, n, kd, dAband, ldab, dV, ldv, dTau,
                                         dC, ldc, dD, dE, dQ, ldq, dR, ldr, dnorm, hAband, hV, hTau,
                                         hC, hD, hE, hQ, hR, hnorm, errors);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        ormtr_unmtr_hb2st_getPerfData<T, I>(handle, side, trans, m, n, kd, dAband, ldab, dV, ldv,
                                            dTau, dC, ldc, dD, dE, hAband, hV, hTau, hC, hD, hE,
                                            &gpu_time_used, &cpu_time_used, hot_calls,
                                            argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 10*machine_precision as tolerance
    // Normalization by m or n already baked into error.
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, errors[0], 10);
        ROCSOLVER_TEST_CHECK(T, errors[1], 10);
        ROCSOLVER_TEST_CHECK(T, errors[2], 10);
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
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "ortho I-Q^HQ",
                                       "berror A-QBQ^H", "ferror QC-Q#C");
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
