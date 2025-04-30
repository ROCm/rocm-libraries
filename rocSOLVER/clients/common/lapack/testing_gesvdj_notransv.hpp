/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <bool STRIDED, typename T, typename S, typename SS, typename U, typename I>
void gesvdj_notransv_checkBadArgs(const rocblas_handle handle,
                                  const rocblas_svect left_svect,
                                  const rocblas_svect right_svect,
                                  const rocblas_int m,
                                  const rocblas_int n,
                                  T dA,
                                  const rocblas_int lda,
                                  const rocblas_stride stA,
                                  const SS abstol,
                                  S dResidual,
                                  const rocblas_int max_sweeps,
                                  I dSweeps,
                                  S dS,
                                  const rocblas_stride stS,
                                  U dU,
                                  const rocblas_int ldu,
                                  const rocblas_stride stU,
                                  U dV,
                                  const rocblas_int ldv,
                                  const rocblas_stride stV,
                                  I dinfo,
                                  const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, nullptr, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, dResidual, max_sweeps,
                                                    dSweeps, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                                    dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, rocblas_svect_overwrite,
                                                    right_svect, m, n, dA, lda, stA, abstol,
                                                    dResidual, max_sweeps, dSweeps, dS, stS, dU,
                                                    ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect,
                                                    rocblas_svect_overwrite, m, n, dA, lda, stA,
                                                    abstol, dResidual, max_sweeps, dSweeps, dS, stS,
                                                    dU, ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m,
                                                        n, dA, lda, stA, abstol, dResidual,
                                                        max_sweeps, dSweeps, dS, stS, dU, ldu, stU,
                                                        dV, ldv, stV, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    (T) nullptr, lda, stA, abstol, dResidual,
                                                    max_sweeps, dSweeps, dS, stS, dU, ldu, stU, dV,
                                                    ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, (S) nullptr, max_sweeps,
                                                    dSweeps, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                                    dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, dResidual, max_sweeps,
                                                    (I) nullptr, dS, stS, dU, ldu, stU, dV, ldv,
                                                    stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, dResidual, max_sweeps,
                                                    dSweeps, (S) nullptr, stS, dU, ldu, stU, dV,
                                                    ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, dResidual, max_sweeps,
                                                    dSweeps, dS, stS, (U) nullptr, ldu, stU, dV,
                                                    ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, dResidual, max_sweeps,
                                                    dSweeps, dS, stS, dU, ldu, stU, (U) nullptr,
                                                    ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n,
                                                    dA, lda, stA, abstol, dResidual, max_sweeps,
                                                    dSweeps, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                                    (I) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, 0, n,
                                                    (T) nullptr, lda, stA, abstol, dResidual,
                                                    max_sweeps, dSweeps, (S) nullptr, stS,
                                                    (U) nullptr, ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, 0,
                                                    (T) nullptr, lda, stA, abstol, dResidual,
                                                    max_sweeps, dSweeps, (S) nullptr, stS, dU, ldu,
                                                    stU, (U) nullptr, ldv, stV, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m,
                                                        n, dA, lda, stA, abstol, (S) nullptr,
                                                        max_sweeps, (I) nullptr, dS, stS, dU, ldu,
                                                        stU, dV, ldv, stV, (I) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesvdj_notransv_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_svect left_svect = rocblas_svect_singular;
    rocblas_svect right_svect = rocblas_svect_singular;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int lda = 2;
    rocblas_int ldu = 2;
    rocblas_int ldv = 2;
    rocblas_stride stA = 2;
    rocblas_stride stS = 2;
    rocblas_stride stU = 2;
    rocblas_stride stV = 2;
    rocblas_int bc = 1;

    S abstol = 0;
    rocblas_int max_sweeps = 100;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dResidual(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, 1);
        device_strided_batch_vector<S> dS(1, 1, 1, 1);
        device_strided_batch_vector<T> dU(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dResidual.memcheck());
        CHECK_HIP_ERROR(dSweeps.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        gesvdj_notransv_checkBadArgs<STRIDED>(handle, left_svect, right_svect, m, n, dA.data(), lda,
                                              stA, abstol, dResidual.data(), max_sweeps,
                                              dSweeps.data(), dS.data(), stS, dU.data(), ldu, stU,
                                              dV.data(), ldv, stV, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dResidual(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, 1);
        device_strided_batch_vector<S> dS(1, 1, 1, 1);
        device_strided_batch_vector<T> dU(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dResidual.memcheck());
        CHECK_HIP_ERROR(dSweeps.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        gesvdj_notransv_checkBadArgs<STRIDED>(handle, left_svect, right_svect, m, n, dA.data(), lda,
                                              stA, abstol, dResidual.data(), max_sweeps,
                                              dSweeps.data(), dS.data(), stS, dU.data(), ldu, stU,
                                              dV.data(), ldv, stV, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gesvdj_notransv_initData(const rocblas_handle handle,
                              const rocblas_svect left_svect,
                              const rocblas_svect right_svect,
                              const rocblas_int m,
                              const rocblas_int n,
                              Td& dA,
                              const rocblas_int lda,
                              const rocblas_int bc,
                              Th& hA,
                              std::vector<T>& A,
                              bool test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test && (left_svect != rocblas_svect_none || right_svect != rocblas_svect_none))
            {
                for(rocblas_int i = 0; i < m; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                        A[b * lda * n + i + j * lda] = hA[b][i + j * lda];
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED,
          typename T,
          typename SS,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvdj_notransv_getError(const rocblas_handle handle,
                              const rocblas_svect left_svect,
                              const rocblas_svect right_svect,
                              const rocblas_int m,
                              const rocblas_int n,
                              Wd& dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              const SS abstol,
                              Td& dResidual,
                              const rocblas_int max_sweeps,
                              Id& dSweeps,
                              Td& dS,
                              const rocblas_stride stS,
                              Ud& dU,
                              const rocblas_int ldu,
                              const rocblas_stride stU,
                              Ud& dV,
                              const rocblas_int ldv,
                              const rocblas_stride stV,
                              Id& dinfo,
                              const rocblas_int bc,
                              const rocblas_svect left_svectT,
                              const rocblas_svect right_svectT,
                              const rocblas_int mT,
                              const rocblas_int nT,
                              Ud& dUT,
                              const rocblas_int lduT,
                              const rocblas_stride stUT,
                              Ud& dVT,
                              const rocblas_int ldvT,
                              const rocblas_stride stVT,
                              Wh& hA,
                              Th& hResidualRes,
                              Ih& hSweepsRes,
                              Th& hS,
                              Th& hSres,
                              Uh& hU,
                              Uh& Ures,
                              const rocblas_int ldures,
                              Uh& hV,
                              Uh& Vres,
                              const rocblas_int ldvres,
                              Ih& hinfo,
                              Ih& hinfoRes,
                              double* max_err,
                              double* max_errv)
{
    rocblas_int lwork = 5 * std::max(m, n);
    rocblas_int lrwork = (rocblas_is_complex<T> ? 5 * std::min(m, n) : 0);
    std::vector<T> work(lwork);
    std::vector<SS> rwork(lrwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    gesvdj_notransv_initData<true, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA,
                                            A);

    // execute computations:
    // complementary execution to compute all singular vectors if needed
    CHECK_ROCBLAS_ERROR(rocsolver_gesvdj_notransv(
        STRIDED, handle, left_svectT, right_svectT, mT, nT, dA.data(), lda, stA, abstol,
        dResidual.data(), max_sweeps, dSweeps.data(), dS.data(), stS, dUT.data(), lduT, stUT,
        dVT.data(), ldvT, stVT, dinfo.data(), bc));

    if(left_svect == rocblas_svect_none && right_svect != rocblas_svect_none)
        CHECK_HIP_ERROR(Ures.transfer_from(dUT));
    if(right_svect == rocblas_svect_none && left_svect != rocblas_svect_none)
        CHECK_HIP_ERROR(Vres.transfer_from(dVT));

    gesvdj_notransv_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA,
                                             A);

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_gesvd(rocblas_svect_none, rocblas_svect_none, m, n, hA[b], lda, hS[b], hU[b], ldu,
                  hV[b], ldv, work.data(), lwork, rwork.data(), hinfo[b]);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gesvdj_notransv(
        STRIDED, handle, left_svect, right_svect, m, n, dA.data(), lda, stA, abstol,
        dResidual.data(), max_sweeps, dSweeps.data(), dS.data(), stS, dU.data(), ldu, stU,
        dV.data(), ldv, stV, dinfo.data(), bc));

    CHECK_HIP_ERROR(hResidualRes.transfer_from(dResidual));
    CHECK_HIP_ERROR(hSweepsRes.transfer_from(dSweeps));
    CHECK_HIP_ERROR(hSres.transfer_from(dS));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    if(left_svect == rocblas_svect_singular || left_svect == rocblas_svect_all)
        CHECK_HIP_ERROR(Ures.transfer_from(dU));
    if(right_svect == rocblas_svect_singular || right_svect == rocblas_svect_all)
        CHECK_HIP_ERROR(Vres.transfer_from(dV));

    // Check info for non-convergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    // Also check validity of residual
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_GE(hResidualRes[b][0], 0) << "where b = " << b;
        if(hResidualRes[b][0] < 0)
            *max_err += 1;
    }

    // Also check validity of sweeps
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_GE(hResidualRes[b][0], 0) << "where b = " << b;
        EXPECT_LE(hSweepsRes[b][0], max_sweeps) << "where b = " << b;
        if(hSweepsRes[b][0] < 0 || hSweepsRes[b][0] > max_sweeps)
            *max_err += 1;
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err;
    *max_errv = 0;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        // error is ||hS - hSres||
        err = norm_error('F', 1, std::min(m, n), 1, hS[b], hSres[b]);
        *max_err = err > *max_err ? err : *max_err;

        // Check the singular vectors if required
        if(hinfo[b][0] == 0 && (left_svect != rocblas_svect_none || right_svect != rocblas_svect_none))
        {
            err = 0;
            // check singular vectors implicitly (A*v_k = s_k*u_k)
            for(rocblas_int k = 0; k < std::min(m, n); ++k)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    T tmp = 0;
                    for(rocblas_int j = 0; j < n; ++j)
                        tmp += A[b * lda * n + i + j * lda] * Vres[b][j + k * ldvres];
                    tmp -= hSres[b][k] * Ures[b][i + k * ldures];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
            err = std::sqrt(err) / double(snorm('F', m, n, A.data() + b * lda * n, lda));
            *max_errv = err > *max_errv ? err : *max_errv;
        }
    }
}

template <bool STRIDED,
          typename T,
          typename SS,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvdj_notransv_getPerfData(const rocblas_handle handle,
                                 const rocblas_svect left_svect,
                                 const rocblas_svect right_svect,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 Wd& dA,
                                 const rocblas_int lda,
                                 const rocblas_stride stA,
                                 const SS abstol,
                                 Td& dResidual,
                                 const rocblas_int max_sweeps,
                                 Id& dSweeps,
                                 Td& dS,
                                 const rocblas_stride stS,
                                 Ud& dU,
                                 const rocblas_int ldu,
                                 const rocblas_stride stU,
                                 Ud& dV,
                                 const rocblas_int ldv,
                                 const rocblas_stride stV,
                                 Id& dinfo,
                                 const rocblas_int bc,
                                 Wh& hA,
                                 Th& hS,
                                 Uh& hU,
                                 Uh& hV,
                                 Ih& hinfo,
                                 double* gpu_time_used,
                                 double* cpu_time_used,
                                 const rocblas_int hot_calls,
                                 const int profile,
                                 const bool profile_kernels,
                                 const bool perf)
{
    rocblas_int lwork = 5 * std::max(m, n);
    rocblas_int lrwork = 5 * std::min(m, n);
    std::vector<T> work(lwork);
    std::vector<SS> rwork(lrwork);
    std::vector<T> A;

    if(!perf)
    {
        gesvdj_notransv_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc,
                                                 hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_gesvd(left_svect, right_svect, m, n, hA[b], lda, hS[b], hU[b], ldu, hV[b], ldv,
                      work.data(), lwork, rwork.data(), hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gesvdj_notransv_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA,
                                             A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesvdj_notransv_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc,
                                                 hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_gesvdj_notransv(
            STRIDED, handle, left_svect, right_svect, m, n, dA.data(), lda, stA, abstol,
            dResidual.data(), max_sweeps, dSweeps.data(), dS.data(), stS, dU.data(), ldu, stU,
            dV.data(), ldv, stV, dinfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

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
        gesvdj_notransv_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc,
                                                 hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_gesvdj_notransv(STRIDED, handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                                  abstol, dResidual.data(), max_sweeps, dSweeps.data(), dS.data(),
                                  stS, dU.data(), ldu, stU, dV.data(), ldv, stV, dinfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesvdj_notransv(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char leftvC = argus.get<char>("left_svect");
    char rightvC = argus.get<char>("right_svect");
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_int ldu = argus.get<rocblas_int>("ldu", m);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stS = argus.get<rocblas_stride>("strideS", std::min(m, n));
    rocblas_stride stU
        = argus.get<rocblas_stride>("strideU", (leftvC == 'A' ? ldu * m : ldu * std::min(m, n)));
    rocblas_stride stV
        = argus.get<rocblas_stride>("strideV", (rightvC == 'A' ? ldv * n : ldv * std::min(m, n)));

    S abstol = S(argus.get<double>("abstol", 0));
    rocblas_int max_sweeps = argus.get<rocblas_int>("max_sweeps", 100);

    rocblas_svect leftv = char2rocblas_svect(leftvC);
    rocblas_svect rightv = char2rocblas_svect(rightvC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if((rightv != rocblas_svect_none && rightv != rocblas_svect_singular && rightv != rocblas_svect_all)
       || (leftv != rocblas_svect_none && leftv != rocblas_svect_singular
           && leftv != rocblas_svect_all))
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdj_notransv(STRIDED, handle, leftv, rightv, m, n, (T* const*)nullptr,
                                          lda, stA, abstol, (S*)nullptr, max_sweeps,
                                          (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu,
                                          stU, (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdj_notransv(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr, lda,
                                          stA, abstol, (S*)nullptr, max_sweeps,
                                          (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu,
                                          stU, (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    /** TESTING OF SINGULAR VECTORS IS DONE IMPLICITLY, NOT EXPLICITLY COMPARING
        WITH LAPACK. SO, WE ALWAYS NEED TO COMPUTE THE SAME NUMBER OF ELEMENTS OF
        THE RIGHT AND LEFT VECTORS. WHILE DOING THIS, IF MORE VECTORS THAN THE
        SPECIFIED IN THE MAIN CALL NEED TO BE COMPUTED, WE DO SO WITH AN EXTRA CALL **/

    rocblas_svect leftvT = rocblas_svect_none;
    rocblas_svect rightvT = rocblas_svect_none;
    rocblas_int ldvT = 1;
    rocblas_int lduT = 1;
    rocblas_int mT = 0;
    rocblas_int nT = 0;
    bool svects = (leftv != rocblas_svect_none || rightv != rocblas_svect_none);

    if(svects)
    {
        if(leftv == rocblas_svect_none)
        {
            leftvT = rocblas_svect_singular;
            lduT = m;
            mT = m;
            nT = n;
        }
        if(rightv == rocblas_svect_none)
        {
            rightvT = rocblas_svect_singular;
            ldvT = n;
            mT = m;
            nT = n;
        }
    }

    // determine sizes
    rocblas_int ldures = 1;
    rocblas_int ldvres = 1;
    size_t size_Sres = 0;
    size_t size_Ures = 0;
    size_t size_Vres = 0;
    size_t size_UT = 0;
    size_t size_VT = 0;
    size_t size_A = size_t(lda) * n;
    size_t size_S = size_t(std::min(m, n));
    size_t size_U = (leftvC == 'A' ? size_t(ldu) * m : size_t(ldu) * std::min(m, n));
    size_t size_V = (rightvC == 'A' ? size_t(ldv) * n : size_t(ldv) * std::min(m, n));
    if(argus.unit_check || argus.norm_check)
    {
        size_Sres = size_S;
        if(svects)
        {
            if(leftv == rocblas_svect_none)
            {
                size_UT = size_t(lduT) * std::min(mT, nT);
                size_Ures = size_UT;
                ldures = lduT;
            }
            else
            {
                size_Ures = size_U;
                ldures = ldu;
            }

            if(rightv == rocblas_svect_none)
            {
                size_VT = size_t(ldvT) * std::min(mT, nT);
                size_Vres = size_VT;
                ldvres = ldvT;
            }
            else
            {
                size_Vres = size_V;
                ldvres = ldv;
            }
        }
    }
    rocblas_stride stUT = size_UT;
    rocblas_stride stVT = size_VT;
    rocblas_stride stUres = size_Ures;
    rocblas_stride stVres = size_Vres;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0, max_errorv = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || bc < 0)
        || ((leftv == rocblas_svect_all || leftv == rocblas_svect_singular) && ldu < m)
        || ((rightv == rocblas_svect_all || rightv == rocblas_svect_singular) && ldv < n);

    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdj_notransv(STRIDED, handle, leftv, rightv, m, n, (T* const*)nullptr,
                                          lda, stA, abstol, (S*)nullptr, max_sweeps,
                                          (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu,
                                          stU, (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdj_notransv(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr, lda,
                                          stA, abstol, (S*)nullptr, max_sweeps,
                                          (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu,
                                          stU, (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
        {
            CHECK_ALLOC_QUERY(rocsolver_gesvdj_notransv(
                STRIDED, handle, leftv, rightv, m, n, (T* const*)nullptr, lda, stA, abstol,
                (S*)nullptr, max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu,
                stU, (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc));
            CHECK_ALLOC_QUERY(rocsolver_gesvdj_notransv(
                STRIDED, handle, leftvT, rightvT, mT, nT, (T* const*)nullptr, lda, stA, abstol,
                (S*)nullptr, max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, lduT,
                stUT, (T*)nullptr, ldvT, stVT, (rocblas_int*)nullptr, bc));
        }
        else
        {
            CHECK_ALLOC_QUERY(rocsolver_gesvdj_notransv(
                STRIDED, handle, leftv, rightv, m, n, (T*)nullptr, lda, stA, abstol, (S*)nullptr,
                max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc));
            CHECK_ALLOC_QUERY(rocsolver_gesvdj_notransv(
                STRIDED, handle, leftvT, rightvT, mT, nT, (T*)nullptr, lda, stA, abstol,
                (S*)nullptr, max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stS, (T*)nullptr, lduT,
                stUT, (T*)nullptr, ldvT, stVT, (rocblas_int*)nullptr, bc));
        }

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hResidualRes(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hSweepsRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hS(size_S, 1, stS, bc);
    host_strided_batch_vector<T> hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T> hU(size_U, 1, stU, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hSres(size_Sres, 1, stS, bc);
    host_strided_batch_vector<T> Vres(size_Vres, 1, stVres, bc);
    host_strided_batch_vector<T> Ures(size_Ures, 1, stUres, bc);
    // device
    device_strided_batch_vector<S> dResidual(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, bc);
    device_strided_batch_vector<S> dS(size_S, 1, stS, bc);
    device_strided_batch_vector<T> dV(size_V, 1, stV, bc);
    device_strided_batch_vector<T> dU(size_U, 1, stU, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<T> dVT(size_VT, 1, stVT, bc);
    device_strided_batch_vector<T> dUT(size_UT, 1, stUT, bc);
    if(size_VT)
        CHECK_HIP_ERROR(dVT.memcheck());
    if(size_UT)
        CHECK_HIP_ERROR(dUT.memcheck());
    CHECK_HIP_ERROR(dResidual.memcheck());
    CHECK_HIP_ERROR(dSweeps.memcheck());
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_U)
        CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || m == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdj_notransv(STRIDED, handle, leftv, rightv, m, n, dA.data(), lda, stA,
                                          abstol, dResidual.data(), max_sweeps, dSweeps.data(),
                                          dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                          dinfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvdj_notransv_getError<STRIDED, T>(
                handle, leftv, rightv, m, n, dA, lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc, leftvT, rightvT, mT, nT, dUT, lduT,
                stUT, dVT, ldvT, stVT, hA, hResidualRes, hSweepsRes, hS, hSres, hU, Ures, ldures,
                hV, Vres, ldvres, hinfo, hinfoRes, &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvdj_notransv_getPerfData<STRIDED, T>(
                handle, leftv, rightv, m, n, dA, lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc, hA, hS, hU, hV, hinfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || m == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gesvdj_notransv(STRIDED, handle, leftv, rightv, m, n, dA.data(), lda, stA,
                                          abstol, dResidual.data(), max_sweeps, dSweeps.data(),
                                          dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                          dinfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvdj_notransv_getError<STRIDED, T>(
                handle, leftv, rightv, m, n, dA, lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc, leftvT, rightvT, mT, nT, dUT, lduT,
                stUT, dVT, ldvT, stVT, hA, hResidualRes, hSweepsRes, hS, hSres, hU, Ures, ldures,
                hV, Vres, ldvres, hinfo, hinfoRes, &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvdj_notransv_getPerfData<STRIDED, T>(
                handle, leftv, rightv, m, n, dA, lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc, hA, hS, hU, hV, hinfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 2 * min(m, n) * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * std::min(m, n));
        if(svects)
            ROCSOLVER_TEST_CHECK(T, max_errorv, 2 * std::min(m, n));
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(svects)
            max_error = (max_error >= max_errorv) ? max_error : max_errorv;

        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "abstol",
                                       "max_sweeps", "strideS", "ldu", "strideU", "ldv", "strideV",
                                       "batch_c");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, abstol, max_sweeps, stS, ldu,
                                       stU, ldv, stV, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "strideA",
                                       "abstol", "max_sweeps", "strideS", "ldu", "strideU", "ldv",
                                       "strideV", "batch_c");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, stA, abstol, max_sweeps, stS,
                                       ldu, stU, ldv, stV, bc);
            }
            else
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "abstol",
                                       "max_sweeps", "ldu", "ldv");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, abstol, max_sweeps, ldu, ldv);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
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
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}
