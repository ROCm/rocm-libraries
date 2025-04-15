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

#include "common/matrix_utils/matrix_utils.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename U, typename I>
void gesdd_checkBadArgs(const rocblas_handle handle,
                        const rocblas_svect left_svect,
                        const rocblas_svect right_svect,
                        const rocblas_int m,
                        const rocblas_int n,
                        T dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
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
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, nullptr, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, rocblas_svect_overwrite, right_svect, m,
                                          n, dA, lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                          dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, rocblas_svect_overwrite, m,
                                          n, dA, lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                          dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                              stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, (T) nullptr,
                                          lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda, stA,
                                          (S) nullptr, stS, dU, ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda, stA,
                                          dS, stS, (U) nullptr, ldu, stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda, stA,
                                          dS, stS, dU, ldu, stU, (U) nullptr, ldv, stV, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA, lda,
                                          stA, dS, stS, dU, ldu, stU, dV, ldv, stV, (I) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, 0, n,
                                          (T) nullptr, lda, stA, (S) nullptr, stS, (U) nullptr, ldu,
                                          stU, dV, ldv, stV, dinfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, 0,
                                          (T) nullptr, lda, stA, (S) nullptr, stS, dU, ldu, stU,
                                          (U) nullptr, ldv, stV, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA,
                                              lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV,
                                              (I) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesdd_bad_arg()
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

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dS(1, 1, 1, 1);
        device_strided_batch_vector<T> dU(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        gesdd_checkBadArgs<STRIDED>(handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                                    dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                    dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dS(1, 1, 1, 1);
        device_strided_batch_vector<T> dU(1, 1, 1, 1);
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        gesdd_checkBadArgs<STRIDED>(handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                                    dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
                                    dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gesdd_initData(const rocblas_handle handle,
                    const rocblas_svect left_svect,
                    const rocblas_svect right_svect,
                    const rocblas_int m,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_int bc,
                    Th& hA,
                    std::vector<T>& A,
                    const bool test = true,
                    const bool singular = false)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            if(!singular)
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
            }
            else
            {
                // form a singular matrix consisting of all ones
                for(rocblas_int i = 0; i < m; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        hA[b][i + j * lda] = 1;
                    }
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
void gesdd_getError(const rocblas_handle handle,
                    const rocblas_svect left_svect,
                    const rocblas_svect right_svect,
                    const rocblas_int m,
                    const rocblas_int n,
                    Wd& dA,
                    const rocblas_int lda,
                    const rocblas_stride stA,
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
    using HMat = HostMatrix<T, rocblas_int>;
    using BDesc = typename HMat::BlockDescriptor;
    rocblas_int lwork = 5 * std::max(m, n);
    rocblas_int lrwork = (rocblas_is_complex<T> ? 5 * std::min(m, n) : 0);
    std::vector<T> work(lwork);
    std::vector<SS> rwork(lrwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    gesdd_initData<true, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // If one of `left_svect` or `right_svect` was requested, this will guarantee
    // that the other is computed as well
    CHECK_ROCBLAS_ERROR(rocsolver_gesdd(STRIDED, handle, left_svectT, right_svectT, mT, nT,
                                        dA.data(), lda, stA, dS.data(), stS, dUT.data(), lduT, stUT,
                                        dVT.data(), ldvT, stVT, dinfo.data(), bc));

    if(left_svect == rocblas_svect_none && right_svect != rocblas_svect_none)
        CHECK_HIP_ERROR(Ures.transfer_from(dUT));
    if(right_svect == rocblas_svect_none && left_svect != rocblas_svect_none)
        CHECK_HIP_ERROR(Vres.transfer_from(dVT));

    gesdd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA.data(),
                                        lda, stA, dS.data(), stS, dU.data(), ldu, stU, dV.data(),
                                        ldv, stV, dinfo.data(), bc));

    CHECK_HIP_ERROR(hSres.transfer_from(dS));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    if(left_svect == rocblas_svect_singular || left_svect == rocblas_svect_all)
        CHECK_HIP_ERROR(Ures.transfer_from(dU));
    if(right_svect == rocblas_svect_singular || right_svect == rocblas_svect_all)
        CHECK_HIP_ERROR(Vres.transfer_from(dV));

    *max_err = 0;
    *max_errv = 0;
    double err;
    const bool no_singular_vectors
        = (left_svect == rocblas_svect_none) && (right_svect == rocblas_svect_none);

    for(rocblas_int b = 0; b < bc; ++b)
    {
        // We expect gesdd to converge for all input matrices
        EXPECT_EQ(hinfoRes[b][0], 0) << "where b = " << b;
        if(hinfoRes[b][0] != 0)
        {
            *max_err += 1;
            continue;
        }
        err = 0.;

        // Number of singular values (i.e., dimension of S) is always smallest
        // number between rows and columns of input matrix A
        rocblas_int dim_S = std::min(m, n);
        rocblas_int ncols_U = dim_S;
        rocblas_int nrows_V = dim_S;

        // Only check singular values
        if(no_singular_vectors)
        {
            // CPU lapack
            cpu_gesvd(rocblas_svect_none, rocblas_svect_none, m, n, hA[b], lda, hS[b], hU[b], ldu,
                      hV[b], ldv, work.data(), lwork, rwork.data(), hinfo[b]);

            // err = ||hS - hSres||_F / ||hS||_F
            err = norm_error('F', 1, dim_S, 1, hS[b], hSres[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        // Check singular vectors and singular values
        else
        {
            // Get input matrix A
            auto AWrap = HMat::Wrap(A.data() + b * lda * n, lda, n);
            auto A = (*AWrap).block(BDesc().nrows(m).ncols(n));

            // Get computed singular values (convert singular values from type
            // S to type T, if required)
            auto svals = *HMat::Convert(hSres[b], dim_S, 1);
            auto S = HMat::Zeros(dim_S, dim_S);
            S.diag(svals);

            // Get computed eigenvectors
            auto U = (*HMat::Wrap(Ures[b], ldures, ncols_U)).block(BDesc().nrows(m).ncols(ncols_U));
            auto Vt = (*HMat::Wrap(Vres[b], ldvres, n)).block(BDesc().nrows(nrows_V).ncols(n));

            // Check orthogonality of left singular vectors if they were requested
            if(left_svect != rocblas_svect_none)
            {
                auto UE = adjoint(U) * U - HMat::Eye(ncols_U, ncols_U);
                err = UE.max_col_norm();
                *max_errv = err > *max_errv ? err : *max_errv;
            }

            // Check orthogonality of right singular vectors if they were requested
            if(right_svect != rocblas_svect_none)
            {
                auto VE = Vt * adjoint(Vt) - HMat::Eye(nrows_V, nrows_V);
                err = VE.max_col_norm();
                *max_errv = err > *max_errv ? err : *max_errv;
            }

            // Check residual error of reconstructed A
            double a_bound = 1.;
            if(m >= n)
            {
                a_bound = (adjoint(A) * A).norm();
            }
            else // (m < n)
            {
                a_bound = (A * adjoint(A)).norm();
            }
            auto AE = A - U * S * Vt;
            err = AE.norm() / a_bound;
            *max_err = err > *max_err ? err : *max_err;
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
void gesdd_getPerfData(const rocblas_handle handle,
                       const rocblas_svect left_svect,
                       const rocblas_svect right_svect,
                       const rocblas_int m,
                       const rocblas_int n,
                       Wd& dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
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
        gesdd_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_gesvd(left_svect, right_svect, m, n, hA[b], lda, hS[b], hU[b], ldu, hV[b], ldv,
                      work.data(), lwork, rwork.data(), hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gesdd_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesdd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n,
                                            dA.data(), lda, stA, dS.data(), stS, dU.data(), ldu,
                                            stU, dV.data(), ldv, stV, dinfo.data(), bc));
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
        gesdd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_gesdd(STRIDED, handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
                        dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV, dinfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gesdd(Arguments& argus)
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
    rocblas_int ldv = argus.get<rocblas_int>("ldv", (rightvC == 'A' ? n : std::min(m, n)));
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stS = argus.get<rocblas_stride>("strideS", std::min(m, n));
    rocblas_stride stU
        = argus.get<rocblas_stride>("strideU", (leftvC == 'A' ? ldu * m : ldu * std::min(m, n)));
    rocblas_stride stV = argus.get<rocblas_stride>("strideV", ldv * n);

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
            EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n,
                                                  (T* const*)nullptr, lda, stA,

                                                  (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                                                  (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr,
                                                  lda, stA, (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                                                  (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    /** Orthogonality and reconstruction errors will be computed explicitly as
     * part of `gesdd_getError` method, which may require an extra call to
     * `rocsolver_gesdd` for the cases in which only one of `left_svect` or
     * `right_svect` is requested.  If such extra call is required, initialize
     * variables `leftvT`, `rightvT`, `ldvT`, `lduT`, `mT`, and `nT`
     * accordingly.
     **/

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
            ldvT = std::min(m, n);
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
    size_t size_V = size_t(ldv) * n;
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
                size_VT = size_t(ldvT) * nT;
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
        || ((rightv == rocblas_svect_all && ldv < n)
            || (rightv == rocblas_svect_singular && ldv < std::min(m, n)));

    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n,
                                                  (T* const*)nullptr, lda, stA, (S*)nullptr, stS,
                                                  (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV,
                                                  (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr,
                                                  lda, stA, (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                                                  (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
        {
            CHECK_ALLOC_QUERY(rocsolver_gesdd(
                STRIDED, handle, leftv, rightv, m, n, (T* const*)nullptr, lda, stA, (S*)nullptr,
                stS, (T*)nullptr, ldu, stU, (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc));
            CHECK_ALLOC_QUERY(rocsolver_gesdd(
                STRIDED, handle, leftvT, rightvT, mT, nT, (T* const*)nullptr, lda, stA, (S*)nullptr,
                stS, (T*)nullptr, lduT, stUT, (T*)nullptr, ldvT, stVT, (rocblas_int*)nullptr, bc));
        }
        else
        {
            CHECK_ALLOC_QUERY(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n, (T*)nullptr,
                                              lda, stA, (S*)nullptr, stS, (T*)nullptr, ldu, stU,
                                              (T*)nullptr, ldv, stV, (rocblas_int*)nullptr, bc));
            CHECK_ALLOC_QUERY(rocsolver_gesdd(STRIDED, handle, leftvT, rightvT, mT, nT, (T*)nullptr,
                                              lda, stA, (S*)nullptr, stS, (T*)nullptr, lduT, stUT,
                                              (T*)nullptr, ldvT, stVT, (rocblas_int*)nullptr, bc));
        }

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hS(size_S, 1, stS, bc);
    host_strided_batch_vector<T> hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T> hU(size_U, 1, stU, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hSres(size_Sres, 1, stS, bc);
    host_strided_batch_vector<T> Vres(size_Vres, 1, stVres, bc);
    host_strided_batch_vector<T> Ures(size_Ures, 1, stUres, bc);
    // device
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
            EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n, dA.data(),
                                                  lda, stA, dS.data(), stS, dU.data(), ldu, stU,
                                                  dV.data(), ldv, stV, dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesdd_getError<STRIDED, T, S>(
                handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV,
                dinfo, bc, leftvT, rightvT, mT, nT, dUT, lduT, stUT, dVT, ldvT, stVT, hA, hS, hSres,
                hU, Ures, ldures, hV, Vres, ldvres, hinfo, hinfoRes, &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesdd_getPerfData<STRIDED, T, S>(handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU,
                                             ldu, stU, dV, ldv, stV, dinfo, bc, hA, hS, hU, hV,
                                             hinfo, &gpu_time_used, &cpu_time_used, hot_calls,
                                             argus.profile, argus.profile_kernels, argus.perf);
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
            EXPECT_ROCBLAS_STATUS(rocsolver_gesdd(STRIDED, handle, leftv, rightv, m, n, dA.data(),
                                                  lda, stA, dS.data(), stS, dU.data(), ldu, stU,
                                                  dV.data(), ldv, stV, dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesdd_getError<STRIDED, T, S>(
                handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU, ldu, stU, dV, ldv, stV,
                dinfo, bc, leftvT, rightvT, mT, nT, dUT, lduT, stUT, dVT, ldvT, stVT, hA, hS, hSres,
                hU, Ures, ldures, hV, Vres, ldvres, hinfo, hinfoRes, &max_error, &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesdd_getPerfData<STRIDED, T, S>(handle, leftv, rightv, m, n, dA, lda, stA, dS, stS, dU,
                                             ldu, stU, dV, ldv, stV, dinfo, bc, hA, hS, hU, hV,
                                             hinfo, &gpu_time_used, &cpu_time_used, hot_calls,
                                             argus.profile, argus.profile_kernels, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 3 * min(m, n) * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 3 * std::min(m, n));
        if(svects)
            ROCSOLVER_TEST_CHECK(T, max_errorv, 3 * std::min(m, n));
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
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "strideS",
                                       "ldu", "strideU", "ldv", "strideV", "batch_c");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, stS, ldu, stU, ldv, stV, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "strideA",
                                       "strideS", "ldu", "strideU", "ldv", "strideV", "batch_c");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, stA, stS, ldu, stU, ldv, stV, bc);
            }
            else
            {
                rocsolver_bench_output("left_svect", "right_svect", "m", "n", "lda", "ldu", "ldv");
                rocsolver_bench_output(leftvC, rightvC, m, n, lda, ldu, ldv);
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

#define EXTERN_TESTING_GESDD(...) extern template void testing_gesdd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GESDD, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
