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
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

#include <iomanip>
#include <iostream>

template <bool STRIDED, typename T, typename I, typename S, typename U, typename V>
void cholqr_checkBadArgs(const rocblas_handle handle,
                         const rocsolver_cholqr_shift cholshift,
                         const rocblas_int cholnum,
                         const I m,
                         const I n,
                         T dA,
                         const I lda,
                         const rocblas_stride stA,
                         U dW,
                         const I ldw,
                         const rocblas_stride stW,
                         S dSigma,
                         V dnr,
                         const I bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, nullptr, cholshift, cholnum, m, n, dA, lda, stA,
                                           dW, ldw, stW, dSigma, dnr, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, rocsolver_cholqr_shift(0), cholnum, m,
                                           n, dA, lda, stA, dW, ldw, stW, dSigma, dnr, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA, lda,
                                               stA, dW, ldw, stW, dSigma, dnr, (I)-1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, (T) nullptr,
                                           lda, stA, dW, ldw, stW, dSigma, dnr, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA, lda, stA,
                                           (U) nullptr, ldw, stW, dSigma, dnr, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA, lda, stA,
                                           dW, ldw, stW, (S) nullptr, dnr, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA, lda, stA,
                                           dW, ldw, stW, dSigma, (V) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, (I)0, n, (T) nullptr,
                                           lda, stA, (U) nullptr, ldw, stW, dSigma, dnr, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, (I)0, (T) nullptr,
                                           lda, stA, (U) nullptr, ldw, stW, dSigma, dnr, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA, lda,
                                               stA, dW, ldw, stW, (S) nullptr, (V) nullptr, (I)0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_cholqr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocsolver_cholqr_shift cholshift = rocsolver_cholqr_shift_computed;
    rocblas_int cholnum = 2;
    I m = 1;
    I n = 1;
    I lda = 1;
    I ldw = 1;
    rocblas_stride stA = 1;
    rocblas_stride stW = 1;
    I bc = 1;

    // memory allocations for any case
    device_strided_batch_vector<T> dW(1, 1, 1, 1);
    device_strided_batch_vector<S> dSigma(1, 1, 1, 1);
    device_strided_batch_vector<I> dnr(1, 1, 1, 1);
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dSigma.memcheck());
    CHECK_HIP_ERROR(dnr.memcheck());

#ifdef ROCSOLVER_ENABLE_CHOLQR
    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());

        // check bad arguments
        cholqr_checkBadArgs<STRIDED>(handle, cholshift, cholnum, m, n, dA.data(), lda, stA,
                                     dW.data(), ldw, stW, dSigma.data(), dnr.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());

        // check bad arguments
        cholqr_checkBadArgs<STRIDED>(handle, cholshift, cholnum, m, n, dA.data(), lda, stA,
                                     dW.data(), ldw, stW, dSigma.data(), dnr.data(), bc);
    }
#endif
}

template <bool CPU, bool GPU, typename T, typename S, typename I, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void cholqr_initData(const rocblas_handle handle,
                     const rocsolver_cholqr_shift cholshift,
                     const rocblas_int cholnum,
                     const I m,
                     const I n,
                     Td& dA,
                     const I lda,
                     const rocblas_stride stA,
                     Ud& dW,
                     const I ldw,
                     const rocblas_stride stW,
                     Sd& dSigma,
                     const I bc,
                     Th& hA,
                     Uh& hW,
                     Sh& hSigma,
                     S sigma,
                     const rocblas_int singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(auto b = 0; b < bc; ++b)
        {
            // start with a well-conditioned, diagonally-dominant matrix
            for(auto j = 0; j < n; j++)
            {
                for(auto i = 0; i < m; i++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 100;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            if(singular)
            {
                // if needed, modify the matrix to be singular, and
                // change the condition number of the matrix
                // to test different configurations of the algorithm

                // a) add two zero or repeated columns/rows at approx
                // 1/3 and 2/3 of the matrix (always same positions for
                // debbuging purposes)
                I mn = std::min(m, n);
                I i_start, i_step, j_start, j_step;
                if(mn == n)
                {
                    // case m >= n
                    i_start = 0;
                    i_step = 1;
                    j_start = mn / 3 + 1;
                    j_step = j_start + 1;
                }
                else
                {
                    // case m < n
                    i_start = mn / 3 + 1;
                    i_step = i_start + 1;
                    j_start = 0;
                    j_step = 1;
                }
                for(auto i = i_start; i < m; i += i_step)
                {
                    for(auto j = j_start; j < n; j += j_step)
                    {
                        T val = (mn == n) ? hA[b][i + (j - j_start) * lda]
                                          : hA[b][(i - i_start) + j * lda];
                        hA[b][i + j * lda] = (singular > 1) ? val : 0;
                    }
                }

                // b) diagonal loading if needed
                if(singular > 1)
                {
                    S eps = std::numeric_limits<S>::epsilon();
                    S val = 1;
                    for(auto k = 1; k < singular; ++k)
                    {
                        val *= 10;
                    }
                    val *= eps;
                    for(auto i = 0; i < mn; ++i)
                    {
                        hA[b][i + i * lda] += val;
                    }
                }
            }
        }

        // Initialize sigma array. When the shift is user provided, the value is in
        // sigma (same value for all matrices in batch). Otherwise, initialize with zero.
        for(I b = 0; b < bc; ++b)
            hSigma[b][0] = cholshift == rocsolver_cholqr_shift_provided ? sigma : S(0.0);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dSigma.transfer_from(hSigma));
    }
}

template <bool STRIDED,
          typename T,
          typename S,
          typename I,
          typename Td,
          typename Ud,
          typename Sd,
          typename Id,
          typename Th,
          typename Uh,
          typename Sh,
          typename Ih>
void cholqr_getError(const rocblas_handle handle,
                     const rocsolver_cholqr_shift cholshift,
                     const rocblas_int cholnum,
                     const I m,
                     const I n,
                     Td& dA,
                     const I lda,
                     const rocblas_stride stA,
                     Ud& dW,
                     const I ldw,
                     const rocblas_stride stW,
                     Sd& dSigma,
                     Id& dnr,
                     const I bc,
                     Th& hA,
                     Th& hARes,
                     Uh& hW,
                     Uh& hWRes,
                     Sh& hSigma,
                     Ih& hnr,
                     S sigma,
                     double* orth_error,
                     double* recon_error,
                     const rocblas_int singular,
                     const bool forgtest)
{
    S eps = std::numeric_limits<S>::epsilon();
    std::vector<double> err1(bc, 10);
    std::vector<double> err2(bc, 0);
    I mn = std::min(m, n);
    I MN = std::max(m, n);

    // input data initialization
    cholqr_initData<true, false, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw, stW,
                                    dSigma, bc, hA, hW, hSigma, sigma, singular);

    // -------------------------------------------
    // compute orthogonality error:
    // -------------------------------------------
    std::vector<T> QtQ(size_t(mn) * mn, T(0));
    std::vector<T> identity(size_t(mn) * mn, T(0));
    cpu_laset('A', mn, mn, T(0), T(1), identity.data(), mn);
    rocblas_int start;
    if(forgtest)
        start = (cholshift == rocsolver_cholqr_shift_none) ? 1 : 2;
    else
        start = cholnum;

    for(auto cnum = start; cnum <= cholnum; ++cnum)
    {
        cholqr_initData<false, true, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                        stW, dSigma, bc, hA, hW, hSigma, sigma, singular);

        // execute GPU cholqr
        CHECK_ROCBLAS_ERROR(rocsolver_cholqr(STRIDED, handle, cholshift, cnum, m, n, dA.data(), lda,
                                             stA, dW.data(), ldw, stW, dSigma.data(), dnr.data(), bc));
        CHECK_HIP_ERROR(hARes.transfer_from(dA));
        CHECK_HIP_ERROR(hnr.transfer_from(dnr));

        for(I b = 0; b < bc; ++b)
        {
            I nr = hnr[b][0];

            // orthogonality error is ||Q^T Q - I|| or ||Q Q^T - I||
            if(mn == n)
            {
                // case m >= n
                cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, nr, nr, MN,
                         T(1), hARes[b], lda, hARes[b], lda, T(0), QtQ.data(), mn);
            }
            else
            {
                // case m < n
                cpu_gemm(rocblas_operation_none, rocblas_operation_conjugate_transpose, nr, nr, MN,
                         T(1), hARes[b], lda, hARes[b], lda, T(0), QtQ.data(), mn);
            }
            err2[b] = norm_error('F', nr, nr, mn, identity.data(), QtQ.data());

            // verify that orthogonality of Q improved
            if(err1[b] > MN * eps)
                ROCSOLVER_TEST_CHECK2(err2[b], err1[b]); // check if err2 < err1

            err1[b] = err2[b];
        }
    }
    double maxerr = *std::max_element(err1.begin(), err1.end());
    *orth_error = maxerr;

    // --------------------------------------------
    // compute reconstruction error
    // --------------------------------------------
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    std::vector<T> Ac(size_t(lda) * n, T(0));

    for(I b = 0; b < bc; ++b)
    {
        // reconstruction error is ||A - QR|| / ||A|| or ||A - LQ|| / ||A||
        if(mn == n)
        {
            // case m >= n
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, m, n, mn, T(1), hARes[b], lda,
                     hWRes[b], ldw, T(0), Ac.data(), lda);
        }
        else
        {
            // case m < n
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, m, n, mn, T(1), hWRes[b], ldw,
                     hARes[b], lda, T(0), Ac.data(), lda);
        }
        err2[b] = norm_error('F', m, n, lda, hA[b], Ac.data());
    }
    maxerr = *std::max_element(err2.begin(), err2.end());
    *recon_error = maxerr;
}

template <bool STRIDED,
          typename T,
          typename S,
          typename I,
          typename Td,
          typename Ud,
          typename Sd,
          typename Id,
          typename Th,
          typename Uh,
          typename Sh,
          typename Ih>
void cholqr_getPerfData(const rocblas_handle handle,
                        const rocsolver_cholqr_shift cholshift,
                        const rocblas_int cholnum,
                        const I m,
                        const I n,
                        Td& dA,
                        const I lda,
                        const rocblas_stride stA,
                        Ud& dW,
                        const I ldw,
                        const rocblas_stride stW,
                        Sd& dSigma,
                        Id& dnr,
                        const I bc,
                        Th& hA,
                        Uh& hW,
                        Sh& hSigma,
                        Ih& hnr,
                        S sigma,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf,
                        const rocblas_int singular)
{
    I mn = std::min(m, n);
    std::vector<T> hWrk(mn);
    std::vector<T> hTau(mn);

    if(!perf)
    {
        cholqr_initData<true, false, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                        stW, dSigma, bc, hA, hW, hSigma, sigma, singular);

        // cpu-lapack performance (using GEQRF or GELQF as reference, only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            if(mn == n)
            {
                // case m >= n
                cpu_geqrf(m, n, hA[b], lda, hTau.data(), hWrk.data(), mn);
            }
            else
            {
                // case m < n
                cpu_gelqf(m, n, hA[b], lda, hTau.data(), hWrk.data(), mn);
            }
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    cholqr_initData<true, false, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw, stW,
                                    dSigma, bc, hA, hW, hSigma, sigma, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        cholqr_initData<false, true, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                        stW, dSigma, bc, hA, hW, hSigma, sigma, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA.data(),
                                             lda, stA, dW.data(), ldw, stW, dSigma.data(),
                                             dnr.data(), bc));
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

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        cholqr_initData<false, true, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                        stW, dSigma, bc, hA, hW, hSigma, sigma, singular);

        timer.start(stream);
        rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n, dA.data(), lda, stA, dW.data(),
                         ldw, stW, dSigma.data(), dnr.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_cholqr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char cholshift_char = argus.get<char>("cholshift");
    rocblas_int cholnum = argus.get<rocblas_int>("cholnum");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I mn = std::min(m, n);
    I lda = argus.get<I>("lda", m);
    I ldw = argus.get<I>("ldw", mn);
    S sigma = S(argus.get<double>("sigma", 0));
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", ldw * mn);

    rocsolver_cholqr_shift cholshift = char2rocsolver_cholqr_shift(cholshift_char);
    I bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stWRes = (argus.unit_check || argus.norm_check) ? stW : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_W = size_t(ldw) * mn;
    double err1 = 0, err2 = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_CHOLQR
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                                   (T* const*)nullptr, lda, stA, (T*)nullptr, ldw,
                                                   stW, (S*)nullptr, (I*)nullptr, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                                   (T*)nullptr, lda, stA, (T*)nullptr, ldw, stW,
                                                   (S*)nullptr, (I*)nullptr, bc),
                                  rocblas_status_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || ldw < mn || bc < 0 || cholnum < 1
                         || (cholshift != rocsolver_cholqr_shift_none && cholnum < 2));
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                                   (T* const*)nullptr, lda, stA, (T*)nullptr, ldw,
                                                   stW, (S*)nullptr, (I*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                                   (T*)nullptr, lda, stA, (T*)nullptr, ldw, stW,
                                                   (S*)nullptr, (I*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                               (T* const*)nullptr, lda, stA, (T*)nullptr, ldw, stW,
                                               (S*)nullptr, (I*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                               (T*)nullptr, lda, stA, (T*)nullptr, ldw, stW,
                                               (S*)nullptr, (I*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations for any case
    host_strided_batch_vector<T> hW(size_W, 1, stW, bc);
    host_strided_batch_vector<T> hWRes(size_WRes, 1, stWRes, bc);
    host_strided_batch_vector<S> hSigma(1, 1, 1, bc);
    host_strided_batch_vector<I> hnr(1, 1, 1, bc);
    device_strided_batch_vector<T> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<S> dSigma(1, 1, 1, bc);
    device_strided_batch_vector<I> dnr(1, 1, 1, bc);
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dSigma.memcheck());
    CHECK_HIP_ERROR(dnr.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                                   dA.data(), lda, stA, dW.data(), ldw, stW,
                                                   dSigma.data(), dnr.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                        stW, dSigma, dnr, bc, hA, hARes, hW, hWRes, hSigma, hnr,
                                        sigma, &err1, &err2, argus.singular, argus.unit_check);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                           stW, dSigma, dnr, bc, hA, hW, hSigma, hnr, sigma,
                                           &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, cholshift, cholnum, m, n,
                                                   dA.data(), lda, stA, dW.data(), ldw, stW,
                                                   dSigma.data(), dnr.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                        stW, dSigma, dnr, bc, hA, hARes, hW, hWRes, hSigma, hnr,
                                        sigma, &err1, &err2, argus.singular, argus.unit_check);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, cholshift, cholnum, m, n, dA, lda, stA, dW, ldw,
                                           stW, dSigma, dnr, bc, hA, hW, hSigma, hnr, sigma,
                                           &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf, argus.singular);
    }

    // Use tolerance based on algorithm and problem size
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, err1, 3 * std::max(m, n)); // orthogonality check
        ROCSOLVER_TEST_CHECK(T, err2, 3 * std::max(m, n)); // reconstruction check
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "ldw", "strideW", "cholshift", "cholnum",
                                       "batch_c");
                rocsolver_bench_output(m, n, lda, ldw, stW, cholshift_char, cholnum, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "ldw", "strideW", "cholshift",
                                       "cholnum", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, ldw, stW, cholshift_char, cholnum, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda", "ldw", "cholshift", "cholnum");
                rocsolver_bench_output(m, n, lda, ldw, cholshift_char, cholnum);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "orth_error", "recon_error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, err1, err2);
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
                rocsolver_bench_output(gpu_time_used, err1, err2);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_CHOLQR(...) extern template void testing_cholqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CHOLQR,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
