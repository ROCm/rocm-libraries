/* **************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template <bool STRIDED, typename T, typename S, typename SS, typename U>
void syevj_heevj_checkBadArgs(const rocblas_handle handle,
                              const rocblas_esort esort,
                              const rocblas_evect evect,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              const SS abstol,
                              S dResidual,
                              const rocblas_int max_sweeps,
                              U dSweeps,
                              S dW,
                              const rocblas_stride stW,
                              U dInfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, nullptr, esort, evect, uplo, n, dA, lda,
                                                stA, abstol, dResidual, max_sweeps, dSweeps, dW,
                                                stW, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, rocblas_esort(0), evect, uplo, n,
                                                dA, lda, stA, abstol, dResidual, max_sweeps,
                                                dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, rocblas_evect(0), uplo, n,
                                                dA, lda, stA, abstol, dResidual, max_sweeps,
                                                dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, rocblas_fill_full, n,
                                                dA, lda, stA, abstol, dResidual, max_sweeps,
                                                dSweeps, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA, lda,
                                                    stA, abstol, dResidual, max_sweeps, dSweeps, dW,
                                                    stW, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, (T) nullptr,
                                                lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                                                dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA, lda,
                                                stA, abstol, (S) nullptr, max_sweeps, dSweeps, dW,
                                                stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA, lda,
                                                stA, abstol, dResidual, max_sweeps, (U) nullptr, dW,
                                                stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA, lda,
                                                stA, abstol, dResidual, max_sweeps, dSweeps,
                                                (S) nullptr, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA, lda,
                                                stA, abstol, dResidual, max_sweeps, dSweeps, dW,
                                                stW, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, 0, (T) nullptr,
                                                lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                                                (S) nullptr, stW, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA, lda,
                                                    stA, abstol, (S) nullptr, max_sweeps,
                                                    (U) nullptr, dW, stW, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syevj_heevj_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect evect = rocblas_evect_none;
    rocblas_esort esort = rocblas_esort_ascending;
    rocblas_fill uplo = rocblas_fill_lower;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stW = 1;
    rocblas_int bc = 1;

    S abstol = 0;
    rocblas_int max_sweeps = 100;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dResidual(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dResidual.memcheck());
        CHECK_HIP_ERROR(dSweeps.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        syevj_heevj_checkBadArgs<STRIDED>(handle, esort, evect, uplo, n, dA.data(), lda, stA,
                                          abstol, dResidual.data(), max_sweeps, dSweeps.data(),
                                          dW.data(), stW, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dResidual(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dResidual.memcheck());
        CHECK_HIP_ERROR(dSweeps.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        syevj_heevj_checkBadArgs<STRIDED>(handle, esort, evect, uplo, n, dA.data(), lda, stA,
                                          abstol, dResidual.data(), max_sweeps, dSweeps.data(),
                                          dW.data(), stW, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevj_heevj_initData(const rocblas_handle handle,
                          const rocblas_evect evect,
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

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test)
            {
                for(rocblas_int i = 0; i < n; i++)
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

template <bool STRIDED, typename T, typename S, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void syevj_heevj_getError(const rocblas_handle handle,
                          const rocblas_esort esort,
                          const rocblas_evect evect,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          const S abstol,
                          Sd& dResidual,
                          const rocblas_int max_sweeps,
                          Id& dSweeps,
                          Sd& dW,
                          const rocblas_stride stW,
                          Id& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Sh& hResidualRes,
                          Ih& hSweepsRes,
                          Sh& hW,
                          Sh& hWRes,
                          Ih& hInfo,
                          Ih& hInfoRes,
                          double* max_err)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    S atol = (abstol <= 0) ? get_epsilon<S>() : abstol;

    int lwork = (COMPLEX ? 2 * n - 1 : 0);
    int lrwork = 3 * n - 1;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    syevj_heevj_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA.data(),
                                              lda, stA, abstol, dResidual.data(), max_sweeps,
                                              dSweeps.data(), dW.data(), stW, dInfo.data(), bc));

    CHECK_HIP_ERROR(hResidualRes.transfer_from(dResidual));
    CHECK_HIP_ERROR(hSweepsRes.transfer_from(dSweeps));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect == rocblas_evect_original)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_syev_heev(evect, uplo, n, hA[b], lda, hW[b], work.data(), lwork, rwork.data(), lrwork,
                      hInfo[b]);

    // (We expect the used input matrices to always converge)
    // Check info for non-convergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
        if(hInfoRes[b][0] != 0)
            *max_err += 1;
    }

    // Also check validity of residual
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_GE(hResidualRes[b][0], 0) << "where b = " << b;
        if(hResidualRes[b][0] < 0)
            *max_err += 1;
        else
        {
            S threshold = snorm('F', n, n, A.data() + b * lda * n, lda) * atol;
            EXPECT_LE(hResidualRes[b][0], threshold) << "where b = " << b;
            if(hResidualRes[b][0] > threshold)
                *max_err += 1;
        }
    }

    // Also check validity of sweeps
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_GE(hSweepsRes[b][0], 0) << "where b = " << b;
        EXPECT_LE(hSweepsRes[b][0], max_sweeps) << "where b = " << b;
        if(hSweepsRes[b][0] < 0 || hSweepsRes[b][0] > max_sweeps)
            *max_err += 1;
    }

    double err = 0;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(evect != rocblas_evect_original)
        {
            // only eigenvalues needed; can compare with LAPACK
            // (no need to test the non-sorted case --lapack return sorted eigenvalues--)

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hInfo[b][0] == 0 && esort == rocblas_esort_ascending)
                err = norm_error('F', 1, n, 1, hW[b], hWRes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hInfo[b][0] == 0)
            {
                // multiply A with each of the n eigenvectors and divide by corresponding
                // eigenvalues
                T alpha;
                T beta = 0;
                for(int j = 0; j < n; j++)
                {
                    alpha = T(1) / hWRes[b][j];
                    cpu_symv_hemv(uplo, n, alpha, A.data() + b * lda * n, lda, hARes[b] + j * lda,
                                  1, beta, hA[b] + j * lda, 1);
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err = norm_error('F', n, n, lda, hA[b], hARes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename S, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void syevj_heevj_getPerfData(const rocblas_handle handle,
                             const rocblas_esort esort,
                             const rocblas_evect evect,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             const S abstol,
                             Sd& dResidual,
                             const rocblas_int max_sweeps,
                             Id& dSweeps,
                             Sd& dW,
                             const rocblas_stride stW,
                             Id& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Sh& hW,
                             Ih& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    int lwork = (COMPLEX ? 2 * n - 1 : 0);
    int lrwork = 3 * n - 1;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<T> A;

    if(!perf)
    {
        syevj_heevj_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_syev_heev(evect, uplo, n, hA[b], lda, hW[b], work.data(), lwork, rwork.data(),
                          lrwork, hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syevj_heevj_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syevj_heevj_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA.data(),
                                                  lda, stA, abstol, dResidual.data(), max_sweeps,
                                                  dSweeps.data(), dW.data(), stW, dInfo.data(), bc));
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
        syevj_heevj_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, dA.data(), lda, stA, abstol,
                              dResidual.data(), max_sweeps, dSweeps.data(), dW.data(), stW,
                              dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syevj_heevj(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char evectC = argus.get<char>("evect");
    char esortC = argus.get<char>("esort");
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideD", n);

    S abstol = S(argus.get<double>("abstol", 0));
    rocblas_int max_sweeps = argus.get<rocblas_int>("max_sweeps", 100);

    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_esort esort = char2rocblas_esort(esortC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo == rocblas_fill_full || evect == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, (T* const*)nullptr, lda,
                                      stA, abstol, (S*)nullptr, max_sweeps, (rocblas_int*)nullptr,
                                      (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n,
                                                        (T*)nullptr, lda, stA, abstol, (S*)nullptr,
                                                        max_sweeps, (rocblas_int*)nullptr,
                                                        (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_W = n;
    size_t size_Ares = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_Wres = (argus.unit_check || argus.norm_check) ? size_W : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n, (T* const*)nullptr, lda,
                                      stA, abstol, (S*)nullptr, max_sweeps, (rocblas_int*)nullptr,
                                      (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n,
                                                        (T*)nullptr, lda, stA, abstol, (S*)nullptr,
                                                        max_sweeps, (rocblas_int*)nullptr,
                                                        (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n,
                                                    (T* const*)nullptr, lda, stA, abstol,
                                                    (S*)nullptr, max_sweeps, (rocblas_int*)nullptr,
                                                    (S*)nullptr, stW, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_syevj_heevj(
                STRIDED, handle, esort, evect, uplo, n, (T*)nullptr, lda, stA, abstol, (S*)nullptr,
                max_sweeps, (rocblas_int*)nullptr, (S*)nullptr, stW, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hResidualRes(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hSweepsRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hW(size_W, 1, stW, bc);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hWRes(size_Wres, 1, stW, bc);
    // device
    device_strided_batch_vector<S> dResidual(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dSweeps(1, 1, 1, bc);
    device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    CHECK_HIP_ERROR(dResidual.memcheck());
    CHECK_HIP_ERROR(dSweeps.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_Ares, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n,
                                                        dA.data(), lda, stA, abstol,
                                                        dResidual.data(), max_sweeps, dSweeps.data(),
                                                        dW.data(), stW, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevj_heevj_getError<STRIDED, T>(handle, esort, evect, uplo, n, dA, lda, stA, abstol,
                                             dResidual, max_sweeps, dSweeps, dW, stW, dInfo, bc, hA,
                                             hARes, hResidualRes, hSweepsRes, hW, hWRes, hInfo,
                                             hInfoRes, &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevj_heevj_getPerfData<STRIDED, T>(
                handle, esort, evect, uplo, n, dA, lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                dW, stW, dInfo, bc, hA, hW, hInfo, &gpu_time_used, &cpu_time_used, hot_calls,
                argus.profile, argus.profile_kernels, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_Ares, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syevj_heevj(STRIDED, handle, esort, evect, uplo, n,
                                                        dA.data(), lda, stA, abstol,
                                                        dResidual.data(), max_sweeps, dSweeps.data(),
                                                        dW.data(), stW, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevj_heevj_getError<STRIDED, T>(handle, esort, evect, uplo, n, dA, lda, stA, abstol,
                                             dResidual, max_sweeps, dSweeps, dW, stW, dInfo, bc, hA,
                                             hARes, hResidualRes, hSweepsRes, hW, hWRes, hInfo,
                                             hInfoRes, &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevj_heevj_getPerfData<STRIDED, T>(
                handle, esort, evect, uplo, n, dA, lda, stA, abstol, dResidual, max_sweeps, dSweeps,
                dW, stW, dInfo, bc, hA, hW, hInfo, &gpu_time_used, &cpu_time_used, hot_calls,
                argus.profile, argus.profile_kernels, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("esort", "evect", "uplo", "n", "lda", "abstol", "max_sweeps",
                                       "strideW", "batch_c");
                rocsolver_bench_output(esortC, evectC, uploC, n, lda, abstol, max_sweeps, stW, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("esort", "evect", "uplo", "n", "lda", "strideA", "abstol",
                                       "max_sweeps", "strideW", "batch_c");
                rocsolver_bench_output(esortC, evectC, uploC, n, lda, stA, abstol, max_sweeps, stW,
                                       bc);
            }
            else
            {
                rocsolver_bench_output("esort", "evect", "uplo", "n", "lda", "abstol", "max_sweeps");
                rocsolver_bench_output(esortC, evectC, uploC, n, lda, abstol, max_sweeps);
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

#define EXTERN_TESTING_SYEVJ_HEEVJ(...) \
    extern template void testing_syevj_heevj<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYEVJ_HEEVJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
