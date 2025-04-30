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

template <bool STRIDED, bool SYTRF, typename T, typename U>
void sytf2_sytrf_checkBadArgs(const rocblas_handle handle,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              U dinfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, nullptr, uplo, n, dA, lda, stA,
                                                dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, rocblas_fill_full, n, dA,
                                                lda, stA, dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                    dIpiv, stP, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T) nullptr, lda,
                                                stA, dIpiv, stP, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                (U) nullptr, stP, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                dIpiv, stP, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, 0, (T) nullptr, lda,
                                                stA, (U) nullptr, stP, dinfo, bc),
                          rocblas_status_success);
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                    dIpiv, stP, (U) nullptr, 0),
                              rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA, lda, stA,
                                                    dIpiv, stP, dinfo, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool SYTRF, typename T>
void testing_sytf2_sytrf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sytf2_sytrf_checkBadArgs<STRIDED, SYTRF>(handle, uplo, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sytf2_sytrf_checkBadArgs<STRIDED, SYTRF>(handle, uplo, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sytf2_sytrf_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv,
                          Uh& hInfo,
                          const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(rocblas_int i = 0; i < n / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    tmp = hA[b][i + j * lda];
                    hA[b][i + j * lda] = hA[b][n - 1 - i + j * lda];
                    hA[b][n - 1 - i + j * lda] = tmp;
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // add some singularities
                // always the same elements for debugging purposes
                // the algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool SYTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sytf2_sytrf_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          Uh& hIpivRes,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const bool singular)
{
    int lwork = (SYTRF ? 64 * n : 0);
    std::vector<T> work(lwork);

    // input data initialization
    sytf2_sytrf_initData<true, true, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                        hIpiv, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        SYTRF ? cpu_sytrf(uplo, n, hA[b], lda, hIpiv[b], work.data(), lwork, hInfo[b])
              : cpu_sytf2(uplo, n, hA[b], lda, hIpiv[b], hInfo[b]);
    }

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', n, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;

        // also check pivoting (count the number of incorrect pivots)
        err = 0;
        for(rocblas_int i = 0; i < n; ++i)
        {
            EXPECT_EQ(hIpiv[b][i], hIpivRes[b][i]) << "where b = " << b << ", i = " << i;
            if(hIpiv[b][i] != hIpivRes[b][i])
                err++;
        }
        *max_err = err > *max_err ? err : *max_err;
    }

    // also check info
    err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <bool STRIDED, bool SYTRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sytf2_sytrf_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             Ud& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             Uh& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const bool singular)
{
    int lwork = (SYTRF ? 64 * n : 0);
    std::vector<T> work(lwork);

    if(!perf)
    {
        sytf2_sytrf_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc,
                                             hA, hIpiv, hInfo, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            SYTRF ? cpu_sytrf(uplo, n, hA[b], lda, hIpiv[b], work.data(), lwork, hInfo[b])
                  : cpu_sytf2(uplo, n, hA[b], lda, hIpiv[b], hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sytf2_sytrf_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA,
                                         hIpiv, hInfo, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sytf2_sytrf_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc,
                                             hA, hIpiv, hInfo, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(), lda,
                                                  stA, dIpiv.data(), stP, dInfo.data(), bc));
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
        sytf2_sytrf_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc,
                                             hA, hIpiv, hInfo, singular);

        start = get_time_us_sync(stream);
        rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(), lda, stA, dIpiv.data(),
                              stP, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool SYTRF, typename T>
void testing_sytf2_sytrf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", n);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // check non-supported values
    if(uplo == rocblas_fill_full)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, stP,
                                                        (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T*)nullptr,
                                                        lda, stA, (rocblas_int*)nullptr, stP,
                                                        (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(
                rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T* const*)nullptr, lda, stA,
                                      (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, (T*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytf2_sytrf_getError<STRIDED, SYTRF, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP,
                                                    dInfo, bc, hA, hARes, hIpiv, hIpivRes, hInfo,
                                                    hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sytf2_sytrf_getPerfData<STRIDED, SYTRF, T>(
                handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytf2_sytrf(STRIDED, SYTRF, handle, uplo, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytf2_sytrf_getError<STRIDED, SYTRF, T>(handle, uplo, n, dA, lda, stA, dIpiv, stP,
                                                    dInfo, bc, hA, hARes, hIpiv, hIpivRes, hInfo,
                                                    hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sytf2_sytrf_getPerfData<STRIDED, SYTRF, T>(
                handle, uplo, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo,
                &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                argus.perf, argus.singular);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "lda");
                rocsolver_bench_output(uploC, n, lda);
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

#define EXTERN_TESTING_SYTF2_SYTRF(...) \
    extern template void testing_sytf2_sytrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYTF2_SYTRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
