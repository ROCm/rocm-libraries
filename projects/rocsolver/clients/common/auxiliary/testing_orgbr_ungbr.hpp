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

template <typename T>
void orgbr_ungbr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_storev storev,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int k,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(nullptr, storev, m, n, k, dA, lda, dIpiv),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(handle, rocblas_storev(0), m, n, k, dA, lda, dIpiv),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(handle, storev, m, n, k, (T) nullptr, lda, dIpiv),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA, lda, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_orgbr_ungbr(handle, rocblas_row_wise, 0, n, 0, (T) nullptr, lda, (T) nullptr),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_orgbr_ungbr(handle, rocblas_column_wise, m, 0, 0, (T) nullptr, lda, (T) nullptr),
        rocblas_status_success);
}

template <typename T>
void testing_orgbr_ungbr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_int k = 1;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());

    // check bad arguments
    orgbr_ungbr_checkBadArgs(handle, storev, m, n, k, dA.data(), lda, dIpiv.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgbr_ungbr_initData(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Th& hA,
                          Th& hIpiv,
                          std::vector<T>& hW,
                          size_t size_W)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        size_t s = std::max(hIpiv.n(), int64_t(2));
        std::vector<S> E(s - 1);
        std::vector<S> D(s);
        std::vector<T> P(s);

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
        // and compute gebrd
        if(storev == rocblas_column_wise)
        {
            for(int i = 0; i < m; ++i)
            {
                for(int j = 0; j < k; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cpu_gebrd(m, k, hA[0], lda, D.data(), E.data(), hIpiv[0], P.data(), hW.data(), size_W);
        }
        else
        {
            for(int i = 0; i < k; ++i)
            {
                for(int j = 0; j < n; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cpu_gebrd(k, n, hA[0], lda, D.data(), E.data(), P.data(), hIpiv[0], hW.data(), size_W);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <typename T, typename Td, typename Th>
void orgbr_ungbr_getError(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Th& hA,
                          Th& hAr,
                          Th& hIpiv,
                          double* max_err)
{
    size_t size_W = std::max(std::max(m, n), k);
    std::vector<T> hW(size_W);

    // initialize data
    orgbr_ungbr_initData<true, true, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                        size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data()));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cpu_orgbr_ungbr(storev, m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void orgbr_ungbr_getPerfData(const rocblas_handle handle,
                             const rocblas_storev storev,
                             const rocblas_int m,
                             const rocblas_int n,
                             const rocblas_int k,
                             Td& dA,
                             const rocblas_int lda,
                             Td& dIpiv,
                             Th& hA,
                             Th& hIpiv,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    size_t size_W = std::max(std::max(m, n), k);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgbr_ungbr_initData<true, false, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                             size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_orgbr_ungbr(storev, m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orgbr_ungbr_initData<true, false, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                         size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgbr_ungbr_initData<false, true, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                             size_W);

        CHECK_ROCBLAS_ERROR(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data()));
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

    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgbr_ungbr_initData<false, true, T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW,
                                             size_W);

        start = get_time_us_sync(stream);
        rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_orgbr_ungbr(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char storevC = argus.get<char>("storev");
    rocblas_int m, n;
    if(storevC == 'R')
    {
        m = argus.get<rocblas_int>("m");
        n = argus.get<rocblas_int>("n", m);
    }
    else
    {
        n = argus.get<rocblas_int>("n");
        m = argus.get<rocblas_int>("m", n);
    }
    rocblas_int k = argus.get<rocblas_int>("k", std::min(m, n));
    rocblas_int lda = argus.get<rocblas_int>("lda", m);

    rocblas_storev storev = char2rocblas_storev(storevC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    // size_P could be zero in test cases that are not quick-return or invalid
    // cases setting it to one to avoid possible memory access errors in the rest
    // of the unit test
    bool row = (storev == rocblas_row_wise);
    size_t size_A = row ? size_t(lda) * n : size_t(lda) * std::max(n, k);
    size_t size_P = row ? std::max(size_t(std::min(n, k)), size_t(1))
                        : std::max(size_t(std::min(m, k)), size_t(1));

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size
        = ((m < 0 || n < 0 || k < 0 || lda < m) || (row && (m > n || m < std::min(n, k)))
           || (!row && (n > m || n < std::min(m, k))));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, (T*)nullptr, lda, (T*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, (T*)nullptr, lda, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T> hIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dIpiv(size_P, 1, size_P, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orgbr_ungbr(handle, storev, m, n, k, dA.data(), lda, dIpiv.data()),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        orgbr_ungbr_getError<T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hAr, hIpiv, &max_error);

    // collect performance data
    if(argus.timing)
        orgbr_ungbr_getPerfData<T>(handle, storev, m, n, k, dA, lda, dIpiv, hA, hIpiv,
                                   &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                   argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    rocblas_int s = row ? n : m;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("storev", "m", "n", "k", "lda");
            rocsolver_bench_output(storevC, m, n, k, lda);

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

#define EXTERN_TESTING_ORGBR_UNGBR(...) \
    extern template void testing_orgbr_ungbr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORGBR_UNGBR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
