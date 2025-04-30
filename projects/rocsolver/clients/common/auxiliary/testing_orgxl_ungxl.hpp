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

template <bool GQL, typename T>
void orgxl_ungxl_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int k,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxl_ungxl(GQL, nullptr, m, n, k, dA, lda, dIpiv),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxl_ungxl(GQL, handle, m, n, k, (T) nullptr, lda, dIpiv),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxl_ungxl(GQL, handle, m, n, k, dA, lda, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxl_ungxl(GQL, handle, m, 0, 0, (T) nullptr, lda, (T) nullptr),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_orgxl_ungxl(GQL, handle, 0, 0, 0, (T) nullptr, lda, (T) nullptr),
                          rocblas_status_success);
}

template <typename T, bool GQL>
void testing_orgxl_ungxl_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
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
    orgxl_ungxl_checkBadArgs<GQL>(handle, m, n, k, dA.data(), lda, dIpiv.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgxl_ungxl_initData(const rocblas_handle handle,
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
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                if(m - i == n - j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute QL factorization
        cpu_geqlf(m, n, hA[0], lda, hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <bool GQL, typename T, typename Td, typename Th>
void orgxl_ungxl_getError(const rocblas_handle handle,
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
    size_t size_W = size_t(n);
    std::vector<T> hW(size_W);

    // initialize data
    orgxl_ungxl_initData<true, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_orgxl_ungxl(GQL, handle, m, n, k, dA.data(), lda, dIpiv.data()));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    GQL ? cpu_orgql_ungql(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W)
        : cpu_org2l_ung2l(m, n, k, hA[0], lda, hIpiv[0], hW.data());

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <bool GQL, typename T, typename Td, typename Th>
void orgxl_ungxl_getPerfData(const rocblas_handle handle,
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
    size_t size_W = size_t(n);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgxl_ungxl_initData<true, false, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        GQL ? cpu_orgql_ungql(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W)
            : cpu_org2l_ung2l(m, n, k, hA[0], lda, hIpiv[0], hW.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orgxl_ungxl_initData<true, false, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgxl_ungxl_initData<false, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_orgxl_ungxl(GQL, handle, m, n, k, dA.data(), lda, dIpiv.data()));
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
        orgxl_ungxl_initData<false, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        start = get_time_us_sync(stream);
        rocsolver_orgxl_ungxl(GQL, handle, m, n, k, dA.data(), lda, dIpiv.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, bool GQL>
void testing_orgxl_ungxl(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int m = argus.get<rocblas_int>("m", n);
    rocblas_int k = argus.get<rocblas_int>("k", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(m);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || k < 0 || lda < m || m < n || k > n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orgxl_ungxl(GQL, handle, m, n, k, (T*)nullptr, lda, (T*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_orgxl_ungxl(GQL, handle, m, n, k, (T*)nullptr, lda, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
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
            rocsolver_orgxl_ungxl(GQL, handle, m, n, k, dA.data(), lda, dIpiv.data()),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        orgxl_ungxl_getError<GQL, T>(handle, m, n, k, dA, lda, dIpiv, hA, hAr, hIpiv, &max_error);

    // collect performance data
    if(argus.timing)
        orgxl_ungxl_getPerfData<GQL, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, &gpu_time_used,
                                        &cpu_time_used, hot_calls, argus.profile,
                                        argus.profile_kernels, argus.perf);

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
            rocsolver_bench_output("m", "n", "k", "lda");
            rocsolver_bench_output(m, n, k, lda);

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

#define EXTERN_TESTING_ORGXL_UNGXL(...) \
    extern template void testing_orgxl_ungxl<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORGXL_UNGXL, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
