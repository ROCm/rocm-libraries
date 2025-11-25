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

template <typename T, typename U>
void sterf_checkBadArgs(const rocblas_handle handle, const rocblas_int n, T dD, T dE, U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sterf(nullptr, n, dD, dE, dInfo), rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sterf(handle, n, (T) nullptr, dE, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sterf(handle, n, dD, (T) nullptr, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sterf(handle, n, dD, dE, (U) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sterf(handle, 0, (T) nullptr, (T) nullptr, dInfo),
                          rocblas_status_success);
}

template <typename T>
void testing_sterf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;

    // memory allocations
    device_strided_batch_vector<T> dD(1, 1, 1, 1);
    device_strided_batch_vector<T> dE(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    sterf_checkBadArgs(handle, n, dD.data(), dE.data(), dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sterf_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dD,
                    Td& dE,
                    Ud& dInfo,
                    Th& hD,
                    Th& hE,
                    Uh& hInfo)
{
    if(CPU)
    {
        rocblas_init<T>(hD, true);
        rocblas_init<T>(hE, true);

        // scale matrix and add random splits
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 400;
            hE[0][i] -= 5;
        }

        // add fixed splits in the matrix to test split handling
        rocblas_int k = n / 2;
        hE[0][k] = 0;
        hE[0][k - 1] = 0;
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void sterf_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dD,
                    Td& dE,
                    Ud& dInfo,
                    Th& hD,
                    Th& hDRes,
                    Th& hE,
                    Th& hERes,
                    Uh& hInfo,
                    double* max_err)
{
    // input data initialization
    sterf_initData<true, true, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sterf(handle, n, dD.data(), dE.data(), dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));

    // CPU lapack
    cpu_sterf(n, hD[0], hE[0]);

    // error is ||hD - hDRes|| / ||hD||
    // using frobenius norm
    *max_err = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void sterf_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       Td& dD,
                       Td& dE,
                       Ud& dInfo,
                       Th& hD,
                       Th& hE,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        sterf_initData<true, false, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_sterf(n, hD[0], hE[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sterf_initData<true, false, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sterf_initData<false, true, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

        CHECK_ROCBLAS_ERROR(rocsolver_sterf(handle, n, dD.data(), dE.data(), dInfo.data()));
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
        sterf_initData<false, true, T>(handle, n, dD, dE, dInfo, hD, hE, hInfo);

        start = get_time_us_sync(stream);
        rocsolver_sterf(handle, n, dD.data(), dE.data(), dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_sterf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");

    rocblas_int hot_calls = argus.iters;

    if(argus.alg_mode == 1)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_set_alg_mode(handle, rocsolver_function_sterf, rocsolver_alg_mode_hybrid),
            rocblas_status_success);

        rocsolver_alg_mode alg_mode;
        EXPECT_ROCBLAS_STATUS(rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &alg_mode),
                              rocblas_status_success);

        EXPECT_EQ(alg_mode, rocsolver_alg_mode_hybrid);
    }

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_ERes = (argus.unit_check || argus.norm_check) ? size_E : 0;

    // check invalid sizes
    bool invalid_size = (n < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_sterf(handle, n, (T*)nullptr, (T*)nullptr, (rocblas_int*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_sterf(handle, n, (T*)nullptr, (T*)nullptr, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<T> hDRes(size_DRes, 1, size_DRes, 1);
    host_strided_batch_vector<T> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hERes(size_ERes, 1, size_ERes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    device_strided_batch_vector<T> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<T> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sterf(handle, n, dD.data(), dE.data(), dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        sterf_getError<T>(handle, n, dD, dE, dInfo, hD, hDRes, hE, hERes, hInfo, &max_error);

    // collect performance data
    if(argus.timing)
        sterf_getPerfData<T>(handle, n, dD, dE, dInfo, hD, hE, hInfo, &gpu_time_used, &cpu_time_used,
                             hot_calls, argus.profile, argus.profile_kernels, argus.perf);

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
            rocsolver_bench_output("n");
            rocsolver_bench_output(n);

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

#define EXTERN_TESTING_STERF(...) extern template void testing_sterf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_STERF, FOREACH_REAL_TYPE, APPLY_STAMP)
