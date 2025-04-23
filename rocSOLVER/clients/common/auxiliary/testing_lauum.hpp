/* **************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
void lauum_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        T A,
                        const rocblas_int lda)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lauum(nullptr, uplo, n, A, lda), rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_lauum(handle, rocblas_fill_full, n, A, lda),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lauum(handle, uplo, n, (T) nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lauum(handle, uplo, 0, (T) nullptr, lda), rocblas_status_success);
}

template <typename T>
void testing_lauum_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());

    // check bad arguments
    lauum_checkBadArgs(handle, uplo, n, dA.data(), lda);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void lauum_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        // LAPACK intends that lauum only be called on matrices with a real diagonal
        for(int i = 0; i < n; i++)
        {
            hA[0][i + i * lda] = std::real(hA[0][i + i * lda]);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Th>
void lauum_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA,
                    Th& hAr,
                    double* max_err)
{
    // initialize data
    lauum_initData<true, true, T>(handle, uplo, n, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_lauum(handle, uplo, n, dA.data(), lda));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cpu_lauum<T>(uplo, n, hA[0], lda);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius
    *max_err = norm_error('F', n, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void lauum_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       Td& dA,
                       const rocblas_int lda,
                       Th& hA,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        lauum_initData<true, false, T>(handle, uplo, n, dA, lda, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_lauum<T>(uplo, n, hA[0], lda);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    lauum_initData<true, false, T>(handle, uplo, n, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lauum_initData<false, true, T>(handle, uplo, n, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_lauum(handle, uplo, n, dA.data(), lda));
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
        lauum_initData<false, true, T>(handle, uplo, n, dA, lda, hA);

        start = get_time_us_sync(stream);
        rocsolver_lauum(handle, uplo, n, dA.data(), lda);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_lauum(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);

    rocblas_int hot_calls = argus.iters;
    rocblas_fill uplo = char2rocblas_fill(uploC);

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lauum(handle, uplo, n, (T*)nullptr, lda),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(n) * lda;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lauum(handle, uplo, n, (T*)nullptr, lda),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lauum(handle, uplo, n, (T*)nullptr, lda));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);

    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lauum(handle, uplo, n, dA.data(), lda),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lauum_getError<T>(handle, uplo, n, dA, lda, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        lauum_getPerfData<T>(handle, uplo, n, dA, lda, hA, &gpu_time_used, &cpu_time_used,
                             hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using machine precision for tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 1);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "lda");
            rocsolver_bench_output(uploC, n, lda);

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

#define EXTERN_TESTING_LAUUM(...) extern template void testing_lauum<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LAUUM, FOREACH_SCALAR_TYPE, APPLY_STAMP)
