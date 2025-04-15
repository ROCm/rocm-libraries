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

template <typename T, typename S>
void latrd_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        const rocblas_int k,
                        T dA,
                        const rocblas_int lda,
                        S dE,
                        T dTau,
                        T dW,
                        const rocblas_int ldw)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(nullptr, uplo, n, k, dA, lda, dE, dTau, dW, ldw),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, rocblas_fill_full, n, k, dA, lda, dE, dTau, dW, ldw),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, (T) nullptr, lda, dE, dTau, dW, ldw),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA, lda, (S) nullptr, dTau, dW, ldw),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA, lda, dE, (T) nullptr, dW, ldw),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA, lda, dE, dTau, (T) nullptr, ldw),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, 0, dA, lda, dE, dTau, (T) nullptr, ldw),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, 0, 0, (T) nullptr, lda, (S) nullptr,
                                          (T) nullptr, (T) nullptr, ldw),
                          rocblas_status_success);
}

template <typename T>
void testing_latrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int k = 1;
    rocblas_int lda = 1;
    rocblas_int ldw = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    device_strided_batch_vector<T> dW(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());

    // check bad arguments
    latrd_checkBadArgs(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(), dW.data(), ldw);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void latrd_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j || (i == j + 1) || (i == j - 1))
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void latrd_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] = hA[0][i + j * lda].real() + 400;
                else if((i == j + 1) || (i == j - 1))
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Sd, typename Td, typename Sh, typename Th>
void latrd_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dA,
                    const rocblas_int lda,
                    Sd& dE,
                    Td& dTau,
                    Td& dW,
                    const rocblas_int ldw,
                    Th& hA,
                    Th& hARes,
                    Sh& hE,
                    Th& hTau,
                    Th& hW,
                    Th& hWRes,
                    double* max_err)
{
    // input data initialization
    latrd_initData<true, true, T>(handle, n, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(),
                                        dW.data(), ldw));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));

    // CPU lapack
    cpu_latrd(uplo, n, k, hA[0], lda, hE[0], hTau[0], hW[0], ldw);

    // error is max(||hA - hARes|| / ||hA||, ||hW - hWRes|| / ||hW||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    rocblas_int offset = (uplo == rocblas_fill_lower) ? k : 0;
    *max_err = 0;
    err = norm_error('F', n, n, lda, hA[0], hARes[0]);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', n - k, k, ldw, hW[0] + offset, hWRes[0] + offset);
    *max_err = err > *max_err ? err : *max_err;
}

template <typename T, typename Sd, typename Td, typename Sh, typename Th>
void latrd_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int k,
                       Td& dA,
                       const rocblas_int lda,
                       Sd& dE,
                       Td& dTau,
                       Td& dW,
                       const rocblas_int ldw,
                       Th& hA,
                       Sh& hE,
                       Th& hTau,
                       Th& hW,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        latrd_initData<true, false, T>(handle, n, dA, lda, hA);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        memset(hW[0], 0, ldw * k * sizeof(T));
        cpu_latrd(uplo, n, k, hA[0], lda, hE[0], hTau[0], hW[0], ldw);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    latrd_initData<true, false, T>(handle, n, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        latrd_initData<false, true, T>(handle, n, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(),
                                            dTau.data(), dW.data(), ldw));
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
        latrd_initData<false, true, T>(handle, n, dA, lda, hA);

        start = get_time_us_sync(stream);
        rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(), dTau.data(), dW.data(), ldw);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_latrd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int k = argus.get<rocblas_int>("k", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldw = argus.get<rocblas_int>("ldw", n);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, (T*)nullptr, lda, (S*)nullptr,
                                              (T*)nullptr, (T*)nullptr, ldw),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = lda * n;
    size_t size_E = n;
    size_t size_tau = n;
    size_t size_W = ldw * k;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || k < 0 || k > n || lda < n || ldw < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, (T*)nullptr, lda, (S*)nullptr,
                                              (T*)nullptr, (T*)nullptr, ldw),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_latrd(handle, uplo, n, k, (T*)nullptr, lda, (S*)nullptr,
                                          (T*)nullptr, (T*)nullptr, ldw));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);
    host_strided_batch_vector<T> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<T> hWRes(size_WRes, 1, size_WRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    device_strided_batch_vector<T> dW(size_W, 1, size_W, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());

    // check quick return
    if(k == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_latrd(handle, uplo, n, k, dA.data(), lda, dE.data(),
                                              dTau.data(), dW.data(), ldw),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        latrd_getError<T>(handle, uplo, n, k, dA, lda, dE, dTau, dW, ldw, hA, hARes, hE, hTau, hW,
                          hWRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        latrd_getPerfData<T>(handle, uplo, n, k, dA, lda, dE, dTau, dW, ldw, hA, hE, hTau, hW,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using k*n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, k * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "k", "lda", "ldw");
            rocsolver_bench_output(uploC, n, k, lda, ldw);
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

#define EXTERN_TESTING_LATRD(...) extern template void testing_latrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LATRD, FOREACH_SCALAR_TYPE, APPLY_STAMP)
