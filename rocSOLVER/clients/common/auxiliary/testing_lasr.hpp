/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
void lasr_checkBadArgs(const rocblas_handle handle,
                       const rocblas_side side,
                       const rocblas_pivot pivot,
                       const rocblas_direct direct,
                       const rocblas_int m,
                       const rocblas_int n,
                       S dC,
                       S dS,
                       T dA,
                       const rocblas_int lda)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(nullptr, side, pivot, direct, m, n, dC, dS, dA, lda),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasr(handle, rocblas_side(0), pivot, direct, m, n, dC, dS, dA, lda),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasr(handle, side, rocblas_pivot(0), direct, m, n, dC, dS, dA, lda),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasr(handle, side, pivot, rocblas_direct(0), m, n, dC, dS, dA, lda),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, side, pivot, direct, m, n, (S) nullptr, dS, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, side, pivot, direct, m, n, dC, (S) nullptr, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, side, pivot, direct, m, n, dC, dS, (T) nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, rocblas_side_left, pivot, direct, 0, n,
                                         (S) nullptr, (S) nullptr, (T) nullptr, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasr(handle, rocblas_side_right, pivot, direct, 0, n, dC, dS, (T) nullptr, lda),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, rocblas_side_right, pivot, direct, m, 0,
                                         (S) nullptr, (S) nullptr, (T) nullptr, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasr(handle, rocblas_side_left, pivot, direct, m, 0, dC, dS, (T) nullptr, lda),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, rocblas_side_left, pivot, direct, 1, n,
                                         (S) nullptr, (S) nullptr, dA, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, rocblas_side_right, pivot, direct, m, 1,
                                         (S) nullptr, (S) nullptr, dA, lda),
                          rocblas_status_success);
}

template <typename T>
void testing_lasr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_pivot pivot = rocblas_pivot_variable;
    rocblas_direct direct = rocblas_forward_direction;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int lda = 2;

    // memory allocation
    device_strided_batch_vector<S> dC(1, 1, 1, 1);
    device_strided_batch_vector<S> dS(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dA.memcheck());

    // check bad arguments
    lasr_checkBadArgs(handle, side, pivot, direct, m, n, dC.data(), dS.data(), dA.data(), lda);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Sd, typename Th, typename Sh>
void lasr_initData(const rocblas_handle handle,
                   const rocblas_side side,
                   const rocblas_pivot pivot,
                   const rocblas_direct direct,
                   const rocblas_int m,
                   const rocblas_int n,
                   Sd& dC,
                   Sd& dS,
                   Td& dA,
                   const rocblas_int lda,
                   Sh& hC,
                   Sh& hS,
                   Th& hA)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        rocblas_init<T>(hA, true);

        // construct C and S such that C^2 + S^2 = 1
        rocblas_init<S>(hC, true);
        rocblas_int size = (side == rocblas_side_left) ? m - 1 : n - 1;

        for(rocblas_int j = 0; j < size; ++j)
        {
            S temp = hC[0][j];
            temp = (temp - 5) / 5.0;
            hC[0][j] = temp;
            hS[0][j] = sqrt(1 - temp * temp);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(dS.transfer_from(hS));
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Sd, typename Th, typename Sh>
void lasr_getError(const rocblas_handle handle,
                   const rocblas_side side,
                   const rocblas_pivot pivot,
                   const rocblas_direct direct,
                   const rocblas_int m,
                   const rocblas_int n,
                   Sd& dC,
                   Sd& dS,
                   Td& dA,
                   const rocblas_int lda,
                   Sh& hC,
                   Sh& hS,
                   Th& hA,
                   Th& hAr,
                   double* max_err)
{
    // initialize data
    lasr_initData<true, true, T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_lasr(handle, side, pivot, direct, m, n, dC.data(), dS.data(), dA.data(), lda));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cpu_lasr(side, pivot, direct, m, n, hC[0], hS[0], hA[0], lda);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Sd, typename Th, typename Sh>
void lasr_getPerfData(const rocblas_handle handle,
                      const rocblas_side side,
                      const rocblas_pivot pivot,
                      const rocblas_direct direct,
                      const rocblas_int m,
                      const rocblas_int n,
                      Sd& dC,
                      Sd& dS,
                      Td& dA,
                      const rocblas_int lda,
                      Sh& hC,
                      Sh& hS,
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
        lasr_initData<true, false, T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_lasr(side, pivot, direct, m, n, hC[0], hS[0], hA[0], lda);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    lasr_initData<true, false, T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lasr_initData<false, true, T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_lasr(handle, side, pivot, direct, m, n, dC.data(), dS.data(),
                                           dA.data(), lda));
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
        lasr_initData<false, true, T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA);

        start = get_time_us_sync(stream);
        rocsolver_lasr(handle, side, pivot, direct, m, n, dC.data(), dS.data(), dA.data(), lda);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_lasr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char pivotC = argus.get<char>("pivot");
    char directC = argus.get<char>("direct");
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_pivot pivot = char2rocblas_pivot(pivotC);
    rocblas_direct direct = char2rocblas_direct(directC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, side, pivot, direct, m, n, (S*)nullptr,
                                             (S*)nullptr, (T*)nullptr, lda),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lasr(handle, side, pivot, direct, m, n, (S*)nullptr,
                                             (S*)nullptr, (T*)nullptr, lda),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    bool right = (side == rocblas_side_right);
    size_t size_CS = 0;
    if(left && m > 1)
        size_CS = size_t(m - 1);
    if(right && n > 1)
        size_CS = size_t(n - 1);
    size_t size_A = size_t(lda) * n;
    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lasr(handle, side, pivot, direct, m, n, (S*)nullptr,
                                         (S*)nullptr, (T*)nullptr, lda));

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
    host_strided_batch_vector<S> hC(size_CS, 1, size_CS, 1);
    host_strided_batch_vector<S> hS(size_CS, 1, size_CS, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dC(size_CS, 1, size_CS, 1);
    device_strided_batch_vector<S> dS(size_CS, 1, size_CS, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_CS)
    {
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
    }

    // check quick return
    bool quickreturn = (left && m < 2) || (right && n < 2) || n == 0 || m == 0;
    if(quickreturn)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_lasr(handle, side, pivot, direct, m, n, dC.data(), dS.data(), dA.data(), lda),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check computations
    if(argus.unit_check || argus.norm_check)
        lasr_getError<T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA, hAr,
                         &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        lasr_getPerfData<T>(handle, side, pivot, direct, m, n, dC, dS, dA, lda, hC, hS, hA,
                            &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                            argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    rocblas_int s = left ? m : n;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("side", "pivot", "direct", "m", "n", "lda");
            rocsolver_bench_output(sideC, pivotC, directC, m, n, lda);

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

#define EXTERN_TESTING_LASR(...) extern template void testing_lasr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LASR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
