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

template <typename T, typename I>
void larf_checkBadArgs(const rocblas_handle handle,
                       const rocblas_side side,
                       const I m,
                       const I n,
                       T dx,
                       const I inc,
                       T dt,
                       T dA,
                       const I lda)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(nullptr, side, m, n, dx, inc, dt, dA, lda),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, rocblas_side_both, m, n, dx, inc, dt, dA, lda),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, side, m, n, (T) nullptr, inc, dt, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, side, m, n, dx, inc, (T) nullptr, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, side, m, n, dx, inc, dt, (T) nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, rocblas_side_left, (I)0, n, (T) nullptr, inc,
                                         (T) nullptr, (T) nullptr, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_larf(handle, rocblas_side_right, m, (I)0, (T) nullptr, inc,
                                         (T) nullptr, (T) nullptr, lda),
                          rocblas_status_success);
}

template <typename T, typename I>
void testing_larf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    I m = 1;
    I n = 1;
    I inc = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dx(1, 1, 1, 1);
    device_strided_batch_vector<T> dt(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dt.memcheck());

    // check bad arguments
    larf_checkBadArgs(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda);
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void larf_initData(const rocblas_handle handle,
                   const rocblas_side side,
                   const I m,
                   const I n,
                   Td& dx,
                   const I inc,
                   Td& dt,
                   Td& dA,
                   const I lda,
                   Th& xx,
                   Th& hx,
                   Th& ht,
                   Th& hA)
{
    if(CPU)
    {
        I order = xx.n();

        rocblas_init<T>(hA, true);
        rocblas_init<T>(xx, true);

        // compute householder reflector
        cpu_larfg(order, xx[0], xx[0] + abs(inc), abs(inc), ht[0]);
        xx[0][0] = 1;
        for(I i = 0; i < order; i++)
        {
            if(inc < 0)
                hx[0][i * abs(inc)] = xx[0][(order - 1 - i) * abs(inc)];
            else
                hx[0][i * inc] = xx[0][i * inc];
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dt.transfer_from(ht));
    }
}

template <typename T, typename I, typename Td, typename Th>
void larf_getError(const rocblas_handle handle,
                   const rocblas_side side,
                   const I m,
                   const I n,
                   Td& dx,
                   const I inc,
                   Td& dt,
                   Td& dA,
                   const I lda,
                   Th& xx,
                   Th& hx,
                   Th& ht,
                   Th& hA,
                   Th& hAr,
                   double* max_err)
{
    size_t size_w = (side == rocblas_side_left) ? size_t(n) : size_t(m);
    std::vector<T> hw(size_w);

    // initialize data
    larf_initData<true, true, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cpu_larf(side, m, n, hx[0], inc, ht[0], hA[0], lda, hw.data());

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename I, typename Td, typename Th>
void larf_getPerfData(const rocblas_handle handle,
                      const rocblas_side side,
                      const I m,
                      const I n,
                      Td& dx,
                      const I inc,
                      Td& dt,
                      Td& dA,
                      const I lda,
                      Th& xx,
                      Th& hx,
                      Th& ht,
                      Th& hA,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const int profile,
                      const bool profile_kernels,
                      const bool perf)
{
    size_t size_w = (side == rocblas_side_left) ? size_t(n) : size_t(m);
    std::vector<T> hw(size_w);

    if(!perf)
    {
        larf_initData<true, false, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_larf(side, m, n, hx[0], inc, ht[0], hA[0], lda, hw.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    larf_initData<true, false, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larf_initData<false, true, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

        CHECK_ROCBLAS_ERROR(
            rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda));
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
        larf_initData<false, true, T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA);

        start = get_time_us_sync(stream);
        rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, typename I>
void testing_larf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I inc = argus.get<I>("incx");
    I lda = argus.get<I>("lda", m);

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_larf(handle, side, m, n, (T*)nullptr, inc, (T*)nullptr, (T*)nullptr, lda),
            rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    size_t size_A = size_t(lda) * n;
    size_t size_x = left ? size_t(m) : size_t(n);
    size_t stx = size_x * abs(inc);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || !inc || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_larf(handle, side, m, n, (T*)nullptr, inc, (T*)nullptr, (T*)nullptr, lda),
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
            rocsolver_larf(handle, side, m, n, (T*)nullptr, inc, (T*)nullptr, (T*)nullptr, lda));

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
    host_strided_batch_vector<T> hx(size_x, abs(inc), stx, 1);
    host_strided_batch_vector<T> xx(size_x, abs(inc), stx, 1);
    host_strided_batch_vector<T> ht(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dx(size_x, abs(inc), stx, 1);
    device_strided_batch_vector<T> dt(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_x)
        CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dt.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_larf(handle, side, m, n, dx.data(), inc, dt.data(), dA.data(), lda),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larf_getError<T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        larf_getPerfData<T>(handle, side, m, n, dx, inc, dt, dA, lda, xx, hx, ht, hA,
                            &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                            argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using size_x * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, size_x);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("side", "m", "n", "inc", "lda");
            rocsolver_bench_output(sideC, m, n, inc, lda);

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

#define EXTERN_TESTING_LARF(...) extern template void testing_larf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LARF, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
