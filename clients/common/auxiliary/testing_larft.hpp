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

template <typename T>
void larft_checkBadArgs(const rocblas_handle handle,
                        const rocblas_direct direct,
                        const rocblas_storev storev,
                        const rocblas_int n,
                        const rocblas_int k,
                        T dV,
                        const rocblas_int ldv,
                        T dt,
                        T dT,
                        const rocblas_int ldt)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_larft(nullptr, direct, storev, n, k, dV, ldv, dt, dT, ldt),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_larft(handle, rocblas_direct(0), storev, n, k, dV, ldv, dt, dT, ldt),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_larft(handle, direct, rocblas_storev(0), n, k, dV, ldv, dt, dT, ldt),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larft(handle, direct, storev, n, k, (T) nullptr, ldv, dt, dT, ldt),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larft(handle, direct, storev, n, k, dV, ldv, (T) nullptr, dT, ldt),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larft(handle, direct, storev, n, k, dV, ldv, dt, (T) nullptr, ldt),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larft(handle, direct, storev, 0, k, (T) nullptr, ldv, dt, dT, ldt),
                          rocblas_status_success);
}

template <typename T>
void testing_larft_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_direct direct = rocblas_forward_direction;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_int k = 1;
    rocblas_int n = 1;
    rocblas_int ldv = 1;
    rocblas_int ldt = 1;

    // memory allocation
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dt(1, 1, 1, 1);
    device_strided_batch_vector<T> dT(1, 1, 1, 1);
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dT.memcheck());
    CHECK_HIP_ERROR(dt.memcheck());

    // check bad arguments
    larft_checkBadArgs(handle, direct, storev, n, k, dV.data(), ldv, dt.data(), dT.data(), ldt);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void larft_initData(const rocblas_handle handle,
                    const rocblas_direct direct,
                    const rocblas_storev storev,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dt,
                    Td& dT,
                    const rocblas_int ldt,
                    Th& hV,
                    Th& ht,
                    Th& hT,
                    std::vector<T>& hw,
                    size_t size_w)
{
    if(CPU)
    {
        rocblas_init<T>(hV, true);

        // scale to avoid singularities
        // and create householder reflectors
        if(storev == rocblas_column_wise)
        {
            for(int j = 0; j < k; ++j)
            {
                for(int i = 0; i < n; ++i)
                {
                    if(i == j)
                        hV[0][i + j * ldv] += 400;
                    else
                        hV[0][i + j * ldv] -= 4;
                }
            }

            if(direct == rocblas_forward_direction)
                cpu_geqrf(n, k, hV[0], ldv, ht[0], hw.data(), k);
            else
                cpu_geqlf(n, k, hV[0], ldv, ht[0], hw.data(), k);
        }
        else
        {
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i < k; ++i)
                {
                    if(i == j)
                        hV[0][i + j * ldv] += 400;
                    else
                        hV[0][i + j * ldv] -= 4;
                }
            }

            if(direct == rocblas_forward_direction)
                cpu_gelqf(k, n, hV[0], ldv, ht[0], hw.data(), k);
            else
                cpu_gerqf(k, n, hV[0], ldv, ht[0], hw.data(), k);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dt.transfer_from(ht));
    }
}

template <typename T, typename Td, typename Th>
void larft_getError(const rocblas_handle handle,
                    const rocblas_direct direct,
                    const rocblas_storev storev,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dt,
                    Td& dT,
                    const rocblas_int ldt,
                    Th& hV,
                    Th& ht,
                    Th& hT,
                    Th& hTr,
                    double* max_err)
{
    size_t size_w = size_t(k);
    std::vector<T> hw(size_w);

    // initialize data
    larft_initData<true, true, T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht, hT,
                                  hw, size_w);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_larft(handle, direct, storev, n, k, dV.data(), ldv, dt.data(), dT.data(), ldt));
    CHECK_HIP_ERROR(hTr.transfer_from(dT));

    // CPU lapack
    cpu_larft(direct, storev, n, k, hV[0], ldv, ht[0], hT[0], ldt);

    // error is ||hT - hTr|| / ||hT||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = (direct == rocblas_forward_direction)
        ? norm_error_upperTr('F', k, k, ldt, hT[0], hTr[0])
        : norm_error_lowerTr('F', k, k, ldt, hT[0], hTr[0]);
}

template <typename T, typename Td, typename Th>
void larft_getPerfData(const rocblas_handle handle,
                       const rocblas_direct direct,
                       const rocblas_storev storev,
                       const rocblas_int n,
                       const rocblas_int k,
                       Td& dV,
                       const rocblas_int ldv,
                       Td& dt,
                       Td& dT,
                       const rocblas_int ldt,
                       Th& hV,
                       Th& ht,
                       Th& hT,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    size_t size_w = size_t(k);
    std::vector<T> hw(size_w);

    if(!perf)
    {
        larft_initData<true, false, T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht,
                                       hT, hw, size_w);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_larft(direct, storev, n, k, hV[0], ldv, ht[0], hT[0], ldt);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    larft_initData<true, false, T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht, hT,
                                   hw, size_w);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larft_initData<false, true, T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht,
                                       hT, hw, size_w);

        CHECK_ROCBLAS_ERROR(rocsolver_larft(handle, direct, storev, n, k, dV.data(), ldv, dt.data(),
                                            dT.data(), ldt));
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
        larft_initData<false, true, T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht,
                                       hT, hw, size_w);

        start = get_time_us_sync(stream);
        rocsolver_larft(handle, direct, storev, n, k, dV.data(), ldv, dt.data(), dT.data(), ldt);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_larft(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char directC = argus.get<char>("direct");
    char storevC = argus.get<char>("storev");
    rocblas_int k = argus.get<rocblas_int>("k");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldv = argus.get<rocblas_int>("ldv", storevC == 'C' ? n : k);
    rocblas_int ldt = argus.get<rocblas_int>("ldt", k);

    rocblas_direct direct = char2rocblas_direct(directC);
    rocblas_storev storev = char2rocblas_storev(storevC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    bool row = (storev == rocblas_row_wise);
    size_t size_T = size_t(ldt) * k;
    size_t size_tau = size_t(k);
    size_t size_V = row ? size_t(ldv) * n : size_t(ldv) * k;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Tr = (argus.unit_check || argus.norm_check) ? size_T : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || k < 1 || ldt < k || (row && ldv < k) || (!row && ldv < n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larft(handle, direct, storev, n, k, (T*)nullptr, ldv,
                                              (T*)nullptr, (T*)nullptr, ldt),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_larft(handle, direct, storev, n, k, (T*)nullptr, ldv,
                                          (T*)nullptr, (T*)nullptr, ldt));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hT(size_T, 1, size_T, 1);
    host_strided_batch_vector<T> hTr(size_Tr, 1, size_Tr, 1);
    host_strided_batch_vector<T> ht(size_tau, 1, size_tau, 1);
    host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dT(size_T, 1, size_T, 1);
    device_strided_batch_vector<T> dt(size_tau, 1, size_tau, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_T)
        CHECK_HIP_ERROR(dT.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dt.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larft(handle, direct, storev, n, k, dV.data(), ldv,
                                              dt.data(), dT.data(), ldt),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larft_getError<T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht, hT, hTr,
                          &max_error);

    // collect performance data
    if(argus.timing)
        larft_getPerfData<T>(handle, direct, storev, n, k, dV, ldv, dt, dT, ldt, hV, ht, hT,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
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
            rocsolver_bench_output("direct", "storev", "n", "k", "ldv", "ldt");
            rocsolver_bench_output(directC, storevC, n, k, ldv, ldt);

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

#define EXTERN_TESTING_LARFT(...) extern template void testing_larft<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LARFT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
