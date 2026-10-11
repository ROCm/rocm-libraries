/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/misc/rocsolver_timer.hpp"

template <typename T, typename I>
void larfg_checkBadArgs(const rocblas_handle handle, const I n, T da, T dx, const I inc, T dtau)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_larfg(nullptr, n, da, dx, inc, dtau),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larfg(handle, n, (T) nullptr, dx, inc, dtau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfg(handle, n, da, (T) nullptr, inc, dtau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfg(handle, n, da, dx, inc, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larfg(handle, (I)0, (T) nullptr, (T) nullptr, inc, (T) nullptr),
                          rocblas_status_success);
}

template <typename T, typename I>
void testing_larfg_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    I n = 2;
    I inc = 1;

    // memory allocation
    device_strided_batch_vector<T> da(1, 1, 1, 1);
    device_strided_batch_vector<T> dx(1, 1, 1, 1);
    device_strided_batch_vector<T> dtau(1, 1, 1, 1);
    CHECK_HIP_ERROR(da.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dtau.memcheck());

    // check bad arguments
    larfg_checkBadArgs(handle, n, da.data(), dx.data(), inc, dtau.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void larfg_initData(const rocblas_handle handle,
                    int singular,
                    const I n,
                    Td& da,
                    Td& dx,
                    const I inc,
                    Td& dtau,
                    Th& ha,
                    Th& hx,
                    Th& htau)
{
    if(CPU)
    {
        rocblas_init<T>(ha, true);
        rocblas_init<T>(hx, true);

        // "singular" case sets imag( alpha ) = 0 and x = 0.
        if(singular == 1)
        {
            ha[0][0] = std::real(ha[0][0]);
            for(int i = 0; i < n - 1; ++i)
                hx[0][i * inc] = 0;
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(da.transfer_from(ha));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
    }
}

template <typename T, typename I, typename Td, typename Th>
void larfg_getError(const rocblas_handle handle,
                    int singular,
                    const I n,
                    Td& da,
                    Td& dx,
                    const I inc,
                    Td& dtau,
                    Th& ha,
                    Th& ha_res,
                    Th& hx,
                    Th& hx_res,
                    Th& htau,
                    Th& htau_res,
                    double* max_err)
{
    using std::real, std::imag, std::abs;

    // initialize data
    larfg_initData<true, true, T>(handle, singular, n, da, dx, inc, dtau, ha, hx, htau);

    // Degenerate case counts as "singular", where imag( alpha ) = 0 and x = 0.
    if(n == 1 && imag(ha[0][0]) == 0)
        singular = 1;

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_larfg(handle, n, da.data(), dx.data(), inc, dtau.data()));
    CHECK_HIP_ERROR(ha_res.transfer_from(da));
    CHECK_HIP_ERROR(hx_res.transfer_from(dx));
    CHECK_HIP_ERROR(htau_res.transfer_from(dtau));

    // CPU lapack
    cpu_larfg(n, ha[0], hx[0], inc, htau[0]);

    // error is max( ||hx - hx_res||_inf,
    //               |alpha - alpha_res| / |alpha|,
    //               |tau - tau_res| ).
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)

    // For "singular" vector, tau = 2 instead of LAPACK's convention that tau = 0.
    // Hence alpha = -alpha_lapack.
    double err_alpha, err_tau;
    if(singular)
    {
        err_alpha = abs(ha[0][0] + ha_res[0][0]);
        err_tau = abs(htau_res[0][0] - 2);
    }
    else
    {
        err_alpha = abs(ha[0][0] - ha_res[0][0]);
        err_tau = abs(htau[0][0] - htau_res[0][0]);
    }
    if(abs(ha[0][0]) != 0)
        err_alpha /= abs(ha[0][0]);

    // using norm-1 which is infinity norm for this 1-by-(n-1) data setup
    *max_err = norm_error('O', 1, n - 1, inc, hx[0], hx_res[0]);
    *max_err = rocblas_max_nan(*max_err, err_alpha);
    *max_err = rocblas_max_nan(*max_err, err_tau);
}

template <typename T, typename I, typename Td, typename Th>
void larfg_getPerfData(const rocblas_handle handle,
                       int singular,
                       const I n,
                       Td& da,
                       Td& dx,
                       const I inc,
                       Td& dtau,
                       Th& ha,
                       Th& hx,
                       Th& htau,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        larfg_initData<true, false, T>(handle, singular, n, da, dx, inc, dtau, ha, hx, htau);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_larfg(n, ha[0], hx[0], inc, htau[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    larfg_initData<true, false, T>(handle, singular, n, da, dx, inc, dtau, ha, hx, htau);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larfg_initData<false, true, T>(handle, singular, n, da, dx, inc, dtau, ha, hx, htau);

        CHECK_ROCBLAS_ERROR(rocsolver_larfg(handle, n, da.data(), dx.data(), inc, dtau.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

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
        larfg_initData<false, true, T>(handle, singular, n, da, dx, inc, dtau, ha, hx, htau);

        timer.start(stream);
        rocsolver_larfg(handle, n, da.data(), dx.data(), inc, dtau.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, typename I>
void testing_larfg(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    I n = argus.get<I>("n");
    I inc = argus.get<I>("incx");
    int singular = argus.get<int>("singular", 0);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    // size_x could be zero in test cases that are not quick-return or invalid
    // cases setting it to one to avoid possible memory access errors in the rest
    // of the unit test
    size_t size_x = n > 1 ? size_t(n - 1) : 1;
    size_t stx = size_x * inc;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_xr = (argus.unit_check || argus.norm_check) ? size_x : 0;
    size_t stxr = (argus.unit_check || argus.norm_check) ? stx : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || inc < 1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfg(handle, n, (T*)nullptr, (T*)nullptr, inc, (T*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_larfg(handle, n, (T*)nullptr, (T*)nullptr, inc, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hx(size_x, inc, stx, 1);
    host_strided_batch_vector<T> hx_res(size_xr, inc, stxr, 1);
    host_strided_batch_vector<T> ha(1, 1, 1, 1);
    host_strided_batch_vector<T> ha_res(1, 1, 1, 1);
    host_strided_batch_vector<T> htau(1, 1, 1, 1);
    host_strided_batch_vector<T> htau_res(1, 1, 1, 1);
    device_strided_batch_vector<T> dx(size_x, inc, stx, 1);
    device_strided_batch_vector<T> da(1, 1, 1, 1);
    device_strided_batch_vector<T> dtau(1, 1, 1, 1);
    CHECK_HIP_ERROR(da.memcheck());
    if(size_x)
        CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dtau.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfg(handle, n, da.data(), dx.data(), inc, dtau.data()),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larfg_getError<T>(handle, singular, n, da, dx, inc, dtau, ha, ha_res, hx, hx_res, htau,
                          htau_res, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        larfg_getPerfData<T>(handle, singular, n, da, dx, inc, dtau, ha, hx, htau, &gpu_time_used,
                             &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                             argus.perf);

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
            rocsolver_bench_output("n", "inc");
            rocsolver_bench_output(n, inc);

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

#define EXTERN_TESTING_LARFG(...) extern template void testing_larfg<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LARFG, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
