/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename U>
void lahr2_checkBadArgs(const rocblas_handle handle,
                        const rocblas_int n,
                        const rocblas_int k,
                        const rocblas_int nb,
                        U dA,
                        const rocblas_int lda,
                        T* dTau,
                        T* dT,
                        const rocblas_int ldt,
                        T* dY,
                        const rocblas_int ldy)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(nullptr, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, (U) nullptr, lda, dTau, dT, ldt, dY, ldy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, dA, lda, (T*)nullptr, dT, ldt, dY, ldy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, dA, lda, dTau, (T*)nullptr, ldt, dY, ldy),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, dA, lda, dTau, dT, ldt, (T*)nullptr, ldy),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, 0, k, nb, (U) nullptr, lda, dTau, dT, ldt, dY, ldy),
                          rocblas_status_success);
}

template <typename T>
void testing_lahr2_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 5;
    rocblas_int k = 1;
    rocblas_int nb = 2;
    rocblas_int lda = 5;
    rocblas_int ldt = 2;
    rocblas_int ldy = 5;

#ifdef ROCSOLVER_ENABLE_HESSENBERG
    // memory allocations
    device_strided_batch_vector<T> dA(lda * n, 1, lda * n, 1);
    device_strided_batch_vector<T> dTau(nb, 1, nb, 1);
    device_strided_batch_vector<T> dT(ldt * nb, 1, ldt * nb, 1);
    device_strided_batch_vector<T> dY(ldy * nb, 1, ldy * nb, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dT.memcheck());
    CHECK_HIP_ERROR(dY.memcheck());

    // check bad arguments
    lahr2_checkBadArgs(handle, n, k, nb, dA.data(), lda, dTau.data(), dT.data(), ldt, dY.data(), ldy);
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void lahr2_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int k,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Td& dTau,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dY,
                    const rocblas_int ldy,
                    Th& hA,
                    Th& hTau,
                    Th& hT,
                    Th& hY)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dT.transfer_from(hT));
        CHECK_HIP_ERROR(dY.transfer_from(hY));
    }
}

template <typename T, typename Td, typename Th>
void lahr2_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int k,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Td& dTau,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dY,
                    const rocblas_int ldy,
                    Th& hA,
                    Th& hARes,
                    Th& hTau,
                    Th& hTauRes,
                    Th& hT,
                    Th& hTRes,
                    Th& hY,
                    Th& hYRes,
                    double* max_err)
{
    // input data initialization
    lahr2_initData<true, true, T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hTau, hT,
                                  hY);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_lahr2(handle, n, k, nb, dA.data(), lda, dTau.data(), dT.data(),
                                        ldt, dY.data(), ldy));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hTauRes.transfer_from(dTau));
    CHECK_HIP_ERROR(hTRes.transfer_from(dT));
    CHECK_HIP_ERROR(hYRes.transfer_from(dY));

    // CPU lapack
    cpu_lahr2(n, k, nb, hA[0], lda, hTau[0], hT[0], ldt, hY[0], ldy);

    // error is max(||hA - hARes|| / ||hA||, ||hY - hYRes|| / ||hY||)
    // using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F', n, n, lda, hA[0], hARes[0]);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', n, nb, ldy, hY[0], hYRes[0]);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', nb, nb, ldt, hT[0], hTRes[0]);
    *max_err = err > *max_err ? err : *max_err;
}

template <typename T, typename Td, typename Th>
void lahr2_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       const rocblas_int k,
                       const rocblas_int nb,
                       Td& dA,
                       const rocblas_int lda,
                       Td& dTau,
                       Td& dT,
                       const rocblas_int ldt,
                       Td& dY,
                       const rocblas_int ldy,
                       Th& hA,
                       Th& hTau,
                       Th& hT,
                       Th& hY,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        lahr2_initData<true, false, T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hTau,
                                       hT, hY);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        cpu_lahr2(n, k, nb, hA[0], lda, hTau[0], hT[0], ldt, hY[0], ldy);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    lahr2_initData<true, false, T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hTau, hT,
                                   hY);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lahr2_initData<false, true, T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hTau,
                                       hT, hY);

        CHECK_ROCBLAS_ERROR(rocsolver_lahr2(handle, n, k, nb, dA.data(), lda, dTau.data(),
                                            dT.data(), ldt, dY.data(), ldy));
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

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        lahr2_initData<false, true, T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hTau,
                                       hT, hY);

        timer.start(stream);
        rocsolver_lahr2(handle, n, k, nb, dA.data(), lda, dTau.data(), dT.data(), ldt, dY.data(),
                        ldy);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T>
void testing_lahr2(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int k = argus.get<rocblas_int>("k", 1);
    rocblas_int nb = argus.get<rocblas_int>("nb", std::max(0, n - k));
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldt = argus.get<rocblas_int>("ldt", nb);
    rocblas_int ldy = argus.get<rocblas_int>("ldy", n);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_tau = nb;
    size_t size_T = ldt * nb;
    size_t size_Y = ldy * nb;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_tauRes = (argus.unit_check || argus.norm_check) ? size_tau : 0;
    size_t size_TRes = (argus.unit_check || argus.norm_check) ? size_T : 0;
    size_t size_YRes = (argus.unit_check || argus.norm_check) ? size_Y : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_HESSENBERG
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, (T*)nullptr, lda, (T*)nullptr,
                                              (T*)nullptr, ldt, (T*)nullptr, ldy),
                              rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes (n<=1 is quick return, not invalid)
    bool invalid_size = (n < 0 || k < 1 || nb < 0 || (n > 1 && (nb < 1 || k >= n || nb > n - k))
                         || lda < std::max(1, n) || ldt < nb || ldy < std::max(1, n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, (T*)nullptr, lda, (T*)nullptr,
                                              (T*)nullptr, ldt, (T*)nullptr, ldy),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lahr2(handle, n, k, nb, (T*)nullptr, lda, (T*)nullptr,
                                          (T*)nullptr, ldt, (T*)nullptr, ldy));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);
    host_strided_batch_vector<T> hTauRes(size_tauRes, 1, size_tauRes, 1);
    host_strided_batch_vector<T> hT(size_T, 1, size_T, 1);
    host_strided_batch_vector<T> hTRes(size_TRes, 1, size_TRes, 1);
    host_strided_batch_vector<T> hY(size_Y, 1, size_Y, 1);
    host_strided_batch_vector<T> hYRes(size_YRes, 1, size_YRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    device_strided_batch_vector<T> dT(size_T, 1, size_T, 1);
    device_strided_batch_vector<T> dY(size_Y, 1, size_Y, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_T)
        CHECK_HIP_ERROR(dT.memcheck());
    if(size_Y)
        CHECK_HIP_ERROR(dY.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lahr2(handle, n, k, nb, dA.data(), lda, dTau.data(),
                                              dT.data(), ldt, dY.data(), ldy),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lahr2_getError<T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hARes, hTau,
                          hTauRes, hT, hTRes, hY, hYRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        lahr2_getPerfData<T>(handle, n, k, nb, dA, lda, dTau, dT, ldt, dY, ldy, hA, hTau, hT, hY,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using 8 * nb * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 8 * nb * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "k", "nb", "lda", "ldt", "ldy");
            rocsolver_bench_output(n, k, nb, lda, ldt, ldy);
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

#define EXTERN_TESTING_LAHR2(...) extern template void testing_lahr2<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LAHR2, FOREACH_SCALAR_TYPE, APPLY_STAMP)
