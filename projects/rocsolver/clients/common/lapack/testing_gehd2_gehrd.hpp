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

template <bool STRIDED, bool GEHRD, typename T, typename U>
void gehd2_gehrd_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int ilo,
                              const rocblas_int ihi,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gehd2_gehrd(STRIDED, GEHRD, nullptr, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, dA, lda,
                                                    stA, dIpiv, stP, (rocblas_int)-1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, (T) nullptr,
                                                lda, stA, dIpiv, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, dA, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, (rocblas_int)0, ilo, ihi,
                                                (T) nullptr, lda, stA, (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, dA, lda,
                                                    stA, dIpiv, stP, (rocblas_int)0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GEHRD, typename T>
void testing_gehd2_gehrd_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;
    rocblas_int ilo = 1;
    rocblas_int ihi = 2;
    rocblas_int lda = 2;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_HESSENBERG
    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gehd2_gehrd_checkBadArgs<STRIDED, GEHRD>(handle, n, ilo, ihi, dA.data(), lda, stA,
                                                 dIpiv.data(), stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gehd2_gehrd_checkBadArgs<STRIDED, GEHRD>(handle, n, ilo, ihi, dA.data(), lda, stA,
                                                 dIpiv.data(), stP, bc);
    }
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gehd2_gehrd_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int ilo,
                          const rocblas_int ihi,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else if(i == j + 1)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GEHRD, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gehd2_gehrd_getError(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int ilo,
                          const rocblas_int ihi,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          Uh& hIpivRes,
                          double* max_err)
{
    std::vector<T> hW(n);

    // input data initialization
    gehd2_gehrd_initData<true, true, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, dA.data(), lda,
                                              stA, dIpiv.data(), stP, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        GEHRD ? cpu_gehrd(n, ilo, ihi, hA[b], lda, hIpiv[b], hW.data(), n)
              : cpu_gehd2(n, ilo, ihi, hA[b], lda, hIpiv[b], hW.data());
    }

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', n, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;

        err = norm_error('F', 1, ihi - ilo, 1, hIpiv[b] + ilo - 1, hIpivRes[b] + ilo - 1);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, bool GEHRD, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gehd2_gehrd_getPerfData(const rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int ilo,
                             const rocblas_int ihi,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    std::vector<T> hW(n);

    if(!perf)
    {
        gehd2_gehrd_initData<true, false, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GEHRD ? cpu_gehrd(n, ilo, ihi, hA[b], lda, hIpiv[b], hW.data(), n)
                  : cpu_gehd2(n, ilo, ihi, hA[b], lda, hIpiv[b], hW.data());
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gehd2_gehrd_initData<true, false, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA,
                                         hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gehd2_gehrd_initData<false, true, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        CHECK_ROCBLAS_ERROR(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, dA.data(),
                                                  lda, stA, dIpiv.data(), stP, bc));
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
        gehd2_gehrd_initData<false, true, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        timer.start(stream);
        rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi, dA.data(), lda, stA,
                              dIpiv.data(), stP, bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, bool GEHRD, typename T>
void testing_gehd2_gehrd(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ilo = argus.get<rocblas_int>("ilo", 1);
    rocblas_int ihi = argus.get<rocblas_int>("ihi", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", std::max(n - 1, 0));

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = n > 1 ? size_t(n - 1) : 0;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_HESSENBERG
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                        (T* const*)nullptr, lda, stA, (T*)nullptr,
                                                        stP, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                        (T*)nullptr, lda, stA, (T*)nullptr, stP, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0 || n && (ilo < 1 || ihi < ilo || ihi > n));
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                        (T* const*)nullptr, lda, stA, (T*)nullptr,
                                                        stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                        (T*)nullptr, lda, stA, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                    (T* const*)nullptr, lda, stA, (T*)nullptr, stP,
                                                    bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                    (T*)nullptr, lda, stA, (T*)nullptr, stP, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED && STRIDED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<T> hIpivRes(size_PRes, 1, stP, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                        dA.data(), lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gehd2_gehrd_getError<STRIDED, GEHRD, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP,
                                                    bc, hA, hARes, hIpiv, hIpivRes, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gehd2_gehrd_getPerfData<STRIDED, GEHRD, T>(
                handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<T> hIpivRes(size_PRes, 1, stP, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gehd2_gehrd(STRIDED, GEHRD, handle, n, ilo, ihi,
                                                        dA.data(), lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gehd2_gehrd_getError<STRIDED, GEHRD, T>(handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP,
                                                    bc, hA, hARes, hIpiv, hIpivRes, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gehd2_gehrd_getPerfData<STRIDED, GEHRD, T>(
                handle, n, ilo, ihi, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 12 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 12 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("n", "ilo", "ihi", "lda", "strideP", "batch_c");
                rocsolver_bench_output(n, ilo, ihi, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("n", "ilo", "ihi", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(n, ilo, ihi, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("n", "ilo", "ihi", "lda");
                rocsolver_bench_output(n, ilo, ihi, lda);
            }
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

#define EXTERN_TESTING_GEHD2_GEHRD(...) \
    extern template void testing_gehd2_gehrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEHD2_GEHRD,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
