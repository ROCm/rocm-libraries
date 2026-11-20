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

template <typename T>
void orghr_unghr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int ilo,
                              const rocblas_int ihi,
                              T dA,
                              const rocblas_int lda,
                              T dTau)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_orghr_unghr(nullptr, n, ilo, ihi, dA, lda, dTau),
                          rocblas_status_invalid_handle);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orghr_unghr(handle, n, ilo, ihi, (T) nullptr, lda, dTau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_orghr_unghr(handle, n, ilo, ihi, dA, lda, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_orghr_unghr(handle, 0, 1, 0, (T) nullptr, lda, (T) nullptr),
                          rocblas_status_success);
}

template <typename T>
void testing_orghr_unghr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;
    rocblas_int ilo = 1;
    rocblas_int ihi = 2;
    rocblas_int lda = 2;

#ifdef ROCSOLVER_ENABLE_XXGHR
    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());

    // check bad arguments
    orghr_unghr_checkBadArgs(handle, n, ilo, ihi, dA.data(), lda, dTau.data());
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orghr_unghr_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int ilo,
                          const rocblas_int ihi,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dTau,
                          Th& hA,
                          Th& hTau,
                          std::vector<T>& hW,
                          size_t size_W)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale to avoid singularities
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(i == j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute gehrd to produce Householder vectors in A and tau in hTau
        cpu_gehrd(n, ilo, ihi, hA[0], lda, hTau[0], hW.data(), (rocblas_int)size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dTau.transfer_from(hTau));
    }
}

template <typename T, typename Td, typename Th>
void orghr_unghr_getError(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int ilo,
                          const rocblas_int ihi,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dTau,
                          Th& hA,
                          Th& hAr,
                          Th& hTau,
                          double* max_err)
{
    size_t size_W = n * 32;
    std::vector<T> hW(size_W);

    // initialize data
    orghr_unghr_initData<true, true, T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hTau, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_orghr_unghr(handle, n, ilo, ihi, dA.data(), lda, dTau.data()));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cpu_orghr_unghr(n, ilo, ihi, hA[0], lda, hTau[0], hW.data(), (rocblas_int)size_W);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', n, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void orghr_unghr_getPerfData(const rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int ilo,
                             const rocblas_int ihi,
                             Td& dA,
                             const rocblas_int lda,
                             Td& dTau,
                             Th& hA,
                             Th& hTau,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    size_t size_W = n * 32;
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orghr_unghr_initData<true, false, T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hTau, hW,
                                             size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_orghr_unghr(n, ilo, ihi, hA[0], lda, hTau[0], hW.data(), (rocblas_int)size_W);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orghr_unghr_initData<true, false, T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hTau, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orghr_unghr_initData<false, true, T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hTau, hW,
                                             size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_orghr_unghr(handle, n, ilo, ihi, dA.data(), lda, dTau.data()));
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
        orghr_unghr_initData<false, true, T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hTau, hW,
                                             size_W);

        timer.start(stream);
        rocsolver_orghr_unghr(handle, n, ilo, ihi, dA.data(), lda, dTau.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T>
void testing_orghr_unghr(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ilo = argus.get<rocblas_int>("ilo");
    rocblas_int ihi = argus.get<rocblas_int>("ihi");

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = n > 1 ? size_t(n - 1) : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_XXGHR
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orghr_unghr(handle, n, ilo, ihi, (T*)nullptr, lda, (T*)nullptr),
            rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || (n && (ilo < 1 || ihi < ilo || ihi > n)));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_orghr_unghr(handle, n, ilo, ihi, (T*)nullptr, lda, (T*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_orghr_unghr(handle, n, ilo, ihi, (T*)nullptr, lda, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T> hTau(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dTau(size_P, 1, size_P, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dTau.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_orghr_unghr(handle, n, ilo, ihi, dA.data(), lda, dTau.data()),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        orghr_unghr_getError<T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hAr, hTau, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        orghr_unghr_getPerfData<T>(handle, n, ilo, ihi, dA, lda, dTau, hA, hTau, &gpu_time_used,
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
            rocsolver_bench_output("n", "lda", "ilo", "ihi");
            rocsolver_bench_output(n, lda, ilo, ihi);

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

#define EXTERN_TESTING_ORGHR_UNGHR(...) \
    extern template void testing_orghr_unghr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORGHR_UNGHR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
