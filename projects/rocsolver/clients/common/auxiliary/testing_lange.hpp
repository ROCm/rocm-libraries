/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename I, typename S>
void lange_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_norm_type norm_type,
                        const I m,
                        const I n,
                        T dA,
                        const I lda,
                        S dnorms)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lange(nullptr, norm_type, m, n, dA, lda, dnorms),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lange(handle, static_cast<rocsolver_norm_type>(0), m, n, dA, lda, dnorms),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, m, n, (T) nullptr, lda, dnorms),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, m, n, dA, lda, (S) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, (I)0, n, (T) nullptr, lda, (S) nullptr),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, m, (I)0, (T) nullptr, lda, (S) nullptr),
                          rocblas_status_success);
}

template <typename T, typename I>
void testing_lange_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocsolver_norm_type norm_type = rocsolver_norm_type_max;
    I m = 1;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dnorms(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dnorms.memcheck());

    // check bad arguments
    lange_checkBadArgs(handle, norm_type, m, n, dA.data(), lda, dnorms.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void lange_initData(const rocblas_handle handle,
                    const rocsolver_norm_type norm_type,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dnorms,
                    Th& hA,
                    Sh& hnorms)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void lange_getError(const rocblas_handle handle,
                    const rocsolver_norm_type norm_type,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dnorms,
                    Th& hA,
                    Sh& hnorms,
                    Sh& hnorms_res,
                    double* max_err)
{
    // Workspace for CPU lange (max needed is for 1-norm or infinity-norm)
    size_t size_work = std::max(m, n);
    std::vector<S> work(size_work);

    // initialize data
    lange_initData<true, true, T, I, S>(handle, norm_type, m, n, dA, lda, dnorms, hA, hnorms);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, norm_type, m, n, dA.data(), lda, dnorms.data()));
    CHECK_HIP_ERROR(hnorms_res.transfer_from(dnorms));

    // CPU lapack
    char norm = rocsolver2char_norm_type(norm_type);
    hnorms[0][0] = cpu_lange<T, S>(norm, m, n, hA[0], lda, work.data());

    // error is ||hnorms - hnorms_res|| / ||hnorms||
    // using absolute value since we only have a single scalar
    *max_err = std::abs(hnorms[0][0] - hnorms_res[0][0]) / std::abs(hnorms[0][0]);
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void lange_getPerfData(const rocblas_handle handle,
                       const rocsolver_norm_type norm_type,
                       const I m,
                       const I n,
                       Td& dA,
                       const I lda,
                       Sd& dnorms,
                       Th& hA,
                       Sh& hnorms,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    // Workspace for CPU lange
    size_t size_work = std::max(m, n);
    std::vector<S> work(size_work);

    // only init CPU data once as it is not overwritten
    lange_initData<true, false, T, I, S>(handle, norm_type, m, n, dA, lda, dnorms, hA, hnorms);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        char norm = rocsolver2char_norm_type(norm_type);
        *cpu_time_used = get_time_us_no_sync();
        hnorms[0][0] = cpu_lange<T, S>(norm, m, n, hA[0], lda, work.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lange_initData<false, true, T, I, S>(handle, norm_type, m, n, dA, lda, dnorms, hA, hnorms);

        CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, norm_type, m, n, dA.data(), lda, dnorms.data()));
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
        lange_initData<false, true, T, I, S>(handle, norm_type, m, n, dA, lda, dnorms, hA, hnorms);

        timer.start(stream);
        rocsolver_lange(handle, norm_type, m, n, dA.data(), lda, dnorms.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, typename I>
void testing_lange(Arguments& argus)
{
    using S = real_t<T>;

    // get arguments
    rocblas_local_handle handle;
    char norm_typeC = argus.get<char>("norm_type");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);

    rocsolver_norm_type norm_type = char2rocsolver_norm_type(norm_typeC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_frobenius
       && norm_type != rocsolver_norm_type_infinity && norm_type != rocsolver_norm_type_max)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, m, n, (T*)nullptr, lda, (S*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_norms = 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_norms_res = (argus.unit_check || argus.norm_check) ? size_norms : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, m, n, (T*)nullptr, lda, (S*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lange(handle, norm_type, m, n, (T*)nullptr, lda, (S*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<S> hnorms(size_norms, 1, size_norms, 1);
    host_strided_batch_vector<S> hnorms_res(size_norms_res, 1, size_norms_res, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dnorms(size_norms, 1, size_norms, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dnorms.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lange(handle, norm_type, m, n, dA.data(), lda, dnorms.data()),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lange_getError<T, I, S>(handle, norm_type, m, n, dA, lda, dnorms, hA, hnorms, hnorms_res,
                                &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        lange_getPerfData<T, I, S>(handle, norm_type, m, n, dA, lda, dnorms, hA, hnorms,
                                   &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                   argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    if(argus.unit_check)
    {
        if(norm_type == rocsolver_norm_type_one)
        {
            ROCSOLVER_TEST_CHECK(T, max_error, m); // column sums of m elements
        }
        else if(norm_type == rocsolver_norm_type_infinity)
        {
            ROCSOLVER_TEST_CHECK(T, max_error, n); // row sums of n elements
        }
        else if(norm_type == rocsolver_norm_type_max)
        {
            ROCSOLVER_TEST_CHECK(T, max_error, 1); // no summation
        }
        else
        {
            ROCSOLVER_TEST_CHECK(T, max_error, m * n); // Frobenius: sum of m*n terms
        }
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("norm_type", "m", "n", "lda");
            rocsolver_bench_output(norm_typeC, m, n, lda);

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

#define EXTERN_TESTING_LANGE(...) extern template void testing_lange<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LANGE, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
