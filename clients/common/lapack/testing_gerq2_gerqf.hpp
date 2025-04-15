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

template <bool STRIDED, bool GERQF, typename T, typename U>
void gerq2_gerqf_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gerq2_gerqf(STRIDED, GERQF, nullptr, m, n, dA, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA, lda, stA, dIpiv, stP, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, (T) nullptr, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA, lda, stA, (U) nullptr, stP, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, 0, n, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, 0, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA, lda, stA, dIpiv, stP, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GERQF, typename T>
void testing_gerq2_gerqf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gerq2_gerqf_checkBadArgs<STRIDED, GERQF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gerq2_gerqf_checkBadArgs<STRIDED, GERQF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gerq2_gerqf_initData(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
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
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
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

template <bool STRIDED, bool GERQF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gerq2_gerqf_getError(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          double* max_err)
{
    std::vector<T> hW(m);

    // input data initialization
    gerq2_gerqf_initData<true, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        GERQF ? cpu_gerqf(m, n, hA[b], lda, hIpiv[b], hW.data(), m)
              : cpu_gerq2(m, n, hA[b], lda, hIpiv[b], hW.data());
    }

    // error is ||hA - hARes|| / ||hA|| (ideally ||QR - Qres Rres|| / ||QR||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, bool GERQF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gerq2_gerqf_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
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
    std::vector<T> hW(m);

    if(!perf)
    {
        gerq2_gerqf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GERQF ? cpu_gerqf(m, n, hA[b], lda, hIpiv[b], hW.data(), m)
                  : cpu_gerq2(m, n, hA[b], lda, hIpiv[b], hW.data());
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gerq2_gerqf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gerq2_gerqf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        CHECK_ROCBLAS_ERROR(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA.data(), lda, stA,
                                                  dIpiv.data(), stP, bc));
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
        gerq2_gerqf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        start = get_time_us_sync(stream);
        rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP,
                              bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool GERQF, typename T>
void testing_gerq2_gerqf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", min(m, n));

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n,
                                                        (T* const*)nullptr, lda, stA, (T*)nullptr,
                                                        stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, (T*)nullptr,
                                                        lda, stA, (T*)nullptr, stP, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, (T* const*)nullptr,
                                                    lda, stA, (T*)nullptr, stP, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, (T*)nullptr, lda,
                                                    stA, (T*)nullptr, stP, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gerq2_gerqf_getError<STRIDED, GERQF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                                    hARes, hIpiv, &max_error);

        // collect performance data
        if(argus.timing)
            gerq2_gerqf_getPerfData<STRIDED, GERQF, T>(
                handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gerq2_gerqf(STRIDED, GERQF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gerq2_gerqf_getError<STRIDED, GERQF, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                                    hARes, hIpiv, &max_error);

        // collect performance data
        if(argus.timing)
            gerq2_gerqf_getPerfData<STRIDED, GERQF, T>(
                handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using m * machine_precision as tolerance
    // (for possibly singular of ill-conditioned matrices we could use m*min(m,n))
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, m);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
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

#define EXTERN_TESTING_GERQ2_GERQF(...) \
    extern template void testing_gerq2_gerqf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GERQ2_GERQF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
