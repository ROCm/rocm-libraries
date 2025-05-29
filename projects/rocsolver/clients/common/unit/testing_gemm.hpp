/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_utility.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void gemm_initData(const rocblas_handle handle, Td& dA, Th& hA, Td& dB, Th& hB, Td& dC, Th& hC)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);
        rocblas_init<T>(hC, true);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename I, typename Td, typename Ud, typename Th, typename Uh>
void gemm_getError(const rocblas_handle handle,
                   const rocblas_operation transA,
                   const rocblas_operation transB,
                   const I m,
                   const I n,
                   const I k,
                   Ud& dalpha,
                   Td& dA,
                   const I inca,
                   const I lda,
                   const rocblas_stride stA,
                   Td& dB,
                   const I incb,
                   const I ldb,
                   const rocblas_stride stB,
                   Ud& dbeta,
                   Td& dC,
                   const I incc,
                   const I ldc,
                   const rocblas_stride stC,
                   const I bc,
                   Uh& halpha,
                   Uh& hbeta,
                   Th& hA,
                   Th& hB,
                   Th& hC,
                   Th& hCRes,
                   double* max_err)
{
    // input data initialization
    gemm_initData<true, true, T, I>(handle, dA, hA, dB, hB, dC, hC);

    // execute computations
    // GPU
    CHECK_ROCBLAS_ERROR(
        (rocsolver::rocsolver_gemm<T, I>)(handle, transA, transB, m, n, k, dalpha.data(), dA.data(),
                                          0, inca, lda, stA, dB.data(), 0, incb, ldb, stB,
                                          dbeta.data(), dC.data(), 0, incc, ldc, stC, bc, nullptr));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // CPU lapack
    for(I b = 0; b < bc; ++b)
    {
        cpu_gemm(transA, transB, m, n, k, halpha[0][0], hA[b], lda, hB[b], ldb, hbeta[0][0], hC[b],
                 ldc);
    }

    // error is ||hC - hCRes|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(I b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, ldc, hC[b], hCRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename I, typename Td, typename Ud, typename Th, typename Uh>
void gemm_getPerfData(const rocblas_handle handle,
                      const rocblas_operation transA,
                      const rocblas_operation transB,
                      const I m,
                      const I n,
                      const I k,
                      Ud& dalpha,
                      Td& dA,
                      const I inca,
                      const I lda,
                      const rocblas_stride stA,
                      Td& dB,
                      const I incb,
                      const I ldb,
                      const rocblas_stride stB,
                      Ud& dbeta,
                      Td& dC,
                      const I incc,
                      const I ldc,
                      const rocblas_stride stC,
                      const I bc,
                      Uh& halpha,
                      Uh& hbeta,
                      Th& hA,
                      Th& hB,
                      Th& hC,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const int profile,
                      const bool profile_kernels,
                      const bool perf)
{
    if(!perf)
    {
        gemm_initData<true, false, T, I>(handle, dA, hA, dB, hB, dC, hC);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            cpu_gemm(transA, transB, m, n, k, halpha[0][0], hA[b], lda, hB[b], ldb, hbeta[0][0],
                     hC[b], ldc);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gemm_initData<true, false, T, I>(handle, dA, hA, dB, hB, dC, hC);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gemm_initData<false, true, T, I>(handle, dA, hA, dB, hB, dC, hC);

        CHECK_ROCBLAS_ERROR((rocsolver::rocsolver_gemm<T, I>)(handle, transA, transB, m, n, k,
                                                              dalpha.data(), dA.data(), 0, inca,
                                                              lda, stA, dB.data(), 0, incb, ldb,
                                                              stB, dbeta.data(), dC.data(), 0, incc,
                                                              ldc, stC, bc, nullptr));
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
        gemm_initData<false, true, T, I>(handle, dA, hA, dB, hB, dC, hC);

        start = get_time_us_sync(stream);
        CHECK_ROCBLAS_ERROR((rocsolver::rocsolver_gemm<T, I>)(handle, transA, transB, m, n, k,
                                                              dalpha.data(), dA.data(), 0, inca,
                                                              lda, stA, dB.data(), 0, incb, ldb,
                                                              stB, dbeta.data(), dC.data(), 0, incc,
                                                              ldc, stC, bc, nullptr));
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_gemm(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I k = argus.get<I>("k", m);
    I inca = argus.get<I>("inca", 1);
    I incb = argus.get<I>("incb", 1);
    I incc = argus.get<I>("incc", 1);

    char tA = argus.get<char>("transA", 'N');
    char tB = argus.get<char>("transB", 'N');
    rocblas_operation transA = char2rocblas_operation(tA);
    rocblas_operation transB = char2rocblas_operation(tB);
    I mk = transA == rocblas_operation_none ? m : k;
    I km = transA == rocblas_operation_none ? k : m;
    I kn = transB == rocblas_operation_none ? k : n;
    I nk = transB == rocblas_operation_none ? n : k;

    I lda = argus.get<I>("lda", mk);
    I ldb = argus.get<I>("ldb", kn);
    I ldc = argus.get<I>("ldc", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * km);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nk);
    rocblas_stride stC = argus.get<rocblas_stride>("strideC", ldc * n);

    T alpha = argus.get<T>("alpha", 1);
    T beta = argus.get<T>("beta", 1);
    I bc = argus.batch_count;
    I hot_calls = argus.iters;

    rocblas_stride stCRes = (argus.unit_check || argus.norm_check) ? stC : 0;

    size_t size_A = size_t(lda) * km;
    size_t size_B = size_t(ldb) * nk;
    size_t size_C = size_t(ldc) * n;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || m < 0 || k < 0 || ldc < m || inca < 1 || incb < 1 || incc < 1
                         || bc < 0 || lda < mk || ldb < kn);

    if(invalid_size)
    {
        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, 0);
        return;
    }

    // check quick return
    if(n == 0 || m == 0 || k == 0 || bc == 0)
    {
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T> halpha(1, 1, 1, 1);
    host_strided_batch_vector<T> hbeta(1, 1, 1, 1);
    device_strided_batch_vector<T> dalpha(1, 1, 1, 1);
    device_strided_batch_vector<T> dbeta(1, 1, 1, 1);

    halpha[0][0] = alpha;
    hbeta[0][0] = beta;

    CHECK_HIP_ERROR(dalpha.transfer_from(halpha));
    CHECK_HIP_ERROR(dbeta.transfer_from(hbeta));

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, inca, bc);
        host_batch_vector<T> hB(size_B, incb, bc);
        host_batch_vector<T> hC(size_C, incc, bc);
        host_batch_vector<T> hCRes(size_CRes, incc, bc);
        device_batch_vector<T> dA(size_A, inca, bc);
        device_batch_vector<T> dB(size_B, incb, bc);
        device_batch_vector<T> dC(size_C, incc, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gemm_getError<T, I>(handle, transA, transB, m, n, k, dalpha, dA, inca, lda, stA, dB,
                                incb, ldb, stB, dbeta, dC, incc, ldc, stC, bc, halpha, hbeta, hA,
                                hB, hC, hCRes, &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            gemm_getPerfData<T, I>(handle, transA, transB, m, n, k, dalpha, dA, inca, lda, stA, dB,
                                   incb, ldb, stB, dbeta, dC, incc, ldc, stC, bc, halpha, hbeta, hA,
                                   hB, hC, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                   argus.profile_kernels, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, inca, stA, bc);
        host_strided_batch_vector<T> hB(size_B, incb, stB, bc);
        host_strided_batch_vector<T> hC(size_C, incc, stC, bc);
        host_strided_batch_vector<T> hCRes(size_CRes, incc, stCRes, bc);
        device_strided_batch_vector<T> dA(size_A, inca, stA, bc);
        device_strided_batch_vector<T> dB(size_B, incb, stB, bc);
        device_strided_batch_vector<T> dC(size_C, incc, stC, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gemm_getError<T, I>(handle, transA, transB, m, n, k, dalpha, dA, inca, lda, stA, dB,
                                incb, ldb, stB, dbeta, dC, incc, ldc, stC, bc, halpha, hbeta, hA,
                                hB, hC, hCRes, &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            gemm_getPerfData<T, I>(handle, transA, transB, m, n, k, dalpha, dA, inca, lda, stA, dB,
                                   incb, ldb, stB, dbeta, dC, incc, ldc, stC, bc, halpha, hbeta, hA,
                                   hB, hC, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                   argus.profile_kernels, argus.perf);
        }
    }

    // validate results for rocsolver-test
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
                rocsolver_bench_output("m", "n", "k", "lda", "ldb", "ldc", "transA", "transB",
                                       "batch_count");
                rocsolver_bench_output(m, n, k, lda, ldb, ldc, tA, tB, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "k", "lda", "ldb", "ldc", "strideA", "strideB",
                                       "strideC", "transA", "transB", "batch_count");
                rocsolver_bench_output(m, n, k, lda, ldb, ldc, stA, stB, stC, tA, tB, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "k", "lda", "ldb", "ldc", "transA", "transB");
                rocsolver_bench_output(m, n, k, lda, ldb, ldc, tA, tB);
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

#define EXTERN_TESTING_GEMM(...) extern template void testing_gemm<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEMM,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
