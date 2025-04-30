/* **************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * ===========================================================================
 *    testing_getrf_large tests the correctness of getrf, getrs, and gemm. We
 *    use an implicit test that solves Ax = b for x (using getrf and getrs),
 *    then computes Ax (using gemm) and compare with b. For the sizes tested,
 *    this is much faster than calling getrf on the CPU.
 * ===========================================================================
 */

template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Ud,
          typename Th,
          typename Uh,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void getrf_large_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nrhs,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          Td& dX,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hB,
                          Uh& hIpiv,
                          Uh& hInfo,
                          const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(rocblas_int i = 0; i < n / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    tmp = hA[b][i + j * lda];
                    hA[b][i + j * lda] = hA[b][n - 1 - i + j * lda];
                    hA[b][n - 1 - i + j * lda] = tmp;
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes).
                // The algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                    hA[b][i + j * lda] = 0;
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                    hA[b][i + j * lda] = 0;
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                    hA[b][i + j * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dX.transfer_from(hB));
    }
}

template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Ud,
          typename Th,
          typename Uh,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void getrf_large_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nrhs,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          Td& dX,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hB,
                          Uh& hIpiv,
                          Uh& hInfo,
                          const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += T(400, 400);
                    else
                        hA[b][i + j * lda] -= T(4, 4);
                }
            }

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(rocblas_int i = 0; i < n / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    tmp = hA[b][i + j * lda];
                    hA[b][i + j * lda] = hA[b][n - 1 - i + j * lda];
                    hA[b][n - 1 - i + j * lda] = tmp;
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes).
                // The algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                    hA[b][i + j * lda] = 0;
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                    hA[b][i + j * lda] = 0;
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                    hA[b][i + j * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dX.transfer_from(hB));
    }
}

template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_large_getError(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nrhs,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Td& dB,
                          const rocblas_int ldb,
                          const rocblas_stride stB,
                          Td& dX,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          Ud& dInfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hB,
                          Th& hBRes,
                          Uh& hIpiv,
                          Uh& hInfo,
                          Uh& hInfoRes,
                          double* max_err,
                          const bool singular)
{
    // Input data initialization for Matrix A
    getrf_large_initData<true, true, T>(handle, n, nrhs, dA, lda, stA, dB, ldb, stB, dX, dIpiv, stP,
                                        dInfo, bc, hA, hB, hIpiv, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, dInfo.data(), bc));

    // Solve Ax = b for x
    CHECK_ROCBLAS_ERROR(rocsolver_getrs(STRIDED, handle, rocblas_operation_none, n, nrhs, dA, lda,
                                        stA, dIpiv, stP, dX, ldb, stB, bc));

    // Resetting the value of dA.
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    // Compute Ax
    T alpha = T(1), beta = T(0);
    CHECK_ROCBLAS_ERROR(rocblas_gemm(STRIDED, handle, rocblas_operation_none,
                                     rocblas_operation_none, n, nrhs, n, &alpha, dA, lda, stA, dX,
                                     ldb, stB, &beta, dB, ldb, stB, bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    { // Pass the matrices here
        err = norm_error('F', n, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

// Function for perfromance data.
template <bool STRIDED, bool GETRF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_large_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             Ud& dInfo,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             Uh& hInfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const bool singular)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution
    *gpu_time_used = nan(""); // no timing on gpu-lapack execution
}

template <bool BATCHED, bool STRIDED, bool GETRF, typename T>
void testing_getrf_large(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", n);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // determine sizes using the leading Dimensions, which are typically greater than the rows
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * nrhs;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        // if(BATCHED)
        //     EXPECT_ROCBLAS_STATUS(
        //         rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T* const*)nullptr, lda, stA,
        //                               (rocblas_int*)nullptr, stP, (rocblas_int*)nullptr, bc),
        //         rocblas_status_invalid_size);
        // else
        //     EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T*)nullptr,
        //                                                 lda, stA, (rocblas_int*)nullptr, stP,
        //                                                 (rocblas_int*)nullptr, bc),
        //                           rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T* const*)nullptr,
                                                    lda, stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf(STRIDED, GETRF, handle, n, n, (T*)nullptr, lda,
                                                    stA, (rocblas_int*)nullptr, stP,
                                                    (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dX(size_B, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);

        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
        {
            CHECK_HIP_ERROR(dB.memcheck());
            CHECK_HIP_ERROR(dX.memcheck());
        }
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            // Modify the parameters passed.
            getrf_large_getError<STRIDED, GETRF, T>(handle, n, nrhs, dA, lda, stA, dB, ldb, stB, dX,
                                                    dIpiv, stP, dInfo, bc, hA, hB, hBRes, hIpiv,
                                                    hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getrf_large_getPerfData<STRIDED, GETRF, T>(
                handle, n, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dX(size_B, 1, stB, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);

        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
        {
            CHECK_HIP_ERROR(dB.memcheck());
            CHECK_HIP_ERROR(dX.memcheck());
        }
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getrf_large_getError<STRIDED, GETRF, T>(handle, n, nrhs, dA, lda, stA, dB, ldb, stB, dX,
                                                    dIpiv, stP, dInfo, bc, hA, hB, hBRes, hIpiv,
                                                    hInfo, hInfoRes, &max_error, argus.singular);

        // The perf function must return NAN
        // collect performance data
        if(argus.timing)
            getrf_large_getPerfData<STRIDED, GETRF, T>(
                handle, n, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, min(n, n));
    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("n", "nrhs", "lda", "ldb", "strideP", "batch_c");
                rocsolver_bench_output(n, nrhs, lda, ldb, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("n", "nrhs", "lda", "strideA", "ldb", "strideB", "strideP",
                                       "batch_c");
                rocsolver_bench_output(n, nrhs, lda, stA, ldb, stB, stP, bc);
            }
            else
            {
                rocsolver_bench_output("n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(n, nrhs, lda, ldb);
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
