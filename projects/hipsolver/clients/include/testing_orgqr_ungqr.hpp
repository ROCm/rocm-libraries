/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"

template <testAPI_t API, typename T, typename U>
void orgqr_ungqr_checkBadArgs(const hipsolverHandle_t handle,
                              const int               m,
                              const int               n,
                              const int               k,
                              T                       dA,
                              const int               lda,
                              T                       dIpiv,
                              T                       dWork,
                              const int               lwork,
                              U                       dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgqr_ungqr(API, nullptr, m, n, k, dA, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgqr_ungqr(API, handle, m, n, k, (T) nullptr, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgqr_ungqr(API, handle, m, n, k, dA, lda, (T) nullptr, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgqr_ungqr(API, handle, m, n, k, dA, lda, dIpiv, dWork, lwork, (U) nullptr),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, typename T>
void testing_orgqr_ungqr_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    int                    k   = 1;
    int                    m   = 1;
    int                    n   = 1;
    int                    lda = 1;

    // memory allocation
    device_strided_batch_vector<T>   dA(1, 1, 1, 1);
    device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_orgqr_ungqr_bufferSize(API, handle, m, n, k, dA.data(), lda, dIpiv.data(), &size_W);
    size_t                         bytes_W = sizeof(T) * size_W;
    device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    orgqr_ungqr_checkBadArgs<API>(
        handle, m, n, k, dA.data(), lda, dIpiv.data(), dWork.data(), size_W, dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgqr_ungqr_initData(const hipsolverHandle_t handle,
                          const int               m,
                          const int               n,
                          const int               k,
                          Td&                     dA,
                          const int               lda,
                          Td&                     dIpiv,
                          Th&                     hA,
                          Th&                     hIpiv,
                          std::vector<T>&         hW,
                          size_t                  size_W)
{
    if(CPU)
    {
        int info;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                if(i == j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute QR factorization
        cpu_geqrf(m, n, hA[0], lda, hIpiv[0], hW.data(), size_W, &info);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgqr_ungqr_getError(const hipsolverHandle_t handle,
                          const int               m,
                          const int               n,
                          const int               k,
                          Td&                     dA,
                          const int               lda,
                          Td&                     dIpiv,
                          Td&                     dWork,
                          const int               lwork,
                          Ud&                     dInfo,
                          Th&                     hA,
                          Th&                     hARes,
                          Th&                     hIpiv,
                          Uh&                     hInfo,
                          Uh&                     hInfoRes,
                          double*                 max_err)
{
    size_t         size_W = size_t(n);
    std::vector<T> hW(size_W);

    // initialize data
    orgqr_ungqr_initData<true, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_orgqr_ungqr(
        API, handle, m, n, k, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    cpu_orgqr_ungqr(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W, hInfo[0]);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, lda, hA[0], hARes[0]);

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err += 1;
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgqr_ungqr_getPerfData(const hipsolverHandle_t handle,
                             const int               m,
                             const int               n,
                             const int               k,
                             Td&                     dA,
                             const int               lda,
                             Td&                     dIpiv,
                             Td&                     dWork,
                             const int               lwork,
                             Ud&                     dInfo,
                             Th&                     hA,
                             Th&                     hIpiv,
                             Uh&                     hInfo,
                             double*                 gpu_time_used,
                             double*                 cpu_time_used,
                             const int               hot_calls,
                             const bool              perf)
{
    size_t         size_W = size_t(n);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgqr_ungqr_initData<true, false, T>(
            handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_orgqr_ungqr(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W, hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orgqr_ungqr_initData<true, false, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgqr_ungqr_initData<false, true, T>(
            handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        CHECK_ROCBLAS_ERROR(hipsolver_orgqr_ungqr(
            API, handle, m, n, k, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgqr_ungqr_initData<false, true, T>(
            handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        start = get_time_us_sync(stream);
        hipsolver_orgqr_ungqr(
            API, handle, m, n, k, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, typename T>
void testing_orgqr_ungqr(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    int                    n   = argus.get<int>("n");
    int                    m   = argus.get<int>("m", n);
    int                    k   = argus.get<int>("k", n);
    int                    lda = argus.get<int>("lda", m);

    int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_P    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || k < 0 || lda < m || n > m || k > n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            hipsolver_orgqr_ungqr(
                API, handle, m, n, k, (T*)nullptr, lda, (T*)nullptr, (T*)nullptr, 0, (int*)nullptr),
            HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_orgqr_ungqr_bufferSize(API, handle, m, n, k, (T*)nullptr, lda, (T*)nullptr, &size_W);
    size_t bytes_W = sizeof(T) * size_W;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_W);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T>     hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T>     hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T>     hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    host_strided_batch_vector<int>   hInfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T>   dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T>   dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    device_strided_batch_vector<T>   dWork(bytes_W, 1, bytes_W, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
        orgqr_ungqr_getError<API, T>(handle,
                                     m,
                                     n,
                                     k,
                                     dA,
                                     lda,
                                     dIpiv,
                                     dWork,
                                     size_W,
                                     dInfo,
                                     hA,
                                     hARes,
                                     hIpiv,
                                     hInfo,
                                     hInfoRes,
                                     &max_error);

    // collect performance data
    if(argus.timing)
        orgqr_ungqr_getPerfData<API, T>(handle,
                                        m,
                                        n,
                                        k,
                                        dA,
                                        lda,
                                        dIpiv,
                                        dWork,
                                        size_W,
                                        dInfo,
                                        hA,
                                        hIpiv,
                                        hInfo,
                                        &gpu_time_used,
                                        &cpu_time_used,
                                        hot_calls,
                                        argus.perf);

    // validate results for rocsolver-test
    // using m * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, m);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("m", "n", "k", "lda");
            rocsolver_bench_output(m, n, k, lda);

            std::cerr << "\n============================================\n";
            std::cerr << "Results:\n";
            std::cerr << "============================================\n";
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            std::cerr << std::endl;
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
