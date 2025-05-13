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
void orgtr_ungtr_checkBadArgs(const hipsolverHandle_t   handle,
                              const hipsolverFillMode_t uplo,
                              const int                 n,
                              T                         dA,
                              const int                 lda,
                              T                         dIpiv,
                              T                         dWork,
                              const int                 lwork,
                              U                         dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgtr_ungtr(API, nullptr, uplo, n, dA, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgtr_ungtr(
            API, handle, hipsolverFillMode_t(-1), n, dA, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgtr_ungtr(API, handle, uplo, n, (T) nullptr, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgtr_ungtr(API, handle, uplo, n, dA, lda, (T) nullptr, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgtr_ungtr(API, handle, uplo, n, dA, lda, dIpiv, dWork, lwork, (U) nullptr),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, typename T>
void testing_orgtr_ungtr_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolverFillMode_t    uplo = HIPSOLVER_FILL_MODE_UPPER;
    int                    n    = 2;
    int                    lda  = 2;

    // memory allocation
    device_strided_batch_vector<T>   dA(1, 1, 1, 1);
    device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_orgtr_ungtr_bufferSize(API, handle, uplo, n, dA.data(), lda, dIpiv.data(), &size_W);
    size_t                         bytes_W = sizeof(T) * size_W;
    device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    orgtr_ungtr_checkBadArgs<API>(
        handle, uplo, n, dA.data(), lda, dIpiv.data(), dWork.data(), size_W, dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgtr_ungtr_initData(const hipsolverHandle_t   handle,
                          const hipsolverFillMode_t uplo,
                          const int                 n,
                          Td&                       dA,
                          const int                 lda,
                          Td&                       dIpiv,
                          Th&                       hA,
                          Th&                       hIpiv,
                          std::vector<T>&           hW,
                          size_t                    size_W)
{
    if(CPU)
    {
        using S          = decltype(std::real(T{}));
        size_t         s = max(hIpiv.n(), int64_t(2));
        std::vector<S> E(s - 1);
        std::vector<S> D(s);

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

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

        // compute sytrd/hetrd
        cpu_sytrd_hetrd(uplo, n, hA[0], lda, D.data(), E.data(), hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgtr_ungtr_getError(const hipsolverHandle_t   handle,
                          const hipsolverFillMode_t uplo,
                          const int                 n,
                          Td&                       dA,
                          const int                 lda,
                          Td&                       dIpiv,
                          Td&                       dWork,
                          const int                 lwork,
                          Ud&                       dInfo,
                          Th&                       hA,
                          Th&                       hARes,
                          Th&                       hIpiv,
                          Uh&                       hInfo,
                          Uh&                       hInfoRes,
                          double*                   max_err)
{
    size_t         size_W = n * 32;
    std::vector<T> hW(size_W);

    // initialize data
    orgtr_ungtr_initData<true, true, T>(handle, uplo, n, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_orgtr_ungtr(
        API, handle, uplo, n, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    cpu_orgtr_ungtr(uplo, n, hA[0], lda, hIpiv[0], hW.data(), size_W, hInfo[0]);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', n, n, lda, hA[0], hARes[0]);

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err += 1;
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgtr_ungtr_getPerfData(const hipsolverHandle_t   handle,
                             const hipsolverFillMode_t uplo,
                             const int                 n,
                             Td&                       dA,
                             const int                 lda,
                             Td&                       dIpiv,
                             Td&                       dWork,
                             const int                 lwork,
                             Ud&                       dInfo,
                             Th&                       hA,
                             Th&                       hIpiv,
                             Uh&                       hInfo,
                             double*                   gpu_time_used,
                             double*                   cpu_time_used,
                             const int                 hot_calls,
                             const bool                perf)
{
    size_t         size_W = n * 32;
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgtr_ungtr_initData<true, false, T>(
            handle, uplo, n, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_orgtr_ungtr(uplo, n, hA[0], lda, hIpiv[0], hW.data(), size_W, hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orgtr_ungtr_initData<true, false, T>(handle, uplo, n, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgtr_ungtr_initData<false, true, T>(
            handle, uplo, n, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        CHECK_ROCBLAS_ERROR(hipsolver_orgtr_ungtr(
            API, handle, uplo, n, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgtr_ungtr_initData<false, true, T>(
            handle, uplo, n, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        start = get_time_us_sync(stream);
        hipsolver_orgtr_ungtr(
            API, handle, uplo, n, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, typename T>
void testing_orgtr_ungtr(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   uploC = argus.get<char>("uplo");
    int                    n     = argus.get<int>("n");
    int                    lda   = argus.get<int>("lda", n);

    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    // size_P could be zero in test cases that are not quick-return or invalid
    // cases setting it to one to avoid possible memory access errors in the rest
    // of the unit test
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            hipsolver_orgtr_ungtr(
                API, handle, uplo, n, (T*)nullptr, lda, (T*)nullptr, (T*)nullptr, 0, (int*)nullptr),
            HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_orgtr_ungtr_bufferSize(API, handle, uplo, n, (T*)nullptr, lda, (T*)nullptr, &size_W);
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
        orgtr_ungtr_getError<API, T>(handle,
                                     uplo,
                                     n,
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
        orgtr_ungtr_getPerfData<API, T>(handle,
                                        uplo,
                                        n,
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
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("uplo", "n", "lda");
            rocsolver_bench_output(uploC, n, lda);

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
