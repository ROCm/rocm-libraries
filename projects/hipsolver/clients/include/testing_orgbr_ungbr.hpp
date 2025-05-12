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
void orgbr_ungbr_checkBadArgs(const hipsolverHandle_t   handle,
                              const hipsolverSideMode_t side,
                              const int                 m,
                              const int                 n,
                              const int                 k,
                              T                         dA,
                              const int                 lda,
                              T                         dIpiv,
                              T                         dWork,
                              const int                 lwork,
                              U                         dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgbr_ungbr(API, nullptr, side, m, n, k, dA, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgbr_ungbr(
            API, handle, hipsolverSideMode_t(-1), m, n, k, dA, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgbr_ungbr(
            API, handle, side, m, n, k, (T) nullptr, lda, dIpiv, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgbr_ungbr(
            API, handle, side, m, n, k, dA, lda, (T) nullptr, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_orgbr_ungbr(
            API, handle, side, m, n, k, dA, lda, dIpiv, dWork, lwork, (U) nullptr),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, typename T>
void testing_orgbr_ungbr_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolverSideMode_t    side = HIPSOLVER_SIDE_LEFT;
    int                    k    = 1;
    int                    m    = 1;
    int                    n    = 1;
    int                    lda  = 1;

    // memory allocation
    device_strided_batch_vector<T>   dA(1, 1, 1, 1);
    device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_orgbr_ungbr_bufferSize(
        API, handle, side, m, n, k, dA.data(), lda, dIpiv.data(), &size_W);
    size_t                         bytes_W = sizeof(T) * size_W;
    device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    orgbr_ungbr_checkBadArgs<API>(
        handle, side, m, n, k, dA.data(), lda, dIpiv.data(), dWork.data(), size_W, dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgbr_ungbr_initData(const hipsolverHandle_t   handle,
                          const hipsolverSideMode_t side,
                          const int                 m,
                          const int                 n,
                          const int                 k,
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
        using S = decltype(std::real(T{}));

        int            info;
        size_t         s = max(hIpiv.n(), int64_t(2));
        std::vector<S> E(s - 1);
        std::vector<S> D(s);
        std::vector<T> P(s);

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
        // and compute gebrd
        if(side == HIPSOLVER_SIDE_LEFT)
        {
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
            cpu_gebrd(
                m, k, hA[0], lda, D.data(), E.data(), hIpiv[0], P.data(), hW.data(), size_W, &info);
        }
        else
        {
            for(int i = 0; i < k; ++i)
            {
                for(int j = 0; j < n; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cpu_gebrd(
                k, n, hA[0], lda, D.data(), E.data(), P.data(), hIpiv[0], hW.data(), size_W, &info);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgbr_ungbr_getError(const hipsolverHandle_t   handle,
                          const hipsolverSideMode_t side,
                          const int                 m,
                          const int                 n,
                          const int                 k,
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
    size_t         size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    // initialize data
    orgbr_ungbr_initData<true, true, T>(
        handle, side, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_orgbr_ungbr(API,
                                              handle,
                                              side,
                                              m,
                                              n,
                                              k,
                                              dA.data(),
                                              lda,
                                              dIpiv.data(),
                                              dWork.data(),
                                              lwork,
                                              dInfo.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    cpu_orgbr_ungbr(side, m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W, hInfo[0]);

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
void orgbr_ungbr_getPerfData(const hipsolverHandle_t   handle,
                             const hipsolverSideMode_t side,
                             const int                 m,
                             const int                 n,
                             const int                 k,
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
    size_t         size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgbr_ungbr_initData<true, false, T>(
            handle, side, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_orgbr_ungbr(side, m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W, hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orgbr_ungbr_initData<true, false, T>(
        handle, side, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgbr_ungbr_initData<false, true, T>(
            handle, side, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        CHECK_ROCBLAS_ERROR(hipsolver_orgbr_ungbr(API,
                                                  handle,
                                                  side,
                                                  m,
                                                  n,
                                                  k,
                                                  dA.data(),
                                                  lda,
                                                  dIpiv.data(),
                                                  dWork.data(),
                                                  lwork,
                                                  dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgbr_ungbr_initData<false, true, T>(
            handle, side, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        start = get_time_us_sync(stream);
        hipsolver_orgbr_ungbr(API,
                              handle,
                              side,
                              m,
                              n,
                              k,
                              dA.data(),
                              lda,
                              dIpiv.data(),
                              dWork.data(),
                              lwork,
                              dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, typename T>
void testing_orgbr_ungbr(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   sideC = argus.get<char>("side");
    int                    m, n;
    if(sideC == 'R')
    {
        m = argus.get<int>("m");
        n = argus.get<int>("n", m);
    }
    else
    {
        n = argus.get<int>("n");
        m = argus.get<int>("m", n);
    }
    int k   = argus.get<int>("k", min(m, n));
    int lda = argus.get<int>("lda", m);

    hipsolverSideMode_t side      = char2hipsolver_side(sideC);
    int                 hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    // size_P could be zero in test cases that are not quick-return or invalid
    // cases setting it to one to avoid possible memory access errors in the rest
    // of the unit test
    bool   row    = (side == HIPSOLVER_SIDE_RIGHT);
    size_t size_A = row ? size_t(lda) * n : size_t(lda) * max(n, k);
    size_t size_P = row ? max(min(n, k), 1) : max(min(m, k), 1);

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = ((m < 0 || n < 0 || k < 0 || lda < m) || (row && (m > n || m < min(n, k)))
                         || (!row && (n > m || n < min(m, k))));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_orgbr_ungbr(API,
                                                    handle,
                                                    side,
                                                    m,
                                                    n,
                                                    k,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_orgbr_ungbr_bufferSize(
        API, handle, side, m, n, k, (T*)nullptr, lda, (T*)nullptr, &size_W);
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
        orgbr_ungbr_getError<API, T>(handle,
                                     side,
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
        orgbr_ungbr_getPerfData<API, T>(handle,
                                        side,
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
    // using s * machine_precision as tolerance
    int s = row ? n : m;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("side", "m", "n", "k", "lda");
            rocsolver_bench_output(sideC, m, n, k, lda);

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
