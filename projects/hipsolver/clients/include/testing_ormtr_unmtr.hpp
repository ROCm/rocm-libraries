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

template <testAPI_t API, bool COMPLEX, typename T, typename U>
void ormtr_unmtr_checkBadArgs(const hipsolverHandle_t    handle,
                              const hipsolverSideMode_t  side,
                              const hipsolverFillMode_t  uplo,
                              const hipsolverOperation_t trans,
                              const int                  m,
                              const int                  n,
                              T                          dA,
                              const int                  lda,
                              T                          dIpiv,
                              T                          dC,
                              const int                  ldc,
                              T                          dWork,
                              const int                  lwork,
                              U                          dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_ormtr_unmtr(
            API, nullptr, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                hipsolverSideMode_t(-1),
                                                uplo,
                                                trans,
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                side,
                                                hipsolverFillMode_t(-1),
                                                trans,
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                side,
                                                uplo,
                                                hipsolverOperation_t(-1),
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    if(COMPLEX)
        EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                    handle,
                                                    side,
                                                    uplo,
                                                    HIPSOLVER_OP_T,
                                                    m,
                                                    n,
                                                    dA,
                                                    lda,
                                                    dIpiv,
                                                    dC,
                                                    ldc,
                                                    dWork,
                                                    lwork,
                                                    dInfo),
                              HIPSOLVER_STATUS_INVALID_VALUE);
    else
        EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                    handle,
                                                    side,
                                                    uplo,
                                                    HIPSOLVER_OP_C,
                                                    m,
                                                    n,
                                                    dA,
                                                    lda,
                                                    dIpiv,
                                                    dC,
                                                    ldc,
                                                    dWork,
                                                    lwork,
                                                    dInfo),
                              HIPSOLVER_STATUS_INVALID_VALUE);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                side,
                                                uplo,
                                                trans,
                                                m,
                                                n,
                                                (T) nullptr,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                side,
                                                uplo,
                                                trans,
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                (T) nullptr,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                side,
                                                uplo,
                                                trans,
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                dIpiv,
                                                (T) nullptr,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                handle,
                                                side,
                                                uplo,
                                                trans,
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                (U) nullptr),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, typename T, bool COMPLEX = is_complex<T>>
void testing_ormtr_unmtr_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolverSideMode_t    side  = HIPSOLVER_SIDE_LEFT;
    hipsolverFillMode_t    uplo  = HIPSOLVER_FILL_MODE_UPPER;
    hipsolverOperation_t   trans = HIPSOLVER_OP_N;
    int                    m     = 2;
    int                    n     = 2;
    int                    lda   = 2;
    int                    ldc   = 2;

    // memory allocation
    device_strided_batch_vector<T>   dA(1, 1, 1, 1);
    device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<T>   dC(1, 1, 1, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_ormtr_unmtr_bufferSize(API,
                                     handle,
                                     side,
                                     uplo,
                                     trans,
                                     m,
                                     n,
                                     dA.data(),
                                     lda,
                                     dIpiv.data(),
                                     dC.data(),
                                     ldc,
                                     &size_W);
    size_t bytes_W
        = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;
    device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    ormtr_unmtr_checkBadArgs<API, COMPLEX>(handle,
                                           side,
                                           uplo,
                                           trans,
                                           m,
                                           n,
                                           dA.data(),
                                           lda,
                                           dIpiv.data(),
                                           dC.data(),
                                           ldc,
                                           dWork.data(),
                                           size_W,
                                           dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormtr_unmtr_initData(const hipsolverHandle_t    handle,
                          const hipsolverSideMode_t  side,
                          const hipsolverFillMode_t  uplo,
                          const hipsolverOperation_t trans,
                          const int                  m,
                          const int                  n,
                          Td&                        dA,
                          const int                  lda,
                          Td&                        dIpiv,
                          Td&                        dC,
                          const int                  ldc,
                          Th&                        hA,
                          Th&                        hIpiv,
                          Th&                        hC,
                          std::vector<T>&            hW,
                          size_t                     size_W)
{
    if(CPU)
    {
        using S           = decltype(std::real(T{}));
        int            nq = (side == HIPSOLVER_SIDE_LEFT) ? m : n;
        std::vector<S> E(nq - 1);
        std::vector<S> D(nq);

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);
        rocblas_init<T>(hC, true);

        // scale to avoid singularities
        for(int i = 0; i < nq; ++i)
        {
            for(int j = 0; j < nq; ++j)
            {
                if(i == j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute sytrd/hetrd
        cpu_sytrd_hetrd(uplo, nq, hA[0], lda, D.data(), E.data(), hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void ormtr_unmtr_getError(const hipsolverHandle_t    handle,
                          const hipsolverSideMode_t  side,
                          const hipsolverFillMode_t  uplo,
                          const hipsolverOperation_t trans,
                          const int                  m,
                          const int                  n,
                          Td&                        dA,
                          const int                  lda,
                          Td&                        dIpiv,
                          Td&                        dC,
                          const int                  ldc,
                          Td&                        dWork,
                          const int                  lwork,
                          Ud&                        dInfo,
                          Th&                        hA,
                          Th&                        hIpiv,
                          Th&                        hC,
                          Th&                        hCRes,
                          Uh&                        hInfo,
                          Uh&                        hInfoRes,
                          double*                    max_err)
{
    size_t         size_W = (side == HIPSOLVER_SIDE_LEFT ? m : n) * 32;
    std::vector<T> hW(size_W);

    // initialize data
    ormtr_unmtr_initData<true, true, T>(
        handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_ormtr_unmtr(API,
                                              handle,
                                              side,
                                              uplo,
                                              trans,
                                              m,
                                              n,
                                              dA.data(),
                                              lda,
                                              dIpiv.data(),
                                              dC.data(),
                                              ldc,
                                              dWork.data(),
                                              lwork,
                                              dInfo.data()));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    cpu_ormtr_unmtr(
        side, uplo, trans, m, n, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(), size_W, hInfo[0]);

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCRes[0]);

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err += 1;
}

template <testAPI_t API, typename T, typename Td, typename Ud, typename Th, typename Uh>
void ormtr_unmtr_getPerfData(const hipsolverHandle_t    handle,
                             const hipsolverSideMode_t  side,
                             const hipsolverFillMode_t  uplo,
                             const hipsolverOperation_t trans,
                             const int                  m,
                             const int                  n,
                             Td&                        dA,
                             const int                  lda,
                             Td&                        dIpiv,
                             Td&                        dC,
                             const int                  ldc,
                             Td&                        dWork,
                             const int                  lwork,
                             Ud&                        dInfo,
                             Th&                        hA,
                             Th&                        hIpiv,
                             Th&                        hC,
                             Uh&                        hInfo,
                             double*                    gpu_time_used,
                             double*                    cpu_time_used,
                             const int                  hot_calls,
                             const bool                 perf)
{
    size_t         size_W = (side == HIPSOLVER_SIDE_LEFT ? m : n) * 32;
    std::vector<T> hW(size_W);

    if(!perf)
    {
        ormtr_unmtr_initData<true, false, T>(
            handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_ormtr_unmtr(
            side, uplo, trans, m, n, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(), size_W, hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    ormtr_unmtr_initData<true, false, T>(
        handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormtr_unmtr_initData<false, true, T>(
            handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(hipsolver_ormtr_unmtr(API,
                                                  handle,
                                                  side,
                                                  uplo,
                                                  trans,
                                                  m,
                                                  n,
                                                  dA.data(),
                                                  lda,
                                                  dIpiv.data(),
                                                  dC.data(),
                                                  ldc,
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
        ormtr_unmtr_initData<false, true, T>(
            handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

        start = get_time_us_sync(stream);
        hipsolver_ormtr_unmtr(API,
                              handle,
                              side,
                              uplo,
                              trans,
                              m,
                              n,
                              dA.data(),
                              lda,
                              dIpiv.data(),
                              dC.data(),
                              ldc,
                              dWork.data(),
                              lwork,
                              dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, typename T, bool COMPLEX = is_complex<T>>
void testing_ormtr_unmtr(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   sideC  = argus.get<char>("side");
    char                   uploC  = argus.get<char>("uplo");
    char                   transC = argus.get<char>("trans");
    int                    m, n;
    if(sideC == 'L')
    {
        m = argus.get<int>("m");
        n = argus.get<int>("n", m);
    }
    else
    {
        n = argus.get<int>("n");
        m = argus.get<int>("m", n);
    }
    int nq  = (sideC == 'L' ? m : n);
    int lda = argus.get<int>("lda", nq);
    int ldc = argus.get<int>("ldc", m);

    hipsolverSideMode_t  side      = char2hipsolver_side(sideC);
    hipsolverFillMode_t  uplo      = char2hipsolver_fill(uploC);
    hipsolverOperation_t trans     = char2hipsolver_operation(transC);
    int                  hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value
        = ((COMPLEX && trans == HIPSOLVER_OP_T) || (!COMPLEX && trans == HIPSOLVER_OP_C));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                    handle,
                                                    side,
                                                    uplo,
                                                    trans,
                                                    m,
                                                    n,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    ldc,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    bool   left   = (side == HIPSOLVER_SIDE_LEFT);
    size_t size_P = size_t(nq);
    size_t size_C = size_t(ldc) * n;

    size_t size_A    = size_t(lda) * nq;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || ldc < m || lda < nq);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_ormtr_unmtr(API,
                                                    handle,
                                                    side,
                                                    uplo,
                                                    trans,
                                                    m,
                                                    n,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    ldc,
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
    hipsolver_ormtr_unmtr_bufferSize(API,
                                     handle,
                                     side,
                                     uplo,
                                     trans,
                                     m,
                                     n,
                                     (T*)nullptr,
                                     lda,
                                     (T*)nullptr,
                                     (T*)nullptr,
                                     ldc,
                                     &size_W);
    size_t bytes_W
        = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_W);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T>     hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T>     hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<T>     hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<T>     hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    host_strided_batch_vector<int>   hInfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T>   dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T>   dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T>   dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    device_strided_batch_vector<T>   dWork(bytes_W, 1, bytes_W, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormtr_unmtr_getError<API, T>(handle,
                                     side,
                                     uplo,
                                     trans,
                                     m,
                                     n,
                                     dA,
                                     lda,
                                     dIpiv,
                                     dC,
                                     ldc,
                                     dWork,
                                     size_W,
                                     dInfo,
                                     hA,
                                     hIpiv,
                                     hC,
                                     hCRes,
                                     hInfo,
                                     hInfoRes,
                                     &max_error);

    // collect performance data
    if(argus.timing)
        ormtr_unmtr_getPerfData<API, T>(handle,
                                        side,
                                        uplo,
                                        trans,
                                        m,
                                        n,
                                        dA,
                                        lda,
                                        dIpiv,
                                        dC,
                                        ldc,
                                        dWork,
                                        size_W,
                                        dInfo,
                                        hA,
                                        hIpiv,
                                        hC,
                                        hInfo,
                                        &gpu_time_used,
                                        &cpu_time_used,
                                        hot_calls,
                                        argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    int s = left ? m : n;
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
            rocsolver_bench_output("side", "uplo", "trans", "m", "n", "lda", "ldc");
            rocsolver_bench_output(sideC, uploC, transC, m, n, lda, ldc);

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
