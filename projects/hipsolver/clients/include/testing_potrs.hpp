/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <testAPI_t API, typename T, typename U, typename V>
void potrs_checkBadArgs(const hipsolverHandle_t   handle,
                        const hipsolverFillMode_t uplo,
                        const int                 n,
                        const int                 nrhs,
                        T                         dA,
                        const int                 lda,
                        const int                 stA,
                        T                         dB,
                        const int                 ldb,
                        const int                 stB,
                        V                         dWork,
                        const int                 lwork,
                        U                         dInfo,
                        const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_potrs(
            API, nullptr, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, dWork, lwork, dInfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                          handle,
                                          hipsolverFillMode_t(-1),
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dB,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                          handle,
                                          uplo,
                                          n,
                                          nrhs,
                                          (T) nullptr,
                                          lda,
                                          stA,
                                          dB,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                          handle,
                                          uplo,
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          (T) nullptr,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_potrs(
            API, handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, dWork, lwork, (U) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_potrs_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    int                    n    = 1;
    int                    nrhs = 1;
    int                    lda  = 1;
    int                    ldb  = 1;
    int                    stA  = 1;
    int                    stB  = 1;
    int                    bc   = 1;
    hipsolverFillMode_t    uplo = HIPSOLVER_FILL_MODE_UPPER;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T>           dA(1, 1, 1);
        device_batch_vector<T>           dB(1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_potrs_bufferSize(
            API, handle, uplo, n, nrhs, dA.data(), lda, dB.data(), ldb, &size_W, bc);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        potrs_checkBadArgs<API>(handle,
                                uplo,
                                n,
                                nrhs,
                                dA.data(),
                                lda,
                                stA,
                                dB.data(),
                                ldb,
                                stB,
                                dWork.data(),
                                size_W,
                                dInfo.data(),
                                bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_potrs_bufferSize(
            API, handle, uplo, n, nrhs, dA.data(), lda, dB.data(), ldb, &size_W, bc);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        potrs_checkBadArgs<API>(handle,
                                uplo,
                                n,
                                nrhs,
                                dA.data(),
                                lda,
                                stA,
                                dB.data(),
                                ldb,
                                stB,
                                dWork.data(),
                                size_W,
                                dInfo.data(),
                                bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void potrs_initData(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const int                 n,
                    const int                 nrhs,
                    Td&                       dA,
                    const int                 lda,
                    const int                 stA,
                    Td&                       dB,
                    const int                 ldb,
                    const int                 stB,
                    const int                 bc,
                    Th&                       hA,
                    Th&                       hB)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);
        int info;

        for(int b = 0; b < bc; ++b)
        {
            // scale to ensure positive definiteness
            for(int i = 0; i < n; i++)
                hA[b][i + i * lda] = hA[b][i + i * lda] * conj(hA[b][i + i * lda]) * 400;

            // do the Cholesky factorization of matrix A w/ the reference LAPACK routine
            cpu_potrf(uplo, n, hA[b], lda, &info);
        }
    }

    if(GPU)
    {
        // now copy matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <testAPI_t API,
          typename T,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh>
void potrs_getError(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const int                 n,
                    const int                 nrhs,
                    Td&                       dA,
                    const int                 lda,
                    const int                 stA,
                    Td&                       dB,
                    const int                 ldb,
                    const int                 stB,
                    Vd&                       dWork,
                    const int                 lwork,
                    Ud&                       dInfo,
                    const int                 bc,
                    Th&                       hA,
                    Th&                       hB,
                    Th&                       hBRes,
                    Uh&                       hInfo,
                    Uh&                       hInfoRes,
                    double*                   max_err)
{
    // input data initialization
    potrs_initData<true, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_potrs(API,
                                        handle,
                                        uplo,
                                        n,
                                        nrhs,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dB.data(),
                                        ldb,
                                        stB,
                                        dWork.data(),
                                        lwork,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
    {
        cpu_potrs(uplo, n, nrhs, hA[b], lda, hB[b], ldb, hInfo[b]);
    }

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('I', n, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }

    // check info
    err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <testAPI_t API,
          typename T,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh>
void potrs_getPerfData(const hipsolverHandle_t   handle,
                       const hipsolverFillMode_t uplo,
                       const int                 n,
                       const int                 nrhs,
                       Td&                       dA,
                       const int                 lda,
                       const int                 stA,
                       Td&                       dB,
                       const int                 ldb,
                       const int                 stB,
                       Vd&                       dWork,
                       const int                 lwork,
                       Ud&                       dInfo,
                       const int                 bc,
                       Th&                       hA,
                       Th&                       hB,
                       Uh&                       hInfo,
                       double*                   gpu_time_used,
                       double*                   cpu_time_used,
                       const int                 hot_calls,
                       const bool                perf)
{
    if(!perf)
    {
        potrs_initData<true, false, T>(
            handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cpu_potrs(uplo, n, nrhs, hA[b], lda, hB[b], ldb, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    potrs_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        potrs_initData<false, true, T>(
            handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

        CHECK_ROCBLAS_ERROR(hipsolver_potrs(API,
                                            handle,
                                            uplo,
                                            n,
                                            nrhs,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dB.data(),
                                            ldb,
                                            stB,
                                            dWork.data(),
                                            lwork,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        potrs_initData<false, true, T>(
            handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB);

        start = get_time_us_sync(stream);
        hipsolver_potrs(API,
                        handle,
                        uplo,
                        n,
                        nrhs,
                        dA.data(),
                        lda,
                        stA,
                        dB.data(),
                        ldb,
                        stB,
                        dWork.data(),
                        lwork,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_potrs(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   uploC = argus.get<char>("uplo");
    int                    n     = argus.get<int>("n");
    int                    nrhs  = argus.get<int>("nrhs", n);
    int                    lda   = argus.get<int>("lda", n);
    int                    ldb   = argus.get<int>("ldb", n);
    int                    stA   = argus.get<int>("strideA", lda * n);
    int                    stB   = argus.get<int>("strideB", ldb * nrhs);

    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    int stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    if(uplo != HIPSOLVER_FILL_MODE_UPPER && uplo != HIPSOLVER_FILL_MODE_LOWER)
    {
        if(BATCHED)
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  nrhs,
                                                  (T**)nullptr,
                                                  lda,
                                                  stA,
                                                  (T**)nullptr,
                                                  ldb,
                                                  stB,
                                                  (T*)nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  nrhs,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (T*)nullptr,
                                                  ldb,
                                                  stB,
                                                  (T*)nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_B    = size_t(ldb) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  nrhs,
                                                  (T**)nullptr,
                                                  lda,
                                                  stA,
                                                  (T**)nullptr,
                                                  ldb,
                                                  stB,
                                                  (T*)nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_potrs(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  nrhs,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (T*)nullptr,
                                                  ldb,
                                                  stB,
                                                  (T*)nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    if(BATCHED)
        hipsolver_potrs_bufferSize(
            API, handle, uplo, n, nrhs, (T**)nullptr, lda, (T**)nullptr, ldb, &size_W, bc);
    else
        hipsolver_potrs_bufferSize(
            API, handle, uplo, n, nrhs, (T*)nullptr, lda, (T*)nullptr, ldb, &size_W, bc);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T>             hA(size_A, 1, bc);
        host_batch_vector<T>             hB(size_B, 1, bc);
        host_batch_vector<T>             hBRes(size_BRes, 1, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_batch_vector<T>           dA(size_A, 1, bc);
        device_batch_vector<T>           dB(size_B, 1, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            potrs_getError<API, T>(handle,
                                   uplo,
                                   n,
                                   nrhs,
                                   dA,
                                   lda,
                                   stA,
                                   dB,
                                   ldb,
                                   stB,
                                   dWork,
                                   size_W,
                                   dInfo,
                                   bc,
                                   hA,
                                   hB,
                                   hBRes,
                                   hInfo,
                                   hInfoRes,
                                   &max_error);

        // collect performance data
        if(argus.timing)
            potrs_getPerfData<API, T>(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      dA,
                                      lda,
                                      stA,
                                      dB,
                                      ldb,
                                      stB,
                                      dWork,
                                      size_W,
                                      dInfo,
                                      bc,
                                      hA,
                                      hB,
                                      hInfo,
                                      &gpu_time_used,
                                      &cpu_time_used,
                                      hot_calls,
                                      argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>     hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T>     hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T>   dB(size_B, 1, stB, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            potrs_getError<API, T>(handle,
                                   uplo,
                                   n,
                                   nrhs,
                                   dA,
                                   lda,
                                   stA,
                                   dB,
                                   ldb,
                                   stB,
                                   dWork,
                                   size_W,
                                   dInfo,
                                   bc,
                                   hA,
                                   hB,
                                   hBRes,
                                   hInfo,
                                   hInfoRes,
                                   &max_error);

        // collect performance data
        if(argus.timing)
            potrs_getPerfData<API, T>(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      dA,
                                      lda,
                                      stA,
                                      dB,
                                      ldb,
                                      stB,
                                      dWork,
                                      size_W,
                                      dInfo,
                                      bc,
                                      hA,
                                      hB,
                                      hInfo,
                                      &gpu_time_used,
                                      &cpu_time_used,
                                      hot_calls,
                                      argus.perf);
    }

    // validate results for rocsolver-test
    // using m * machine_precision as tolerance
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
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb", "batch_c");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output(
                    "uplo", "n", "nrhs", "lda", "ldb", "strideA", "strideB", "batch_c");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb, stA, stB, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb);
            }
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
