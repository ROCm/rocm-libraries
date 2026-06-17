/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <testAPI_t API,
          typename I,
          typename SIZE,
          typename Td,
          typename Id,
          typename Ud,
          typename INTd>
void sytrs_checkBadArgs(const hipsolverHandle_t   handle,
                        const hipsolverFillMode_t uplo,
                        const I                   n,
                        const I                   nrhs,
                        Td                        dA,
                        const I                   lda,
                        Id                        dIpiv,
                        Td                        dB,
                        const I                   ldb,
                        Ud                        dWork,
                        const SIZE                dlwork,
                        Ud                        hWork,
                        const SIZE                hlwork,
                        INTd                      dInfo,
                        const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                          nullptr,
                                          uplo,
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          dIpiv,
                                          dB,
                                          ldb,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                          handle,
                                          hipsolverFillMode_t(-1),
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          dIpiv,
                                          dB,
                                          ldb,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                          handle,
                                          uplo,
                                          n,
                                          nrhs,
                                          (Td) nullptr,
                                          lda,
                                          dIpiv,
                                          dB,
                                          ldb,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                          handle,
                                          uplo,
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          (Id) nullptr,
                                          dB,
                                          ldb,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                          handle,
                                          uplo,
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          dIpiv,
                                          (Td) nullptr,
                                          ldb,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                          handle,
                                          uplo,
                                          n,
                                          nrhs,
                                          dA,
                                          lda,
                                          dIpiv,
                                          dB,
                                          ldb,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          (INTd) nullptr,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_sytrs_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolverFillMode_t    uplo = HIPSOLVER_FILL_MODE_UPPER;
    I                      n    = 1;
    I                      nrhs = 1;
    I                      lda  = 1;
    I                      ldb  = 1;
    int                    bc   = 1;

    if(BATCHED)
    {
        // // memory allocations
        // (no batched DnX API for sytrs)
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<I>   dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        SIZE bytes_dW, bytes_hW;
        hipsolver_sytrs_bufferSize(API,
                                   handle,
                                   uplo,
                                   n,
                                   nrhs,
                                   dA.data(),
                                   lda,
                                   dIpiv.data(),
                                   dB.data(),
                                   ldb,
                                   &bytes_dW,
                                   &bytes_hW);
        SIZE size_dW = (bytes_dW + sizeof(T) - 1) / sizeof(T);
        SIZE size_hW = (bytes_hW + sizeof(T) - 1) / sizeof(T);

        host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        sytrs_checkBadArgs<API>(handle,
                                uplo,
                                n,
                                nrhs,
                                dA.data(),
                                lda,
                                dIpiv.data(),
                                dB.data(),
                                ldb,
                                dWork.data(),
                                bytes_dW,
                                hWork.data(),
                                bytes_hW,
                                dInfo.data(),
                                bc);
    }
}

template <bool CPU,
          bool GPU,
          typename T,
          typename I,
          typename Td,
          typename Id,
          typename Th,
          typename Ih,
          typename INTh>
void sytrs_initData(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const I                   n,
                    const I                   nrhs,
                    Td&                       dA,
                    const I                   lda,
                    Id&                       dIpiv,
                    Td&                       dB,
                    const I                   ldb,
                    const int                 bc,
                    Th&                       hA,
                    Ih&                       hIpiv,
                    INTh&                     hIpiv_cpu,
                    Th&                       hB)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

        for(int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(I i = 0; i < n; i++)
            {
                for(I j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // shuffle rows to test pivoting
            // always the same permutation for debugging purposes
            for(I i = 0; i < n / 2; i++)
            {
                for(I j = 0; j < n; j++)
                {
                    std::swap(hA[b][i + j * lda], hA[b][n - 1 - i + j * lda]);
                }
            }

            // do the LDL^T factorization of matrix A w/ the reference LAPACK routine
            int            sytrf_lwork = lda * n;
            std::vector<T> hWork(sytrf_lwork);
            int            info = 0;
            cpu_sytrf(uplo, n, hA[b], lda, hIpiv_cpu[b], hWork.data(), sytrf_lwork, &info);

            // copy ipiv from int to int64_t
            for(I i = 0; i < n; i++)
                hIpiv[b][i] = hIpiv_cpu[b][i];
        }
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <testAPI_t API,
          typename T,
          typename I,
          typename SIZE,
          typename Td,
          typename Id,
          typename INTd,
          typename Th,
          typename Ih,
          typename INTh>
void sytrs_getError(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const I                   n,
                    const I                   nrhs,
                    Td&                       dA,
                    const I                   lda,
                    Id&                       dIpiv,
                    Td&                       dB,
                    const I                   ldb,
                    void*                     dWork,
                    const SIZE                dlwork,
                    void*                     hWork,
                    const SIZE                hlwork,
                    INTd&                     dInfo,
                    const int                 bc,
                    Th&                       hA,
                    Ih&                       hIpiv,
                    INTh&                     hIpiv_cpu,
                    Th&                       hB,
                    Th&                       hBRes,
                    INTh&                     hInfo,
                    INTh&                     hInfoRes,
                    double*                   max_err)
{
    // input data initialization
    sytrs_initData<true, true, T>(
        handle, uplo, n, nrhs, dA, lda, dIpiv, dB, ldb, bc, hA, hIpiv, hIpiv_cpu, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_sytrs(API,
                                        handle,
                                        uplo,
                                        n,
                                        nrhs,
                                        dA.data(),
                                        lda,
                                        dIpiv.data(),
                                        dB.data(),
                                        ldb,
                                        dWork,
                                        dlwork,
                                        hWork,
                                        hlwork,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
    {
        cpu_sytrs(uplo, n, nrhs, hA[b], lda, hIpiv_cpu[b], hB[b], ldb, hInfo[b]);
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
          typename I,
          typename SIZE,
          typename Td,
          typename Id,
          typename INTd,
          typename Th,
          typename Ih,
          typename INTh>
void sytrs_getPerfData(const hipsolverHandle_t   handle,
                       const hipsolverFillMode_t uplo,
                       const I                   n,
                       const I                   nrhs,
                       Td&                       dA,
                       const I                   lda,
                       Id&                       dIpiv,
                       Td&                       dB,
                       const I                   ldb,
                       void*                     dWork,
                       const SIZE                dlwork,
                       void*                     hWork,
                       const SIZE                hlwork,
                       INTd&                     dInfo,
                       const int                 bc,
                       Th&                       hA,
                       Ih&                       hIpiv,
                       INTh&                     hIpiv_cpu,
                       Th&                       hB,
                       INTh&                     hInfo,
                       double*                   gpu_time_used,
                       double*                   cpu_time_used,
                       const int                 hot_calls,
                       const bool                perf)
{
    if(!perf)
    {
        sytrs_initData<true, false, T>(
            handle, uplo, n, nrhs, dA, lda, dIpiv, dB, ldb, bc, hA, hIpiv, hIpiv_cpu, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cpu_sytrs(uplo, n, nrhs, hA[b], lda, hIpiv_cpu[b], hB[b], ldb, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sytrs_initData<true, false, T>(
        handle, uplo, n, nrhs, dA, lda, dIpiv, dB, ldb, bc, hA, hIpiv, hIpiv_cpu, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sytrs_initData<false, true, T>(
            handle, uplo, n, nrhs, dA, lda, dIpiv, dB, ldb, bc, hA, hIpiv, hIpiv_cpu, hB);

        CHECK_ROCBLAS_ERROR(hipsolver_sytrs(API,
                                            handle,
                                            uplo,
                                            n,
                                            nrhs,
                                            dA.data(),
                                            lda,
                                            dIpiv.data(),
                                            dB.data(),
                                            ldb,
                                            dWork,
                                            dlwork,
                                            hWork,
                                            hlwork,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        sytrs_initData<false, true, T>(
            handle, uplo, n, nrhs, dA, lda, dIpiv, dB, ldb, bc, hA, hIpiv, hIpiv_cpu, hB);

        start = get_time_us_sync(stream);
        hipsolver_sytrs(API,
                        handle,
                        uplo,
                        n,
                        nrhs,
                        dA.data(),
                        lda,
                        dIpiv.data(),
                        dB.data(),
                        ldb,
                        dWork,
                        dlwork,
                        hWork,
                        hlwork,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_sytrs(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   uploC = argus.get<char>("uplo");
    I                      n     = argus.get<int>("n");
    I                      nrhs  = argus.get<int>("nrhs", n);
    I                      lda   = argus.get<int>("lda", n);
    I                      ldb   = argus.get<int>("ldb", n);

    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    // check non-supported values
    if(uplo != HIPSOLVER_FILL_MODE_UPPER && uplo != HIPSOLVER_FILL_MODE_LOWER)
    {
        if(BATCHED)
        {
            // (no batched DnX API for sytrs)
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  nrhs,
                                                  (T*)nullptr,
                                                  lda,
                                                  (I*)nullptr,
                                                  (T*)nullptr,
                                                  ldb,
                                                  (void*)nullptr,
                                                  (SIZE)0,
                                                  (void*)nullptr,
                                                  (SIZE)0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_ENUM);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_B    = size_t(ldb) * nrhs;
    size_t size_P    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // (no batched DnX API for sytrs)
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_sytrs(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  nrhs,
                                                  (T*)nullptr,
                                                  lda,
                                                  (I*)nullptr,
                                                  (T*)nullptr,
                                                  ldb,
                                                  (void*)nullptr,
                                                  (SIZE)0,
                                                  (void*)nullptr,
                                                  (SIZE)0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    SIZE bytes_dW, bytes_hW;
    hipsolver_sytrs_bufferSize(API,
                               handle,
                               uplo,
                               n,
                               nrhs,
                               (T*)nullptr,
                               lda,
                               (I*)nullptr,
                               (T*)nullptr,
                               ldb,
                               &bytes_dW,
                               &bytes_hW);
    SIZE size_dW = (bytes_dW + sizeof(T) - 1) / sizeof(T);
    SIZE size_hW = (bytes_hW + sizeof(T) - 1) / sizeof(T);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_dW);
        return;
    }

    if(BATCHED)
    {
        // (no batched DnX API for sytrs)
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, size_A, bc);
        host_strided_batch_vector<T>     hB(size_B, 1, size_B, bc);
        host_strided_batch_vector<T>     hBRes(size_BRes, 1, size_BRes, bc);
        host_strided_batch_vector<I>     hIpiv(size_P, 1, size_P, bc);
        host_strided_batch_vector<int>   hIpiv_cpu(size_P, 1, size_P, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        host_strided_batch_vector<T>     hWork(size_hW, 1, size_hW, 1);
        device_strided_batch_vector<T>   dA(size_A, 1, size_A, bc);
        device_strided_batch_vector<T>   dB(size_B, 1, size_B, bc);
        device_strided_batch_vector<I>   dIpiv(size_P, 1, size_P, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_dW, 1, size_dW, 1);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytrs_getError<API, T>(handle,
                                   uplo,
                                   n,
                                   nrhs,
                                   dA,
                                   lda,
                                   dIpiv,
                                   dB,
                                   ldb,
                                   dWork.data(),
                                   bytes_dW,
                                   hWork.data(),
                                   bytes_hW,
                                   dInfo,
                                   bc,
                                   hA,
                                   hIpiv,
                                   hIpiv_cpu,
                                   hB,
                                   hBRes,
                                   hInfo,
                                   hInfoRes,
                                   &max_error);

        // collect performance data
        if(argus.timing)
            sytrs_getPerfData<API, T>(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      dA,
                                      lda,
                                      dIpiv,
                                      dB,
                                      ldb,
                                      dWork.data(),
                                      bytes_dW,
                                      hWork.data(),
                                      bytes_hW,
                                      dInfo,
                                      bc,
                                      hA,
                                      hIpiv,
                                      hIpiv_cpu,
                                      hB,
                                      hInfo,
                                      &gpu_time_used,
                                      &cpu_time_used,
                                      hot_calls,
                                      argus.perf);
    }

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
            rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb");
            rocsolver_bench_output(uploC, n, nrhs, lda, ldb);

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
