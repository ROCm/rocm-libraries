/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipsolver_timer.hpp"

template <testAPI_t API,
          typename I,
          typename SIZE,
          typename Td,
          typename Ud,
          typename TdWork,
          typename ThWork>
void trtri_checkBadArgs(const hipsolverHandle_t   handle,
                        const hipsolverFillMode_t uplo,
                        const hipsolverDiagType_t diag,
                        const I                   n,
                        Td                        dA,
                        const I                   lda,
                        TdWork                    dWork,
                        const SIZE                dlwork,
                        ThWork                    hWork,
                        const SIZE                hlwork,
                        Ud                        dInfo,
                        const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_trtri(
            API, nullptr, uplo, diag, n, dA, lda, dWork, dlwork, hWork, hlwork, dInfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_trtri(API,
                                          handle,
                                          hipsolverFillMode_t(-1),
                                          diag,
                                          n,
                                          dA,
                                          lda,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_trtri(API,
                                          handle,
                                          uplo,
                                          hipsolverDiagType_t(-1),
                                          n,
                                          dA,
                                          lda,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_trtri(
            API, handle, uplo, diag, n, (Td) nullptr, lda, dWork, dlwork, hWork, hlwork, dInfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_trtri(
            API, handle, uplo, diag, n, dA, lda, dWork, dlwork, hWork, hlwork, (Ud) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_trtri(
            API, handle, uplo, diag, 0, (Td) nullptr, lda, dWork, dlwork, hWork, hlwork, dInfo, bc),
        HIPSOLVER_STATUS_SUCCESS);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_trtri_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    I                      n    = 10;
    I                      lda  = 10;
    hipsolverFillMode_t    uplo = HIPSOLVER_FILL_MODE_LOWER;
    hipsolverDiagType_t    diag = HIPSOLVER_DIAG_NON_UNIT;
    int                    bc   = 1;

    if(BATCHED)
    {
        // memory allocations
        // device_batch_vector<T>         dA(1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // SIZE size_dW, size_hW;
        // hipsolver_trtri_bufferSize(API, handle, uplo, diag, n, dA.data(), lda, &size_dW, &size_hW);
        // host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        // device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        // if(size_dW)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // trtri_checkBadArgs<API>(handle, uplo, diag, n, dA.data(), lda,
        //                         dWork.data(), size_dW, hWork.data(), size_hW, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        SIZE size_dW, size_hW;
        hipsolver_trtri_bufferSize(API, handle, uplo, diag, n, dA.data(), lda, &size_dW, &size_hW);
        host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        trtri_checkBadArgs<API>(handle,
                                uplo,
                                diag,
                                n,
                                dA.data(),
                                lda,
                                dWork.data(),
                                size_dW,
                                hWork.data(),
                                size_hW,
                                dInfo.data(),
                                bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void trtri_initData(const int n, Td& dA, const int lda, const int bc, Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / 10.0 + 1;
                    else
                        hA[b][i + j * lda] = (hA[b][i + j * lda] - 4) / 10.0;
                }
            }
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          typename T,
          typename I,
          typename SIZE,
          typename Td,
          typename Ud,
          typename TdWork,
          typename Th,
          typename Uh,
          typename ThWork>
void trtri_getError(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const hipsolverDiagType_t diag,
                    const I                   n,
                    Td&                       dA,
                    const I                   lda,
                    TdWork&                   dWork,
                    const SIZE                lworkOnDevice,
                    ThWork&                   hWork,
                    const SIZE                lworkOnHost,
                    Ud&                       dInfo,
                    const int                 bc,
                    Th&                       hA,
                    Th&                       hARes,
                    Uh&                       hInfo,
                    Uh&                       hInfoRes,
                    double*                   max_err)
{
    // input data initialization
    trtri_initData<true, true, T>(n, dA, lda, bc, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_trtri(API,
                                        handle,
                                        uplo,
                                        diag,
                                        n,
                                        dA.data(),
                                        lda,
                                        dWork.data(),
                                        lworkOnDevice,
                                        hWork.data(),
                                        lworkOnHost,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_trtri(uplo, diag, n, hA[b], lda, hInfo[0]);

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            // for unit diagonal, trtri does not define the output diagonal values,
            // so zero them out in both results before comparing
            // specifically cuSOLVER does not mention if they reference or do not reference the diagonal values
            if(diag == HIPSOLVER_DIAG_UNIT)
            {
                for(I i = 0; i < n; i++)
                {
                    hA[b][i + i * lda]    = T(0);
                    hARes[b][i + i * lda] = T(0);
                }
            }

            if(uplo == HIPSOLVER_FILL_MODE_UPPER)
                *max_err = norm_error_upperTr('F', n, n, lda, hA[b], hARes[b]);
            else
                *max_err = norm_error_lowerTr('F', n, n, lda, hA[b], hARes[b]);
        }
    }

    // check info for singularities
    double err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
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
          typename Ud,
          typename TdWork,
          typename Th,
          typename Uh,
          typename ThWork>
void trtri_getPerfData(const hipsolverHandle_t   handle,
                       const hipsolverFillMode_t uplo,
                       const hipsolverDiagType_t diag,
                       const I                   n,
                       Td&                       dA,
                       const I                   lda,
                       TdWork&                   dWork,
                       const SIZE                lworkOnDevice,
                       ThWork&                   hWork,
                       const SIZE                lworkOnHost,
                       Ud&                       dInfo,
                       const int                 bc,
                       Th&                       hA,
                       Uh&                       hInfo,
                       double*                   gpu_time_used,
                       double*                   cpu_time_used,
                       const int                 hot_calls,
                       const bool                perf)
{
    if(!perf)
    {
        trtri_initData<true, false, T>(n, dA, lda, bc, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_trtri(uplo, diag, n, hA[b], lda, hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trtri_initData<true, false, T>(n, dA, lda, bc, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trtri_initData<false, true, T>(n, dA, lda, bc, hA);

        CHECK_ROCBLAS_ERROR(hipsolver_trtri(API,
                                            handle,
                                            uplo,
                                            diag,
                                            n,
                                            dA.data(),
                                            lda,
                                            dWork.data(),
                                            lworkOnDevice,
                                            hWork.data(),
                                            lworkOnHost,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    hipsolver_timer timer;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        trtri_initData<false, true, T>(n, dA, lda, bc, hA);

        timer.start(stream);
        hipsolver_trtri(API,
                        handle,
                        uplo,
                        diag,
                        n,
                        dA.data(),
                        lda,
                        dWork.data(),
                        lworkOnDevice,
                        hWork.data(),
                        lworkOnHost,
                        dInfo.data(),
                        bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_trtri(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   uploC = argus.get<char>("uplo");
    char                   diagC = argus.get<char>("diag");
    I                      n     = argus.get<rocblas_int>("n");
    I                      lda   = argus.get<rocblas_int>("lda");

    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    hipsolverDiagType_t diag      = char2hipsolver_diag(diagC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_trtri(API,
                                              handle,
                                              uplo,
                                              diag,
                                              n,
                                              (T*)nullptr,
                                              lda,
                                              (T*)nullptr,
                                              0,
                                              (T*)nullptr,
                                              0,
                                              (int*)nullptr,
                                              bc),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T>     hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T>     hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    host_strided_batch_vector<int>   hInfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T>   dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    SIZE size_dW, size_hW;
    hipsolver_trtri_bufferSize(API, handle, uplo, diag, n, dA.data(), lda, &size_dW, &size_hW);
    host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
    device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
    if(size_dW)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_trtri(API,
                                              handle,
                                              uplo,
                                              diag,
                                              n,
                                              dA.data(),
                                              lda,
                                              dWork.data(),
                                              size_dW,
                                              hWork.data(),
                                              size_hW,
                                              dInfo.data(),
                                              bc),
                              HIPSOLVER_STATUS_SUCCESS);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        trtri_getError<API, T>(handle,
                               uplo,
                               diag,
                               n,
                               dA,
                               lda,
                               dWork,
                               size_dW,
                               hWork,
                               size_hW,
                               dInfo,
                               bc,
                               hA,
                               hARes,
                               hInfo,
                               hInfoRes,
                               &max_error);

    // collect performance data
    if(argus.timing)
        trtri_getPerfData<API, T>(handle,
                                  uplo,
                                  diag,
                                  n,
                                  dA,
                                  lda,
                                  dWork,
                                  size_dW,
                                  hWork,
                                  size_hW,
                                  dInfo,
                                  bc,
                                  hA,
                                  hInfo,
                                  &gpu_time_used,
                                  &cpu_time_used,
                                  argus.perf ? 1 : argus.iters,
                                  argus.perf);

    // validate results for rocsolver-test
    // using n * machine_epsilon as tolerance
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
            rocsolver_bench_output("uplo", "diag", "n", "lda");
            rocsolver_bench_output(uploC, diagC, n, lda);

            std::cerr << "\n============================================\n";
            std::cerr << "Results:\n";
            std::cerr << "============================================\n";
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
