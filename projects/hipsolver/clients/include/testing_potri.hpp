/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
void potri_checkBadArgs(const hipsolverHandle_t   handle,
                        const hipsolverFillMode_t uplo,
                        const int                 n,
                        T                         dA,
                        const int                 lda,
                        const int                 stA,
                        U                         dWork,
                        const int                 lwork,
                        V                         dinfo,
                        const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_potri(API, nullptr, uplo, n, dA, lda, stA, dWork, lwork, dinfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(
        hipsolver_potri(
            API, handle, hipsolverFillMode_t(-1), n, dA, lda, stA, dWork, lwork, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_potri(API, handle, uplo, n, (T) nullptr, lda, stA, dWork, lwork, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_potri(API, handle, uplo, n, dA, lda, stA, dWork, lwork, (V) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_potri_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolverFillMode_t    uplo = HIPSOLVER_FILL_MODE_UPPER;
    int                    n    = 1;
    int                    lda  = 1;
    int                    stA  = 1;
    int                    bc   = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dinfo.memcheck());

        // int size_W;
        // hipsolver_potri_bufferSize(API, handle, uplo, n, dA.data(), lda, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // potri_checkBadArgs<API>(
        //     handle, uplo, n, dA.data(), lda, stA, dWork.data(), size_W, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        int size_W;
        hipsolver_potri_bufferSize(API, handle, uplo, n, dA.data(), lda, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        potri_checkBadArgs<API>(
            handle, uplo, n, dA.data(), lda, stA, dWork.data(), size_W, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void potri_initData(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const int                 n,
                    Td&                       dA,
                    const int                 lda,
                    const int                 stA,
                    Ud&                       dInfo,
                    const int                 bc,
                    Th&                       hA,
                    Uh&                       hInfo)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to ensure positive definiteness
            for(rocblas_int i = 0; i < n; i++)
                hA[b][i + i * lda] = hA[b][i + i * lda] * conj(hA[b][i + i * lda]) * 400;

            // do the Cholesky factorization of matrix A w/ the reference LAPACK routine
            cpu_potrf(uplo, n, hA[b], lda, hInfo[b]);
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          typename T,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh>
void potri_getError(const hipsolverHandle_t   handle,
                    const hipsolverFillMode_t uplo,
                    const int                 n,
                    Td&                       dA,
                    const int                 lda,
                    const int                 stA,
                    Vd&                       dWork,
                    const int                 lwork,
                    Ud&                       dInfo,
                    const int                 bc,
                    Th&                       hA,
                    Th&                       hARes,
                    Uh&                       hInfo,
                    Uh&                       hInfoRes,
                    double*                   max_err)
{
    // input data initialization
    potri_initData<true, true, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_potri(
        API, handle, uplo, n, dA.data(), lda, stA, dWork.data(), lwork, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cpu_potri(uplo, n, hA[b], lda, hInfo[b]);

    // check info for singularities
    double err = 0;
    *max_err   = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            if(uplo == HIPSOLVER_FILL_MODE_UPPER)
                err = norm_error_upperTr('F', n, n, lda, hA[b], hARes[b]);
            else
                err = norm_error_lowerTr('F', n, n, lda, hA[b], hARes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <testAPI_t API,
          typename T,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh>
void potri_getPerfData(const hipsolverHandle_t   handle,
                       const hipsolverFillMode_t uplo,
                       const int                 n,
                       Td&                       dA,
                       const int                 lda,
                       const int                 stA,
                       Vd&                       dWork,
                       const int                 lwork,
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
        potri_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_potri(uplo, n, hA[b], lda, hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    potri_initData<true, false, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        potri_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_potri(
            API, handle, uplo, n, dA.data(), lda, stA, dWork.data(), lwork, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        potri_initData<false, true, T>(handle, uplo, n, dA, lda, stA, dInfo, bc, hA, hInfo);

        start = get_time_us_sync(stream);
        hipsolver_potri(
            API, handle, uplo, n, dA.data(), lda, stA, dWork.data(), lwork, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_potri(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   uploC = argus.get<char>("uplo");
    int                    n     = argus.get<int>("n");
    int                    lda   = argus.get<int>("lda", n);
    int                    stA   = argus.get<int>("strideA", lda * n);
    int                    bc    = argus.batch_count;

    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    if(uplo != HIPSOLVER_FILL_MODE_UPPER && uplo != HIPSOLVER_FILL_MODE_LOWER)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_potri(API,
            //                                       handle,
            //                                       uplo,
            //                                       n,
            //                                       (T**)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(
                hipsolver_potri(
                    API, handle, uplo, n, (T*)nullptr, lda, stA, (T*)nullptr, 0, (int*)nullptr, bc),
                HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_potri(API,
            //                                       handle,
            //                                       uplo,
            //                                       n,
            //                                       (T**)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(
                hipsolver_potri(
                    API, handle, uplo, n, (T*)nullptr, lda, stA, (T*)nullptr, 0, (int*)nullptr, bc),
                HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_potri_bufferSize(API, handle, uplo, n, (T*)nullptr, lda, &size_W);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>             hA(size_A, 1, bc);
        // host_batch_vector<T>             hARes(size_ARes, 1, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     potri_getError<API, T>(handle,
        //                                uplo,
        //                                n,
        //                                dA,
        //                                lda,
        //                                stA,
        //                                dWork,
        //                                size_W,
        //                                dInfo,
        //                                bc,
        //                                hA,
        //                                hARes,
        //                                hInfo,
        //                                hInfoRes,
        //                                &max_error);

        // // collect performance data
        // if(argus.timing)
        //     potri_getPerfData<API, T>(handle,
        //                                   uplo,
        //                                   n,
        //                                   dA,
        //                                   lda,
        //                                   stA,
        //                                   dWork,
        //                                   size_W,
        //                                   dInfo,
        //                                   bc,
        //                                   hA,
        //                                   hInfo,
        //                                   &gpu_time_used,
        //                                   &cpu_time_used,
        //                                   hot_calls,
        //                                   argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>     hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            potri_getError<API, T>(handle,
                                   uplo,
                                   n,
                                   dA,
                                   lda,
                                   stA,
                                   dWork,
                                   size_W,
                                   dInfo,
                                   bc,
                                   hA,
                                   hARes,
                                   hInfo,
                                   hInfoRes,
                                   &max_error);

        // collect performance data
        if(argus.timing)
            potri_getPerfData<API, T>(handle,
                                      uplo,
                                      n,
                                      dA,
                                      lda,
                                      stA,
                                      dWork,
                                      size_W,
                                      dInfo,
                                      bc,
                                      hA,
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
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "batch_c");
                rocsolver_bench_output(uploC, n, lda, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "lda", "strideA", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stA, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "lda");
                rocsolver_bench_output(uploC, n, lda);
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
}
