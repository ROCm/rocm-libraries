/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <testAPI_t API, bool STRIDED, typename T, typename S, typename U>
void syev_heev_checkBadArgs(const hipsolverHandle_t   handle,
                            const hipsolverDnParams_t params,
                            const hipsolverEigMode_t  jobz,
                            const hipsolverFillMode_t uplo,
                            const int64_t             n,
                            T                         dA,
                            const int64_t             lda,
                            S                         dW,
                            T                         dWork,
                            const size_t              lworkOnDevice,
                            T                         hWork,
                            const size_t              lworkOnHost,
                            U                         dinfo,
                            const int64_t             batchSize)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              nullptr,
                                              params,
                                              jobz,
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              dW,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              dinfo,
                                              batchSize),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              handle,
                                              params,
                                              hipsolverEigMode_t(-1),
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              dW,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              dinfo,
                                              batchSize),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              handle,
                                              params,
                                              jobz,
                                              hipsolverFillMode_t(-1),
                                              n,
                                              dA,
                                              lda,
                                              dW,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              dinfo,
                                              batchSize),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              handle,
                                              (hipsolverDnParams_t) nullptr,
                                              jobz,
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              dW,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              dinfo,
                                              batchSize),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              handle,
                                              params,
                                              jobz,
                                              uplo,
                                              n,
                                              (T) nullptr,
                                              lda,
                                              dW,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              dinfo,
                                              batchSize),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              handle,
                                              params,
                                              jobz,
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              (S) nullptr,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              dinfo,
                                              batchSize),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                              STRIDED,
                                              handle,
                                              params,
                                              jobz,
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              dW,
                                              dWork,
                                              lworkOnDevice,
                                              hWork,
                                              lworkOnHost,
                                              (U) nullptr,
                                              batchSize),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_syev_heev_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    hipsolverEigMode_t     jobz      = HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t    uplo      = HIPSOLVER_FILL_MODE_LOWER;
    int64_t                n         = 1;
    int64_t                lda       = 1;
    int64_t                batchSize = 1;

    // memory allocations
    device_strided_batch_vector<T>   dA(1, 1, 1, 1);
    device_strided_batch_vector<S>   dW(1, 1, 1, 1);
    device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    SIZE size_dW, size_hW;
    hipsolver_syev_heev_bufferSize(API,
                                   STRIDED,
                                   handle,
                                   params,
                                   jobz,
                                   uplo,
                                   n,
                                   dA.data(),
                                   lda,
                                   dW.data(),
                                   &size_dW,
                                   &size_hW,
                                   batchSize);
    host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
    device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
    if(size_dW)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    syev_heev_checkBadArgs<API, STRIDED>(handle,
                                         params,
                                         jobz,
                                         uplo,
                                         n,
                                         dA.data(),
                                         lda,
                                         dW.data(),
                                         dWork.data(),
                                         size_dW,
                                         hWork.data(),
                                         size_hW,
                                         dinfo.data(),
                                         batchSize);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syev_heev_initData(const hipsolverHandle_t  handle,
                        const hipsolverEigMode_t evect,
                        const int                n,
                        Td&                      dA,
                        const int                lda,
                        const int                bc,
                        Th&                      hA,
                        std::vector<T>&          A,
                        bool                     test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test && evect == HIPSOLVER_EIG_MODE_VECTOR)
            {
                for(int i = 0; i < n; i++)
                {
                    for(int j = 0; j < n; j++)
                        A[b * lda * n + i + j * lda] = hA[b][i + j * lda];
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          bool      STRIDED,
          typename T,
          typename I,
          typename SIZE,
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Ih>
void syev_heev_getError(const hipsolverHandle_t   handle,
                        const hipsolverDnParams_t params,
                        const hipsolverEigMode_t  evect,
                        const hipsolverFillMode_t uplo,
                        const I                   n,
                        Td&                       dA,
                        const I                   lda,
                        Sd&                       dW,
                        Td&                       dWork,
                        const SIZE                lworkOnDevice,
                        Th&                       hWork,
                        const SIZE                lworkOnHost,
                        Id&                       dinfo,
                        const int                 bc,
                        Th&                       hA,
                        Th&                       hAres,
                        Sh&                       hW,
                        Sh&                       hWres,
                        Ih&                       hinfo,
                        Ih&                       hinfoRes,
                        double*                   max_err)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));

    I lrwork, ltwork;
    if(!COMPLEX)
    {
        ltwork = std::max(I(1), 3 * n - 1);
        lrwork = 0;
    }
    else
    {
        lrwork = std::max(I(1), 3 * n - 2);
        ltwork = std::max(I(1), 2 * n - 1);
    }

    std::vector<T> work(ltwork);
    std::vector<S> hE(lrwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    syev_heev_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_syev_heev(API,
                                            STRIDED,
                                            handle,
                                            params,
                                            evect,
                                            uplo,
                                            n,
                                            dA.data(),
                                            lda,
                                            dW.data(),
                                            dWork.data(),
                                            lworkOnDevice,
                                            hWork.data(),
                                            lworkOnHost,
                                            dinfo.data(),
                                            bc));

    CHECK_HIP_ERROR(hWres.transfer_from(dW));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(evect == HIPSOLVER_EIG_MODE_VECTOR)
        CHECK_HIP_ERROR(hAres.transfer_from(dA));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cpu_syev_heev(evect, uplo, n, hA[b], lda, hW[b], work.data(), ltwork, hE.data(), hinfo[b]);

    // Check info for non-convergence
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
        {
            *max_err += 1;
        }
    }

    double err = 0;

    for(int b = 0; b < bc; ++b)
    {
        if(evect != HIPSOLVER_EIG_MODE_VECTOR)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hinfo[b][0] == 0)
            {
                err = norm_error('F', 1, n, 1, hW[b], hWres[b]);
            }
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hinfo[b][0] == 0)
            {
                // multiply A with each of the n eigenvectors and divide by corresponding
                // eigenvalues
                T alpha;
                T beta = 0;
                for(int j = 0; j < n; j++)
                {
                    alpha = T(1) / hWres[b][j];
                    cpu_symv_hemv(uplo,
                                  n,
                                  alpha,
                                  A.data() + b * lda * n,
                                  lda,
                                  hAres[b] + j * lda,
                                  1,
                                  beta,
                                  hA[b] + j * lda,
                                  1);
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err      = norm_error('F', n, n, lda, hA[b], hAres[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <testAPI_t API,
          bool      STRIDED,
          typename T,
          typename I,
          typename SIZE,
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Ih>
void syev_heev_getPerfData(const hipsolverHandle_t   handle,
                           const hipsolverDnParams_t params,
                           const hipsolverEigMode_t  evect,
                           const hipsolverFillMode_t uplo,
                           const I                   n,
                           Td&                       dA,
                           const I                   lda,
                           Sd&                       dW,
                           Td&                       dWork,
                           const SIZE                lworkOnDevice,
                           Th&                       hWork,
                           const SIZE                lworkOnHost,
                           Id&                       dinfo,
                           const int                 bc,
                           Th&                       hA,
                           Sh&                       hW,
                           Ih&                       hinfo,
                           double*                   gpu_time_used,
                           double*                   cpu_time_used,
                           const int                 hot_calls,
                           const bool                perf)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));

    I lrwork, ltwork;
    if(!COMPLEX)
    {
        ltwork = std::max(I(1), 3 * n - 1);
        lrwork = 0;
    }
    else
    {
        lrwork = std::max(I(1), 3 * n - 2);
        ltwork = std::max(I(1), 2 * n - 1);
    }

    std::vector<T> work(ltwork);
    std::vector<S> hE(lrwork);
    std::vector<T> A;

    if(!perf)
    {
        syev_heev_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_syev_heev(
                evect, uplo, n, hA[b], lda, hW[b], work.data(), ltwork, hE.data(), hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syev_heev_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syev_heev_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(hipsolver_syev_heev(API,
                                                STRIDED,
                                                handle,
                                                params,
                                                evect,
                                                uplo,
                                                n,
                                                dA.data(),
                                                lda,
                                                dW.data(),
                                                dWork.data(),
                                                lworkOnDevice,
                                                hWork.data(),
                                                lworkOnHost,
                                                dinfo.data(),
                                                bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        syev_heev_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        hipsolver_syev_heev(API,
                            STRIDED,
                            handle,
                            params,
                            evect,
                            uplo,
                            n,
                            dA.data(),
                            lda,
                            dW.data(),
                            dWork.data(),
                            lworkOnDevice,
                            hWork.data(),
                            lworkOnHost,
                            dinfo.data(),
                            bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API,
          bool      BATCHED,
          bool      STRIDED,
          typename T,
          typename I    = int64_t,
          typename SIZE = size_t>
void testing_syev_heev(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    char                   evectC = argus.get<char>("jobz");
    char                   uploC  = argus.get<char>("uplo");
    I                      n      = argus.get<int>("n");
    I                      lda    = argus.get<int>("lda", n);
    I                      stA    = lda * n;
    I                      stW    = n;

    hipsolverEigMode_t  evect     = char2hipsolver_evect(evectC);
    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_W    = n;
    size_t size_Ares = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_Wres = (argus.unit_check || argus.norm_check) ? size_W : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_syev_heev(API,
                                                  STRIDED,
                                                  handle,
                                                  params,
                                                  evect,
                                                  uplo,
                                                  n,
                                                  (T*)nullptr,
                                                  lda,
                                                  (S*)nullptr,
                                                  (T*)nullptr,
                                                  (SIZE)0,
                                                  (T*)nullptr,
                                                  (SIZE)0,
                                                  (int*)nullptr,
                                                  bc),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    SIZE size_dW, size_hW;
    hipsolver_syev_heev_bufferSize(API,
                                   STRIDED,
                                   handle,
                                   params,
                                   evect,
                                   uplo,
                                   n,
                                   (T*)nullptr,
                                   lda,
                                   (S*)nullptr,
                                   &size_dW,
                                   &size_hW,
                                   bc);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_dW);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S>   hW(size_W, 1, stW, bc);
    host_strided_batch_vector<int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S>   hWres(size_Wres, 1, stW, bc);
    // device
    device_strided_batch_vector<S>   dW(size_W, 1, stW, bc);
    device_strided_batch_vector<int> dinfo(1, 1, 1, bc);
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // memory allocations
    host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
    host_strided_batch_vector<T>   hAres(size_Ares, 1, stA, bc);
    host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1); // size_hW accounts for bc
    device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
    device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1); // size_dW accounts for bc
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_dW)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        syev_heev_getError<API, STRIDED, T>(handle,
                                            params,
                                            evect,
                                            uplo,
                                            n,
                                            dA,
                                            lda,
                                            dW,
                                            dWork,
                                            size_dW,
                                            hWork,
                                            size_hW,
                                            dinfo,
                                            bc,
                                            hA,
                                            hAres,
                                            hW,
                                            hWres,
                                            hinfo,
                                            hinfoRes,
                                            &max_error);
    }

    // collect performance data
    if(argus.timing)
    {
        syev_heev_getPerfData<API, STRIDED, T>(handle,
                                               params,
                                               evect,
                                               uplo,
                                               n,
                                               dA,
                                               lda,
                                               dW,
                                               dWork,
                                               size_dW,
                                               hWork,
                                               size_hW,
                                               dinfo,
                                               bc,
                                               hA,
                                               hW,
                                               hinfo,
                                               &gpu_time_used,
                                               &cpu_time_used,
                                               hot_calls,
                                               argus.perf);
    }

    // validate results for rocsolver-test
    // using 4 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 4 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("jobz", "uplo", "n", "lda", "batch_c");
            rocsolver_bench_output(evectC, uploC, n, lda, bc);
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
