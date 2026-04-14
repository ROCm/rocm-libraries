/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
void syevj_heevj_checkBadArgs(const hipsolverHandle_t    handle,
                              const hipsolverEigMode_t   evect,
                              const hipsolverFillMode_t  uplo,
                              const int                  n,
                              T                          dA,
                              const int                  lda,
                              const int                  stA,
                              S                          dD,
                              const int                  stD,
                              T                          dWork,
                              const int                  lwork,
                              U                          dinfo,
                              const hipsolverSyevjInfo_t params,
                              const int                  bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                STRIDED,
                                                nullptr,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dinfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                STRIDED,
                                                handle,
                                                hipsolverEigMode_t(-1),
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dinfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                STRIDED,
                                                handle,
                                                evect,
                                                hipsolverFillMode_t(-1),
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dinfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                STRIDED,
                                                handle,
                                                evect,
                                                uplo,
                                                n,
                                                (T) nullptr,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dinfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                STRIDED,
                                                handle,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                (S) nullptr,
                                                stD,
                                                dWork,
                                                lwork,
                                                dinfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                STRIDED,
                                                handle,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                (U) nullptr,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_syevj_heevj_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle     handle;
    hipsolver_local_syevj_info params;
    hipsolverEigMode_t         evect = HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t        uplo  = HIPSOLVER_FILL_MODE_LOWER;
    int                        n     = 1;
    int                        lda   = 1;
    int                        stA   = 1;
    int                        stD   = 1;
    int                        bc    = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        // device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dD.memcheck());
        // CHECK_HIP_ERROR(dinfo.memcheck());

        // int size_W;
        // hipsolver_syevj_heevj_bufferSize(
        //     API, handle, evect, uplo, n, dA.data(), lda, dD.data(), &size_W, params, bc);
        // size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;
        // device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // syevj_heevj_checkBadArgs<API, STRIDED>(handle,
        //                                        evect,
        //                                        uplo,
        //                                        n,
        //                                        dA.data(),
        //                                        lda,
        //                                        stA,
        //                                        dD.data(),
        //                                        stD,
        //                                        dWork.data(),
        //                                        size_W,
        //                                        dinfo.data(),
        //                                        params,
        //                                        bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        int size_W;
        hipsolver_syevj_heevj_bufferSize(
            API, STRIDED, handle, evect, uplo, n, dA.data(), lda, dD.data(), &size_W, params, bc);
        size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr
                             ? size_W
                             : sizeof(T) * size_W;
        device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        syevj_heevj_checkBadArgs<API, STRIDED>(handle,
                                               evect,
                                               uplo,
                                               n,
                                               dA.data(),
                                               lda,
                                               stA,
                                               dD.data(),
                                               stD,
                                               dWork.data(),
                                               size_W,
                                               dinfo.data(),
                                               params,
                                               bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevj_heevj_initData(const hipsolverHandle_t  handle,
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
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test)
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
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Uh,
          typename Ih>
void syevj_heevj_getError(const hipsolverHandle_t   handle,
                          const hipsolverEigMode_t  evect,
                          const hipsolverFillMode_t uplo,
                          const int                 n,
                          Td&                       dA,
                          const int                 lda,
                          const int                 stA,
                          Sd&                       dD,
                          const int                 stD,
                          Td&                       dWork,
                          const int                 lwork,
                          Id&                       dinfo,
                          hipsolverSyevjInfo_t      params,
                          const double              abstol,
                          const int                 max_sweeps,
                          const int                 sort_eig,
                          const int                 bc,
                          Th&                       hA,
                          Th&                       hAres,
                          Sh&                       hD,
                          Sh&                       hDres,
                          Ih&                       hinfo,
                          Ih&                       hinfoRes,
                          Uh&                       hResidualRes,
                          Ih&                       hSweepsRes,
                          double*                   max_err)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));
    S atol                 = (abstol <= 0) ? get_epsilon<T>() : abstol;

    int sizeE, ltwork;
    if(!COMPLEX)
    {
        sizeE  = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 2 * n + 1 : 1 + 6 * n + 2 * n * n);
        ltwork = 0;
    }
    else
    {
        sizeE  = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n : 1 + 5 * n + 2 * n * n);
        ltwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n + 1 : 2 * n + n * n);
    }
    int liwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : 3 + 5 * n);

    std::vector<T>   work(ltwork);
    std::vector<S>   hE(sizeE);
    std::vector<int> iwork(liwork);
    std::vector<T>   A(lda * n * bc);

    // input data initialization
    syevj_heevj_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_syevj_heevj(API,
                                              STRIDED,
                                              handle,
                                              evect,
                                              uplo,
                                              n,
                                              dA.data(),
                                              lda,
                                              stA,
                                              dD.data(),
                                              stD,
                                              dWork.data(),
                                              lwork,
                                              dinfo.data(),
                                              params,
                                              bc));

    CHECK_HIP_ERROR(hDres.transfer_from(dD));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(evect == HIPSOLVER_EIG_MODE_VECTOR)
        CHECK_HIP_ERROR(hAres.transfer_from(dA));

    hipsolverXsyevjGetResidual(handle, params, hResidualRes.data());
    hipsolverXsyevjGetSweeps(handle, params, hSweepsRes.data());

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cpu_syevd_heevd(evect,
                        uplo,
                        n,
                        hA[b],
                        lda,
                        hD[b],
                        work.data(),
                        ltwork,
                        hE.data(),
                        sizeE,
                        iwork.data(),
                        liwork,
                        hinfo[b]);

    // Check info for non-convergence
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    if(!STRIDED)
    {
        // Also check validity of residual
        for(rocblas_int b = 0; b < bc; ++b)
        {
            EXPECT_GE(hResidualRes[b][0], 0) << "where b = " << b;
            if(hResidualRes[b][0] < 0)
                *max_err += 1;
            else
            {
                S threshold = snorm('F', n, n, A.data() + b * lda * n, lda) * atol;
                EXPECT_LE(hResidualRes[b][0], threshold) << "where b = " << b;
                if(hResidualRes[b][0] > threshold)
                    *max_err += 1;
            }
        }

        // Also check validity of sweeps
        for(rocblas_int b = 0; b < bc; ++b)
        {
            EXPECT_GE(hSweepsRes[b][0], 0) << "where b = " << b;
            EXPECT_LE(hSweepsRes[b][0], max_sweeps) << "where b = " << b;
            if(hSweepsRes[b][0] < 0 || hSweepsRes[b][0] > max_sweeps)
                *max_err += 1;
        }
    }

    double err = 0;

    for(int b = 0; b < bc; ++b)
    {
        if(evect != HIPSOLVER_EIG_MODE_VECTOR)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hD - hDRes|| / ||hD||
            // using frobenius norm
            if(hinfo[b][0] == 0 && sort_eig)
                err = norm_error('F', 1, n, 1, hD[b], hDres[b]);
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
                    alpha = T(1) / hDres[b][j];
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
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Ih>
void syevj_heevj_getPerfData(const hipsolverHandle_t   handle,
                             const hipsolverEigMode_t  evect,
                             const hipsolverFillMode_t uplo,
                             const int                 n,
                             Td&                       dA,
                             const int                 lda,
                             const int                 stA,
                             Sd&                       dD,
                             const int                 stD,
                             Td&                       dWork,
                             const int                 lwork,
                             Id&                       dinfo,
                             hipsolverSyevjInfo_t      params,
                             const int                 bc,
                             Th&                       hA,
                             Sh&                       hD,
                             Ih&                       hinfo,
                             double*                   gpu_time_used,
                             double*                   cpu_time_used,
                             const int                 hot_calls,
                             const bool                perf)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));

    int sizeE, ltwork;
    if(!COMPLEX)
    {
        sizeE  = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 2 * n + 1 : 1 + 6 * n + 2 * n * n);
        ltwork = 0;
    }
    else
    {
        sizeE  = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n : 1 + 5 * n + 2 * n * n);
        ltwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n + 1 : 2 * n + n * n);
    }
    int liwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : 3 + 5 * n);

    std::vector<T>   work(ltwork);
    std::vector<S>   hE(sizeE);
    std::vector<int> iwork(liwork);
    std::vector<T>   A;

    if(!perf)
    {
        syevj_heevj_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_syevd_heevd(evect,
                            uplo,
                            n,
                            hA[b],
                            lda,
                            hD[b],
                            work.data(),
                            ltwork,
                            hE.data(),
                            sizeE,
                            iwork.data(),
                            liwork,
                            hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syevj_heevj_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syevj_heevj_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(hipsolver_syevj_heevj(API,
                                                  STRIDED,
                                                  handle,
                                                  evect,
                                                  uplo,
                                                  n,
                                                  dA.data(),
                                                  lda,
                                                  stA,
                                                  dD.data(),
                                                  stD,
                                                  dWork.data(),
                                                  lwork,
                                                  dinfo.data(),
                                                  params,
                                                  bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        syevj_heevj_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        hipsolver_syevj_heevj(API,
                              STRIDED,
                              handle,
                              evect,
                              uplo,
                              n,
                              dA.data(),
                              lda,
                              stA,
                              dD.data(),
                              stD,
                              dWork.data(),
                              lwork,
                              dinfo.data(),
                              params,
                              bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_syevj_heevj(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle     handle;
    hipsolver_local_syevj_info params;
    char                       evectC = argus.get<char>("jobz");
    char                       uploC  = argus.get<char>("uplo");
    int                        n      = argus.get<int>("n");
    int                        lda    = argus.get<int>("lda", n);
    int                        stA    = lda * n;
    int                        stD    = n;

    double      abstol     = argus.get<double>("tolerance", 2 * get_epsilon<T>());
    rocblas_int max_sweeps = argus.get<int>("max_sweeps", 100);
    rocblas_int sort_eig   = argus.get<int>("sort_eig", 1);
    hipsolverXsyevjSetTolerance(params, abstol);
    hipsolverXsyevjSetMaxSweeps(params, max_sweeps);
    hipsolverXsyevjSetSortEig(params, sort_eig);

    hipsolverEigMode_t  evect     = char2hipsolver_evect(evectC);
    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_D    = n;
    size_t size_Ares = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_Dres = (argus.unit_check || argus.norm_check) ? size_D : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
            //                                             STRIDED,
            //                                             handle,
            //                                             evect,
            //                                             uplo,
            //                                             n,
            //                                             (T* const*)nullptr,
            //                                             lda,
            //                                             stA,
            //                                             (S*)nullptr,
            //                                             stD,
            //                                             (T*)nullptr,
            //                                             0,
            //                                             (int*)nullptr,
            //                                             params,
            //                                             bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_syevj_heevj(API,
                                                        STRIDED,
                                                        handle,
                                                        evect,
                                                        uplo,
                                                        n,
                                                        (T*)nullptr,
                                                        lda,
                                                        stA,
                                                        (S*)nullptr,
                                                        stD,
                                                        (T*)nullptr,
                                                        0,
                                                        (int*)nullptr,
                                                        params,
                                                        bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_syevj_heevj_bufferSize(
        API, STRIDED, handle, evect, uplo, n, (T*)nullptr, lda, (S*)nullptr, &size_W, params, bc);
    size_t bytes_W
        = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_W);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<double> hResidualRes(1, 1, 1, bc);
    host_strided_batch_vector<int>    hSweepsRes(1, 1, 1, bc);
    host_strided_batch_vector<S>      hD(size_D, 1, stD, bc);
    host_strided_batch_vector<int>    hinfo(1, 1, 1, bc);
    host_strided_batch_vector<int>    hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S>      hDres(size_Dres, 1, stD, bc);
    // device
    device_strided_batch_vector<S>   dD(size_D, 1, stD, bc);
    device_strided_batch_vector<int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dWork(bytes_W, 1, bytes_W, 1); // bytes_W accounts for bc
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>           hA(size_A, 1, bc);
        // host_batch_vector<T>           hAres(size_Ares, 1, bc);
        // device_batch_vector<T>         dA(size_A, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        // {
        //     syevj_heevj_getError<API, STRIDED, T>(handle,
        //                                          evect,
        //                                          uplo,
        //                                          n,
        //                                          dA,
        //                                          lda,
        //                                          stA,
        //                                          dD,
        //                                          stD,
        //                                          dWork,
        //                                          size_W,
        //                                          dinfo,
        //                                          params,
        //                                          abstol,
        //                                          max_sweeps,
        //                                          sort_eig,
        //                                          bc,
        //                                          hA,
        //                                          hAres,
        //                                          hD,
        //                                          hDres,
        //                                          hinfo,
        //                                          hinfoRes,
        //                                          hResidualRes,
        //                                          hSweepsRes,
        //                                          &max_error);
        // }

        // // collect performance data
        // if(argus.timing)
        // {
        //     syevj_heevj_getPerfData<API, STRIDED, T>(handle,
        //                                              evect,
        //                                              uplo,
        //                                              n,
        //                                              dA,
        //                                              lda,
        //                                              stA,
        //                                              dD,
        //                                              stD,
        //                                              dWork,
        //                                              size_W,
        //                                              dinfo,
        //                                              params,
        //                                              bc,
        //                                              hA,
        //                                              hD,
        //                                              hinfo,
        //                                              &gpu_time_used,
        //                                              &cpu_time_used,
        //                                              hot_calls,
        //                                              argus.perf);
        // }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>   hAres(size_Ares, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevj_heevj_getError<API, STRIDED, T>(handle,
                                                  evect,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dD,
                                                  stD,
                                                  dWork,
                                                  size_W,
                                                  dinfo,
                                                  params,
                                                  abstol,
                                                  max_sweeps,
                                                  sort_eig,
                                                  bc,
                                                  hA,
                                                  hAres,
                                                  hD,
                                                  hDres,
                                                  hinfo,
                                                  hinfoRes,
                                                  hResidualRes,
                                                  hSweepsRes,
                                                  &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevj_heevj_getPerfData<API, STRIDED, T>(handle,
                                                     evect,
                                                     uplo,
                                                     n,
                                                     dA,
                                                     lda,
                                                     stA,
                                                     dD,
                                                     stD,
                                                     dWork,
                                                     size_W,
                                                     dinfo,
                                                     params,
                                                     bc,
                                                     hA,
                                                     hD,
                                                     hinfo,
                                                     &gpu_time_used,
                                                     &cpu_time_used,
                                                     hot_calls,
                                                     argus.perf);
        }
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
            if(BATCHED)
            {
                rocsolver_bench_output("jobz",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "strideD",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig",
                                       "batch_c");
                rocsolver_bench_output(
                    evectC, uploC, n, lda, stD, abstol, max_sweeps, sort_eig, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("jobz",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "strideA",
                                       "strideD",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig",
                                       "batch_c");
                rocsolver_bench_output(
                    evectC, uploC, n, lda, stA, stD, abstol, max_sweeps, sort_eig, bc);
            }
            else
            {
                rocsolver_bench_output(
                    "jobz", "uplo", "n", "lda", "tolerance", "max_sweeps", "sort_eig");
                rocsolver_bench_output(evectC, uploC, n, lda, abstol, max_sweeps, sort_eig);
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
