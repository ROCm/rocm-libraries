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

template <testAPI_t API, typename I, typename SIZE, typename Td, typename Id, typename INTd>
void getrs_checkBadArgs(const hipsolverHandle_t    handle,
                        const hipsolverDnParams_t  params,
                        const hipsolverOperation_t trans,
                        const I                    m,
                        const I                    nrhs,
                        Td                         dA,
                        const I                    lda,
                        const I                    stA,
                        Id                         dIpiv,
                        const I                    stP,
                        Td                         dB,
                        const I                    ldb,
                        const I                    stB,
                        Td                         dWork,
                        const SIZE                 lwork,
                        INTd                       dInfo,
                        const int                  bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                          nullptr,
                                          params,
                                          trans,
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
                                          dB,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                          handle,
                                          params,
                                          hipsolverOperation_t(-1),
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
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
    if constexpr(!std::is_same<I, int>::value)
        EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                              handle,
                                              (hipsolverDnParams_t) nullptr,
                                              trans,
                                              m,
                                              nrhs,
                                              dA,
                                              lda,
                                              stA,
                                              dIpiv,
                                              stP,
                                              dB,
                                              ldb,
                                              stB,
                                              dWork,
                                              lwork,
                                              dInfo,
                                              bc),
                              HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                          handle,
                                          params,
                                          trans,
                                          m,
                                          nrhs,
                                          (Td) nullptr,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
                                          dB,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                          handle,
                                          params,
                                          trans,
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          (Id) nullptr,
                                          stP,
                                          dB,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                          handle,
                                          params,
                                          trans,
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
                                          (Td) nullptr,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                          handle,
                                          params,
                                          trans,
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
                                          dB,
                                          ldb,
                                          stB,
                                          dWork,
                                          lwork,
                                          (INTd) nullptr,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_getrs_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    I                      m     = 1;
    I                      nrhs  = 1;
    I                      lda   = 1;
    I                      ldb   = 1;
    I                      stA   = 1;
    I                      stP   = 1;
    I                      stB   = 1;
    int                    bc    = 1;
    hipsolverOperation_t   trans = HIPSOLVER_OP_N;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_batch_vector<T>           dB(1, 1, 1);
        // device_strided_batch_vector<I>   dIpiv(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dB.memcheck());
        // CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // SIZE size_W;
        // hipsolver_getrs_bufferSize(API, handle, params, trans, m, nrhs, dA.data(), lda, dIpiv.data(), dB.data(), ldb, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // getrs_checkBadArgs<API>(handle,
        //                             params,
        //                             trans,
        //                             m,
        //                             nrhs,
        //                             dA.data(),
        //                             lda,
        //                             stA,
        //                             dIpiv.data(),
        //                             stP,
        //                             dB.data(),
        //                             ldb,
        //                             stB,
        //                             dInfo.data(),
        //                             bc);
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

        SIZE size_W;
        hipsolver_getrs_bufferSize(API,
                                   handle,
                                   params,
                                   trans,
                                   m,
                                   nrhs,
                                   dA.data(),
                                   lda,
                                   dIpiv.data(),
                                   dB.data(),
                                   ldb,
                                   &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        getrs_checkBadArgs<API>(handle,
                                params,
                                trans,
                                m,
                                nrhs,
                                dA.data(),
                                lda,
                                stA,
                                dIpiv.data(),
                                stP,
                                dB.data(),
                                ldb,
                                stB,
                                dWork.data(),
                                size_W,
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
void getrs_initData(const hipsolverHandle_t    handle,
                    const hipsolverDnParams_t  params,
                    const hipsolverOperation_t trans,
                    const I                    m,
                    const I                    nrhs,
                    Td&                        dA,
                    const I                    lda,
                    const I                    stA,
                    Id&                        dIpiv,
                    const I                    stP,
                    Td&                        dB,
                    const I                    ldb,
                    const I                    stB,
                    const int                  bc,
                    Th&                        hA,
                    Ih&                        hIpiv,
                    INTh&                      hIpiv_cpu,
                    Th&                        hB)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(I i = 0; i < m; i++)
            {
                for(I j = 0; j < m; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }

        // do the LU decomposition of matrix A w/ the reference LAPACK routine
        for(int b = 0; b < bc; ++b)
        {
            int info;
            cpu_getrf(m, m, hA[b], lda, hIpiv_cpu[b], &info);

            for(I i = 0; i < m; i++)
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
void getrs_getError(const hipsolverHandle_t    handle,
                    const hipsolverDnParams_t  params,
                    const hipsolverOperation_t trans,
                    const I                    m,
                    const I                    nrhs,
                    Td&                        dA,
                    const I                    lda,
                    const I                    stA,
                    Id&                        dIpiv,
                    const I                    stP,
                    Td&                        dB,
                    const I                    ldb,
                    const I                    stB,
                    Td&                        dWork,
                    const SIZE                 lwork,
                    INTd&                      dInfo,
                    const int                  bc,
                    Th&                        hA,
                    Ih&                        hIpiv,
                    INTh&                      hIpiv_cpu,
                    Th&                        hB,
                    Th&                        hBRes,
                    INTh&                      hInfo,
                    INTh&                      hInfoRes,
                    double*                    max_err)
{
    // input data initialization
    getrs_initData<true, true, T>(handle,
                                  params,
                                  trans,
                                  m,
                                  nrhs,
                                  dA,
                                  lda,
                                  stA,
                                  dIpiv,
                                  stP,
                                  dB,
                                  ldb,
                                  stB,
                                  bc,
                                  hA,
                                  hIpiv,
                                  hIpiv_cpu,
                                  hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_getrs(API,
                                        handle,
                                        params,
                                        trans,
                                        m,
                                        nrhs,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dIpiv.data(),
                                        stP,
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
        cpu_getrs(trans, m, nrhs, hA[b], lda, hIpiv_cpu[b], hB[b], ldb, hInfo[b]);
    }

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('I', m, nrhs, ldb, hB[b], hBRes[b]);
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
void getrs_getPerfData(const hipsolverHandle_t    handle,
                       const hipsolverDnParams_t  params,
                       const hipsolverOperation_t trans,
                       const I                    m,
                       const I                    nrhs,
                       Td&                        dA,
                       const I                    lda,
                       const I                    stA,
                       Id&                        dIpiv,
                       const I                    stP,
                       Td&                        dB,
                       const I                    ldb,
                       const I                    stB,
                       Td&                        dWork,
                       const SIZE                 lwork,
                       INTd&                      dInfo,
                       const int                  bc,
                       Th&                        hA,
                       Ih&                        hIpiv,
                       INTh&                      hIpiv_cpu,
                       Th&                        hB,
                       INTh&                      hInfo,
                       double*                    gpu_time_used,
                       double*                    cpu_time_used,
                       const int                  hot_calls,
                       const bool                 perf)
{
    if(!perf)
    {
        getrs_initData<true, false, T>(handle,
                                       params,
                                       trans,
                                       m,
                                       nrhs,
                                       dA,
                                       lda,
                                       stA,
                                       dIpiv,
                                       stP,
                                       dB,
                                       ldb,
                                       stB,
                                       bc,
                                       hA,
                                       hIpiv,
                                       hIpiv_cpu,
                                       hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cpu_getrs(trans, m, nrhs, hA[b], lda, hIpiv_cpu[b], hB[b], ldb, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getrs_initData<true, false, T>(handle,
                                   params,
                                   trans,
                                   m,
                                   nrhs,
                                   dA,
                                   lda,
                                   stA,
                                   dIpiv,
                                   stP,
                                   dB,
                                   ldb,
                                   stB,
                                   bc,
                                   hA,
                                   hIpiv,
                                   hIpiv_cpu,
                                   hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getrs_initData<false, true, T>(handle,
                                       params,
                                       trans,
                                       m,
                                       nrhs,
                                       dA,
                                       lda,
                                       stA,
                                       dIpiv,
                                       stP,
                                       dB,
                                       ldb,
                                       stB,
                                       bc,
                                       hA,
                                       hIpiv,
                                       hIpiv_cpu,
                                       hB);

        CHECK_ROCBLAS_ERROR(hipsolver_getrs(API,
                                            handle,
                                            params,
                                            trans,
                                            m,
                                            nrhs,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dIpiv.data(),
                                            stP,
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
        getrs_initData<false, true, T>(handle,
                                       params,
                                       trans,
                                       m,
                                       nrhs,
                                       dA,
                                       lda,
                                       stA,
                                       dIpiv,
                                       stP,
                                       dB,
                                       ldb,
                                       stB,
                                       bc,
                                       hA,
                                       hIpiv,
                                       hIpiv_cpu,
                                       hB);

        start = get_time_us_sync(stream);
        hipsolver_getrs(API,
                        handle,
                        params,
                        trans,
                        m,
                        nrhs,
                        dA.data(),
                        lda,
                        stA,
                        dIpiv.data(),
                        stP,
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

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_getrs(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    char                   transC = argus.get<char>("trans");
    I                      m      = argus.get<int>("n");
    I                      nrhs   = argus.get<int>("nrhs", m);
    I                      lda    = argus.get<int>("lda", m);
    I                      ldb    = argus.get<int>("ldb", m);
    I                      stA    = argus.get<int>("strideA", lda * m);
    I                      stP    = argus.get<int>("strideP", m);
    I                      stB    = argus.get<int>("strideB", ldb * nrhs);

    hipsolverOperation_t trans     = char2hipsolver_operation(transC);
    int                  bc        = argus.batch_count;
    int                  hot_calls = argus.iters;

    I stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * m;
    size_t size_B    = size_t(ldb) * nrhs;
    size_t size_P    = size_t(m);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || nrhs < 0 || lda < m || ldb < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
            //                                       handle,
            //                                       params,
            //                                       trans,
            //                                       m,
            //                                       nrhs,
            //                                       (T* const*)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (I*)nullptr,
            //                                       stP,
            //                                       (T* const*)nullptr,
            //                                       ldb,
            //                                       stB,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_getrs(API,
                                                  handle,
                                                  params,
                                                  trans,
                                                  m,
                                                  nrhs,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (I*)nullptr,
                                                  stP,
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
    SIZE size_W;
    hipsolver_getrs_bufferSize(API,
                               handle,
                               params,
                               trans,
                               m,
                               nrhs,
                               (T*)nullptr,
                               lda,
                               (I*)nullptr,
                               (T*)nullptr,
                               ldb,
                               &size_W);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>             hA(size_A, 1, bc);
        // host_batch_vector<T>             hB(size_B, 1, bc);
        // host_batch_vector<T>             hBRes(size_BRes, 1, bc);
        // host_strided_batch_vector<I>     hIpiv(size_P, 1, stP, bc);
        // host_strided_batch_vector<int>   hIpiv_cpu(size_P, 1, stP, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_batch_vector<T>           dB(size_B, 1, bc);
        // device_strided_batch_vector<I>   dIpiv(size_P, 1, stP, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_B)
        //     CHECK_HIP_ERROR(dB.memcheck());
        // if(size_P)
        //     CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     getrs_getError<API, T>(handle,
        //                                params,
        //                                trans,
        //                                m,
        //                                nrhs,
        //                                dA,
        //                                lda,
        //                                stA,
        //                                dIpiv,
        //                                stP,
        //                                dB,
        //                                ldb,
        //                                stB,
        //                                dWork,
        //                                size_W,
        //                                dInfo,
        //                                bc,
        //                                hA,
        //                                hIpiv,
        //                                hIpiv_cpu,
        //                                hB,
        //                                hBRes,
        //                                hInfo,
        //                                hInfoRes,
        //                                &max_error);

        // // collect performance data
        // if(argus.timing)
        //     getrs_getPerfData<API, T>(handle,
        //                                   params,
        //                                   trans,
        //                                   m,
        //                                   nrhs,
        //                                   dA,
        //                                   lda,
        //                                   stA,
        //                                   dIpiv,
        //                                   stP,
        //                                   dB,
        //                                   ldb,
        //                                   stB,
        //                                   dWork,
        //                                   size_W,
        //                                   dInfo,
        //                                   bc,
        //                                   hA,
        //                                   hIpiv,
        //                                   hIpiv_cpu,
        //                                   hB,
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
        host_strided_batch_vector<T>     hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T>     hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<I>     hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hIpiv_cpu(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T>   dB(size_B, 1, stB, bc);
        device_strided_batch_vector<I>   dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getrs_getError<API, T>(handle,
                                   params,
                                   trans,
                                   m,
                                   nrhs,
                                   dA,
                                   lda,
                                   stA,
                                   dIpiv,
                                   stP,
                                   dB,
                                   ldb,
                                   stB,
                                   dWork,
                                   size_W,
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
            getrs_getPerfData<API, T>(handle,
                                      params,
                                      trans,
                                      m,
                                      nrhs,
                                      dA,
                                      lda,
                                      stA,
                                      dIpiv,
                                      stP,
                                      dB,
                                      ldb,
                                      stB,
                                      dWork,
                                      size_W,
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
            if(BATCHED)
            {
                rocsolver_bench_output("trans", "n", "nrhs", "lda", "ldb", "strideP", "batch_c");
                rocsolver_bench_output(transC, m, nrhs, lda, ldb, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output(
                    "trans", "n", "nrhs", "lda", "ldb", "strideA", "strideP", "strideB", "batch_c");
                rocsolver_bench_output(transC, m, nrhs, lda, ldb, stA, stP, stB, bc);
            }
            else
            {
                rocsolver_bench_output("trans", "n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(transC, m, nrhs, lda, ldb);
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
