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

template <testAPI_t API, typename U>
void gels_checkBadArgs(const hipsolverHandle_t handle,
                       const int               m,
                       const int               n,
                       const int               nrhs,
                       U                       dA,
                       const int               lda,
                       const int               stA,
                       U                       dB,
                       const int               ldb,
                       const int               stB,
                       U                       dX,
                       const int               ldx,
                       const int               stX,
                       U                       dWork,
                       const size_t            lwork,
                       int*                    niters,
                       int*                    info,
                       const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
                                         false,
                                         nullptr,
                                         m,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         info,
                                         bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
                                         false,
                                         handle,
                                         m,
                                         n,
                                         nrhs,
                                         (U) nullptr,
                                         lda,
                                         stA,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         info,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
                                         false,
                                         handle,
                                         m,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         (U) nullptr,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         info,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
                                         false,
                                         handle,
                                         m,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dB,
                                         ldb,
                                         stB,
                                         (U) nullptr,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         info,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
                                         false,
                                         handle,
                                         m,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         nullptr,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gels_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    int                    m    = 1;
    int                    n    = 1;
    int                    nrhs = 1;
    int                    lda  = 1;
    int                    ldb  = 1;
    int                    ldx  = 1;
    int                    stA  = 1;
    int                    stB  = 1;
    int                    stX  = 1;
    int                    bc   = 1;

    if(BATCHED)
    {
        // // memory allocations
        // host_strided_batch_vector<int>   hNIters(1, 1, 1, bc);
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_batch_vector<T>           dB(1, 1, 1);
        // device_batch_vector<T>           dX(1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dB.memcheck());
        // CHECK_HIP_ERROR(dX.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // size_t size_W;
        // hipsolver_gels_bufferSize(
        //     API, handle, m, n, nrhs, dA.data(), lda, dB.data(), ldb, dX.data(), ldx, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // gels_checkBadArgs<API>(handle,
        //                        m,
        //                        n,
        //                        nrhs,
        //                        dA.data(),
        //                        lda,
        //                        stA,
        //                        dB.data(),
        //                        ldb,
        //                        stB,
        //                        dX.data(),
        //                        ldx,
        //                        stX,
        //                        dWork.data(),
        //                        size_W,
        //                        hNIters.data(),
        //                        dInfo.data(),
        //                        bc);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<int>   hNIters(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<T>   dX(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dX.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        size_t size_W;
        hipsolver_gels_bufferSize(
            API, handle, m, n, nrhs, dA.data(), lda, dB.data(), ldb, dX.data(), ldx, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        gels_checkBadArgs<API>(handle,
                               m,
                               n,
                               nrhs,
                               dA.data(),
                               lda,
                               stA,
                               dB.data(),
                               ldb,
                               stB,
                               dX.data(),
                               ldx,
                               stX,
                               dWork.data(),
                               size_W,
                               hNIters.data(),
                               dInfo.data(),
                               bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_initData(const hipsolverHandle_t handle,
                   const int               m,
                   const int               n,
                   const int               nrhs,
                   Td&                     dA,
                   const int               lda,
                   const int               stA,
                   Td&                     dB,
                   const int               ldb,
                   const int               stB,
                   Ud&                     dInfo,
                   const int               bc,
                   Th&                     hA,
                   Th&                     hB,
                   Th&                     hX,
                   Uh&                     hInfo)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

        const int                          max_index = std::max(0, std::min(m, n) - 1);
        std::uniform_int_distribution<int> sample_index(0, max_index);
        std::bernoulli_distribution        coinflip(0.5);

        const int ldx = max(m, n);

        for(int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // populate hX with values from hB
            for(int i = 0; i < m; i++)
                for(int j = 0; j < nrhs; j++)
                    hX[b][i + j * ldx] = hB[b][i + j * ldb];
        }
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <testAPI_t API,
          bool      INPLACE,
          typename T,
          typename Td,
          typename Ud,
          typename Th,
          typename Uh>
void gels_getError(const hipsolverHandle_t handle,
                   const int               m,
                   const int               n,
                   const int               nrhs,
                   Td&                     dA,
                   const int               lda,
                   const int               stA,
                   Td&                     dB,
                   const int               ldb,
                   const int               stB,
                   Td&                     dX,
                   const int               ldx,
                   const int               stX,
                   Td&                     dWork,
                   const size_t            lwork,
                   Ud&                     dInfo,
                   const int               bc,
                   Th&                     hA,
                   Th&                     hB,
                   Th&                     hBRes,
                   Th&                     hX,
                   Th&                     hXRes,
                   Uh&                     hNIters,
                   Uh&                     hInfo,
                   Uh&                     hInfoRes,
                   double*                 max_err)
{
    int            sizeW = max(1, min(m, n) + max(min(m, n), nrhs));
    std::vector<T> hW(sizeW);

    // input data initialization
    gels_initData<true, true, T>(
        handle, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc, hA, hB, hX, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_gels(API,
                                       INPLACE,
                                       handle,
                                       m,
                                       n,
                                       nrhs,
                                       dA.data(),
                                       lda,
                                       stA,
                                       dB.data(),
                                       ldb,
                                       stB,
                                       dX.data(),
                                       ldx,
                                       stX,
                                       dWork.data(),
                                       lwork,
                                       hNIters.data(),
                                       dInfo.data(),
                                       bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
    {
        cpu_gels(
            HIPSOLVER_OP_N, m, n, nrhs, hA[b], lda, hX[b], max(m, n), hW.data(), sizeW, hInfo[b]);
    }

    // error is ||hX - hXRes|| / ||hX||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        if(!INPLACE)
        {
            err      = norm_error('F', m, nrhs, ldb, hB[b], hBRes[b]);
            *max_err = err > *max_err ? err : *max_err;

            if(hInfo[b][0] == 0)
            {
                err      = norm_error('I', n, nrhs, max(m, n), hX[b], hXRes[b], ldx);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
        else
        {
            if(hInfo[b][0] == 0)
            {
                err      = norm_error('I', n, nrhs, max(m, n), hX[b], hBRes[b], ldb);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }

    // also check info for singularities
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
          bool      INPLACE,
          typename T,
          typename Td,
          typename Ud,
          typename Th,
          typename Uh>
void gels_getPerfData(const hipsolverHandle_t handle,
                      const int               m,
                      const int               n,
                      const int               nrhs,
                      Td&                     dA,
                      const int               lda,
                      const int               stA,
                      Td&                     dB,
                      const int               ldb,
                      const int               stB,
                      Td&                     dX,
                      const int               ldx,
                      const int               stX,
                      Td&                     dWork,
                      const size_t            lwork,
                      Ud&                     dInfo,
                      const int               bc,
                      Th&                     hA,
                      Th&                     hB,
                      Th&                     hX,
                      Uh&                     hNIters,
                      Uh&                     hInfo,
                      double*                 gpu_time_used,
                      double*                 cpu_time_used,
                      const int               hot_calls,
                      const bool              perf)
{
    int            sizeW = max(1, min(m, n) + max(min(m, n), nrhs));
    std::vector<T> hW(sizeW);

    if(!perf)
    {
        gels_initData<true, false, T>(
            handle, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc, hA, hB, hX, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cpu_gels(HIPSOLVER_OP_N,
                     m,
                     n,
                     nrhs,
                     hA[b],
                     lda,
                     hX[b],
                     max(m, n),
                     hW.data(),
                     sizeW,
                     hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gels_initData<true, false, T>(
        handle, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc, hA, hB, hX, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gels_initData<false, true, T>(
            handle, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc, hA, hB, hX, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_gels(API,
                                           INPLACE,
                                           handle,
                                           m,
                                           n,
                                           nrhs,
                                           dA.data(),
                                           lda,
                                           stA,
                                           dB.data(),
                                           ldb,
                                           stB,
                                           dX.data(),
                                           ldx,
                                           stX,
                                           dWork.data(),
                                           lwork,
                                           hNIters.data(),
                                           dInfo.data(),
                                           bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        gels_initData<false, true, T>(
            handle, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc, hA, hB, hX, hInfo);

        start = get_time_us_sync(stream);
        hipsolver_gels(API,
                       INPLACE,
                       handle,
                       m,
                       n,
                       nrhs,
                       dA.data(),
                       lda,
                       stA,
                       dB.data(),
                       ldb,
                       stB,
                       dX.data(),
                       ldx,
                       stX,
                       dWork.data(),
                       lwork,
                       hNIters.data(),
                       dInfo.data(),
                       bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API,
          bool      BATCHED,
          bool      STRIDED,
          bool      INPLACE,
          typename T,
          bool COMPLEX = is_complex<T>>
void testing_gels(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    int                    m    = argus.get<int>("m");
    int                    n    = argus.get<int>("n", m);
    int                    nrhs = argus.get<int>("nrhs", n);
    int                    lda  = argus.get<int>("lda", m);
    int                    ldb  = argus.get<int>("ldb", m);
    int                    ldx  = argus.get<int>("ldx", n);
    int                    stA  = argus.get<int>("strideA", lda * n);
    int                    stB  = argus.get<int>("strideB", ldb * nrhs);
    int                    stX  = argus.get<int>("strideX", ldx * nrhs);

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    int stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;
    int stXRes = (argus.unit_check || argus.norm_check) ? stX : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_B    = size_t(ldb) * nrhs;
    size_t size_X    = size_t(ldx) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || nrhs < 0 || lda < m || ldb < m || ldx < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
            //                                      INPLACE,
            //                                      handle,
            //                                      m,
            //                                      n,
            //                                      nrhs,
            //                                      (T* const*)nullptr,
            //                                      lda,
            //                                      stA,
            //                                      (T* const*)nullptr,
            //                                      ldb,
            //                                      stB,
            //                                      (T* const*)nullptr,
            //                                      ldx,
            //                                      stX,
            //                                      (T*)nullptr,
            //                                      0,
            //                                      (int*)nullptr,
            //                                      (int*)nullptr,
            //                                      bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_gels(API,
                                                 INPLACE,
                                                 handle,
                                                 m,
                                                 n,
                                                 nrhs,
                                                 (T*)nullptr,
                                                 lda,
                                                 stA,
                                                 (T*)nullptr,
                                                 ldb,
                                                 stB,
                                                 (T*)nullptr,
                                                 ldx,
                                                 stX,
                                                 (T*)nullptr,
                                                 0,
                                                 (int*)nullptr,
                                                 (int*)nullptr,
                                                 bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    size_t size_W;
    hipsolver_gels_bufferSize(
        API, handle, m, n, nrhs, (T*)nullptr, lda, (T*)nullptr, ldb, (T*)nullptr, ldx, &size_W);

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
        // host_batch_vector<T>             hX(max(m, n) * nrhs, 1, bc);
        // host_batch_vector<T>             hXRes(size_XRes, 1, bc);
        // host_strided_batch_vector<int>   hNIters(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_batch_vector<T>           dB(size_B, 1, bc);
        // device_batch_vector<T>           dX(size_X, 1, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_B)
        //     CHECK_HIP_ERROR(dB.memcheck());
        // if(size_X)
        //     CHECK_HIP_ERROR(dX.memcheck());
        // if(bc)
        //     CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     gels_getError<API, INPLACE, T>(handle,
        //                                    m,
        //                                    n,
        //                                    nrhs,
        //                                    dA,
        //                                    lda,
        //                                    stA,
        //                                    dB,
        //                                    ldb,
        //                                    stB,
        //                                    dX,
        //                                    ldx,
        //                                    stX,
        //                                    dWork,
        //                                    size_W,
        //                                    dInfo,
        //                                    bc,
        //                                    hA,
        //                                    hB,
        //                                    hBRes,
        //                                    hX,
        //                                    hXRes,
        //                                    hNIters,
        //                                    hInfo,
        //                                    hInfoRes,
        //                                    &max_error);

        // // collect performance data
        // if(argus.timing)
        //     gels_getPerfData<API, INPLACE, T>(handle,
        //                                       m,
        //                                       n,
        //                                       nrhs,
        //                                       dA,
        //                                       lda,
        //                                       stA,
        //                                       dB,
        //                                       ldb,
        //                                       stB,
        //                                       dX,
        //                                       ldx,
        //                                       stX,
        //                                       dWork,
        //                                       size_W,
        //                                       dInfo,
        //                                       bc,
        //                                       hA,
        //                                       hB,
        //                                       hX,
        //                                       hNIters,
        //                                       hInfo,
        //                                       &gpu_time_used,
        //                                       &cpu_time_used,
        //                                       hot_calls,
        //                                       argus.perf);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>     hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T>     hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<T>     hX(max(m, n) * nrhs, 1, max(m, n) * nrhs, bc);
        host_strided_batch_vector<T>     hXRes(size_XRes, 1, stXRes, bc);
        host_strided_batch_vector<int>   hNIters(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T>   dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T>   dX(size_X, 1, stX, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());
        if(bc)
            CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            gels_getError<API, INPLACE, T>(handle,
                                           m,
                                           n,
                                           nrhs,
                                           dA,
                                           lda,
                                           stA,
                                           dB,
                                           ldb,
                                           stB,
                                           dX,
                                           ldx,
                                           stX,
                                           dWork,
                                           size_W,
                                           dInfo,
                                           bc,
                                           hA,
                                           hB,
                                           hBRes,
                                           hX,
                                           hXRes,
                                           hNIters,
                                           hInfo,
                                           hInfoRes,
                                           &max_error);

        // collect performance data
        if(argus.timing)
            gels_getPerfData<API, INPLACE, T>(handle,
                                              m,
                                              n,
                                              nrhs,
                                              dA,
                                              lda,
                                              stA,
                                              dB,
                                              ldb,
                                              stB,
                                              dX,
                                              ldx,
                                              stX,
                                              dWork,
                                              size_W,
                                              dInfo,
                                              bc,
                                              hA,
                                              hB,
                                              hX,
                                              hNIters,
                                              hInfo,
                                              &gpu_time_used,
                                              &cpu_time_used,
                                              hot_calls,
                                              argus.perf);
    }
    // validate results for rocsolver-test
    // using max(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, max(m, n));

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
                rocsolver_bench_output("m", "n", "nrhs", "lda", "ldb", "ldx", "batch_c");
                rocsolver_bench_output(m, n, nrhs, lda, ldb, ldx, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m",
                                       "n",
                                       "nrhs",
                                       "lda",
                                       "ldb",
                                       "ldx",
                                       "strideA",
                                       "strideB",
                                       "strideX",
                                       "batch_c");
                rocsolver_bench_output(m, n, nrhs, lda, ldb, ldx, stA, stB, stX, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "nrhs", "lda", "ldb", "ldx");
                rocsolver_bench_output(m, n, nrhs, lda, ldb, ldx);
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
