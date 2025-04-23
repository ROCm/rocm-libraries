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

template <testAPI_t API, typename T, typename U>
void gesv_checkBadArgs(const hipsolverHandle_t handle,
                       const int               n,
                       const int               nrhs,
                       T                       dA,
                       const int               lda,
                       const int               stA,
                       U                       dIpiv,
                       const int               stP,
                       T                       dB,
                       const int               ldb,
                       const int               stB,
                       T                       dX,
                       const int               ldx,
                       const int               stX,
                       T                       dWork,
                       const size_t            lwork,
                       U                       niters,
                       U                       dInfo,
                       const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                         false,
                                         nullptr,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dIpiv,
                                         stP,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         dInfo,
                                         bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                         false,
                                         handle,
                                         n,
                                         nrhs,
                                         (T) nullptr,
                                         lda,
                                         stA,
                                         dIpiv,
                                         stP,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         dInfo,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                         false,
                                         handle,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         (U) nullptr,
                                         stP,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         dInfo,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                         false,
                                         handle,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dIpiv,
                                         stP,
                                         (T) nullptr,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         dInfo,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                         false,
                                         handle,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dIpiv,
                                         stP,
                                         dB,
                                         ldb,
                                         stB,
                                         (T) nullptr,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         dInfo,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                         false,
                                         handle,
                                         n,
                                         nrhs,
                                         dA,
                                         lda,
                                         stA,
                                         dIpiv,
                                         stP,
                                         dB,
                                         ldb,
                                         stB,
                                         dX,
                                         ldx,
                                         stX,
                                         dWork,
                                         lwork,
                                         niters,
                                         (U) nullptr,
                                         bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gesv_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    int                    n    = 1;
    int                    nrhs = 1;
    int                    lda  = 1;
    int                    ldb  = 1;
    int                    ldx  = 1;
    int                    stA  = 1;
    int                    stP  = 1;
    int                    stB  = 1;
    int                    stX  = 1;
    int                    bc   = 1;

    if(BATCHED)
    {
        // // memory allocations
        // host_strided_batch_vector<int>   hNiters(1, 1, 1, 1);
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_batch_vector<T>           dB(1, 1, 1);
        // device_batch_vector<T>           dX(1, 1, 1);
        // device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dB.memcheck());
        // CHECK_HIP_ERROR(dX.memcheck());
        // CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // size_t size_W;
        // hipsolver_gesv_bufferSize(API,
        //                           handle,
        //                           n,
        //                           nrhs,
        //                           dA.data(),
        //                           lda,
        //                           dIpiv.data(),
        //                           dB.data(),
        //                           ldb,
        //                           dX.data(),
        //                           ldx,
        //                           &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // gesv_checkBadArgs<API>(handle,
        //                        n,
        //                        nrhs,
        //                        dA.data(),
        //                        lda,
        //                        stA,
        //                        dIpiv.data(),
        //                        stP,
        //                        dB.data(),
        //                        ldb,
        //                        stB,
        //                        dX.data(),
        //                        ldx,
        //                        stX,
        //                        dWork.data(),
        //                        size_W,
        //                        hNiters.data(),
        //                        dInfo.data(),
        //                        bc);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<int>   hNiters(1, 1, 1, 1);
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<T>   dX(1, 1, 1, 1);
        device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dX.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        size_t size_W;
        hipsolver_gesv_bufferSize(API,
                                  handle,
                                  n,
                                  nrhs,
                                  dA.data(),
                                  lda,
                                  dIpiv.data(),
                                  dB.data(),
                                  ldb,
                                  dX.data(),
                                  ldx,
                                  &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        gesv_checkBadArgs<API>(handle,
                               n,
                               nrhs,
                               dA.data(),
                               lda,
                               stA,
                               dIpiv.data(),
                               stP,
                               dB.data(),
                               ldb,
                               stB,
                               dX.data(),
                               ldx,
                               stX,
                               dWork.data(),
                               size_W,
                               hNiters.data(),
                               dInfo.data(),
                               bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gesv_initData(const hipsolverHandle_t handle,
                   const int               n,
                   const int               nrhs,
                   Td&                     dA,
                   const int               lda,
                   const int               stA,
                   Ud&                     dIpiv,
                   const int               stP,
                   Td&                     dB,
                   const int               ldb,
                   const int               stB,
                   const int               bc,
                   Th&                     hA,
                   Uh&                     hIpiv,
                   Th&                     hB)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

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
          bool      INPLACE,
          typename T,
          typename Td,
          typename Ud,
          typename Th,
          typename Uh>
void gesv_getError(const hipsolverHandle_t handle,
                   const int               n,
                   const int               nrhs,
                   Td&                     dA,
                   const int               lda,
                   const int               stA,
                   Ud&                     dIpiv,
                   const int               stP,
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
                   Uh&                     hIpiv,
                   Th&                     hB,
                   Th&                     hBRes,
                   Uh&                     hNiters,
                   Uh&                     hInfo,
                   Uh&                     hInfoRes,
                   double*                 max_err)
{
    // input data initialization
    gesv_initData<true, true, T>(
        handle, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_gesv(API,
                                       INPLACE,
                                       handle,
                                       n,
                                       nrhs,
                                       dA.data(),
                                       lda,
                                       stA,
                                       dIpiv.data(),
                                       stP,
                                       dB.data(),
                                       ldb,
                                       stB,
                                       dX.data(),
                                       ldx,
                                       stX,
                                       dWork.data(),
                                       lwork,
                                       hNiters.data(),
                                       dInfo.data(),
                                       bc));
    if(!INPLACE)
        CHECK_HIP_ERROR(hBRes.transfer_from(dX));
    else
        CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
    {
        cpu_gesv(n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb, hInfo[b]);
    }

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            err      = norm_error('I', n, nrhs, ldb, hB[b], hBRes[b], (!INPLACE ? ldx : ldb));
            *max_err = err > *max_err ? err : *max_err;
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
void gesv_getPerfData(const hipsolverHandle_t handle,
                      const int               n,
                      const int               nrhs,
                      Td&                     dA,
                      const int               lda,
                      const int               stA,
                      Ud&                     dIpiv,
                      const int               stP,
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
                      Uh&                     hIpiv,
                      Th&                     hB,
                      Uh&                     hNiters,
                      Uh&                     hInfo,
                      double*                 gpu_time_used,
                      double*                 cpu_time_used,
                      const int               hot_calls,
                      const bool              perf)
{
    if(!perf)
    {
        gesv_initData<true, false, T>(
            handle, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cpu_gesv(n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gesv_initData<true, false, T>(
        handle, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesv_initData<false, true, T>(
            handle, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

        CHECK_ROCBLAS_ERROR(hipsolver_gesv(API,
                                           INPLACE,
                                           handle,
                                           n,
                                           nrhs,
                                           dA.data(),
                                           lda,
                                           stA,
                                           dIpiv.data(),
                                           stP,
                                           dB.data(),
                                           ldb,
                                           stB,
                                           dX.data(),
                                           ldx,
                                           stX,
                                           dWork.data(),
                                           lwork,
                                           hNiters.data(),
                                           dInfo.data(),
                                           bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        gesv_initData<false, true, T>(
            handle, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

        start = get_time_us_sync(stream);
        hipsolver_gesv(API,
                       INPLACE,
                       handle,
                       n,
                       nrhs,
                       dA.data(),
                       lda,
                       stA,
                       dIpiv.data(),
                       stP,
                       dB.data(),
                       ldb,
                       stB,
                       dX.data(),
                       ldx,
                       stX,
                       dWork.data(),
                       lwork,
                       hNiters.data(),
                       dInfo.data(),
                       bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, bool INPLACE, typename T>
void testing_gesv(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    int                    n    = argus.get<int>("n");
    int                    nrhs = argus.get<int>("nrhs", n);
    int                    lda  = argus.get<int>("lda", n);
    int                    ldb  = argus.get<int>("ldb", n);
    int                    ldx  = argus.get<int>("ldx", n);
    int                    stA  = argus.get<int>("strideA", lda * n);
    int                    stP  = argus.get<int>("strideP", n);
    int                    stB  = argus.get<int>("strideB", ldb * nrhs);
    int                    stX  = argus.get<int>("strideX", ldx * nrhs);

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    int stBRes = (argus.unit_check || argus.norm_check) ? stX : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_B    = size_t(ldb) * nrhs;
    size_t size_X    = size_t(ldx) * nrhs;
    size_t size_P    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? (!INPLACE ? size_X : size_B) : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || ldx < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
            //                                      INPLACE,
            //                                      handle,
            //                                      n,
            //                                      nrhs,
            //                                      (T* const*)nullptr,
            //                                      lda,
            //                                      stA,
            //                                      (int*)nullptr,
            //                                      stP,
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
            EXPECT_ROCBLAS_STATUS(hipsolver_gesv(API,
                                                 INPLACE,
                                                 handle,
                                                 n,
                                                 nrhs,
                                                 (T*)nullptr,
                                                 lda,
                                                 stA,
                                                 (int*)nullptr,
                                                 stP,
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
    hipsolver_gesv_bufferSize(API,
                              handle,
                              n,
                              nrhs,
                              (T*)nullptr,
                              lda,
                              (int*)nullptr,
                              (T*)nullptr,
                              ldb,
                              (T*)nullptr,
                              ldx,
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
        // host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        // host_strided_batch_vector<int>   hNiters(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_batch_vector<T>           dB(size_B, 1, bc);
        // device_batch_vector<T>           dX(size_X, 1, bc);
        // device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_B)
        //     CHECK_HIP_ERROR(dB.memcheck());
        // if(size_X)
        //     CHECK_HIP_ERROR(dX.memcheck());
        // if(size_P)
        //     CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     gesv_getError<API, INPLACE, T>(handle,
        //                                    n,
        //                                    nrhs,
        //                                    dA,
        //                                    lda,
        //                                    stA,
        //                                    dIpiv,
        //                                    stP,
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
        //                                    hIpiv,
        //                                    hB,
        //                                    hBRes,
        //                                    hNiters,
        //                                    hInfo,
        //                                    hInfoRes,
        //                                    &max_error);

        // // collect performance data
        // if(argus.timing)
        //     gesv_getPerfData<API, INPLACE, T>(handle,
        //                                       n,
        //                                       nrhs,
        //                                       dA,
        //                                       lda,
        //                                       stA,
        //                                       dIpiv,
        //                                       stP,
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
        //                                       hIpiv,
        //                                       hB,
        //                                       hNiters,
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
        host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hNiters(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T>   dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T>   dX(size_X, 1, stX, bc);
        device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            gesv_getError<API, INPLACE, T>(handle,
                                           n,
                                           nrhs,
                                           dA,
                                           lda,
                                           stA,
                                           dIpiv,
                                           stP,
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
                                           hIpiv,
                                           hB,
                                           hBRes,
                                           hNiters,
                                           hInfo,
                                           hInfoRes,
                                           &max_error);

        // collect performance data
        if(argus.timing)
            gesv_getPerfData<API, INPLACE, T>(handle,
                                              n,
                                              nrhs,
                                              dA,
                                              lda,
                                              stA,
                                              dIpiv,
                                              stP,
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
                                              hIpiv,
                                              hB,
                                              hNiters,
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
                rocsolver_bench_output("n", "nrhs", "lda", "ldb", "ldx", "strideP", "batch_c");
                rocsolver_bench_output(n, nrhs, lda, ldb, ldx, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("n",
                                       "nrhs",
                                       "lda",
                                       "ldb",
                                       "ldx",
                                       "strideA",
                                       "strideP",
                                       "strideB",
                                       "strideX",
                                       "batch_c");
                rocsolver_bench_output(n, nrhs, lda, ldb, ldx, stA, stP, stB, stX, bc);
            }
            else
            {
                rocsolver_bench_output("n", "nrhs", "lda", "ldb", "ldx");
                rocsolver_bench_output(n, nrhs, lda, ldb, ldx);
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
