/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>

#include "clientcommon.hpp"

template <testAPI_t API, typename I, typename SIZE, typename Td, typename Th>
void larft_checkBadArgs(const hipsolverHandle_t     handle,
                        const hipsolverDnParams_t   params,
                        const hipsolverDirectMode_t direct,
                        const hipsolverStorevMode_t storev,
                        const I                     n,
                        const I                     k,
                        Td                          dV,
                        const I                     ldv,
                        const I                     stV,
                        Td                          dTau,
                        const I                     stTau,
                        Td                          dT,
                        const I                     ldt,
                        const I                     stT,
                        Td                          dWork,
                        const SIZE                  dlwork,
                        Th                          hWork,
                        const SIZE                  hlwork,
                        const int                   bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                          nullptr,
                                          params,
                                          direct,
                                          storev,
                                          n,
                                          k,
                                          dV,
                                          ldv,
                                          stV,
                                          dTau,
                                          stTau,
                                          dT,
                                          ldt,
                                          stT,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                          handle,
                                          params,
                                          hipsolverDirectMode_t(-1),
                                          storev,
                                          n,
                                          k,
                                          dV,
                                          ldv,
                                          stV,
                                          dTau,
                                          stTau,
                                          dT,
                                          ldt,
                                          stT,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                          handle,
                                          params,
                                          direct,
                                          hipsolverStorevMode_t(-1),
                                          n,
                                          k,
                                          dV,
                                          ldv,
                                          stV,
                                          dTau,
                                          stTau,
                                          dT,
                                          ldt,
                                          stT,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    if constexpr(!std::is_same<I, int>::value)
        EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                              handle,
                                              (hipsolverDnParams_t) nullptr,
                                              direct,
                                              storev,
                                              n,
                                              k,
                                              dV,
                                              ldv,
                                              stV,
                                              dTau,
                                              stTau,
                                              dT,
                                              ldt,
                                              stT,
                                              dWork,
                                              dlwork,
                                              hWork,
                                              hlwork,
                                              bc),
                              HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                          handle,
                                          params,
                                          direct,
                                          storev,
                                          n,
                                          k,
                                          (Td) nullptr,
                                          ldv,
                                          stV,
                                          dTau,
                                          stTau,
                                          dT,
                                          ldt,
                                          stT,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                          handle,
                                          params,
                                          direct,
                                          storev,
                                          n,
                                          k,
                                          dV,
                                          ldv,
                                          stV,
                                          (Td) nullptr,
                                          stTau,
                                          dT,
                                          ldt,
                                          stT,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                          handle,
                                          params,
                                          direct,
                                          storev,
                                          n,
                                          k,
                                          dV,
                                          ldv,
                                          stV,
                                          dTau,
                                          stTau,
                                          (Td) nullptr,
                                          ldt,
                                          stT,
                                          dWork,
                                          dlwork,
                                          hWork,
                                          hlwork,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_larft_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    hipsolverDirectMode_t  direct = HIPSOLVER_DIRECT_FORWARD;
    hipsolverStorevMode_t  storev = HIPSOLVER_STOREV_COLUMNWISE;
    I                      n      = 10;
    I                      k      = 5;
    I                      ldv    = 10;
    I                      ldt    = 5;
    I                      stV    = 1;
    I                      stTau  = 1;
    I                      stT    = 1;
    int                    bc     = 1;

#ifdef HIPSOLVER_ENABLE_EIGENSOLVERS_64
    if(BATCHED)
    {
        // memory allocations
        // device_batch_vector<T>         dV(1, 1, 1);
        // device_strided_batch_vector<T> dTau(1, 1, 1, 1);
        // device_strided_batch_vector<T> dT(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dV.memcheck());
        // CHECK_HIP_ERROR(dTau.memcheck());
        // CHECK_HIP_ERROR(dT.memcheck());

        // SIZE size_dW, size_hW;
        // hipsolver_larft_bufferSize(
        //     API, handle, params, direct, storev, n, k, dV.data(), ldv, dTau.data(), dT.data(), ldt, &size_dW, &size_hW);
        // host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        // device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        // if(size_dW)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // larft_checkBadArgs<API>(handle,
        //                         params,
        //                         direct,
        //                         storev,
        //                         n,
        //                         k,
        //                         dV.data(),
        //                         ldv,
        //                         stV,
        //                         dTau.data(),
        //                         stTau,
        //                         dT.data(),
        //                         ldt,
        //                         stT,
        //                         dWork.data(),
        //                         size_dW,
        //                         hWork.data(),
        //                         size_hW,
        //                         bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        device_strided_batch_vector<T> dTau(1, 1, 1, 1);
        device_strided_batch_vector<T> dT(1, 1, 1, 1);
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dTau.memcheck());
        CHECK_HIP_ERROR(dT.memcheck());

        SIZE size_dW, size_hW;
        hipsolver_larft_bufferSize(API,
                                   handle,
                                   params,
                                   direct,
                                   storev,
                                   n,
                                   k,
                                   dV.data(),
                                   ldv,
                                   dTau.data(),
                                   dT.data(),
                                   ldt,
                                   &size_dW,
                                   &size_hW);
        host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        larft_checkBadArgs<API>(handle,
                                params,
                                direct,
                                storev,
                                n,
                                k,
                                dV.data(),
                                ldv,
                                stV,
                                dTau.data(),
                                stTau,
                                dT.data(),
                                ldt,
                                stT,
                                dWork.data(),
                                size_dW,
                                hWork.data(),
                                size_hW,
                                bc);
    }
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void larft_initData(const hipsolverHandle_t     handle,
                    const hipsolverDirectMode_t direct,
                    const hipsolverStorevMode_t storev,
                    const int                   n,
                    const int                   k,
                    Td&                         dV,
                    const int                   ldv,
                    const int                   stV,
                    Td&                         dTau,
                    const int                   stTau,
                    Td&                         dT,
                    const int                   ldt,
                    const int                   stT,
                    const int                   bc,
                    Th&                         hV,
                    Th&                         hTau,
                    Th&                         hT,
                    std::vector<T>&             hw,
                    size_t                      size_w)
{
    if(CPU)
    {
        rocblas_init<T>(hV, true);

        // scale to avoid singularities
        // and create householder reflectors
        if(storev == HIPSOLVER_STOREV_COLUMNWISE)
        {
            for(int j = 0; j < k; ++j)
            {
                for(int i = 0; i < n; ++i)
                {
                    if(i == j)
                        hV[0][i + j * ldv] += 400;
                    else
                        hV[0][i + j * ldv] -= 4;
                }
            }

            int info;
            if(direct == HIPSOLVER_DIRECT_FORWARD)
                cpu_geqrf(n, k, hV[0], ldv, hTau[0], hw.data(), size_w, &info);
            else
                cpu_geqlf(n, k, hV[0], ldv, hTau[0], hw.data(), size_w, &info);
        }
        else
        {
            for(int j = 0; j < n; ++j)
            {
                for(int i = 0; i < k; ++i)
                {
                    if(i == j)
                        hV[0][i + j * ldv] += 400;
                    else
                        hV[0][i + j * ldv] -= 4;
                }
            }

            int info;
            if(direct == HIPSOLVER_DIRECT_FORWARD)
                cpu_gelqf(k, n, hV[0], ldv, hTau[0], hw.data(), size_w, &info);
            else
                cpu_gerqf(k, n, hV[0], ldv, hTau[0], hw.data(), size_w, &info);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dTau.transfer_from(hTau));
    }
}

template <testAPI_t API, typename I, typename SIZE, typename T, typename Td, typename Th>
void larft_getError(const hipsolverHandle_t     handle,
                    const hipsolverDnParams_t   params,
                    const hipsolverDirectMode_t direct,
                    const hipsolverStorevMode_t storev,
                    const I                     n,
                    const I                     k,
                    Td&                         dV,
                    const I                     ldv,
                    const I                     stV,
                    Td&                         dTau,
                    const I                     stTau,
                    Td&                         dT,
                    const I                     ldt,
                    const I                     stT,
                    Td&                         dWork,
                    const SIZE                  dlwork,
                    Th&                         hWork,
                    const SIZE                  hlwork,
                    const int                   bc,
                    Th&                         hV,
                    Th&                         hTau,
                    Th&                         hT,
                    Th&                         hTRes,
                    double*                     max_err)
{
    size_t         size_w = size_t(k);
    std::vector<T> hw(size_w);

    // input data initialization
    larft_initData<true, true, T>(handle,
                                  direct,
                                  storev,
                                  n,
                                  k,
                                  dV,
                                  ldv,
                                  stV,
                                  dTau,
                                  stTau,
                                  dT,
                                  ldt,
                                  stT,
                                  bc,
                                  hV,
                                  hTau,
                                  hT,
                                  hw,
                                  size_w);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_larft(API,
                                        handle,
                                        params,
                                        direct,
                                        storev,
                                        n,
                                        k,
                                        dV.data(),
                                        ldv,
                                        stV,
                                        dTau.data(),
                                        stTau,
                                        dT.data(),
                                        ldt,
                                        stT,
                                        dWork.data(),
                                        dlwork,
                                        hWork.data(),
                                        hlwork,
                                        bc));
    CHECK_HIP_ERROR(hTRes.transfer_from(dT));

    // CPU lapack
    cpu_larft(direct, storev, n, k, hV[0], ldv, hTau[0], hT[0], ldt);

    // error is ||hT - hTRes|| / ||hT||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = (direct == HIPSOLVER_DIRECT_FORWARD)
                   ? norm_error_upperTr('F', k, k, ldt, hT[0], hTRes[0])
                   : norm_error_lowerTr('F', k, k, ldt, hT[0], hTRes[0]);
}

template <testAPI_t API, typename I, typename SIZE, typename T, typename Td, typename Th>
void larft_getPerfData(const hipsolverHandle_t     handle,
                       const hipsolverDnParams_t   params,
                       const hipsolverDirectMode_t direct,
                       const hipsolverStorevMode_t storev,
                       const I                     n,
                       const I                     k,
                       Td&                         dV,
                       const I                     ldv,
                       const I                     stV,
                       Td&                         dTau,
                       const I                     stTau,
                       Td&                         dT,
                       const I                     ldt,
                       const I                     stT,
                       Td&                         dWork,
                       const SIZE                  dlwork,
                       Th&                         hWork,
                       const SIZE                  hlwork,
                       const int                   bc,
                       Th&                         hV,
                       Th&                         hTau,
                       Th&                         hT,
                       double*                     gpu_time_used,
                       double*                     cpu_time_used,
                       const int                   hot_calls,
                       const bool                  perf)
{
    size_t         size_w = size_t(k);
    std::vector<T> hw(size_w);

    if(!perf)
    {
        larft_initData<true, false, T>(handle,
                                       direct,
                                       storev,
                                       n,
                                       k,
                                       dV,
                                       ldv,
                                       stV,
                                       dTau,
                                       stTau,
                                       dT,
                                       ldt,
                                       stT,
                                       bc,
                                       hV,
                                       hTau,
                                       hT,
                                       hw,
                                       size_w);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_larft(direct, storev, n, k, hV[0], ldv, hTau[0], hT[0], ldt);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    larft_initData<true, false, T>(handle,
                                   direct,
                                   storev,
                                   n,
                                   k,
                                   dV,
                                   ldv,
                                   stV,
                                   dTau,
                                   stTau,
                                   dT,
                                   ldt,
                                   stT,
                                   bc,
                                   hV,
                                   hTau,
                                   hT,
                                   hw,
                                   size_w);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larft_initData<false, true, T>(handle,
                                       direct,
                                       storev,
                                       n,
                                       k,
                                       dV,
                                       ldv,
                                       stV,
                                       dTau,
                                       stTau,
                                       dT,
                                       ldt,
                                       stT,
                                       bc,
                                       hV,
                                       hTau,
                                       hT,
                                       hw,
                                       size_w);

        CHECK_ROCBLAS_ERROR(hipsolver_larft(API,
                                            handle,
                                            params,
                                            direct,
                                            storev,
                                            n,
                                            k,
                                            dV.data(),
                                            ldv,
                                            stV,
                                            dTau.data(),
                                            stTau,
                                            dT.data(),
                                            ldt,
                                            stT,
                                            dWork.data(),
                                            dlwork,
                                            hWork.data(),
                                            hlwork,
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        larft_initData<false, true, T>(handle,
                                       direct,
                                       storev,
                                       n,
                                       k,
                                       dV,
                                       ldv,
                                       stV,
                                       dTau,
                                       stTau,
                                       dT,
                                       ldt,
                                       stT,
                                       bc,
                                       hV,
                                       hTau,
                                       hT,
                                       hw,
                                       size_w);

        start = get_time_us_sync(stream);
        hipsolver_larft(API,
                        handle,
                        params,
                        direct,
                        storev,
                        n,
                        k,
                        dV.data(),
                        ldv,
                        stV,
                        dTau.data(),
                        stTau,
                        dT.data(),
                        ldt,
                        stT,
                        dWork.data(),
                        dlwork,
                        hWork.data(),
                        hlwork,
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_larft(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    char                   directC = argus.get<char>("direct");
    char                   storevC = argus.get<char>("storev");
    I                      n       = argus.get<rocblas_int>("n");
    I                      k       = argus.get<rocblas_int>("k", n);
    I                      ldv     = argus.get<rocblas_int>("ldv", n);
    I                      ldt     = argus.get<rocblas_int>("ldt", k);
    I                      stV
        = argus.get<rocblas_int>("strideV", (storevC == 'C' || storevC == 'c') ? ldv * k : ldv * n);
    I stTau = argus.get<rocblas_int>("strideTau", k);
    I stT   = argus.get<rocblas_int>("strideT", ldt * k);

    hipsolverDirectMode_t direct = char2hipsolver_direct(directC);
    hipsolverStorevMode_t storev = char2hipsolver_storev(storevC);

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    I stVRes   = (argus.unit_check || argus.norm_check) ? stV : 0;
    I stTauRes = (argus.unit_check || argus.norm_check) ? stTau : 0;
    I stTRes   = (argus.unit_check || argus.norm_check) ? stT : 0;

#ifndef HIPSOLVER_ENABLE_EIGENSOLVERS_64
    // 64-bit API disabled: entry points must report not_implemented.
    if constexpr(std::is_same<I, int64_t>::value)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                              handle,
                                              params,
                                              direct,
                                              storev,
                                              n,
                                              k,
                                              (T*)nullptr,
                                              ldv,
                                              stV,
                                              (T*)nullptr,
                                              stTau,
                                              (T*)nullptr,
                                              ldt,
                                              stT,
                                              (T*)nullptr,
                                              0,
                                              (T*)nullptr,
                                              0,
                                              bc),
                              HIPSOLVER_STATUS_NOT_SUPPORTED);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check non-supported values
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_AMD__)
    if(storev != HIPSOLVER_STOREV_COLUMNWISE)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
            //                                       handle,
            //                                       params,
            //                                       direct,
            //                                       storev,
            //                                       n,
            //                                       k,
            //                                       (T*)nullptr,
            //                                       ldv,
            //                                       stV,
            //                                       (T*)nullptr,
            //                                       stTau,
            //                                       (T*)nullptr,
            //                                       ldt,
            //                                       stT,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       bc),
            //                       HIPSOLVER_STATUS_NOT_SUPPORTED);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                                  handle,
                                                  params,
                                                  direct,
                                                  storev,
                                                  n,
                                                  k,
                                                  (T*)nullptr,
                                                  ldv,
                                                  stV,
                                                  (T*)nullptr,
                                                  stTau,
                                                  (T*)nullptr,
                                                  ldt,
                                                  stT,
                                                  (T*)nullptr,
                                                  0,
                                                  (T*)nullptr,
                                                  0,
                                                  bc),
                                  HIPSOLVER_STATUS_NOT_SUPPORTED);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }
#endif

    // determine sizes
    bool   col       = (storev == HIPSOLVER_STOREV_COLUMNWISE);
    size_t size_V    = col ? size_t(ldv) * k : size_t(ldv) * n;
    size_t size_Tau  = size_t(k);
    size_t size_T    = size_t(ldt) * k;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_VRes   = (argus.unit_check || argus.norm_check) ? size_V : 0;
    size_t size_TauRes = (argus.unit_check || argus.norm_check) ? size_Tau : 0;
    size_t size_TRes   = (argus.unit_check || argus.norm_check) ? size_T : 0;

    // check invalid sizes
    bool invalid_size = col ? (n < 0 || k < 0 || k > n || ldv < n || ldt < k || bc < 0)
                            : (n < 0 || k < 0 || k > n || ldv < k || ldt < k || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
            //                                       handle,
            //                                       params,
            //                                       direct,
            //                                       storev,
            //                                       n,
            //                                       k,
            //                                       (T*)nullptr,
            //                                       ldv,
            //                                       stV,
            //                                       (T*)nullptr,
            //                                       stTau,
            //                                       (T*)nullptr,
            //                                       ldt,
            //                                       stT,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                                  handle,
                                                  params,
                                                  direct,
                                                  storev,
                                                  n,
                                                  k,
                                                  (T*)nullptr,
                                                  ldv,
                                                  stV,
                                                  (T*)nullptr,
                                                  stTau,
                                                  (T*)nullptr,
                                                  ldt,
                                                  stT,
                                                  (T*)nullptr,
                                                  0,
                                                  (T*)nullptr,
                                                  0,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query
    SIZE size_dW, size_hW;
    hipsolver_larft_bufferSize(API,
                               handle,
                               params,
                               direct,
                               storev,
                               n,
                               k,
                               (T*)nullptr,
                               ldv,
                               (T*)nullptr,
                               (T*)nullptr,
                               ldt,
                               &size_dW,
                               &size_hW);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_dW);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T>   hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T>   hTau(size_Tau, 1, stTau, bc);
    host_strided_batch_vector<T>   hT(size_T, 1, stT, bc);
    host_strided_batch_vector<T>   hTRes(size_TRes, 1, stTRes, bc);
    host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
    device_strided_batch_vector<T> dV(size_V, 1, stV, bc);
    device_strided_batch_vector<T> dTau(size_Tau, 1, stTau, bc);
    device_strided_batch_vector<T> dT(size_T, 1, stT, bc);
    device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_Tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_T)
        CHECK_HIP_ERROR(dT.memcheck());
    if(size_dW)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check quick return
    if(n == 0 || k == 0 || bc == 0)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_larft(API,
                                              handle,
                                              params,
                                              direct,
                                              storev,
                                              n,
                                              k,
                                              dV.data(),
                                              ldv,
                                              stV,
                                              dTau.data(),
                                              stTau,
                                              dT.data(),
                                              ldt,
                                              stT,
                                              dWork.data(),
                                              size_dW,
                                              hWork.data(),
                                              size_hW,
                                              bc),
                              HIPSOLVER_STATUS_SUCCESS);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larft_getError<API, I, SIZE, T>(handle,
                                        params,
                                        direct,
                                        storev,
                                        n,
                                        k,
                                        dV,
                                        ldv,
                                        stV,
                                        dTau,
                                        stTau,
                                        dT,
                                        ldt,
                                        stT,
                                        dWork,
                                        size_dW,
                                        hWork,
                                        size_hW,
                                        bc,
                                        hV,
                                        hTau,
                                        hT,
                                        hTRes,
                                        &max_error);

    // collect performance data
    if(argus.timing)
        larft_getPerfData<API, I, SIZE, T>(handle,
                                           params,
                                           direct,
                                           storev,
                                           n,
                                           k,
                                           dV,
                                           ldv,
                                           stV,
                                           dTau,
                                           stTau,
                                           dT,
                                           ldt,
                                           stT,
                                           dWork,
                                           size_dW,
                                           hWork,
                                           size_hW,
                                           bc,
                                           hV,
                                           hTau,
                                           hT,
                                           &gpu_time_used,
                                           &cpu_time_used,
                                           hot_calls,
                                           argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver test
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("direct", "storev", "n", "k", "ldv", "ldt");
            rocsolver_bench_output(directC, storevC, n, k, ldv, ldt);
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
