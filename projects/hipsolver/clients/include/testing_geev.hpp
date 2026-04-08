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

#include "clientcommon.hpp"

template <testAPI_t API, typename I, typename SIZE, typename Td, typename INTd, typename Th>
void geev_checkBadArgs(const hipsolverHandle_t   handle,
                       const hipsolverDnParams_t params,
                       const hipsolverEigMode_t  jobvl,
                       const hipsolverEigMode_t  jobvr,
                       const I                   n,
                       Td                        dA,
                       const I                   lda,
                       Td                        dW,
                       Td                        dVL,
                       const I                   ldvl,
                       Td                        dVR,
                       const I                   ldvr,
                       Td                        dWork,
                       const SIZE                dlwork,
                       Th                        hWork,
                       const SIZE                hlwork,
                       INTd                      dinfo)
{
    // TODO
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_geev_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    hipsolverEigMode_t     jobvl = HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverEigMode_t     jobvr = HIPSOLVER_EIG_MODE_NOVECTOR;
    I                      n     = 1;
    I                      lda   = 1;
    I                      ldvl  = 1;
    I                      ldvr  = 1;

    if(BATCHED)
    {
        // unsupported
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dW(1, 1, 1, 1);
        device_strided_batch_vector<T>   dVL(1, 1, 1, 1);
        device_strided_batch_vector<T>   dVR(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        SIZE size_dW, size_hW;
        hipsolver_geev_bufferSize(API,
                                  handle,
                                  params,
                                  jobvl,
                                  jobvr,
                                  n,
                                  dA.data(),
                                  lda,
                                  dW.data(),
                                  dVL.data(),
                                  ldvl,
                                  dVR.data(),
                                  ldvr,
                                  &size_dW,
                                  &size_hW);
        host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        geev_checkBadArgs<API>(handle,
                               params,
                               jobvl,
                               jobvr,
                               n,
                               dA.data(),
                               lda,
                               dW.data(),
                               dVL.data(),
                               ldvl,
                               dVR.data(),
                               ldvr,
                               dWork.data(),
                               size_dW,
                               hWork.data(),
                               size_hW,
                               dInfo.data());
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void geev_initData(const hipsolverHandle_t   handle,
                   const hipsolverDnParams_t params,
                   const I                   n,
                   Td&                       dA,
                   const I                   lda,
                   Th&                       hA)
{
    if(CPU)
    {
        // TODO
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          typename T,
          typename I,
          typename SIZE,
          typename Td,
          typename INTd,
          typename Th,
          typename INTh>
void geev_getError(const hipsolverHandle_t   handle,
                   const hipsolverDnParams_t params,
                   const hipsolverEigMode_t  jobvl,
                   const hipsolverEigMode_t  jobvr,
                   const I                   n,
                   Td&                       dA,
                   const I                   lda,
                   Td&                       dW,
                   Td&                       dVL,
                   const I                   ldvl,
                   Td&                       dVR,
                   const I                   ldvr,
                   Td&                       dWork,
                   const SIZE                dlwork,
                   Th&                       hWork,
                   const SIZE                hlwork,
                   INTd&                     dInfo,
                   Th&                       hA,
                   Th&                       hARes,
                   Th&                       hW,
                   Th&                       hWRes,
                   Th&                       hVL,
                   Th&                       hVLRes,
                   Th&                       hVR,
                   Th&                       hVRRes,
                   INTh&                     hInfo,
                   INTh&                     hInfoRes,
                   double*                   max_err)
{
    // input data initialization
    geev_initData<true, true, T>(handle, params, n, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_geev(API,
                                       handle,
                                       params,
                                       jobvl,
                                       jobvr,
                                       n,
                                       dA.data(),
                                       lda,
                                       dW.data(),
                                       dVL.data(),
                                       ldvl,
                                       dVR.data(),
                                       ldvr,
                                       dWork.data(),
                                       dlwork,
                                       hWork.data(),
                                       hlwork,
                                       dInfo.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    if(jobvl != HIPSOLVER_EIG_MODE_NOVECTOR)
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
    if(jobvr != HIPSOLVER_EIG_MODE_NOVECTOR)
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // TODO
    *max_err = 1;
}

template <testAPI_t API,
          typename T,
          typename I,
          typename SIZE,
          typename Td,
          typename INTd,
          typename Th>
void geev_getPerfData(const hipsolverHandle_t   handle,
                      const hipsolverDnParams_t params,
                      const hipsolverEigMode_t  jobvl,
                      const hipsolverEigMode_t  jobvr,
                      const I                   n,
                      Td&                       dA,
                      const I                   lda,
                      Td&                       dW,
                      Td&                       dVL,
                      const I                   ldvl,
                      Td&                       dVR,
                      const I                   ldvr,
                      Td&                       dWork,
                      const SIZE                dlwork,
                      Th&                       hWork,
                      const SIZE                hlwork,
                      INTd&                     dInfo,
                      Th&                       hA,
                      double*                   gpu_time_used,
                      double*                   cpu_time_used,
                      const int                 hot_calls,
                      const bool                perf)
{
    if(!perf)
    {
        // TODO
    }

    geev_initData<true, false, T>(handle, params, n, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geev_initData<false, true, T>(handle, params, n, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(hipsolver_geev(API,
                                           handle,
                                           params,
                                           jobvl,
                                           jobvr,
                                           n,
                                           dA.data(),
                                           lda,
                                           dW.data(),
                                           dVL.data(),
                                           ldvl,
                                           dVR.data(),
                                           ldvr,
                                           dWork.data(),
                                           dlwork,
                                           hWork.data(),
                                           hlwork,
                                           dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        geev_initData<false, true, T>(handle, params, n, dA, lda, hA);

        start = get_time_us_sync(stream);
        hipsolver_geev(API,
                       handle,
                       params,
                       jobvl,
                       jobvr,
                       n,
                       dA.data(),
                       lda,
                       dW.data(),
                       dVL.data(),
                       ldvl,
                       dVR.data(),
                       ldvr,
                       dWork.data(),
                       dlwork,
                       hWork.data(),
                       hlwork,
                       dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_geev(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    char                   jobvlC = argus.get<char>("jobvl");
    char                   jobvrC = argus.get<char>("jobvr");
    I                      n      = argus.get<int>("n");
    I                      lda    = argus.get<int>("lda", n);
    I                      ldvl   = argus.get<int>("ldvl", n);
    I                      ldvr   = argus.get<int>("ldvr", n);

    hipsolverEigMode_t jobvl     = char2hipsolver_evect(jobvlC);
    hipsolverEigMode_t jobvr     = char2hipsolver_evect(jobvrC);
    int                bc        = argus.batch_count;
    int                hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_W    = is_complex<T> ? size_t(n) : size_t(2 * n);
    size_t size_VL   = jobvlC == 'N' ? 0 : size_t(ldvl) * n;
    size_t size_VR   = jobvrC == 'N' ? 0 : size_t(ldvr) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes  = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes  = (argus.unit_check || argus.norm_check) ? size_W : 0;
    size_t size_VLRes = (argus.unit_check || argus.norm_check) ? size_VL : 0;
    size_t size_VRRes = (argus.unit_check || argus.norm_check) ? size_VR : 0;

    // check invalid sizes
    bool invalid_size    = (n < 0 || lda < n || bc < 0);
    bool invalid_size_vl = ldvl < (jobvl == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : n);
    bool invalid_size_vr = ldvr < (jobvr == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : n);
    if(invalid_size || invalid_size_vl || invalid_size_vr)
    {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
        if(BATCHED)
        {
            // unsupported
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                                 handle,
                                                 params,
                                                 jobvl,
                                                 jobvr,
                                                 n,
                                                 (T*)nullptr,
                                                 lda,
                                                 (T*)nullptr,
                                                 (T*)nullptr,
                                                 ldvl,
                                                 (T*)nullptr,
                                                 ldvr,
                                                 (T*)nullptr,
                                                 0,
                                                 (T*)nullptr,
                                                 0,
                                                 (int*)nullptr),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }
#endif

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    SIZE size_dW, size_hW;
    hipsolver_geev_bufferSize(API,
                              handle,
                              params,
                              jobvl,
                              jobvr,
                              n,
                              (T*)nullptr,
                              lda,
                              (T*)nullptr,
                              (T*)nullptr,
                              ldvl,
                              (T*)nullptr,
                              ldvr,
                              &size_dW,
                              &size_hW);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_dW);
        return;
    }

    if(BATCHED)
    {
        // unsupported
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, size_A, bc);
        host_strided_batch_vector<T>     hARes(size_ARes, 1, size_ARes, bc);
        host_strided_batch_vector<T>     hW(size_W, 1, size_W, bc);
        host_strided_batch_vector<T>     hWRes(size_WRes, 1, size_WRes, bc);
        host_strided_batch_vector<T>     hVL(size_VL, 1, size_VL, bc);
        host_strided_batch_vector<T>     hVLRes(size_VLRes, 1, size_VLRes, bc);
        host_strided_batch_vector<T>     hVR(size_VR, 1, size_VR, bc);
        host_strided_batch_vector<T>     hVRRes(size_VRRes, 1, size_VRRes, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        host_strided_batch_vector<T>     hWork(size_hW, 1, size_hW, 1); // size_hW accounts for bc
        device_strided_batch_vector<T>   dA(size_A, 1, size_A, bc);
        device_strided_batch_vector<T>   dW(size_W, 1, size_W, bc);
        device_strided_batch_vector<T>   dVL(size_VL, 1, size_VL, bc);
        device_strided_batch_vector<T>   dVR(size_VR, 1, size_VR, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_dW, 1, size_dW, 1); // size_dW accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dW.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            geev_getError<API, T>(handle,
                                  params,
                                  jobvl,
                                  jobvr,
                                  n,
                                  dA,
                                  lda,
                                  dW,
                                  dVL,
                                  ldvl,
                                  dVR,
                                  ldvr,
                                  dWork,
                                  size_dW,
                                  hWork,
                                  size_hW,
                                  dInfo,
                                  hA,
                                  hARes,
                                  hW,
                                  hWRes,
                                  hVL,
                                  hVLRes,
                                  hVR,
                                  hVRRes,
                                  hInfo,
                                  hInfoRes,
                                  &max_error);

        // collect performance data
        if(argus.timing)
            geev_getPerfData<API, T>(handle,
                                     params,
                                     jobvl,
                                     jobvr,
                                     n,
                                     dA,
                                     lda,
                                     dW,
                                     dVL,
                                     ldvl,
                                     dVR,
                                     ldvr,
                                     dWork,
                                     size_dW,
                                     hWork,
                                     size_hW,
                                     dInfo,
                                     hA,
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
            rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "ldvl", "ldvr");
            rocsolver_bench_output(jobvlC, jobvrC, n, lda, ldvl, ldvr);

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
