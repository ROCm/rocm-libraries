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

#include "clientcommon.hpp"
#include "hipsolver_timer.hpp"

template <bool BATCHED,
          bool NPVT,
          typename I,
          typename Td,
          typename Twork,
          typename Id,
          typename INTd>
void getri_checkBadArgs(const hipsolverHandle_t handle,
                        const I                 n,
                        Td                      dA,
                        const I                 lda,
                        Td                      dC,
                        const I                 ldc,
                        Twork                   dWork,
                        const I                 lwork,
                        Id                      dIpiv,
                        const I                 stP,
                        INTd                    dinfo,
                        const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getriBatched(nullptr, n, dA, lda, dC, ldc, dWork, lwork, dIpiv, stP, dinfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getriBatched(
            handle, n, (Td) nullptr, lda, dC, ldc, dWork, lwork, dIpiv, stP, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getriBatched(
            handle, n, dA, lda, (Td) nullptr, ldc, dWork, lwork, dIpiv, stP, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getriBatched(
            handle, n, dA, lda, dC, ldc, dWork, lwork, dIpiv, stP, (INTd) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API,
          bool      BATCHED,
          bool      STRIDED,
          bool      NPVT,
          typename T,
          typename I,
          typename SIZE>
void testing_getri_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    I                      n     = 1;
    I                      lda   = 1;
    I                      stP   = 1;
    I                      lwork = 1;
    int                    bc    = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T>           dA(1, 1, 1);
        device_batch_vector<T>           dC(1, 1, 1);
        device_strided_batch_vector<T>   dWork(1, 1, 1, 1);
        device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dWork.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getri_checkBadArgs<BATCHED, NPVT, I, T**, T*, int*, int*>(handle,
                                                                  n,
                                                                  dA.data(),
                                                                  lda,
                                                                  dC.data(),
                                                                  lda,
                                                                  dWork.data(),
                                                                  lwork,
                                                                  dIpiv.data(),
                                                                  stP,
                                                                  dInfo.data(),
                                                                  bc);
    }
}

template <bool NPVT,
          bool CPU,
          bool GPU,
          typename T,
          typename I,
          typename Td,
          typename Id,
          typename INTd,
          typename Th,
          typename Ih,
          typename INTh>
void getriBatched_initData(const hipsolverHandle_t handle,
                           const I                 n,
                           Td&                     dA,
                           const I                 lda,
                           Td&                     dC,
                           const I                 ldc,
                           Id&                     dIpiv,
                           const I                 stP,
                           INTd&                   dInfo,
                           const int               bc,
                           Th&                     hA,
                           Th&                     hC,
                           Ih&                     hIpiv,
                           INTh&                   hInfo)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            // make it well-conditioned for inversion
            for(I i = 0; i < n; i++)
            {
                for(I j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = (hA[b][i + j * lda] / 10.0) + 10;
                    else
                        hA[b][i + j * lda] = (hA[b][i + j * lda] - 4) / 10.0;
                }
            }

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(rocblas_int i = 0; i < n / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    tmp                        = hA[b][i + j * lda];
                    hA[b][i + j * lda]         = hA[b][n - 1 - i + j * lda];
                    hA[b][n - 1 - i + j * lda] = tmp;
                }
            }

            // compute LU factorization as getri requires LU-factorized input
            cpu_getrf(n, n, hA[b], lda, hIpiv[b], hInfo[b]);
        }
    }

    if(GPU)
    {
        // now copy LU-factorized data and pivots to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <bool NPVT,
          typename T,
          typename I,
          typename Td,
          typename Twork,
          typename Id,
          typename INTd,
          typename Th,
          typename Ih,
          typename INTh>
void getriBatched_getError(const hipsolverHandle_t handle,
                           const I                 n,
                           Td&                     dA,
                           const I                 lda,
                           Td&                     dC,
                           const I                 ldc,
                           Id&                     dIpiv,
                           const I                 stP,
                           Twork&                  dWork,
                           const I                 lwork,
                           INTd&                   dInfo,
                           const int               bc,
                           Th&                     hA,
                           Th&                     hC,
                           Th&                     hCRes,
                           Ih&                     hIpiv,
                           Ih&                     hIpivRes,
                           INTh&                   hInfo,
                           INTh&                   hInfoRes,
                           double*                 max_err)
{
    // input data initialization (includes cpu_getrf to compute LU factorization)
    getriBatched_initData<NPVT, true, true, T>(
        handle, n, dA, lda, dC, ldc, dIpiv, stP, dInfo, bc, hA, hC, hIpiv, hInfo);

    // save LU-factorized matrix for CPU reference
    Th hA_LU(hA.n(), 1, hA.stride(), bc);
    for(int b = 0; b < bc; ++b)
    {
        for(I i = 0; i < n * lda; ++i)
            hA_LU[b][i] = hA[b][i];
    }

    // execute computations
    // GPU lapack - A is input (LU), C is output (inverse)
    CHECK_ROCBLAS_ERROR(hipsolver_getriBatched(handle,
                                               n,
                                               dA.data(),
                                               lda,
                                               dC.data(),
                                               ldc,
                                               dWork.data(),
                                               lwork,
                                               dIpiv.data(),
                                               stP,
                                               dInfo.data(),
                                               bc));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack - compute inverse from LU factorization
    for(int b = 0; b < bc; ++b)
        cpu_getri(n, hA_LU[b], lda, hIpiv[b], hInfo[b]);

    // expecting original matrix to be non-singular
    // error is ||hA_LU_inv - hCRes|| / ||hA_LU_inv||
    // using frobenius norm
    // NOTE: hA_LU has lda, hCRes has ldc
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('F', n, n, lda, hA_LU[b], hCRes[b], ldc);
        *max_err = err > *max_err ? err : *max_err;
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

template <bool NPVT,
          typename T,
          typename I,
          typename Td,
          typename Twork,
          typename Id,
          typename INTd,
          typename Th,
          typename Ih,
          typename INTh>
void getriBatched_getPerfData(const hipsolverHandle_t handle,
                              const I                 n,
                              Td&                     dA,
                              const I                 lda,
                              Td&                     dC,
                              const I                 ldc,
                              Id&                     dIpiv,
                              const I                 stP,
                              Twork&                  dWork,
                              const I                 lwork,
                              INTd&                   dInfo,
                              const int               bc,
                              Th&                     hA,
                              Th&                     hC,
                              Ih&                     hIpiv,
                              INTh&                   hInfo,
                              double*                 gpu_time_used,
                              double*                 cpu_time_used,
                              const int               hot_calls,
                              const bool              perf)
{
    if(!perf)
    {
        getriBatched_initData<NPVT, true, false, T>(
            handle, n, dA, lda, dC, ldc, dIpiv, stP, dInfo, bc, hA, hC, hIpiv, hInfo);

        // save LU-factorized matrix for CPU reference
        Th hA_LU(hA.n(), 1, hA.stride(), bc);
        for(int b = 0; b < bc; ++b)
        {
            for(I i = 0; i < n * lda; ++i)
                hA_LU[b][i] = hA[b][i];
        }

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_getri(n, hA_LU[b], lda, hIpiv[b], hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getriBatched_initData<NPVT, true, false, T>(
        handle, n, dA, lda, dC, ldc, dIpiv, stP, dInfo, bc, hA, hC, hIpiv, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getriBatched_initData<NPVT, false, true, T>(
            handle, n, dA, lda, dC, ldc, dIpiv, stP, dInfo, bc, hA, hC, hIpiv, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_getriBatched(handle,
                                                   n,
                                                   dA.data(),
                                                   lda,
                                                   dC.data(),
                                                   ldc,
                                                   dWork.data(),
                                                   lwork,
                                                   dIpiv.data(),
                                                   stP,
                                                   dInfo.data(),
                                                   bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    hipsolver_timer timer;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        getriBatched_initData<NPVT, false, true, T>(
            handle, n, dA, lda, dC, ldc, dIpiv, stP, dInfo, bc, hA, hC, hIpiv, hInfo);

        timer.start(stream);
        hipsolver_getriBatched(handle,
                               n,
                               dA.data(),
                               lda,
                               dC.data(),
                               ldc,
                               dWork.data(),
                               lwork,
                               dIpiv.data(),
                               stP,
                               dInfo.data(),
                               bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <testAPI_t API,
          bool      BATCHED,
          bool      STRIDED,
          bool      NPVT,
          typename T,
          typename I,
          typename SIZE>
void testing_getri(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    I                      n   = argus.get<int>("n");
    I                      lda = argus.get<int>("lda", n);
    I                      ldc = argus.get<int>("ldc", n);
    I                      stP = argus.get<int>("strideP", n);

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    I stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_C    = size_t(ldc) * n;
    size_t size_P    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldc < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_getriBatched(handle,
                                                         n,
                                                         (T**)nullptr,
                                                         lda,
                                                         (T**)nullptr,
                                                         ldc,
                                                         (T*)nullptr,
                                                         0,
                                                         (int*)nullptr,
                                                         stP,
                                                         (int*)nullptr,
                                                         bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    I lwork;
    hipsolver_getriBatched_bufferSize(
        handle, n, (T**)nullptr, lda, (T**)nullptr, ldc, stP, &lwork, bc);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, lwork);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T>             hA(size_A, 1, bc);
        host_batch_vector<T>             hC(size_C, 1, bc);
        host_batch_vector<T>             hCRes(size_CRes, 1, bc);
        host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_batch_vector<T>           dA(size_A, 1, bc);
        device_batch_vector<T>           dC(size_C, 1, bc);
        device_strided_batch_vector<T>   dWork(lwork, 1, lwork, bc);
        device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        if(lwork)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getriBatched_getError<NPVT,
                                  T,
                                  I,
                                  device_batch_vector<T>,
                                  device_strided_batch_vector<T>,
                                  device_strided_batch_vector<int>,
                                  device_strided_batch_vector<int>,
                                  host_batch_vector<T>,
                                  host_strided_batch_vector<int>,
                                  host_strided_batch_vector<int>>(handle,
                                                                  n,
                                                                  dA,
                                                                  lda,
                                                                  dC,
                                                                  ldc,
                                                                  dIpiv,
                                                                  stP,
                                                                  dWork,
                                                                  lwork,
                                                                  dInfo,
                                                                  bc,
                                                                  hA,
                                                                  hC,
                                                                  hCRes,
                                                                  hIpiv,
                                                                  hIpivRes,
                                                                  hInfo,
                                                                  hInfoRes,
                                                                  &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            getriBatched_getPerfData<NPVT,
                                     T,
                                     I,
                                     device_batch_vector<T>,
                                     device_strided_batch_vector<T>,
                                     device_strided_batch_vector<int>,
                                     device_strided_batch_vector<int>,
                                     host_batch_vector<T>,
                                     host_strided_batch_vector<int>,
                                     host_strided_batch_vector<int>>(handle,
                                                                     n,
                                                                     dA,
                                                                     lda,
                                                                     dC,
                                                                     ldc,
                                                                     dIpiv,
                                                                     stP,
                                                                     dWork,
                                                                     lwork,
                                                                     dInfo,
                                                                     bc,
                                                                     hA,
                                                                     hC,
                                                                     hIpiv,
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
                rocsolver_bench_output("n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(n, lda, stP, bc);
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
