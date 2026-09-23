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
#include "hipsolver_timer.hpp"

template <testAPI_t API, typename I, typename Td, typename Id, typename INTd, typename TdWork>
void getri_checkBadArgs(const hipsolverHandle_t handle,
                        const I                 n,
                        Td                      dA,
                        const I                 lda,
                        Id                      dIpiv,
                        const I                 stP,
                        Td                      dC,
                        const I                 ldc,
                        TdWork                  dWork,
                        const I                 lwork,
                        INTd                    dinfo,
                        const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getri(
            API, false, nullptr, n, dA, lda, dIpiv, stP, dC, ldc, dWork, lwork, dinfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getri(
            API, false, handle, n, (Td) nullptr, lda, dIpiv, stP, dC, ldc, dWork, lwork, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getri(
            API, false, handle, n, dA, lda, dIpiv, stP, (Td) nullptr, ldc, dWork, lwork, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getri(
            API, false, handle, n, dA, lda, dIpiv, stP, dC, ldc, dWork, lwork, (INTd) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_getri_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    I                      n     = 1;
    I                      lda   = 1;
    I                      ldc   = 1;
    I                      stP   = 1;
    I                      lwork = 1;
    int                    bc    = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T>           dA(1, 1, 1);
        device_batch_vector<T>           dC(1, 1, 1);
        device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        SIZE size_W;
        hipsolver_getri_bufferSize(
            API, handle, n, dA.data(), lda, dIpiv.data(), stP, dC.data(), ldc, &size_W, bc);
        SIZE                           size_W_elems = (size_W + sizeof(T) - 1) / sizeof(T);
        device_strided_batch_vector<T> dWork(size_W_elems, 1, size_W_elems, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        getri_checkBadArgs<API>(handle,
                                n,
                                dA.data(),
                                lda,
                                dIpiv.data(),
                                stP,
                                dC.data(),
                                lda,
                                dWork.data(),
                                lwork,
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
void getri_initData(const hipsolverHandle_t handle,
                    const I                 n,
                    Td&                     dA,
                    const I                 lda,
                    Id&                     dIpiv,
                    const I                 stP,
                    Td&                     dC,
                    const I                 ldc,
                    INTd&                   dInfo,
                    const int               bc,
                    Th&                     hA,
                    Ih&                     hIpiv,
                    Th&                     hC,
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

            if(!NPVT)
            {
                // shuffle rows to test pivoting
                // always the same permutation for debugging purposes
                for(rocblas_int i = 0; i < n / 2; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        tmp                        = hA[b][i + j * lda];
                        hA[b][i + j * lda]         = hA[b][n - 1 - i + j * lda];
                        hA[b][n - 1 - i + j * lda] = tmp;
                    }
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

template <testAPI_t API,
          bool      NPVT,
          typename T,
          typename I,
          typename SIZE,
          typename Td,
          typename Id,
          typename INTd,
          typename TdWork,
          typename Th,
          typename Ih,
          typename INTh>
void getri_getError(const hipsolverHandle_t handle,
                    const I                 n,
                    Td&                     dA,
                    const I                 lda,
                    Id&                     dIpiv,
                    const I                 stP,
                    Td&                     dC,
                    const I                 ldc,
                    TdWork&                 dWork,
                    const SIZE              lwork,
                    INTd&                   dInfo,
                    const int               bc,
                    Th&                     hA,
                    Ih&                     hIpiv,
                    Ih&                     hIpivRes,
                    Th&                     hC,
                    Th&                     hCRes,
                    INTh&                   hInfo,
                    INTh&                   hInfoRes,
                    double*                 max_err)
{
    rocblas_int    sizeW = std::max(1, n);
    std::vector<T> hW(sizeW);

    // input data initialization (includes cpu_getrf to compute LU factorization)
    getri_initData<NPVT, true, true, T>(
        handle, n, dA, lda, dIpiv, stP, dC, ldc, dInfo, bc, hA, hIpiv, hC, hInfo);

    // execute computations
    // GPU lapack - A is input (LU), C is output (inverse)
    CHECK_ROCBLAS_ERROR(hipsolver_getri(API,
                                        NPVT,
                                        handle,
                                        n,
                                        dA.data(),
                                        lda,
                                        dIpiv.data(),
                                        stP,
                                        dC.data(),
                                        ldc,
                                        dWork.data(),
                                        lwork,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    if(!NPVT)
        CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack - compute inverse from LU factorization
    for(int b = 0; b < bc; ++b)
        cpu_getri(n, hA[b], lda, hIpiv[b], hW.data(), sizeW, hInfo[b]);

    // expecting original matrix to be non-singular
    // error is ||hA - hCRes|| / ||hA||
    // using frobenius norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('F', n, n, lda, hA[b], hCRes[b], ldc);
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

template <testAPI_t API,
          bool      NPVT,
          typename T,
          typename I,
          typename SIZE,
          typename Td,
          typename Twork,
          typename Id,
          typename INTd,
          typename Th,
          typename Ih,
          typename INTh>
void getri_getPerfData(const hipsolverHandle_t handle,
                       const I                 n,
                       Td&                     dA,
                       const I                 lda,
                       Id&                     dIpiv,
                       const I                 stP,
                       Td&                     dC,
                       const I                 ldc,
                       Twork&                  dWork,
                       const SIZE              lwork,
                       INTd&                   dInfo,
                       const int               bc,
                       Th&                     hA,
                       Ih&                     hIpiv,
                       Th&                     hC,
                       INTh&                   hInfo,
                       double*                 gpu_time_used,
                       double*                 cpu_time_used,
                       const int               hot_calls,
                       const bool              perf)
{
    if(!perf)
    {
        rocblas_int    sizeW = std::max(1, n);
        std::vector<T> hW(sizeW);

        getri_initData<NPVT, true, false, T>(
            handle, n, dA, lda, dIpiv, stP, dC, ldc, dInfo, bc, hA, hIpiv, hC, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_getri(n, hA[b], lda, hIpiv[b], hW.data(), sizeW, hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getri_initData<NPVT, true, false, T>(
        handle, n, dA, lda, dIpiv, stP, dC, ldc, dInfo, bc, hA, hIpiv, hC, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getri_initData<NPVT, false, true, T>(
            handle, n, dA, lda, dIpiv, stP, dC, ldc, dInfo, bc, hA, hIpiv, hC, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_getri(API,
                                            NPVT,
                                            handle,
                                            n,
                                            dA.data(),
                                            lda,
                                            dIpiv.data(),
                                            stP,
                                            dC.data(),
                                            ldc,
                                            dWork.data(),
                                            lwork,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    hipsolver_timer timer;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        getri_initData<NPVT, false, true, T>(
            handle, n, dA, lda, dIpiv, stP, dC, ldc, dInfo, bc, hA, hIpiv, hC, hInfo);

        timer.start(stream);
        hipsolver_getri(API,
                        NPVT,
                        handle,
                        n,
                        dA.data(),
                        lda,
                        dIpiv.data(),
                        stP,
                        dC.data(),
                        ldc,
                        dWork.data(),
                        lwork,
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
            EXPECT_ROCBLAS_STATUS(hipsolver_getri(API,
                                                  NPVT,
                                                  handle,
                                                  n,
                                                  (T**)nullptr,
                                                  lda,
                                                  (int*)nullptr,
                                                  stP,
                                                  (T**)nullptr,
                                                  ldc,
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
    hipsolver_getri_bufferSize(
        API, handle, n, (T**)nullptr, lda, (int*)nullptr, stP, (T**)nullptr, ldc, &size_W, bc);
    SIZE size_W_elems = (size_W + sizeof(T) - 1) / sizeof(T);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T>           hA(size_A, 1, bc);
        host_batch_vector<T>           hC(size_C, 1, bc);
        host_batch_vector<T>           hCRes(size_CRes, 1, bc);
        host_strided_batch_vector<int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T>         dA(size_A, 1, bc);
        device_batch_vector<T>         dC(size_C, 1, bc);
        device_strided_batch_vector<T> dWork(
            size_W_elems, 1, size_W_elems, 1); // size_W accounts for bc
        device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getri_getError<API, NPVT, T>(handle,
                                         n,
                                         dA,
                                         lda,
                                         dIpiv,
                                         stP,
                                         dC,
                                         ldc,
                                         dWork,
                                         size_W,
                                         dInfo,
                                         bc,
                                         hA,
                                         hIpiv,
                                         hIpivRes,
                                         hC,
                                         hCRes,
                                         hInfo,
                                         hInfoRes,
                                         &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            getri_getPerfData<API, NPVT, T>(handle,
                                            n,
                                            dA,
                                            lda,
                                            dIpiv,
                                            stP,
                                            dC,
                                            ldc,
                                            dWork,
                                            size_W,
                                            dInfo,
                                            bc,
                                            hA,
                                            hIpiv,
                                            hC,
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
                rocsolver_bench_output("n", "lda", "ldc", "strideP", "batch_c");
                rocsolver_bench_output(n, lda, ldc, stP, bc);
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
