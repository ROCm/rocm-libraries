/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "hipsolverSp.hpp"

template <bool HOST, typename T>
void csrlsvqr_checkBadArgs(hipsolverSpHandle_t       handle,
                           const int                 n,
                           const int                 nnzA,
                           const hipsparseMatDescr_t descrA,
                           int*                      ptrA,
                           int*                      indA,
                           T                         valA,
                           T                         B,
                           T                         X,
                           int*                      singularity)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, nullptr, n, nnzA, descrA, valA, ptrA, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, handle, n, nnzA, nullptr, valA, ptrA, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, handle, n, nnzA, descrA, (T) nullptr, ptrA, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, handle, n, nnzA, descrA, valA, (int*)nullptr, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, handle, n, nnzA, descrA, valA, ptrA, (int*)nullptr, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, handle, n, nnzA, descrA, valA, ptrA, indA, (T) nullptr, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvqr(
            HOST, handle, n, nnzA, descrA, valA, ptrA, indA, B, 0, 0, (T) nullptr, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <bool HOST, typename T>
void testing_csrlsvqr_bad_arg()
{
    // safe arguments
    hipsolverSp_local_handle handle;
    int                      n    = 1;
    int                      nnzA = 1;

    hipsparse_local_mat_descr descrA;
    hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
    hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO);

    if(HOST)
    {
        // memory allocations
        host_strided_batch_vector<int> singularity(1, 1, 1, 1);
        host_strided_batch_vector<int> ptrA(1, 1, 1, 1);
        host_strided_batch_vector<int> indA(1, 1, 1, 1);
        host_strided_batch_vector<T>   valA(1, 1, 1, 1);
        host_strided_batch_vector<T>   B(1, 1, 1, 1);
        host_strided_batch_vector<T>   X(1, 1, 1, 1);

        // check bad arguments
        csrlsvqr_checkBadArgs<HOST>(handle,
                                    n,
                                    nnzA,
                                    descrA,
                                    ptrA.data(),
                                    indA.data(),
                                    valA.data(),
                                    B.data(),
                                    X.data(),
                                    singularity.data());
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<int>   singularity(1, 1, 1, 1);
        device_strided_batch_vector<int> ptrA(1, 1, 1, 1);
        device_strided_batch_vector<int> indA(1, 1, 1, 1);
        device_strided_batch_vector<T>   valA(1, 1, 1, 1);
        device_strided_batch_vector<T>   B(1, 1, 1, 1);
        device_strided_batch_vector<T>   X(1, 1, 1, 1);
        CHECK_HIP_ERROR(ptrA.memcheck());
        CHECK_HIP_ERROR(indA.memcheck());
        CHECK_HIP_ERROR(valA.memcheck());
        CHECK_HIP_ERROR(B.memcheck());
        CHECK_HIP_ERROR(X.memcheck());

        // check bad arguments
        csrlsvqr_checkBadArgs<HOST>(handle,
                                    n,
                                    nnzA,
                                    descrA,
                                    ptrA.data(),
                                    indA.data(),
                                    valA.data(),
                                    B.data(),
                                    X.data(),
                                    singularity.data());
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrlsvqr_initData(hipsolverSpHandle_t handle,
                       const int           n,
                       const int           nnzA,
                       hipsparseMatDescr_t descrA,
                       Ud&                 dptrA,
                       Ud&                 dindA,
                       Td&                 dvalA,
                       Td&                 dB,
                       Uh&                 hptrA,
                       Uh&                 hindA,
                       Th&                 hvalA,
                       Th&                 hB,
                       Th&                 hX,
                       const fs::path      testcase,
                       bool                test = true)
{
    if(CPU)
    {
        fs::path file;

        // read-in A
        file = testcase / "ptrA";
        read_matrix(file.string(), 1, n + 1, hptrA.data(), 1);
        file = testcase / "indA";
        read_matrix(file.string(), 1, nnzA, hindA.data(), 1);
        file = testcase / "valA";
        read_matrix(file.string(), 1, nnzA, hvalA.data(), 1);

        // read-in B
        file = testcase / "B_1";
        read_matrix(file.string(), n, 1, hB.data(), n);

        // get results (matrix X) if validation is required
        if(test)
        {
            // read-in X
            file = testcase / "X_1";
            read_matrix(file.string(), n, 1, hX.data(), n);
        }

        // change to base 1, if applicable
        hipsparseIndexBase_t indbase = hipsparseGetMatIndexBase(descrA);
        if(indbase == HIPSPARSE_INDEX_BASE_ONE)
        {
            for(rocblas_int i = 0; i <= n; i++)
                hptrA[0][i]++;

            for(rocblas_int i = 0; i < nnzA; i++)
                hindA[0][i]++;
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dptrA.transfer_from(hptrA));
        CHECK_HIP_ERROR(dindA.transfer_from(hindA));
        CHECK_HIP_ERROR(dvalA.transfer_from(hvalA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool HOST, typename T, typename S, typename Td, typename Ud, typename Th, typename Uh>
void csrlsvqr_getError(hipsolverSpHandle_t       handle,
                       const int                 n,
                       const int                 nnzA,
                       const hipsparseMatDescr_t descrA,
                       Ud&                       dptrA,
                       Ud&                       dindA,
                       Td&                       dvalA,
                       Td&                       dB,
                       const S                   tolerance,
                       const int                 reorder,
                       Td&                       dX,
                       Uh&                       hptrA,
                       Uh&                       hindA,
                       Th&                       hvalA,
                       Th&                       hB,
                       Th&                       hX,
                       Th&                       hXRes,
                       Uh&                       hSingularity,
                       double*                   max_err,
                       const fs::path            testcase)
{
    // input data initialization
    csrlsvqr_initData<true, true, T>(
        handle, n, nnzA, descrA, dptrA, dindA, dvalA, dB, hptrA, hindA, hvalA, hB, hX, testcase);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_csrlsvqr(HOST,
                                           handle,
                                           n,
                                           nnzA,
                                           descrA,
                                           dvalA.data(),
                                           dptrA.data(),
                                           dindA.data(),
                                           dB.data(),
                                           tolerance,
                                           reorder,
                                           dX.data(),
                                           hSingularity.data()));

    CHECK_HIP_ERROR(hXRes.transfer_from(dX));

    // compare computed results with original result
    double err;
    *max_err = 0;

    err      = norm_error('I', n, 1, n, hX[0], hXRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    // TODO: Add singular matrices and check info
    err = 0;
    EXPECT_EQ(hSingularity[0][0], -1);
    if(hSingularity[0][0] != -1)
        err++;
    *max_err += err;
}

template <bool HOST, typename T, typename S, typename Td, typename Ud, typename Th, typename Uh>
void csrlsvqr_getPerfData(hipsolverSpHandle_t       handle,
                          const int                 n,
                          const int                 nnzA,
                          const hipsparseMatDescr_t descrA,
                          Ud&                       dptrA,
                          Ud&                       dindA,
                          Td&                       dvalA,
                          Td&                       dB,
                          const S                   tolerance,
                          const int                 reorder,
                          Td&                       dX,
                          Uh&                       hptrA,
                          Uh&                       hindA,
                          Th&                       hvalA,
                          Th&                       hB,
                          Th&                       hX,
                          Uh&                       hSingularity,
                          double*                   gpu_time_used,
                          double*                   cpu_time_used,
                          const int                 hot_calls,
                          const bool                perf,
                          const fs::path            testcase)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution

    csrlsvqr_initData<true, false, T>(
        handle, n, nnzA, descrA, dptrA, dindA, dvalA, dB, hptrA, hindA, hvalA, hB, hX, testcase);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        csrlsvqr_initData<false, true, T>(handle,
                                          n,
                                          nnzA,
                                          descrA,
                                          dptrA,
                                          dindA,
                                          dvalA,
                                          dB,
                                          hptrA,
                                          hindA,
                                          hvalA,
                                          hB,
                                          hX,
                                          testcase);

        CHECK_ROCBLAS_ERROR(hipsolver_csrlsvqr(HOST,
                                               handle,
                                               n,
                                               nnzA,
                                               descrA,
                                               dvalA.data(),
                                               dptrA.data(),
                                               dindA.data(),
                                               dB.data(),
                                               tolerance,
                                               reorder,
                                               dX.data(),
                                               hSingularity.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        csrlsvqr_initData<false, true, T>(handle,
                                          n,
                                          nnzA,
                                          descrA,
                                          dptrA,
                                          dindA,
                                          dvalA,
                                          dB,
                                          hptrA,
                                          hindA,
                                          hvalA,
                                          hB,
                                          hX,
                                          testcase);

        start = get_time_us_sync(stream);
        hipsolver_csrlsvqr(HOST,
                           handle,
                           n,
                           nnzA,
                           descrA,
                           dvalA.data(),
                           dptrA.data(),
                           dindA.data(),
                           dB.data(),
                           tolerance,
                           reorder,
                           dX.data(),
                           hSingularity.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool HOST, typename T>
void testing_csrlsvqr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolverSp_local_handle handle;
    int                      n         = argus.get<int>("n");
    int                      nnzA      = argus.get<int>("nnzA");
    double                   tolerance = argus.get<double>("tolerance", 0);
    int                      reorder   = argus.get<int>("reorder", 0);
    int                      base1     = argus.get<int>("base1", 0);
    int                      hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzA < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_csrlsvqr(HOST,
                                                 handle,
                                                 n,
                                                 nnzA,
                                                 (hipsparseMatDescr_t) nullptr,
                                                 (T*)nullptr,
                                                 (int*)nullptr,
                                                 (int*)nullptr,
                                                 (T*)nullptr,
                                                 tolerance,
                                                 reorder,
                                                 (T*)nullptr,
                                                 (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // determine existing test case
    if(n > 0)
    {
        if(n <= 35)
            n = 20;
        else if(n <= 75)
            n = 50;
        else if(n <= 175)
            n = 100;
        else
            n = 250;
    }

    if(n <= 50) // small case
    {
        if(nnzA <= 80)
            nnzA = 60;
        else if(nnzA <= 120)
            nnzA = 100;
        else
            nnzA = 140;
    }
    else // large case
    {
        if(nnzA <= 400)
            nnzA = 300;
        else if(nnzA <= 600)
            nnzA = 500;
        else
            nnzA = 700;
    }

    // read/set corresponding nnzA
    fs::path testcase;
    if(n > 0)
    {
        fs::path    file;
        std::string folder
            = std::string("posmat_") + std::to_string(n) + "_" + std::to_string(nnzA);
        testcase = get_sparse_data_dir() / folder;

        file = testcase / "ptrA";
        read_last(file.string(), &nnzA);
    }

    // determine sizes
    size_t size_ptrA = size_t(n) + 1;
    size_t size_indA = size_t(nnzA);
    size_t size_valA = size_t(nnzA);
    size_t size_BX   = size_t(n);

    size_t size_BXres = 0;
    if(argus.unit_check || argus.norm_check)
        size_BXres = size_BX;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations (all cases)
    hipsparse_local_mat_descr descrA;
    hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
    if(base1 == 0)
        hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO);
    else
        hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ONE);

    host_strided_batch_vector<int> hptrA(size_ptrA, 1, size_ptrA, 1);
    host_strided_batch_vector<int> hindA(size_indA, 1, size_indA, 1);
    host_strided_batch_vector<T>   hvalA(size_valA, 1, size_valA, 1);
    host_strided_batch_vector<T>   hB(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T>   hX(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T>   hXRes(size_BXres, 1, size_BXres, 1);
    host_strided_batch_vector<int> hSingularity(1, 1, 1, 1);

    if(HOST)
    {
        // memory allocations
        host_strided_batch_vector<int> dptrA(size_ptrA, 1, size_ptrA, 1);
        host_strided_batch_vector<int> dindA(size_indA, 1, size_indA, 1);
        host_strided_batch_vector<T>   dvalA(size_valA, 1, size_valA, 1);
        host_strided_batch_vector<T>   dB(size_BX, 1, size_BX, 1);
        host_strided_batch_vector<T>   dX(size_BX, 1, size_BX, 1);
        CHECK_HIP_ERROR(dptrA.memcheck());
        if(size_indA)
            CHECK_HIP_ERROR(dindA.memcheck());
        if(size_valA)
            CHECK_HIP_ERROR(dvalA.memcheck());
        if(size_BX)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_BX)
            CHECK_HIP_ERROR(dX.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            csrlsvqr_getError<HOST, T>(handle,
                                       n,
                                       nnzA,
                                       descrA,
                                       dptrA,
                                       dindA,
                                       dvalA,
                                       dB,
                                       tolerance,
                                       reorder,
                                       dX,
                                       hptrA,
                                       hindA,
                                       hvalA,
                                       hB,
                                       hX,
                                       hXRes,
                                       hSingularity,
                                       &max_error,
                                       testcase);

        // collect performance data
        if(argus.timing)
            csrlsvqr_getPerfData<HOST, T>(handle,
                                          n,
                                          nnzA,
                                          descrA,
                                          dptrA,
                                          dindA,
                                          dvalA,
                                          dB,
                                          tolerance,
                                          reorder,
                                          dX,
                                          hptrA,
                                          hindA,
                                          hvalA,
                                          hB,
                                          hX,
                                          hSingularity,
                                          &gpu_time_used,
                                          &cpu_time_used,
                                          hot_calls,
                                          argus.perf,
                                          testcase);
    }

    else
    {
        // memory allocations
        device_strided_batch_vector<int> dptrA(size_ptrA, 1, size_ptrA, 1);
        device_strided_batch_vector<int> dindA(size_indA, 1, size_indA, 1);
        device_strided_batch_vector<T>   dvalA(size_valA, 1, size_valA, 1);
        device_strided_batch_vector<T>   dB(size_BX, 1, size_BX, 1);
        device_strided_batch_vector<T>   dX(size_BX, 1, size_BX, 1);
        CHECK_HIP_ERROR(dptrA.memcheck());
        if(size_indA)
            CHECK_HIP_ERROR(dindA.memcheck());
        if(size_valA)
            CHECK_HIP_ERROR(dvalA.memcheck());
        if(size_BX)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_BX)
            CHECK_HIP_ERROR(dX.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            csrlsvqr_getError<HOST, T>(handle,
                                       n,
                                       nnzA,
                                       descrA,
                                       dptrA,
                                       dindA,
                                       dvalA,
                                       dB,
                                       tolerance,
                                       reorder,
                                       dX,
                                       hptrA,
                                       hindA,
                                       hvalA,
                                       hB,
                                       hX,
                                       hXRes,
                                       hSingularity,
                                       &max_error,
                                       testcase);

        // collect performance data
        if(argus.timing)
            csrlsvqr_getPerfData<HOST, T>(handle,
                                          n,
                                          nnzA,
                                          descrA,
                                          dptrA,
                                          dindA,
                                          dvalA,
                                          dB,
                                          tolerance,
                                          reorder,
                                          dX,
                                          hptrA,
                                          hindA,
                                          hvalA,
                                          hB,
                                          hX,
                                          hSingularity,
                                          &gpu_time_used,
                                          &cpu_time_used,
                                          hot_calls,
                                          argus.perf,
                                          testcase);
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
            rocsolver_bench_output("n", "nnzA");
            rocsolver_bench_output(n, nnzA);

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
