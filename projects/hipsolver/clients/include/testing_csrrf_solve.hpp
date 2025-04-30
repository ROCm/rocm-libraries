/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"

template <typename T>
void csrrf_solve_checkBadArgs(hipsolverRfHandle_t handle,
                              const int           n,
                              const int           nrhs,
                              const int           nnzT,
                              int*                ptrT,
                              int*                indT,
                              T                   valT,
                              int*                pivP,
                              int*                pivQ,
                              T                   work,
                              const int           ldw,
                              T                   B,
                              const int           ldb)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolverRfSolve(nullptr, pivP, pivQ, nrhs, work, ldw, B, ldb),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);
}

template <typename T>
void testing_csrrf_solve_bad_arg()
{
    // safe arguments
    hipsolverRf_local_handle handle;
    int                      n    = 1;
    int                      nrhs = 1;
    int                      nnzT = 1;
    int                      ldb  = 1;

    // memory allocations
    device_strided_batch_vector<int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T>   valT(1, 1, 1, 1);
    device_strided_batch_vector<int> pivP(1, 1, 1, 1);
    device_strided_batch_vector<int> pivQ(1, 1, 1, 1);
    device_strided_batch_vector<T>   B(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());
    CHECK_HIP_ERROR(pivP.memcheck());
    CHECK_HIP_ERROR(pivQ.memcheck());
    CHECK_HIP_ERROR(B.memcheck());

    int                            size_W = n * nrhs;
    device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    csrrf_solve_checkBadArgs(handle,
                             n,
                             nrhs,
                             nnzT,
                             ptrT.data(),
                             indT.data(),
                             valT.data(),
                             pivP.data(),
                             pivQ.data(),
                             dWork.data(),
                             n,
                             B.data(),
                             ldb);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_solve_initData(hipsolverRfHandle_t handle,
                          const int           n,
                          const int           nrhs,
                          const int           nnzA,
                          Ud&                 dptrA,
                          Ud&                 dindA,
                          Td&                 dvalA,
                          const int           nnzL,
                          Ud&                 dptrL,
                          Ud&                 dindL,
                          Td&                 dvalL,
                          const int           nnzU,
                          Ud&                 dptrU,
                          Ud&                 dindU,
                          Td&                 dvalU,
                          Ud&                 dpivP,
                          Ud&                 dpivQ,
                          Td&                 dB,
                          const int           ldb,
                          Uh&                 hptrA,
                          Uh&                 hindA,
                          Th&                 hvalA,
                          Uh&                 hptrL,
                          Uh&                 hindL,
                          Th&                 hvalL,
                          Uh&                 hptrU,
                          Uh&                 hindU,
                          Th&                 hvalU,
                          Uh&                 hpivP,
                          Uh&                 hpivQ,
                          Th&                 hB,
                          Th&                 hX,
                          const fs::path      testcase,
                          bool                test = true)
{
    if(CPU)
    {
        fs::path    file;
        std::string filename;

        // read-in A
        file = testcase / "ptrA";
        read_matrix(file.string(), 1, n + 1, hptrA.data(), 1);
        file = testcase / "indA";
        read_matrix(file.string(), 1, nnzA, hindA.data(), 1);
        file = testcase / "valA";
        read_matrix(file.string(), 1, nnzA, hvalA.data(), 1);

        // read-in L
        file = testcase / "ptrL";
        read_matrix(file.string(), 1, n + 1, hptrL.data(), 1);
        file = testcase / "indL";
        read_matrix(file.string(), 1, nnzL, hindL.data(), 1);
        file = testcase / "valL";
        read_matrix(file.string(), 1, nnzL, hvalL.data(), 1);

        // read-in U
        file = testcase / "ptrU";
        read_matrix(file.string(), 1, n + 1, hptrU.data(), 1);
        file = testcase / "indU";
        read_matrix(file.string(), 1, nnzU, hindU.data(), 1);
        file = testcase / "valU";
        read_matrix(file.string(), 1, nnzU, hvalU.data(), 1);

        // read-in P
        file = testcase / "P";
        read_matrix(file.string(), 1, n, hpivP.data(), 1);

        // read-in Q
        file = testcase / "Q";
        read_matrix(file.string(), 1, n, hpivQ.data(), 1);

        // read-in B
        filename = std::string("B_") + std::to_string(nrhs);
        file     = testcase / filename;
        read_matrix(file.string(), n, nrhs, hB.data(), ldb);

        // get results (matrix X) if validation is required
        if(test)
        {
            // read-in X
            filename = std::string("X_") + std::to_string(nrhs);
            file     = testcase / filename;
            read_matrix(file.string(), n, nrhs, hX.data(), ldb);
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dptrA.transfer_from(hptrA));
        CHECK_HIP_ERROR(dindA.transfer_from(hindA));
        CHECK_HIP_ERROR(dvalA.transfer_from(hvalA));
        CHECK_HIP_ERROR(dptrL.transfer_from(hptrL));
        CHECK_HIP_ERROR(dindL.transfer_from(hindL));
        CHECK_HIP_ERROR(dvalL.transfer_from(hvalL));
        CHECK_HIP_ERROR(dptrU.transfer_from(hptrU));
        CHECK_HIP_ERROR(dindU.transfer_from(hindU));
        CHECK_HIP_ERROR(dvalU.transfer_from(hvalU));
        CHECK_HIP_ERROR(dpivP.transfer_from(hpivP));
        CHECK_HIP_ERROR(dpivQ.transfer_from(hpivQ));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_solve_getError(hipsolverRfHandle_t handle,
                          const int           n,
                          const int           nrhs,
                          const int           nnzA,
                          Ud&                 dptrA,
                          Ud&                 dindA,
                          Td&                 dvalA,
                          const int           nnzL,
                          Ud&                 dptrL,
                          Ud&                 dindL,
                          Td&                 dvalL,
                          const int           nnzU,
                          Ud&                 dptrU,
                          Ud&                 dindU,
                          Td&                 dvalU,
                          Ud&                 dpivP,
                          Ud&                 dpivQ,
                          Td&                 dB,
                          const int           ldb,
                          Td&                 dWork,
                          Uh&                 hptrA,
                          Uh&                 hindA,
                          Th&                 hvalA,
                          Uh&                 hptrL,
                          Uh&                 hindL,
                          Th&                 hvalL,
                          Uh&                 hptrU,
                          Uh&                 hindU,
                          Th&                 hvalU,
                          Uh&                 hpivP,
                          Uh&                 hpivQ,
                          Th&                 hB,
                          Th&                 hX,
                          Th&                 hXres,
                          double*             max_err,
                          const fs::path      testcase)
{
    // input data initialization
    csrrf_solve_initData<true, true, T>(handle,
                                        n,
                                        nrhs,
                                        nnzA,
                                        dptrA,
                                        dindA,
                                        dvalA,
                                        nnzL,
                                        dptrL,
                                        dindL,
                                        dvalL,
                                        nnzU,
                                        dptrU,
                                        dindU,
                                        dvalU,
                                        dpivP,
                                        dpivQ,
                                        dB,
                                        ldb,
                                        hptrA,
                                        hindA,
                                        hvalA,
                                        hptrL,
                                        hindL,
                                        hvalL,
                                        hptrU,
                                        hindU,
                                        hvalU,
                                        hpivP,
                                        hpivQ,
                                        hB,
                                        hX,
                                        testcase);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolverRfSetupDevice(n,
                                               nnzA,
                                               dptrA.data(),
                                               dindA.data(),
                                               dvalA.data(),
                                               nnzL,
                                               dptrL.data(),
                                               dindL.data(),
                                               dvalL.data(),
                                               nnzU,
                                               dptrU.data(),
                                               dindU.data(),
                                               dvalU.data(),
                                               dpivP.data(),
                                               dpivQ.data(),
                                               handle));

    CHECK_ROCBLAS_ERROR(hipsolverRfAnalyze(handle));

    CHECK_ROCBLAS_ERROR(hipsolverRfResetValues(
        n, nnzA, dptrA.data(), dindA.data(), dvalA.data(), dpivP.data(), dpivQ.data(), handle));

    CHECK_ROCBLAS_ERROR(hipsolverRfRefactor(handle));

    CHECK_ROCBLAS_ERROR(
        hipsolverRfSolve(handle, dpivP, dpivQ, nrhs, dWork.data(), n, dB.data(), ldb));

    CHECK_HIP_ERROR(hXres.transfer_from(dB));

    // compare computed results with original result
    *max_err = norm_error('I', n, nrhs, ldb, hX[0], hXres[0]);
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_solve_getPerfData(hipsolverRfHandle_t handle,
                             const int           n,
                             const int           nrhs,
                             const int           nnzA,
                             Ud&                 dptrA,
                             Ud&                 dindA,
                             Td&                 dvalA,
                             const int           nnzL,
                             Ud&                 dptrL,
                             Ud&                 dindL,
                             Td&                 dvalL,
                             const int           nnzU,
                             Ud&                 dptrU,
                             Ud&                 dindU,
                             Td&                 dvalU,
                             Ud&                 dpivP,
                             Ud&                 dpivQ,
                             Td&                 dB,
                             const int           ldb,
                             Td&                 dWork,
                             Uh&                 hptrA,
                             Uh&                 hindA,
                             Th&                 hvalA,
                             Uh&                 hptrL,
                             Uh&                 hindL,
                             Th&                 hvalL,
                             Uh&                 hptrU,
                             Uh&                 hindU,
                             Th&                 hvalU,
                             Uh&                 hpivP,
                             Uh&                 hpivQ,
                             Th&                 hB,
                             Th&                 hX,
                             double*             gpu_time_used,
                             double*             cpu_time_used,
                             const int           hot_calls,
                             const bool          perf,
                             const fs::path      testcase)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution
    *gpu_time_used = nan(""); // no timing on gpu-lapack execution
}

template <typename T>
void testing_csrrf_solve(Arguments& argus)
{
    // get arguments
    hipsolverRf_local_handle handle;
    int                      n         = argus.get<int>("n");
    int                      nrhs      = argus.get<int>("nrhs", n);
    int                      nnzA      = argus.get<int>("nnzA");
    int                      ldb       = argus.get<int>("ldb", n);
    int                      hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || nnzA < 0 || ldb < n);
    if(invalid_size)
    {
        // EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzA, (int*)nullptr,
        //                                             (int*)nullptr, (T*)nullptr,
        //                                             (int*)nullptr, (int*)nullptr,
        //                                             rfinfo, (T*)nullptr, ldb),
        //                       rocblas_status_invalid_size);

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

    // read/set corresponding nnzL and nnzU
    int      nnzL, nnzU;
    fs::path testcase;
    if(n > 0)
    {
        fs::path    file;
        std::string folder = std::string("mat_") + std::to_string(n) + "_" + std::to_string(nnzA);
        testcase           = get_sparse_data_dir() / folder;

        file = testcase / "ptrA";
        read_last(file.string(), &nnzA);

        file = testcase / "ptrL";
        read_last(file.string(), &nnzL);

        file = testcase / "ptrU";
        read_last(file.string(), &nnzU);
    }

    // determine existing right-hand-side
    if(nrhs > 0)
    {
        nrhs = 1;
    }

    // determine sizes
    size_t size_ptrA = size_t(n) + 1;
    size_t size_indA = size_t(nnzA);
    size_t size_valA = size_t(nnzA);
    size_t size_ptrL = size_t(n) + 1;
    size_t size_indL = size_t(nnzL);
    size_t size_valL = size_t(nnzL);
    size_t size_ptrU = size_t(n) + 1;
    size_t size_indU = size_t(nnzU);
    size_t size_valU = size_t(nnzU);
    size_t size_pivP = size_t(n);
    size_t size_pivQ = size_t(n);
    size_t size_BX   = size_t(ldb) * nrhs;

    size_t size_BXres = 0;
    if(argus.unit_check || argus.norm_check)
        size_BXres = size_BX;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<int> hptrA(size_ptrA, 1, size_ptrA, 1);
    host_strided_batch_vector<int> hindA(size_indA, 1, size_indA, 1);
    host_strided_batch_vector<T>   hvalA(size_valA, 1, size_valA, 1);
    host_strided_batch_vector<int> hptrL(size_ptrL, 1, size_ptrL, 1);
    host_strided_batch_vector<int> hindL(size_indL, 1, size_indL, 1);
    host_strided_batch_vector<T>   hvalL(size_valL, 1, size_valL, 1);
    host_strided_batch_vector<int> hptrU(size_ptrU, 1, size_ptrU, 1);
    host_strided_batch_vector<int> hindU(size_indU, 1, size_indU, 1);
    host_strided_batch_vector<T>   hvalU(size_valU, 1, size_valU, 1);
    host_strided_batch_vector<int> hpivP(size_pivP, 1, size_pivP, 1);
    host_strided_batch_vector<int> hpivQ(size_pivQ, 1, size_pivQ, 1);
    host_strided_batch_vector<T>   hB(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T>   hX(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T>   hXres(size_BXres, 1, size_BXres, 1);

    device_strided_batch_vector<int> dptrA(size_ptrA, 1, size_ptrA, 1);
    device_strided_batch_vector<int> dindA(size_indA, 1, size_indA, 1);
    device_strided_batch_vector<T>   dvalA(size_valA, 1, size_valA, 1);
    device_strided_batch_vector<int> dptrL(size_ptrL, 1, size_ptrL, 1);
    device_strided_batch_vector<int> dindL(size_indL, 1, size_indL, 1);
    device_strided_batch_vector<T>   dvalL(size_valL, 1, size_valL, 1);
    device_strided_batch_vector<int> dptrU(size_ptrU, 1, size_ptrU, 1);
    device_strided_batch_vector<int> dindU(size_indU, 1, size_indU, 1);
    device_strided_batch_vector<T>   dvalU(size_valU, 1, size_valU, 1);
    device_strided_batch_vector<int> dpivP(size_pivP, 1, size_pivP, 1);
    device_strided_batch_vector<int> dpivQ(size_pivQ, 1, size_pivQ, 1);
    device_strided_batch_vector<T>   dB(size_BX, 1, size_BX, 1);
    CHECK_HIP_ERROR(dptrA.memcheck());
    CHECK_HIP_ERROR(dptrL.memcheck());
    CHECK_HIP_ERROR(dptrU.memcheck());
    if(size_indA)
        CHECK_HIP_ERROR(dindA.memcheck());
    if(size_valA)
        CHECK_HIP_ERROR(dvalA.memcheck());
    if(size_indL)
        CHECK_HIP_ERROR(dindL.memcheck());
    if(size_valL)
        CHECK_HIP_ERROR(dvalL.memcheck());
    if(size_indU)
        CHECK_HIP_ERROR(dindU.memcheck());
    if(size_valU)
        CHECK_HIP_ERROR(dvalU.memcheck());
    if(size_pivP)
        CHECK_HIP_ERROR(dpivP.memcheck());
    if(size_pivQ)
        CHECK_HIP_ERROR(dpivQ.memcheck());
    if(size_BX)
        CHECK_HIP_ERROR(dB.memcheck());

    int                            size_W = n * nrhs;
    device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_solve_getError<T>(handle,
                                n,
                                nrhs,
                                nnzA,
                                dptrA,
                                dindA,
                                dvalA,
                                nnzL,
                                dptrL,
                                dindL,
                                dvalL,
                                nnzU,
                                dptrU,
                                dindU,
                                dvalU,
                                dpivP,
                                dpivQ,
                                dB,
                                ldb,
                                dWork,
                                hptrA,
                                hindA,
                                hvalA,
                                hptrL,
                                hindL,
                                hvalL,
                                hptrU,
                                hindU,
                                hvalU,
                                hpivP,
                                hpivQ,
                                hB,
                                hX,
                                hXres,
                                &max_error,
                                testcase);

    // collect performance data
    if(argus.timing)
        csrrf_solve_getPerfData<T>(handle,
                                   n,
                                   nrhs,
                                   nnzA,
                                   dptrA,
                                   dindA,
                                   dvalA,
                                   nnzL,
                                   dptrL,
                                   dindL,
                                   dvalL,
                                   nnzU,
                                   dptrU,
                                   dindU,
                                   dvalU,
                                   dpivP,
                                   dpivQ,
                                   dB,
                                   ldb,
                                   dWork,
                                   hptrA,
                                   hindA,
                                   hvalA,
                                   hptrL,
                                   hindL,
                                   hvalL,
                                   hptrU,
                                   hindU,
                                   hvalU,
                                   hpivP,
                                   hpivQ,
                                   hB,
                                   hX,
                                   &gpu_time_used,
                                   &cpu_time_used,
                                   hot_calls,
                                   argus.perf,
                                   testcase);

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
            rocsolver_bench_output("n", "nrhs", "nnzA", "ldb");
            rocsolver_bench_output(n, nrhs, nnzA, ldb);

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
