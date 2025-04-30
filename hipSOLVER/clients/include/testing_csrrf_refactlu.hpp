/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T>
void csrrf_refactlu_checkBadArgs(hipsolverRfHandle_t handle,
                                 const int           n,
                                 const int           nnzA,
                                 int*                ptrA,
                                 int*                indA,
                                 T                   valA,
                                 const int           nnzT,
                                 int*                ptrT,
                                 int*                indT,
                                 T                   valT,
                                 int*                pivP,
                                 int*                pivQ)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolverRfRefactor(nullptr), HIPSOLVER_STATUS_NOT_INITIALIZED);
}

template <typename T>
void testing_csrrf_refactlu_bad_arg()
{
    // safe arguments
    hipsolverRf_local_handle handle;
    int                      n    = 1;
    int                      nnzA = 1;
    int                      nnzT = 1;

    // memory allocations
    device_strided_batch_vector<int> ptrA(1, 1, 1, 1);
    device_strided_batch_vector<int> indA(1, 1, 1, 1);
    device_strided_batch_vector<T>   valA(1, 1, 1, 1);
    device_strided_batch_vector<int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T>   valT(1, 1, 1, 1);
    device_strided_batch_vector<int> pivP(1, 1, 1, 1);
    device_strided_batch_vector<int> pivQ(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrA.memcheck());
    CHECK_HIP_ERROR(indA.memcheck());
    CHECK_HIP_ERROR(valA.memcheck());
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());
    CHECK_HIP_ERROR(pivP.memcheck());
    CHECK_HIP_ERROR(pivQ.memcheck());

    // check bad arguments
    csrrf_refactlu_checkBadArgs(handle,
                                n,
                                nnzA,
                                ptrA.data(),
                                indA.data(),
                                valA.data(),
                                nnzT,
                                ptrT.data(),
                                indT.data(),
                                valT.data(),
                                pivP.data(),
                                pivQ.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_refactlu_initData(hipsolverRfHandle_t handle,
                             const int           n,
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
                             Uh&                 hptrA,
                             Uh&                 hindA,
                             Th&                 hvalA,
                             Uh&                 hptrL,
                             Uh&                 hindL,
                             Th&                 hvalL,
                             Uh&                 hptrU,
                             Uh&                 hindU,
                             Th&                 hvalU,
                             Th&                 hvalT,
                             Uh&                 hpivP,
                             Uh&                 hpivQ,
                             const fs::path      testcase)
{
    int nnzT = nnzL - n + nnzU;

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

        // read-in T
        file = testcase / "valT";
        read_matrix(file.string(), 1, nnzT, hvalT.data(), 1);

        // read-in P
        file = testcase / "P";
        read_matrix(file.string(), 1, n, hpivP.data(), 1);

        // read-in Q
        file = testcase / "Q";
        read_matrix(file.string(), 1, n, hpivQ.data(), 1);
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
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_refactlu_getError(hipsolverRfHandle_t handle,
                             const int           n,
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
                             Uh&                 hptrA,
                             Uh&                 hindA,
                             Th&                 hvalA,
                             Uh&                 hptrL,
                             Uh&                 hindL,
                             Th&                 hvalL,
                             Uh&                 hptrU,
                             Uh&                 hindU,
                             Th&                 hvalU,
                             Th&                 hvalT,
                             Uh&                 hpivP,
                             Uh&                 hpivQ,
                             double*             max_err,
                             const fs::path      testcase)
{
    int  nnzLRes;
    int* hptrLRes;
    int* hindLRes;
    T*   hvalLRes;

    int  nnzURes;
    int* hptrURes;
    int* hindURes;
    T*   hvalURes;

    int  nnzTRes;
    int* hptrTRes;
    int* hindTRes;
    T*   hvalTRes;

    // input data initialization
    csrrf_refactlu_initData<true, true, T>(handle,
                                           n,
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
                                           hptrA,
                                           hindA,
                                           hvalA,
                                           hptrL,
                                           hindL,
                                           hvalL,
                                           hptrU,
                                           hindU,
                                           hvalU,
                                           hvalT,
                                           hpivP,
                                           hpivQ,
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

    // compare computed bundled factors with original result
    CHECK_ROCBLAS_ERROR(
        hipsolverRfExtractBundledFactorsHost(handle, &nnzTRes, &hptrTRes, &hindTRes, &hvalTRes));

    EXPECT_EQ(nnzTRes, nnzL - n + nnzU) << "where b = " << 0;
    if(nnzTRes == nnzL - n + nnzU)
        *max_err = norm_error('F', 1, nnzTRes, 1, hvalT[0], hvalTRes);
    else
        *max_err = 1;

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // compare computed split factors with original result
    CHECK_ROCBLAS_ERROR(hipsolverRfExtractSplitFactorsHost(handle,
                                                           &nnzLRes,
                                                           &hptrLRes,
                                                           &hindLRes,
                                                           &hvalLRes,
                                                           &nnzURes,
                                                           &hptrURes,
                                                           &hindURes,
                                                           &hvalURes));

    EXPECT_EQ(nnzLRes, nnzL) << "where b = " << 0;
    EXPECT_EQ(nnzURes, nnzU) << "where b = " << 0;
    if(nnzLRes == nnzL && nnzURes == nnzU)
    {
        double errorL, errorU;
        errorL   = norm_error('F', 1, nnzLRes, 1, hvalL[0], hvalLRes);
        errorU   = norm_error('F', 1, nnzURes, 1, hvalU[0], hvalURes);
        *max_err = max({*max_err, errorL, errorU});
    }
    else
        *max_err = 1;
#endif
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_refactlu_getPerfData(hipsolverRfHandle_t handle,
                                const int           n,
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
                                Uh&                 hptrA,
                                Uh&                 hindA,
                                Th&                 hvalA,
                                Uh&                 hptrL,
                                Uh&                 hindL,
                                Th&                 hvalL,
                                Uh&                 hptrU,
                                Uh&                 hindU,
                                Th&                 hvalU,
                                Th&                 hvalT,
                                Uh&                 hpivP,
                                Uh&                 hpivQ,
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
void testing_csrrf_refactlu(Arguments& argus)
{
    // get arguments
    hipsolverRf_local_handle handle;
    int                      n         = argus.get<int>("n");
    int                      nnzA      = argus.get<int>("nnzA");
    int                      hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzA < 0);
    if(invalid_size)
    {
        // EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(
        //                           handle, n, nnzA, (int*)nullptr, (int*)nullptr,
        //                           (T*)nullptr, nnzT, (int*)nullptr, (int*)nullptr,
        //                           (T*)nullptr, (int*)nullptr, (int*)nullptr, rfinfo),
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
    size_t size_valT = size_t(nnzL) - n + nnzU;
    size_t size_pivP = size_t(n);
    size_t size_pivQ = size_t(n);

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
    host_strided_batch_vector<T>   hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<int> hpivP(size_pivP, 1, size_pivP, 1);
    host_strided_batch_vector<int> hpivQ(size_pivQ, 1, size_pivQ, 1);

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

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_refactlu_getError<T>(handle,
                                   n,
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
                                   hptrA,
                                   hindA,
                                   hvalA,
                                   hptrL,
                                   hindL,
                                   hvalL,
                                   hptrU,
                                   hindU,
                                   hvalU,
                                   hvalT,
                                   hpivP,
                                   hpivQ,
                                   &max_error,
                                   testcase);

    // collect performance data
    if(argus.timing)
        csrrf_refactlu_getPerfData<T>(handle,
                                      n,
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
                                      hptrA,
                                      hindA,
                                      hvalA,
                                      hptrL,
                                      hindL,
                                      hvalL,
                                      hptrU,
                                      hindU,
                                      hvalU,
                                      hvalT,
                                      hpivP,
                                      hpivQ,
                                      &gpu_time_used,
                                      &cpu_time_used,
                                      hot_calls,
                                      argus.perf,
                                      testcase);

    // validate results for rocsolver-test
    // using n * machine precision for tolerance
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
