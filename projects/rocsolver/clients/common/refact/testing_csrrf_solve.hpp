/* **************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <typename T>
void csrrf_solve_checkBadArgs(rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              const rocblas_int nnzT,
                              rocblas_int* ptrT,
                              rocblas_int* indT,
                              T valT,
                              rocblas_int* pivP,
                              rocblas_int* pivQ,
                              T B,
                              const rocblas_int ldb,
                              rocsolver_rfinfo rfinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_csrrf_solve(nullptr, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb, rfinfo),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, (rocblas_int*)nullptr, indT,
                                                valT, pivP, pivQ, B, ldb, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, ptrT, (rocblas_int*)nullptr,
                                                valT, pivP, pivQ, B, ldb, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, (T) nullptr,
                                                pivP, pivQ, B, ldb, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT,
                                                (rocblas_int*)nullptr, pivQ, B, ldb, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP,
                                                (rocblas_int*)nullptr, B, ldb, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ,
                                                (T) nullptr, ldb, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_csrrf_solve(handle, n, nrhs, nnzT, ptrT, indT, valT, pivP, pivQ, B, ldb, nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, 0, nrhs, nnzT, ptrT, indT, valT,
                                                (rocblas_int*)nullptr, (rocblas_int*)nullptr, B,
                                                ldb, rfinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, 0, nnzT, ptrT, indT, valT, pivP, pivQ,
                                                (T) nullptr, ldb, rfinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, 0, 0, nnzT, ptrT, indT, valT,
                                                (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                (T) nullptr, ldb, rfinfo),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    // N/A
}

template <typename T>
void testing_csrrf_solve_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = 1;
    rocblas_int nrhs = 1;
    rocblas_int nnzT = 1;
    rocblas_int ldb = 1;

    // memory allocations
    device_strided_batch_vector<rocblas_int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T> valT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> pivP(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> pivQ(1, 1, 1, 1);
    device_strided_batch_vector<T> B(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());
    CHECK_HIP_ERROR(pivP.memcheck());
    CHECK_HIP_ERROR(pivQ.memcheck());
    CHECK_HIP_ERROR(B.memcheck());

    // check bad arguments
    csrrf_solve_checkBadArgs(handle, n, nrhs, nnzT, ptrT.data(), indT.data(), valT.data(),
                             pivP.data(), pivQ.data(), B.data(), ldb, rfinfo);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_solve_initData(rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nrhs,
                          const rocblas_int nnzT,
                          Ud& dptrT,
                          Ud& dindT,
                          Td& dvalT,
                          Ud& dpivP,
                          Ud& dpivQ,
                          Td& dB,
                          const rocblas_int ldb,
                          Uh& hptrT,
                          Uh& hindT,
                          Th& hvalT,
                          Uh& hpivP,
                          Uh& hpivQ,
                          Th& hB,
                          Th& hX,
                          const fs::path testcase,
                          const rocsolver_rfinfo_mode mode,
                          bool test = true)
{
    if(CPU)
    {
        fs::path file;

        // read-in T
        file = testcase / "ptrT";
        read_matrix(file.string(), 1, n + 1, hptrT.data(), 1);
        file = testcase / "indT";
        read_matrix(file.string(), 1, nnzT, hindT.data(), 1);
        file = testcase / "valT";
        read_matrix(file.string(), 1, nnzT, hvalT.data(), 1);

        // read-in P
        if(mode == rocsolver_rfinfo_mode_lu)
        {
            file = testcase / "P";
            read_matrix(file.string(), 1, n, hpivP.data(), 1);
        }

        // read-in Q
        file = testcase / "Q";
        read_matrix(file.string(), 1, n, hpivQ.data(), 1);

        // read-in B
        file = testcase / fs::path(fmt::format("B_{}", nrhs));
        read_matrix(file.string(), n, nrhs, hB.data(), ldb);

        // get results (matrix X) if validation is required
        if(test)
        {
            // read-in X
            file = testcase / fs::path(fmt::format("X_{}", nrhs));
            read_matrix(file.string(), n, nrhs, hX.data(), ldb);
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dptrT.transfer_from(hptrT));
        CHECK_HIP_ERROR(dindT.transfer_from(hindT));
        CHECK_HIP_ERROR(dvalT.transfer_from(hvalT));
        CHECK_HIP_ERROR(dB.transfer_from(hB));

        if(mode == rocsolver_rfinfo_mode_lu)
            CHECK_HIP_ERROR(dpivP.transfer_from(hpivP));
        else
            CHECK_HIP_ERROR(dpivP.transfer_from(hpivQ));
        CHECK_HIP_ERROR(dpivQ.transfer_from(hpivQ));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_solve_getError(rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nrhs,
                          const rocblas_int nnzT,
                          Ud& dptrT,
                          Ud& dindT,
                          Td& dvalT,
                          Ud& dpivP,
                          Ud& dpivQ,
                          Td& dB,
                          const rocblas_int ldb,
                          rocsolver_rfinfo rfinfo,
                          Uh& hptrT,
                          Uh& hindT,
                          Th& hvalT,
                          Uh& hpivP,
                          Uh& hpivQ,
                          Th& hB,
                          Th& hX,
                          Th& hXres,
                          double* max_err,
                          const fs::path testcase,
                          const rocsolver_rfinfo_mode mode)
{
    // input data initialization
    csrrf_solve_initData<true, true, T>(handle, n, nrhs, nnzT, dptrT, dindT, dvalT, dpivP, dpivQ,
                                        dB, ldb, hptrT, hindT, hvalT, hpivP, hpivQ, hB, hX,
                                        testcase, mode);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(
        handle, n, nrhs, nnzT, dptrT.data(), dindT.data(), dvalT.data(), nnzT, dptrT.data(),
        dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), dB.data(), ldb, rfinfo));

    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, dptrT.data(), dindT.data(),
                                              dvalT.data(), dpivP.data(), dpivQ.data(), dB.data(),
                                              ldb, rfinfo));

    CHECK_HIP_ERROR(hXres.transfer_from(dB));

    // compare computed results with original result
    *max_err = norm_error('I', n, nrhs, ldb, hX[0], hXres[0]);
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_solve_getPerfData(rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int nrhs,
                             const rocblas_int nnzT,
                             Ud& dptrT,
                             Ud& dindT,
                             Td& dvalT,
                             Ud& dpivP,
                             Ud& dpivQ,
                             Td& dB,
                             const rocblas_int ldb,
                             rocsolver_rfinfo rfinfo,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ,
                             Th& hB,
                             Th& hX,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const fs::path testcase,
                             const rocsolver_rfinfo_mode mode)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution

    csrrf_solve_initData<true, true, T>(handle, n, nrhs, nnzT, dptrT, dindT, dvalT, dpivP, dpivQ,
                                        dB, ldb, hptrT, hindT, hvalT, hpivP, hpivQ, hB, hX,
                                        testcase, mode, false);

    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(
        handle, n, nrhs, nnzT, dptrT.data(), dindT.data(), dvalT.data(), nnzT, dptrT.data(),
        dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), dB.data(), ldb, rfinfo));

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        csrrf_solve_initData<false, true, T>(handle, n, nrhs, nnzT, dptrT, dindT, dvalT, dpivP,
                                             dpivQ, dB, ldb, hptrT, hindT, hvalT, hpivP, hpivQ, hB,
                                             hX, testcase, mode, false);

        CHECK_ROCBLAS_ERROR(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, dptrT.data(), dindT.data(),
                                                  dvalT.data(), dpivP.data(), dpivQ.data(),
                                                  dB.data(), ldb, rfinfo));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        csrrf_solve_initData<false, true, T>(handle, n, nrhs, nnzT, dptrT, dindT, dvalT, dpivP,
                                             dpivQ, dB, ldb, hptrT, hindT, hvalT, hpivP, hpivQ, hB,
                                             hX, testcase, mode, false);

        start = get_time_us_sync(stream);
        rocsolver_csrrf_solve(handle, n, nrhs, nnzT, dptrT.data(), dindT.data(), dvalT.data(),
                              dpivP.data(), dpivQ.data(), dB.data(), ldb, rfinfo);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_csrrf_solve(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs", n);
    rocblas_int nnzT = argus.get<rocblas_int>("nnzT");
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    char modeC = argus.get<char>("rfinfo_mode", '1');
    rocblas_int hot_calls = argus.iters;

    rocsolver_rfinfo_mode mode = char2rocsolver_rfinfo_mode(modeC);
    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(rfinfo, mode));

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || nnzT < 0 || ldb < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, (rocblas_int*)nullptr,
                                                    (rocblas_int*)nullptr, (T*)nullptr,
                                                    (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                    (T*)nullptr, ldb, rfinfo),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // determine existing test case
    rocblas_int nnzA = nnzT;
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

    // read/set corresponding nnzT
    fs::path testcase;
    if(n > 0)
    {
        std::string matname;
        if(mode == rocsolver_rfinfo_mode_lu)
            matname = fmt::format("mat_{}_{}", n, nnzA);
        else
            matname = fmt::format("posmat_{}_{}", n, nnzA);

        testcase = get_sparse_data_dir() / fs::path(matname);
        fs::path file = testcase / "ptrT";
        read_last(file.string(), &nnzT);
    }

    // determine existing right-hand-side
    if(nrhs > 0)
    {
        if(nrhs <= 5)
            nrhs = 1;
        else if(nrhs <= 20)
            nrhs = 10;
        else
            nrhs = 30;
    }

    // memory size query if necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_csrrf_solve(
            handle, n, nrhs, nnzT, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr,
            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, ldb, rfinfo));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // determine sizes
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);
    size_t size_pivP = size_t(n);
    size_t size_pivQ = size_t(n);
    size_t size_BX = size_t(ldb) * nrhs;

    size_t size_BXres = 0;
    if(argus.unit_check || argus.norm_check)
        size_BXres = size_BX;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<rocblas_int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<rocblas_int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<T> hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<rocblas_int> hpivP(size_pivP, 1, size_pivP, 1);
    host_strided_batch_vector<rocblas_int> hpivQ(size_pivQ, 1, size_pivQ, 1);
    host_strided_batch_vector<T> hB(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T> hX(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T> hXres(size_BXres, 1, size_BXres, 1);

    device_strided_batch_vector<rocblas_int> dptrT(size_ptrT, 1, size_ptrT, 1);
    device_strided_batch_vector<rocblas_int> dindT(size_indT, 1, size_indT, 1);
    device_strided_batch_vector<T> dvalT(size_valT, 1, size_valT, 1);
    device_strided_batch_vector<rocblas_int> dpivP(size_pivP, 1, size_pivP, 1);
    device_strided_batch_vector<rocblas_int> dpivQ(size_pivQ, 1, size_pivQ, 1);
    device_strided_batch_vector<T> dB(size_BX, 1, size_BX, 1);
    CHECK_HIP_ERROR(dptrT.memcheck());
    if(size_indT)
        CHECK_HIP_ERROR(dindT.memcheck());
    if(size_valT)
        CHECK_HIP_ERROR(dvalT.memcheck());
    if(size_pivP)
        CHECK_HIP_ERROR(dpivP.memcheck());
    if(size_pivQ)
        CHECK_HIP_ERROR(dpivQ.memcheck());
    if(size_BX)
        CHECK_HIP_ERROR(dB.memcheck());

    // check quick return
    if(n == 0 || nrhs == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, dptrT.data(),
                                                    dindT.data(), dvalT.data(), dpivP.data(),
                                                    dpivQ.data(), dB.data(), ldb, rfinfo),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_solve_getError<T>(handle, n, nrhs, nnzT, dptrT, dindT, dvalT, dpivP, dpivQ, dB, ldb,
                                rfinfo, hptrT, hindT, hvalT, hpivP, hpivQ, hB, hX, hXres,
                                &max_error, testcase, mode);

    // collect performance data
    if(argus.timing)
        csrrf_solve_getPerfData<T>(handle, n, nrhs, nnzT, dptrT, dindT, dvalT, dpivP, dpivQ, dB,
                                   ldb, rfinfo, hptrT, hindT, hvalT, hpivP, hpivQ, hB, hX,
                                   &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                   argus.profile_kernels, argus.perf, testcase, mode);

    // validate results for rocsolver-test
    // using 20 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nrhs", "nnzT", "ldb");
            rocsolver_bench_output(n, nrhs, nnzT, ldb);

            rocsolver_bench_header("Results:");
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
            rocsolver_bench_endl();
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

#define EXTERN_TESTING_CSRRF_SOLVE(...) \
    extern template void testing_csrrf_solve<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CSRRF_SOLVE, FOREACH_REAL_TYPE, APPLY_STAMP)
