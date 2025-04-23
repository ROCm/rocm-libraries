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

/*
 * ===========================================================================
 *    testing_csrrf_workflow is a modified version of testing_csrrf_analysis
 *    that tests invalid workflows for csrrf functions that need analysis
 *    to be performed first. checkBadArgs has been removed.
 * ===========================================================================
 */

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_workflow_initData(rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int nrhs,
                             const rocblas_int nnzA,
                             Ud& dptrA,
                             Ud& dindA,
                             Td& dvalA,
                             const rocblas_int nnzT,
                             Ud& dptrT,
                             Ud& dindT,
                             Td& dvalT,
                             Ud& dpivP,
                             Ud& dpivQ,
                             Td& dB,
                             const rocblas_int ldb,
                             Uh& hptrA,
                             Uh& hindA,
                             Th& hvalA,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ,
                             Th& hB,
                             const fs::path testcase,
                             const rocsolver_rfinfo_mode mode)
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
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dptrA.transfer_from(hptrA));
        CHECK_HIP_ERROR(dindA.transfer_from(hindA));
        CHECK_HIP_ERROR(dvalA.transfer_from(hvalA));
        CHECK_HIP_ERROR(dptrT.transfer_from(hptrT));
        CHECK_HIP_ERROR(dindT.transfer_from(hindT));
        CHECK_HIP_ERROR(dvalT.transfer_from(hvalT));

        if(mode == rocsolver_rfinfo_mode_lu)
            CHECK_HIP_ERROR(dpivP.transfer_from(hpivP));
        else
            CHECK_HIP_ERROR(dpivP.transfer_from(hpivQ));
        CHECK_HIP_ERROR(dpivQ.transfer_from(hpivQ));

        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_workflow_getError(rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int nrhs,
                             const rocblas_int nnzM,
                             Ud& dptrM,
                             Ud& dindM,
                             Td& dvalM,
                             const rocblas_int nnzT,
                             Ud& dptrT,
                             Ud& dindT,
                             Td& dvalT,
                             Ud& dpivP,
                             Ud& dpivQ,
                             Td& dB,
                             const rocblas_int ldb,
                             rocsolver_rfinfo rfinfo,
                             Uh& hptrM,
                             Uh& hindM,
                             Th& hvalM,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ,
                             Th& hB,
                             double* max_err,
                             const fs::path testcase,
                             const rocsolver_rfinfo_mode mode,
                             const rocsolver_rfinfo_mode analysis_mode)
{
    // input data initialization
    csrrf_workflow_initData<true, true, T>(
        handle, n, nrhs, nnzM, dptrM, dindM, dvalM, nnzT, dptrT, dindT, dvalT, dpivP, dpivQ, dB,
        ldb, hptrM, hindM, hvalM, hptrT, hindT, hvalT, hpivP, hpivQ, hB, testcase, analysis_mode);

    // execute computations
    // GPU lapack

    // cannot perform re-factorization or solve before analysis
    if(mode == rocsolver_rfinfo_mode_lu)
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzM, dptrM.data(), dindM.data(),
                                                       dvalM.data(), nnzT, dptrT.data(),
                                                       dindT.data(), dvalT.data(), dpivP.data(),
                                                       dpivQ.data(), rfinfo),
                              rocblas_status_internal_error);
    else
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactchol(
                                  handle, n, nnzM, dptrM.data(), dindM.data(), dvalM.data(), nnzT,
                                  dptrT.data(), dindT.data(), dvalT.data(), dpivQ.data(), rfinfo),
                              rocblas_status_internal_error);

    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, dptrT.data(), dindT.data(),
                                                dvalT.data(), dpivP.data(), dpivQ.data(), dB.data(),
                                                ldb, rfinfo),
                          rocblas_status_internal_error);

    // perform analysis
    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(
        handle, n, nrhs, nnzM, dptrM.data(), dindM.data(), dvalM.data(), nnzT, dptrT.data(),
        dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), dB.data(), ldb, rfinfo));

    // cannot perform re-factorization or solve when mode doesn't match
    if(mode == rocsolver_rfinfo_mode_lu)
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzM, dptrM.data(), dindM.data(),
                                                       dvalM.data(), nnzT, dptrT.data(),
                                                       dindT.data(), dvalT.data(), dpivP.data(),
                                                       dpivQ.data(), rfinfo),
                              rocblas_status_internal_error);
    else
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactchol(
                                  handle, n, nnzM, dptrM.data(), dindM.data(), dvalM.data(), nnzT,
                                  dptrT.data(), dindT.data(), dvalT.data(), dpivQ.data(), rfinfo),
                              rocblas_status_internal_error);

    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(rfinfo, mode));
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_solve(handle, n, nrhs, nnzT, dptrT.data(), dindT.data(),
                                                dvalT.data(), dpivP.data(), dpivQ.data(), dB.data(),
                                                ldb, rfinfo),
                          rocblas_status_internal_error);

    // No error to calculate...
    // (analysis phase is required by refact and solve, so its results are validated there)
    *max_err = 0;
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_workflow_getPerfData(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                const rocblas_int nnzM,
                                Ud& dptrM,
                                Ud& dindM,
                                Td& dvalM,
                                const rocblas_int nnzT,
                                Ud& dptrT,
                                Ud& dindT,
                                Td& dvalT,
                                Ud& dpivP,
                                Ud& dpivQ,
                                Td& dB,
                                const rocblas_int ldb,
                                rocsolver_rfinfo rfinfo,
                                Uh& hptrM,
                                Uh& hindM,
                                Th& hvalM,
                                Uh& hptrT,
                                Uh& hindT,
                                Th& hvalT,
                                Uh& hpivP,
                                Uh& hpivQ,
                                Th& hB,
                                double* gpu_time_used,
                                double* cpu_time_used,
                                const rocblas_int hot_calls,
                                const int profile,
                                const bool profile_kernels,
                                const bool perf,
                                const fs::path testcase,
                                const rocsolver_rfinfo_mode mode,
                                const rocsolver_rfinfo_mode analysis_mode)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution
    *gpu_time_used = nan(""); // no timing on gpu-lapack execution
}

template <typename T>
void testing_csrrf_workflow(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs", 0);
    rocblas_int nnzM = argus.get<rocblas_int>("nnzM");
    rocblas_int nnzT = argus.get<rocblas_int>("nnzT");
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    char modeC = argus.get<char>("rfinfo_mode");
    char analysis_modeC = argus.get<char>("analysis_mode");
    rocblas_int hot_calls = argus.iters;

    rocsolver_rfinfo_mode mode = char2rocsolver_rfinfo_mode(modeC);
    rocsolver_rfinfo_mode analysis_mode = char2rocsolver_rfinfo_mode(analysis_modeC);
    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(rfinfo, analysis_mode));

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzM < 0 || nnzT < 0 || nrhs < 0 || ldb < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_csrrf_analysis(handle, n, nrhs, nnzM, (rocblas_int*)nullptr,
                                     (rocblas_int*)nullptr, (T*)nullptr, nnzT, (rocblas_int*)nullptr,
                                     (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
                                     (rocblas_int*)nullptr, (T*)nullptr, ldb, rfinfo),
            rocblas_status_invalid_size);

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
        if(nnzM <= 80)
            nnzM = 60;
        else if(nnzM <= 120)
            nnzM = 100;
        else
            nnzM = 140;
    }
    else // large case
    {
        if(nnzM <= 400)
            nnzM = 300;
        else if(nnzM <= 600)
            nnzM = 500;
        else
            nnzM = 700;
    }

    // read/set corresponding nnzT
    fs::path testcase;
    if(n > 0)
    {
        std::string matname;
        if(analysis_mode == rocsolver_rfinfo_mode_lu)
            matname = fmt::format("mat_{}_{}", n, nnzM);
        else
            matname = fmt::format("posmat_{}_{}", n, nnzM);

        testcase = get_sparse_data_dir() / fs::path(matname);
        fs::path fileA = testcase / "ptrA";
        read_last(fileA.string(), &nnzM);
        fs::path fileT = testcase / "ptrT";
        read_last(fileT.string(), &nnzT);
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
        CHECK_ALLOC_QUERY(rocsolver_csrrf_analysis(
            handle, n, nrhs, nnzM, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, nnzT,
            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
            (rocblas_int*)nullptr, (T*)nullptr, ldb, rfinfo));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // determine sizes
    size_t size_ptrM = size_t(n) + 1;
    size_t size_indM = size_t(nnzM);
    size_t size_valM = size_t(nnzM);
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);
    size_t size_pivP = size_t(n);
    size_t size_pivQ = size_t(n);
    size_t size_BX = size_t(ldb) * nrhs;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<rocblas_int> hptrM(size_ptrM, 1, size_ptrM, 1);
    host_strided_batch_vector<rocblas_int> hindM(size_indM, 1, size_indM, 1);
    host_strided_batch_vector<T> hvalM(size_valM, 1, size_valM, 1);
    host_strided_batch_vector<rocblas_int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<rocblas_int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<T> hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<rocblas_int> hpivP(size_pivP, 1, size_pivP, 1);
    host_strided_batch_vector<rocblas_int> hpivQ(size_pivQ, 1, size_pivQ, 1);
    host_strided_batch_vector<T> hB(size_BX, 1, size_BX, 1);

    device_strided_batch_vector<rocblas_int> dptrM(size_ptrM, 1, size_ptrM, 1);
    device_strided_batch_vector<rocblas_int> dindM(size_indM, 1, size_indM, 1);
    device_strided_batch_vector<T> dvalM(size_valM, 1, size_valM, 1);
    device_strided_batch_vector<rocblas_int> dptrT(size_ptrT, 1, size_ptrT, 1);
    device_strided_batch_vector<rocblas_int> dindT(size_indT, 1, size_indT, 1);
    device_strided_batch_vector<T> dvalT(size_valT, 1, size_valT, 1);
    device_strided_batch_vector<rocblas_int> dpivP(size_pivP, 1, size_pivP, 1);
    device_strided_batch_vector<rocblas_int> dpivQ(size_pivQ, 1, size_pivQ, 1);
    device_strided_batch_vector<T> dB(size_BX, 1, size_BX, 1);
    CHECK_HIP_ERROR(dptrM.memcheck());
    CHECK_HIP_ERROR(dptrT.memcheck());
    if(size_indM)
        CHECK_HIP_ERROR(dindM.memcheck());
    if(size_valM)
        CHECK_HIP_ERROR(dvalM.memcheck());
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
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_csrrf_analysis(handle, n, nrhs, nnzM, dptrM.data(), dindM.data(),
                                     dvalM.data(), nnzT, dptrT.data(), dindT.data(), dvalT.data(),
                                     dpivP.data(), dpivQ.data(), dB.data(), ldb, rfinfo),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_workflow_getError<T>(handle, n, nrhs, nnzM, dptrM, dindM, dvalM, nnzT, dptrT, dindT,
                                   dvalT, dpivP, dpivQ, dB, ldb, rfinfo, hptrM, hindM, hvalM, hptrT,
                                   hindT, hvalT, hpivP, hpivQ, hB, &max_error, testcase, mode,
                                   analysis_mode);

    // collect performance data
    if(argus.timing)
        csrrf_workflow_getPerfData<T>(handle, n, nrhs, nnzM, dptrM, dindM, dvalM, nnzT, dptrT,
                                      dindT, dvalT, dpivP, dpivQ, dB, ldb, rfinfo, hptrM, hindM,
                                      hvalM, hptrT, hindT, hvalT, hpivP, hpivQ, hB, &gpu_time_used,
                                      &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                                      argus.perf, testcase, mode, analysis_mode);

    // validate results for rocsolver-test
    // N/A

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nrhs", "nnzM", "nnzT", "ldb");
            rocsolver_bench_output(n, nrhs, nnzM, nnzT, ldb);

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
