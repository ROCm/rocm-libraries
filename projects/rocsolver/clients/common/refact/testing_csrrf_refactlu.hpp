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
void csrrf_refactlu_checkBadArgs(rocblas_handle handle,
                                 const rocblas_int n,
                                 const rocblas_int nnzA,
                                 rocblas_int* ptrA,
                                 rocblas_int* indA,
                                 T valA,
                                 const rocblas_int nnzT,
                                 rocblas_int* ptrT,
                                 rocblas_int* indT,
                                 T valT,
                                 rocblas_int* pivP,
                                 rocblas_int* pivQ,
                                 rocsolver_rfinfo rfinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(nullptr, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, (rocblas_int*)nullptr, indA,
                                                   valA, nnzT, ptrT, indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, (rocblas_int*)nullptr,
                                                   valA, nnzT, ptrT, indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, (T) nullptr, nnzT,
                                                   ptrT, indT, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT,
                                                   (rocblas_int*)nullptr, indT, valT, pivP, pivQ,
                                                   rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   (rocblas_int*)nullptr, valT, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, (T) nullptr, pivP, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, valT, (rocblas_int*)nullptr, pivQ, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, valT, pivP, (rocblas_int*)nullptr, rfinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, valT, pivP, pivQ, nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, 0, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                   indT, valT, (rocblas_int*)nullptr,
                                                   (rocblas_int*)nullptr, rfinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_csrrf_refactlu(handle, 0, 0, ptrA, (rocblas_int*)nullptr, (T) nullptr, nnzT, ptrT,
                                 indT, valT, (rocblas_int*)nullptr, (rocblas_int*)nullptr, rfinfo),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_csrrf_refactlu(handle, 0, nnzA, ptrA, indA, valA, 0, ptrT, (rocblas_int*)nullptr,
                                 (T) nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr, rfinfo),
        rocblas_status_success);

    // quick return with zero batch_count if applicable
    // N/A
}

template <typename T>
void testing_csrrf_refactlu_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = 1;
    rocblas_int nnzA = 1;
    rocblas_int nnzT = 1;

    // memory allocations
    device_strided_batch_vector<rocblas_int> ptrA(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indA(1, 1, 1, 1);
    device_strided_batch_vector<T> valA(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T> valT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> pivP(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> pivQ(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrA.memcheck());
    CHECK_HIP_ERROR(indA.memcheck());
    CHECK_HIP_ERROR(valA.memcheck());
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());
    CHECK_HIP_ERROR(pivP.memcheck());
    CHECK_HIP_ERROR(pivQ.memcheck());

    // check bad arguments
    csrrf_refactlu_checkBadArgs(handle, n, nnzA, ptrA.data(), indA.data(), valA.data(), nnzT,
                                ptrT.data(), indT.data(), valT.data(), pivP.data(), pivQ.data(),
                                rfinfo);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_refactlu_initData(rocblas_handle handle,
                             const rocblas_int n,
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
                             Uh& hptrA,
                             Uh& hindA,
                             Th& hvalA,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ,
                             const fs::path testcase)
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
        CHECK_HIP_ERROR(dptrT.transfer_from(hptrT));
        CHECK_HIP_ERROR(dindT.transfer_from(hindT));
        CHECK_HIP_ERROR(dvalT.transfer_from(hvalT));
        CHECK_HIP_ERROR(dpivP.transfer_from(hpivP));
        CHECK_HIP_ERROR(dpivQ.transfer_from(hpivQ));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_refactlu_getError(rocblas_handle handle,
                             const rocblas_int n,
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
                             rocsolver_rfinfo rfinfo,
                             Uh& hptrA,
                             Uh& hindA,
                             Th& hvalA,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             Uh& hpivP,
                             Uh& hpivQ,
                             Th& hvalTres,
                             double* max_err,
                             const fs::path testcase)
{
    // input data initialization
    csrrf_refactlu_initData<true, true, T>(handle, n, nnzA, dptrA, dindA, dvalA, nnzT, dptrT, dindT,
                                           dvalT, dpivP, dpivQ, hptrA, hindA, hvalA, hptrT, hindT,
                                           hvalT, hpivP, hpivQ, testcase);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(
        handle, n, 0, nnzA, dptrA.data(), dindA.data(), dvalA.data(), nnzT, dptrT.data(),
        dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), (T*)nullptr, n, rfinfo));

    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_refactlu(handle, n, nnzA, dptrA.data(), dindA.data(),
                                                 dvalA.data(), nnzT, dptrT.data(), dindT.data(),
                                                 dvalT.data(), dpivP.data(), dpivQ.data(), rfinfo));

    CHECK_HIP_ERROR(hvalTres.transfer_from(dvalT));

    // compare computed results with original result
    *max_err = norm_error('F', 1, nnzT, 1, hvalT[0], hvalTres[0]);
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_refactlu_getPerfData(rocblas_handle handle,
                                const rocblas_int n,
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
                                rocsolver_rfinfo rfinfo,
                                Uh& hptrA,
                                Uh& hindA,
                                Th& hvalA,
                                Uh& hptrT,
                                Uh& hindT,
                                Th& hvalT,
                                Uh& hpivP,
                                Uh& hpivQ,
                                double* gpu_time_used,
                                double* cpu_time_used,
                                const rocblas_int hot_calls,
                                const int profile,
                                const bool profile_kernels,
                                const bool perf,
                                const fs::path testcase)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution

    csrrf_refactlu_initData<true, true, T>(handle, n, nnzA, dptrA, dindA, dvalA, nnzT, dptrT, dindT,
                                           dvalT, dpivP, dpivQ, hptrA, hindA, hvalA, hptrT, hindT,
                                           hvalT, hpivP, hpivQ, testcase);

    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_analysis(
        handle, n, 0, nnzA, dptrA.data(), dindA.data(), dvalA.data(), nnzT, dptrT.data(),
        dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), (T*)nullptr, n, rfinfo));

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        csrrf_refactlu_initData<false, true, T>(handle, n, nnzA, dptrA, dindA, dvalA, nnzT, dptrT,
                                                dindT, dvalT, dpivP, dpivQ, hptrA, hindA, hvalA,
                                                hptrT, hindT, hvalT, hpivP, hpivQ, testcase);

        CHECK_ROCBLAS_ERROR(rocsolver_csrrf_refactlu(
            handle, n, nnzA, dptrA.data(), dindA.data(), dvalA.data(), nnzT, dptrT.data(),
            dindT.data(), dvalT.data(), dpivP.data(), dpivQ.data(), rfinfo));
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
        csrrf_refactlu_initData<false, true, T>(handle, n, nnzA, dptrA, dindA, dvalA, nnzT, dptrT,
                                                dindT, dvalT, dpivP, dpivQ, hptrA, hindA, hvalA,
                                                hptrT, hindT, hvalT, hpivP, hpivQ, testcase);

        start = get_time_us_sync(stream);
        rocsolver_csrrf_refactlu(handle, n, nnzA, dptrA.data(), dindA.data(), dvalA.data(), nnzT,
                                 dptrT.data(), dindT.data(), dvalT.data(), dpivP.data(),
                                 dpivQ.data(), rfinfo);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_csrrf_refactlu(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocsolver_local_rfinfo rfinfo(handle);
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nnzA = argus.get<rocblas_int>("nnzA");
    rocblas_int nnzT = argus.get<rocblas_int>("nnzT");
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzA < 0 || nnzT < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(
                                  handle, n, nnzA, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                  (T*)nullptr, nnzT, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                  (T*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr, rfinfo),
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
        testcase = get_sparse_data_dir() / fs::path(fmt::format("mat_{}_{}", n, nnzA));
        fs::path fileA = testcase / "ptrA";
        read_last(fileA.string(), &nnzA);
        fs::path fileT = testcase / "ptrT";
        read_last(fileT.string(), &nnzT);
    }

    // memory size query if necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_csrrf_refactlu(
            handle, n, nnzA, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, nnzT,
            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
            (rocblas_int*)nullptr, rfinfo));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // determine sizes
    size_t size_ptrA = size_t(n) + 1;
    size_t size_indA = size_t(nnzA);
    size_t size_valA = size_t(nnzA);
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);
    size_t size_pivP = size_t(n);
    size_t size_pivQ = size_t(n);

    size_t size_valTres = 0;
    if(argus.unit_check || argus.norm_check)
        size_valTres = size_valT;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<rocblas_int> hptrA(size_ptrA, 1, size_ptrA, 1);
    host_strided_batch_vector<rocblas_int> hindA(size_indA, 1, size_indA, 1);
    host_strided_batch_vector<T> hvalA(size_valA, 1, size_valA, 1);
    host_strided_batch_vector<rocblas_int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<rocblas_int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<T> hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<T> hvalTres(size_valTres, 1, size_valTres, 1);
    host_strided_batch_vector<rocblas_int> hpivP(size_pivP, 1, size_pivP, 1);
    host_strided_batch_vector<rocblas_int> hpivQ(size_pivQ, 1, size_pivQ, 1);

    device_strided_batch_vector<rocblas_int> dptrA(size_ptrA, 1, size_ptrA, 1);
    device_strided_batch_vector<rocblas_int> dindA(size_indA, 1, size_indA, 1);
    device_strided_batch_vector<T> dvalA(size_valA, 1, size_valA, 1);
    device_strided_batch_vector<rocblas_int> dptrT(size_ptrT, 1, size_ptrT, 1);
    device_strided_batch_vector<rocblas_int> dindT(size_indT, 1, size_indT, 1);
    device_strided_batch_vector<T> dvalT(size_valT, 1, size_valT, 1);
    device_strided_batch_vector<rocblas_int> dpivP(size_pivP, 1, size_pivP, 1);
    device_strided_batch_vector<rocblas_int> dpivQ(size_pivQ, 1, size_pivQ, 1);
    CHECK_HIP_ERROR(dptrA.memcheck());
    CHECK_HIP_ERROR(dptrT.memcheck());
    if(size_indA)
        CHECK_HIP_ERROR(dindA.memcheck());
    if(size_valA)
        CHECK_HIP_ERROR(dvalA.memcheck());
    if(size_indT)
        CHECK_HIP_ERROR(dindT.memcheck());
    if(size_valT)
        CHECK_HIP_ERROR(dvalT.memcheck());
    if(size_pivP)
        CHECK_HIP_ERROR(dpivP.memcheck());
    if(size_pivQ)
        CHECK_HIP_ERROR(dpivQ.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_refactlu(handle, n, nnzA, dptrA.data(), dindA.data(),
                                                       dvalA.data(), nnzT, dptrT.data(),
                                                       dindT.data(), dvalT.data(), dpivP.data(),
                                                       dpivQ.data(), rfinfo),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_refactlu_getError<T>(handle, n, nnzA, dptrA, dindA, dvalA, nnzT, dptrT, dindT, dvalT,
                                   dpivP, dpivQ, rfinfo, hptrA, hindA, hvalA, hptrT, hindT, hvalT,
                                   hpivP, hpivQ, hvalTres, &max_error, testcase);

    // collect performance data
    if(argus.timing)
        csrrf_refactlu_getPerfData<T>(handle, n, nnzA, dptrA, dindA, dvalA, nnzT, dptrT, dindT,
                                      dvalT, dpivP, dpivQ, rfinfo, hptrA, hindA, hvalA, hptrT, hindT,
                                      hvalT, hpivP, hpivQ, &gpu_time_used, &cpu_time_used, hot_calls,
                                      argus.profile, argus.profile_kernels, argus.perf, testcase);

    // validate results for rocsolver-test
    // using 2 * n * machine precision for tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nnzA", "nnzT");
            rocsolver_bench_output(n, nnzA, nnzT);

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

#define EXTERN_TESTING_CSRRF_REFACTLU(...) \
    extern template void testing_csrrf_refactlu<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CSRRF_REFACTLU, FOREACH_REAL_TYPE, APPLY_STAMP)
