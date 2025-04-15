/* **************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename U>
void stebz_checkBadArgs(const rocblas_handle handle,
                        const rocblas_erange erange,
                        const rocblas_eorder eorder,
                        const rocblas_int n,
                        const T vl,
                        const T vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        const T abstol,
                        U dD,
                        U dE,
                        rocblas_int* dnev,
                        rocblas_int* dnsplit,
                        U dW,
                        rocblas_int* dIblock,
                        rocblas_int* dIsplit,
                        rocblas_int* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(nullptr, erange, eorder, n, vl, vu, il, iu, abstol, dD,
                                          dE, dnev, dnsplit, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, rocblas_erange(0), eorder, n, vl, vu, il, iu,
                                          abstol, dD, dE, dnev, dnsplit, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, rocblas_eorder(0), n, vl, vu, il, iu,
                                          abstol, dD, dE, dnev, dnsplit, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                          (U) nullptr, dE, dnev, dnsplit, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD,
                                          (U) nullptr, dnev, dnsplit, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE,
                                          (rocblas_int*)nullptr, dnsplit, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE,
                                          dnev, (rocblas_int*)nullptr, dW, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE,
                                          dnev, dnsplit, (U) nullptr, dIblock, dIsplit, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE,
                                          dnev, dnsplit, dW, (rocblas_int*)nullptr, dIsplit, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE,
                                          dnev, dnsplit, dW, dIblock, (rocblas_int*)nullptr, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE,
                                          dnev, dnsplit, dW, dIblock, dIsplit, (rocblas_int*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, 0, vl, vu, il, iu, abstol,
                                          (U) nullptr, (U) nullptr, dnev, dnsplit, (U) nullptr,
                                          (rocblas_int*)nullptr, (rocblas_int*)nullptr, dinfo),
                          rocblas_status_success);
}

template <typename T>
void testing_stebz_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;
    rocblas_erange erange = rocblas_erange_all;
    rocblas_eorder eorder = rocblas_eorder_entire;
    T vl = 0;
    T vu = 0;
    rocblas_int il = 0;
    rocblas_int iu = 0;
    T abstol = 0;

    // memory allocations
    device_strided_batch_vector<T> dD(1, 1, 1, 1);
    device_strided_batch_vector<T> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dW(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnev(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnsplit(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIblock(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIsplit(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dnev.memcheck());
    CHECK_HIP_ERROR(dnsplit.memcheck());
    CHECK_HIP_ERROR(dIblock.memcheck());
    CHECK_HIP_ERROR(dIsplit.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    stebz_checkBadArgs(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD.data(), dE.data(),
                       dnev.data(), dnsplit.data(), dW.data(), dIblock.data(), dIsplit.data(),
                       dinfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void stebz_initData(const rocblas_handle handle, const rocblas_int n, Td& dD, Td& dE, Th& hD, Th& hE)
{
    if(CPU)
    {
        rocblas_init<T>(hD, true);
        rocblas_init<T>(hE, true);

        // scale matrix and add fixed splits in the matrix to test split handling
        // (scaling ensures that all eigenvalues are in [-20, 20])
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 10;
            hE[0][i] = (hE[0][i] - 5) / 10;
            if(i == n / 4 || i == n / 2 || i == n - 1)
                hE[0][i] = 0;
            if(i == n / 7 || i == n / 5 || i == n / 3)
                hD[0][i] *= -1;
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void stebz_getError(const rocblas_handle handle,
                    const rocblas_erange erange,
                    const rocblas_eorder eorder,
                    const rocblas_int n,
                    const T vl,
                    const T vu,
                    const rocblas_int il,
                    const rocblas_int iu,
                    const T abstol,
                    Td& dD,
                    Td& dE,
                    Ud& dnev,
                    Ud& dnsplit,
                    Td& dW,
                    Ud& dIblock,
                    Ud& dIsplit,
                    Ud& dinfo,
                    Th& hD,
                    Th& hE,
                    Uh& hnev,
                    Uh& hnevRes,
                    Uh& hnsplit,
                    Uh& hnsplitRes,
                    Th& hW,
                    Th& hWRes,
                    Uh& hIblock,
                    Uh& hIblockRes,
                    Uh& hIsplit,
                    Uh& hIsplitRes,
                    Uh& hinfo,
                    Uh& hinfoRes,
                    double* max_err)
{
    std::vector<T> work(4 * n);
    std::vector<int> iwork(3 * n);

    // input data initialization
    stebz_initData<true, true, T>(handle, n, dD, dE, hD, hE);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                        dD.data(), dE.data(), dnev.data(), dnsplit.data(),
                                        dW.data(), dIblock.data(), dIsplit.data(), dinfo.data()));
    CHECK_HIP_ERROR(hnevRes.transfer_from(dnev));
    CHECK_HIP_ERROR(hnsplitRes.transfer_from(dnsplit));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hIblockRes.transfer_from(dIblock));
    CHECK_HIP_ERROR(hIsplitRes.transfer_from(dIsplit));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    // CPU lapack
    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    T atol = (abstol == 0) ? 2 * get_safemin<T>() : abstol;
    cpu_stebz(erange, eorder, n, vl, vu, il, iu, atol, hD[0], hE[0], hnev[0], hnsplit[0], hW[0],
              hIblock[0], hIsplit[0], work.data(), iwork.data(), hinfo[0]);

    // check info
    EXPECT_EQ(hinfo[0][0], hinfoRes[0][0]);
    if(hinfo[0][0] != hinfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    // check number of split blocks
    rocblas_int ns = hnsplit[0][0];
    *max_err += std::abs(ns - hnsplitRes[0][0]);
    EXPECT_EQ(hnsplit[0][0], hnsplitRes[0][0]);

    // check split blocks limits
    for(int k = 0; k < ns; ++k)
    {
        *max_err += std::abs(hIsplit[0][k] - hIsplitRes[0][k]);
        EXPECT_EQ(hIsplit[0][k], hIsplitRes[0][k]) << "where k = " << k;
    }

    // if finding eigenvalues succeded, check values
    if(hinfo[0][0] == 0)
    {
        // check number of computed eigenvalues
        rocblas_int nn = hnev[0][0];
        *max_err += std::abs(nn - hnevRes[0][0]);
        EXPECT_EQ(hnev[0][0], hnevRes[0][0]);

        // check block indices
        // (note: as very close eigenvalues could be considered to belong to different
        //  blocks by the CPU and GPU algorithms, only check the block index of distinguishable
        //  eigenvalues)
        for(int k = 0; k < nn; ++k)
        {
            int difb = std::abs(hIblock[0][k] - hIblockRes[0][k]);
            T difv = std::abs(hW[0][k] - hWRes[0][k]) / hW[0][k];
            if(difv > n * get_epsilon<T>())
            {
                EXPECT_EQ(hIblock[0][k], hIblockRes[0][k]) << "where k = " << k;
                if(difb > 0)
                    *max_err += difb;
            }
        }

        // error is ||hW - hWRes|| / ||hW||
        // using frobenius norm
        double err = norm_error('F', 1, nn, 1, hW[0], hWRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void stebz_getPerfData(const rocblas_handle handle,
                       const rocblas_erange erange,
                       const rocblas_eorder eorder,
                       const rocblas_int n,
                       const T vl,
                       const T vu,
                       const rocblas_int il,
                       const rocblas_int iu,
                       const T abstol,
                       Td& dD,
                       Td& dE,
                       Ud& dnev,
                       Ud& dnsplit,
                       Td& dW,
                       Ud& dIblock,
                       Ud& dIsplit,
                       Ud& dinfo,
                       Th& hD,
                       Th& hE,
                       Uh& hnev,
                       Uh& hnsplit,
                       Th& hW,
                       Uh& hIblock,
                       Uh& hIsplit,
                       Uh& hinfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        std::vector<T> work(4 * n);
        std::vector<int> iwork(3 * n);
        // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
        T atol = (abstol == 0) ? 2 * get_safemin<T>() : abstol;

        stebz_initData<true, false, T>(handle, n, dD, dE, hD, hE);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_stebz(erange, eorder, n, vl, vu, il, iu, atol, hD[0], hE[0], hnev[0], hnsplit[0], hW[0],
                  hIblock[0], hIsplit[0], work.data(), iwork.data(), hinfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stebz_initData<true, false, T>(handle, n, dD, dE, hD, hE);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stebz_initData<false, true, T>(handle, n, dD, dE, hD, hE);

        CHECK_ROCBLAS_ERROR(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                            dD.data(), dE.data(), dnev.data(), dnsplit.data(),
                                            dW.data(), dIblock.data(), dIsplit.data(), dinfo.data()));
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
        stebz_initData<false, true, T>(handle, n, dD, dE, hD, hE);

        start = get_time_us_sync(stream);
        rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD.data(), dE.data(),
                        dnev.data(), dnsplit.data(), dW.data(), dIblock.data(), dIsplit.data(),
                        dinfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_stebz(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char erangeC = argus.get<char>("erange");
    char eorderC = argus.get<char>("eorder");
    rocblas_int n = argus.get<rocblas_int>("n");
    T vl = T(argus.get<double>("vl", 0));
    T vu = T(argus.get<double>("vu", erangeC == 'V' ? 1 : 0));
    rocblas_int il = argus.get<rocblas_int>("il", erangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", erangeC == 'I' ? 1 : 0);
    T abstol = T(argus.get<double>("abstol"));

    rocblas_erange erange = char2rocblas_erange(erangeC);
    rocblas_eorder eorder = char2rocblas_eorder(eorderC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    size_t size_W = n;
    size_t size_iblock = n;
    size_t size_isplit = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;
    size_t size_iblockRes = (argus.unit_check || argus.norm_check) ? size_iblock : 0;
    size_t size_isplitRes = (argus.unit_check || argus.norm_check) ? size_isplit : 0;

    // check invalid sizes
    bool invalid_size = (n < 0) || (erange == rocblas_erange_value && vl >= vu)
        || (erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))
        || (erange == rocblas_erange_index && (il < 1 || iu < 0));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol, (T*)nullptr,
                            (T*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr,
                            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (rocblas_int*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                          (T*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
                                          (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
                                          (rocblas_int*)nullptr, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<T> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<T> hWRes(size_WRes, 1, size_WRes, 1);
    host_strided_batch_vector<rocblas_int> hIblock(size_iblock, 1, size_iblock, 1);
    host_strided_batch_vector<rocblas_int> hIblockRes(size_iblockRes, 1, size_iblockRes, 1);
    host_strided_batch_vector<rocblas_int> hIsplit(size_isplit, 1, size_isplit, 1);
    host_strided_batch_vector<rocblas_int> hIsplitRes(size_isplitRes, 1, size_isplitRes, 1);
    host_strided_batch_vector<rocblas_int> hnev(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnevRes(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnsplit(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnsplitRes(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<T> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dW(size_W, 1, size_W, 1);
    device_strided_batch_vector<rocblas_int> dIblock(size_iblock, 1, size_iblock, 1);
    device_strided_batch_vector<rocblas_int> dIsplit(size_isplit, 1, size_isplit, 1);
    device_strided_batch_vector<rocblas_int> dnev(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnsplit(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);

    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(size_iblock)
        CHECK_HIP_ERROR(dIblock.memcheck());
    if(size_isplit)
        CHECK_HIP_ERROR(dIsplit.memcheck());
    CHECK_HIP_ERROR(dnev.memcheck());
    CHECK_HIP_ERROR(dnsplit.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stebz(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                              dD.data(), dE.data(), dnev.data(), dnsplit.data(),
                                              dW.data(), dIblock.data(), dIsplit.data(), dinfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stebz_getError<T>(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE, dnev, dnsplit,
                          dW, dIblock, dIsplit, dinfo, hD, hE, hnev, hnevRes, hnsplit, hnsplitRes,
                          hW, hWRes, hIblock, hIblockRes, hIsplit, hIsplitRes, hinfo, hinfoRes,
                          &max_error);

    // collect performance data
    if(argus.timing)
        stebz_getPerfData<T>(handle, erange, eorder, n, vl, vu, il, iu, abstol, dD, dE, dnev,
                             dnsplit, dW, dIblock, dIsplit, dinfo, hD, hE, hnev, hnsplit, hW,
                             hIblock, hIsplit, hinfo, &gpu_time_used, &cpu_time_used, hot_calls,
                             argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("erange", "eorder", "n", "vl", "vu", "il", "iu", "abstol");
            rocsolver_bench_output(erangeC, eorderC, n, vl, vu, il, iu, abstol);

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

#define EXTERN_TESTING_STEBZ(...) extern template void testing_stebz<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_STEBZ, FOREACH_REAL_TYPE, APPLY_STAMP)
