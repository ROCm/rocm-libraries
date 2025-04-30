/* **************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename S, typename U>
void stedcx_checkBadArgs(const rocblas_handle handle,
                         const rocblas_evect evect,
                         const rocblas_erange erange,
                         const rocblas_int n,
                         const T vl,
                         const T vu,
                         const rocblas_int il,
                         const rocblas_int iu,
                         S dD,
                         S dE,
                         rocblas_int* dnev,
                         S dW,
                         U dC,
                         const rocblas_int ldc,
                         rocblas_int* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(nullptr, evect, erange, n, vl, vu, il, iu, dD, dE, dnev,
                                           dW, dC, ldc, dinfo),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, rocblas_erange(0), n, vl, vu, il, iu, dD,
                                           dE, dnev, dW, dC, ldc, dinfo),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, rocblas_evect(0), erange, n, vl, vu, il, iu, dD,
                                           dE, dnev, dW, dC, ldc, dinfo),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, (S) nullptr,
                                           dE, dnev, dW, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD,
                                           (S) nullptr, dnev, dW, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD, dE,
                                           (rocblas_int*)nullptr, dW, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD, dE, dnev,
                                           (S) nullptr, dC, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD, dE, dnev,
                                           dW, (U) nullptr, ldc, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD, dE, dnev,
                                           dW, dC, ldc, (rocblas_int*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, 0, vl, vu, il, iu, (S) nullptr,
                                           (S) nullptr, dnev, (S) nullptr, (U) nullptr, ldc, dinfo),
                          rocblas_status_success);
}

template <typename T>
void testing_stedcx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 2;
    rocblas_int ldc = 2;
    rocblas_erange erange = rocblas_erange_all;
    rocblas_evect evect = rocblas_evect_original;
    S vl = 0;
    S vu = 0;
    rocblas_int il = 0;
    rocblas_int iu = 0;

    // memory allocations
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<S> dW(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dnev(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dnev.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    stedcx_checkBadArgs(handle, evect, erange, n, vl, vu, il, iu, dD.data(), dE.data(), dnev.data(),
                        dW.data(), dC.data(), ldc, dinfo.data());
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Sh, typename Th>
void stedcx_initData(const rocblas_handle handle,
                     const rocblas_evect evect,
                     const rocblas_int n,
                     Sd& dD,
                     Sd& dE,
                     Td& dC,
                     const rocblas_int ldc,
                     Sh& hD,
                     Sh& hE,
                     Th& hC)
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

    // initialize C to the identity matrix
    if(evect == rocblas_evect_original)
    {
        for(rocblas_int j = 0; j < n; j++)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                if(i == j)
                    hC[0][i + j * ldc] = 1;
                else
                    hC[0][i + j * ldc] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(evect == rocblas_evect_original)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename S, typename Td, typename Sd, typename Id, typename Th, typename Sh, typename Ih>
void stedcx_getError(const rocblas_handle handle,
                     const rocblas_evect evect,
                     const rocblas_erange erange,
                     const rocblas_int n,
                     const S vl,
                     const S vu,
                     const rocblas_int il,
                     const rocblas_int iu,
                     Sd& dD,
                     Sd& dE,
                     Id& dnev,
                     Sd& dW,
                     Td& dC,
                     const rocblas_int ldc,
                     Id& dinfo,
                     Sh& hD,
                     Sh& hE,
                     Ih& hnev,
                     Ih& hnevRes,
                     Sh& hW,
                     Sh& hWRes,
                     Th& hC,
                     Th& hCRes,
                     Ih& hinfo,
                     Ih& hinfoRes,
                     double* max_err)
{
    std::vector<S> work(4 * n);
    std::vector<int> iwork(3 * n);
    std::vector<rocblas_int> hIblock(n);
    std::vector<rocblas_int> hIsplit(n);
    rocblas_int hnsplit;
    S atol = 2 * get_safemin<S>();

    // input data initialization
    stedcx_initData<true, true, S>(handle, evect, n, dD, dE, dC, ldc, hD, hE, hC);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD.data(),
                                         dE.data(), dnev.data(), dW.data(), dC.data(), ldc,
                                         dinfo.data()));
    CHECK_HIP_ERROR(hnevRes.transfer_from(dnev));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    // CPU lapack
    cpu_stebz(erange, rocblas_eorder_entire, n, vl, vu, il, iu, atol, hD[0], hE[0], hnev[0],
              &hnsplit, hW[0], hIblock.data(), hIsplit.data(), work.data(), iwork.data(), hinfo[0]);

    // check info
    EXPECT_EQ(hinfo[0][0], hinfoRes[0][0]);
    if(hinfo[0][0] != hinfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    // if finding eigenvalues succeded, check values
    if(hinfoRes[0][0] == 0)
    {
        // check number of computed eigenvalues
        rocblas_int nn = hnevRes[0][0];
        *max_err += std::abs(nn - hnev[0][0]);
        EXPECT_EQ(hnev[0][0], hnevRes[0][0]);

        // error is ||hW - hWRes|| / ||hW||
        // using frobenius norm
        double err = norm_error('F', 1, nn, 1, hW[0], hWRes[0]);
        *max_err = err > *max_err ? err : *max_err;

        if(evect != rocblas_evect_none)
        {
            // C should be orthonormal, if it is then C^T*C should be the identity
            if(nn > 0)
            {
                std::vector<T> CCres(nn * nn, 0.0);
                std::vector<T> I(nn * nn, 0.0);

                for(rocblas_int i = 0; i < nn; i++)
                    I[i + i * nn] = T(1);

                cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, nn, nn, n,
                         T(1), hCRes[0], ldc, hCRes[0], ldc, T(0), CCres.data(), nn);
                err = norm_error('F', nn, nn, nn, I.data(), CCres.data());
                *max_err = err > *max_err ? err : *max_err;
            }

            // for each of the nev eigenvalues w_j, verify that the associated eigenvector is in the
            // null space of (A - w_i * I)
            T alpha, t1, t2;
            for(int j = 0; j < nn; j++)
            {
                for(int i = 0; i < n; i++)
                {
                    alpha = hWRes[0][j] - hD[0][i];
                    hC[0][i + j * ldc] = hCRes[0][i + j * ldc] * alpha;
                }
                t1 = hCRes[0][j * ldc];
                hCRes[0][j * ldc] = hE[0][0] * hCRes[0][1 + j * ldc];
                for(int i = 1; i < n - 1; i++)
                {
                    t2 = hCRes[0][i + j * ldc];
                    hCRes[0][i + j * ldc]
                        = hE[0][i - 1] * t1 + hE[0][i] * hCRes[0][(i + 1) + j * ldc];
                    t1 = t2;
                }
                hCRes[0][(n - 1) + j * ldc] = hE[0][n - 2] * t1;
            }

            // error is then ||hC - hCRes|| / ||hC||
            // using frobenius norm
            err = norm_error('F', n, nn, ldc, hC[0], hCRes[0]);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <typename T, typename S, typename Td, typename Sd, typename Id, typename Th, typename Sh, typename Ih>
void stedcx_getPerfData(const rocblas_handle handle,
                        const rocblas_evect evect,
                        const rocblas_erange erange,
                        const rocblas_int n,
                        const S vl,
                        const S vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        Sd& dD,
                        Sd& dE,
                        Id& dnev,
                        Sd& dW,
                        Td& dC,
                        const rocblas_int ldc,
                        Id& dinfo,
                        Sh& hD,
                        Sh& hE,
                        Ih& hnev,
                        Sh& hW,
                        Th& hC,
                        Ih& hinfo,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf)
{
    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = nan("");
    }

    stedcx_initData<true, false, S>(handle, evect, n, dD, dE, dC, ldc, hD, hE, hC);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stedcx_initData<false, true, S>(handle, evect, n, dD, dE, dC, ldc, hD, hE, hC);

        CHECK_ROCBLAS_ERROR(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD.data(),
                                             dE.data(), dnev.data(), dW.data(), dC.data(), ldc,
                                             dinfo.data()));
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
        stedcx_initData<false, true, S>(handle, evect, n, dD, dE, dC, ldc, hD, hE, hC);

        start = get_time_us_sync(stream);
        rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD.data(), dE.data(),
                         dnev.data(), dW.data(), dC.data(), ldc, dinfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_stedcx(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char evectC = argus.get<char>("evect");
    char erangeC = argus.get<char>("erange");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldc = argus.get<rocblas_int>("ldc", n);
    S vl = S(argus.get<double>("vl", 0));
    S vu = S(argus.get<double>("vu", erangeC == 'V' ? 1 : 0));
    rocblas_int il = argus.get<rocblas_int>("il", erangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", erangeC == 'I' ? 1 : 0);

    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_erange erange = char2rocblas_erange(erangeC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = n;
    size_t size_W = n;
    size_t size_C = ldc * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size = (n < 0) || (ldc < n) || (erange == rocblas_erange_value && vl >= vu)
        || (erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))
        || (erange == rocblas_erange_index && (il < 1 || iu < 0));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu,
                                               (S*)nullptr, (S*)nullptr, (rocblas_int*)nullptr,
                                               (S*)nullptr, (T*)nullptr, ldc, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, (S*)nullptr,
                                           (S*)nullptr, (rocblas_int*)nullptr, (S*)nullptr,
                                           (T*)nullptr, ldc, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<S> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<S> hWRes(size_WRes, 1, size_WRes, 1);
    host_strided_batch_vector<rocblas_int> hnev(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hnevRes(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<S> dW(size_W, 1, size_W, 1);
    device_strided_batch_vector<rocblas_int> dnev(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);

    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dnev.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stedcx(handle, evect, erange, n, vl, vu, il, iu, dD.data(),
                                               dE.data(), dnev.data(), dW.data(), dC.data(), ldc,
                                               dinfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stedcx_getError<T>(handle, evect, erange, n, vl, vu, il, iu, dD, dE, dnev, dW, dC, ldc, dinfo,
                           hD, hE, hnev, hnevRes, hW, hWRes, hC, hCRes, hinfo, hinfoRes, &max_error);

    // collect performance data
    if(argus.timing)
        stedcx_getPerfData<T>(handle, evect, erange, n, vl, vu, il, iu, dD, dE, dnev, dW, dC, ldc,
                              dinfo, hD, hE, hnev, hW, hC, hinfo, &gpu_time_used, &cpu_time_used,
                              hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using 3 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 3 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("evect", "erange", "n", "vl", "vu", "il", "iu", "ldc");
            rocsolver_bench_output(evectC, erangeC, n, vl, vu, il, iu, ldc);

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

#define EXTERN_TESTING_STEDCX(...) extern template void testing_stedcx<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_STEDCX, FOREACH_SCALAR_TYPE, APPLY_STAMP)
