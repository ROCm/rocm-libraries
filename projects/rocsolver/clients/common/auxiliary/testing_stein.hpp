/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
void stein_checkBadArgs(const rocblas_handle handle,
                        const rocblas_int n,
                        S dD,
                        S dE,
                        U dNev,
                        S dW,
                        U dIblock,
                        U dIsplit,
                        T dZ,
                        const rocblas_int ldz,
                        U dIfail,
                        U dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(nullptr, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dIfail, dInfo),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, (S) nullptr, dE, dNev, dW, dIblock, dIsplit,
                                          dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD, (S) nullptr, dNev, dW, dIblock, dIsplit,
                                          dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD, dE, (U) nullptr, dW, dIblock, dIsplit, dZ,
                                          ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD, dE, dNev, (S) nullptr, dIblock, dIsplit,
                                          dZ, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, (U) nullptr, dIsplit, dZ, ldz, dIfail, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, (U) nullptr, dZ, ldz, dIfail, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, dIsplit,
                                          (T) nullptr, ldz, dIfail, dInfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, (U) nullptr, dInfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz,
                                          dIfail, (U) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, 0, (S) nullptr, (S) nullptr, dNev, (S) nullptr,
                                          (U) nullptr, (U) nullptr, (T) nullptr, ldz, (U) nullptr,
                                          dInfo),
                          rocblas_status_success);
}

template <typename T>
void testing_stein_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int ldz = 1;

    // memory allocations
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
    device_strided_batch_vector<S> dW(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIblock(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIsplit(1, 1, 1, 1);
    device_strided_batch_vector<T> dZ(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIfail(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dNev.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dIblock.memcheck());
    CHECK_HIP_ERROR(dIsplit.memcheck());
    CHECK_HIP_ERROR(dZ.memcheck());
    CHECK_HIP_ERROR(dIfail.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    stein_checkBadArgs(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(), dIblock.data(),
                       dIsplit.data(), dZ.data(), ldz, dIfail.data(), dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Ud, typename Sh, typename Uh>
void stein_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int nev,
                    Sd& dD,
                    Sd& dE,
                    Ud& dNev,
                    Sd& dW,
                    Ud& dIblock,
                    Ud& dIsplit,
                    Sh& hD,
                    Sh& hE,
                    Uh& hNev,
                    Sh& hW,
                    Uh& hIblock,
                    Uh& hIsplit)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        rocblas_init<S>(hD, true);
        rocblas_init<S>(hE, true);

        rocblas_int nsplit, info;
        size_t lwork = 4 * n;
        size_t liwork = 3 * n;
        std::vector<S> work(lwork);
        std::vector<rocblas_int> iwork(liwork);

        // scale matrix
        for(rocblas_int i = 0; i < n; i++)
        {
            hD[0][i] += 10;
            hE[0][i] -= 5;
            if(i == n / 4 || i == n / 2 || i == n - 1)
                hE[0][i] = 0;
            if(i == n / 7 || i == n / 5 || i == n / 3)
                hD[0][i] *= -1;
        }

        // compute a subset of the eigenvalues
        S il = n - nev + 1;
        S iu = n;
        S abstol = 2 * get_safemin<S>();
        cpu_stebz(rocblas_erange_index, rocblas_eorder_blocks, n, S(0), S(0), il, iu, abstol, hD[0],
                  hE[0], hNev[0], &nsplit, hW[0], hIblock[0], hIsplit[0], work.data(), iwork.data(),
                  &info);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));
        CHECK_HIP_ERROR(dNev.transfer_from(hNev));
        CHECK_HIP_ERROR(dW.transfer_from(hW));
        CHECK_HIP_ERROR(dIblock.transfer_from(hIblock));
        CHECK_HIP_ERROR(dIsplit.transfer_from(hIsplit));
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stein_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int nev,
                    Sd& dD,
                    Sd& dE,
                    Ud& dNev,
                    Sd& dW,
                    Ud& dIblock,
                    Ud& dIsplit,
                    Td& dZ,
                    const rocblas_int ldz,
                    Ud& dIfail,
                    Ud& dInfo,
                    Sh& hD,
                    Sh& hE,
                    Uh& hNev,
                    Sh& hW,
                    Uh& hIblock,
                    Uh& hIsplit,
                    Th& hZ,
                    Th& hZRes,
                    Uh& hIfail,
                    Uh& hIfailRes,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err)
{
    using S = decltype(std::real(T{}));

    size_t lwork = 5 * n;
    size_t liwork = n;
    size_t lifail = n;
    std::vector<S> work(lwork);
    std::vector<rocblas_int> iwork(liwork);

    // input data initialization
    stein_initData<true, true, T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev,
                                  hW, hIblock, hIsplit);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(),
                                        dIblock.data(), dIsplit.data(), dZ.data(), ldz,
                                        dIfail.data(), dInfo.data()));
    CHECK_HIP_ERROR(hZRes.transfer_from(dZ));
    CHECK_HIP_ERROR(hIfailRes.transfer_from(dIfail));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    cpu_stein(n, hD[0], hE[0], hNev[0], hW[0], hIblock[0], hIsplit[0], hZ[0], ldz, work.data(),
              iwork.data(), hIfail[0], hInfo[0]);

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    double err;

    if(hInfo[0][0] == 0)
    {
        rocblas_int nn = hNev[0][0];

        // check ifail
        err = 0;
        for(int j = 0; j < nn; j++)
        {
            EXPECT_EQ(hIfailRes[0][j], 0) << "j = " << j;
            if(hIfailRes[0][j] != 0)
                err++;
        }
        *max_err = err > *max_err ? err : *max_err;

        // need to implicitly test eigenvectors due to non-uniqueness of eigenvectors under scaling

        // Z should be orthonormal, if it is then Z^T*Z should be the identity
        if(nn > 0)
        {
            std::vector<T> ZZres(nn * nn, 0.0);
            std::vector<T> I(nn * nn, 0.0);

            for(rocblas_int i = 0; i < nn; i++)
                I[i + i * nn] = T(1);

            cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, nn, nn, n, T(1),
                     hZRes[0], ldz, hZRes[0], ldz, T(0), ZZres.data(), nn);
            err = norm_error('F', nn, nn, nn, I.data(), ZZres.data());
            *max_err = err > *max_err ? err : *max_err;
        }

        // for each of the nev eigenvalues w_j, verify that the associated eigenvector is in the
        // null space of (A - w_i * I)
        T alpha, t1, t2;
        for(int j = 0; j < nn; j++)
        {
            for(int i = 0; i < n; i++)
            {
                alpha = hW[0][j] - hD[0][i];
                hZ[0][i + j * ldz] = hZRes[0][i + j * ldz] * alpha;
            }
            t1 = hZRes[0][j * ldz];
            hZRes[0][j * ldz] = hE[0][0] * hZRes[0][1 + j * ldz];
            for(int i = 1; i < n - 1; i++)
            {
                t2 = hZRes[0][i + j * ldz];
                hZRes[0][i + j * ldz] = hE[0][i - 1] * t1 + hE[0][i] * hZRes[0][(i + 1) + j * ldz];
                t1 = t2;
            }
            hZRes[0][(n - 1) + j * ldz] = hE[0][n - 2] * t1;
        }

        // error is then ||hZ - hZRes|| / ||hZ||
        // using frobenius norm
        err = norm_error('F', n, nn, ldz, hZ[0], hZRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
    else
    {
        // check ifail
        err = 0;
        for(int j = 0; j < hInfo[0][0]; j++)
        {
            EXPECT_NE(hIfailRes[0][j], 0) << "j = " << j;
            if(hIfailRes[0][j] == 0)
                err++;
        }
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void stein_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       const rocblas_int nev,
                       Sd& dD,
                       Sd& dE,
                       Ud& dNev,
                       Sd& dW,
                       Ud& dIblock,
                       Ud& dIsplit,
                       Td& dZ,
                       const rocblas_int ldz,
                       Ud& dIfail,
                       Ud& dInfo,
                       Sh& hD,
                       Sh& hE,
                       Uh& hNev,
                       Sh& hW,
                       Uh& hIblock,
                       Uh& hIsplit,
                       Th& hZ,
                       Uh& hIfail,
                       Uh& hInfo,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    using S = decltype(std::real(T{}));

    size_t lwork = 5 * n;
    size_t liwork = n;
    size_t lifail = n;
    std::vector<S> work(lwork);
    std::vector<rocblas_int> iwork(liwork);

    if(!perf)
    {
        stein_initData<true, false, T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE,
                                       hNev, hW, hIblock, hIsplit);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_stein(n, hD[0], hE[0], hNev[0], hW[0], hIblock[0], hIsplit[0], hZ[0], ldz, work.data(),
                  iwork.data(), hIfail[0], hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stein_initData<true, false, T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE, hNev,
                                   hW, hIblock, hIsplit);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stein_initData<false, true, T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE,
                                       hNev, hW, hIblock, hIsplit);

        CHECK_ROCBLAS_ERROR(rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(),
                                            dIblock.data(), dIsplit.data(), dZ.data(), ldz,
                                            dIfail.data(), dInfo.data()));
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
        stein_initData<false, true, T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, hD, hE,
                                       hNev, hW, hIblock, hIsplit);

        start = get_time_us_sync(stream);
        rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(), dW.data(), dIblock.data(),
                        dIsplit.data(), dZ.data(), ldz, dIfail.data(), dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_stein(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nev = argus.get<rocblas_int>("nev", n < 5 ? n : 5);
    rocblas_int ldz = argus.get<rocblas_int>("ldz", n);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D = n;
    size_t size_E = size_D;
    size_t size_W = size_D;
    size_t size_iblock = size_D;
    size_t size_isplit = size_D;
    size_t size_Z = ldz * n;
    size_t size_ifail = size_D;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ZRes = (argus.unit_check || argus.norm_check) ? size_Z : 0;
    size_t size_ifailRes = (argus.unit_check || argus.norm_check) ? size_ifail : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || ldz < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_stein(handle, n, (S*)nullptr, (S*)nullptr, (rocblas_int*)nullptr, (S*)nullptr,
                            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, ldz,
                            (rocblas_int*)nullptr, (rocblas_int*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_stein(handle, n, (S*)nullptr, (S*)nullptr,
                                          (rocblas_int*)nullptr, (S*)nullptr, (rocblas_int*)nullptr,
                                          (rocblas_int*)nullptr, (T*)nullptr, ldz,
                                          (rocblas_int*)nullptr, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    // host
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<rocblas_int> hNev(1, 1, 1, 1);
    host_strided_batch_vector<S> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<rocblas_int> hIblock(size_iblock, 1, size_iblock, 1);
    host_strided_batch_vector<rocblas_int> hIsplit(size_isplit, 1, size_isplit, 1);
    host_strided_batch_vector<T> hZ(size_Z, 1, size_Z, 1);
    host_strided_batch_vector<T> hZRes(size_ZRes, 1, size_ZRes, 1);
    host_strided_batch_vector<rocblas_int> hIfail(size_ifail, 1, size_ifail, 1);
    host_strided_batch_vector<rocblas_int> hIfailRes(size_ifailRes, 1, size_ifailRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
    // device
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
    device_strided_batch_vector<S> dW(size_W, 1, size_W, 1);
    device_strided_batch_vector<rocblas_int> dIblock(size_iblock, 1, size_iblock, 1);
    device_strided_batch_vector<rocblas_int> dIsplit(size_isplit, 1, size_isplit, 1);
    device_strided_batch_vector<T> dZ(size_Z, 1, size_Z, 1);
    device_strided_batch_vector<rocblas_int> dIfail(size_ifail, 1, size_ifail, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dNev.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    if(size_iblock)
        CHECK_HIP_ERROR(dIblock.memcheck());
    if(size_isplit)
        CHECK_HIP_ERROR(dIsplit.memcheck());
    if(size_Z)
        CHECK_HIP_ERROR(dZ.memcheck());
    if(size_ifail)
        CHECK_HIP_ERROR(dIfail.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_stein(handle, n, dD.data(), dE.data(), dNev.data(),
                                              dW.data(), dIblock.data(), dIsplit.data(), dZ.data(),
                                              ldz, dIfail.data(), dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stein_getError<T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dIfail,
                          dInfo, hD, hE, hNev, hW, hIblock, hIsplit, hZ, hZRes, hIfail, hIfailRes,
                          hInfo, hInfoRes, &max_error);

    // collect performance data
    if(argus.timing)
        stein_getPerfData<T>(handle, n, nev, dD, dE, dNev, dW, dIblock, dIsplit, dZ, ldz, dIfail,
                             dInfo, hD, hE, hNev, hW, hIblock, hIsplit, hZ, hIfail, hInfo,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nev", "ldz");
            rocsolver_bench_output(n, nev, ldz);

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

#define EXTERN_TESTING_STEIN(...) extern template void testing_stein<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_STEIN, FOREACH_SCALAR_TYPE, APPLY_STAMP)
