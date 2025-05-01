/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
void labrd_checkBadArgs(const rocblas_handle handle,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int nb,
                        T dA,
                        const rocblas_int lda,
                        S dD,
                        S dE,
                        U dTauq,
                        U dTaup,
                        T dX,
                        const rocblas_int ldx,
                        T dY,
                        const rocblas_int ldy)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(nullptr, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, (T) nullptr, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, (S) nullptr, dE, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, (S) nullptr, dTauq, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, (U) nullptr, dTaup, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, (U) nullptr, dX, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, (T) nullptr, ldx, dY, ldy),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_labrd(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, (T) nullptr, ldy),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, 0, n, 0, (T) nullptr, lda, dD, dE, dTauq, dTaup,
                                          (T) nullptr, ldx, dY, ldy),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, 0, 0, (T) nullptr, lda, dD, dE, dTauq, dTaup,
                                          dX, ldx, (T) nullptr, ldy),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, 0, dA, lda, (S) nullptr, (S) nullptr,
                                          (U) nullptr, (U) nullptr, (T) nullptr, ldx, (T) nullptr,
                                          ldy),
                          rocblas_status_success);
}

template <typename T>
void testing_labrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int nb = 1;
    rocblas_int lda = 1;
    rocblas_int ldx = 1;
    rocblas_int ldy = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dD(1, 1, 1, 1);
    device_strided_batch_vector<S> dE(1, 1, 1, 1);
    device_strided_batch_vector<T> dTauq(1, 1, 1, 1);
    device_strided_batch_vector<T> dTaup(1, 1, 1, 1);
    device_strided_batch_vector<T> dX(1, 1, 1, 1);
    device_strided_batch_vector<T> dY(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dTauq.memcheck());
    CHECK_HIP_ERROR(dTaup.memcheck());
    CHECK_HIP_ERROR(dX.memcheck());
    CHECK_HIP_ERROR(dY.memcheck());

    // check bad arguments
    labrd_checkBadArgs(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(), dTauq.data(),
                       dTaup.data(), dX.data(), ldx, dY.data(), ldy);
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void labrd_initData(const rocblas_handle handle,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Sd& dD,
                    Sd& dE,
                    Ud& dTauq,
                    Ud& dTaup,
                    Td& dX,
                    const rocblas_int ldx,
                    Td& dY,
                    const rocblas_int ldy,
                    Th& hA,
                    Sh& hD,
                    Sh& hE,
                    Uh& hTauq,
                    Uh& hTaup,
                    Th& hX,
                    Th& hY)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int i = 0; i < m; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j || (m >= n && j == i + 1) || (m < n && i == j + 1))
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void labrd_getError(const rocblas_handle handle,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Sd& dD,
                    Sd& dE,
                    Ud& dTauq,
                    Ud& dTaup,
                    Td& dX,
                    const rocblas_int ldx,
                    Td& dY,
                    const rocblas_int ldy,
                    Th& hA,
                    Th& hARes,
                    Sh& hD,
                    Sh& hE,
                    Uh& hTauq,
                    Uh& hTaup,
                    Th& hX,
                    Th& hXRes,
                    Th& hY,
                    Th& hYRes,
                    double* max_err)
{
    // input data initialization
    labrd_initData<true, true, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy,
                                  hA, hD, hE, hTauq, hTaup, hX, hY);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(),
                                        dTauq.data(), dTaup.data(), dX.data(), ldx, dY.data(), ldy));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));
    CHECK_HIP_ERROR(hYRes.transfer_from(dY));

    // CPU lapack
    cpu_labrd(m, n, nb, hA[0], lda, hD[0], hE[0], hTauq[0], hTaup[0], hX[0], ldx, hY[0], ldy);

    // error is max(||hA - hARes|| / ||hA||, ||hX - hXRes|| / ||hX||, ||hY -
    // hYRes|| / ||hY||) (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F', m, n, lda, hA[0], hARes[0]);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', m - nb, nb, ldx, hX[0] + nb, hXRes[0] + nb);
    *max_err = err > *max_err ? err : *max_err;
    err = norm_error('F', n - nb, nb, ldy, hY[0] + nb, hYRes[0] + nb);
    *max_err = err > *max_err ? err : *max_err;
}

template <typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void labrd_getPerfData(const rocblas_handle handle,
                       const rocblas_int m,
                       const rocblas_int n,
                       const rocblas_int nb,
                       Td& dA,
                       const rocblas_int lda,
                       Sd& dD,
                       Sd& dE,
                       Ud& dTauq,
                       Ud& dTaup,
                       Td& dX,
                       const rocblas_int ldx,
                       Td& dY,
                       const rocblas_int ldy,
                       Th& hA,
                       Sh& hD,
                       Sh& hE,
                       Uh& hTauq,
                       Uh& hTaup,
                       Th& hX,
                       Th& hY,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        labrd_initData<true, false, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY,
                                       ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        memset(hX[0], 0, ldx * nb * sizeof(T));
        memset(hY[0], 0, ldy * nb * sizeof(T));
        cpu_labrd(m, n, nb, hA[0], lda, hD[0], hE[0], hTauq[0], hTaup[0], hX[0], ldx, hY[0], ldy);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    labrd_initData<true, false, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY,
                                   ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        labrd_initData<false, true, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY,
                                       ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

        CHECK_ROCBLAS_ERROR(rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(),
                                            dTauq.data(), dTaup.data(), dX.data(), ldx, dY.data(),
                                            ldy));
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
        labrd_initData<false, true, T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY,
                                       ldy, hA, hD, hE, hTauq, hTaup, hX, hY);

        start = get_time_us_sync(stream);
        rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(), dE.data(), dTauq.data(),
                        dTaup.data(), dX.data(), ldx, dY.data(), ldy);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_labrd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int nb = argus.get<rocblas_int>("k", std::min(m, n));
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_int ldx = argus.get<rocblas_int>("ldx", m);
    rocblas_int ldy = argus.get<rocblas_int>("ldy", n);

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_D = nb;
    size_t size_E = nb;
    size_t size_Q = nb;
    size_t size_P = nb;
    size_t size_X = ldx * nb;
    size_t size_Y = ldy * nb;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;
    size_t size_YRes = (argus.unit_check || argus.norm_check) ? size_Y : 0;

    // check invalid sizes
    bool invalid_size
        = (m < 0 || n < 0 || nb < 0 || nb > std::min(m, n) || lda < m || ldx < m || ldy < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, nb, (T*)nullptr, lda, (S*)nullptr,
                                              (S*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr,
                                              ldx, (T*)nullptr, ldy),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_labrd(handle, m, n, nb, (T*)nullptr, lda, (S*)nullptr,
                                          (S*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr, ldx,
                                          (T*)nullptr, ldy));

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
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<S> hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S> hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<T> hTauq(size_Q, 1, size_Q, 1);
    host_strided_batch_vector<T> hTaup(size_P, 1, size_P, 1);
    host_strided_batch_vector<T> hX(size_X, 1, size_X, 1);
    host_strided_batch_vector<T> hXRes(size_XRes, 1, size_XRes, 1);
    host_strided_batch_vector<T> hY(size_Y, 1, size_Y, 1);
    host_strided_batch_vector<T> hYRes(size_YRes, 1, size_YRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T> dTauq(size_Q, 1, size_Q, 1);
    device_strided_batch_vector<T> dTaup(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dX(size_X, 1, size_X, 1);
    device_strided_batch_vector<T> dY(size_Y, 1, size_Y, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_Q)
        CHECK_HIP_ERROR(dTauq.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dTaup.memcheck());
    if(size_X)
        CHECK_HIP_ERROR(dX.memcheck());
    if(size_Y)
        CHECK_HIP_ERROR(dY.memcheck());

    // check quick return
    if(m == 0 || n == 0 || nb == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_labrd(handle, m, n, nb, dA.data(), lda, dD.data(),
                                              dE.data(), dTauq.data(), dTaup.data(), dX.data(), ldx,
                                              dY.data(), ldy),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        labrd_getError<T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy, hA,
                          hARes, hD, hE, hTauq, hTaup, hX, hXRes, hY, hYRes, &max_error);

    // collect performance data
    if(argus.timing)
        labrd_getPerfData<T>(handle, m, n, nb, dA, lda, dD, dE, dTauq, dTaup, dX, ldx, dY, ldy, hA,
                             hD, hE, hTauq, hTaup, hX, hY, &gpu_time_used, &cpu_time_used,
                             hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using nb * max(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, nb * std::max(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("m", "n", "nb", "lda", "ldx", "ldy");
            rocsolver_bench_output(m, n, nb, lda, ldx, ldy);
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

#define EXTERN_TESTING_LABRD(...) extern template void testing_labrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LABRD, FOREACH_SCALAR_TYPE, APPLY_STAMP)
