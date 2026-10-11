/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <limits>

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <typename T, typename I, typename S>
void pocon_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const I n,
                        T dA,
                        const I lda,
                        const S* danorm,
                        S* drcond)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_pocon(nullptr, uplo, n, dA, lda, danorm, drcond),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_pocon(handle, rocblas_fill_full, n, dA, lda, danorm, drcond),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_pocon(handle, uplo, n, (T) nullptr, lda, danorm, drcond),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_pocon(handle, uplo, n, dA, lda, (S*)nullptr, drcond),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_pocon(handle, uplo, n, dA, lda, danorm, (S*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_pocon(handle, uplo, (I)0, (T) nullptr, lda, (S*)nullptr, (S*)nullptr),
        rocblas_status_success);
}

template <typename T, typename I>
void testing_pocon_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> danorm(1, 1, 1, 1);
    device_strided_batch_vector<S> drcond(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(danorm.memcheck());
    CHECK_HIP_ERROR(drcond.memcheck());

    // check bad arguments
    pocon_checkBadArgs(handle, uplo, n, dA.data(), lda, danorm.data(), drcond.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void pocon_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& danorm,
                    Th& hA,
                    Sh& hanorm)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale to ensure positive definiteness
        for(I i = 0; i < n; i++)
            hA[0][i + i * lda] = hA[0][i + i * lda] * sconj(hA[0][i + i * lda]) * 400;

        // 1-norm of the symmetric/Hermitian matrix
        std::vector<S> work(n);
        char uploC = rocblas2char_fill(uplo);
        if constexpr(rocblas_is_complex<T>)
            hanorm[0][0] = cpu_lanhe<T, S>('1', uploC, n, hA[0], lda, work.data());
        else
            hanorm[0][0] = cpu_lansy<T, S>('1', uploC, n, hA[0], lda, work.data());

        // factor matrix with CPU POTRF (in place)
        rocblas_int info;
        cpu_potrf(uplo, n, hA[0], lda, &info);
    }

    if(GPU)
    {
        // copy factored data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(danorm.transfer_from(hanorm));
    }
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void pocon_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& danorm,
                    Sd& drcond,
                    Th& hA,
                    Sh& hanorm,
                    Sh& hrcond,
                    Sh& hrcond_res,
                    double* max_err)
{
    // initialize data
    pocon_initData<true, true, T, I, S>(handle, uplo, n, dA, lda, danorm, hA, hanorm);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_pocon(handle, uplo, n, dA.data(), lda, danorm.data(), drcond.data()));
    CHECK_HIP_ERROR(hrcond_res.transfer_from(drcond));

    // CPU lapack
    std::vector<T> work(3 * n);
    std::vector<S> rwork(n);
    std::vector<rocblas_int> iwork(n);
    hrcond[0][0] = cpu_pocon<T, S>(uplo, n, hA[0], lda, hanorm[0][0], work.data(), rwork.data(),
                                   iwork.data());

    // error is |hrcond - hrcond_res| / |hrcond|
    // (an exact zero is expected if hrcond is zero)
    S gold = hrcond[0][0];
    S comp = hrcond_res[0][0];
    if(gold == comp)
        *max_err = 0;
    else if(gold == 0 || !std::isfinite(comp))
        *max_err = std::numeric_limits<double>::infinity();
    else
        *max_err = std::abs(double(gold) - double(comp)) / std::abs(double(gold));
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void pocon_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const I n,
                       Td& dA,
                       const I lda,
                       Sd& danorm,
                       Sd& drcond,
                       Th& hA,
                       Sh& hanorm,
                       Sh& hrcond,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    // only init CPU data once as it is not overwritten
    pocon_initData<true, false, T, I, S>(handle, uplo, n, dA, lda, danorm, hA, hanorm);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        std::vector<T> work(3 * n);
        std::vector<S> rwork(n);
        std::vector<rocblas_int> iwork(n);
        *cpu_time_used = get_time_us_no_sync();
        hrcond[0][0] = cpu_pocon<T, S>(uplo, n, hA[0], lda, hanorm[0][0], work.data(), rwork.data(),
                                       iwork.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        pocon_initData<false, true, T, I, S>(handle, uplo, n, dA, lda, danorm, hA, hanorm);

        CHECK_ROCBLAS_ERROR(
            rocsolver_pocon(handle, uplo, n, dA.data(), lda, danorm.data(), drcond.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(int iter = 0; iter < hot_calls; iter++)
    {
        pocon_initData<false, true, T, I, S>(handle, uplo, n, dA, lda, danorm, hA, hanorm);

        timer.start(stream);
        rocsolver_pocon(handle, uplo, n, dA.data(), lda, danorm.data(), drcond.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, typename I>
void testing_pocon(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    I n = argus.get<I>("n");
    I lda = argus.get<I>("lda", n);

    // the handle can be in device pointer mode (the API scalars are on the device in both modes)
    if(argus.get<char>("pointer_mode", 'H') == 'D')
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_pocon(handle, uplo, n, (T*)nullptr, lda, (S*)nullptr, (S*)nullptr),
            rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_anorm = 1;
    size_t size_rcond = 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_rcond_res = (argus.unit_check || argus.norm_check) ? size_rcond : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_pocon(handle, uplo, n, (T*)nullptr, lda, (S*)nullptr, (S*)nullptr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(
            rocsolver_pocon(handle, uplo, n, (T*)nullptr, lda, (S*)nullptr, (S*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<S> hanorm(size_anorm, 1, size_anorm, 1);
    host_strided_batch_vector<S> hrcond(size_rcond, 1, size_rcond, 1);
    host_strided_batch_vector<S> hrcond_res(size_rcond_res, 1, size_rcond_res, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> danorm(size_anorm, 1, size_anorm, 1);
    device_strided_batch_vector<S> drcond(size_rcond, 1, size_rcond, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(danorm.memcheck());
    CHECK_HIP_ERROR(drcond.memcheck());

    // check quick return
    if(n == 0)
    {
        // rcond must be set to one
        hrcond[0][0] = S(-1);
        CHECK_HIP_ERROR(drcond.transfer_from(hrcond));

        EXPECT_ROCBLAS_STATUS(
            rocsolver_pocon(handle, uplo, n, dA.data(), lda, danorm.data(), drcond.data()),
            rocblas_status_success);

        CHECK_HIP_ERROR(hrcond.transfer_from(drcond));
        EXPECT_EQ(hrcond[0][0], S(1));

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        pocon_getError<T, I, S>(handle, uplo, n, dA, lda, danorm, drcond, hA, hanorm, hrcond,
                                hrcond_res, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        pocon_getPerfData<T, I, S>(handle, uplo, n, dA, lda, danorm, drcond, hA, hanorm, hrcond,
                                   &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                                   argus.profile_kernels, argus.perf);

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
            rocsolver_bench_output("uplo", "n", "lda");
            rocsolver_bench_output(uploC, n, lda);

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

#define EXTERN_TESTING_POCON(...) extern template void testing_pocon<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_POCON, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
