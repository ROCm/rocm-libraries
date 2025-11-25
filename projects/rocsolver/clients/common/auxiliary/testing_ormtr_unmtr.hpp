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

template <bool COMPLEX, typename T>
void ormtr_unmtr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_side side,
                              const rocblas_fill uplo,
                              const rocblas_operation trans,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv,
                              T dC,
                              const rocblas_int ldc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr(nullptr, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr(handle, rocblas_side(0), uplo, trans, m, n, dA, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr(handle, side, rocblas_fill(0), trans, m, n, dA, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, side, uplo, rocblas_operation(0), m, n, dA,
                                                lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    if(COMPLEX)
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, side, uplo, rocblas_operation_transpose,
                                                    m, n, dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);
    else
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, side, uplo,
                                                    rocblas_operation_conjugate_transpose, m, n, dA,
                                                    lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, (T) nullptr, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, dA, lda, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, dA, lda, dIpiv, (T) nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, rocblas_side_left, uplo, trans, 0, n,
                                                (T) nullptr, lda, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, rocblas_side_right, uplo, trans, m, 0,
                                                (T) nullptr, lda, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
}

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormtr_unmtr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int lda = 2;
    rocblas_int ldc = 2;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    // check bad arguments
    ormtr_unmtr_checkBadArgs<COMPLEX>(handle, side, uplo, trans, m, n, dA.data(), lda, dIpiv.data(),
                                      dC.data(), ldc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormtr_unmtr_initData(const rocblas_handle handle,
                          const rocblas_side side,
                          const rocblas_fill uplo,
                          const rocblas_operation trans,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Td& dC,
                          const rocblas_int ldc,
                          Th& hA,
                          Th& hIpiv,
                          Th& hC,
                          std::vector<T>& hW,
                          size_t size_W)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        rocblas_int nq = (side == rocblas_side_left) ? m : n;
        std::vector<S> E(nq - 1);
        std::vector<S> D(nq);

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);
        rocblas_init<T>(hC, true);

        // scale to avoid singularities
        for(int i = 0; i < nq; ++i)
        {
            for(int j = 0; j < nq; ++j)
            {
                if(i == j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute sytrd/hetrd
        cpu_sytrd_hetrd(uplo, nq, hA[0], lda, D.data(), E.data(), hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Td, typename Th>
void ormtr_unmtr_getError(const rocblas_handle handle,
                          const rocblas_side side,
                          const rocblas_fill uplo,
                          const rocblas_operation trans,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          Td& dIpiv,
                          Td& dC,
                          const rocblas_int ldc,
                          Th& hA,
                          Th& hIpiv,
                          Th& hC,
                          Th& hCr,
                          double* max_err)
{
    size_t size_W = (side == rocblas_side_left ? m : n) * 32;
    std::vector<T> hW(size_W);

    // initialize data
    ormtr_unmtr_initData<true, true, T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc,
                                        hA, hIpiv, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, dA.data(), lda,
                                              dIpiv.data(), dC.data(), ldc));
    CHECK_HIP_ERROR(hCr.transfer_from(dC));

    // CPU lapack
    cpu_ormtr_unmtr(side, uplo, trans, m, n, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(), size_W);

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCr[0]);
}

template <typename T, typename Td, typename Th>
void ormtr_unmtr_getPerfData(const rocblas_handle handle,
                             const rocblas_side side,
                             const rocblas_fill uplo,
                             const rocblas_operation trans,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             Td& dIpiv,
                             Td& dC,
                             const rocblas_int ldc,
                             Th& hA,
                             Th& hIpiv,
                             Th& hC,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    size_t size_W = (side == rocblas_side_left ? m : n) * 32;
    std::vector<T> hW(size_W);

    if(!perf)
    {
        ormtr_unmtr_initData<true, false, T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC,
                                             ldc, hA, hIpiv, hC, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_ormtr_unmtr(side, uplo, trans, m, n, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(), size_W);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    ormtr_unmtr_initData<true, false, T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc,
                                         hA, hIpiv, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormtr_unmtr_initData<false, true, T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC,
                                             ldc, hA, hIpiv, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, dA.data(), lda,
                                                  dIpiv.data(), dC.data(), ldc));
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

    for(int iter = 0; iter < hot_calls; iter++)
    {
        ormtr_unmtr_initData<false, true, T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC,
                                             ldc, hA, hIpiv, hC, hW, size_W);

        start = get_time_us_sync(stream);
        rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, dA.data(), lda, dIpiv.data(),
                              dC.data(), ldc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormtr_unmtr(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char uploC = argus.get<char>("uplo");
    char transC = argus.get<char>("trans");
    rocblas_int m, n;
    if(sideC == 'L')
    {
        m = argus.get<rocblas_int>("m");
        n = argus.get<rocblas_int>("n", m);
    }
    else
    {
        n = argus.get<rocblas_int>("n");
        m = argus.get<rocblas_int>("m", n);
    }
    rocblas_int nq = (sideC == 'L' ? m : n);
    rocblas_int lda = argus.get<rocblas_int>("lda", nq);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", m);

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value
        = (side == rocblas_side_both || (COMPLEX && trans == rocblas_operation_transpose)
           || (!COMPLEX && trans == rocblas_operation_conjugate_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, (T*)nullptr,
                                                    lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    size_t size_P = size_t(nq);
    size_t size_C = size_t(ldc) * n;

    size_t size_A = size_t(lda) * nq;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || ldc < m || lda < nq);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, (T*)nullptr,
                                                    lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, (T*)nullptr, lda,
                                                (T*)nullptr, (T*)nullptr, ldc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCr(size_Cr, 1, size_Cr, 1);
    host_strided_batch_vector<T> hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T> dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormtr_unmtr(handle, side, uplo, trans, m, n, dA.data(), lda,
                                                    dIpiv.data(), dC.data(), ldc),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormtr_unmtr_getError<T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA, hIpiv,
                                hC, hCr, &max_error);

    // collect performance data
    if(argus.timing)
        ormtr_unmtr_getPerfData<T>(handle, side, uplo, trans, m, n, dA, lda, dIpiv, dC, ldc, hA,
                                   hIpiv, hC, &gpu_time_used, &cpu_time_used, hot_calls,
                                   argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    rocblas_int s = left ? m : n;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("side", "uplo", "trans", "m", "n", "lda", "ldc");
            rocsolver_bench_output(sideC, uploC, transC, m, n, lda, ldc);

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

#define EXTERN_TESTING_ORMTR_UNMTR(...) \
    extern template void testing_ormtr_unmtr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORMTR_UNMTR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
