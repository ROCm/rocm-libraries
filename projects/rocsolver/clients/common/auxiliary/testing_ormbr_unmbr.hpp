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

template <bool COMPLEX, typename T>
void ormbr_unmbr_checkBadArgs(const rocblas_handle handle,
                              const rocblas_storev storev,
                              const rocblas_side side,
                              const rocblas_operation trans,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int k,
                              T dA,
                              const rocblas_int lda,
                              T dIpiv,
                              T dC,
                              const rocblas_int ldc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormbr_unmbr(nullptr, storev, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side(0), trans, m, n, k, dA,
                                                lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, rocblas_storev(0), side, trans, m, n, k, dA,
                                                lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, rocblas_operation(0), m, n, k,
                                                dA, lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_value);
    if(COMPLEX)
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, rocblas_operation_transpose,
                                                    m, n, k, dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);
    else
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side,
                                                    rocblas_operation_conjugate_transpose, m, n, k,
                                                    dA, lda, dIpiv, dC, ldc),
                              rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, (T) nullptr,
                                                lda, dIpiv, dC, ldc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA, lda, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA, lda,
                                                dIpiv, (T) nullptr, ldc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side_left, trans, 0, n, k,
                                                (T) nullptr, lda, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side_right, trans, m, 0, k,
                                                (T) nullptr, lda, (T) nullptr, (T) nullptr, ldc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, rocblas_side_left, trans, m, n, 0,
                                                (T) nullptr, lda, (T) nullptr, dC, ldc),
                          rocblas_status_success);
}

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormbr_unmbr_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_int k = 1;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldc = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    // check bad arguments
    ormbr_unmbr_checkBadArgs<COMPLEX>(handle, storev, side, trans, m, n, k, dA.data(), lda,
                                      dIpiv.data(), dC.data(), ldc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormbr_unmbr_initData(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
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
        size_t s = std::max(hIpiv.n(), int64_t(2));
        std::vector<S> E(s - 1);
        std::vector<S> D(s);
        std::vector<T> P(s);
        rocblas_int nq = (side == rocblas_side_left) ? m : n;

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);
        rocblas_init<T>(hC, true);

        // scale to avoid singularities
        // and compute gebrd
        if(storev == rocblas_column_wise)
        {
            for(int i = 0; i < nq; ++i)
            {
                for(int j = 0; j < s; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cpu_gebrd(nq, s, hA[0], lda, D.data(), E.data(), hIpiv[0], P.data(), hW.data(), size_W);
        }
        else
        {
            for(int i = 0; i < s; ++i)
            {
                for(int j = 0; j < nq; ++j)
                {
                    if(i == j)
                        hA[0][i + j * lda] += 400;
                    else
                        hA[0][i + j * lda] -= 4;
                }
            }
            cpu_gebrd(s, nq, hA[0], lda, D.data(), E.data(), P.data(), hIpiv[0], hW.data(), size_W);
        }
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
void ormbr_unmbr_getError(const rocblas_handle handle,
                          const rocblas_storev storev,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_int m,
                          const rocblas_int n,
                          const rocblas_int k,
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
    size_t size_W = std::max(std::max(m, n), k);
    std::vector<T> hW(size_W);

    // initialize data
    ormbr_unmbr_initData<true, true, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC,
                                        ldc, hA, hIpiv, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(), lda,
                                              dIpiv.data(), dC.data(), ldc));
    CHECK_HIP_ERROR(hCr.transfer_from(dC));

    // CPU lapack
    cpu_ormbr_unmbr(storev, side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(),
                    size_W);

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCr[0]);
}

template <typename T, typename Td, typename Th>
void ormbr_unmbr_getPerfData(const rocblas_handle handle,
                             const rocblas_storev storev,
                             const rocblas_side side,
                             const rocblas_operation trans,
                             const rocblas_int m,
                             const rocblas_int n,
                             const rocblas_int k,
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
    size_t size_W = std::max(std::max(m, n), k);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        ormbr_unmbr_initData<true, false, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv,
                                             dC, ldc, hA, hIpiv, hC, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_ormbr_unmbr(storev, side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(),
                        size_W);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    ormbr_unmbr_initData<true, false, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC,
                                         ldc, hA, hIpiv, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormbr_unmbr_initData<false, true, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv,
                                             dC, ldc, hA, hIpiv, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(),
                                                  lda, dIpiv.data(), dC.data(), ldc));
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
        ormbr_unmbr_initData<false, true, T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv,
                                             dC, ldc, hA, hIpiv, hC, hW, size_W);

        start = get_time_us_sync(stream);
        rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(), lda, dIpiv.data(),
                              dC.data(), ldc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormbr_unmbr(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char storevC = argus.get<char>("storev");
    char sideC = argus.get<char>("side");
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
    rocblas_int k = argus.get<rocblas_int>("k", std::min(m, n));
    rocblas_int nq = (sideC == 'L' ? m : n);
    rocblas_int lda = argus.get<rocblas_int>("lda", storevC == 'C' ? nq : std::min(nq, k));
    rocblas_int ldc = argus.get<rocblas_int>("ldc", m);

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_storev storev = char2rocblas_storev(storevC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value
        = (side == rocblas_side_both || (COMPLEX && trans == rocblas_operation_transpose)
           || (!COMPLEX && trans == rocblas_operation_conjugate_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k,
                                                    (T*)nullptr, lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    bool left = (side == rocblas_side_left);
    size_t size_P = size_t(std::min(nq, k));
    size_t size_C = size_t(ldc) * n;

    bool row = (storev == rocblas_row_wise);
    size_t size_A = row ? size_t(lda) * nq : size_t(lda) * size_P;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = ((m < 0 || n < 0 || k < 0 || ldc < m) || (row && lda < std::min(nq, k))
                         || (!row && lda < nq));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k,
                                                    (T*)nullptr, lda, (T*)nullptr, (T*)nullptr, ldc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, (T*)nullptr,
                                                lda, (T*)nullptr, (T*)nullptr, ldc));

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
    if(n == 0 || m == 0 || k == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ormbr_unmbr(handle, storev, side, trans, m, n, k, dA.data(),
                                                    lda, dIpiv.data(), dC.data(), ldc),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormbr_unmbr_getError<T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA,
                                hIpiv, hC, hCr, &max_error);

    // collect performance data
    if(argus.timing)
        ormbr_unmbr_getPerfData<T>(handle, storev, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc,
                                   hA, hIpiv, hC, &gpu_time_used, &cpu_time_used, hot_calls,
                                   argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    rocblas_int s = left ? m : n;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("storev", "side", "trans", "m", "n", "k", "lda", "ldc");
            rocsolver_bench_output(storevC, sideC, transC, m, n, k, lda, ldc);

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

#define EXTERN_TESTING_ORMBR_UNMBR(...) \
    extern template void testing_ormbr_unmbr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORMBR_UNMBR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
