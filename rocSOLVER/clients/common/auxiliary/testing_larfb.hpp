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

template <typename T>
void larfb_checkBadArgs(const rocblas_handle handle,
                        const rocblas_side side,
                        const rocblas_operation trans,
                        const rocblas_direct direct,
                        const rocblas_storev storev,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int k,
                        T dV,
                        const rocblas_int ldv,
                        T dT,
                        const rocblas_int ldt,
                        T dA,
                        const rocblas_int lda)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_larfb(nullptr, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt, dA, lda),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, rocblas_side(0), trans, direct, storev, m, n, k,
                                          dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, rocblas_operation(0), direct, storev, m, n,
                                          k, dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, rocblas_direct(0), storev, m, n, k,
                                          dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, rocblas_storev(0), m, n, k,
                                          dV, ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, (T) nullptr,
                                          ldv, dT, ldt, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV, ldv,
                                          (T) nullptr, ldt, dA, lda),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                          ldt, (T) nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, rocblas_side_left, trans, direct, storev, 0, n, k,
                                          (T) nullptr, ldv, dT, ldt, (T) nullptr, lda),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, rocblas_side_right, trans, direct, storev, m, 0,
                                          k, (T) nullptr, ldv, dT, ldt, (T) nullptr, lda),
                          rocblas_status_success);
}

template <typename T>
void testing_larfb_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_direct direct = rocblas_forward_direction;
    rocblas_storev storev = rocblas_column_wise;
    rocblas_int k = 1;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int ldv = 1;
    rocblas_int ldt = 1;
    rocblas_int lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dT(1, 1, 1, 1);
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dT.memcheck());
    CHECK_HIP_ERROR(dA.memcheck());

    // check bad arguments
    larfb_checkBadArgs(handle, side, trans, direct, storev, m, n, k, dV.data(), ldv, dT.data(), ldt,
                       dA.data(), lda);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void larfb_initData(const rocblas_handle handle,
                    const rocblas_side side,
                    const rocblas_operation trans,
                    const rocblas_direct direct,
                    const rocblas_storev storev,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hV,
                    Th& hT,
                    Th& hA,
                    std::vector<T>& hW,
                    size_t sizeW)
{
    if(CPU)
    {
        bool left = (side == rocblas_side_left);
        bool forward = (direct == rocblas_forward_direction);
        bool column = (storev == rocblas_column_wise);
        std::vector<T> htau(k);

        rocblas_init<T>(hV, true);
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hT, true);

        // scale to avoid singularities
        // create householder reflectors and triangular factor
        if(left)
        {
            if(column)
            {
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < k; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cpu_geqrf(m, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cpu_geqlf(m, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }
            else
            {
                for(int i = 0; i < k; ++i)
                {
                    for(int j = 0; j < m; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cpu_gelqf(k, m, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cpu_gerqf(k, m, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }

            cpu_larft(direct, storev, m, k, hV[0], ldv, htau.data(), hT[0], ldt);
        }
        else
        {
            if(column)
            {
                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < k; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cpu_geqrf(n, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cpu_geqlf(n, k, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }
            else
            {
                for(int i = 0; i < k; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        if(i == j)
                            hV[0][i + j * ldv] += 400;
                        else
                            hV[0][i + j * ldv] -= 4;
                    }
                }

                if(forward)
                    cpu_gelqf(k, n, hV[0], ldv, htau.data(), hW.data(), sizeW);
                else
                    cpu_gerqf(k, n, hV[0], ldv, htau.data(), hW.data(), sizeW);
            }

            cpu_larft(direct, storev, n, k, hV[0], ldv, htau.data(), hT[0], ldt);
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dT.transfer_from(hT));
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Th>
void larfb_getError(const rocblas_handle handle,
                    const rocblas_side side,
                    const rocblas_operation trans,
                    const rocblas_direct direct,
                    const rocblas_storev storev,
                    const rocblas_int m,
                    const rocblas_int n,
                    const rocblas_int k,
                    Td& dV,
                    const rocblas_int ldv,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hV,
                    Th& hT,
                    Th& hA,
                    Th& hAr,
                    double* max_err)
{
    bool left = (side == rocblas_side_left);
    rocblas_int ldw = left ? n : m;
    size_t sizeW = size_t(ldw) * k;
    std::vector<T> hW(sizeW);

    // initialize data
    larfb_initData<true, true, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt,
                                  dA, lda, hV, hT, hA, hW, sizeW);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV.data(),
                                        ldv, dT.data(), ldt, dA.data(), lda));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cpu_larfb(side, trans, direct, storev, m, n, k, hV[0], ldv, hT[0], ldt, hA[0], lda, hW.data(),
              ldw);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <typename T, typename Td, typename Th>
void larfb_getPerfData(const rocblas_handle handle,
                       const rocblas_side side,
                       const rocblas_operation trans,
                       const rocblas_direct direct,
                       const rocblas_storev storev,
                       const rocblas_int m,
                       const rocblas_int n,
                       const rocblas_int k,
                       Td& dV,
                       const rocblas_int ldv,
                       Td& dT,
                       const rocblas_int ldt,
                       Td& dA,
                       const rocblas_int lda,
                       Th& hV,
                       Th& hT,
                       Th& hA,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    bool left = (side == rocblas_side_left);
    rocblas_int ldw = left ? n : m;
    size_t sizeW = size_t(ldw) * k;
    std::vector<T> hW(sizeW);

    if(!perf)
    {
        larfb_initData<true, false, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                       ldt, dA, lda, hV, hT, hA, hW, sizeW);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_larfb(side, trans, direct, storev, m, n, k, hV[0], ldv, hT[0], ldt, hA[0], lda,
                  hW.data(), ldw);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    larfb_initData<true, false, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt,
                                   dA, lda, hV, hT, hA, hW, sizeW);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        larfb_initData<false, true, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                       ldt, dA, lda, hV, hT, hA, hW, sizeW);

        CHECK_ROCBLAS_ERROR(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV.data(),
                                            ldv, dT.data(), ldt, dA.data(), lda));
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
        larfb_initData<false, true, T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT,
                                       ldt, dA, lda, hV, hT, hA, hW, sizeW);

        start = get_time_us_sync(stream);
        rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, dV.data(), ldv, dT.data(),
                        ldt, dA.data(), lda);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_larfb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    char transC = argus.get<char>("trans");
    char directC = argus.get<char>("direct");
    char storevC = argus.get<char>("storev");
    rocblas_int k = argus.get<rocblas_int>("k");
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", storevC == 'R' ? k : (sideC == 'L' ? m : n));
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_int ldt = argus.get<rocblas_int>("ldt", k);

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_direct direct = char2rocblas_direct(directC);
    rocblas_storev storev = char2rocblas_storev(storevC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k,
                                              (T*)nullptr, ldv, (T*)nullptr, ldt, (T*)nullptr, lda),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    bool row = (storev == rocblas_row_wise);
    bool left = (side == rocblas_side_left);

    size_t size_V = size_t(ldv) * k;
    if(row)
        size_V = left ? size_t(ldv) * m : size_t(ldv) * n;

    size_t size_T = size_t(ldt) * k;
    size_t size_A = size_t(lda) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || k < 1 || ldt < k || lda < m || (row && ldv < k)
                         || (!row && !left && ldv < n) || (!row && left && ldv < m));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k,
                                              (T*)nullptr, ldv, (T*)nullptr, ldt, (T*)nullptr, lda),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k, (T*)nullptr,
                                          ldv, (T*)nullptr, ldt, (T*)nullptr, lda));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hT(size_T, 1, size_T, 1);
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dT(size_T, 1, size_T, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_T)
        CHECK_HIP_ERROR(dT.memcheck());
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_larfb(handle, side, trans, direct, storev, m, n, k,
                                              dV.data(), ldv, dT.data(), ldt, dA.data(), lda),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        larfb_getError<T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt, dA, lda,
                          hV, hT, hA, hAr, &max_error);

    // collect performance data
    if(argus.timing)
        larfb_getPerfData<T>(handle, side, trans, direct, storev, m, n, k, dV, ldv, dT, ldt, dA,
                             lda, hV, hT, hA, &gpu_time_used, &cpu_time_used, hot_calls,
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
            rocsolver_bench_output("side", "trans", "direct", "storev", "m", "n", "k", "ldv", "ldt",
                                   "lda");
            rocsolver_bench_output(sideC, transC, directC, storevC, m, n, k, ldv, ldt, lda);

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

#define EXTERN_TESTING_LARFB(...) extern template void testing_larfb<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LARFB, FOREACH_SCALAR_TYPE, APPLY_STAMP)
