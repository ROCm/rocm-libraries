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
#include "common/misc/rocsolver_timer.hpp"

template <bool COMPLEX, typename T>
void ormtr_unmtr_hb2st_checkBadArgs(const rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int kd,
                                    T dV,
                                    const rocblas_int ldv,
                                    T dTau,
                                    T dC,
                                    const rocblas_int ldc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            nullptr, side, trans, m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, rocblas_side(0), trans, m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, rocblas_operation(0), m, n, kd, dV, ldv, dTau, dC, ldc),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd, (T) nullptr, ldv, dTau, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd, dV, ldv, (T) nullptr, dC, ldc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd, dV, ldv, dTau, (T) nullptr, ldc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, 0, n, kd,
            (T) nullptr, ldv, (T) nullptr, (T) nullptr, ldc),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, 0, kd,
            (T) nullptr, ldv, (T) nullptr, (T) nullptr, ldc),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, 0,
            (T) nullptr, ldv, (T) nullptr, (T) nullptr, ldc),
        rocblas_status_success);
}

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormtr_unmtr_hb2st_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_left;
    rocblas_operation trans = rocblas_operation_conjugate_transpose;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int kd = 1;
    rocblas_int ldv = 2;
    rocblas_int ldc = 2;

    // memory allocation
    device_strided_batch_vector<T> dV(1, 1, 1, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    // check bad arguments
    ormtr_unmtr_hb2st_checkBadArgs<COMPLEX>(
        handle, side, trans, m, n, kd,
        dV.data(), ldv, dTau.data(), dC.data(), ldc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormtr_unmtr_hb2st_initData(const rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int kd,
                                Td& dV,
                                const rocblas_int ldv,
                                Td& dTau,
                                Td& dC,
                                const rocblas_int ldc,
                                Th& hV,
                                Th& hTau,
                                Th& hC,
                                std::vector<T>& hW,
                                size_t size_W)
{
    if(CPU)
    {
        rocblas_init<T>(hV, true);
        rocblas_init<T>(hTau, true);
        rocblas_init<T>(hC, true);

        // TODO: generate proper test data
        // This should involve calling HB2ST or a reference implementation
        // to generate the Householder reflectors stored in V and tau
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dV.transfer_from(hV));
        CHECK_HIP_ERROR(dTau.transfer_from(hTau));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <typename T, typename Td, typename Th>
void ormtr_unmtr_hb2st_getError(const rocblas_handle handle,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int kd,
                                Td& dV,
                                const rocblas_int ldv,
                                Td& dTau,
                                Td& dC,
                                const rocblas_int ldc,
                                Th& hV,
                                Th& hTau,
                                Th& hC,
                                Th& hCr,
                                double* max_err)
{
    size_t size_W = (side == rocblas_side_left ? m : n) * 32;
    std::vector<T> hW(size_W);

    // initialize data
    ormtr_unmtr_hb2st_initData<true, true, T>(
        handle, side, trans, m, n, kd,
        dV, ldv, dTau, dC, ldc, hV, hTau, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd,
            dV.data(), ldv, dTau.data(), dC.data(), ldc));
    CHECK_HIP_ERROR(hCr.transfer_from(dC));

    // CPU lapack
    // TODO: implement CPU reference for ormtr_unmtr_hb2st
    // cpu_ormtr_unmtr_hb2st( side, m, n, kd, hV[0], ldv, hTau[0], hC[0], ldc, hW.data(), size_W);

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCr[0]);
}

template <typename T, typename Td, typename Th>
void ormtr_unmtr_hb2st_getPerfData(const rocblas_handle handle,
                                   const rocblas_side side,
                                   const rocblas_operation trans,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int kd,
                                   Td& dV,
                                   const rocblas_int ldv,
                                   Td& dTau,
                                   Td& dC,
                                   const rocblas_int ldc,
                                   Th& hV,
                                   Th& hTau,
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

    // todo: No CPU implementation available.
    // unmtr_hb2st is in PLASMA but not LAPACK.

    // Initialize CPU data.
    ormtr_unmtr_hb2st_initData<true, false, T>(
        handle, side, trans, m, n, kd,
        dV, ldv, dTau, dC, ldc, hV, hTau, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormtr_unmtr_hb2st_initData<false, true, T>(
            handle, side, trans, m, n, kd,
            dV, ldv, dTau, dC, ldc, hV, hTau, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc));
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
        ormtr_unmtr_hb2st_initData<false, true, T>(
            handle, side, trans, m, n, kd,
            dV, ldv, dTau, dC, ldc, hV, hTau, hC, hW, size_W);

        timer.start(stream);
        rocsolver_ormtr_unmtr_hb2st(
            handle, side, trans, m, n, kd,
            dV.data(), ldv, dTau.data(), dC.data(), ldc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_ormtr_unmtr_hb2st(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
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
    rocblas_int kd = argus.get<rocblas_int>("kd", 1);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", m);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", m);

    rocblas_side side = char2rocblas_side(sideC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value = (side == rocblas_side_both
                          || (COMPLEX && trans == rocblas_operation_transpose));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc),
            rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_V = size_t(ldv) * m;
    size_t size_tau = size_t(kd);
    size_t size_C = size_t(ldc) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || kd < 0 || ldc < m || ldv < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc),
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
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                (T*)nullptr, ldv, (T*)nullptr, (T*)nullptr, ldc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCr(size_Cr, 1, size_Cr, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);
    host_strided_batch_vector<T> hV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());

    // check quick return
    if(m == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_ormtr_unmtr_hb2st(
                handle, side, trans, m, n, kd,
                dV.data(), ldv, dTau.data(), dC.data(), ldc),
            rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        ormtr_unmtr_hb2st_getError<T>(
            handle, side, trans, m, n, kd,
            dV, ldv, dTau, dC, ldc, hV, hTau, hC, hCr, &max_error);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        ormtr_unmtr_hb2st_getPerfData<T>(
            handle, side, trans, m, n, kd, dV, ldv, dTau, dC, ldc,
            hV, hTau, hC, &gpu_time_used, &cpu_time_used, hot_calls,
            argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    rocblas_int s = (side == rocblas_side_left) ? m : n;
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, s);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("side", "trans", "m", "n", "kd", "ldv", "ldc");
            rocsolver_bench_output(sideC, transC, m, n, kd, ldv, ldc);

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

#define EXTERN_TESTING_ORMTR_UNMTR_HB2ST(...) \
    extern template void testing_ormtr_unmtr_hb2st<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_ORMTR_UNMTR_HB2ST, FOREACH_SCALAR_TYPE, APPLY_STAMP)
