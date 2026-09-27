/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <bool STRIDED, bool GELQF, typename T, typename U>
void gelq2_gelqf_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              U dIpiv,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gelq2_gelqf(STRIDED, GELQF, nullptr, m, n, dA, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA, lda, stA, dIpiv, stP, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, (T) nullptr, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA, lda, stA, (U) nullptr, stP, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, 0, n, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, 0, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA, lda, stA, dIpiv, stP, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GELQF, typename T>
void testing_gelq2_gelqf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gelq2_gelqf_checkBadArgs<STRIDED, GELQF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        gelq2_gelqf_checkBadArgs<STRIDED, GELQF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gelq2_gelqf_initData(const rocblas_handle handle,
                          const std::string& matrix,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Uh& hIpiv)
{
    if(CPU)
    {
        if(matrix == "identity")
        {
            for(rocblas_int b = 0; b < bc; ++b)
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < m; i++)
                        hA[b][i + j * lda] = (i == j ? T(1) : T(0));
        }
        else if(matrix == "randint")
        {
            rocblas_init<T>(hA, true);
        }
        else
        {
            throw std::runtime_error("unknown matrix type: " + matrix);
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GELQF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gelq2_gelqf_getError(const rocblas_handle handle,
                          const std::string& matrix,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Ud& dIpiv,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Uh& hIpiv,
                          double max_errors[3])
{
    using S = decltype(std::real(T{}));

    // todo: fix const-correctness in rocsolver_gemm
    T one = T(1);
    T negone = T(-1);

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_int min_mn = std::min(m, n);

    // Work arrays for cpu_lange, cpu_gelqf, etc.
    // todo: query for optimal size; currently estimate nb=64.
    rocblas_int nb = 64;
    std::vector<T> hW(std::max(m, n) * nb);
    std::vector<S> hrwork(std::max(m, n));

    // input data initialization
    gelq2_gelqf_initData<true, true, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                        hIpiv);

    // GPU scalar for lange output, shared by all checks below.
    device_strided_batch_vector<S> dnorm(1, 1, 1, 1);
    CHECK_HIP_ERROR(dnorm.memcheck());
    host_strided_batch_vector<S> hnorm(1, 1, 1, 1);

    // Compute norm( A ) on GPU before gelqf overwrites dA.
    std::vector<S> A_norms(bc);
    for(rocblas_int b = 0; b < bc; ++b)
    {
        CHECK_ROCBLAS_ERROR(
            rocsolver_lange(handle, rocsolver_norm_type_one, m, n, dA[b], lda, dnorm[0]));
        CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
        A_norms[b] = hnorm[0][0];
    }

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpiv.transfer_from(dIpiv));

    //--------------------
    // Check 0: Backward error: norm( L*Q - A ) / (n * norm( A )), using 1-norm.
    // Done before cpu_gelqf so hA still holds the original A.
    // L*Q is computed via unmlq applied to L extracted from the GPU factored output.
    //
    // For LQ factorization: L is m x min_mn lower trapezoid, stored in the
    // first min_mn columns of A. The Householder reflectors are stored in the
    // first min_mn rows of A.
    //
    // For each batch element b:
    //   1. Extract L into hC (m x n): L[i][j] for i >= j and j < min_mn, else 0.
    //   2. Upload hC to GPU (dC).
    //   3. Apply unmlq: dC = dC * Q (side=right, trans=none).
    //   4. Transfer result back, compute norm( L*Q - A ).
    //
    // If m > n (example with m = 5, n = 3):
    //      L = [ l     ]               V = [ 1 v v ]  } first
    //          [ l l   ]                   [   1 v ]  } min_mn
    //          [ l l l ]                   [     1 ]  } rows
    //          [ l l l ]                   [       ]
    //          [ l l l ]                   [       ]
    //
    // If m <= n (typical use case; example with m = 3, n = 5):
    //      L = [ l         ]           V = [ 1 v v v v ]
    //          [ l l       ]               [   1 v v v ]
    //          [ l l l     ]               [     1 v v ]
    //           {     }
    //      first min_mn cols

    // Allocate GPU work buffer for one matrix (m x n), reused per batch element.
    size_t size_C = size_t(lda) * n;
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    CHECK_HIP_ERROR(dC.memcheck());

    max_errors[0] = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // Copy L: lower trapezoid of A, zero elsewhere.
        // todo: laset and lacpy on GPU.
        std::vector<T> hC(lda * n, T(0));
        cpu_lacpy(rocblas_fill_lower, m, min_mn, hARes[b], lda, hC.data(), lda);

        // Upload L to GPU.
        CHECK_HIP_ERROR(hipMemcpy(dC[0], hC.data(), sizeof(T) * lda * n, hipMemcpyHostToDevice));

        // Compute L*Q on GPU using unmlq.
        CHECK_ROCBLAS_ERROR(rocsolver_ormlx_unmlx(GELQF, handle, rocblas_side_right,
                                                  rocblas_operation_none, m, n, min_mn, dA[b], lda,
                                                  dIpiv[b], dC[0], lda));

        // Transfer L*Q back to host.
        // todo: geadd and lange on GPU.
        CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC[0], sizeof(T) * lda * n, hipMemcpyDeviceToHost));

        // Compute norm( L*Q - A ); hA[b] is still the original A.
        for(rocblas_int j = 0; j < n; ++j)
            for(rocblas_int i = 0; i < m; ++i)
                hC[i + j * lda] -= hA[b][i + j * lda];

        double err = cpu_lange('1', m, n, hC.data(), lda, hrwork.data());
        err /= n;
        if(A_norms[b] != 0)
            err /= A_norms[b];
        max_errors[0] = rocblas_max_nan(err, max_errors[0]);
    }

    //--------------------
    // Check 1: Orthogonality: norm( I - Q * Q^H ) / n, using 1-norm.
    // Q is generated explicitly via unglq from the GPU factored output.
    // Q is min_mn x n; Q * Q^H is min_mn x min_mn.

    // GPU work buffer for Q (min_mn x n, leading dim lda).
    size_t size_Q = size_t(lda) * n;
    device_strided_batch_vector<T> dQ(size_Q, 1, size_Q, 1);
    CHECK_HIP_ERROR(dQ.memcheck());

    // GPU work buffer for I - Q * Q^H (min_mn x min_mn, leading dim min_mn).
    size_t size_R = size_t(min_mn) * min_mn;
    device_strided_batch_vector<T> dR(size_R, 1, size_R, 1);
    CHECK_HIP_ERROR(dR.memcheck());

    // Build identity matrix on host for upload into dR before each gemm.
    std::vector<T> hR_id(min_mn * min_mn, T(0));
    for(rocblas_int i = 0; i < min_mn; ++i)
        hR_id[i + i * min_mn] = T(1);

    max_errors[1] = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // Copy dA[b] to dQ (unglq uses first min_mn rows; copying all lda*n is safe).
        CHECK_HIP_ERROR(hipMemcpy(dQ[0], dA[b], sizeof(T) * lda * n, hipMemcpyDeviceToDevice));

        // Generate explicit Q (min_mn x n) in dQ via unglq.
        CHECK_ROCBLAS_ERROR(
            rocsolver_orglx_unglx(GELQF, handle, min_mn, n, min_mn, dQ[0], lda, dIpiv[b]));

        // Set dR = I (min_mn x min_mn).
        // todo: replace with laset
        CHECK_HIP_ERROR(
            hipMemcpy(dR[0], hR_id.data(), sizeof(T) * min_mn * min_mn, hipMemcpyHostToDevice));

        // Compute dR = I - Q * Q^H.
        CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle, rocblas_operation_none,
                                           rocblas_operation_conjugate_transpose, min_mn, min_mn,
                                           n, // opts
                                           &negone, dQ[0], lda, 0, // Q
                                           dQ[0], lda, 0, // Q^H
                                           &one, dR[0], min_mn, 0, // R
                                           1));

        // Compute norm( I - Q * Q^H ).
        CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, rocsolver_norm_type_one, min_mn, min_mn, dR[0],
                                            min_mn, dnorm[0]));
        CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));

        double err = hnorm[0][0] / n;
        max_errors[1] = rocblas_max_nan(err, max_errors[1]);
    }

    //--------------------
    // Check 2: Comparison with CPU LAPACK.
    // Runs last so hA holds the original A for all checks above.
    for(rocblas_int b = 0; b < bc; ++b)
    {
        GELQF ? cpu_gelqf(m, n, hA[b], lda, hIpiv[b], hW.data(), m)
              : cpu_gelq2(m, n, hA[b], lda, hIpiv[b], hW.data());
    }

    // forward comparison: ||hA - hARes|| / ||hA|| (GPU vs CPU factored form)
    // using frobenius norm
    // (This does not account for numerical reproducibility issues.
    // Checks 0 and 1 above are more robust.)
    max_errors[2] = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        double err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        max_errors[2] = rocblas_max_nan(err, max_errors[2]);
    }

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool STRIDED, bool GELQF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gelq2_gelqf_getPerfData(const rocblas_handle handle,
                             const std::string& matrix,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Ud& dIpiv,
                             const rocblas_stride stP,
                             const rocblas_int bc,
                             Th& hA,
                             Uh& hIpiv,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    std::vector<T> hW(m);

    if(!perf)
    {
        gelq2_gelqf_initData<true, false, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GELQF ? cpu_gelqf(m, n, hA[b], lda, hIpiv[b], hW.data(), m)
                  : cpu_gelq2(m, n, hA[b], lda, hIpiv[b], hW.data());
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gelq2_gelqf_initData<true, false, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                         hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gelq2_gelqf_initData<false, true, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        CHECK_ROCBLAS_ERROR(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(), lda, stA,
                                                  dIpiv.data(), stP, bc));
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

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        gelq2_gelqf_initData<false, true, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        timer.start(stream);
        rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP,
                              bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, bool GELQF, typename T>
void testing_gelq2_gelqf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", min(m, n));
    std::string matrix = argus.get<std::string>("matrix", "randint");

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m, n));
    double max_errors[3] = {0, 0, 0}, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n,
                                                        (T* const*)nullptr, lda, stA, (T*)nullptr,
                                                        stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, (T*)nullptr,
                                                        lda, stA, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, (T* const*)nullptr,
                                                    lda, stA, (T*)nullptr, stP, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, (T*)nullptr, lda,
                                                    stA, (T*)nullptr, stP, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gelq2_gelqf_getError<STRIDED, GELQF, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP,
                                                    bc, hA, hARes, hIpiv, max_errors);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gelq2_gelqf_getPerfData<STRIDED, GELQF, T>(
                handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gelq2_gelqf(STRIDED, GELQF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gelq2_gelqf_getError<STRIDED, GELQF, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP,
                                                    bc, hA, hARes, hIpiv, max_errors);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gelq2_gelqf_getPerfData<STRIDED, GELQF, T>(
                handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 15*machine_precision as tolerance (LAPACK uses 30 ulp/2).
    // max_errors is already normalized, e.g., by n.
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_errors[0], 15);
        ROCSOLVER_TEST_CHECK(T, max_errors[1], 15);
        // Do not check forward comparison with LAPACK, since it is unreliable.
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "backward error",
                                       "orthogonality", "forward comparison");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_errors[0], max_errors[1],
                                       max_errors[2]);
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
                rocsolver_bench_output(gpu_time_used, max_errors[0], max_errors[1], max_errors[2]);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_GELQ2_GELQF(...) \
    extern template void testing_gelq2_gelqf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GELQ2_GELQF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
