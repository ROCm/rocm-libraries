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

template <bool STRIDED, bool GEQLF, typename T, typename U>
void geql2_geqlf_checkBadArgs(const rocblas_handle handle,
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
        rocsolver_geql2_geqlf(STRIDED, GEQLF, nullptr, m, n, dA, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA, lda, stA, dIpiv, stP, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, (T) nullptr, lda, stA, dIpiv, stP, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA, lda, stA, (U) nullptr, stP, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, 0, n, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, 0, (T) nullptr, lda, stA,
                                                (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA, lda, stA, dIpiv, stP, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GEQLF, typename T>
void testing_geql2_geqlf_bad_arg()
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
        geql2_geqlf_checkBadArgs<STRIDED, GEQLF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
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
        geql2_geqlf_checkBadArgs<STRIDED, GEQLF>(handle, m, n, dA.data(), lda, stA, dIpiv.data(),
                                                 stP, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void geql2_geqlf_initData(const rocblas_handle handle,
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

template <bool STRIDED, bool GEQLF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void geql2_geqlf_getError(const rocblas_handle handle,
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

    // Work arrays for cpu_lange, cpu_geqlf, etc.
    // todo: query for optimal size; currently estimate nb=64.
    rocblas_int nb = 64;
    std::vector<T> hW(std::max(m, n) * nb);
    std::vector<S> hrwork(std::max(m, n));

    // input data initialization
    geql2_geqlf_initData<true, true, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                        hIpiv);

    // GPU scalar for lange output, shared by all checks below.
    device_strided_batch_vector<S> dnorm(1, 1, 1, 1);
    CHECK_HIP_ERROR(dnorm.memcheck());
    host_strided_batch_vector<S> hnorm(1, 1, 1, 1);

    // Compute norm( A ) on GPU before geqlf overwrites dA.
    std::vector<S> A_norms(bc);
    for(rocblas_int b = 0; b < bc; ++b)
    {
        CHECK_ROCBLAS_ERROR(
            rocsolver_lange(handle, rocsolver_norm_type_one, m, n, dA[b], lda, dnorm[0]));
        CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
        A_norms[b] = hnorm[0][0];
    }

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA.data(), lda, stA,
                                              dIpiv.data(), stP, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpiv.transfer_from(dIpiv));

    //--------------------
    // Check 0: Backward error: norm( Q*L - A ) / (m * norm( A )), using 1-norm.
    // Done before cpu_geqlf so hA still holds the original A.
    // Q*L is computed via unmql applied to L extracted from the GPU factored output.
    //
    // For QL factorization: L is min_mn x n lower trapezoid, stored in the
    // last min_mn rows of A (rows m-min_mn .. m-1). The Householder reflectors
    // are stored in the last min_mn columns of A.
    //
    // For each batch element b:
    //   1. Extract L (lower trapezoid) from hARes[b] into hC, zeroing above diagonal n-m.
    //   2. Upload hC to GPU (dC).
    //   3. Apply unmql: dC = Q * dC.
    //   4. Transfer result back to host.
    //   5. Compute norm( Q*L - A ).
    //
    // If m > n (example with m = 5, n = 3):
    //      L = [       ]               V = [ v v v ]
    //          [       ]                   [ v v v ]
    //          [ l     ]  } last           [ 1 v v ]
    //          [ l l   ]  } min_mn         [   1 v ]
    //          [ l l l ]  } rows           [     1 ]
    //
    // If m <= n (typical use case; example with m = 3, n = 5):
    //      L = [ l l l     ]           V = [     1 v v ]
    //          [ l l l l   ]               [       1 v ]
    //          [ l l l l l ]               [         1 ]
    //                                           {     }
    //                                      last min_mn cols

    // Allocate GPU work buffer for one matrix (m x n), reused per batch element.
    size_t size_C = size_t(lda) * n;
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    CHECK_HIP_ERROR(dC.memcheck());

    max_errors[0] = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // Copy L: lower trapezoid in last min_mn rows,
        // on & below diagonal (n-m), zero elsewhere.
        std::vector<T> hC(lda * n);
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = std::max(0, j + m - n); i < m; ++i)
            {
                hC[i + j * lda] = hARes[b][i + j * lda];
            }
        }

        // Upload L to GPU.
        CHECK_HIP_ERROR(hipMemcpy(dC[0], hC.data(), sizeof(T) * lda * n, hipMemcpyHostToDevice));

        // Compute Q*L on GPU using unmql (side=left, trans=none).
        // unmql expects A to point to the last k=min_mn columns of the GEQLF output.
        CHECK_ROCBLAS_ERROR(
            rocsolver_ormxl_unmxl(GEQLF, handle, rocblas_side_left, rocblas_operation_none, m, n,
                                  min_mn, dA[b] + lda * (n - min_mn), lda, dIpiv[b], dC[0], lda));

        // Transfer Q*L back to host.
        CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC[0], sizeof(T) * lda * n, hipMemcpyDeviceToHost));

        // Compute norm( Q*L - A ); hA[b] is still the original A.
        for(rocblas_int j = 0; j < n; ++j)
            for(rocblas_int i = 0; i < m; ++i)
                hC[i + j * lda] -= hA[b][i + j * lda];

        double err = cpu_lange('1', m, n, hC.data(), lda, hrwork.data());
        err /= m;
        if(A_norms[b] != 0)
            err /= A_norms[b];
        max_errors[0] = rocblas_max_nan(err, max_errors[0]);
    }

    //--------------------
    // Check 1: Orthogonality: norm( I - Q^H * Q ) / m, using 1-norm.
    // Q is generated explicitly via ungql from the GPU factored output.
    // ungql expects A to point to the last k=min_mn columns of the GEQLF output.
    // Q is m x min_mn; Q^H * Q is min_mn x min_mn.

    // GPU work buffer for Q (m x min_mn, leading dim lda).
    size_t size_Q = size_t(lda) * min_mn;
    device_strided_batch_vector<T> dQ(size_Q, 1, size_Q, 1);
    CHECK_HIP_ERROR(dQ.memcheck());

    // GPU work buffer for I - Q^H * Q (min_mn x min_mn, leading dim min_mn).
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
        // Copy last min_mn columns of dA[b] (QL reflectors) to dQ.
        // When m >= n: n-min_mn = 0, copies all n = min_mn columns from the start.
        // When m <  n: copies columns n-min_mn .. n-1.
        CHECK_HIP_ERROR(hipMemcpy(dQ[0], dA[b] + lda * (n - min_mn), sizeof(T) * lda * min_mn,
                                  hipMemcpyDeviceToDevice));

        // Generate explicit Q (m x min_mn) via ungql.
        CHECK_ROCBLAS_ERROR(
            rocsolver_orgxl_ungxl(GEQLF, handle, m, min_mn, min_mn, dQ[0], lda, dIpiv[b]));

        // Set dR = I (min_mn x min_mn).
        CHECK_HIP_ERROR(
            hipMemcpy(dR[0], hR_id.data(), sizeof(T) * min_mn * min_mn, hipMemcpyHostToDevice));

        // Compute dR = I - Q^H * Q.
        CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle, rocblas_operation_conjugate_transpose,
                                           rocblas_operation_none, min_mn, min_mn, m, // opts
                                           &negone, dQ[0], lda, 0, // Q^H
                                           dQ[0], lda, 0, // Q
                                           &one, dR[0], min_mn, 0, // R
                                           1));

        // Compute norm( I - Q^H * Q ).
        CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, rocsolver_norm_type_one, min_mn, min_mn, dR[0],
                                            min_mn, dnorm[0]));
        CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));

        double err = hnorm[0][0] / m;
        max_errors[1] = rocblas_max_nan(err, max_errors[1]);
    }

    //--------------------
    // Check 2: Comparison with CPU LAPACK.
    // Runs last so hA holds the original A for all checks above.
    for(rocblas_int b = 0; b < bc; ++b)
    {
        GEQLF ? cpu_geqlf(m, n, hA[b], lda, hIpiv[b], hW.data(), hW.size())
              : cpu_geql2(m, n, hA[b], lda, hIpiv[b], hW.data());
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

template <bool STRIDED, bool GEQLF, typename T, typename Td, typename Ud, typename Th, typename Uh>
void geql2_geqlf_getPerfData(const rocblas_handle handle,
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
    std::vector<T> hW(n);

    if(!perf)
    {
        geql2_geqlf_initData<true, false, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GEQLF ? cpu_geqlf(m, n, hA[b], lda, hIpiv[b], hW.data(), n)
                  : cpu_geql2(m, n, hA[b], lda, hIpiv[b], hW.data());
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    geql2_geqlf_initData<true, false, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                         hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geql2_geqlf_initData<false, true, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        CHECK_ROCBLAS_ERROR(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA.data(), lda, stA,
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
        geql2_geqlf_initData<false, true, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA,
                                             hIpiv);

        timer.start(stream);
        rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA.data(), lda, stA, dIpiv.data(), stP,
                              bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, bool GEQLF, typename T>
void testing_geql2_geqlf(Arguments& argus)
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
            EXPECT_ROCBLAS_STATUS(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n,
                                                        (T* const*)nullptr, lda, stA, (T*)nullptr,
                                                        stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, (T*)nullptr,
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
            CHECK_ALLOC_QUERY(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, (T* const*)nullptr,
                                                    lda, stA, (T*)nullptr, stP, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, (T*)nullptr, lda,
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
            EXPECT_ROCBLAS_STATUS(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            geql2_geqlf_getError<STRIDED, GEQLF, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP,
                                                    bc, hA, hARes, hIpiv, max_errors);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            geql2_geqlf_getPerfData<STRIDED, GEQLF, T>(
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
            EXPECT_ROCBLAS_STATUS(rocsolver_geql2_geqlf(STRIDED, GEQLF, handle, m, n, dA.data(),
                                                        lda, stA, dIpiv.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            geql2_geqlf_getError<STRIDED, GEQLF, T>(handle, matrix, m, n, dA, lda, stA, dIpiv, stP,
                                                    bc, hA, hARes, hIpiv, max_errors);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            geql2_geqlf_getPerfData<STRIDED, GEQLF, T>(
                handle, matrix, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 15*machine_precision as tolerance (LAPACK uses 30 ulp/2).
    // max_errors is already normalized, e.g., by m.
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

#define EXTERN_TESTING_GEQL2_GEQLF(...) \
    extern template void testing_geql2_geqlf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEQL2_GEQLF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
