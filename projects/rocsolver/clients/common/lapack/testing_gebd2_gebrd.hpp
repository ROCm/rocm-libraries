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

template <bool STRIDED, bool GEBRD, typename T, typename S, typename U>
void gebd2_gebrd_checkBadArgs(const rocblas_handle handle,
                              const rocblas_int m,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              S dD,
                              const rocblas_stride stD,
                              S dE,
                              const rocblas_stride stE,
                              U dTauq,
                              const rocblas_stride stQ,
                              U dTaup,
                              const rocblas_stride stP,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, nullptr, m, n, dA, lda, stA, dD,
                                                stD, dE, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD,
                                                    stD, dE, stE, dTauq, stQ, dTaup, stP, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, (T) nullptr, lda, stA,
                                                dD, stD, dE, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA,
                                                (S) nullptr, stD, dE, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD, stD,
                                                (S) nullptr, stE, dTauq, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD, stD,
                                                dE, stE, (U) nullptr, stQ, dTaup, stP, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD, stD,
                                                dE, stE, dTauq, stQ, (U) nullptr, stP, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, 0, n, (T) nullptr, lda, stA,
                                                (S) nullptr, stD, (S) nullptr, stE, (U) nullptr,
                                                stQ, (U) nullptr, stP, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, 0, (T) nullptr, lda, stA,
                                                (S) nullptr, stD, (S) nullptr, stE, (U) nullptr,
                                                stQ, (U) nullptr, stP, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA, lda, stA, dD,
                                                    stD, dE, stE, dTauq, stQ, dTaup, stP, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GEBRD, typename T>
void testing_gebd2_gebrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 2;
    rocblas_int n = 2;
    rocblas_int lda = 2;
    rocblas_stride stA = 1;
    rocblas_stride stD = 1;
    rocblas_stride stE = 1;
    rocblas_stride stQ = 1;
    rocblas_stride stP = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<T> dTauq(1, 1, 1, 1);
        device_strided_batch_vector<T> dTaup(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTauq.memcheck());
        CHECK_HIP_ERROR(dTaup.memcheck());

        // check bad arguments
        gebd2_gebrd_checkBadArgs<STRIDED, GEBRD>(handle, m, n, dA.data(), lda, stA, dD.data(), stD,
                                                 dE.data(), stE, dTauq.data(), stQ, dTaup.data(),
                                                 stP, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<T> dTauq(1, 1, 1, 1);
        device_strided_batch_vector<T> dTaup(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTauq.memcheck());
        CHECK_HIP_ERROR(dTaup.memcheck());

        // check bad arguments
        gebd2_gebrd_checkBadArgs<STRIDED, GEBRD>(handle, m, n, dA.data(), lda, stA, dD.data(), stD,
                                                 dE.data(), stE, dTauq.data(), stQ, dTaup.data(),
                                                 stP, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void gebd2_gebrd_initData(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Sd& dD,
                          const rocblas_stride stD,
                          Sd& dE,
                          const rocblas_stride stE,
                          Ud& dTauq,
                          const rocblas_stride stQ,
                          Ud& dTaup,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Sh& hD,
                          Sh& hE,
                          Uh& hTauq,
                          Uh& hTaup)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j || (m >= n && j == i + 1) || (m < n && i == j + 1))
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GEBRD, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void gebd2_gebrd_getError(const rocblas_handle handle,
                          const rocblas_int m,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Sd& dD,
                          const rocblas_stride stD,
                          Sd& dE,
                          const rocblas_stride stE,
                          Ud& dTauq,
                          const rocblas_stride stQ,
                          Ud& dTaup,
                          const rocblas_stride stP,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hARes,
                          Sh& hD,
                          Sh& hE,
                          Uh& hTauq,
                          Uh& hTaup,
                          double* max_err)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    constexpr bool VERIFY_IMPLICIT_TEST = false;

    std::vector<T> hW(std::max(m, n));

    // input data initialization
    gebd2_gebrd_initData<true, true, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ,
                                        dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

    // execute computations
    // use verify_implicit_test to check correctness of the implicit test using
    // CPU lapack
    if(!VERIFY_IMPLICIT_TEST)
    {
        // GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(), lda, stA,
                                                  dD.data(), stD, dE.data(), stE, dTauq.data(), stQ,
                                                  dTaup.data(), stP, bc));
        CHECK_HIP_ERROR(hARes.transfer_from(dA));
        CHECK_HIP_ERROR(hTauq.transfer_from(dTauq));
        CHECK_HIP_ERROR(hTaup.transfer_from(dTaup));
    }
    else
    {
        // CPU lapack
        for(rocblas_int b = 0; b < bc; ++b)
        {
            memcpy(hARes[b], hA[b], lda * n * sizeof(T));
            GEBRD
            ? cpu_gebrd(m, n, hARes[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data(),
                        std::max(m, n))
            : cpu_gebd2(m, n, hARes[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data());
        }
    }

    // reconstruct A from the factorization for implicit testing
    std::vector<T> vec(std::max(m, n));
    vec[0] = 1;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        T* a = hARes[b];
        T* tauq = hTauq[b];
        T* taup = hTaup[b];

        if(m >= n)
        {
            for(int j = n - 1; j >= 0; j--)
            {
                if(j < n - 1)
                {
                    if(COMPLEX)
                    {
                        cpu_lacgv(1, taup + j, 1);
                        cpu_lacgv(n - j - 1, a + j + (j + 1) * lda, lda);
                    }
                    for(int i = 1; i < n - j - 1; i++)
                    {
                        vec[i] = a[j + (j + i + 1) * lda];
                        a[j + (j + i + 1) * lda] = 0;
                    }
                    cpu_larf(rocblas_side_right, m - j, n - j - 1, vec.data(), 1, taup + j,
                             a + j + (j + 1) * lda, lda, hW.data());
                    if(COMPLEX)
                        cpu_lacgv(1, taup + j, 1);
                }

                for(int i = 1; i < m - j; i++)
                {
                    vec[i] = a[(j + i) + j * lda];
                    a[(j + i) + j * lda] = 0;
                }
                cpu_larf(rocblas_side_left, m - j, n - j, vec.data(), 1, tauq + j, a + j + j * lda,
                         lda, hW.data());
            }
        }
        else
        {
            for(int j = m - 1; j >= 0; j--)
            {
                if(j < m - 1)
                {
                    for(int i = 1; i < m - j - 1; i++)
                    {
                        vec[i] = a[(j + i + 1) + j * lda];
                        a[(j + i + 1) + j * lda] = 0;
                    }
                    cpu_larf(rocblas_side_left, m - j - 1, n - j, vec.data(), 1, tauq + j,
                             a + (j + 1) + j * lda, lda, hW.data());
                }

                if(COMPLEX)
                {
                    cpu_lacgv(1, taup + j, 1);
                    cpu_lacgv(n - j, a + j + j * lda, lda);
                }
                for(int i = 1; i < n - j; i++)
                {
                    vec[i] = a[j + (j + i) * lda];
                    a[j + (j + i) * lda] = 0;
                }
                cpu_larf(rocblas_side_right, m - j, n - j, vec.data(), 1, taup + j, a + j + j * lda,
                         lda, hW.data());
                if(COMPLEX)
                    cpu_lacgv(1, taup + j, 1);
            }
        }
    }

    // error is ||hA - hARes|| / ||hA||
    // using frobenius norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, bool GEBRD, typename T, typename Sd, typename Td, typename Ud, typename Sh, typename Th, typename Uh>
void gebd2_gebrd_getPerfData(const rocblas_handle handle,
                             const rocblas_int m,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Sd& dD,
                             const rocblas_stride stD,
                             Sd& dE,
                             const rocblas_stride stE,
                             Ud& dTauq,
                             const rocblas_stride stQ,
                             Ud& dTaup,
                             const rocblas_stride stP,
                             const rocblas_int bc,
                             Th& hA,
                             Sh& hD,
                             Sh& hE,
                             Uh& hTauq,
                             Uh& hTaup,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    std::vector<T> hW(std::max(m, n));

    if(!perf)
    {
        gebd2_gebrd_initData<true, false, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                             stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            GEBRD
            ? cpu_gebrd(m, n, hA[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data(), std::max(m, n))
            : cpu_gebd2(m, n, hA[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data());
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gebd2_gebrd_initData<true, false, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ,
                                         dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gebd2_gebrd_initData<false, true, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                             stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

        CHECK_ROCBLAS_ERROR(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(), lda, stA,
                                                  dD.data(), stD, dE.data(), stE, dTauq.data(), stQ,
                                                  dTaup.data(), stP, bc));
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
        gebd2_gebrd_initData<false, true, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq,
                                             stQ, dTaup, stP, bc, hA, hD, hE, hTauq, hTaup);

        start = get_time_us_sync(stream);
        rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(), lda, stA, dD.data(), stD,
                              dE.data(), stE, dTauq.data(), stQ, dTaup.data(), stP, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool GEBRD, typename T>
void testing_gebd2_gebrd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stD = argus.get<rocblas_stride>("strideD", std::min(m, n));
    rocblas_stride stE = argus.get<rocblas_stride>("strideE", std::min(m, n) - 1);
    rocblas_stride stQ = argus.get<rocblas_stride>("strideQ", std::min(m, n));
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", std::min(m, n));

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_D = std::min(m, n);
    size_t size_E = std::min(m, n) - 1;
    size_t size_Q = std::min(m, n);
    size_t size_P = std::min(m, n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n,
                                                        (T* const*)nullptr, lda, stA, (S*)nullptr,
                                                        stD, (S*)nullptr, stE, (T*)nullptr, stQ,
                                                        (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, (T*)nullptr,
                                                        lda, stA, (S*)nullptr, stD, (S*)nullptr,
                                                        stE, (T*)nullptr, stQ, (T*)nullptr, stP, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, (T* const*)nullptr,
                                                    lda, stA, (S*)nullptr, stD, (S*)nullptr, stE,
                                                    (T*)nullptr, stQ, (T*)nullptr, stP, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, (T*)nullptr, lda,
                                                    stA, (S*)nullptr, stD, (S*)nullptr, stE,
                                                    (T*)nullptr, stQ, (T*)nullptr, stP, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S> hE(size_E, 1, stE, bc);
        host_strided_batch_vector<T> hTaup(size_P, 1, stP, bc);
        host_strided_batch_vector<T> hTauq(size_Q, 1, stQ, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
        device_strided_batch_vector<T> dTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T> dTaup(size_P, 1, stP, bc);
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

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dTauq.data(), stQ, dTaup.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebd2_gebrd_getError<STRIDED, GEBRD, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE,
                                                    dTauq, stQ, dTaup, stP, bc, hA, hARes, hD, hE,
                                                    hTauq, hTaup, &max_error);

        // collect performance data
        if(argus.timing)
            gebd2_gebrd_getPerfData<STRIDED, GEBRD, T>(
                handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ, dTaup, stP, bc, hA, hD,
                hE, hTauq, hTaup, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S> hE(size_E, 1, stE, bc);
        host_strided_batch_vector<T> hTaup(size_P, 1, stP, bc);
        host_strided_batch_vector<T> hTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
        device_strided_batch_vector<T> dTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T> dTaup(size_P, 1, stP, bc);
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

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebd2_gebrd(STRIDED, GEBRD, handle, m, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dTauq.data(), stQ, dTaup.data(), stP, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebd2_gebrd_getError<STRIDED, GEBRD, T>(handle, m, n, dA, lda, stA, dD, stD, dE, stE,
                                                    dTauq, stQ, dTaup, stP, bc, hA, hARes, hD, hE,
                                                    hTauq, hTaup, &max_error);

        // collect performance data
        if(argus.timing)
            gebd2_gebrd_getPerfData<STRIDED, GEBRD, T>(
                handle, m, n, dA, lda, stA, dD, stD, dE, stE, dTauq, stQ, dTaup, stP, bc, hA, hD,
                hE, hTauq, hTaup, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using m*n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, m * n);

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

#define EXTERN_TESTING_GEBD2_GEBRD(...) \
    extern template void testing_gebd2_gebrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBD2_GEBRD,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
