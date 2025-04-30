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

#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool BATCHED, bool STRIDED, typename U>
void gels_outofplace_checkBadArgs(const rocblas_handle handle,
                                  const rocblas_operation trans,
                                  const rocblas_int m,
                                  const rocblas_int n,
                                  const rocblas_int nrhs,
                                  U dA,
                                  const rocblas_int lda,
                                  const rocblas_stride stA,
                                  U dB,
                                  const rocblas_int ldb,
                                  const rocblas_stride stB,
                                  U dX,
                                  const rocblas_int ldx,
                                  const rocblas_stride stX,
                                  rocblas_int* info,
                                  const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, nullptr, trans, m, n, nrhs, dA, lda,
                                                    stA, dB, ldb, stB, dX, ldx, stX, info, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, rocblas_operation(0), m, n,
                                                    nrhs, dA, lda, stA, dB, ldb, stB, dX, ldx, stX,
                                                    info, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA, lda,
                                                        stA, dB, ldb, stB, dX, ldx, stX, info, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, (U) nullptr,
                                                    lda, stA, dB, ldb, stB, dX, ldx, stX, info, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA,
                                                    (U) nullptr, ldb, stB, dX, ldx, stX, info, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA, lda, stA,
                                                    dB, ldb, stB, (U) nullptr, ldx, stX, info, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA, lda,
                                                    stA, dB, ldb, stB, dX, ldx, stX, nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, rocblas_operation_none, 0, n,
                                                    nrhs, (U) nullptr, lda, stA, (U) nullptr, ldb,
                                                    stB, dX, ldx, stX, info, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, 0, n, nrhs, (U) nullptr,
                                                    lda, stA, dB, ldb, stB, (U) nullptr, ldx, stX,
                                                    info, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, rocblas_operation_none, m, 0,
                                                    nrhs, (U) nullptr, lda, stA, dB, ldb, stB,
                                                    (U) nullptr, ldx, stX, info, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, 0, nrhs, (U) nullptr,
                                                    lda, stA, (U) nullptr, ldb, stB, dX, ldx, stX,
                                                    info, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, 0, dA, lda, stA,
                                                    (U) nullptr, ldb, stB, (U) nullptr, ldx, stX,
                                                    info, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, 0, 0, nrhs, (U) nullptr,
                                                    lda, stA, (U) nullptr, ldb, stB, (U) nullptr,
                                                    ldx, stX, info, bc),
                          rocblas_status_success);
    if(BATCHED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA, lda,
                                                        stA, dB, ldb, stB, dX, ldx, stX, nullptr, 0),
                              rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA, lda,
                                                        stA, dB, ldb, stB, dX, ldx, stX, info, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gels_outofplace_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int n = 1;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldx = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    rocblas_stride stX = 1;
    rocblas_int bc = 1;
    rocblas_operation trans = (!rocblas_is_complex<T> ? rocblas_operation_transpose
                                                      : rocblas_operation_conjugate_transpose);
    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_batch_vector<T> dX(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dX.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        gels_outofplace_checkBadArgs<BATCHED, STRIDED>(handle, trans, m, n, nrhs, dA.data(), lda,
                                                       stA, dB.data(), ldb, stB, dX.data(), ldx,
                                                       stX, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<T> dX(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dX.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        gels_outofplace_checkBadArgs<BATCHED, STRIDED>(handle, trans, m, n, nrhs, dA.data(), lda,
                                                       stA, dB.data(), ldb, stB, dX.data(), ldx,
                                                       stX, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_outofplace_initData(const rocblas_handle handle,
                              const rocblas_operation trans,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              Td& dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              Td& dB,
                              const rocblas_int ldb,
                              const rocblas_stride stB,
                              Ud& dInfo,
                              const rocblas_int bc,
                              Th& hA,
                              Th& hB,
                              Th& hX,
                              Uh& hInfo,
                              const bool singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

        const rocblas_int max_index = std::max(0, std::min(m, n) - 1);
        std::uniform_int_distribution<int> sample_index(0, max_index);
        std::bernoulli_distribution coinflip(0.5);

        const rocblas_int rowsB = (trans == rocblas_operation_none) ? m : n;
        const rocblas_int ldx = std::max(m, n);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(rocblas_int i = 0; i < m; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // populate hX with values from hB
            for(rocblas_int i = 0; i < rowsB; i++)
                for(rocblas_int j = 0; j < nrhs; j++)
                    hX[b][i + j * ldx] = hB[b][i + j * ldb];

            // add some singularities
            // always the same elements for debugging purposes
            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                do
                {
                    if(n <= m)
                    {
                        // zero random col
                        rocblas_int j = sample_index(rocblas_rng);
                        for(rocblas_int i = 0; i < m; i++)
                            hA[b][i + j * lda] = 0;
                    }
                    else
                    {
                        // zero random row
                        rocblas_int i = sample_index(rocblas_rng);
                        for(rocblas_int j = 0; j < n; j++)
                            hA[b][i + j * lda] = 0;
                    }
                } while(coinflip(rocblas_rng));
            }
        }
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_outofplace_getError(const rocblas_handle handle,
                              const rocblas_operation trans,
                              const rocblas_int m,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              Td& dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              Td& dB,
                              const rocblas_int ldb,
                              const rocblas_stride stB,
                              Td& dX,
                              const rocblas_int ldx,
                              const rocblas_stride stX,
                              Ud& dInfo,
                              const rocblas_int bc,
                              Th& hA,
                              Th& hB,
                              Th& hBRes,
                              Th& hX,
                              Th& hXRes,
                              Uh& hInfo,
                              Uh& hInfoRes,
                              double* max_err,
                              const bool singular)
{
    rocblas_int sizeW = std::max(1, std::min(m, n) + std::max(std::min(m, n), nrhs));
    std::vector<T> hW(sizeW);

    // input data initialization
    gels_outofplace_initData<true, true, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb, stB,
                                            dInfo, bc, hA, hB, hX, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA.data(),
                                                  lda, stA, dB.data(), ldb, stB, dX.data(), ldx,
                                                  stX, dInfo.data(), bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cpu_gels(trans, m, n, nrhs, hA[b], lda, hX[b], std::max(m, n), hW.data(), sizeW, hInfo[b]);
    }

    // error is ||hX - hXRes|| / ||hX||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        const rocblas_int rowsB = (trans == rocblas_operation_none) ? m : n;
        err = norm_error('F', rowsB, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;

        if(hInfo[b][0] == 0)
        {
            const rocblas_int rowsX = (trans == rocblas_operation_none) ? n : m;
            err = norm_error('I', rowsX, nrhs, std::max(m, n), hX[b], hXRes[b], ldx);
            *max_err = err > *max_err ? err : *max_err;
        }
    }

    // also check info for singularities
    err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void gels_outofplace_getPerfData(const rocblas_handle handle,
                                 const rocblas_operation trans,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 const rocblas_int nrhs,
                                 Td& dA,
                                 const rocblas_int lda,
                                 const rocblas_stride stA,
                                 Td& dB,
                                 const rocblas_int ldb,
                                 const rocblas_stride stB,
                                 Td& dX,
                                 const rocblas_int ldx,
                                 const rocblas_stride stX,
                                 Ud& dInfo,
                                 const rocblas_int bc,
                                 Th& hA,
                                 Th& hB,
                                 Th& hX,
                                 Uh& hInfo,
                                 double* gpu_time_used,
                                 double* cpu_time_used,
                                 const rocblas_int hot_calls,
                                 const int profile,
                                 const bool profile_kernels,
                                 const bool perf,
                                 const bool singular)
{
    rocblas_int sizeW = std::max(1, std::min(m, n) + std::max(std::min(m, n), nrhs));
    std::vector<T> hW(sizeW);

    if(!perf)
    {
        gels_outofplace_initData<true, false, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb,
                                                 stB, dInfo, bc, hA, hB, hX, hInfo, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cpu_gels(trans, m, n, nrhs, hA[b], lda, hX[b], std::max(m, n), hW.data(), sizeW,
                     hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gels_outofplace_initData<true, false, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb, stB,
                                             dInfo, bc, hA, hB, hX, hInfo, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gels_outofplace_initData<false, true, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb,
                                                 stB, dInfo, bc, hA, hB, hX, hInfo, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA.data(),
                                                      lda, stA, dB.data(), ldb, stB, dX.data(), ldx,
                                                      stX, dInfo.data(), bc));
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
        gels_outofplace_initData<false, true, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb,
                                                 stB, dInfo, bc, hA, hB, hX, hInfo, singular);

        start = get_time_us_sync(stream);
        rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, dA.data(), lda, stA,
                                  dB.data(), ldb, stB, dX.data(), ldx, stX, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T, bool COMPLEX = rocblas_is_complex<T>>
void testing_gels_outofplace(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char transC = argus.get<char>("trans");
    rocblas_int m = argus.get<rocblas_int>("m");
    rocblas_int n = argus.get<rocblas_int>("n", m);
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", m);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", transC == 'N' ? m : n);
    rocblas_int ldx = argus.get<rocblas_int>("ldx", transC == 'N' ? n : m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);
    rocblas_stride stX = argus.get<rocblas_stride>("strideX", ldx * nrhs);

    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;
    rocblas_stride stXRes = (argus.unit_check || argus.norm_check) ? stX : 0;

    // check non-supported values
    bool invalid_value = ((COMPLEX && trans == rocblas_operation_transpose)
                          || (!COMPLEX && trans == rocblas_operation_conjugate_transpose));
    if(invalid_value)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, (T* const*)nullptr,
                                          lda, stA, (T* const*)nullptr, ldb, stB,
                                          (T* const*)nullptr, ldx, stX, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs,
                                                            (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                            stB, (T*)nullptr, ldx, stX,
                                                            (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * nrhs;
    size_t size_X = size_t(ldx) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_XRes = (argus.unit_check || argus.norm_check) ? size_X : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || nrhs < 0 || lda < m
                         || (trans == rocblas_operation_none && (ldb < m || ldx < n))
                         || (trans != rocblas_operation_none && (ldb < n || ldx < m)) || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs, (T* const*)nullptr,
                                          lda, stA, (T* const*)nullptr, ldb, stB,
                                          (T* const*)nullptr, ldx, stX, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs,
                                                            (T*)nullptr, lda, stA, (T*)nullptr, ldb,
                                                            stB, (T*)nullptr, ldx, stX,
                                                            (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_gels_outofplace(
                STRIDED, handle, trans, m, n, nrhs, (T* const*)nullptr, lda, stA, (T* const*)nullptr,
                ldb, stB, (T* const*)nullptr, ldx, stX, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gels_outofplace(
                STRIDED, handle, trans, m, n, nrhs, (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                (T*)nullptr, ldx, stX, (rocblas_int*)nullptr, bc));

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
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_batch_vector<T> hX(std::max(m, n) * nrhs, 1, bc);
        host_batch_vector<T> hXRes(size_XRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dX(size_X, 1, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());
        if(bc)
            CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs,
                                                            dA.data(), lda, stA, dB.data(), ldb, stB,
                                                            dX.data(), ldx, stX, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gels_outofplace_getError<STRIDED, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb,
                                                 stB, dX, ldx, stX, dInfo, bc, hA, hB, hBRes, hX,
                                                 hXRes, hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            gels_outofplace_getPerfData<STRIDED, T>(
                handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dX, ldx, stX, dInfo, bc, hA,
                hB, hX, hInfo, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                argus.profile_kernels, argus.perf, argus.singular);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<T> hX(std::max(m, n) * nrhs, 1, std::max(m, n) * nrhs, bc);
        host_strided_batch_vector<T> hXRes(size_XRes, 1, stXRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dX(size_X, 1, stX, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_X)
            CHECK_HIP_ERROR(dX.memcheck());
        if(bc)
            CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gels_outofplace(STRIDED, handle, trans, m, n, nrhs,
                                                            dA.data(), lda, stA, dB.data(), ldb, stB,
                                                            dX.data(), ldx, stX, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gels_outofplace_getError<STRIDED, T>(handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb,
                                                 stB, dX, ldx, stX, dInfo, bc, hA, hB, hBRes, hX,
                                                 hXRes, hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            gels_outofplace_getPerfData<STRIDED, T>(
                handle, trans, m, n, nrhs, dA, lda, stA, dB, ldb, stB, dX, ldx, stX, dInfo, bc, hA,
                hB, hX, hInfo, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                argus.profile_kernels, argus.perf, argus.singular);
    }
    // validate results for rocsolver-test
    // using max(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, std::max(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("trans", "m", "n", "nrhs", "lda", "ldb", "ldx", "batch_c");
                rocsolver_bench_output(transC, m, n, nrhs, lda, ldb, ldx, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("trans", "m", "n", "nrhs", "lda", "ldb", "ldx", "strideA",
                                       "strideB", "strideX", "batch_c");
                rocsolver_bench_output(transC, m, n, nrhs, lda, ldb, ldx, stA, stB, stX, bc);
            }
            else
            {
                rocsolver_bench_output("trans", "m", "n", "nrhs", "lda", "ldb", "ldx");
                rocsolver_bench_output(transC, m, n, nrhs, lda, ldb, ldx);
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
