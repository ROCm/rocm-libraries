/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSyrkBatchedModel = ArgumentModel<e_a_type,
                                              e_uplo,
                                              e_transA,
                                              e_N,
                                              e_K,
                                              e_alpha,
                                              e_lda,
                                              e_beta,
                                              e_ldc,
                                              e_batch_count>;

inline void testname_syrk_batched(const Arguments& arg, std::string& name)
{
    hipblasSyrkBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_syrk_batched_bad_arg(const Arguments& arg)
{
    using Ts = hipblas_internal_type<T>;
    auto hipblasSyrkBatchedFn
        = arg.api == FORTRAN ? hipblasSyrkBatched<T, true> : hipblasSyrkBatched<T, false>;
    auto hipblasSyrkBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasSyrkBatched_64<T, true> : hipblasSyrkBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N           = 101;
    int64_t            K           = 100;
    int64_t            lda         = 102;
    int64_t            ldc         = 104;
    int64_t            batch_count = 2;
    hipblasOperation_t transA      = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;

    size_t rows = (transA != HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);
    size_t cols = (transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);

    // Allocate device memory
    device_batch_matrix<T> dA(rows, cols, lda, batch_count);
    device_batch_matrix<T> dC(N, N, ldc, batch_count);

    device_vector<T> d_alpha(1), d_zero(1), d_beta(1), d_one(1);
    Ts               h_alpha{1}, h_zero{0}, h_beta{2}, h_one{1};
    if constexpr(is_complex<T>)
        h_one = {1, 0};

    const Ts* alpha = &h_alpha;
    const Ts* beta  = &h_beta;
    const Ts* one   = &h_one;
    const Ts* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_one, one, sizeof(*one), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            one   = d_one;
            zero  = d_zero;
        }

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasSyrkBatchedFn,
                    (nullptr,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     beta,
                     dC.ptr_on_device(),
                     ldc,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyrkBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     transA,
                     N,
                     K,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     beta,
                     dC.ptr_on_device(),
                     ldc,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyrkBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     transA,
                     N,
                     K,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     beta,
                     dC.ptr_on_device(),
                     ldc,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSyrkBatchedFn,
                    (handle,
                     uplo,
                     (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                     N,
                     K,
                     alpha,
                     dA.ptr_on_device(),
                     lda,
                     beta,
                     dC.ptr_on_device(),
                     ldc,
                     batch_count));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkBatchedFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         nullptr,
                         dA.ptr_on_device(),
                         lda,
                         beta,
                         dC.ptr_on_device(),
                         ldc,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyrkBatchedFn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         dA.ptr_on_device(),
                         lda,
                         nullptr,
                         dC.ptr_on_device(),
                         ldc,
                         batch_count));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyrkBatchedFn,
                            (handle,
                             uplo,
                             transA,
                             N,
                             K,
                             alpha,
                             nullptr,
                             lda,
                             beta,
                             dC.ptr_on_device(),
                             ldc,
                             batch_count));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyrkBatchedFn,
                            (handle,
                             uplo,
                             transA,
                             N,
                             K,
                             alpha,
                             dA.ptr_on_device(),
                             lda,
                             beta,
                             nullptr,
                             ldc,
                             batch_count));

                // 64-bit interface test
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                 : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyrkBatchedFn,
                            (handle,
                             uplo,
                             transA,
                             c_i32_overflow,
                             c_i32_overflow,
                             zero,
                             nullptr,
                             c_i32_overflow,
                             one,
                             nullptr,
                             c_i32_overflow,
                             c_i32_overflow));
            }

            // If k == 0 && beta == 1, A, C may be nullptr
            DAPI_CHECK(
                hipblasSyrkBatchedFn,
                (handle, uplo, transA, N, 0, alpha, nullptr, lda, one, nullptr, ldc, batch_count));

            // If alpha == 0 && beta == 1, A, C may be nullptr
            DAPI_CHECK(
                hipblasSyrkBatchedFn,
                (handle, uplo, transA, N, K, zero, nullptr, lda, one, nullptr, ldc, batch_count));
        }

        // If N == 0 batch_count == 0, can have nullptrs
        DAPI_CHECK(hipblasSyrkBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    nullptr,
                    ldc,
                    batch_count));
        DAPI_CHECK(hipblasSyrkBatchedFn,
                   (handle, uplo, transA, N, K, nullptr, nullptr, lda, nullptr, nullptr, ldc, 0));
    }
}

template <typename T>
void testing_syrk_batched(const Arguments& arg)
{
    using Ts = hipblas_internal_type<T>;
    auto hipblasSyrkBatchedFn
        = arg.api == FORTRAN ? hipblasSyrkBatched<T, true> : hipblasSyrkBatched<T, false>;
    auto hipblasSyrkBatchedFn_64
        = arg.api == FORTRAN_64 ? hipblasSyrkBatched_64<T, true> : hipblasSyrkBatched_64<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    int64_t            N           = arg.N;
    int64_t            K           = arg.K;
    int64_t            lda         = arg.lda;
    int64_t            ldc         = arg.ldc;
    int64_t            batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
                        || (transA != HIPBLAS_OP_N && lda < K) || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSyrkBatchedFn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     nullptr,
                     ldc,
                     batch_count));
        return;
    }

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    size_t rows = (transA != HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);
    size_t cols = (transA == HIPBLAS_OP_N ? std::max(K, int64_t(1)) : N);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_batch_matrix<T> hA(rows, cols, lda, batch_count);
    host_batch_matrix<T> hC_host(N, N, ldc, batch_count);
    host_batch_matrix<T> hC_device(N, N, ldc, batch_count);
    host_batch_matrix<T> hC_cpu(N, N, ldc, batch_count);

    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hC_host.memcheck());
    CHECK_HIP_ERROR(hC_device.memcheck());
    CHECK_HIP_ERROR(hC_cpu.memcheck());

    // Allocate device memory
    device_batch_matrix<T> dA(rows, cols, lda, batch_count);
    device_batch_matrix<T> dC(N, N, ldc, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(
        hC_host, arg, hipblas_client_beta_sets_nan, hipblas_symmetric_matrix, false);

    hC_device.copy_from(hC_host);
    hC_cpu.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasSyrkBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    reinterpret_cast<Ts*>(&h_alpha),
                    dA.ptr_on_device(),
                    lda,
                    reinterpret_cast<Ts*>(&h_beta),
                    dC.ptr_on_device(),
                    ldc,
                    batch_count));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasSyrkBatchedFn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    d_alpha,
                    dA.ptr_on_device(),
                    lda,
                    d_beta,
                    dC.ptr_on_device(),
                    ldc,
                    batch_count));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_syrk<T>(uplo, transA, N, K, h_alpha, hA[b], lda, h_beta, hC_cpu[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, hC_cpu, hC_host);
            unit_check_general<T>(N, N, batch_count, ldc, hC_cpu, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, ldc, hC_cpu, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', N, N, ldc, hC_cpu, hC_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasSyrkBatchedFn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           d_alpha,
                           dA.ptr_on_device(),
                           lda,
                           d_beta,
                           dC.ptr_on_device(),
                           ldc,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasSyrkBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              syrk_gflop_count<T>(N, K),
                                              syrk_gbyte_count<T>(N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}
