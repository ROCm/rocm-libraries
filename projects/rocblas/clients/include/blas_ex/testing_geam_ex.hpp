/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "cblas_interface.hpp"
#include "client_utility.hpp"
#include "flops.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"

/* ============================================================================================ */

template <typename T>
void testing_geam_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_geam_ex_fn = arg.api & c_API_FORTRAN ? rocblas_geam_ex : rocblas_geam_ex_fortran;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M = 100;
        const rocblas_int N = 99;
        const rocblas_int K = 98;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;
        const rocblas_int ldd = 100;

        rocblas_datatype a_type       = arg.a_type;
        rocblas_datatype b_type       = arg.b_type;
        rocblas_datatype c_type       = arg.c_type;
        rocblas_datatype d_type       = arg.d_type;
        rocblas_datatype compute_type = arg.compute_type;

        rocblas_geam_ex_operation geam_ex_op = arg.geam_ex_op;

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, one_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        rocblas_int A_row = transA == rocblas_operation_none ? M : K;
        rocblas_int A_col = transA == rocblas_operation_none ? K : M;
        rocblas_int B_row = transB == rocblas_operation_none ? K : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : K;

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<T>, dB, (B_row, B_col, ldb));
        DEVICE_MEMCHECK(device_matrix<T>, dC, (M, N, ldc));
        DEVICE_MEMCHECK(device_matrix<T>, dD, (M, N, ldd));

        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(nullptr,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_handle);

        // invalid values
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 (rocblas_operation)rocblas_fill_full,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 (rocblas_operation)rocblas_fill_full,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_value);

        // invalid sizes not done in yaml test

        // C == D leading dims must match
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dC,
                                                 d_type,
                                                 ldc + 1,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_size);

        // alpha/beta
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 nullptr,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 nullptr,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_pointer);

        // invalid pointers
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 nullptr,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {

            EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     alpha,
                                                     nullptr,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     beta,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     compute_type,
                                                     geam_ex_op),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     alpha,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     nullptr,
                                                     b_type,
                                                     ldb,
                                                     beta,
                                                     dC,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     compute_type,
                                                     geam_ex_op),
                                  rocblas_status_invalid_pointer);

            EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     alpha,
                                                     dA,
                                                     a_type,
                                                     lda,
                                                     dB,
                                                     b_type,
                                                     ldb,
                                                     beta,
                                                     nullptr,
                                                     c_type,
                                                     ldc,
                                                     dD,
                                                     d_type,
                                                     ldd,
                                                     compute_type,
                                                     geam_ex_op),
                                  rocblas_status_invalid_pointer);
        }

        // M==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 0,
                                                 N,
                                                 K,
                                                 alpha,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 nullptr,
                                                 c_type,
                                                 ldc,
                                                 nullptr,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_success);

        // N==0 then all may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 0,
                                                 K,
                                                 alpha,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 nullptr,
                                                 c_type,
                                                 ldc,
                                                 nullptr,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_success);

        // K == 0 then alpha, A, and B may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_success);

        // alpha==0 then A/B may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 zero,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 beta,
                                                 dC,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_success);

        // beta==0 then C may be nullptr
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 dA,
                                                 a_type,
                                                 lda,
                                                 dB,
                                                 b_type,
                                                 ldb,
                                                 zero,
                                                 nullptr,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_success);

        // If alpha==0 && beta==0 then A, B and C can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 zero,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 zero,
                                                 nullptr,
                                                 c_type,
                                                 ldc,
                                                 dD,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_geam_ex(const Arguments& arg)
{
    auto rocblas_geam_ex_fn = arg.api & c_API_FORTRAN ? rocblas_geam_ex : rocblas_geam_ex_fortran;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;
    rocblas_int ldd = arg.ldd;

    rocblas_datatype a_type       = arg.a_type;
    rocblas_datatype b_type       = arg.b_type;
    rocblas_datatype c_type       = arg.c_type;
    rocblas_datatype d_type       = arg.d_type;
    rocblas_datatype compute_type = arg.compute_type;

    rocblas_geam_ex_operation geam_ex_op = arg.geam_ex_op;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    T* dD_in_place;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;

    double rocblas_error_1 = std::numeric_limits<double>::max();
    double rocblas_error_2 = std::numeric_limits<double>::max();
    double rocblas_error   = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    size_t size_D = size_t(ldd) * N;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M;
    if(invalid_size || !M || !N)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_geam_ex_fn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 nullptr,
                                                 nullptr,
                                                 a_type,
                                                 lda,
                                                 nullptr,
                                                 b_type,
                                                 ldb,
                                                 nullptr,
                                                 nullptr,
                                                 c_type,
                                                 ldc,
                                                 nullptr,
                                                 d_type,
                                                 ldd,
                                                 compute_type,
                                                 geam_ex_op),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (A_row, A_col, lda));
    HOST_MEMCHECK(host_matrix<T>, hA_copy, (A_row, A_col, lda));
    HOST_MEMCHECK(host_matrix<T>, hB, (B_row, B_col, ldb));
    HOST_MEMCHECK(host_matrix<T>, hB_copy, (B_row, B_col, ldb));
    HOST_MEMCHECK(host_matrix<T>, hC, (M, N, ldc));
    HOST_MEMCHECK(host_matrix<T>, hD_1, (M, N, ldd));
    HOST_MEMCHECK(host_matrix<T>, hD_2, (M, N, ldd));
    HOST_MEMCHECK(host_matrix<T>, hD_gold, (M, N, ldd));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<T>, h_beta, (1));

    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (A_row, A_col, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dB, (B_row, B_col, ldb));
    DEVICE_MEMCHECK(device_matrix<T>, dC, (M, N, ldc));
    DEVICE_MEMCHECK(device_matrix<T>, dD, (M, N, ldd));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(hB, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
    rocblas_init_matrix(hD_1, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    hA_copy = hA;
    hB_copy = hB;
    hD_2    = hD_1;
    hD_gold = hD_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
    CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));
    CHECK_HIP_ERROR(dD.transfer_from(hD_1));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR(rocblas_geam_ex_fn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   &alpha,
                                                   dA,
                                                   a_type,
                                                   lda,
                                                   dB,
                                                   b_type,
                                                   ldb,
                                                   &beta,
                                                   dC,
                                                   c_type,
                                                   ldc,
                                                   dD,
                                                   d_type,
                                                   ldd,
                                                   compute_type,
                                                   geam_ex_op));
            handle.post_test(arg);
            CHECK_HIP_ERROR(hD_1.transfer_from(dD));
            CHECK_HIP_ERROR(dD.transfer_from(hD_2));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_ROCBLAS_ERROR(rocblas_geam_ex_fn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   d_alpha,
                                                   dA,
                                                   a_type,
                                                   lda,
                                                   dB,
                                                   b_type,
                                                   ldb,
                                                   d_beta,
                                                   dC,
                                                   c_type,
                                                   ldc,
                                                   dD,
                                                   d_type,
                                                   ldd,
                                                   compute_type,
                                                   geam_ex_op));

            CHECK_HIP_ERROR(hD_2.transfer_from(dD));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_matrix<T>, hD_copy, (M, N, ldd));

                // multi-GPU support
                int device_id, device_count;
                CHECK_HIP_ERROR(limit_device_count(device_count, (int)arg.devices));

                for(int dev_id = 0; dev_id < device_count; dev_id++)
                {
                    CHECK_HIP_ERROR(hipGetDevice(&device_id));
                    if(device_id != dev_id)
                        CHECK_HIP_ERROR(hipSetDevice(dev_id));

                    //New rocblas handle for new device
                    rocblas_local_handle handle_copy{arg};

                    //Allocate device memory in new device
                    DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (A_row, A_col, lda));
                    DEVICE_MEMCHECK(device_matrix<T>, dB_copy, (B_row, B_col, ldb));
                    DEVICE_MEMCHECK(device_matrix<T>, dC_copy, (M, N, ldc));
                    DEVICE_MEMCHECK(device_matrix<T>, dD_copy, (M, N, ldd));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(h_beta));
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(dC_copy.transfer_from(hC));
                    CHECK_HIP_ERROR(dD_copy.transfer_from(hD_1));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {

                        CHECK_ROCBLAS_ERROR(rocblas_geam_ex_fn(handle_copy,
                                                               transA,
                                                               transB,
                                                               M,
                                                               N,
                                                               K,
                                                               d_alpha_copy,
                                                               dA_copy,
                                                               a_type,
                                                               lda,
                                                               dB_copy,
                                                               b_type,
                                                               ldb,
                                                               d_beta_copy,
                                                               dC_copy,
                                                               c_type,
                                                               ldc,
                                                               dD_copy,
                                                               d_type,
                                                               ldd,
                                                               compute_type,
                                                               geam_ex_op));
                        CHECK_HIP_ERROR(hD_copy.transfer_from(dD_copy));
                        unit_check_general<T>(M, N, ldd, hD_2, hD_copy);
                    }
                }
                return;
            }
        }

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        auto ref_geam_ex_fn = geam_ex_op == rocblas_geam_ex_operation_min_plus
                                  ? ref_geam_min_plus<T>
                                  : ref_geam_plus_min<T>;

        ref_geam_ex_fn(transA,
                       transB,
                       M,
                       N,
                       K,
                       h_alpha[0],
                       (T*)hA,
                       lda,
                       (T*)hB,
                       ldb,
                       h_beta[0],
                       (T*)hC,
                       ldc,
                       (T*)hD_gold,
                       ldd);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldd, hD_gold, hD_1);
            }

            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<T>('F', M, N, ldd, (T*)hD_gold, (T*)hD_1);
            }
        }

        if(arg.pointer_mode_device)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldd, hD_gold, hD_2);
            }

            if(arg.norm_check)
            {
                rocblas_error_2 = norm_check_general<T>('F', M, N, ldd, (T*)hD_gold, (T*)hD_2);
            }
        }

        // inplace check for dD == dC
        {
            dD_in_place = dC;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            auto status_h = rocblas_geam_ex_fn(handle,
                                               transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               &alpha,
                                               dA,
                                               a_type,
                                               lda,
                                               dB,
                                               b_type,
                                               ldb,
                                               &beta,
                                               dC,
                                               c_type,
                                               ldc,
                                               dD_in_place,
                                               d_type,
                                               ldd,
                                               compute_type,
                                               geam_ex_op);

            if(ldc != ldd)
            {
                EXPECT_ROCBLAS_STATUS(status_h, rocblas_status_invalid_size);
            }
            else
            {
                CHECK_ROCBLAS_ERROR(status_h);

                CHECK_HIP_ERROR(
                    hipMemcpy(hD_1, dD_in_place, sizeof(T) * size_D, hipMemcpyDeviceToHost));

                // reference calculation
                ref_geam_ex_fn(transA,
                               transB,
                               M,
                               N,
                               K,
                               h_alpha[0],
                               (T*)hA_copy,
                               lda,
                               (T*)hB_copy,
                               ldb,
                               h_beta[0],
                               (T*)hC,
                               ldc,
                               (T*)hD_gold,
                               ldd);

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldd, hD_gold, hD_1);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldd, (T*)hD_gold, (T*)hD_1);
                }
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            rocblas_geam_ex_fn(handle,
                               transA,
                               transB,
                               M,
                               N,
                               K,
                               &alpha,
                               dA,
                               a_type,
                               lda,
                               dB,
                               b_type,
                               ldb,
                               &beta,
                               dC,
                               c_type,
                               ldc,
                               dD,
                               d_type,
                               ldd,
                               compute_type,
                               geam_ex_op);
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_geam_ex_fn(handle,
                               transA,
                               transB,
                               M,
                               N,
                               K,
                               &alpha,
                               dA,
                               a_type,
                               lda,
                               dB,
                               b_type,
                               ldb,
                               &beta,
                               dC,
                               c_type,
                               ldc,
                               dD,
                               d_type,
                               ldd,
                               compute_type,
                               geam_ex_op);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_ldb,
                      e_beta,
                      e_ldc,
                      e_ldd>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         geam_min_plus_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
