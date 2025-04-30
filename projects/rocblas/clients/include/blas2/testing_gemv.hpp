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
 *
 * ************************************************************************ */

#pragma once

#include "testing_common.hpp"

template <typename T>
void testing_gemv_bad_arg(const Arguments& arg)
{
    auto rocblas_gemv_fn = arg.api & c_API_FORTRAN ? rocblas_gemv<T, true> : rocblas_gemv<T, false>;
    auto rocblas_gemv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemv_64<T, true> : rocblas_gemv_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_operation transA = rocblas_operation_none;
        const int64_t           M      = 100;
        const int64_t           N      = 100;
        const int64_t           lda    = 100;
        const int64_t           incx   = 1;
        const int64_t           incy   = 1;

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, one_d, (1));

        const T alpha_h(1), beta_h(1), zero_h(0), one_h(1);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* zero  = &zero_h;
        const T* one   = &one_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
        }

        // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
        // Allocate host memory
        HOST_MEMCHECK(host_matrix<T>, hA, (M, N, lda));
        HOST_MEMCHECK(host_vector<T>, hx, (N, incx));
        HOST_MEMCHECK(host_vector<T>, hy, (N, incy));

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (M, N, lda));
        DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));
        DEVICE_MEMCHECK(device_vector<T>, dy, (N, incy));

        // Initialize data on host memory
        rocblas_init_matrix(
            hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
        rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
        rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_gemv_fn,
                    (nullptr, transA, M, N, alpha, dA, lda, dx, incx, beta, dy, incy));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gemv_fn,
                    (handle,
                     (rocblas_operation)rocblas_fill_full,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dx,
                     incx,
                     beta,
                     dy,
                     incy));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gemv_fn,
                    (handle, transA, M, N, nullptr, dA, lda, dx, incx, beta, dy, incy));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gemv_fn,
                    (handle, transA, M, N, alpha, dA, lda, dx, incx, nullptr, dy, incy));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gemv_fn,
                        (handle, transA, M, N, alpha, nullptr, lda, dx, incx, beta, dy, incy));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gemv_fn,
                        (handle, transA, M, N, alpha, dA, lda, nullptr, incx, beta, dy, incy));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gemv_fn,
                        (handle, transA, M, N, alpha, dA, lda, dx, incx, beta, nullptr, incy));
        }

        // If M==0, then all pointers may be nullptr without error
        DAPI_CHECK(
            rocblas_gemv_fn,
            (handle, transA, 0, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy));

        // If N==0, then all pointers may be nullptr without error
        DAPI_CHECK(
            rocblas_gemv_fn,
            (handle, transA, M, 0, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy));

        // If alpha==0, then A and X may be nullptr without error
        DAPI_CHECK(rocblas_gemv_fn,
                   (handle, transA, M, N, zero, nullptr, lda, nullptr, incx, beta, dy, incy));

        // If alpha==0 && beta==1, then A, X and Y may be nullptr without error
        DAPI_CHECK(rocblas_gemv_fn,
                   (handle, transA, M, N, zero, nullptr, lda, nullptr, incx, one, nullptr, incy));
    }
}

template <typename T>
void testing_gemv(const Arguments& arg)
{
    auto rocblas_gemv_fn = arg.api & c_API_FORTRAN ? rocblas_gemv<T, true> : rocblas_gemv<T, false>;
    auto rocblas_gemv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemv_64<T, true> : rocblas_gemv_64<T, false>;

    int64_t           M       = arg.M;
    int64_t           N       = arg.N;
    int64_t           lda     = arg.lda;
    int64_t           incx    = arg.incx;
    int64_t           incy    = arg.incy;
    T                 h_alpha = arg.get_alpha<T>();
    T                 h_beta  = arg.get_beta<T>();
    rocblas_operation transA  = char2rocblas_operation(arg.transA);
    bool              HMM     = arg.HMM;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy;
    if(invalid_size || !M || !N)
    {
        DAPI_EXPECT(
            invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
            rocblas_gemv_fn,
            (handle, transA, M, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy));

        return;
    }

    size_t dim_x;
    size_t dim_y;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (M, N, lda));
    HOST_MEMCHECK(host_vector<T>, hx, (dim_x, incx));
    HOST_MEMCHECK(host_vector<T>, hy, (dim_y, incy));
    HOST_MEMCHECK(host_vector<T>, hy_gold, (dim_y, incy));
    HOST_MEMCHECK(host_vector<T>, halpha, (1));
    HOST_MEMCHECK(host_vector<T>, hbeta, (1));
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (M, N, lda, HMM));
    DEVICE_MEMCHECK(device_vector<T>, dx, (dim_x, incx, HMM));
    DEVICE_MEMCHECK(device_vector<T>, dy, (dim_y, incy, HMM));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1, 1, HMM));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1, 1, HMM));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_gold = hy: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;
    double error_host;
    double error_device;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gemv_fn,
                       (handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy, incy));
            handle.post_test(arg);

            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));
            CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gemv_fn,
                       (handle, transA, M, N, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                //Transfer original results from device to host
                CHECK_HIP_ERROR(hy.transfer_from(dy));
                //Host buffer to store results subsequent iterations
                HOST_MEMCHECK(host_vector<T>, hy_copy, (dim_y, incy));

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
                    DEVICE_MEMCHECK(device_vector<T>, dy_copy, (dim_y, incy, HMM));
                    DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (M, N, lda, HMM));
                    DEVICE_MEMCHECK(device_vector<T>, dx_copy, (dim_x, incx, HMM));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1, 1, HMM));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1, 1, HMM));

                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(hbeta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_gemv_fn,
                                   (handle_copy,
                                    transA,
                                    M,
                                    N,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    dx_copy,
                                    incx,
                                    d_beta_copy,
                                    dy_copy,
                                    incy));

                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));
                        unit_check_general<T>(1, dim_y, incy, hy, hy_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        ref_gemv<T>(transA, M, N, h_alpha, (T*)hA, lda, (T*)hx, incx, h_beta, (T*)hy_gold, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_hy_to_gold = [&] {
            if(arg.unit_check)
            {
                bool use_near = reduction_requires_near<T>(arg, dim_x);
                if(use_near)
                {
                    const double tol = dim_x * sum_error_tolerance<T>;
                    near_check_general<T>(1, dim_y, incy, hy_gold, hy, tol);
                }
                else
                {
                    unit_check_general<T>(1, dim_y, incy, hy_gold, hy);
                }
            }
            double error = 0;
            if(arg.norm_check)
                error = norm_check_general<T>('F', 1, dim_y, incy, hy_gold, hy);
            return error;
        };

        if(arg.pointer_mode_host)
        {
            error_host = compare_hy_to_gold();
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hy.transfer_from(dy));
            error_device = compare_hy_to_gold();
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_gemv_fn,
                          (handle, transA, M, N, &h_alpha, dA, lda, dx, incx, &h_beta, dy, incy));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_M, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            gemv_gflop_count<T>(transA, M, N),
            gemv_gbyte_count<T>(transA, M, N),
            cpu_time_used,
            error_host,
            error_device);
    }
}
