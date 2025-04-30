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

//The Template parameter Ti, Tex and To is to test the special cases where the input/compute/output types could be HSH (Half, single, half), HSS (Half, single, single),
// TST (rocblas_bfloat16, single, rocblas_bfloat16), TSS (rocblas_bfloat16, single, single)
// Ti==Tex==To (float, double, rocblas_complex_float, rocblas_complex double)
template <typename Ti, typename Tex = Ti, typename To = Tex>
void testing_gemv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gemv_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_gemv_strided_batched<Ti, Tex, To, true>
                                               : rocblas_gemv_strided_batched<Ti, Tex, To, false>;
    auto rocblas_gemv_strided_batched_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemv_strided_batched_64<Ti, Tex, To, true>
                                  : rocblas_gemv_strided_batched_64<Ti, Tex, To, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_operation transA      = rocblas_operation_none;
        const int64_t           M           = 100;
        const int64_t           N           = 100;
        const int64_t           lda         = 100;
        const int64_t           incx        = 1;
        const int64_t           incy        = 1;
        const int64_t           stride_a    = 10000;
        const int64_t           stride_x    = 100;
        const int64_t           stride_y    = 100;
        const int64_t           batch_count = 2;

        DEVICE_MEMCHECK(device_vector<Tex>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<Tex>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<Tex>, zero_d, (1));
        DEVICE_MEMCHECK(device_vector<Tex>, one_d, (1));

        const Tex alpha_h(1), beta_h(1), zero_h(0), one_h(1);

        const Tex* alpha = &alpha_h;
        const Tex* beta  = &beta_h;
        const Tex* zero  = &zero_h;
        const Tex* one   = &one_h;

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

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<Ti>, dA, (M, N, lda, stride_a, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<Ti>, dx, (N, incx, stride_x, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<To>, dy, (M, incy, stride_y, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_gemv_strided_batched_fn,
                    (nullptr,
                     transA,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gemv_strided_batched_fn,
                    (handle,
                     (rocblas_operation)rocblas_fill_full,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gemv_strided_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     nullptr,
                     dA,
                     lda,
                     stride_a,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gemv_strided_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dx,
                     incx,
                     stride_x,
                     nullptr,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gemv_strided_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         alpha,
                         nullptr,
                         lda,
                         stride_a,
                         dx,
                         incx,
                         stride_x,
                         beta,
                         dy,
                         incy,
                         stride_y,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gemv_strided_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         stride_a,
                         nullptr,
                         incx,
                         stride_x,
                         beta,
                         dy,
                         incy,
                         stride_y,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gemv_strided_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         stride_a,
                         dx,
                         incx,
                         stride_x,
                         beta,
                         nullptr,
                         incy,
                         stride_y,
                         batch_count));
        }

        // When M==0, all pointers may be nullptr without error
        DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                   (handle,
                    transA,
                    0,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));

        // When N==0, all pointers may be nullptr without error
        DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));

        // When alpha==0, A and x may be nullptr without error
        DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    zero,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    incx,
                    stride_x,
                    beta,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        // When alpha==0 && beta==1, A, x and y may be nullptr without error
        DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    zero,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    incx,
                    stride_x,
                    one,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));

        // When batch_count==0, all pointers may be nullptr without error
        DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    0));
    }
}

//The Template parameter Ti, Tex and To is to test the special cases where the input/compute/output types could be HSH (Half, single, half), HSS (Half, single, single),
// TST (rocblas_bfloat16, single, rocblas_bfloat16), TSS (rocblas_bfloat16, single, single)
// Ti==Tex==To (float, double, rocblas_complex_float, rocblas_complex double)
template <typename Ti, typename Tex = Ti, typename To = Tex>
void testing_gemv_strided_batched(const Arguments& arg)
{
    auto rocblas_gemv_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_gemv_strided_batched<Ti, Tex, To, true>
                                               : rocblas_gemv_strided_batched<Ti, Tex, To, false>;
    auto rocblas_gemv_strided_batched_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemv_strided_batched_64<Ti, Tex, To, true>
                                  : rocblas_gemv_strided_batched_64<Ti, Tex, To, false>;

    int64_t           M           = arg.M;
    int64_t           N           = arg.N;
    int64_t           lda         = arg.lda;
    int64_t           incx        = arg.incx;
    int64_t           incy        = arg.incy;
    Tex               h_alpha     = arg.get_alpha<Tex>();
    Tex               h_beta      = arg.get_beta<Tex>();
    rocblas_operation transA      = char2rocblas_operation(arg.transA);
    int64_t           stride_a    = arg.stride_a;
    int64_t           stride_x    = arg.stride_x;
    int64_t           stride_y    = arg.stride_y;
    int64_t           batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    size_t dim_x, row_A;
    size_t dim_y, col_A;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
        row_A = M;
        col_A = N;
    }
    else
    {
        dim_x = M;
        dim_y = N;
        row_A = N;
        col_A = M;
    }

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_gemv_strided_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     stride_a,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     nullptr,
                     incy,
                     stride_y,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<Ti>, hA, (M, N, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<Ti>, hx, (dim_x, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<To>, hy, (dim_y, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<To>, hy_gold, (dim_y, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_vector<Tex>, halpha, (1));
    HOST_MEMCHECK(host_vector<Tex>, hbeta, (1));
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<Ti>, dA, (M, N, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<Ti>, dx, (dim_x, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<To>, dy, (dim_y, incy, stride_y, batch_count));
    DEVICE_MEMCHECK(device_vector<Tex>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<Tex>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    hy_gold.copy_from(hy);

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
            DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                       (handle,
                        transA,
                        M,
                        N,
                        &h_alpha,
                        dA,
                        lda,
                        stride_a,
                        dx,
                        incx,
                        stride_x,
                        &h_beta,
                        dy,
                        incy,
                        stride_y,
                        batch_count));
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
            DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                       (handle,
                        transA,
                        M,
                        N,
                        d_alpha,
                        dA,
                        lda,
                        stride_a,
                        dx,
                        incx,
                        stride_x,
                        d_beta,
                        dy,
                        incy,
                        stride_y,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                //Transfer original results from device to host
                CHECK_HIP_ERROR(hy.transfer_from(dy));
                //Host buffer to store results subsequent iterations
                HOST_MEMCHECK(
                    host_strided_batch_vector<To>, hy_copy, (dim_y, incy, stride_y, batch_count));

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

                    // Allocate device memory
                    DEVICE_MEMCHECK(device_strided_batch_matrix<Ti>,
                                    dA_copy,
                                    (M, N, lda, stride_a, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_vector<Ti>,
                                    dx_copy,
                                    (dim_x, incx, stride_x, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_vector<To>,
                                    dy_copy,
                                    (dim_y, incy, stride_y, batch_count));
                    DEVICE_MEMCHECK(device_vector<Tex>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<Tex>, d_beta_copy, (1));

                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(hbeta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_gemv_strided_batched_fn,
                                   (handle_copy,
                                    transA,
                                    M,
                                    N,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    stride_a,
                                    dx_copy,
                                    incx,
                                    stride_x,
                                    d_beta_copy,
                                    dy_copy,
                                    incy,
                                    stride_y,
                                    batch_count));
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));

                        unit_check_general<To>(1, dim_y, incy, stride_y, hy, hy_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int64_t b = 0; b < batch_count; ++b)
        {
            ref_gemv<Ti, To>(
                transA, M, N, h_alpha, hA[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_hy_to_gold = [&] {
            if(arg.unit_check)
            {
                bool use_near = reduction_requires_near<To>(arg, dim_x);
                if(use_near)
                {
                    const double tol = dim_x * sum_error_tolerance<To>;
                    near_check_general<To>(1, dim_y, incy, stride_y, hy_gold, hy, batch_count, tol);
                }
                else
                {
                    unit_check_general<To>(1, dim_y, incy, stride_y, hy_gold, hy, batch_count);
                }
            }
            double error = 0;
            if(arg.norm_check)
                error = norm_check_general<To>(
                    'F', 1, dim_y, incy, stride_y, hy_gold, hy, batch_count);
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

            DAPI_DISPATCH(rocblas_gemv_strided_batched_fn,
                          (handle,
                           transA,
                           M,
                           N,
                           &h_alpha,
                           dA,
                           lda,
                           stride_a,
                           dx,
                           incx,
                           stride_x,
                           &h_beta,
                           dy,
                           incy,
                           stride_y,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_beta,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<Tex>(rocblas_cout,
                           arg,
                           gpu_time_used,
                           gemv_gflop_count<Tex>(transA, M, N),
                           gemv_gbyte_count<Tex>(transA, M, N),
                           cpu_time_used,
                           error_host,
                           error_device);
    }
}
