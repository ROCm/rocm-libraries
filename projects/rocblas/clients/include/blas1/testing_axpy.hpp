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

#include "testing_common.hpp"

#include "blas1/rocblas_axpy.hpp"
#include "src64/blas1/rocblas_axpy_64.hpp"

/* ============================================================================================ */
template <typename T>
void testing_axpy_bad_arg(const Arguments& arg)
{
    auto rocblas_axpy_fn = arg.api & c_API_FORTRAN ? rocblas_axpy<T, true> : rocblas_axpy<T, false>;
    auto rocblas_axpy_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_axpy_64<T, true> : rocblas_axpy_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));
        DEVICE_MEMCHECK(device_vector<T>, dy, (N, incy));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_axpy_fn,
                    (nullptr, N, alpha, dx, incx, dy, incy));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_axpy_fn,
                    (handle, N, nullptr, dx, incx, dy, incy));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_axpy_fn,
                        (handle, N, alpha, nullptr, incx, dy, incy));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_axpy_fn,
                        (handle, N, alpha, dx, incx, nullptr, incy));
        }

        // If N == 0, then alpha, X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_fn,
                    (handle, 0, nullptr, nullptr, incx, nullptr, incy));
        // If alpha == 0, then X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_fn,
                    (handle, N, zero, nullptr, incx, nullptr, incy));
    }
}

template <typename T>
void testing_axpy(const Arguments& arg)
{
    auto rocblas_axpy_fn = arg.api & c_API_FORTRAN ? rocblas_axpy<T, true> : rocblas_axpy<T, false>;
    auto rocblas_axpy_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_axpy_64<T, true> : rocblas_axpy_64<T, false>;

    int64_t              N       = arg.N;
    int64_t              incx    = arg.incx;
    int64_t              incy    = arg.incy;
    T                    h_alpha = arg.get_alpha<T>();
    bool                 HMM     = arg.HMM;
    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_axpy_fn, (handle, N, nullptr, nullptr, incx, nullptr, incy));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<T>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<T>, hy, (N, incy));
    HOST_MEMCHECK(host_vector<T>, hy_gold, (N, incy));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1, 1, HMM));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hy_gold = hy;

    double cpu_time_used;
    double rocblas_error_host   = 0.0;
    double rocblas_error_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // Allocate device memory
        DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx, HMM));
        DEVICE_MEMCHECK(device_vector<T>, dy, (N, incy, HMM));

        // copy data from CPU to device
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        if(arg.pointer_mode_host)
        {
            // ROCBLAS pointer mode host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            if(arg.api != INTERNAL)
            {
                DAPI_CHECK(rocblas_axpy_fn, (handle, N, &h_alpha, dx, incx, dy, incy));
            }
            else
            {
                // only checking offsets not alpha stride
                rocblas_stride offset_x = arg.lda;
                rocblas_stride offset_y = arg.ldb;
                DAPI_CHECK(rocblas_internal_axpy_template,
                           (handle,
                            N,
                            &h_alpha,
                            0,
                            dx + offset_x,
                            -offset_x,
                            incx,
                            arg.stride_x,
                            dy + offset_y,
                            -offset_y,
                            incy,
                            arg.stride_y,
                            1));
            }
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            // ROCBLAS pointer mode device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            CHECK_HIP_ERROR(dy.transfer_from(hy_gold)); // hy_gold not computed yet so still hy
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_axpy_fn, (handle, N, d_alpha, dx, incx, dy, incy));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<T>, hy_copy, (N, incy));
                CHECK_HIP_ERROR(hy.transfer_from(dy));

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
                    DEVICE_MEMCHECK(device_vector<T>, dx_copy, (N, incx, HMM));
                    DEVICE_MEMCHECK(device_vector<T>, dy_copy, (N, incy, HMM));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1, 1, HMM));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(
                        hipMemcpy(d_alpha_copy, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_axpy_fn,
                                   (handle_copy, N, d_alpha_copy, dx_copy, incx, dy_copy, incy));
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));
                        unit_check_general<T>(1, N, incy, hy, hy_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        ref_axpy<T>(N, h_alpha, hx, incx, hy_gold, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, hy_gold, hy);
            }

            if(arg.norm_check)
            {
                rocblas_error_host = norm_check_general<T>('F', 1, N, incy, hy_gold, hy);
            }
        }

        if(arg.pointer_mode_device)
        {
            // check device mode results
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, hy_gold, hy);
            }

            if(arg.norm_check)
            {
                rocblas_error_device = norm_check_general<T>('F', 1, N, incy, hy_gold, hy);
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        // Information on flush_memory_size and flush_batch_count
        // - To time axpy it is called number_hot_calls times.
        // - if the size of dx and dy are small enough they will be cached
        //   and reused number_hot_calls-1 times.
        // - This "hot-cache" timing will give higher performance than if the
        //   cache is flushed
        // - arg.flush_batch_count or arg.flush_memory_size can be used to avoid caching of dx and dy
        // - if arg.flush_memory_size is specified, then flush_batch_count is calculated
        // - only one of arg.flush_memory_size or arg.flush_batch_count can be used, not both
        // - Note that this is only used in timing code, not in testing code.
        // - The method is as outlined in
        //   "Achieving accurate and context-sensitive timing for code optimization" by Whaley and Castaldo.
        // - In the number_hot_calls timing loop it cycles through the arg.flush_batch_count copies
        //   of dx_rot_buff and dy_rot_buff, and if flush_memory_size is large enough they will be evicted
        //   from cache before they are reused.
        // - The individual vectors in the dx_rot_buff and dy_rot_buff rotating buffers are aligned on the
        //   same byte boundaries provided by hipMalloc.
        size_t stride_x         = N * (incx >= 0 ? incx : -incx);
        size_t stride_y         = N * (incy >= 0 ? incy : -incy);
        stride_x                = stride_x == 0 ? 1 : stride_x;
        stride_y                = stride_y == 0 ? 1 : stride_y;
        size_t aligned_stride_x = align_stride<T>(stride_x);
        size_t aligned_stride_y = align_stride<T>(stride_y);

        size_t flush_batch_count = 1;
        if(arg.timing)
        {
            size_t x_size          = N * sizeof(T);
            size_t y_size          = N * sizeof(T);
            size_t x_y_cached_size = x_size + y_size;

            flush_batch_count = calculate_flush_batch_count(
                arg.flush_batch_count, arg.flush_memory_size, x_y_cached_size);
        }

        // allocate device rotating buffer arrays
        device_strided_batch_vector<T> dx_rot_buff(
            N, incx, aligned_stride_x, flush_batch_count, HMM);
        device_strided_batch_vector<T> dy_rot_buff(
            N, incy, aligned_stride_y, flush_batch_count, HMM);

        CHECK_HIP_ERROR(dx_rot_buff.broadcast_one_vector_from(hx));
        CHECK_HIP_ERROR(dy_rot_buff.broadcast_one_vector_from(hy));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            int flush_index = (iter + 1) % flush_batch_count;
            DAPI_DISPATCH(rocblas_axpy_fn,
                          (handle,
                           N,
                           &h_alpha,
                           dx_rot_buff[flush_index],
                           incx,
                           dy_rot_buff[flush_index],
                           incy));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy>{}.log_args<T>(rocblas_cout,
                                                                  arg,
                                                                  gpu_time_used,
                                                                  axpy_gflop_count<T>(N),
                                                                  axpy_gbyte_count<T>(N),
                                                                  cpu_time_used,
                                                                  rocblas_error_host,
                                                                  rocblas_error_device);
    }
}
