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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasAxpyStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_alpha, e_incx, e_incy, e_stride_scale, e_batch_count>;

inline void testname_axpy_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasAxpyStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_axpy_strided_batched_bad_arg(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAxpyStridedBatchedFn
        = FORTRAN ? hipblasAxpyStridedBatched<T, true> : hipblasAxpyStridedBatched<T, false>;
    auto hipblasAxpyStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasAxpyStridedBatched_64<T, true>
                                              : hipblasAxpyStridedBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t       N           = 100;
        int64_t       incx        = 1;
        int64_t       incy        = 1;
        int64_t       batch_count = 2;
        hipblasStride stride_x    = N;
        hipblasStride stride_y    = N;

        device_vector<T>               d_alpha(1), d_zero(1);
        device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
        device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

        const Ts  h_alpha{1}, h_zero{0};
        const Ts* alpha = &h_alpha;
        const Ts* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(h_alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, &h_zero, sizeof(h_zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasAxpyStridedBatchedFn,
                    (nullptr, N, alpha, dx, incx, stride_x, dy, incy, stride_y, batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasAxpyStridedBatchedFn,
                    (handle, N, nullptr, dx, incx, stride_x, dy, incy, stride_y, batch_count));

        // Can only check for nullptr for dx/dy with host mode because
        //device mode may not check as it could be quick-return success
        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasAxpyStridedBatchedFn,
                (handle, N, alpha, nullptr, incx, stride_x, dy, incy, stride_y, batch_count));
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasAxpyStridedBatchedFn,
                (handle, N, alpha, dx, incx, stride_x, nullptr, incy, stride_y, batch_count));
        }

        DAPI_CHECK(
            hipblasAxpyStridedBatchedFn,
            (handle, 0, nullptr, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count));
        DAPI_CHECK(
            hipblasAxpyStridedBatchedFn,
            (handle, N, zero, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count));
        DAPI_CHECK(hipblasAxpyStridedBatchedFn,
                   (handle, N, nullptr, nullptr, incx, stride_x, nullptr, incy, stride_y, 0));
    }
}

template <typename T>
void testing_axpy_strided_batched(const Arguments& arg)
{
    using Ts     = hipblas_internal_type<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAxpyStridedBatchedFn
        = FORTRAN ? hipblasAxpyStridedBatched<T, true> : hipblasAxpyStridedBatched<T, false>;
    auto hipblasAxpyStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasAxpyStridedBatched_64<T, true>
                                              : hipblasAxpyStridedBatched_64<T, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int64_t incy         = arg.incy;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;
    T       alpha        = arg.get_alpha<T>();

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;

    hipblasStride stride_x = N * abs_incx * stride_scale;
    hipblasStride stride_y = N * abs_incy * stride_scale;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(
            hipblasAxpyStridedBatchedFn,
            (handle, N, nullptr, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hx_cpu(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy_host(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_device(N, incy, stride_y, batch_count);
    host_strided_batch_vector<T> hy_cpu(N, incy, stride_y, batch_count);

    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, arg, hipblas_client_alpha_sets_nan, false);

    hx_cpu.copy_from(hx);
    hy_cpu.copy_from(hy_host);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasAxpyStridedBatchedFn,
                   (handle, N, d_alpha, dx, incx, stride_x, dy, incy, stride_y, batch_count));

        CHECK_HIP_ERROR(hy_device.transfer_from(dy));
        CHECK_HIP_ERROR(dy.transfer_from(hy_host));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasAxpyStridedBatchedFn,
                   (handle,
                    N,
                    reinterpret_cast<Ts*>(&alpha),
                    dx,
                    incx,
                    stride_x,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hy_host.transfer_from(dy));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_axpy<T>(N, alpha, hx_cpu[b], incx, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(
                1, N, batch_count, abs_incy, stride_y, hy_cpu.data(), hy_host.data());
            unit_check_general<T>(
                1, N, batch_count, abs_incy, stride_y, hy_cpu.data(), hy_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, N, abs_incy, stride_y, hy_cpu.data(), hy_host.data(), batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, N, abs_incy, stride_y, hy_cpu.data(), hy_device.data(), batch_count);
        }
    } // end of if unit check

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

            DAPI_CHECK(hipblasAxpyStridedBatchedFn,
                       (handle, N, d_alpha, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     axpy_gflop_count<T>(N),
                                                     axpy_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
