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

using hipblasSyr2Model = ArgumentModel<e_a_type, e_uplo, e_N, e_alpha, e_incx, e_incy, e_lda>;

inline void testname_syr2(const Arguments& arg, std::string& name)
{
    hipblasSyr2Model{}.test_name(arg, name);
}

template <typename T>
void testing_syr2_bad_arg(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    auto hipblasSyr2Fn = arg.api == FORTRAN ? hipblasSyr2<T, true> : hipblasSyr2<T, false>;
    auto hipblasSyr2Fn_64
        = arg.api == FORTRAN_64 ? hipblasSyr2_64<T, true> : hipblasSyr2_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N    = 100;
        int64_t           lda  = 100;
        int64_t           incx = 1;
        int64_t           incy = 1;

        device_vector<T> d_alpha(1), d_zero(1);

        const Ts  h_alpha{1}, h_zero{0};
        const Ts* alpha = &h_alpha;
        const Ts* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        // Allocate device memory
        device_matrix<T> dA(N, N, lda);
        device_vector<T> dx(N, incx);
        device_vector<T> dy(N, incy);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasSyr2Fn,
                    (nullptr, uplo, N, alpha, dx, incx, dy, incy, dA, lda));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSyr2Fn,
                    (handle, HIPBLAS_FILL_MODE_FULL, N, alpha, dx, incx, dy, incy, dA, lda));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_ENUM,
            hipblasSyr2Fn,
            (handle, (hipblasFillMode_t)HIPBLAS_OP_N, N, alpha, dx, incx, dy, incy, dA, lda));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSyr2Fn,
                        (handle, uplo, N, nullptr, dx, incx, dy, incy, dA, lda));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // For device mode in rocBLAS we don't have checks for dA, dx as we may be able to quick return
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyr2Fn,
                            (handle, uplo, N, alpha, nullptr, incx, dy, incy, dA, lda));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyr2Fn,
                            (handle, uplo, N, alpha, dx, incx, nullptr, incy, dA, lda));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyr2Fn,
                            (handle, uplo, N, alpha, dx, incx, dy, incy, nullptr, lda));

                // testing the 64-bit interface for n and lda
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                 : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSyr2Fn,
                            (handle,
                             uplo,
                             c_i32_overflow,
                             zero,
                             nullptr,
                             incx,
                             nullptr,
                             incy,
                             nullptr,
                             c_i32_overflow));
            }

            // With alpha == 0, can have all nullptrs
            DAPI_CHECK(hipblasSyr2Fn,
                       (handle, uplo, N, zero, nullptr, incx, nullptr, incy, nullptr, lda));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasSyr2Fn,
                   (handle, uplo, 0, nullptr, nullptr, incx, nullptr, incy, nullptr, lda));
    }
}

template <typename T>
void testing_syr2(const Arguments& arg)
{
    using Ts           = hipblas_internal_type<T>;
    auto hipblasSyr2Fn = arg.api == FORTRAN ? hipblasSyr2<T, true> : hipblasSyr2<T, false>;
    auto hipblasSyr2Fn_64
        = arg.api == FORTRAN_64 ? hipblasSyr2_64<T, true> : hipblasSyr2_64<T, false>;

    hipblasFillMode_t uplo = char2hipblas_fill(arg.uplo);
    int64_t           N    = arg.N;
    int64_t           incx = arg.incx;
    int64_t           incy = arg.incy;
    int64_t           lda  = arg.lda;

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy || lda < N || lda < 1;
    if(invalid_size || !N)
    {
        DAPI_EXPECT((invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS),
                    hipblasSyr2Fn,
                    (handle, uplo, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(N, N, lda);
    host_matrix<T> hA_cpu(N, N, lda);
    host_matrix<T> hA_host(N, N, lda);
    host_matrix<T> hA_device(N, N, lda);
    host_vector<T> hx(N, incx);
    host_vector<T> hy(N, incy);

    // Allocate device memory
    device_matrix<T> dA(N, N, lda);
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);
    device_vector<T> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    T h_alpha = arg.get_alpha<T>();

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_symmetric_matrix, true, false);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan);

    //copy vector
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasSyr2Fn,
                   (handle, uplo, N, reinterpret_cast<Ts*>(&h_alpha), dx, incx, dy, incy, dA, lda));

        CHECK_HIP_ERROR(hA_host.transfer_from(dA));
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasSyr2Fn, (handle, uplo, N, d_alpha, dx, incx, dy, incy, dA, lda));

        CHECK_HIP_ERROR(hA_device.transfer_from(dA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_syr2<T>(uplo, N, h_alpha, hx.data(), incx, hy.data(), incy, hA_cpu.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, lda, hA_cpu.data(), hA_host.data());
            unit_check_general<T>(N, N, lda, hA_cpu.data(), hA_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, lda, hA_cpu.data(), hA_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', N, N, lda, hA_cpu.data(), hA_device.data());
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasSyr2Fn, (handle, uplo, N, d_alpha, dx, incx, dy, incy, dA, lda));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSyr2Model{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       syr2_gflop_count<T>(N),
                                       syr2_gbyte_count<T>(N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
