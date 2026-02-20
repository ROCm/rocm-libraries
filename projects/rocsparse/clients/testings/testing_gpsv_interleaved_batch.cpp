/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_gpsv_interleaved_batch_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle               handle       = local_handle;
    rocsparse_gpsv_interleaved_alg alg          = rocsparse_gpsv_interleaved_alg_default;
    rocsparse_int                  m            = safe_size;
    rocsparse_int                  batch_count  = safe_size;
    rocsparse_int                  batch_stride = safe_size;
    T*                             ds           = (T*)0x4;
    T*                             dl           = (T*)0x4;
    T*                             d            = (T*)0x4;
    T*                             du           = (T*)0x4;
    T*                             dw           = (T*)0x4;
    T*                             x            = (T*)0x4;
    size_t*                        buffer_size  = (size_t*)0x4;
    void*                          temp_buffer  = (void*)0x4;

#define PARAMS_BUFFER_SIZE \
    handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer_size
#define PARAMS_SOLVE handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, temp_buffer

    bad_arg_analysis(rocsparse_gpsv_interleaved_batch_buffer_size<T>, PARAMS_BUFFER_SIZE);
    bad_arg_analysis(rocsparse_gpsv_interleaved_batch<T>, PARAMS_SOLVE);

    // m < 3
    m = 2;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gpsv_interleaved_batch_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gpsv_interleaved_batch<T>(PARAMS_SOLVE),
                            rocsparse_status_invalid_size);
    m = safe_size;

    // batch_stride < batch_count
    batch_count  = 4;
    batch_stride = 2;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gpsv_interleaved_batch_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gpsv_interleaved_batch<T>(PARAMS_SOLVE),
                            rocsparse_status_invalid_size);

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_SOLVE
}

template <typename T>
void testing_gpsv_interleaved_batch(const Arguments& arg)
{
    rocsparse_int                  m            = arg.M;
    rocsparse_int                  batch_count  = arg.batch_count;
    rocsparse_int                  batch_stride = arg.batch_stride;
    rocsparse_gpsv_interleaved_alg alg          = arg.gpsv_interleaved_alg;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

#define PARAMS_BUFFER_SIZE \
    handle, alg, m, dds, ddl, dd, ddu, ddw, dx, batch_count, batch_stride, &buffer_size
#define PARAMS_SOLVE handle, alg, m, dds, ddl, dd, ddu, ddw, dx, batch_count, batch_stride, dbuffer

    if(batch_stride < batch_count)
    {
        return;
    }

    rocsparse_seedrand();

    // Host pentadiagonal matrix
    host_vector<T> hds(m * batch_stride);
    host_vector<T> hdl(m * batch_stride);
    host_vector<T> hd(m * batch_stride);
    host_vector<T> hdu(m * batch_stride);
    host_vector<T> hdw(m * batch_stride);

    // initialize interleaved pentadiagonal matrix
    for(rocsparse_int b = 0; b < batch_count; ++b)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hds[batch_stride * i + b] = random_cached_generator<T>(1, 8);
            hdl[batch_stride * i + b] = random_cached_generator<T>(1, 8);
            hd[batch_stride * i + b]  = random_cached_generator<T>(17, 32);
            hdu[batch_stride * i + b] = random_cached_generator<T>(1, 8);
            hdw[batch_stride * i + b] = random_cached_generator<T>(1, 8);
        }

        hds[batch_stride * 0 + b]       = static_cast<T>(0);
        hds[batch_stride * 1 + b]       = static_cast<T>(0);
        hdl[batch_stride * 0 + b]       = static_cast<T>(0);
        hdu[batch_stride * (m - 1) + b] = static_cast<T>(0);
        hdw[batch_stride * (m - 1) + b] = static_cast<T>(0);
        hdw[batch_stride * (m - 2) + b] = static_cast<T>(0);
    }

    // Host dense rhs
    host_vector<T> hx(m * batch_stride);

    for(rocsparse_int b = 0; b < batch_count; ++b)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hx[batch_stride * i + b] = random_cached_generator<T>(-10, 10);
        }
    }

    host_vector<T> hx_original(m * batch_count);

    // Convert solution to non strided
    for(rocsparse_int b = 0; b < batch_count; ++b)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hx_original[batch_count * i + b] = hx[batch_stride * i + b];
        }
    }

    // Device pentadiagonal matrix
    device_vector<T> dds(hds);
    device_vector<T> ddl(hdl);
    device_vector<T> dd(hd);
    device_vector<T> ddu(hdu);
    device_vector<T> ddw(hdw);

    // Device dense rhs
    device_vector<T> dx(hx);

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_gpsv_interleaved_batch_buffer_size<T>(PARAMS_BUFFER_SIZE));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gpsv_interleaved_batch<T>(PARAMS_SOLVE));

        hx.transfer_from(dx);

        // Check
        std::vector<T> hresult(m * batch_count);

        for(rocsparse_int b = 0; b < batch_count; b++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(rocsparse_int i = 0; i < m; ++i)
            {
                T sum = hd[batch_stride * i + b] * hx[batch_stride * (i - 0) + b];

                sum += (i - 2 >= 0) ? hds[batch_stride * i + b] * hx[batch_stride * (i - 2) + b]
                                    : static_cast<T>(0);
                sum += (i - 1 >= 0) ? hdl[batch_stride * i + b] * hx[batch_stride * (i - 1) + b]
                                    : static_cast<T>(0);
                sum += (i + 1 < m) ? hdu[batch_stride * i + b] * hx[batch_stride * (i + 1) + b]
                                   : static_cast<T>(0);
                sum += (i + 2 < m) ? hdw[batch_stride * i + b] * hx[batch_stride * (i + 2) + b]
                                   : static_cast<T>(0);

                // Store the result in non strided way
                hresult[batch_count * i + b] = sum;
            }
        }

        // Only check the actual relevant content
        near_check_segments<T>(m * batch_count, hx_original.data(), hresult.data());
    }

    if(arg.timing)
    {

        const double gpu_solve_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_gpsv_interleaved_batch<T>, PARAMS_SOLVE);

        double gbyte_count = gpsv_interleaved_batch_gbyte_count<T>(m, batch_count);

        double gpu_gbyte = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::batch_count,
                            batch_count,
                            display_key_t::batch_stride,
                            batch_stride,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_SOLVE
}

#define INSTANTIATE(TYPE)                                                             \
    template void testing_gpsv_interleaved_batch_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gpsv_interleaved_batch<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_gpsv_interleaved_batch_extra(const Arguments& arg) {}
