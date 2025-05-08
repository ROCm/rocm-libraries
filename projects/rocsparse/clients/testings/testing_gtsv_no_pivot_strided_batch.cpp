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
void testing_gtsv_no_pivot_strided_batch_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle       = local_handle;
    rocsparse_int    m            = safe_size;
    rocsparse_int    batch_count  = safe_size;
    rocsparse_int    batch_stride = safe_size;
    const T*         dl           = (const T*)0x4;
    const T*         d            = (const T*)0x4;
    const T*         du           = (const T*)0x4;
    T*               x            = (T*)0x4;
    size_t*          buffer_size  = (size_t*)0x4;
    void*            temp_buffer  = (void*)0x4;

    int       nargs_to_exclude_solve   = 1;
    const int args_to_exclude_solve[1] = {8};

#define PARAMS_BUFFER_SIZE handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size
#define PARAMS_SOLVE handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer

    bad_arg_analysis(rocsparse_gtsv_no_pivot_strided_batch_buffer_size<T>, PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(rocsparse_gtsv_no_pivot_strided_batch<T>,
                            nargs_to_exclude_solve,
                            args_to_exclude_solve,
                            PARAMS_SOLVE);

    // m > 512
    batch_stride = m = 513;
    temp_buffer      = (void*)nullptr;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gtsv_no_pivot_strided_batch<T>(PARAMS_SOLVE),
                            rocsparse_status_invalid_pointer);
    batch_stride = m = safe_size;
    temp_buffer      = (void*)0x4;

    // m <= 1
    m = 1;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_gtsv_no_pivot_strided_batch_buffer_size<T>(PARAMS_BUFFER_SIZE),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gtsv_no_pivot_strided_batch<T>(PARAMS_SOLVE),
                            rocsparse_status_invalid_size);
    m = safe_size;

    // batch_stride < m
    m            = 4;
    batch_stride = 2;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_gtsv_no_pivot_strided_batch_buffer_size<T>(PARAMS_BUFFER_SIZE),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gtsv_no_pivot_strided_batch<T>(PARAMS_SOLVE),
                            rocsparse_status_invalid_size);

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_SOLVE
}

template <typename T>
void testing_gtsv_no_pivot_strided_batch(const Arguments& arg)
{
    rocsparse_int m            = arg.M;
    rocsparse_int batch_count  = arg.N;
    rocsparse_int batch_stride = arg.denseld;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

#define PARAMS_BUFFER_SIZE handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, &buffer_size
#define PARAMS_SOLVE handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, dbuffer

    if(batch_stride < m)
    {
        return;
    }

    rocsparse_seedrand();

    // Host tri-diagonal matrix
    host_vector<T> hdl(batch_stride * batch_count, static_cast<T>(7));
    host_vector<T> hd(batch_stride * batch_count, static_cast<T>(7));
    host_vector<T> hdu(batch_stride * batch_count, static_cast<T>(7));

    // initialize tri-diagonal matrix
    for(rocsparse_int j = 0; j < batch_count; ++j)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hdl[j * batch_stride + i] = random_cached_generator<T>(1, 8);
            hd[j * batch_stride + i]  = random_cached_generator<T>(17, 32);
            hdu[j * batch_stride + i] = random_cached_generator<T>(1, 8);
        }

        hdl[j * batch_stride + 0]     = static_cast<T>(0);
        hdu[j * batch_stride + m - 1] = static_cast<T>(0);
    }

    // Host dense rhs
    host_vector<T> hx(batch_stride * batch_count, static_cast<T>(7));

    for(rocsparse_int j = 0; j < batch_count; ++j)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hx[j * batch_stride + i] = random_cached_generator<T>(-10, 10);
        }
    }

    host_vector<T> hx_original = hx;

    // Device tri-diagonal matrix
    device_vector<T> ddl(batch_stride * batch_count);
    device_vector<T> dd(batch_stride * batch_count);
    device_vector<T> ddu(batch_stride * batch_count);

    // Device dense rhs
    device_vector<T> dx(batch_stride * batch_count);

    // Copy to device
    ddl.transfer_from(hdl);
    dd.transfer_from(hd);
    ddu.transfer_from(hdu);
    dx.transfer_from(hx);

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_gtsv_no_pivot_strided_batch_buffer_size<T>(PARAMS_BUFFER_SIZE));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gtsv_no_pivot_strided_batch<T>(PARAMS_SOLVE));

        hx.transfer_from(dx);

        // Check
        std::vector<T> hresult(batch_stride * batch_count, static_cast<T>(7));

        for(rocsparse_int j = 0; j < batch_count; j++)
        {
            rocsparse_int offset = batch_stride * j;

            hresult[offset] = hd[offset + 0] * hx[offset] + hdu[offset + 0] * hx[offset + 1];
            hresult[offset + m - 1] = hdl[offset + m - 1] * hx[offset + m - 2]
                                      + hd[offset + m - 1] * hx[offset + m - 1];
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(rocsparse_int i = 1; i < m - 1; i++)
            {
                hresult[offset + i] = hdl[offset + i] * hx[offset + i - 1]
                                      + hd[offset + i] * hx[offset + i]
                                      + hdu[offset + i] * hx[offset + i + 1];
            }
        }

        near_check_segments<T>(batch_stride * batch_count, hx_original.data(), hresult.data());
    }

    if(arg.timing)
    {
        const double gpu_solve_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_gtsv_no_pivot_strided_batch<T>, PARAMS_SOLVE);

        double gbyte_count = gtsv_strided_batch_gbyte_count<T>(m, batch_count);
        double gpu_gbyte   = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);
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

#define INSTANTIATE(TYPE)                                                                  \
    template void testing_gtsv_no_pivot_strided_batch_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gtsv_no_pivot_strided_batch<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_gtsv_no_pivot_strided_batch_extra(const Arguments& arg) {}
