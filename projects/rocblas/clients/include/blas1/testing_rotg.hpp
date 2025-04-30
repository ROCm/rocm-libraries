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

template <typename T, typename U = T>
void testing_rotg_bad_arg(const Arguments& arg)
{
    auto rocblas_rotg_fn
        = arg.api & c_API_FORTRAN ? rocblas_rotg<T, U, true> : rocblas_rotg<T, U, false>;
    auto rocblas_rotg_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_rotg_64<T, U, true> : rocblas_rotg_64<T, U, false>;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_vector<T> da(1, 1);
    device_vector<T> db(1, 1);
    device_vector<U> dc(1, 1);
    device_vector<T> ds(1, 1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    DAPI_EXPECT(rocblas_status_invalid_handle, rocblas_rotg_fn, (nullptr, da, db, dc, ds));
    DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_rotg_fn, (handle, nullptr, db, dc, ds));
    DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_rotg_fn, (handle, da, nullptr, dc, ds));
    DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_rotg_fn, (handle, da, db, nullptr, ds));
    DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_rotg_fn, (handle, da, db, dc, nullptr));
}

template <typename T, typename U = T>
void testing_rotg(const Arguments& arg)
{
    auto rocblas_rotg_fn
        = arg.api & c_API_FORTRAN ? rocblas_rotg<T, U, true> : rocblas_rotg<T, U, false>;
    auto rocblas_rotg_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_rotg_64<T, U, true> : rocblas_rotg_64<T, U, false>;

    rocblas_local_handle handle{arg};
    double               gpu_time_used, cpu_time_used;
    double               error_host, error_device;
    const U              rel_error = std::numeric_limits<U>::epsilon() * 10;

    host_vector<T> a(1, 1);
    host_vector<T> b(1, 1);
    host_vector<U> c(1, 1);
    host_vector<T> s(1, 1);

    // Initialize data on host memory
    a[0] = arg.get_alpha<T>(); // reuse alpha in place of a to keep number of arguments small
    b[0] = arg.get_beta<T>(); // reuse beta  in place of b to keep number of arguments small
    c[0] = U(0);
    s[0] = T(0);

    // CPU BLAS
    host_vector<T> ha_gold = a;
    host_vector<T> hb_gold = b;
    host_vector<U> hc_gold = c;
    host_vector<T> hs_gold = s;
    cpu_time_used          = get_time_us_no_sync();
    ref_rotg<T, U>(ha_gold, hb_gold, hc_gold, hs_gold);
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    // Test rocblas_pointer_mode_host
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // Naming: `h` is in CPU (host) memory(eg ha), `d` is in GPU (device) memory (eg da).
            // Allocate host memory
            host_vector<T> ha = a;
            host_vector<T> hb = b;
            host_vector<U> hc = c;
            host_vector<T> hs = s;
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_rotg_fn, (handle, ha, hb, hc, hs));
            handle.post_test(arg);

            if(arg.unit_check)
            {
                near_check_general<T>(1, 1, 1, ha_gold, ha, rel_error);
                near_check_general<T>(1, 1, 1, hb_gold, hb, rel_error);
                near_check_general<U>(1, 1, 1, hc_gold, hc, rel_error);
                near_check_general<T>(1, 1, 1, hs_gold, hs, rel_error);
            }

            if(arg.norm_check)
            {
                error_host = norm_check_general<T>('F', 1, 1, 1, ha_gold, ha);
                error_host += norm_check_general<T>('F', 1, 1, 1, hb_gold, hb);
                error_host += norm_check_general<U>('F', 1, 1, 1, hc_gold, hc);
                error_host += norm_check_general<T>('F', 1, 1, 1, hs_gold, hs);
            }
        }
    }

    // Test rocblas_pointer_mode_device
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_device)
        {
            // Allocate device memory
            device_vector<T> da(1, 1);
            device_vector<T> db(1, 1);
            device_vector<U> dc(1, 1);
            device_vector<T> ds(1, 1);

            // Check device memory allocation
            CHECK_DEVICE_ALLOCATION(da.memcheck());
            CHECK_DEVICE_ALLOCATION(db.memcheck());
            CHECK_DEVICE_ALLOCATION(dc.memcheck());
            CHECK_DEVICE_ALLOCATION(ds.memcheck());
            CHECK_DEVICE_ALLOCATION(ds.memcheck());

            // Transfer from CPU to GPU
            CHECK_HIP_ERROR(da.transfer_from(a));
            CHECK_HIP_ERROR(db.transfer_from(b));
            CHECK_HIP_ERROR(dc.transfer_from(c));
            CHECK_HIP_ERROR(ds.transfer_from(s));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_rotg_fn, (handle, da, db, dc, ds));
            handle.post_test(arg);

            host_vector<T> ha(1);
            host_vector<T> hb(1);
            host_vector<U> hc(1);
            host_vector<T> hs(1);

            // Transfer from GPU to CPU
            CHECK_HIP_ERROR(ha.transfer_from(da));
            CHECK_HIP_ERROR(hb.transfer_from(db));
            CHECK_HIP_ERROR(hc.transfer_from(dc));
            CHECK_HIP_ERROR(hs.transfer_from(ds));

            if(arg.repeatability_check)
            {
                host_vector<T> ha_copy(1);
                host_vector<T> hb_copy(1);
                host_vector<U> hc_copy(1);
                host_vector<T> hs_copy(1);
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
                    device_vector<T> da_copy(1, 1);
                    device_vector<T> db_copy(1, 1);
                    device_vector<U> dc_copy(1, 1);
                    device_vector<T> ds_copy(1, 1);

                    // Check device memory allocation
                    CHECK_DEVICE_ALLOCATION(da_copy.memcheck());
                    CHECK_DEVICE_ALLOCATION(db_copy.memcheck());
                    CHECK_DEVICE_ALLOCATION(dc_copy.memcheck());
                    CHECK_DEVICE_ALLOCATION(ds_copy.memcheck());
                    CHECK_DEVICE_ALLOCATION(ds_copy.memcheck());

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(da_copy.transfer_from(a));
                        CHECK_HIP_ERROR(db_copy.transfer_from(b));
                        CHECK_HIP_ERROR(dc_copy.transfer_from(c));
                        CHECK_HIP_ERROR(ds_copy.transfer_from(s));

                        DAPI_CHECK(rocblas_rotg_fn,
                                   (handle_copy, da_copy, db_copy, dc_copy, ds_copy));

                        CHECK_HIP_ERROR(ha_copy.transfer_from(da_copy));
                        CHECK_HIP_ERROR(hb_copy.transfer_from(db_copy));
                        CHECK_HIP_ERROR(hc_copy.transfer_from(dc_copy));
                        CHECK_HIP_ERROR(hs_copy.transfer_from(ds_copy));
                        unit_check_general<T>(1, 1, 1, ha, ha_copy);
                        unit_check_general<T>(1, 1, 1, hb, hb_copy);
                        unit_check_general<U>(1, 1, 1, hc, hc_copy);
                        unit_check_general<T>(1, 1, 1, hs, hs_copy);
                    }
                }
                return;
            }

            if(arg.unit_check)
            {
                near_check_general<T>(1, 1, 1, ha_gold, ha, rel_error);
                near_check_general<T>(1, 1, 1, hb_gold, hb, rel_error);
                near_check_general<U>(1, 1, 1, hc_gold, hc, rel_error);
                near_check_general<T>(1, 1, 1, hs_gold, hs, rel_error);
            }

            if(arg.norm_check)
            {
                error_device = norm_check_general<T>('F', 1, 1, 1, ha_gold, ha);
                error_device += norm_check_general<T>('F', 1, 1, 1, hb_gold, hb);
                error_device += norm_check_general<U>('F', 1, 1, 1, hc_gold, hc);
                error_device += norm_check_general<T>('F', 1, 1, 1, hs_gold, hs);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        host_vector<T> ha = a;
        host_vector<T> hb = b;
        host_vector<U> hc = c;
        host_vector<T> hs = s;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; ++iter)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            ha = a;
            hb = b;
            hc = c;
            hs = s;
            DAPI_DISPATCH(rocblas_rotg_fn, (handle, ha, hb, hc, hs));
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used) / arg.iters;

        rocblas_cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            rocblas_cout << ",norm_error_host_ptr,norm_error_dev_ptr";
        rocblas_cout << std::endl;

        rocblas_cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            rocblas_cout << ',' << error_host << ',' << error_device;
        rocblas_cout << std::endl;
    }
}
