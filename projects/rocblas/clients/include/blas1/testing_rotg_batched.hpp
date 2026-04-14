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
void testing_rotg_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_rotg_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_rotg_batched<T, U, true>
                                                              : rocblas_rotg_batched<T, U, false>;
    auto rocblas_rotg_batched_fn_64 = arg.api & c_API_FORTRAN
                                          ? rocblas_rotg_batched_64<T, U, true>
                                          : rocblas_rotg_batched_64<T, U, false>;

    int64_t             batch_count = 5;
    static const size_t safe_size   = 1;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    device_batch_vector<T> da(1, 1, batch_count);
    device_batch_vector<T> db(1, 1, batch_count);
    device_batch_vector<U> dc(1, 1, batch_count);
    device_batch_vector<T> ds(1, 1, batch_count);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_rotg_batched_fn,
                (nullptr,
                 da.ptr_on_device(),
                 db.ptr_on_device(),
                 dc.ptr_on_device(),
                 ds.ptr_on_device(),
                 batch_count));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rotg_batched_fn,
        (handle, nullptr, db.ptr_on_device(), dc.ptr_on_device(), ds.ptr_on_device(), batch_count));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rotg_batched_fn,
        (handle, da.ptr_on_device(), nullptr, dc.ptr_on_device(), ds.ptr_on_device(), batch_count));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rotg_batched_fn,
        (handle, da.ptr_on_device(), db.ptr_on_device(), nullptr, ds.ptr_on_device(), batch_count));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rotg_batched_fn,
        (handle, da.ptr_on_device(), db.ptr_on_device(), dc.ptr_on_device(), nullptr, batch_count));
}

template <typename T, typename U = T>
void testing_rotg_batched(const Arguments& arg)
{
    auto rocblas_rotg_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_rotg_batched<T, U, true>
                                                              : rocblas_rotg_batched<T, U, false>;
    auto rocblas_rotg_batched_fn_64 = arg.api & c_API_FORTRAN
                                          ? rocblas_rotg_batched_64<T, U, true>
                                          : rocblas_rotg_batched_64<T, U, false>;

    int64_t batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    double gpu_time_used, cpu_time_used;
    double norm_error_host = 0.0, norm_error_device = 0.0;

    const U rel_error = std::numeric_limits<U>::epsilon() * 100;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_rotg_batched_fn,
                   (handle, nullptr, nullptr, nullptr, nullptr, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg ha), `d` is in GPU (device) memory (eg da).
    // Allocate host memory
    host_batch_vector<T> ha(1, 1, batch_count);
    host_batch_vector<T> hb(1, 1, batch_count);
    host_batch_vector<U> hc(1, 1, batch_count);
    host_batch_vector<T> hs(1, 1, batch_count);

    bool enable_near_check_general = true;

    host_batch_vector<T> ha_gold(1, 1, batch_count);
    host_batch_vector<T> hb_gold(1, 1, batch_count);
    host_batch_vector<U> hc_gold(1, 1, batch_count);
    host_batch_vector<T> hs_gold(1, 1, batch_count);

    // Initialize data on host memory
    rocblas_init_vector(ha, arg, rocblas_client_never_set_nan, true);
    rocblas_init_vector(hb, arg, rocblas_client_never_set_nan, false);
    rocblas_init_vector(hc, arg, rocblas_client_never_set_nan, false);
    rocblas_init_vector(hs, arg, rocblas_client_never_set_nan, false);

    ha[0][0] = arg.get_alpha<T>(); // reuse alpha in place of a to keep number of arguments small
    hb[0][0] = arg.get_beta<T>(); // reuse beta  in place of b to keep number of arguments small
    hc[0][0] = U(0);
    hs[0][0] = T(0);

    ha_gold.copy_from(ha);
    hb_gold.copy_from(hb);
    hc_gold.copy_from(hc);
    hs_gold.copy_from(hs);

    cpu_time_used = get_time_us_no_sync();
    for(size_t b = 0; b < batch_count; b++)
    {
        ref_rotg<T, U>(ha_gold[b], hb_gold[b], hc_gold[b], hs_gold[b]);
    }
    cpu_time_used = get_time_us_no_sync() - cpu_time_used;

    // Test rocblas_pointer_mode_host
    if(arg.pointer_mode_host)
    {

        host_batch_vector<T> ra(1, 1, batch_count);
        host_batch_vector<T> rb(1, 1, batch_count);
        host_batch_vector<U> rc(1, 1, batch_count);
        host_batch_vector<T> rs(1, 1, batch_count);

        ra.copy_from(ha);
        rb.copy_from(hb);
        rc.copy_from(hc);
        rs.copy_from(hs);

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        handle.pre_test(arg);
        DAPI_CHECK(rocblas_rotg_batched_fn, (handle, ra, rb, rc, rs, batch_count));
        handle.post_test(arg);

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, 1, ra, ha_gold, rel_error, batch_count);
            near_check_general<T>(1, 1, 1, rb, hb_gold, rel_error, batch_count);
            near_check_general<U>(1, 1, 1, rc, hc_gold, rel_error, batch_count);
            near_check_general<T>(1, 1, 1, rs, hs_gold, rel_error, batch_count);
        }

        if(arg.norm_check)
        {
            norm_error_host = norm_check_general<T>('F', 1, 1, 1, ra, ha_gold, batch_count);
            norm_error_host += norm_check_general<T>('F', 1, 1, 1, rb, hb_gold, batch_count);
            norm_error_host += norm_check_general<U>('F', 1, 1, 1, rc, hc_gold, batch_count);
            norm_error_host += norm_check_general<T>('F', 1, 1, 1, rs, hs_gold, batch_count);
        }
    }

    // Test rocblas_pointer_mode_device
    if(arg.pointer_mode_device)
    {
        // Allocate device memory
        device_batch_vector<T> da(1, 1, batch_count);
        device_batch_vector<T> db(1, 1, batch_count);
        device_batch_vector<U> dc(1, 1, batch_count);
        device_batch_vector<T> ds(1, 1, batch_count);

        // Check device memory allocation
        CHECK_DEVICE_ALLOCATION(da.memcheck());
        CHECK_DEVICE_ALLOCATION(db.memcheck());
        CHECK_DEVICE_ALLOCATION(dc.memcheck());
        CHECK_DEVICE_ALLOCATION(ds.memcheck());

        // Transfer from CPU to GPU
        CHECK_HIP_ERROR(da.transfer_from(ha));
        CHECK_HIP_ERROR(db.transfer_from(hb));
        CHECK_HIP_ERROR(dc.transfer_from(hc));
        CHECK_HIP_ERROR(ds.transfer_from(hs));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        handle.pre_test(arg);
        DAPI_CHECK(rocblas_rotg_batched_fn,
                   (handle,
                    da.ptr_on_device(),
                    db.ptr_on_device(),
                    dc.ptr_on_device(),
                    ds.ptr_on_device(),
                    batch_count));
        handle.post_test(arg);

        host_batch_vector<T> ra(1, 1, batch_count);
        host_batch_vector<T> rb(1, 1, batch_count);
        host_batch_vector<U> rc(1, 1, batch_count);
        host_batch_vector<T> rs(1, 1, batch_count);

        // Transfer from GPU to CPU
        CHECK_HIP_ERROR(ra.transfer_from(da));
        CHECK_HIP_ERROR(rb.transfer_from(db));
        CHECK_HIP_ERROR(rc.transfer_from(dc));
        CHECK_HIP_ERROR(rs.transfer_from(ds));

        if(arg.repeatability_check)
        {
            host_batch_vector<T> ra_copy(1, 1, batch_count);
            host_batch_vector<T> rb_copy(1, 1, batch_count);
            host_batch_vector<U> rc_copy(1, 1, batch_count);
            host_batch_vector<T> rs_copy(1, 1, batch_count);

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
                device_batch_vector<T> da_copy(1, 1, batch_count);
                device_batch_vector<T> db_copy(1, 1, batch_count);
                device_batch_vector<U> dc_copy(1, 1, batch_count);
                device_batch_vector<T> ds_copy(1, 1, batch_count);

                // Check device memory allocation
                CHECK_DEVICE_ALLOCATION(da_copy.memcheck());
                CHECK_DEVICE_ALLOCATION(db_copy.memcheck());
                CHECK_DEVICE_ALLOCATION(dc_copy.memcheck());
                CHECK_DEVICE_ALLOCATION(ds_copy.memcheck());

                CHECK_ROCBLAS_ERROR(
                    rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                for(int runs = 0; runs < arg.iters; runs++)
                {
                    CHECK_HIP_ERROR(da_copy.transfer_from(ha));
                    CHECK_HIP_ERROR(db_copy.transfer_from(hb));
                    CHECK_HIP_ERROR(dc_copy.transfer_from(hc));
                    CHECK_HIP_ERROR(ds_copy.transfer_from(hs));

                    DAPI_CHECK(rocblas_rotg_batched_fn,
                               (handle_copy,
                                da_copy.ptr_on_device(),
                                db_copy.ptr_on_device(),
                                dc_copy.ptr_on_device(),
                                ds_copy.ptr_on_device(),
                                batch_count));

                    CHECK_HIP_ERROR(ra_copy.transfer_from(da_copy));
                    CHECK_HIP_ERROR(rb_copy.transfer_from(db_copy));
                    CHECK_HIP_ERROR(rc_copy.transfer_from(dc_copy));
                    CHECK_HIP_ERROR(rs_copy.transfer_from(ds_copy));

                    unit_check_general<T>(1, 1, 1, ra, ra_copy, batch_count);
                    unit_check_general<T>(1, 1, 1, rb, rb_copy, batch_count);
                    unit_check_general<U>(1, 1, 1, rc, rc_copy, batch_count);
                    unit_check_general<T>(1, 1, 1, rs, rs_copy, batch_count);
                }
            }
            return;
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, 1, ra, ha_gold, batch_count, rel_error);
            near_check_general<T>(1, 1, 1, rb, hb_gold, batch_count, rel_error);
            near_check_general<U>(1, 1, 1, rc, hc_gold, batch_count, rel_error);
            near_check_general<T>(1, 1, 1, rs, hs_gold, batch_count, rel_error);
        }

        if(arg.norm_check)
        {
            norm_error_device = norm_check_general<T>('F', 1, 1, 1, ra, ha_gold, batch_count);
            norm_error_device += norm_check_general<T>('F', 1, 1, 1, rb, hb_gold, batch_count);
            norm_error_device += norm_check_general<U>('F', 1, 1, 1, rc, hc_gold, batch_count);
            norm_error_device += norm_check_general<T>('F', 1, 1, 1, rs, hs_gold, batch_count);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        // Device mode will be much quicker
        // (TODO: or is there another reason we are typically using host_mode for timing?)
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        device_batch_vector<T> da(1, 1, batch_count);
        device_batch_vector<T> db(1, 1, batch_count);
        device_batch_vector<U> dc(1, 1, batch_count);
        device_batch_vector<T> ds(1, 1, batch_count);
        CHECK_HIP_ERROR(da.transfer_from(ha));
        CHECK_HIP_ERROR(db.transfer_from(hb));
        CHECK_HIP_ERROR(dc.transfer_from(hc));
        CHECK_HIP_ERROR(ds.transfer_from(hs));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_rotg_batched_fn,
                          (handle,
                           da.ptr_on_device(),
                           db.ptr_on_device(),
                           dc.ptr_on_device(),
                           ds.ptr_on_device(),
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_batch_count>{}.log_args<T>(rocblas_cout,
                                                   arg,
                                                   gpu_time_used,
                                                   ArgumentLogging::NA_value,
                                                   ArgumentLogging::NA_value,
                                                   cpu_time_used,
                                                   norm_error_host,
                                                   norm_error_device);
    }
}
