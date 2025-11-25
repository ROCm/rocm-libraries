/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef TESTING_GTSV2_NOPIVOT_STRIDED_BATCH_HPP
#define TESTING_GTSV2_NOPIVOT_STRIDED_BATCH_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_gtsv2_strided_batch_bad_arg(void)
{
    // Dont do bad argument checking for cuda
#if(!defined(CUDART_VERSION))
    int safe_size    = 100;
    int m            = 10;
    int batch_count  = 10;
    int batch_stride = m;

    // Create handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto ddl_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dd_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto ddu_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    T*    ddl  = (T*)ddl_managed.get();
    T*    dd   = (T*)dd_managed.get();
    T*    ddu  = (T*)ddu_managed.get();
    T*    dx   = (T*)dx_managed.get();
    void* dbuf = (void*)dbuf_managed.get();

    size_t bsize;

    // gtsv2StridedBatch_bufferSize
    verify_hipsparse_status_invalid_handle(hipsparseXgtsv2StridedBatch_bufferSizeExt(
        nullptr, m, ddl, dd, ddu, dx, batch_count, batch_stride, &bsize));
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2StridedBatch_bufferSizeExt(
            handle, -1, ddl, dd, ddu, dx, batch_count, batch_stride, &bsize),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2StridedBatch_bufferSizeExt(
            handle, m, ddl, dd, ddu, dx, -1, batch_stride, &bsize),
        "Error: batch_count is invalid");
    verify_hipsparse_status_invalid_value(hipsparseXgtsv2StridedBatch_bufferSizeExt(
                                              handle, m, ddl, dd, ddu, dx, batch_count, -1, &bsize),
                                          "Error: batch_stride is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2StridedBatch_bufferSizeExt(
            handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, nullptr),
        "Error: bsize is nullptr");

    // gtsv2StridedBatch
    verify_hipsparse_status_invalid_handle(
        hipsparseXgtsv2StridedBatch(nullptr, m, ddl, dd, ddu, dx, batch_count, batch_stride, dbuf));
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2StridedBatch(handle, -1, ddl, dd, ddu, dx, batch_count, batch_stride, dbuf),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2StridedBatch(handle, m, ddl, dd, ddu, dx, -1, batch_stride, dbuf),
        "Error: batch_count is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgtsv2StridedBatch(handle, m, ddl, dd, ddu, dx, batch_count, -1, dbuf),
        "Error: batch_stride is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2StridedBatch(
            handle, m, (const T*)nullptr, dd, ddu, dx, batch_count, batch_stride, dbuf),
        "Error: ddl is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2StridedBatch(
            handle, m, ddl, (const T*)nullptr, ddu, dx, batch_count, batch_stride, dbuf),
        "Error: dd is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2StridedBatch(
            handle, m, ddl, dd, (const T*)nullptr, dx, batch_count, batch_stride, dbuf),
        "Error: ddu is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgtsv2StridedBatch(
            handle, m, ddl, dd, ddu, (T*)nullptr, batch_count, batch_stride, dbuf),
        "Error: dx is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_gtsv2_strided_batch(Arguments argus)
{
    int m           = argus.M;
    int batch_count = argus.batch_count;

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    int batch_stride = 2 * m;

    // Host structures
    std::vector<T> hdl(batch_stride * batch_count, make_DataType<T>(1));
    std::vector<T> hd(batch_stride * batch_count, make_DataType<T>(2));
    std::vector<T> hdu(batch_stride * batch_count, make_DataType<T>(1));
    std::vector<T> hx(batch_stride * batch_count, make_DataType<T>(3));

    for(int i = 0; i < batch_count; i++)
    {
        hdl[batch_stride * i + 0]     = make_DataType<T>(0);
        hdu[batch_stride * i + m - 1] = make_DataType<T>(0);
    }

    std::vector<T> hx_original = hx;

    // allocate memory on device
    auto ddl_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_stride * batch_count), device_free};
    auto dd_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_stride * batch_count), device_free};
    auto ddu_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_stride * batch_count), device_free};
    auto dx_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * batch_stride * batch_count), device_free};

    T* ddl = (T*)ddl_managed.get();
    T* dd  = (T*)dd_managed.get();
    T* ddu = (T*)ddu_managed.get();
    T* dx  = (T*)dx_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(ddl, hdl.data(), sizeof(T) * batch_stride * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dd, hd.data(), sizeof(T) * batch_stride * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(ddu, hdu.data(), sizeof(T) * batch_stride * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dx, hx.data(), sizeof(T) * batch_stride * batch_count, hipMemcpyHostToDevice));

    // Query SparseToDense buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXgtsv2StridedBatch_bufferSizeExt(
        handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXgtsv2StridedBatch(
            handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(
            hx.data(), dx, sizeof(T) * batch_stride * batch_count, hipMemcpyDeviceToHost));

        // Check
        std::vector<T> hresult(batch_stride * batch_count, make_DataType<T>(3));
        for(int j = 0; j < batch_count; j++)
        {
            hresult[batch_stride * j]
                = testing_mult(hd[batch_stride * j + 0], hx[batch_stride * j])
                  + testing_mult(hdu[batch_stride * j + 0], hx[batch_stride * j + 1]);
            hresult[batch_stride * j + m - 1]
                = testing_mult(hdl[batch_stride * j + m - 1], hx[batch_stride * j + m - 2])
                  + testing_mult(hd[batch_stride * j + m - 1], hx[batch_stride * j + m - 1]);
            for(int i = 1; i < m - 1; i++)
            {
                hresult[batch_stride * j + i]
                    = testing_mult(hdl[batch_stride * j + i], hx[batch_stride * j + i - 1])
                      + testing_mult(hd[batch_stride * j + i], hx[batch_stride * j + i])
                      + testing_mult(hdu[batch_stride * j + i], hx[batch_stride * j + i + 1]);
            }
        }

        unit_check_near<T>(1, batch_stride * batch_count, 1, hx_original.data(), hresult.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgtsv2StridedBatch(
                handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgtsv2StridedBatch(
                handle, m, ddl, dd, ddu, dx, batch_count, batch_stride, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = gtsv_strided_batch_gbyte_count<T>(m, batch_count);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::batch_count,
                            batch_count,
                            display_key_t::batch_stride,
                            batch_stride,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GTSV2_NOPIVOT_STRIDED_BATCH_HPP
