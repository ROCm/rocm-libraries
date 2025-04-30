/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_GPSV_INTERLEAVED_BATCH_HPP
#define TESTING_GPSV_INTERLEAVED_BATCH_HPP

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
void testing_gpsv_interleaved_batch_bad_arg(void)
{
    // Dont do bad argument checking for cuda
#if(!defined(CUDART_VERSION))
    int safe_size   = 100;
    int algo        = 0;
    int m           = 10;
    int batch_count = 10;

    // Create handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dds_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto ddl_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dd_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto ddu_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto ddw_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    T*    dds  = (T*)dds_managed.get();
    T*    ddl  = (T*)ddl_managed.get();
    T*    dd   = (T*)dd_managed.get();
    T*    ddu  = (T*)ddu_managed.get();
    T*    ddw  = (T*)ddw_managed.get();
    T*    dx   = (T*)dx_managed.get();
    void* dbuf = (void*)dbuf_managed.get();

    size_t bsize;

    // gpsvInterleavedBatch_bufferSizeExt
    verify_hipsparse_status_invalid_handle(hipsparseXgpsvInterleavedBatch_bufferSizeExt(
        nullptr, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, &bsize));
    verify_hipsparse_status_invalid_value(
        hipsparseXgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, -1, dds, ddl, dd, ddu, ddw, dx, batch_count, &bsize),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, dds, ddl, dd, ddu, ddw, dx, -1, &bsize),
        "Error: batch_count is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch_bufferSizeExt(
            handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, nullptr),
        "Error: bsize is nullptr");

    // gpsvInterleavedBatch
    verify_hipsparse_status_invalid_handle(hipsparseXgpsvInterleavedBatch(
        nullptr, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, dbuf));
    verify_hipsparse_status_invalid_value(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, -1, dds, ddl, dd, ddu, ddw, dx, batch_count, dbuf),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_value(
        hipsparseXgpsvInterleavedBatch(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, -1, dbuf),
        "Error: batch_count is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, (T*)nullptr, ddl, dd, ddu, ddw, dx, batch_count, dbuf),
        "Error: dds is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, (T*)nullptr, dd, ddu, ddw, dx, batch_count, dbuf),
        "Error: ddl is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, ddl, (T*)nullptr, ddu, ddw, dx, batch_count, dbuf),
        "Error: dd is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, ddl, dd, (T*)nullptr, ddw, dx, batch_count, dbuf),
        "Error: ddu is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, ddl, dd, ddu, (T*)nullptr, dx, batch_count, dbuf),
        "Error: ddw is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, ddl, dd, ddu, ddw, (T*)nullptr, batch_count, dbuf),
        "Error: dx is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, nullptr),
        "Error: bsize is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_gpsv_interleaved_batch(Arguments argus)
{
    int m           = argus.M;
    int batch_count = argus.batch_count;
    int algo        = argus.gpsv_alg;

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<T> hds(m * batch_count, make_DataType<T>(1));
    std::vector<T> hdl(m * batch_count, make_DataType<T>(1));
    std::vector<T> hd(m * batch_count, make_DataType<T>(2));
    std::vector<T> hdu(m * batch_count, make_DataType<T>(1));
    std::vector<T> hdw(m * batch_count, make_DataType<T>(1));
    std::vector<T> hx(m * batch_count, make_DataType<T>(3));

    for(int i = 0; i < batch_count; i++)
    {
        hds[i]                         = make_DataType<T>(0);
        hds[batch_count + i]           = make_DataType<T>(0);
        hdl[i]                         = make_DataType<T>(0);
        hdu[batch_count * (m - 1) + i] = make_DataType<T>(0);
        hdw[batch_count * (m - 1) + i] = make_DataType<T>(0);
        hdw[batch_count * (m - 2) + i] = make_DataType<T>(0);
    }

    std::vector<T> hx_original = hx;

    // allocate memory on device
    auto dds_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * batch_count), device_free};
    auto ddl_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * batch_count), device_free};
    auto dd_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * batch_count), device_free};
    auto ddu_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * batch_count), device_free};
    auto ddw_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * batch_count), device_free};
    auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * batch_count), device_free};

    T* dds = (T*)dds_managed.get();
    T* ddl = (T*)ddl_managed.get();
    T* dd  = (T*)dd_managed.get();
    T* ddu = (T*)ddu_managed.get();
    T* ddw = (T*)ddw_managed.get();
    T* dx  = (T*)dx_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dds, hds.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ddl, hdl.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dd, hd.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ddu, hdu.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ddw, hdw.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));

    // Query SparseToDense buffer
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXgpsvInterleavedBatch_bufferSizeExt(
        handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXgpsvInterleavedBatch(
            handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, buffer));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx.data(), dx, sizeof(T) * m * batch_count, hipMemcpyDeviceToHost));

        // Check
        std::vector<T> hresult(m * batch_count, make_DataType<T>(3));

        for(int b = 0; b < batch_count; b++)
        {
            for(int i = 0; i < m; ++i)
            {
                T sum = testing_mult(hd[batch_count * i + b], hx[batch_count * i + b]);

                sum = sum
                      + ((i - 2 >= 0)
                             ? testing_mult(hds[batch_count * i + b], hx[batch_count * (i - 2) + b])
                             : make_DataType<T>(0));
                sum = sum
                      + ((i - 1 >= 0)
                             ? testing_mult(hdl[batch_count * i + b], hx[batch_count * (i - 1) + b])
                             : make_DataType<T>(0));
                sum = sum
                      + ((i + 1 < m)
                             ? testing_mult(hdu[batch_count * i + b], hx[batch_count * (i + 1) + b])
                             : make_DataType<T>(0));
                sum = sum
                      + ((i + 2 < m)
                             ? testing_mult(hdw[batch_count * i + b], hx[batch_count * (i + 2) + b])
                             : make_DataType<T>(0));

                hresult[batch_count * i + b] = sum;
            }
        }

        unit_check_near<T>(1, m * batch_count, 1, hx_original.data(), hresult.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgpsvInterleavedBatch(
                handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgpsvInterleavedBatch(
                handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batch_count, buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = gpsv_interleaved_batch_gbyte_count<T>(m, batch_count);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::batch_count,
                            batch_count,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GPSV_INTERLEAVED_BATCH_HPP
