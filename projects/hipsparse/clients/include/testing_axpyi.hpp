/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_AXPYI_HPP
#define TESTING_AXPYI_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_axpyi_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION))
    int nnz       = 100;
    int safe_size = 100;
    T   alpha     = make_DataType<T>(0.6);

    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dxVal_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dxInd_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dy_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*   dxVal = (T*)dxVal_managed.get();
    int* dxInd = (int*)dxInd_managed.get();
    T*   dy    = (T*)dy_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseXaxpyi(handle, nnz, &alpha, dxVal, (int*)nullptr, dy, idx_base),
        "Error: xInd is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXaxpyi(handle, nnz, &alpha, (T*)nullptr, dxInd, dy, idx_base),
        "Error: xVal is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXaxpyi(handle, nnz, &alpha, dxVal, dxInd, (T*)nullptr, idx_base),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXaxpyi(handle, nnz, (T*)nullptr, dxVal, dxInd, dy, idx_base),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXaxpyi((hipsparseHandle_t) nullptr, nnz, &alpha, dxVal, dxInd, dy, idx_base));
#endif
}

template <typename T>
hipsparseStatus_t testing_axpyi(const Arguments& argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  N        = argus.N;
    int                  nnz      = argus.nnz;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    hipsparseIndexBase_t idx_base = argus.baseA;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<int> hxInd(nnz);
    std::vector<T>   hxVal(nnz);
    std::vector<T>   hy_1(N);
    std::vector<T>   hy_2(N);
    std::vector<T>   hy_gold(N);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hxInd.data(), nnz, 1, N);
    hipsparseInit<T>(hxVal, 1, nnz);
    hipsparseInit<T>(hy_1, 1, N);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dxInd_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dxVal_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dy_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dxInd   = (int*)dxInd_managed.get();
    T*   dxVal   = (T*)dxVal_managed.get();
    T*   dy_1    = (T*)dy_1_managed.get();
    T*   dy_2    = (T*)dy_2_managed.get();
    T*   d_alpha = (T*)d_alpha_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dxVal, hxVal.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXaxpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy_1, idx_base));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXaxpyi(handle, nnz, d_alpha, dxVal, dxInd, dy_2, idx_base));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * N, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * N, hipMemcpyDeviceToHost));

        // CPU
        for(int i = 0; i < nnz; ++i)
        {
            hy_gold[hxInd[i] - idx_base]
                = hy_gold[hxInd[i] - idx_base] + testing_mult(h_alpha, hxVal[i]);
        }

        unit_check_general(1, N, 1, hy_gold.data(), hy_1.data());
        unit_check_general(1, N, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXaxpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy_1, idx_base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXaxpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy_1, idx_base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = axpyi_gflop_count(nnz);
        double gbyte_count = axpby_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info(display_key_t::size,
                            N,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_AXPYI_HPP
