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
#ifndef TESTING_ROTI_HPP
#define TESTING_ROTI_HPP

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
void testing_roti_bad_arg(const Arguments& argus)
{
    int nnz       = 100;
    int safe_size = 100;
    T   c         = 3.7;
    T   s         = 1.2;

    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*   dx_val = (T*)dx_val_managed.get();
    int* dx_ind = (int*)dx_ind_managed.get();
    T*   dy     = (T*)dy_managed.get();

    verify_hipsparse_status_invalid_value(
        hipsparseXroti(handle, -1, dx_val, dx_ind, dy, &c, &s, idx_base), "Error: nnz is invalid");

    // cusparse returns success when passed nullptrs
#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_pointer(
        hipsparseXroti(handle, nnz, dx_val, (int*)nullptr, dy, &c, &s, idx_base),
        "Error: x_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXroti(handle, nnz, (T*)nullptr, dx_ind, dy, &c, &s, idx_base),
        "Error: x_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXroti(handle, nnz, dx_val, dx_ind, (T*)nullptr, &c, &s, idx_base),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXroti(handle, nnz, dx_val, dx_ind, dy, (T*)nullptr, &s, idx_base),
        "Error: c is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXroti(handle, nnz, dx_val, dx_ind, dy, &c, (T*)nullptr, idx_base),
        "Error: s is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXroti(nullptr, nnz, dx_val, dx_ind, dy, &c, &s, idx_base));
#endif
}

template <typename T>
hipsparseStatus_t testing_roti(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  N        = argus.N;
    int                  nnz      = argus.nnz;
    T                    c        = argus.get_alpha<T>();
    T                    s        = argus.get_beta<T>();
    hipsparseIndexBase_t idx_base = argus.baseA;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<int> hx_ind(nnz);
    std::vector<T>   hx_val_1(nnz);
    std::vector<T>   hx_val_2(nnz);
    std::vector<T>   hx_val_gold(nnz);
    std::vector<T>   hy_1(N);
    std::vector<T>   hy_2(N);
    std::vector<T>   hy_gold(N);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, N);
    hipsparseInit<T>(hx_val_1, 1, nnz);
    hipsparseInit<T>(hy_1, 1, N);

    hx_val_2    = hx_val_1;
    hx_val_gold = hx_val_1;
    hy_2        = hy_1;
    hy_gold     = hy_1;

    // allocate memory on device
    auto dx_ind_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dx_val_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_val_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_1_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dy_2_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dc_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto ds_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dx_ind   = (int*)dx_ind_managed.get();
    T*   dx_val_1 = (T*)dx_val_1_managed.get();
    T*   dx_val_2 = (T*)dx_val_2_managed.get();
    T*   dy_1     = (T*)dy_1_managed.get();
    T*   dy_2     = (T*)dy_2_managed.get();
    T*   dc       = (T*)dc_managed.get();
    T*   ds       = (T*)ds_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val_1, hx_val_1.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dx_val_2, hx_val_2.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, &c, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, &s, sizeof(T), hipMemcpyHostToDevice));

        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXroti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXroti(handle, nnz, dx_val_2, dx_ind, dy_2, dc, ds, idx_base));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_1.data(), dx_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_2.data(), dx_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * N, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * N, hipMemcpyDeviceToHost));

        // CPU
        for(int i = 0; i < nnz; ++i)
        {
            int idx = hx_ind[i] - idx_base;

            T x = hx_val_gold[i];
            T y = hy_gold[idx];

            hx_val_gold[i] = c * x + s * y;
            hy_gold[idx]   = c * y - s * x;
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_1.data());
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_2.data());
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
                hipsparseXroti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXroti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = roti_gflop_count(nnz);
        double gbyte_count = roti_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info(display_key_t::nnz,
                            nnz,
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

#endif // TESTING_ROTI_HPP
