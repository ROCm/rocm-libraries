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
#ifndef TESTING_ROT_HPP
#define TESTING_ROT_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <typeinfo>

using namespace hipsparse_test;

void testing_rot_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
    int64_t size = 100;
    int64_t nnz  = 100;

    float c_coeff = 3.7;
    float s_coeff = 1.2;

    hipsparseIndexType_t idxType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(float) * size), device_free};

    float* dx_val = (float*)dx_val_managed.get();
    int*   dx_ind = (int*)dx_ind_managed.get();
    float* dy     = (float*)dy_managed.get();

    // Structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    verify_hipsparse_status_success(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, dataType), "Success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, size, dy, dataType), "Success");

    // Rot
    verify_hipsparse_status_invalid_handle(hipsparseRot(nullptr, &c_coeff, &s_coeff, x, y));
    verify_hipsparse_status_invalid_pointer(hipsparseRot(handle, nullptr, &s_coeff, x, y),
                                            "Error: c_coeff is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseRot(handle, &c_coeff, nullptr, x, y),
                                            "Error: s_coeff is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseRot(handle, &c_coeff, &s_coeff, nullptr, y),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseRot(handle, &c_coeff, &s_coeff, x, nullptr),
                                            "Error: y is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "Success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_rot(Arguments argus)
{
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
    I size = argus.N;
    I nnz  = argus.nnz;

    T hc_coeff = make_DataType<T>(argus.alpha);
    T hs_coeff = make_DataType<T>(argus.beta);

    hipsparseIndexBase_t idxBase = argus.baseA;

    // Index and data type
    hipsparseIndexType_t idxType  = getIndexType<I>();
    hipDataType          dataType = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hx_ind(nnz);
    std::vector<T> hx_val_1(nnz);
    std::vector<T> hx_val_2(nnz);
    std::vector<T> hx_val_gold(nnz);
    std::vector<T> hy_1(size);
    std::vector<T> hy_2(size);
    std::vector<T> hy_gold(size);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, size);
    hipsparseInit<T>(hx_val_1, 1, nnz);
    hipsparseInit<T>(hy_1, 1, size);

    hx_val_2    = hx_val_1;
    hx_val_gold = hx_val_1;
    hy_2        = hy_1;
    hy_gold     = hy_1;

    // Allocate memory on device
    auto dx_ind_managed   = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dx_val_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_val_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_1_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * size), device_free};
    auto dy_2_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * size), device_free};
    auto dc_coeff_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto ds_coeff_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dx_ind   = (I*)dx_ind_managed.get();
    T* dx_val_1 = (T*)dx_val_1_managed.get();
    T* dx_val_2 = (T*)dx_val_2_managed.get();
    T* dy_1     = (T*)dy_1_managed.get();
    T* dy_2     = (T*)dy_2_managed.get();
    T* dc_coeff = (T*)dc_coeff_managed.get();
    T* ds_coeff = (T*)ds_coeff_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val_1, hx_val_1.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val_2, hx_val_2.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc_coeff, &hc_coeff, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds_coeff, &hs_coeff, sizeof(T), hipMemcpyHostToDevice));

    // Create structures
    hipsparseSpVecDescr_t x1, x2;
    hipsparseDnVecDescr_t y1, y2;

    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateSpVec(&x1, size, nnz, dx_ind, dx_val_1, idxType, idxBase, dataType));
    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateSpVec(&x2, size, nnz, dx_ind, dx_val_2, idxType, idxBase, dataType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y1, size, dy_1, dataType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y2, size, dy_2, dataType));

    if(argus.unit_check)
    {
        // hipSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseRot(handle, &hc_coeff, &hs_coeff, x1, y1));

        // hipSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseRot(handle, dc_coeff, ds_coeff, x2, y2));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_1.data(), dx_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_2.data(), dx_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * size, hipMemcpyDeviceToHost));

        // CPU
        for(int64_t i = 0; i < nnz; ++i)
        {
            I idx = hx_ind[i] - idxBase;

            T x = hx_val_gold[i];
            T y = hy_gold[idx];

            hx_val_gold[i] = testing_mult(hc_coeff, x) + testing_mult(hs_coeff, y);
            hy_gold[idx]   = testing_mult(hc_coeff, y) - testing_mult(hs_coeff, x);
        }

        // Verify results against host
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_1.data());
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_2.data());
        unit_check_general(1, size, 1, hy_gold.data(), hy_1.data());
        unit_check_general(1, size, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseRot(handle, &hc_coeff, &hs_coeff, x1, y1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseRot(handle, &hc_coeff, &hs_coeff, x1, y1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = roti_gflop_count<I>(nnz);
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

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpVec(x1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpVec(x2));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y1));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y2));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_ROT_HPP
