/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_SPVV_HPP
#define TESTING_SPVV_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_graph.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <typeinfo>

using namespace hipsparse_test;

template <typename I, typename X, typename Y, typename T>
void testing_spvv_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    int64_t size = 100;
    int64_t nnz  = 100;

    T result;

    hipsparseOperation_t opType      = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexType_t idxType     = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase     = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          xType       = getDataType<X>();
    hipDataType          yType       = getDataType<Y>();
    hipDataType          computeType = getDataType<T>();

    hipsparseLocalHandle_t handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(X) * nnz), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(Y) * size), device_free};

    X*   dx_val = (X*)dx_val_managed.get();
    int* dx_ind = (int*)dx_ind_managed.get();
    Y*   dy     = (Y*)dy_managed.get();

    // Structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    verify_hipsparse_status_success(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, xType), "Success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, size, dy, yType), "Success");

    // SpVV bufferSize
    size_t bufferSize;
    verify_hipsparse_status_invalid_handle(
        hipsparseSpVV_bufferSize(nullptr, opType, x, y, &result, computeType, &bufferSize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, nullptr, y, &result, computeType, &bufferSize),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, x, nullptr, &result, computeType, &bufferSize),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, x, y, nullptr, computeType, &bufferSize),
        "Error: result is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV_bufferSize(handle, opType, x, y, &result, computeType, nullptr),
        "Error: bufferSize is nullptr");

    // SpVV
    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, 100));

    verify_hipsparse_status_invalid_handle(
        hipsparseSpVV(nullptr, opType, x, y, &result, computeType, buffer));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, nullptr, y, &result, computeType, buffer),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, x, nullptr, &result, computeType, buffer),
        "Error: y is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, x, y, nullptr, computeType, buffer),
        "Error: result is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVV(handle, opType, x, y, &result, computeType, nullptr),
        "Error: buffer is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "Success");

    CHECK_HIP_ERROR(hipFree(buffer));
#endif
}

template <typename I, typename X, typename Y, typename T>
void testing_spvv(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    I                    size    = argus.N;
    I                    nnz     = argus.nnz;
    hipsparseOperation_t trans   = argus.transA;
    hipsparseIndexBase_t idxBase = argus.baseA;

    // Index, vector element and compute data types
    hipsparseIndexType_t idxType     = getIndexType<I>();
    hipDataType          xType       = getDataType<X>();
    hipDataType          yType       = getDataType<Y>();
    hipDataType          computeType = getDataType<T>();

    // hipSPARSE handle
    hipsparseLocalHandle_t handle(argus);

    hipStream_t stream;
    CHECK_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Host structures
    std::vector<I> hx_ind(nnz);
    std::vector<X> hx_val(nnz);
    std::vector<Y> hy(size);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, idxBase, size + idxBase);
    hipsparseInit<X>(hx_val, 1, nnz);
    hipsparseInit<Y>(hy, 1, size);

    // Allocate memory on device
    auto dx_ind_managed  = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dx_val_managed  = hipsparse_unique_ptr{device_malloc(sizeof(X) * nnz), device_free};
    auto dy_managed      = hipsparse_unique_ptr{device_malloc(sizeof(Y) * size), device_free};
    auto dresult_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    I* dx_ind  = (I*)dx_ind_managed.get();
    X* dx_val  = (X*)dx_val_managed.get();
    Y* dy      = (Y*)dy_managed.get();
    T* dresult = (T*)dresult_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(X) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(Y) * size, hipMemcpyHostToDevice));

    // Create structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, xType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, size, dy, yType));

    T hresult;
    T hresult_gold;
    T hresult_copied_from_device;

    // SpVV_bufferSize
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(
        testing::hipsparseSpVV_bufferSize(handle, trans, x, y, &hresult, computeType, &bufferSize));

    void* externalBuffer;
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer, bufferSize));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            testing::hipsparseSpVV(handle, trans, x, y, &hresult, computeType, externalBuffer));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            testing::hipsparseSpVV(handle, trans, x, y, dresult, computeType, externalBuffer));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(&hresult_copied_from_device, dresult, sizeof(T), hipMemcpyDeviceToHost));

        // CPU solution
        host_spvv<I, X, Y, T>(
            nnz, hx_val.data(), hx_ind.data(), hy.data(), &hresult_gold, trans, idxBase);

        // Verify results against host
        unit_check_general(1, 1, 1, &hresult_gold, &hresult);
        unit_check_general(1, 1, 1, &hresult_gold, &hresult_copied_from_device);
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
                testing::hipsparseSpVV(handle, trans, x, y, &hresult, computeType, externalBuffer));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                testing::hipsparseSpVV(handle, trans, x, y, &hresult, computeType, externalBuffer));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = doti_gflop_count(nnz);
        double gbyte_count = doti_gbyte_count<X, Y>(nnz);

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

    CHECK_HIP_ERROR(hipFree(externalBuffer));

    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));

#endif
}

#endif // TESTING_SPVV_HPP
