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
#ifndef TESTING_GEMVI_HPP
#define TESTING_GEMVI_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_gemvi_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int m   = 100;
    int n   = 100;
    int nnz = 100;
    int lda = 100;

    static constexpr hipsparseOperation_t opType  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t                  idxBase = HIPSPARSE_INDEX_BASE_ZERO;

    float alpha = 0.6;
    float beta  = 0.1;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto A_managed    = hipsparse_unique_ptr{device_malloc(sizeof(float) * m * n), device_free};
    auto x_managed    = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto xInd_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto y_managed    = hipsparse_unique_ptr{device_malloc(sizeof(float) * m), device_free};

    float* A    = (float*)A_managed.get();
    float* x    = (float*)x_managed.get();
    int*   xInd = (int*)xInd_managed.get();
    float* y    = (float*)y_managed.get();

    // gemvi
    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, 100));

    verify_hipsparse_status_invalid_handle(hipsparseSgemvi(
        nullptr, opType, m, n, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer));
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, nullptr, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, nullptr, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, nullptr, xInd, &beta, y, idxBase, buffer),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, nullptr, &beta, y, idxBase, buffer),
        "Error: xInd is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, xInd, nullptr, y, idxBase, buffer),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, nnz, x, xInd, &beta, nullptr, idxBase, buffer),
        "Error: y is nullptr");

    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, -1, n, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, -1, &alpha, A, lda, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: n is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, -1, nnz, x, xInd, &beta, y, idxBase, buffer),
        "Error: lda is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, -1, x, xInd, &beta, y, idxBase, buffer),
        "Error: nnz is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseSgemvi(
            handle, opType, m, n, &alpha, A, lda, n + 1, x, xInd, &beta, y, idxBase, buffer),
        "Error: nnz is invalid");

    CHECK_HIP_ERROR(hipFree(buffer));
#endif
}

template <typename T>
hipsparseStatus_t testing_gemvi(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  m        = argus.M;
    int                  n        = argus.N;
    int                  nnz      = argus.nnz;
    T                    alpha    = make_DataType<T>(argus.alpha);
    T                    beta     = make_DataType<T>(argus.beta);
    hipsparseOperation_t trans    = argus.transA;
    hipsparseIndexBase_t idxBase  = argus.baseA;
    std::string          filename = argus.filename;

    int lda = m;

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<T>   hA(m * n);
    std::vector<T>   hx_val(nnz);
    std::vector<int> hx_ind(nnz);
    std::vector<T>   hy(m);
    std::vector<T>   hy_gold(m);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, n);
    hipsparseInit<T>(hx_val, 1, nnz);
    hipsparseInit<T>(hy, 1, m);
    hy_gold = hy;

    for(int i = 0; i < m * n; ++i)
    {
        hA[i] = random_generator<T>();
    }

    // Allocate memory on device
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dA_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * m * n), device_free};

    int* dx_ind = (int*)dx_ind_managed.get();
    T*   dx_val = (T*)dx_val_managed.get();
    T*   dy     = (T*)dy_managed.get();
    T*   dA     = (T*)dA_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * m * n, hipMemcpyHostToDevice));

    // gemvi bufferSize
    int   bufferSize;
    void* externalBuffer;

    CHECK_HIPSPARSE_ERROR(hipsparseXgemvi_bufferSize<T>(handle, trans, m, n, nnz, &bufferSize));
    CHECK_HIP_ERROR(hipMalloc(&externalBuffer, bufferSize));

    if(argus.unit_check)
    {
        // gemvi
        CHECK_HIPSPARSE_ERROR(hipsparseXgemvi(handle,
                                              trans,
                                              m,
                                              n,
                                              &alpha,
                                              dA,
                                              lda,
                                              nnz,
                                              dx_val,
                                              dx_ind,
                                              &beta,
                                              dy,
                                              idxBase,
                                              externalBuffer));

        // CPU
        for(int i = 0; i < m; ++i)
        {
            T sum = make_DataType<T>(0);

            for(int j = 0; j < nnz; ++j)
            {
                sum = testing_fma(hx_val[j], hA[(hx_ind[j] - idxBase) * lda + i], sum);
            }

            hy_gold[i] = testing_fma(alpha, sum, testing_mult(beta, hy_gold[i]));
        }

        // Verify results against host
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * m, hipMemcpyDeviceToHost));

        unit_check_near(m, 1, 1, hy_gold.data(), hy.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgemvi(handle,
                                                  trans,
                                                  m,
                                                  n,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  nnz,
                                                  dx_val,
                                                  dx_ind,
                                                  &beta,
                                                  dy,
                                                  idxBase,
                                                  externalBuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgemvi(handle,
                                                  trans,
                                                  m,
                                                  n,
                                                  &alpha,
                                                  dA,
                                                  lda,
                                                  nnz,
                                                  dx_val,
                                                  dx_ind,
                                                  &beta,
                                                  dy,
                                                  idxBase,
                                                  externalBuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = gemvi_gflop_count(m, nnz);
        double gbyte_count
            = gemvi_gbyte_count<T>((trans == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? m : n,
                                   nnz,
                                   beta != make_DataType<T>(0.0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::trans,
                            hipsparse_operation2string(trans),
                            display_key_t::alpha,
                            alpha,
                            display_key_t::beta,
                            beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(hipFree(externalBuffer));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEMVI_HPP
