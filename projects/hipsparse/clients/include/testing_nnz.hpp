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
#ifndef TESTING_NNZ_HPP
#define TESTING_NNZ_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_nnz_bad_arg(void)
{
    static constexpr size_t               safe_size = 100;
    static constexpr int                  M         = 10;
    static constexpr int                  N         = 10;
    static constexpr int                  lda       = M;
    static constexpr hipsparseDirection_t dirA      = HIPSPARSE_DIRECTION_ROW;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descrA = unique_ptr_descr->descr;

    auto A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto nnzPerRowColumn_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto nnzTotalDevHostPtr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};

    T*   d_A                  = (T*)A_managed.get();
    int* d_nnzPerRowColumn    = (int*)nnzPerRowColumn_managed.get();
    int* d_nnzTotalDevHostPtr = (int*)nnzTotalDevHostPtr_managed.get();

#if(!defined(CUDART_VERSION))
    verify_hipsparse_status_invalid_handle(hipsparseXnnz(
        nullptr, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, d_nnzTotalDevHostPtr));
    verify_hipsparse_status_invalid_pointer(hipsparseXnnz(handle,
                                                          dirA,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          (const T*)d_A,
                                                          lda,
                                                          d_nnzPerRowColumn,
                                                          d_nnzTotalDevHostPtr),
                                            "Error: descrA as invalid pointer must be detected.");
    verify_hipsparse_status_invalid_pointer(hipsparseXnnz(handle,
                                                          dirA,
                                                          M,
                                                          N,
                                                          descrA,
                                                          (const T*)nullptr,
                                                          lda,
                                                          d_nnzPerRowColumn,
                                                          d_nnzTotalDevHostPtr),
                                            "Error: A as invalid pointer must be detected.");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXnnz(
            handle, dirA, M, N, descrA, (const T*)d_A, lda, nullptr, d_nnzTotalDevHostPtr),
        "Error: nnzPerRowColumn as invalid pointer must be detected.");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXnnz(handle, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, nullptr),
        "Error: nnzTotalDevHostPtr as invalid pointer must be detected.");
#endif

    // Testing invalid direction
    try
    {
        hipsparseXnnz(handle,
                      (hipsparseDirection_t)77,
                      -1,
                      -1,
                      descrA,
                      (const T*)nullptr,
                      -1,
                      nullptr,
                      nullptr);

        // An exception should be thrown.
        verify_hipsparse_status_internal_error(
            HIPSPARSE_STATUS_SUCCESS,
            "Error: an exception must be thrown from the conversion of the hipsparseDirection_t.");
    }
    catch(...)
    {
    }

    verify_hipsparse_status_invalid_size(hipsparseXnnz(handle,
                                                       dirA,
                                                       -1,
                                                       N,
                                                       descrA,
                                                       (const T*)d_A,
                                                       lda,
                                                       d_nnzPerRowColumn,
                                                       d_nnzTotalDevHostPtr),
                                         "Error: M < 0 must be detected.");
    verify_hipsparse_status_invalid_size(hipsparseXnnz(handle,
                                                       dirA,
                                                       M,
                                                       -1,
                                                       descrA,
                                                       (const T*)d_A,
                                                       lda,
                                                       d_nnzPerRowColumn,
                                                       d_nnzTotalDevHostPtr),
                                         "Error: N < 0 must be detected.");
    verify_hipsparse_status_invalid_size(hipsparseXnnz(handle,
                                                       dirA,
                                                       M,
                                                       N,
                                                       descrA,
                                                       (const T*)d_A,
                                                       M - 1,
                                                       d_nnzPerRowColumn,
                                                       d_nnzTotalDevHostPtr),
                                         "Error: lda < M must be detected.");
}

template <typename T>
hipsparseStatus_t testing_nnz(Arguments argus)
{
    int                  M    = argus.M;
    int                  N    = argus.N;
    int                  lda  = argus.lda;
    hipsparseDirection_t dirA = argus.dirA;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descrA = unique_ptr_descr->descr;

    if(M == 0 || N == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    // Create the dense matrix.
    int  MN        = (dirA == HIPSPARSE_DIRECTION_ROW) ? M : N;
    auto A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * lda * N), device_free};
    auto nnzPerRowColumn_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * MN), device_free};
    auto nnzTotalDevHostPtr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};

    T*   d_A                  = (T*)A_managed.get();
    int* d_nnzPerRowColumn    = (int*)nnzPerRowColumn_managed.get();
    int* d_nnzTotalDevHostPtr = (int*)nnzTotalDevHostPtr_managed.get();

    std::vector<T>   h_A(lda * N);
    std::vector<int> h_nnzPerRowColumn(MN);
    std::vector<int> hd_nnzPerRowColumn(MN);
    std::vector<int> h_nnzTotalDevHostPtr(1);
    std::vector<int> hd_nnzTotalDevHostPtr(1);

    // Initialize the entire allocated memory.
    for(int i = 0; i < lda; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_A[j * lda + i] = make_DataType<T>(-1);
        }
    }

    // Initialize a random dense matrix.
    srand(0);
    gen_dense_random_sparsity_pattern(M, N, h_A.data(), lda, HIPSPARSE_ORDER_COL, 0.2);

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Compute the reference host first.
        host_nnz<T>(dirA,
                    M,
                    N,
                    descrA,
                    h_A.data(),
                    lda,
                    h_nnzPerRowColumn.data(),
                    h_nnzTotalDevHostPtr.data());

        // Pointer mode device for nnz and call.
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz(handle,
                                            dirA,
                                            M,
                                            N,
                                            descrA,
                                            (const T*)d_A,
                                            lda,
                                            d_nnzPerRowColumn,
                                            d_nnzTotalDevHostPtr));

        // Transfer.
        CHECK_HIP_ERROR(hipMemcpy(
            hd_nnzPerRowColumn.data(), d_nnzPerRowColumn, sizeof(int) * MN, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzTotalDevHostPtr.data(),
                                  d_nnzTotalDevHostPtr,
                                  sizeof(int) * 1,
                                  hipMemcpyDeviceToHost));

        // Check results.
        unit_check_general<int>(1, MN, 1, hd_nnzPerRowColumn.data(), h_nnzPerRowColumn.data());
        unit_check_general<int>(1, 1, 1, hd_nnzTotalDevHostPtr.data(), h_nnzTotalDevHostPtr.data());

        // Pointer mode host for nnz and call.
        int dh_nnz;
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz(
            handle, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, &dh_nnz));

        // Transfer.
        CHECK_HIP_ERROR(hipMemcpy(
            hd_nnzPerRowColumn.data(), d_nnzPerRowColumn, sizeof(int) * MN, hipMemcpyDeviceToHost));

        // Check results.
        unit_check_general<int>(1, MN, 1, hd_nnzPerRowColumn.data(), h_nnzPerRowColumn.data());
        unit_check_general<int>(1, 1, 1, &dh_nnz, h_nnzTotalDevHostPtr.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm-up
        int h_nnz;
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXnnz(
                handle, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, &h_nnz));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXnnz(
                handle, dirA, M, N, descrA, (const T*)d_A, lda, d_nnzPerRowColumn, &h_nnz));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = nnz_gbyte_count<T>(M, N, dirA);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::LD,
                            lda,
                            display_key_t::nnz,
                            h_nnz,
                            display_key_t::direction,
                            hipsparse_direction2string(dirA),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_NNZ_HPP
