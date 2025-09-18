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
#ifndef TESTING_CSR2CSR_COMPRESS_HPP
#define TESTING_CSR2CSR_COMPRESS_HPP

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
#include <iostream>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csr2csr_compress_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  m            = 1;
    int                  n            = 1;
    int                  nnz_A        = 1;
    int                  safe_size    = 1;
    T                    tol          = make_DataType<T>(0);
    hipsparseIndexBase_t csr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);

    auto csr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto csr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto csr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto nnz_per_row_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto nnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* csr_row_ptr_A = (int*)csr_row_ptr_A_managed.get();
    int* csr_col_ind_A = (int*)csr_col_ind_A_managed.get();
    T*   csr_val_A     = (T*)csr_val_A_managed.get();
    int* csr_row_ptr_C = (int*)csr_row_ptr_C_managed.get();
    int* csr_col_ind_C = (int*)csr_col_ind_C_managed.get();
    T*   csr_val_C     = (T*)csr_val_C_managed.get();
    int* nnz_per_row   = (int*)nnz_per_row_managed.get();
    int* nnz_C         = (int*)nnz_C_managed.get();

    int local_ptr[2] = {0, 1};
    CHECK_HIP_ERROR(
        hipMemcpy(csr_row_ptr_A, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(csr_row_ptr_C, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));

    int local_nnz[1] = {1};
    CHECK_HIP_ERROR(hipMemcpy(nnz_per_row, local_nnz, sizeof(int), hipMemcpyHostToDevice));

    verify_hipsparse_status_invalid_handle(hipsparseXnnz_compress(
        nullptr, m, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol));
    verify_hipsparse_status_invalid_pointer(
        hipsparseXnnz_compress(
            handle, m, nullptr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol),
        "Error: Matrix descriptor is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXnnz_compress(handle, m, csr_descr, csr_val_A, nullptr, nnz_per_row, nnz_C, tol),
        "Error: CSR row pointer array is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXnnz_compress(handle, m, csr_descr, csr_val_A, csr_row_ptr_A, nullptr, nnz_C, tol),
        "Error: Number of elements per row array is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXnnz_compress(
            handle, m, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nullptr, tol),
        "Error: Total number of elements pointer is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXnnz_compress(
            handle, -1, csr_descr, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol),
        "Error: Matrix size is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXnnz_compress(handle,
                                                                m,
                                                                csr_descr,
                                                                csr_val_A,
                                                                csr_row_ptr_A,
                                                                nnz_per_row,
                                                                nnz_C,
                                                                make_DataType<T>(-1)),
                                         "Error: Tolerance is invalid");

    verify_hipsparse_status_invalid_handle(hipsparseXcsr2csr_compress(nullptr,
                                                                      m,
                                                                      n,
                                                                      csr_descr,
                                                                      csr_val_A,
                                                                      csr_col_ind_A,
                                                                      csr_row_ptr_A,
                                                                      nnz_A,
                                                                      nnz_per_row,
                                                                      csr_val_C,
                                                                      csr_col_ind_C,
                                                                      csr_row_ptr_C,
                                                                      tol));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       nullptr,
                                                                       csr_val_A,
                                                                       csr_col_ind_A,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       csr_val_C,
                                                                       csr_col_ind_C,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: Matrix descriptor is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       (const T*)nullptr,
                                                                       csr_col_ind_A,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       csr_val_C,
                                                                       csr_col_ind_C,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: CSR matrix values array is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val_A,
                                                                       nullptr,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       csr_val_C,
                                                                       csr_col_ind_C,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: CSR matrix column indices array is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val_A,
                                                                       csr_col_ind_A,
                                                                       nullptr,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       csr_val_C,
                                                                       csr_col_ind_C,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: CSR matrix row pointer array is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val_A,
                                                                       csr_col_ind_A,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nullptr,
                                                                       csr_val_C,
                                                                       csr_col_ind_C,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: Number of elements per row array is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val_A,
                                                                       csr_col_ind_A,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       (T*)nullptr,
                                                                       csr_col_ind_C,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: CSR matrix values array is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val_A,
                                                                       csr_col_ind_A,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       csr_val_C,
                                                                       nullptr,
                                                                       csr_row_ptr_C,
                                                                       tol),
                                            "Error: CSR matrix column indices array is invalid");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2csr_compress(handle,
                                                                       m,
                                                                       n,
                                                                       csr_descr,
                                                                       csr_val_A,
                                                                       csr_col_ind_A,
                                                                       csr_row_ptr_A,
                                                                       nnz_A,
                                                                       nnz_per_row,
                                                                       csr_val_C,
                                                                       csr_col_ind_C,
                                                                       nullptr,
                                                                       tol),
                                            "Error: CSR matrix row pointer array is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2csr_compress(handle,
                                                                    -1,
                                                                    n,
                                                                    csr_descr,
                                                                    csr_val_A,
                                                                    csr_col_ind_A,
                                                                    csr_row_ptr_A,
                                                                    nnz_A,
                                                                    nnz_per_row,
                                                                    csr_val_C,
                                                                    csr_col_ind_C,
                                                                    csr_row_ptr_C,
                                                                    tol),
                                         "Error: Matrix size is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2csr_compress(handle,
                                                                    m,
                                                                    -1,
                                                                    csr_descr,
                                                                    csr_val_A,
                                                                    csr_col_ind_A,
                                                                    csr_row_ptr_A,
                                                                    nnz_A,
                                                                    nnz_per_row,
                                                                    csr_val_C,
                                                                    csr_col_ind_C,
                                                                    csr_row_ptr_C,
                                                                    tol),
                                         "Error: Matrix size is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2csr_compress(handle,
                                                                    m,
                                                                    n,
                                                                    csr_descr,
                                                                    csr_val_A,
                                                                    csr_col_ind_A,
                                                                    csr_row_ptr_A,
                                                                    -1,
                                                                    nnz_per_row,
                                                                    csr_val_C,
                                                                    csr_col_ind_C,
                                                                    csr_row_ptr_C,
                                                                    tol),
                                         "Error: Matrix size is invalid");
    verify_hipsparse_status_invalid_value(hipsparseXcsr2csr_compress(handle,
                                                                     m,
                                                                     n,
                                                                     csr_descr,
                                                                     csr_val_A,
                                                                     csr_col_ind_A,
                                                                     csr_row_ptr_A,
                                                                     nnz_A,
                                                                     nnz_per_row,
                                                                     csr_val_C,
                                                                     csr_col_ind_C,
                                                                     csr_row_ptr_C,
                                                                     static_cast<T>(-1)),
                                          "Error: Tolerance is invalid");
#endif
}

template <typename T>
hipsparseStatus_t testing_csr2csr_compress(Arguments argus)
{
    int                  m        = argus.M;
    int                  n        = argus.N;
    T                    tol      = make_DataType<T>(argus.alpha);
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, idx_base);

    hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE);

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    // Read or construct CSR matrix
    int hnnz_A = 0;
    if(!generate_csr_matrix(
           filename, m, n, hnnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_A), device_free};
    auto dcsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_A), device_free};
    auto dcsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dnnz_per_row_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * m), device_free};

    int* dcsr_row_ptr_A = (int*)dcsr_row_ptr_A_managed.get();
    int* dcsr_col_ind_A = (int*)dcsr_col_ind_A_managed.get();
    T*   dcsr_val_A     = (T*)dcsr_val_A_managed.get();
    int* dcsr_row_ptr_C = (int*)dcsr_row_ptr_C_managed.get();
    int* dnnz_per_row   = (int*)dnnz_per_row_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(int) * hnnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(T) * hnnz_A, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Use both host and device pointers for nnz_C and confirm they give the same answer
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        int hnnz_C;
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz_compress(
            handle, m, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &hnnz_C, tol));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

        auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
        int* dnnz_C         = (int*)dnnz_C_managed.get();
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz_compress(
            handle, m, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, dnnz_C, tol));

        int hnnz_C_copied_from_device;
        CHECK_HIP_ERROR(
            hipMemcpy(&hnnz_C_copied_from_device, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

        unit_check_general(1, 1, 1, &hnnz_C_copied_from_device, &hnnz_C);

        if(hnnz_C == 0)
        {
            return HIPSPARSE_STATUS_SUCCESS;
        }

        // Allocate device memory for compressed CSR columns indices and values
        auto dcsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C), device_free};
        auto dcsr_val_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C), device_free};

        int* dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
        T*   dcsr_val_C     = (T*)dcsr_val_C_managed.get();

        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2csr_compress(handle,
                                                         m,
                                                         n,
                                                         csr_descr,
                                                         dcsr_val_A,
                                                         dcsr_col_ind_A,
                                                         dcsr_row_ptr_A,
                                                         hnnz_A,
                                                         dnnz_per_row,
                                                         dcsr_val_C,
                                                         dcsr_col_ind_C,
                                                         dcsr_row_ptr_C,
                                                         tol));

        // Copy output from device to host
        std::vector<int> hcsr_row_ptr_C(m + 1);
        std::vector<int> hcsr_col_ind_C(hnnz_C);
        std::vector<T>   hcsr_val_C(hnnz_C);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * hnnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(T) * hnnz_C, hipMemcpyDeviceToHost));

        // Host csr2csc conversion
        std::vector<int> hcsr_row_ptr_C_gold;
        std::vector<int> hcsr_col_ind_C_gold;
        std::vector<T>   hcsr_val_gold;

        // Call host conversion here
        host_csr_to_csr_compress<T>(m,
                                    n,
                                    hcsr_row_ptr_A,
                                    hcsr_col_ind_A,
                                    hcsr_val_A,
                                    hcsr_row_ptr_C_gold,
                                    hcsr_col_ind_C_gold,
                                    hcsr_val_gold,
                                    idx_base,
                                    tol);

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        unit_check_general(1, hnnz_C, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        unit_check_general(1, hnnz_C, 1, hcsr_val_gold.data(), hcsr_val_C.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        int hnnz_C;
        CHECK_HIPSPARSE_ERROR(hipsparseXnnz_compress(
            handle, m, csr_descr, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &hnnz_C, tol));

        // Allocate device memory for compressed CSR columns indices and values
        auto dcsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C), device_free};
        auto dcsr_val_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C), device_free};

        int* dcsr_col_ind_C = (int*)dcsr_col_ind_C_managed.get();
        T*   dcsr_val_C     = (T*)dcsr_val_C_managed.get();

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2csr_compress(handle,
                                                             m,
                                                             n,
                                                             csr_descr,
                                                             dcsr_val_A,
                                                             dcsr_col_ind_A,
                                                             dcsr_row_ptr_A,
                                                             hnnz_A,
                                                             dnnz_per_row,
                                                             dcsr_val_C,
                                                             dcsr_col_ind_C,
                                                             dcsr_row_ptr_C,
                                                             tol));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2csr_compress(handle,
                                                             m,
                                                             n,
                                                             csr_descr,
                                                             dcsr_val_A,
                                                             dcsr_col_ind_A,
                                                             dcsr_row_ptr_A,
                                                             hnnz_A,
                                                             dnnz_per_row,
                                                             dcsr_val_C,
                                                             dcsr_col_ind_C,
                                                             dcsr_row_ptr_C,
                                                             tol));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2csr_compress_gbyte_count<T>(m, hnnz_A, hnnz_C);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnzA,
                            hnnz_A,
                            display_key_t::nnzC,
                            hnnz_C,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2CSR_COMPRESS_HPP
