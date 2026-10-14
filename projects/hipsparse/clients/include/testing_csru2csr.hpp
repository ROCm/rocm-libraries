/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CSRU2CSR_HPP
#define TESTING_CSRU2CSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <algorithm>
#include <string>
#include <vector>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csru2csr_bad_arg(const Arguments& argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    int m         = 100;
    int n         = 100;
    int nnz       = 100;
    int safe_size = 100;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<csru2csr_struct> unique_ptr_csru2csr(new csru2csr_struct);
    csru2csrInfo_t                   info = unique_ptr_csru2csr->info;

    size_t bufferSize;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int*   csr_col_ind = (int*)csr_col_ind_managed.get();
    float* csr_val     = (float*)csr_val_managed.get();
    void*  buffer      = (void*)buffer_managed.get();

    // Testing csru2csr_bufferSizeExt for bad args
#ifndef __HIP_PLATFORM_NVIDIA__
    // cusparse seem to not have any error checking
    verify_hipsparse_status_invalid_handle(hipsparseXcsru2csr_bufferSizeExt(
        nullptr, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, info, &bufferSize));
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, n, nnz, (float*)nullptr, csr_row_ptr, csr_col_ind, info, &bufferSize),
        "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csr_val, nullptr, csr_col_ind, info, &bufferSize),
        "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csr_val, csr_row_ptr, nullptr, info, &bufferSize),
        "Error: csr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, nullptr, &bufferSize),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, info, nullptr),
        "Error: bufferSize is nullptr");
#endif
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, -1, n, nnz, csr_val, csr_row_ptr, csr_col_ind, info, &bufferSize),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, -1, nnz, csr_val, csr_row_ptr, csr_col_ind, info, &bufferSize),
        "Error: n is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, n, -1, csr_val, csr_row_ptr, csr_col_ind, info, &bufferSize),
        "Error: nnz is invalid");
#ifndef __HIP_PLATFORM_NVIDIA__
    // cusparse seem to not have any error checking for some parts
    verify_hipsparse_status_success(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, 0, n, 0, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Success");
    verify_hipsparse_status_success(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, m, 0, 0, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Success");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, 0, n, nnz, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Error: nnz is invalid");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr_bufferSizeExt(
            handle, 0, n, 0, (float*)nullptr, nullptr, nullptr, nullptr, nullptr),
        "Error: bufferSize is invalid");
#endif

    // Testing csru2csr for bad args
#ifndef __HIP_PLATFORM_NVIDIA__
    // cusparse seem to not have any error checking for some parts
    verify_hipsparse_status_invalid_handle(hipsparseXcsru2csr(
        nullptr, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer));
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr(
            handle, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, nullptr, buffer),
        "Error: info is nullptr");
#endif
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr(
            handle, m, n, nnz, nullptr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr(
            handle, m, n, nnz, descr, (float*)nullptr, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr(handle, m, n, nnz, descr, csr_val, nullptr, csr_col_ind, info, buffer),
        "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr(handle, m, n, nnz, descr, csr_val, csr_row_ptr, nullptr, info, buffer),
        "Error: csr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsru2csr(
            handle, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, nullptr),
        "Error: buffer is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr(
            handle, -1, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr(
            handle, m, -1, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: n is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr(
            handle, m, n, -1, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: nnz is invalid");
#ifndef __HIP_PLATFORM_NVIDIA__
    // cusparse seem to not have any error checking for some parts
    verify_hipsparse_status_success(
        hipsparseXcsru2csr(
            handle, 0, n, 0, nullptr, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Success");
    verify_hipsparse_status_success(
        hipsparseXcsru2csr(
            handle, m, 0, 0, nullptr, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Success");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsru2csr(
            handle, 0, n, nnz, nullptr, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Error: nnz is invalid");
#endif

    // Testing csr2csru for bad args
#ifndef __HIP_PLATFORM_NVIDIA__
    // cusparse seem to not have any error checking for some parts
    verify_hipsparse_status_invalid_handle(hipsparseXcsr2csru(
        nullptr, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer));
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsr2csru(
            handle, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, nullptr, buffer),
        "Error: info is nullptr");
#endif
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsr2csru(
            handle, m, n, nnz, nullptr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsr2csru(
            handle, m, n, nnz, descr, (float*)nullptr, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsr2csru(handle, m, n, nnz, descr, csr_val, nullptr, csr_col_ind, info, buffer),
        "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsr2csru(handle, m, n, nnz, descr, csr_val, csr_row_ptr, nullptr, info, buffer),
        "Error: csr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsr2csru(
            handle, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, nullptr),
        "Error: buffer is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsr2csru(
            handle, -1, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: m is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsr2csru(
            handle, m, -1, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: n is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsr2csru(
            handle, m, n, -1, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer),
        "Error: nnz is invalid");
#ifndef __HIP_PLATFORM_NVIDIA__
    // cusparse seem to not have any error checking for some parts
    verify_hipsparse_status_success(
        hipsparseXcsr2csru(
            handle, 0, n, 0, nullptr, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Success");
    verify_hipsparse_status_success(
        hipsparseXcsr2csru(
            handle, m, 0, 0, nullptr, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Success");
    verify_hipsparse_status_invalid_size(
        hipsparseXcsr2csru(
            handle, 0, n, nnz, nullptr, (float*)nullptr, nullptr, nullptr, nullptr, &bufferSize),
        "Error: nnz is invalid");
#endif
#endif
}

template <typename T>
void testing_csru2csr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    int                  m        = argus.M;
    int                  n        = argus.N;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    hipsparseSetMatIndexBase(descr, idx_base);

    std::unique_ptr<csru2csr_struct> unique_ptr_info(new csru2csr_struct);
    csru2csrInfo_t                   info = unique_ptr_info->info;

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind_gold;
    std::vector<T>   hcsr_val_gold;

    // Read or construct CSR matrix
    int nnz = 0;
    CHECK_GENERATE_MATRIX_ERROR(generate_csr_matrix(
        filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind_gold, hcsr_val_gold, idx_base));

    // Unsort CSR columns
    std::vector<int> hperm(nnz);
    std::vector<int> hcsr_col_ind_unsorted_gold(nnz);
    std::vector<T>   hcsr_val_unsorted_gold(nnz);

    hcsr_col_ind_unsorted_gold = hcsr_col_ind_gold;
    hcsr_val_unsorted_gold     = hcsr_val_gold;

    for(int i = 0; i < m; ++i)
    {
        int row_begin = hcsr_row_ptr[i] - idx_base;
        int row_end   = hcsr_row_ptr[i + 1] - idx_base;
        int row_nnz   = row_end - row_begin;

        for(int j = row_begin; j < row_end; ++j)
        {
            int rng = row_begin + rand() % row_nnz;

            int temp_col = hcsr_col_ind_unsorted_gold[j] - idx_base;
            T   temp_val = hcsr_val_unsorted_gold[j];

            hcsr_col_ind_unsorted_gold[j] = hcsr_col_ind_unsorted_gold[rng];
            hcsr_val_unsorted_gold[j]     = hcsr_val_unsorted_gold[rng];

            hcsr_col_ind_unsorted_gold[rng] = temp_col + idx_base;
            hcsr_val_unsorted_gold[rng]     = temp_val;
        }
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind_unsorted_gold.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val, hcsr_val_unsorted_gold.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t buffer_size;

    CHECK_HIPSPARSE_ERROR(hipsparseXcsru2csr_bufferSizeExt(
        handle, m, n, nnz, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, &buffer_size));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(argus.unit_check)
    {
        // Sort CSR columns
        CHECK_HIPSPARSE_ERROR(hipsparseXcsru2csr(
            handle, m, n, nnz, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, dbuffer));

        // Copy output from device to host
        std::vector<int> hcsr_col_ind(nnz);
        std::vector<T>   hcsr_val(nnz);

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Unsort CSR columns back to original state
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2csru(
            handle, m, n, nnz, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, dbuffer));

        // Copy output from device to host
        std::vector<int> hcsr_col_ind_unsorted(nnz);
        std::vector<T>   hcsr_val_unsorted(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_unsorted.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_unsorted.data(), dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, nnz, 1, hcsr_col_ind.data(), hcsr_col_ind_gold.data());
        unit_check_general(1, nnz, 1, hcsr_val.data(), hcsr_val_gold.data());
        unit_check_general(
            1, nnz, 1, hcsr_col_ind_unsorted.data(), hcsr_col_ind_unsorted_gold.data());
        unit_check_general(1, nnz, 1, hcsr_val_unsorted.data(), hcsr_val_unsorted_gold.data());
    }
#endif
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
// Reuse one csru2csrInfo across hipsparseDcsru2csr calls with different nnz
static void csru2csr_double_sort_check(hipsparseHandle_t      handle,
                                       hipsparseMatDescr_t    descr,
                                       csru2csrInfo_t         info,
                                       int                    m,
                                       int                    n,
                                       std::vector<int>       row_ptr,
                                       std::vector<int>       col_unsorted)
{
    int nnz = row_ptr[m];

    std::vector<double> val_unsorted(nnz);
    for(int i = 0; i < m; ++i)
        for(int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            val_unsorted[j] = 100.0 * i + col_unsorted[j];

    std::vector<int>    col_gold = col_unsorted;
    std::vector<double> val_gold(nnz);
    for(int i = 0; i < m; ++i)
        std::sort(col_gold.begin() + row_ptr[i], col_gold.begin() + row_ptr[i + 1]);
    for(int i = 0; i < m; ++i)
        for(int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            val_gold[j] = 100.0 * i + col_gold[j];

    int*    drow;
    int*    dcol;
    double* dval;
    CHECK_HIP_ERROR(hipMalloc((void**)&drow, sizeof(int) * (m + 1)));
    CHECK_HIP_ERROR(hipMalloc((void**)&dcol, sizeof(int) * nnz));
    CHECK_HIP_ERROR(hipMalloc((void**)&dval, sizeof(double) * nnz));
    CHECK_HIP_ERROR(
        hipMemcpy(drow, row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol, col_unsorted.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dval, val_unsorted.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));

    size_t buffer_size = 0;
    ASSERT_EQ(hipsparseDcsru2csr_bufferSizeExt(
                  handle, m, n, nnz, dval, drow, dcol, info, &buffer_size),
              HIPSPARSE_STATUS_SUCCESS);
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size > 0 ? buffer_size : 1));

    ASSERT_EQ(hipsparseDcsru2csr(handle, m, n, nnz, descr, dval, drow, dcol, info, dbuffer),
              HIPSPARSE_STATUS_SUCCESS);

    std::vector<int>    col_out(nnz);
    std::vector<double> val_out(nnz);
    CHECK_HIP_ERROR(
        hipMemcpy(col_out.data(), dcol, sizeof(int) * nnz, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(val_out.data(), dval, sizeof(double) * nnz, hipMemcpyDeviceToHost));

    EXPECT_EQ(col_out, col_gold);
    EXPECT_EQ(val_out, val_gold);

    CHECK_HIP_ERROR(hipFree(drow));
    CHECK_HIP_ERROR(hipFree(dcol));
    CHECK_HIP_ERROR(hipFree(dval));
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

TEST(csru2csr_info_reuse, double_realloc_and_empty_quick_return)
{
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t            descr = unique_ptr_descr->descr;
    hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);
    std::unique_ptr<csru2csr_struct> unique_ptr_info(new csru2csr_struct);
    csru2csrInfo_t                   info = unique_ptr_info->info;

    std::vector<int> small_ptr{0, 3, 5};
    std::vector<int> small_col{3, 0, 2, 1, 0};
    std::vector<int> large_ptr{0, 4, 6, 9};
    std::vector<int> large_col{4, 1, 3, 0, 2, 0, 3, 1, 2};

    // First call allocates info->P for nnz = 5
    csru2csr_double_sort_check(handle, descr, info, 2, 4, small_ptr, small_col);
    // info->size (5) != nnz (9): permutation is freed and reallocated
    csru2csr_double_sort_check(handle, descr, info, 3, 5, large_ptr, large_col);

    // Quick returns succeed and do not touch info
    EXPECT_EQ(hipsparseDcsru2csr(
                  handle, 3, 5, 0, descr, (double*)nullptr, nullptr, nullptr, info, nullptr),
              HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseDcsru2csr(
                  handle, 0, 5, 0, descr, (double*)nullptr, nullptr, nullptr, info, nullptr),
              HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseDcsru2csr(
                  handle, 3, 0, 0, descr, (double*)nullptr, nullptr, nullptr, info, nullptr),
              HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseDcsru2csr(
                  handle, 0, 5, 3, descr, (double*)nullptr, nullptr, nullptr, info, nullptr),
              HIPSPARSE_STATUS_INVALID_VALUE);

    // Info is still usable after the quick returns, including shrinking again
    csru2csr_double_sort_check(handle, descr, info, 3, 5, large_ptr, large_col);
    csru2csr_double_sort_check(handle, descr, info, 2, 4, small_ptr, small_col);
}
#endif

#endif // TESTING_CSRU2CSR_HPP
