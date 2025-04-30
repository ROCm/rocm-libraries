/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_COOSORT_HPP
#define TESTING_COOSORT_HPP

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

void testing_coosort_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int m         = 100;
    int n         = 100;
    int nnz       = 100;
    int safe_size = 100;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    size_t buffer_size = 0;

    auto coo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto coo_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto perm_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  coo_row_ind = (int*)coo_row_ind_managed.get();
    int*  coo_col_ind = (int*)coo_col_ind_managed.get();
    int*  perm        = (int*)perm_managed.get();
    void* buffer      = (void*)buffer_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, (int*)nullptr, coo_col_ind, &buffer_size),
        "Error: coo_row_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, coo_row_ind, (int*)nullptr, &buffer_size),
        "Error: coo_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosort_bufferSizeExt(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, (size_t*)nullptr),
        "Error: buffer_size is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcoosort_bufferSizeExt(
        (hipsparseHandle_t) nullptr, m, n, nnz, coo_row_ind, coo_col_ind, &buffer_size));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosortByRow(handle, m, n, nnz, (int*)nullptr, coo_col_ind, perm, buffer),
        "Error: coo_row_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosortByRow(handle, m, n, nnz, coo_row_ind, (int*)nullptr, perm, buffer),
        "Error: coo_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosortByRow(handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, (int*)nullptr),
        "Error: buffer is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcoosortByRow(
        (hipsparseHandle_t) nullptr, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosortByColumn(handle, m, n, nnz, (int*)nullptr, coo_col_ind, perm, buffer),
        "Error: coo_row_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosortByColumn(handle, m, n, nnz, coo_row_ind, (int*)nullptr, perm, buffer),
        "Error: coo_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoosortByColumn(handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, (int*)nullptr),
        "Error: buffer is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcoosortByColumn(
        (hipsparseHandle_t) nullptr, m, n, nnz, coo_row_ind, coo_col_ind, perm, buffer));
#endif
}

hipsparseStatus_t testing_coosort(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  m        = argus.M;
    int                  n        = argus.N;
    int                  by_row   = (argus.transA == HIPSPARSE_OPERATION_NON_TRANSPOSE);
    int                  permute  = argus.permute;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    if(m == 0 || n == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int>   hcoo_row_ind;
    std::vector<int>   hcoo_col_ind;
    std::vector<float> hcoo_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_coo_matrix(filename, m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Unsort COO columns
    std::vector<int>   hcoo_row_ind_unsorted(nnz);
    std::vector<int>   hcoo_col_ind_unsorted(nnz);
    std::vector<float> hcoo_val_unsorted(nnz);

    hcoo_row_ind_unsorted = hcoo_row_ind;
    hcoo_col_ind_unsorted = hcoo_col_ind;
    hcoo_val_unsorted     = hcoo_val;

    for(int i = 0; i < nnz; ++i)
    {
        int rng = rand() % nnz;

        int   temp_row = hcoo_row_ind_unsorted[i];
        int   temp_col = hcoo_col_ind_unsorted[i];
        float temp_val = hcoo_val_unsorted[i];

        hcoo_row_ind_unsorted[i] = hcoo_row_ind_unsorted[rng];
        hcoo_col_ind_unsorted[i] = hcoo_col_ind_unsorted[rng];
        hcoo_val_unsorted[i]     = hcoo_val_unsorted[rng];

        hcoo_row_ind_unsorted[rng] = temp_row;
        hcoo_col_ind_unsorted[rng] = temp_col;
        hcoo_val_unsorted[rng]     = temp_val;
    }

    // If coosort by column, sort host arrays by column
    if(!by_row)
    {
        std::vector<int> hperm(nnz);
        for(int i = 0; i < nnz; ++i)
        {
            hperm[i] = i;
        }

        std::sort(hperm.begin(), hperm.end(), [&](const int& a, const int& b) {
            if(hcoo_col_ind_unsorted[a] < hcoo_col_ind_unsorted[b])
            {
                return true;
            }
            else if(hcoo_col_ind_unsorted[a] == hcoo_col_ind_unsorted[b])
            {
                return (hcoo_row_ind_unsorted[a] < hcoo_row_ind_unsorted[b]);
            }
            else
            {
                return false;
            }
        });

        for(int i = 0; i < nnz; ++i)
        {
            hcoo_row_ind[i] = hcoo_row_ind_unsorted[hperm[i]];
            hcoo_col_ind[i] = hcoo_col_ind_unsorted[hperm[i]];
            hcoo_val[i]     = hcoo_val_unsorted[hperm[i]];
        }
    }

    // Allocate memory on the device
    auto dcoo_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcoo_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcoo_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dcoo_val_sorted_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dperm_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int*   dcoo_row_ind    = (int*)dcoo_row_ind_managed.get();
    int*   dcoo_col_ind    = (int*)dcoo_col_ind_managed.get();
    float* dcoo_val        = (float*)dcoo_val_managed.get();
    float* dcoo_val_sorted = (float*)dcoo_val_sorted_managed.get();

    // Set permutation vector, if asked for
    int* dperm = permute ? (int*)dperm_managed.get() : nullptr;

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcoo_row_ind, hcoo_row_ind_unsorted.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcoo_col_ind, hcoo_col_ind_unsorted.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_val, hcoo_val_unsorted.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXcoosort_bufferSizeExt(
        handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(permute)
    {
        // Initialize perm with identity permutation
        CHECK_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, dperm));
    }

    if(argus.unit_check)
    {
        // Sort CSR columns
        if(by_row)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByRow(
                handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
        }
        else
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByColumn(
                handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
        }

        if(permute)
        {
            // Sort CSR values
            CHECK_HIPSPARSE_ERROR(hipsparseSgthr(
                handle, nnz, dcoo_val, dcoo_val_sorted, dperm, HIPSPARSE_INDEX_BASE_ZERO));
        }

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind_unsorted.data(), dcoo_row_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_col_ind_unsorted.data(), dcoo_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));

        if(permute)
        {
            CHECK_HIP_ERROR(hipMemcpy(hcoo_val_unsorted.data(),
                                      dcoo_val_sorted,
                                      sizeof(float) * nnz,
                                      hipMemcpyDeviceToHost));
        }

        // Unit check
        unit_check_general(1, nnz, 1, hcoo_row_ind.data(), hcoo_row_ind_unsorted.data());
        unit_check_general(1, nnz, 1, hcoo_col_ind.data(), hcoo_col_ind_unsorted.data());

        if(permute)
        {
            unit_check_general(1, nnz, 1, hcoo_val.data(), hcoo_val_unsorted.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            if(by_row)
            {
                CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByRow(
                    handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
            }
            else
            {
                CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByColumn(
                    handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
            }
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            if(by_row)
            {
                CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByRow(
                    handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
            }
            else
            {
                CHECK_HIPSPARSE_ERROR(hipsparseXcoosortByColumn(
                    handle, m, n, nnz, dcoo_row_ind, dcoo_col_ind, dperm, dbuffer));
            }
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = coosort_gbyte_count(nnz, permute);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::permute,
                            (permute ? "yes" : "no"),
                            display_key_t::direction,
                            (by_row ? "row" : "column"),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_COOSORT_HPP
