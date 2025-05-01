/* ************************************************************************
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CSCSORT_HPP
#define TESTING_CSCSORT_HPP

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

void testing_cscsort_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int m         = 100;
    int n         = 100;
    int nnz       = 100;
    int safe_size = 100;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    size_t buffer_size = 0;

    auto csc_col_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csc_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto perm_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  csc_col_ptr = (int*)csc_col_ptr_managed.get();
    int*  csc_row_ind = (int*)csc_row_ind_managed.get();
    int*  perm        = (int*)perm_managed.get();
    void* buffer      = (void*)buffer_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcscsort_bufferSizeExt(
            handle, m, n, nnz, (int*)nullptr, csc_row_ind, &buffer_size),
        "Error: csc_col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcscsort_bufferSizeExt(
            handle, m, n, nnz, csc_col_ptr, (int*)nullptr, &buffer_size),
        "Error: csc_row_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcscsort_bufferSizeExt(
            handle, m, n, nnz, csc_col_ptr, csc_row_ind, (size_t*)nullptr),
        "Error: buffer_size is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcscsort_bufferSizeExt(
        (hipsparseHandle_t) nullptr, m, n, nnz, csc_col_ptr, csc_row_ind, &buffer_size));

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcscsort(handle, m, n, nnz, descr, (int*)nullptr, csc_row_ind, perm, buffer),
        "Error: csc_col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcscsort(handle, m, n, nnz, descr, csc_col_ptr, (int*)nullptr, perm, buffer),
        "Error: csc_row_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcscsort(handle, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, (int*)nullptr),
        "Error: buffer is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcscsort(handle,
                                                              m,
                                                              n,
                                                              nnz,
                                                              (hipsparseMatDescr_t) nullptr,
                                                              csc_col_ptr,
                                                              csc_row_ind,
                                                              perm,
                                                              buffer),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcscsort(
        (hipsparseHandle_t) nullptr, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, buffer));
#endif
}

hipsparseStatus_t testing_cscsort(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  m        = argus.M;
    int                  n        = argus.N;
    int                  permute  = argus.permute;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    if(m == 0 || n == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int>   hcsc_col_ptr;
    std::vector<int>   hcsc_row_ind;
    std::vector<float> hcsc_val;

    // Read or construct CSC matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, n, m, nnz, hcsc_col_ptr, hcsc_row_ind, hcsc_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // Unsort CSC columns
    std::vector<int>   hperm(nnz);
    std::vector<int>   hcsc_row_ind_unsorted(nnz);
    std::vector<float> hcsc_val_unsorted(nnz);

    hcsc_row_ind_unsorted = hcsc_row_ind;
    hcsc_val_unsorted     = hcsc_val;

    for(int i = 0; i < n; ++i)
    {
        int col_begin = hcsc_col_ptr[i] - idx_base;
        int col_end   = hcsc_col_ptr[i + 1] - idx_base;
        int col_nnz   = col_end - col_begin;

        for(int j = col_begin; j < col_end; ++j)
        {
            int rng = col_begin + rand() % col_nnz;

            int   temp_row = hcsc_row_ind_unsorted[j];
            float temp_val = hcsc_val_unsorted[j];

            hcsc_row_ind_unsorted[j] = hcsc_row_ind_unsorted[rng];
            hcsc_val_unsorted[j]     = hcsc_val_unsorted[rng];

            hcsc_row_ind_unsorted[rng] = temp_row;
            hcsc_val_unsorted[rng]     = temp_val;
        }
    }

    // Allocate memory on the device
    auto dcsc_col_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (n + 1)), device_free};
    auto dcsc_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsc_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dcsc_val_sorted_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dperm_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};

    int*   dcsc_col_ptr    = (int*)dcsc_col_ptr_managed.get();
    int*   dcsc_row_ind    = (int*)dcsc_row_ind_managed.get();
    float* dcsc_val        = (float*)dcsc_val_managed.get();
    float* dcsc_val_sorted = (float*)dcsc_val_sorted_managed.get();

    // Set permutation vector, if asked for
#ifdef __HIP_PLATFORM_NVIDIA__
    // cusparse does not allow nullptr
    int* dperm = (int*)dperm_managed.get();
#else
    int* dperm = permute ? (int*)dperm_managed.get() : nullptr;
#endif

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_col_ptr, hcsc_col_ptr.data(), sizeof(int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsc_row_ind, hcsc_row_ind_unsorted.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_val, hcsc_val_unsorted.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXcscsort_bufferSizeExt(
        handle, m, n, nnz, dcsc_col_ptr, dcsc_row_ind, &bufferSize));

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
        // Sort CSC columns
        CHECK_HIPSPARSE_ERROR(hipsparseXcscsort(
            handle, m, n, nnz, descr, dcsc_col_ptr, dcsc_row_ind, dperm, dbuffer));

        if(permute)
        {
            // Sort CSC values
            CHECK_HIPSPARSE_ERROR(hipsparseSgthr(
                handle, nnz, dcsc_val, dcsc_val_sorted, dperm, HIPSPARSE_INDEX_BASE_ZERO));
        }

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsc_row_ind_unsorted.data(), dcsc_row_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));

        if(permute)
        {
            CHECK_HIP_ERROR(hipMemcpy(hcsc_val_unsorted.data(),
                                      dcsc_val_sorted,
                                      sizeof(float) * nnz,
                                      hipMemcpyDeviceToHost));
        }

        // Unit check
        unit_check_general(1, nnz, 1, hcsc_row_ind.data(), hcsc_row_ind_unsorted.data());

        if(permute)
        {
            unit_check_general(1, nnz, 1, hcsc_val.data(), hcsc_val_unsorted.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcscsort(
                handle, m, n, nnz, descr, dcsc_col_ptr, dcsc_row_ind, dperm, dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcscsort(
                handle, m, n, nnz, descr, dcsc_col_ptr, dcsc_row_ind, dperm, dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = cscsort_gbyte_count(n, nnz, permute);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::permute,
                            (permute ? "yes" : "no"),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSCSORT_HPP
