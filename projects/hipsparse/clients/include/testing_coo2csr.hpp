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
#ifndef TESTING_COO2CSR_HPP
#define TESTING_COO2CSR_HPP

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

void testing_coo2csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  m         = 100;
    int                  nnz       = 100;
    int                  safe_size = 100;
    hipsparseIndexBase_t idx_base  = HIPSPARSE_INDEX_BASE_ZERO;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto coo_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* coo_row_ind = (int*)coo_row_ind_managed.get();
    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoo2csr(handle, (int*)nullptr, nnz, m, csr_row_ptr, idx_base),
        "Error: coo_row_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcoo2csr(handle, coo_row_ind, nnz, m, (int*)nullptr, idx_base),
        "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_handle(
        hipsparseXcoo2csr((hipsparseHandle_t) nullptr, coo_row_ind, nnz, m, csr_row_ptr, idx_base));
#endif
}

template <typename T>
hipsparseStatus_t testing_coo2csr(Arguments argus)
{
    int                  m        = argus.M;
    int                  n        = argus.N;
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
    std::vector<int> hcoo_row_ind;
    std::vector<int> hcoo_col_ind;
    std::vector<T>   hcoo_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_coo_matrix(filename, m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<int> hcsr_row_ptr(m + 1);
    std::vector<int> hcsr_row_ptr_gold(m + 1, 0);

    // Allocate memory on the device
    auto dcoo_row_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};

    int* dcoo_row_ind = (int*)dcoo_row_ind_managed.get();
    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXcoo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));

        // CPU
        // coo2csr on host
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr_gold[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr_gold[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_gold[i + 1] += hcsr_row_ptr_gold[i];
        }

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXcoo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXcoo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = coo2csr_gbyte_count<T>(m, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_COO2CSR_HPP
