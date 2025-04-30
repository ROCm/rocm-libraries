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
#ifndef TESTING_HYB2CSR_HPP
#define TESTING_HYB2CSR_HPP

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
void testing_hyb2csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int safe_size = 100;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    testhyb* dhyb = (testhyb*)hyb;

    dhyb->m       = safe_size;
    dhyb->n       = safe_size;
    dhyb->ell_nnz = safe_size;
    dhyb->coo_nnz = safe_size;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseXhyb2csr(handle, descr, hyb, csr_val, (int*)nullptr, csr_col_ind),
        "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, (int*)nullptr),
        "Error: csr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXhyb2csr(handle, descr, hyb, (T*)nullptr, csr_row_ptr, csr_col_ind),
        "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXhyb2csr(
            handle, (hipsparseMatDescr_t) nullptr, hyb, csr_val, csr_row_ptr, csr_col_ind),
        "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXhyb2csr(
            handle, descr, (hipsparseHybMat_t) nullptr, csr_val, csr_row_ptr, csr_col_ind),
        "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXhyb2csr(
        (hipsparseHandle_t) nullptr, descr, hyb, csr_val, csr_row_ptr, csr_col_ind));
#endif
}

template <typename T>
hipsparseStatus_t testing_hyb2csr(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int                  m        = argus.M;
    int                  n        = argus.N;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr_gold;
    std::vector<int> hcsr_col_ind_gold;
    std::vector<T>   hcsr_val_gold;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(
           filename, m, n, nnz, hcsr_row_ptr_gold, hcsr_col_ind_gold, hcsr_val_gold, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
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
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr_gold.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind_gold.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val, hcsr_val_gold.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR to HYB
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2hyb(handle,
                                            m,
                                            n,
                                            descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            hyb,
                                            0,
                                            HIPSPARSE_HYB_PARTITION_AUTO));

    // Set all CSR arrays to zero
    CHECK_HIP_ERROR(hipMemset(dcsr_row_ptr, 0, sizeof(int) * (m + 1)));
    CHECK_HIP_ERROR(hipMemset(dcsr_col_ind, 0, sizeof(int) * nnz));
    CHECK_HIP_ERROR(hipMemset(dcsr_val, 0, sizeof(T) * nnz));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(
            hipsparseXhyb2csr(handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind));

        // Copy output from device to host
        std::vector<int> hcsr_row_ptr(m + 1);
        std::vector<int> hcsr_col_ind(nnz);
        std::vector<T>   hcsr_val(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
        unit_check_general(1, nnz, 1, hcsr_col_ind_gold.data(), hcsr_col_ind.data());
        unit_check_general(1, nnz, 1, hcsr_val_gold.data(), hcsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXhyb2csr(handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(
                hipsparseXhyb2csr(handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        testhyb* dhyb = (testhyb*)hyb;

        double gbyte_count = hyb2csr_gbyte_count<T>(m, nnz, dhyb->ell_nnz, dhyb->coo_nnz);
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
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_HYB2CSR_HPP
