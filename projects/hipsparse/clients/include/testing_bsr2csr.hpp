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
#ifndef TESTING_BSR2CSR_HPP
#define TESTING_BSR2CSR_HPP

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
void testing_bsr2csr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  m            = 1;
    int                  n            = 1;
    int                  safe_size    = 1;
    int                  block_dim    = 2;
    hipsparseIndexBase_t csr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t bsr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDirection_t dir          = HIPSPARSE_DIRECTION_ROW;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();
    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    int local_ptr[2] = {0, 1};
    CHECK_HIP_ERROR(
        hipMemcpy(bsr_row_ptr, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));

    verify_hipsparse_status_invalid_handle(hipsparseXbsr2csr((hipsparseHandle_t) nullptr,
                                                             dir,
                                                             m,
                                                             n,
                                                             bsr_descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             block_dim,
                                                             csr_descr,
                                                             csr_val,
                                                             csr_row_ptr,
                                                             csr_col_ind));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              (hipsparseMatDescr_t) nullptr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind),
                                            "Error: bsr_descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              (T*)nullptr,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind),
                                            "Error: bsr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              bsr_val,
                                                              (int*)nullptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind),
                                            "Error: bsr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              (int*)nullptr,
                                                              block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind),
                                            "Error: bsr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              (hipsparseMatDescr_t) nullptr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind),
                                            "Error: csr_descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              csr_descr,
                                                              (T*)nullptr,
                                                              csr_row_ptr,
                                                              csr_col_ind),
                                            "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              (int*)nullptr,
                                                              csr_col_ind),
                                            "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsr2csr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              (int*)nullptr),
                                            "Error: csr_col_ind is nullptr");

    verify_hipsparse_status_invalid_size(hipsparseXbsr2csr(handle,
                                                           dir,
                                                           -1,
                                                           n,
                                                           bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           block_dim,
                                                           csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind),
                                         "Error: m is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsr2csr(handle,
                                                           dir,
                                                           m,
                                                           -1,
                                                           bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           block_dim,
                                                           csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind),
                                         "Error: n is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsr2csr(handle,
                                                           dir,
                                                           m,
                                                           n,
                                                           bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           -1,
                                                           csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind),
                                         "Error: block_dim is invalid");
#endif
}

template <typename T>
hipsparseStatus_t testing_bsr2csr(Arguments argus)
{
    int                  m            = argus.M;
    int                  n            = argus.N;
    int                  block_dim    = argus.block_dim;
    hipsparseIndexBase_t bsr_idx_base = argus.baseA;
    hipsparseIndexBase_t csr_idx_base = argus.baseB;
    hipsparseDirection_t dir          = argus.dirA;
    std::string          filename     = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    if(m == 0 || n == 0 || block_dim == 1)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse does not support m == 0 or n == 0 for bsr2csr
        // cusparse does not support asynchronous execution if block_dim == 1
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<T>   csr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, csr_idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // m and n can be modifed if we read in a matrix from a file
    int mb = (m + block_dim - 1) / block_dim;
    int nb = (n + block_dim - 1) / block_dim;

    // Host BSR matrix
    std::vector<int> hbsr_row_ptr;
    std::vector<int> hbsr_col_ind;
    std::vector<T>   hbsr_val;

    // Convert CSR matrix to BSR
    int nnzb;
    host_csr_to_bsr<T>(dir,
                       m,
                       n,
                       block_dim,
                       nnzb,
                       csr_idx_base,
                       csr_row_ptr,
                       csr_col_ind,
                       csr_val,
                       bsr_idx_base,
                       hbsr_row_ptr,
                       hbsr_col_ind,
                       hbsr_val);

    // Determine the size of the output CSR matrix based on the size of the input BSR matrix
    m = mb * block_dim;
    n = nb * block_dim;

    // Host CSR matrix
    std::vector<int> hcsr_row_ptr(m + 1);
    std::vector<int> hcsr_col_ind(nnzb * block_dim * block_dim);
    std::vector<T>   hcsr_val(nnzb * block_dim * block_dim);

    // Allocate memory on the device
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(int) * nnzb * block_dim * block_dim), device_free};
    auto dcsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};

    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();
    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val,
                              hbsr_val.data(),
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXbsr2csr(handle,
                                                dir,
                                                mb,
                                                nb,
                                                bsr_descr,
                                                dbsr_val,
                                                dbsr_row_ptr,
                                                dbsr_col_ind,
                                                block_dim,
                                                csr_descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind.data(),
                                  dcsr_col_ind,
                                  sizeof(int) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val.data(),
                                  dcsr_val,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        // Host computed bsr2csr conversion
        std::vector<int> hcsr_row_ptr_gold(m + 1);
        std::vector<int> hcsr_col_ind_gold(nnzb * block_dim * block_dim, 0);
        std::vector<T>   hcsr_val_gold(nnzb * block_dim * block_dim);

        // Host bsr2csr
        host_bsr_to_csr<T>(dir,
                           mb,
                           nb,
                           block_dim,
                           bsr_idx_base,
                           hbsr_row_ptr,
                           hbsr_col_ind,
                           hbsr_val,
                           csr_idx_base,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold);

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
        unit_check_general(
            1, nnzb * block_dim * block_dim, 1, hcsr_col_ind_gold.data(), hcsr_col_ind.data());
        unit_check_general(
            1, nnzb * block_dim * block_dim, 1, hcsr_val_gold.data(), hcsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXbsr2csr(handle,
                                                    dir,
                                                    mb,
                                                    nb,
                                                    bsr_descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind,
                                                    block_dim,
                                                    csr_descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXbsr2csr(handle,
                                                    dir,
                                                    mb,
                                                    nb,
                                                    bsr_descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind,
                                                    block_dim,
                                                    csr_descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = bsr2csr_gbyte_count<T>(mb, block_dim, nnzb);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::Mb,
                            mb,
                            display_key_t::Nb,
                            nb,
                            display_key_t::block_dim,
                            block_dim,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSR2CSR_HPP
