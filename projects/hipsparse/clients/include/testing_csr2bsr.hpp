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
#ifndef TESTING_CSR2BSR_HPP
#define TESTING_CSR2BSR_HPP

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
void testing_csr2bsr_bad_arg(void)
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

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();
    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();

    int local_ptr[2] = {0, 1};
    CHECK_HIP_ERROR(
        hipMemcpy(csr_row_ptr, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(bsr_row_ptr, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));

    int bsr_nnzb;

    verify_hipsparse_status_invalid_handle(hipsparseXcsr2bsrNnz(nullptr,
                                                                dir,
                                                                m,
                                                                n,
                                                                csr_descr,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                block_dim,
                                                                bsr_descr,
                                                                bsr_row_ptr,
                                                                &bsr_nnzb));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsrNnz(handle,
                                                                 dir,
                                                                 m,
                                                                 n,
                                                                 nullptr,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 block_dim,
                                                                 bsr_descr,
                                                                 bsr_row_ptr,
                                                                 &bsr_nnzb),
                                            "Error: csr_descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsrNnz(handle,
                                                                 dir,
                                                                 m,
                                                                 n,
                                                                 csr_descr,
                                                                 nullptr,
                                                                 csr_col_ind,
                                                                 block_dim,
                                                                 bsr_descr,
                                                                 bsr_row_ptr,
                                                                 &bsr_nnzb),
                                            "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsrNnz(handle,
                                                                 dir,
                                                                 m,
                                                                 n,
                                                                 csr_descr,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 block_dim,
                                                                 nullptr,
                                                                 bsr_row_ptr,
                                                                 &bsr_nnzb),
                                            "Error: bsr_descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsrNnz(handle,
                                                                 dir,
                                                                 m,
                                                                 n,
                                                                 csr_descr,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 block_dim,
                                                                 bsr_descr,
                                                                 nullptr,
                                                                 &bsr_nnzb),
                                            "Error: bsr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsrNnz(handle,
                                                                 dir,
                                                                 m,
                                                                 n,
                                                                 csr_descr,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 block_dim,
                                                                 bsr_descr,
                                                                 bsr_row_ptr,
                                                                 nullptr),
                                            "Error: bsr_nnzb is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2bsrNnz(handle,
                                                              dir,
                                                              -1,
                                                              n,
                                                              csr_descr,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_row_ptr,
                                                              &bsr_nnzb),
                                         "Error: m is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2bsrNnz(handle,
                                                              dir,
                                                              m,
                                                              -1,
                                                              csr_descr,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_row_ptr,
                                                              &bsr_nnzb),
                                         "Error: n is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2bsrNnz(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              -1,
                                                              bsr_descr,
                                                              bsr_row_ptr,
                                                              &bsr_nnzb),
                                         "Error: block_dim is invalid");

    verify_hipsparse_status_invalid_handle(hipsparseXcsr2bsr(nullptr,
                                                             dir,
                                                             m,
                                                             n,
                                                             csr_descr,
                                                             csr_val,
                                                             csr_row_ptr,
                                                             csr_col_ind,
                                                             block_dim,
                                                             bsr_descr,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              nullptr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind),
                                            "Error: csr_descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              (T*)nullptr,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind),
                                            "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              nullptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind),
                                            "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              nullptr,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind),
                                            "Error: csr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              nullptr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind),
                                            "Error: bsr_descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              (T*)nullptr,
                                                              bsr_row_ptr,
                                                              bsr_col_ind),
                                            "Error: bsr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              nullptr,
                                                              bsr_col_ind),
                                            "Error: bsr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2bsr(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              nullptr),
                                            "Error: bsr_col_ind is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2bsr(handle,
                                                           dir,
                                                           -1,
                                                           n,
                                                           csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           block_dim,
                                                           bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind),
                                         "Error: m is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2bsr(handle,
                                                           dir,
                                                           m,
                                                           -1,
                                                           csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           block_dim,
                                                           bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind),
                                         "Error: n is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXcsr2bsr(handle,
                                                           dir,
                                                           m,
                                                           n,
                                                           csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           -1,
                                                           bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind),
                                         "Error: block_dim is invalid");
#endif
}

template <typename T>
hipsparseStatus_t testing_csr2bsr(Arguments argus)
{
    int                  m            = argus.M;
    int                  n            = argus.N;
    int                  block_dim    = argus.block_dim;
    hipsparseIndexBase_t csr_idx_base = argus.baseA;
    hipsparseIndexBase_t bsr_idx_base = argus.baseB;
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
        // cusparse does not support m == 0 or n == 0 for csr2bsr
        // cusparse does not support asynchronous execution if block_dim == 1
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(
           filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, csr_idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    int mb = (m + block_dim - 1) / block_dim;
    int nb = (n + block_dim - 1) / block_dim;

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain BSR nnzb first on the host and then using the device and ensure they give the same results
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    int hbsr_nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               dir,
                                               m,
                                               n,
                                               csr_descr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               block_dim,
                                               bsr_descr,
                                               dbsr_row_ptr,
                                               &hbsr_nnzb));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

    auto dbsr_nnzb_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    int* dbsr_nnzb         = (int*)dbsr_nnzb_managed.get();
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               dir,
                                               m,
                                               n,
                                               csr_descr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               block_dim,
                                               bsr_descr,
                                               dbsr_row_ptr,
                                               dbsr_nnzb));

    int hbsr_nnzb_copied_from_device;
    CHECK_HIP_ERROR(
        hipMemcpy(&hbsr_nnzb_copied_from_device, dbsr_nnzb, sizeof(int), hipMemcpyDeviceToHost));

    // Check that using host and device pointer mode gives the same result
    unit_check_general(1, 1, 1, &hbsr_nnzb_copied_from_device, &hbsr_nnzb);

    // Allocate memory on the device
    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * hbsr_nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * hbsr_nnzb * block_dim * block_dim), device_free};

    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                                dir,
                                                m,
                                                n,
                                                csr_descr,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                block_dim,
                                                bsr_descr,
                                                dbsr_val,
                                                dbsr_row_ptr,
                                                dbsr_col_ind));

        // Copy output from device to host
        std::vector<int> hbsr_row_ptr(mb + 1);
        std::vector<int> hbsr_col_ind(hbsr_nnzb);
        std::vector<T>   hbsr_val(hbsr_nnzb * block_dim * block_dim);

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * hbsr_nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                                  dbsr_val,
                                  sizeof(T) * hbsr_nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        // Host csr2bsr conversion
        std::vector<int> hbsr_row_ptr_gold(mb + 1);
        std::vector<int> hbsr_col_ind_gold(hbsr_nnzb, 0);
        std::vector<T>   hbsr_val_gold(hbsr_nnzb * block_dim * block_dim);

        // call host csr2bsr here
        int bsr_nnzb_gold;
        host_csr_to_bsr<T>(dir,
                           m,
                           n,
                           block_dim,
                           bsr_nnzb_gold,
                           csr_idx_base,
                           hcsr_row_ptr,
                           hcsr_col_ind,
                           hcsr_val,
                           bsr_idx_base,
                           hbsr_row_ptr_gold,
                           hbsr_col_ind_gold,
                           hbsr_val_gold);

        // Unit check
        unit_check_general(1, 1, 1, &bsr_nnzb_gold, &hbsr_nnzb);
        unit_check_general(1, mb + 1, 1, hbsr_row_ptr_gold.data(), hbsr_row_ptr.data());
        unit_check_general(1, hbsr_nnzb, 1, hbsr_col_ind_gold.data(), hbsr_col_ind.data());
        unit_check_general(
            1, hbsr_nnzb * block_dim * block_dim, 1, hbsr_val_gold.data(), hbsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                                    dir,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                                    dir,
                                                    m,
                                                    n,
                                                    csr_descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2bsr_gbyte_count<T>(m, mb, nnz, hbsr_nnzb, block_dim);
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
                            hbsr_nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2BSR_HPP
