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
#ifndef TESTING_CSR2HYB_HPP
#define TESTING_CSR2HYB_HPP

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

#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

template <typename T>
void testing_csr2hyb_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int m         = 100;
    int n         = 100;
    int safe_size = 100;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2hyb(handle,
                                                              m,
                                                              n,
                                                              descr,
                                                              csr_val,
                                                              (int*)nullptr,
                                                              csr_col_ind,
                                                              hyb,
                                                              0,
                                                              HIPSPARSE_HYB_PARTITION_AUTO),
                                            "Error: csr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2hyb(handle,
                                                              m,
                                                              n,
                                                              descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              (int*)nullptr,
                                                              hyb,
                                                              0,
                                                              HIPSPARSE_HYB_PARTITION_AUTO),
                                            "Error: csr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsr2hyb(handle,
                                                              m,
                                                              n,
                                                              descr,
                                                              (T*)nullptr,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              hyb,
                                                              0,
                                                              HIPSPARSE_HYB_PARTITION_AUTO),
                                            "Error: csr_val is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsr2hyb((hipsparseHandle_t) nullptr,
                                                             m,
                                                             n,
                                                             descr,
                                                             csr_val,
                                                             csr_row_ptr,
                                                             csr_col_ind,
                                                             hyb,
                                                             0,
                                                             HIPSPARSE_HYB_PARTITION_AUTO));
#endif
}

template <typename T>
hipsparseStatus_t testing_csr2hyb(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int                     m              = argus.M;
    int                     n              = argus.N;
    hipsparseIndexBase_t    idx_base       = argus.baseA;
    hipsparseHybPartition_t part           = argus.part;
    int                     user_ell_width = argus.ell_width;
    std::string             filename       = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    if(m == 0 || n == 0)
    {
        return HIPSPARSE_STATUS_SUCCESS;
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
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // User given ELL width check
    hipsparseStatus_t status;
    if(part == HIPSPARSE_HYB_PARTITION_USER)
    {
        // ELL width -33 means we take a reasonable pre-computed width
        if(user_ell_width == -33)
        {
            user_ell_width = nnz / m;
        }

        // Test invalid user_ell_width
        int max_allowed_ell_nnz_per_row = (2 * nnz - 1) / m + 1;
        if(user_ell_width < 0 || user_ell_width > max_allowed_ell_nnz_per_row)
        {
            status = hipsparseXcsr2hyb(handle,
                                       m,
                                       n,
                                       descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind,
                                       hyb,
                                       user_ell_width,
                                       part);

            verify_hipsparse_status_invalid_value(
                status, "Error: user_ell_width < 0 || user_ell_width > max_ell_width");

            return HIPSPARSE_STATUS_SUCCESS;
        }
    }

    // Max width check
    if(part == HIPSPARSE_HYB_PARTITION_MAX)
    {
        // Compute max ELL width
        int ell_max_width = 0;
        for(int i = 0; i < m; ++i)
        {
            ell_max_width = std::max(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i], ell_max_width);
        }

        int width_limit = (2 * nnz - 1) / m + 1;
        if(ell_max_width > width_limit)
        {
            status = hipsparseXcsr2hyb(handle,
                                       m,
                                       n,
                                       descr,
                                       dcsr_val,
                                       dcsr_row_ptr,
                                       dcsr_col_ind,
                                       hyb,
                                       user_ell_width,
                                       part);

            verify_hipsparse_status_invalid_value(status, "ell_max_width > width_limit");
            return HIPSPARSE_STATUS_SUCCESS;
        }
    }

    // Host structures for verification
    std::vector<int> hhyb_ell_col_ind_gold;
    std::vector<T>   hhyb_ell_val_gold;
    std::vector<int> hhyb_coo_row_ind_gold;
    std::vector<int> hhyb_coo_col_ind_gold;
    std::vector<T>   hhyb_coo_val_gold;

    // Host csr2hyb conversion
    int ell_width = 0;
    int ell_nnz   = 0;
    int coo_nnz   = 0;

    if(part == HIPSPARSE_HYB_PARTITION_AUTO || part == HIPSPARSE_HYB_PARTITION_USER)
    {
        if(part == HIPSPARSE_HYB_PARTITION_AUTO)
        {
            // ELL width is average nnz per row
            ell_width = (nnz - 1) / m + 1;
        }
        else
        {
            // User given ELL width
            ell_width = user_ell_width;
        }

        ell_nnz = ell_width * m;

        // Determine COO nnz
        for(int i = 0; i < m; ++i)
        {
            int row_nnz = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
            if(row_nnz > ell_width)
            {
                coo_nnz += row_nnz - ell_width;
            }
        }
    }
    else if(part == HIPSPARSE_HYB_PARTITION_MAX)
    {
        // Determine max nnz per row
        for(int i = 0; i < m; ++i)
        {
            int row_nnz = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i];
            ell_width   = (row_nnz > ell_width) ? row_nnz : ell_width;
        }
        ell_nnz = ell_width * m;
    }

    // Allocate host memory
    // ELL
    hhyb_ell_col_ind_gold.resize(ell_nnz);
    hhyb_ell_val_gold.resize(ell_nnz);
    // COO
    hhyb_coo_row_ind_gold.resize(coo_nnz);
    hhyb_coo_col_ind_gold.resize(coo_nnz);
    hhyb_coo_val_gold.resize(coo_nnz);

    // Fill HYB
    int coo_idx = 0;
    for(int i = 0; i < m; ++i)
    {
        int p = 0;
        for(int j = hcsr_row_ptr[i] - idx_base; j < hcsr_row_ptr[i + 1] - idx_base; ++j)
        {
            if(p < ell_width)
            {
                int idx                    = ELL_IND(i, p++, m, ell_width);
                hhyb_ell_col_ind_gold[idx] = hcsr_col_ind[j];
                hhyb_ell_val_gold[idx]     = hcsr_val[j];
            }
            else
            {
                hhyb_coo_row_ind_gold[coo_idx] = i + idx_base;
                hhyb_coo_col_ind_gold[coo_idx] = hcsr_col_ind[j];
                hhyb_coo_val_gold[coo_idx]     = hcsr_val[j];
                ++coo_idx;
            }
        }
        for(int j = hcsr_row_ptr[i + 1] - hcsr_row_ptr[i]; j < ell_width; ++j)
        {
            int idx                    = ELL_IND(i, p++, m, ell_width);
            hhyb_ell_col_ind_gold[idx] = -1;
            hhyb_ell_val_gold[idx]     = make_DataType<T>(0.0);
        }
    }

    // Allocate verification structures
    std::vector<int> hhyb_ell_col_ind(ell_nnz);
    std::vector<T>   hhyb_ell_val(ell_nnz);
    std::vector<int> hhyb_coo_row_ind(coo_nnz);
    std::vector<int> hhyb_coo_col_ind(coo_nnz);
    std::vector<T>   hhyb_coo_val(coo_nnz);

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2hyb(
            handle, m, n, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, hyb, user_ell_width, part));

        // Copy output from device to host
        testhyb* dhyb = (testhyb*)hyb;

        // Check if sizes match
        unit_check_general(1, 1, 1, &m, &dhyb->m);
        unit_check_general(1, 1, 1, &n, &dhyb->n);
        unit_check_general(1, 1, 1, &ell_width, &dhyb->ell_width);
        unit_check_general(1, 1, 1, &ell_nnz, &dhyb->ell_nnz);
        unit_check_general(1, 1, 1, &coo_nnz, &dhyb->coo_nnz);

        CHECK_HIP_ERROR(hipMemcpy(hhyb_ell_col_ind.data(),
                                  dhyb->ell_col_ind,
                                  sizeof(int) * ell_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hhyb_ell_val.data(), dhyb->ell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_row_ind.data(),
                                  dhyb->coo_row_ind,
                                  sizeof(int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_col_ind.data(),
                                  dhyb->coo_col_ind,
                                  sizeof(int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hhyb_coo_val.data(), dhyb->coo_val, sizeof(T) * coo_nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, ell_nnz, 1, hhyb_ell_col_ind_gold.data(), hhyb_ell_col_ind.data());
        unit_check_general(1, ell_nnz, 1, hhyb_ell_val_gold.data(), hhyb_ell_val.data());
        unit_check_general(1, coo_nnz, 1, hhyb_coo_row_ind_gold.data(), hhyb_coo_row_ind.data());
        unit_check_general(1, coo_nnz, 1, hhyb_coo_col_ind_gold.data(), hhyb_coo_col_ind.data());
        unit_check_general(1, coo_nnz, 1, hhyb_coo_val_gold.data(), hhyb_coo_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2hyb(handle,
                                                    m,
                                                    n,
                                                    descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    hyb,
                                                    user_ell_width,
                                                    part));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2hyb(handle,
                                                    m,
                                                    n,
                                                    descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    hyb,
                                                    user_ell_width,
                                                    part));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2hyb_gbyte_count<T>(m, nnz, ell_nnz, coo_nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::ell_nnz,
                            ell_nnz,
                            display_key_t::coo_nnz,
                            coo_nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2HYB_HPP
