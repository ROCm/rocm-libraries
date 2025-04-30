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
#ifndef TESTING_BSRMM_HPP
#define TESTING_BSRMM_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <iostream>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_bsrmm_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  mb        = 100;
    int                  n         = 100;
    int                  kb        = 100;
    int                  nnzb      = 100;
    int                  block_dim = 100;
    int                  ldb       = 100;
    int                  ldc       = 100;
    int                  safe_size = 100;
    T                    alpha     = 0.6;
    T                    beta      = 0.2;
    hipsparseDirection_t dirA      = HIPSPARSE_DIRECTION_ROW;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dB_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dC_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();
    T*   dB           = (T*)dB_managed.get();
    T*   dC           = (T*)dC_managed.get();

    verify_hipsparse_status_invalid_handle(hipsparseXbsrmm((hipsparseHandle_t) nullptr,
                                                           dirA,
                                                           transA,
                                                           transB,
                                                           mb,
                                                           n,
                                                           kb,
                                                           nnzb,
                                                           &alpha,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           dB,
                                                           ldb,
                                                           &beta,
                                                           dC,
                                                           ldc));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            (T*)nullptr,
                                                            descr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            &beta,
                                                            dC,
                                                            ldc),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            (hipsparseMatDescr_t) nullptr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            &beta,
                                                            dC,
                                                            ldc),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            descr,
                                                            (T*)nullptr,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            &beta,
                                                            dC,
                                                            ldc),
                                            "Error: dbsr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            descr,
                                                            dbsr_val,
                                                            (int*)nullptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            &beta,
                                                            dC,
                                                            ldc),
                                            "Error: dbsr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            descr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            (int*)nullptr,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            &beta,
                                                            dC,
                                                            ldc),
                                            "Error: dbsr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            descr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            (T*)nullptr,
                                                            ldb,
                                                            &beta,
                                                            dC,
                                                            ldc),
                                            "Error: dB is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            descr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            (T*)nullptr,
                                                            dC,
                                                            ldc),
                                            "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrmm(handle,
                                                            dirA,
                                                            transA,
                                                            transB,
                                                            mb,
                                                            n,
                                                            kb,
                                                            nnzb,
                                                            &alpha,
                                                            descr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            dB,
                                                            ldb,
                                                            &beta,
                                                            (T*)nullptr,
                                                            ldc),
                                            "Error: dC is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmm(handle,
                                                         dirA,
                                                         transA,
                                                         transB,
                                                         -1,
                                                         n,
                                                         kb,
                                                         nnzb,
                                                         &alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         block_dim,
                                                         dB,
                                                         ldb,
                                                         &beta,
                                                         dC,
                                                         ldc),
                                         "Error: mb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmm(handle,
                                                         dirA,
                                                         transA,
                                                         transB,
                                                         mb,
                                                         -1,
                                                         kb,
                                                         nnzb,
                                                         &alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         block_dim,
                                                         dB,
                                                         ldb,
                                                         &beta,
                                                         dC,
                                                         ldc),
                                         "Error: n is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmm(handle,
                                                         dirA,
                                                         transA,
                                                         transB,
                                                         mb,
                                                         n,
                                                         -1,
                                                         nnzb,
                                                         &alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         block_dim,
                                                         dB,
                                                         ldb,
                                                         &beta,
                                                         dC,
                                                         ldc),
                                         "Error: kb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrmm(handle,
                                                         dirA,
                                                         transA,
                                                         transB,
                                                         mb,
                                                         n,
                                                         kb,
                                                         nnzb,
                                                         &alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         0,
                                                         dB,
                                                         ldb,
                                                         &beta,
                                                         dC,
                                                         ldc),
                                         "Error: block_dim is invalid");

    // Test not implemented (mapped to hiparse internal error)
    verify_hipsparse_status_not_supported(hipsparseXbsrmm(handle,
                                                          dirA,
                                                          HIPSPARSE_OPERATION_TRANSPOSE,
                                                          transB,
                                                          mb,
                                                          n,
                                                          kb,
                                                          nnzb,
                                                          &alpha,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          block_dim,
                                                          dB,
                                                          ldb,
                                                          &beta,
                                                          dC,
                                                          ldc),
                                          "Error: Passed value for transA is not supported");
    verify_hipsparse_status_not_supported(hipsparseXbsrmm(handle,
                                                          dirA,
                                                          HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                                          transB,
                                                          mb,
                                                          n,
                                                          kb,
                                                          nnzb,
                                                          &alpha,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          block_dim,
                                                          dB,
                                                          ldb,
                                                          &beta,
                                                          dC,
                                                          ldc),
                                          "Error: Passed value for transA is not supported");
    verify_hipsparse_status_not_supported(hipsparseXbsrmm(handle,
                                                          dirA,
                                                          transA,
                                                          HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                                          mb,
                                                          n,
                                                          kb,
                                                          nnzb,
                                                          &alpha,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          block_dim,
                                                          dB,
                                                          ldb,
                                                          &beta,
                                                          dC,
                                                          ldc),
                                          "Error: Passed value for transB is not supported");
#endif
}

template <typename T>
hipsparseStatus_t testing_bsrmm(Arguments argus)
{
    int                  m         = argus.M;
    int                  n         = argus.N;
    int                  k         = argus.K;
    int                  block_dim = argus.block_dim;
    T                    h_alpha   = make_DataType<T>(argus.alpha);
    T                    h_beta    = make_DataType<T>(argus.beta);
    hipsparseDirection_t dirA      = argus.dirA;
    hipsparseOperation_t transA    = argus.transA;
    hipsparseOperation_t transB    = argus.transB;
    hipsparseIndexBase_t idx_base  = argus.baseA;
    std::string          filename  = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    if(m == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse does not support m == 0 for csr2bsr
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
    if(!generate_csr_matrix(filename, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // m and k can be modifed if we read in a matrix from a file
    int mb = (m + block_dim - 1) / block_dim;
    int kb = (k + block_dim - 1) / block_dim;

    // Allocate memory on device for CSR matrix and BSR row pointer array
    auto dcsr_row_ptrA_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_indA_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_valA_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptrA_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};

    int* dcsr_row_ptrA = (int*)dcsr_row_ptrA_managed.get();
    int* dcsr_col_indA = (int*)dcsr_col_indA_managed.get();
    T*   dcsr_valA     = (T*)dcsr_valA_managed.get();
    int* dbsr_row_ptrA = (int*)dbsr_row_ptrA_managed.get();

    // Copy CSR from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptrA, csr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_indA, csr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_valA, csr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR to BSR
    int nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               dirA,
                                               m,
                                               k,
                                               descr,
                                               dcsr_row_ptrA,
                                               dcsr_col_indA,
                                               block_dim,
                                               descr,
                                               dbsr_row_ptrA,
                                               &nnzb));

    auto dbsr_col_indA_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_valA_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};

    int* dbsr_col_indA = (int*)dbsr_col_indA_managed.get();
    T*   dbsr_valA     = (T*)dbsr_valA_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                            dirA,
                                            m,
                                            k,
                                            descr,
                                            dcsr_valA,
                                            dcsr_row_ptrA,
                                            dcsr_col_indA,
                                            block_dim,
                                            descr,
                                            dbsr_valA,
                                            dbsr_row_ptrA,
                                            dbsr_col_indA));

    // Host BSR matrix
    std::vector<int> hbsr_row_ptrA(mb + 1);
    std::vector<int> hbsr_col_indA(nnzb);
    std::vector<T>   hbsr_valA(nnzb * block_dim * block_dim);

    // Copy BSR matrix to host
    CHECK_HIP_ERROR(hipMemcpy(
        hbsr_row_ptrA.data(), dbsr_row_ptrA, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hbsr_col_indA.data(), dbsr_col_indA, sizeof(int) * nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_valA.data(),
                              dbsr_valA,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));

    m = mb * block_dim;
    k = kb * block_dim;

    // Some matrix properties
    int ldb = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
    int ldc = m;

    int ncol_B = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? n : k);
    int nnz_B  = ldb * ncol_B;
    int nnz_C  = ldc * n;

    // Allocate host memory for dense matrices
    std::vector<T> hB(nnz_B);
    std::vector<T> hC_1(nnz_C);
    std::vector<T> hC_2(nnz_C);
    std::vector<T> hC_gold(nnz_C);

    hipsparseInit<T>(hB, ldb, ncol_B);
    hipsparseInit<T>(hC_1, ldc, n);

    // copy vector is easy in STL; hC_gold = hC_1: save a copy in hy_gold which will be output of
    // CPU
    hC_gold = hC_1;
    hC_2    = hC_1;

    // allocate memory on device
    auto dB_managed      = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dC_1_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto dC_2_managed    = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_C), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dB      = (T*)dB_managed.get();
    T* dC_1    = (T*)dC_1_managed.get();
    T* dC_2    = (T*)dC_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();
    T* d_beta  = (T*)d_beta_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Testing using host pointer mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrmm(handle,
                                              dirA,
                                              transA,
                                              transB,
                                              mb,
                                              n,
                                              kb,
                                              nnzb,
                                              &h_alpha,
                                              descr,
                                              dbsr_valA,
                                              dbsr_row_ptrA,
                                              dbsr_col_indA,
                                              block_dim,
                                              dB,
                                              ldb,
                                              &h_beta,
                                              dC_1,
                                              ldc));

        // Testing using device pointer mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrmm(handle,
                                              dirA,
                                              transA,
                                              transB,
                                              mb,
                                              n,
                                              kb,
                                              nnzb,
                                              d_alpha,
                                              descr,
                                              dbsr_valA,
                                              dbsr_row_ptrA,
                                              dbsr_col_indA,
                                              block_dim,
                                              dB,
                                              ldb,
                                              d_beta,
                                              dC_2,
                                              ldc));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // Host bsrmm
        host_bsrmm<T>(mb,
                      n,
                      kb,
                      block_dim,
                      dirA,
                      transA,
                      transB,
                      h_alpha,
                      hbsr_row_ptrA,
                      hbsr_col_indA,
                      hbsr_valA,
                      hB,
                      ldb,
                      h_beta,
                      hC_gold,
                      ldc,
                      idx_base);

        // Unit check
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_1.data());
        unit_check_near(1, nnz_C, 1, hC_gold.data(), hC_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXbsrmm(handle,
                                                  dirA,
                                                  transA,
                                                  transB,
                                                  mb,
                                                  n,
                                                  kb,
                                                  nnzb,
                                                  &h_alpha,
                                                  descr,
                                                  dbsr_valA,
                                                  dbsr_row_ptrA,
                                                  dbsr_col_indA,
                                                  block_dim,
                                                  dB,
                                                  ldb,
                                                  &h_beta,
                                                  dC_1,
                                                  ldc));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXbsrmm(handle,
                                                  dirA,
                                                  transA,
                                                  transB,
                                                  mb,
                                                  n,
                                                  kb,
                                                  nnzb,
                                                  &h_alpha,
                                                  descr,
                                                  dbsr_valA,
                                                  dbsr_row_ptrA,
                                                  dbsr_col_indA,
                                                  block_dim,
                                                  dB,
                                                  ldb,
                                                  &h_beta,
                                                  dC_1,
                                                  ldc));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count
            = bsrmm_gflop_count(n, nnzb, block_dim, m * n, h_beta != make_DataType<T>(0.0));
        double gbyte_count = bsrmm_gbyte_count<T>(
            mb, nnzb, block_dim, k * n, m * n, h_beta != make_DataType<T>(0.0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::K,
                            k,
                            display_key_t::direction,
                            dirA,
                            display_key_t::transA,
                            transA,
                            display_key_t::transB,
                            transB,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::block_dim,
                            block_dim,
                            display_key_t::nnzB,
                            nnz_B,
                            display_key_t::nnzC,
                            nnz_C,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRMM_HPP
