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
#ifndef TESTING_CSRMV_HPP
#define TESTING_CSRMV_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <cmath>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrmv_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  n         = 100;
    int                  m         = 100;
    int                  nnz       = 100;
    int                  safe_size = 100;
    T                    alpha     = 0.6;
    T                    beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* dptr = (int*)dptr_managed.get();
    int* dcol = (int*)dcol_managed.get();
    T*   dval = (T*)dval_managed.get();
    T*   dx   = (T*)dx_managed.get();
    T*   dy   = (T*)dy_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, (int*)nullptr, dcol, dx, &beta, dy),
        "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, (int*)nullptr, dx, &beta, dy),
        "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, (T*)nullptr, dptr, dcol, dx, &beta, dy),
        "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, (T*)nullptr, &beta, dy),
        "Error: dx is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, dx, &beta, (T*)nullptr),
        "Error: dy is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, (T*)nullptr, descr, dval, dptr, dcol, dx, &beta, dy),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrmv(
            handle, transA, m, n, nnz, &alpha, descr, dval, dptr, dcol, dx, (T*)nullptr, dy),
        "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrmv(handle,
                                                            transA,
                                                            m,
                                                            n,
                                                            nnz,
                                                            &alpha,
                                                            (hipsparseMatDescr_t) nullptr,
                                                            dval,
                                                            dptr,
                                                            dcol,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrmv((hipsparseHandle_t) nullptr,
                                                           transA,
                                                           m,
                                                           n,
                                                           nnz,
                                                           &alpha,
                                                           descr,
                                                           dval,
                                                           dptr,
                                                           dcol,
                                                           dx,
                                                           &beta,
                                                           dy));
#endif
}

template <typename T>
hipsparseStatus_t testing_csrmv(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int                  nrow     = argus.M;
    int                  ncol     = argus.N;
    T                    h_alpha  = make_DataType<T>(argus.alpha);
    T                    h_beta   = make_DataType<T>(argus.beta);
    hipsparseOperation_t transA   = argus.transA;
    hipsparseIndexBase_t idx_base = argus.baseA;
    std::string          filename = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(
           filename, nrow, ncol, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    int m = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? nrow : ncol;
    int n = (transA == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? ncol : nrow;

    std::vector<T> hx(n);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    hipsparseInit<T>(hx, 1, n);
    hipsparseInit<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (nrow + 1)), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * n), device_free};
    auto dy_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dptr    = (int*)dptr_managed.get();
    int* dcol    = (int*)dcol_managed.get();
    T*   dval    = (T*)dval_managed.get();
    T*   dx      = (T*)dx_managed.get();
    T*   dy_1    = (T*)dy_1_managed.get();
    T*   dy_2    = (T*)dy_2_managed.get();
    T*   d_alpha = (T*)d_alpha_managed.get();
    T*   d_beta  = (T*)d_beta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dptr, hcsr_row_ptr.data(), sizeof(int) * (nrow + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrmv(
            handle, transA, nrow, ncol, nnz, &h_alpha, descr, dval, dptr, dcol, dx, &h_beta, dy_1));

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrmv(
            handle, transA, nrow, ncol, nnz, d_alpha, descr, dval, dptr, dcol, dx, d_beta, dy_2));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        host_csrmv(transA,
                   nrow,
                   ncol,
                   nnz,
                   h_alpha,
                   hcsr_row_ptr.data(),
                   hcsr_col_ind.data(),
                   hcsr_val.data(),
                   hx.data(),
                   h_beta,
                   hy_gold.data(),
                   idx_base);

        unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
        unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrmv(handle,
                                                  transA,
                                                  nrow,
                                                  ncol,
                                                  nnz,
                                                  &h_alpha,
                                                  descr,
                                                  dval,
                                                  dptr,
                                                  dcol,
                                                  dx,
                                                  &h_beta,
                                                  dy_1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrmv(handle,
                                                  transA,
                                                  nrow,
                                                  ncol,
                                                  nnz,
                                                  &h_alpha,
                                                  descr,
                                                  dval,
                                                  dptr,
                                                  dcol,
                                                  dx,
                                                  &h_beta,
                                                  dy_1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(nrow, nnz, h_beta != make_DataType<T>(0.0));
        double gbyte_count = csrmv_gbyte_count<T>(nrow, ncol, nnz, h_beta != make_DataType<T>(0.0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            nrow,
                            display_key_t::N,
                            ncol,
                            display_key_t::nnz,
                            nnz,
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
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRMV_HPP
