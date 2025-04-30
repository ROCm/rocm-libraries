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
#ifndef TESTING_CSRGEAM_HPP
#define TESTING_CSRGEAM_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrgeam_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int safe_size = 1;

    T alpha = 1.0;
    T beta  = 1.0;

    int nnz_C;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    auto dAptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto dAcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dAval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dBptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto dBcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dBval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dCptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* dAptr = (int*)dAptr_managed.get();
    int* dAcol = (int*)dAcol_managed.get();
    T*   dAval = (T*)dAval_managed.get();
    int* dBptr = (int*)dBptr_managed.get();
    int* dBcol = (int*)dBcol_managed.get();
    T*   dBval = (T*)dBval_managed.get();
    int* dCptr = (int*)dCptr_managed.get();
    int* dCcol = (int*)dCcol_managed.get();
    T*   dCval = (T*)dCval_managed.get();

    const int        M = safe_size;
    std::vector<int> hcsr_row_ptr_C(M + 1);
    hcsr_row_ptr_C[0] = 0;
    hcsr_row_ptr_C[1] = 1;

    CHECK_HIP_ERROR(
        hipMemcpy(dCptr, hcsr_row_ptr_C.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));

    // testing hipsparseXcsrgeamNnz
    verify_hipsparse_status_invalid_handle(hipsparseXcsrgeamNnz(nullptr,
                                                                safe_size,
                                                                safe_size,
                                                                descr_A,
                                                                safe_size,
                                                                dAptr,
                                                                dAcol,
                                                                descr_B,
                                                                safe_size,
                                                                dBptr,
                                                                dBcol,
                                                                descr_C,
                                                                dCptr,
                                                                &nnz_C));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 nullptr,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid descr_A pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 nullptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid dAptr pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 nullptr,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid dAcol pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 nullptr,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid descr_B pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 nullptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid dBptr pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 nullptr,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid dBcol pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 nullptr,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: invalid descr_C pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 nullptr,
                                                                 &nnz_C),
                                            "Error: invalid dCptr pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeamNnz(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 descr_A,
                                                                 safe_size,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 safe_size,
                                                                 dBptr,
                                                                 dBcol,
                                                                 descr_C,
                                                                 dCptr,
                                                                 nullptr),
                                            "Error: invalid nnz_C pointer");

    // testing hipsparseXcsrgeam
    verify_hipsparse_status_invalid_handle(hipsparseXcsrgeam(nullptr,
                                                             safe_size,
                                                             safe_size,
                                                             &alpha,
                                                             descr_A,
                                                             safe_size,
                                                             dAval,
                                                             dAptr,
                                                             dAcol,
                                                             &beta,
                                                             descr_B,
                                                             safe_size,
                                                             dBval,
                                                             dBptr,
                                                             dBcol,
                                                             descr_C,
                                                             dCval,
                                                             dCptr,
                                                             dCcol));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              (T*)nullptr,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid alpha pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              nullptr,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid descr_A pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              (T*)nullptr,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dAval pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              nullptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dAptr pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              nullptr,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dAcol pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              (T*)nullptr,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid beta pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              nullptr,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid descr_B pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              (T*)nullptr,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dBval pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              nullptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dBptr pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              nullptr,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dBcol pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              nullptr,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid descr_C pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              (T*)nullptr,
                                                              dCptr,
                                                              dCcol),
                                            "Error: invalid dCval pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              nullptr,
                                                              dCcol),
                                            "Error: invalid dCptr pointer");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgeam(handle,
                                                              safe_size,
                                                              safe_size,
                                                              &alpha,
                                                              descr_A,
                                                              safe_size,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              &beta,
                                                              descr_B,
                                                              safe_size,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              nullptr),
                                            "Error: invalid dCcol pointer");

    // testing invalid sizes
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeamNnz(handle,
                                                              -1,
                                                              safe_size,
                                                              descr_A,
                                                              safe_size,
                                                              dAptr,
                                                              dAcol,
                                                              descr_B,
                                                              safe_size,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCptr,
                                                              &nnz_C),
                                         "Error: invalid M size");
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeamNnz(handle,
                                                              safe_size,
                                                              -1,
                                                              descr_A,
                                                              safe_size,
                                                              dAptr,
                                                              dAcol,
                                                              descr_B,
                                                              safe_size,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCptr,
                                                              &nnz_C),
                                         "Error: invalid N size");
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeamNnz(handle,
                                                              safe_size,
                                                              safe_size,
                                                              descr_A,
                                                              -1,
                                                              dAptr,
                                                              dAcol,
                                                              descr_B,
                                                              safe_size,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCptr,
                                                              &nnz_C),
                                         "Error: invalid nnz_A size");
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeamNnz(handle,
                                                              safe_size,
                                                              safe_size,
                                                              descr_A,
                                                              safe_size,
                                                              dAptr,
                                                              dAcol,
                                                              descr_B,
                                                              -1,
                                                              dBptr,
                                                              dBcol,
                                                              descr_C,
                                                              dCptr,
                                                              &nnz_C),
                                         "Error: invalid nnz_B size");

    verify_hipsparse_status_invalid_size(hipsparseXcsrgeam(handle,
                                                           -1,
                                                           safe_size,
                                                           &alpha,
                                                           descr_A,
                                                           safe_size,
                                                           dAval,
                                                           dAptr,
                                                           dAcol,
                                                           &beta,
                                                           descr_B,
                                                           safe_size,
                                                           dBval,
                                                           dBptr,
                                                           dBcol,
                                                           descr_C,
                                                           dCval,
                                                           dCptr,
                                                           dCcol),
                                         "Error: invalid M size");
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeam(handle,
                                                           safe_size,
                                                           -1,
                                                           &alpha,
                                                           descr_A,
                                                           safe_size,
                                                           dAval,
                                                           dAptr,
                                                           dAcol,
                                                           &beta,
                                                           descr_B,
                                                           safe_size,
                                                           dBval,
                                                           dBptr,
                                                           dBcol,
                                                           descr_C,
                                                           dCval,
                                                           dCptr,
                                                           dCcol),
                                         "Error: invalid N size");
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeam(handle,
                                                           safe_size,
                                                           safe_size,
                                                           &alpha,
                                                           descr_A,
                                                           -1,
                                                           dAval,
                                                           dAptr,
                                                           dAcol,
                                                           &beta,
                                                           descr_B,
                                                           safe_size,
                                                           dBval,
                                                           dBptr,
                                                           dBcol,
                                                           descr_C,
                                                           dCval,
                                                           dCptr,
                                                           dCcol),
                                         "Error: invalid nnz_A size");
    verify_hipsparse_status_invalid_size(hipsparseXcsrgeam(handle,
                                                           safe_size,
                                                           safe_size,
                                                           &alpha,
                                                           descr_A,
                                                           safe_size,
                                                           dAval,
                                                           dAptr,
                                                           dAcol,
                                                           &beta,
                                                           descr_B,
                                                           -1,
                                                           dBval,
                                                           dBptr,
                                                           dBcol,
                                                           descr_C,
                                                           dCval,
                                                           dCptr,
                                                           dCcol),
                                         "Error: invalid nnz_B size");
#endif
}

template <typename T>
hipsparseStatus_t testing_csrgeam(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int                  M          = argus.M;
    int                  N          = argus.N;
    T                    h_alpha    = make_DataType<T>(argus.alpha);
    T                    h_beta     = make_DataType<T>(argus.beta);
    hipsparseIndexBase_t idx_base_A = argus.baseA;
    hipsparseIndexBase_t idx_base_B = argus.baseB;
    hipsparseIndexBase_t idx_base_C = argus.baseC;
    std::string          filename   = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_A, idx_base_A));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_B, idx_base_B));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_C, idx_base_C));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    // Read or construct CSR matrix
    int nnz_A = 0;
    if(!generate_csr_matrix(
           filename, M, N, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // B = A
    int              nnz_B = nnz_A;
    std::vector<int> hcsr_row_ptr_B(M + 1, 0);
    std::vector<int> hcsr_col_ind_B(nnz_B);
    std::vector<T>   hcsr_val_B(nnz_B);

    for(int i = 0; i < M + 1; ++i)
    {
        hcsr_row_ptr_B[i] = hcsr_row_ptr_A[i] - idx_base_A + idx_base_B;
    }

    for(int i = 0; i < nnz_A; ++i)
    {
        hcsr_col_ind_B[i] = hcsr_col_ind_A[i] - idx_base_A + idx_base_B;
    }

    hcsr_val_B = hcsr_val_A;

    // Allocate memory on device
    auto dAptr_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dAcol_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto dAval_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dBptr_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dBcol_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_B), device_free};
    auto dBval_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dCptr_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dCptr_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};

    int* dAptr   = (int*)dAptr_managed.get();
    int* dAcol   = (int*)dAcol_managed.get();
    T*   dAval   = (T*)dAval_managed.get();
    int* dBptr   = (int*)dBptr_managed.get();
    int* dBcol   = (int*)dBcol_managed.get();
    T*   dBval   = (T*)dBval_managed.get();
    int* dCptr_1 = (int*)dCptr_1_managed.get();
    int* dCptr_2 = (int*)dCptr_2_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dAptr, hcsr_row_ptr_A.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dAcol, hcsr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dAval, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dBptr, hcsr_row_ptr_B.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dBcol, hcsr_col_ind_B.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dBval, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));

    // hipsparse pointer mode host
    int hnnz_C_1;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgeamNnz(handle,
                                               M,
                                               N,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               descr_C,
                                               dCptr_1,
                                               &hnnz_C_1));

    // Allocate result matrix
    auto dCcol_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C_1), device_free};
    auto dCval_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C_1), device_free};
    auto dCcol_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C_1), device_free};
    auto dCval_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C_1), device_free};

    int* dCcol_1 = (int*)dCcol_1_managed.get();
    T*   dCval_1 = (T*)dCval_1_managed.get();
    int* dCcol_2 = (int*)dCcol_2_managed.get();
    T*   dCval_2 = (T*)dCval_2_managed.get();

    // hipsparse pointer mode device
    auto dalpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto dbeta_managed  = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    T*   d_alpha = (T*)dalpha_managed.get();
    T*   d_beta  = (T*)dbeta_managed.get();
    int* dnnz_C  = (int*)dnnz_C_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgeamNnz(handle,
                                               M,
                                               N,
                                               descr_A,
                                               nnz_A,
                                               dAptr,
                                               dAcol,
                                               descr_B,
                                               nnz_B,
                                               dBptr,
                                               dBcol,
                                               descr_C,
                                               dCptr_2,
                                               dnnz_C));

    // Copy output from device to CPU
    int hnnz_C_2;
    CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        // Compute csrgeam
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgeam(handle,
                                                M,
                                                N,
                                                &h_alpha,
                                                descr_A,
                                                nnz_A,
                                                dAval,
                                                dAptr,
                                                dAcol,
                                                &h_beta,
                                                descr_B,
                                                nnz_B,
                                                dBval,
                                                dBptr,
                                                dBcol,
                                                descr_C,
                                                dCval_1,
                                                dCptr_1,
                                                dCcol_1));

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgeam(handle,
                                                M,
                                                N,
                                                d_alpha,
                                                descr_A,
                                                nnz_A,
                                                dAval,
                                                dAptr,
                                                dAcol,
                                                d_beta,
                                                descr_B,
                                                nnz_B,
                                                dBval,
                                                dBptr,
                                                dBcol,
                                                descr_C,
                                                dCval_2,
                                                dCptr_2,
                                                dCcol_2));

        // Copy output from device to CPU
        std::vector<int> hcsr_row_ptr_C_1(M + 1);
        std::vector<int> hcsr_col_ind_C_1(hnnz_C_1);
        std::vector<T>   hcsr_val_C_1(hnnz_C_1);
        std::vector<int> hcsr_row_ptr_C_2(M + 1);
        std::vector<int> hcsr_col_ind_C_2(hnnz_C_2);
        std::vector<T>   hcsr_val_C_2(hnnz_C_2);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C_1.data(), dCptr_1, sizeof(int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C_1.data(), dCcol_1, sizeof(int) * hnnz_C_1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1.data(), dCval_1, sizeof(T) * hnnz_C_1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr_C_2.data(), dCptr_2, sizeof(int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C_2.data(), dCcol_2, sizeof(int) * hnnz_C_2, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2.data(), dCval_2, sizeof(T) * hnnz_C_2, hipMemcpyDeviceToHost));

        // Compute csrgeam host solution
        std::vector<int> hcsr_row_ptr_C_gold(M + 1);

        int nnz_C_gold = host_csrgeam_nnz(M,
                                          N,
                                          h_alpha,
                                          hcsr_row_ptr_A.data(),
                                          hcsr_col_ind_A.data(),
                                          h_beta,
                                          hcsr_row_ptr_B.data(),
                                          hcsr_col_ind_B.data(),
                                          hcsr_row_ptr_C_gold.data(),
                                          idx_base_A,
                                          idx_base_B,
                                          idx_base_C);

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        host_csrgeam(M,
                     N,
                     h_alpha,
                     hcsr_row_ptr_A.data(),
                     hcsr_col_ind_A.data(),
                     hcsr_val_A.data(),
                     h_beta,
                     hcsr_row_ptr_B.data(),
                     hcsr_col_ind_B.data(),
                     hcsr_val_B.data(),
                     hcsr_row_ptr_C_gold.data(),
                     hcsr_col_ind_C_gold.data(),
                     hcsr_val_C_gold.data(),
                     idx_base_A,
                     idx_base_B,
                     idx_base_C);

        // Check nnz of C
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_2);

        // Check structure and entries of C
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C_1.data());
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C_2.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C_1.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C_2.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_1.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrgeam(handle,
                                                    M,
                                                    N,
                                                    &h_alpha,
                                                    descr_A,
                                                    nnz_A,
                                                    dAval,
                                                    dAptr,
                                                    dAcol,
                                                    &h_beta,
                                                    descr_B,
                                                    nnz_B,
                                                    dBval,
                                                    dBptr,
                                                    dBcol,
                                                    descr_C,
                                                    dCval_1,
                                                    dCptr_1,
                                                    dCcol_1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrgeam(handle,
                                                    M,
                                                    N,
                                                    &h_alpha,
                                                    descr_A,
                                                    nnz_A,
                                                    dAval,
                                                    dAptr,
                                                    dAcol,
                                                    &h_beta,
                                                    descr_B,
                                                    nnz_B,
                                                    dBval,
                                                    dBptr,
                                                    dBcol,
                                                    descr_C,
                                                    dCval_1,
                                                    dCptr_1,
                                                    dCcol_1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = csrgeam_gflop_count<T>(nnz_A, nnz_B, hnnz_C_1, &h_alpha, &h_beta);
        double gbyte_count = csrgeam_gbyte_count<T>(M, nnz_A, nnz_B, hnnz_C_1, &h_alpha, &h_beta);

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnzA,
                            nnz_A,
                            display_key_t::nnzB,
                            nnz_B,
                            display_key_t::nnzC,
                            hnnz_C_1,
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

#endif // TESTING_CSRGEAM_HPP
