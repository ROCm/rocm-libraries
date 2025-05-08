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
#ifndef TESTING_CSRGEMM2_B_HPP
#define TESTING_CSRGEMM2_B_HPP

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
void testing_csrgemm2_b_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int M         = 1;
    int N         = 1;
    int nnz_D     = 1;
    int safe_size = 1;

    T beta = 1.0;

    hipsparseStatus_t status;
    size_t            size;
    int               nnz_C;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_D(new descr_struct);
    hipsparseMatDescr_t           descr_D = unique_ptr_descr_D->descr;

    std::unique_ptr<csrgemm2_struct> unique_ptr_csrgemm2(new csrgemm2_struct);
    csrgemm2Info_t                   info = unique_ptr_csrgemm2->info;

    auto dDptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto dDcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dDval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dCptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dDptr   = (int*)dDptr_managed.get();
    int*  dDcol   = (int*)dDcol_managed.get();
    T*    dDval   = (T*)dDval_managed.get();
    int*  dCptr   = (int*)dCptr_managed.get();
    int*  dCcol   = (int*)dCcol_managed.get();
    T*    dCval   = (T*)dCval_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    std::vector<int> hcsr_row_ptr_C(M + 1);
    hcsr_row_ptr_C[0] = 0;
    hcsr_row_ptr_C[1] = 1;

    CHECK_HIP_ERROR(
        hipMemcpy(dCptr, hcsr_row_ptr_C.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));

    // Scenario: alpha == 0 and beta != 0

    // testing hipsparseXcsrgemm2_bufferSizeExt

    // testing for(nullptr == handle)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(nullptr,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  &beta,
                                                  descr_D,
                                                  nnz_D,
                                                  dDptr,
                                                  dDcol,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == beta)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  (T*)nullptr,
                                                  descr_D,
                                                  nnz_D,
                                                  dDptr,
                                                  dDcol,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == descr_D)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  &beta,
                                                  nullptr,
                                                  nnz_D,
                                                  dDptr,
                                                  dDcol,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_D is nullptr");
    }
    // testing for(nullptr == dDptr)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  &beta,
                                                  descr_D,
                                                  nnz_D,
                                                  nullptr,
                                                  dDcol,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDptr is nullptr");
    }
    // testing for(nullptr == dDcol)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  &beta,
                                                  descr_D,
                                                  nnz_D,
                                                  dDptr,
                                                  nullptr,
                                                  info,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  &beta,
                                                  descr_D,
                                                  nnz_D,
                                                  dDptr,
                                                  dDcol,
                                                  nullptr,
                                                  &size);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == size)
    {
        status = hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                  M,
                                                  N,
                                                  0,
                                                  (T*)nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  &beta,
                                                  descr_D,
                                                  nnz_D,
                                                  dDptr,
                                                  dDcol,
                                                  info,
                                                  nullptr);
        verify_hipsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }

    // testing hipsparseXcsrgemm2Nnz

    // testing for(nullptr == handle)
    {
        status = hipsparseXcsrgemm2Nnz(nullptr,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == descr_D)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_D is nullptr");
    }
    // testing for(nullptr == dDptr)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       nullptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDptr is nullptr");
    }
    // testing for(nullptr == dDcol)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       nullptr,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       nullptr,
                                       dCptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       nullptr,
                                       &nnz_C,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == nnz_C)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       nullptr,
                                       info,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: nnz_C is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = hipsparseXcsrgemm2Nnz(handle,
                                       M,
                                       N,
                                       0,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       descr_D,
                                       nnz_D,
                                       dDptr,
                                       dDcol,
                                       descr_C,
                                       dCptr,
                                       &nnz_C,
                                       nullptr,
                                       dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }

    // testing hipsparseXcsrgemm2

    // testing for(nullptr == handle)
    {
        status = hipsparseXcsrgemm2(nullptr,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_handle(status);
    }
    // testing for(nullptr == beta)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    (T*)nullptr,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == descr_D)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    nullptr,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_D is nullptr");
    }
    // testing for(nullptr == dDval)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    (T*)nullptr,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDval is nullptr");
    }
    // testing for(nullptr == dDptr)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    nullptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDptr is nullptr");
    }
    // testing for(nullptr == dDcol)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    nullptr,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dDcol is nullptr");
    }
    // testing for(nullptr == descr_C)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    nullptr,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");
    }
    // testing for(nullptr == dCval)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    (T*)nullptr,
                                    dCptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCval is nullptr");
    }
    // testing for(nullptr == dCptr)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    nullptr,
                                    dCcol,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCptr is nullptr");
    }
    // testing for(nullptr == dCcol)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    nullptr,
                                    info,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: dCcol is nullptr");
    }
    // testing for(nullptr == info)
    {
        status = hipsparseXcsrgemm2(handle,
                                    M,
                                    N,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    (T*)nullptr,
                                    nullptr,
                                    nullptr,
                                    &beta,
                                    descr_D,
                                    nnz_D,
                                    dDval,
                                    dDptr,
                                    dDcol,
                                    descr_C,
                                    dCval,
                                    dCptr,
                                    dCcol,
                                    nullptr,
                                    dbuffer);
        verify_hipsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
#endif
}

template <typename T>
hipsparseStatus_t testing_csrgemm2_b(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  M          = argus.M;
    int                  N          = argus.N;
    hipsparseIndexBase_t idx_base_C = argus.baseC;
    hipsparseIndexBase_t idx_base_D = argus.baseD;
    std::string          filename   = argus.filename;
    T                    beta       = make_DataType<T>(argus.beta);

    T* h_beta = &beta;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_D(new descr_struct);
    hipsparseMatDescr_t           descr_D = unique_ptr_descr_D->descr;

    std::unique_ptr<csrgemm2_struct> unique_ptr_csrgemm2(new csrgemm2_struct);
    csrgemm2Info_t                   info = unique_ptr_csrgemm2->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_C, idx_base_C));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_D, idx_base_D));

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr_D;
    std::vector<int> hcsr_col_ind_D;
    std::vector<T>   hcsr_val_D;

    // Read or construct CSR matrix
    int nnz_D = 0;
    if(!generate_csr_matrix(
           filename, M, N, nnz_D, hcsr_row_ptr_D, hcsr_col_ind_D, hcsr_val_D, idx_base_D))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;
    std::vector<int> hcsr_row_ptr_B;
    std::vector<int> hcsr_col_ind_B;
    std::vector<T>   hcsr_val_B;

    // Allocate memory on device
    int one        = 1;
    int safe_nnz_D = std::max(nnz_D, one);

    auto dDptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dDcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_nnz_D), device_free};
    auto dDval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_D), device_free};
    auto dCptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dbeta_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dDptr = (int*)dDptr_managed.get();
    int* dDcol = (int*)dDcol_managed.get();
    T*   dDval = (T*)dDval_managed.get();
    int* dCptr = (int*)dCptr_managed.get();
    T*   dbeta = (T*)dbeta_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dDptr, hcsr_row_ptr_D.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dDcol, hcsr_col_ind_D.data(), sizeof(int) * nnz_D, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dDval, hcsr_val_D.data(), sizeof(T) * nnz_D, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrgemm2 buffer size
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                           M,
                                                           N,
                                                           0,
                                                           (T*)nullptr,
                                                           descr_A,
                                                           0,
                                                           nullptr,
                                                           nullptr,
                                                           descr_B,
                                                           0,
                                                           nullptr,
                                                           nullptr,
                                                           h_beta,
                                                           descr_D,
                                                           nnz_D,
                                                           dDptr,
                                                           dDcol,
                                                           info,
                                                           &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    // csrgemm2 nnz

    // hipsparse pointer mode host
    int hnnz_C_1;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2Nnz(handle,
                                                M,
                                                N,
                                                0,
                                                descr_A,
                                                0,
                                                nullptr,
                                                nullptr,
                                                descr_B,
                                                0,
                                                nullptr,
                                                nullptr,
                                                descr_D,
                                                nnz_D,
                                                dDptr,
                                                dDcol,
                                                descr_C,
                                                dCptr,
                                                &hnnz_C_1,
                                                info,
                                                dbuffer));

    // Allocate result matrix
    int safe_nnz_C = std::max(hnnz_C_1, one);

    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_nnz_C), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_C), device_free};

    int* dCcol = (int*)dCcol_managed.get();
    T*   dCval = (T*)dCval_managed.get();

    if(argus.unit_check)
    {
        // hipsparse pointer mode device
        auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
        int* dnnz_C         = (int*)dnnz_C_managed.get();

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2Nnz(handle,
                                                    M,
                                                    N,
                                                    0,
                                                    descr_A,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    descr_B,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    descr_D,
                                                    nnz_D,
                                                    dDptr,
                                                    dDcol,
                                                    descr_C,
                                                    dCptr,
                                                    dnnz_C,
                                                    info,
                                                    dbuffer));

        // Compute csrgemm host solution
        std::vector<int> hcsr_row_ptr_C_gold(M + 1);

        double cpu_time_used = get_time_us();

        int nnz_C_gold = host_csrgemm2_nnz(M,
                                           N,
                                           0,
                                           (T*)nullptr,
                                           hcsr_row_ptr_A.data(),
                                           hcsr_col_ind_A.data(),
                                           hcsr_row_ptr_B.data(),
                                           hcsr_col_ind_B.data(),
                                           h_beta,
                                           hcsr_row_ptr_D.data(),
                                           hcsr_col_ind_D.data(),
                                           hcsr_row_ptr_C_gold.data(),
                                           HIPSPARSE_INDEX_BASE_ZERO,
                                           HIPSPARSE_INDEX_BASE_ZERO,
                                           idx_base_C,
                                           idx_base_D);

        // If nnz_C == 0, we are done
        if(nnz_C_gold == 0)
        {
            return HIPSPARSE_STATUS_SUCCESS;
        }

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        host_csrgemm2(M,
                      N,
                      0,
                      (T*)nullptr,
                      hcsr_row_ptr_A.data(),
                      hcsr_col_ind_A.data(),
                      hcsr_val_A.data(),
                      hcsr_row_ptr_B.data(),
                      hcsr_col_ind_B.data(),
                      hcsr_val_B.data(),
                      h_beta,
                      hcsr_row_ptr_D.data(),
                      hcsr_col_ind_D.data(),
                      hcsr_val_D.data(),
                      hcsr_row_ptr_C_gold.data(),
                      hcsr_col_ind_C_gold.data(),
                      hcsr_val_C_gold.data(),
                      HIPSPARSE_INDEX_BASE_ZERO,
                      HIPSPARSE_INDEX_BASE_ZERO,
                      idx_base_C,
                      idx_base_D);

        cpu_time_used = get_time_us() - cpu_time_used;

        // Copy output from device to CPU
        int hnnz_C_2;
        CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

        // Check nnz of C
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &nnz_C_gold, &hnnz_C_2);

        // Compute csrgemm2
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2(handle,
                                                 M,
                                                 N,
                                                 0,
                                                 (T*)nullptr,
                                                 descr_A,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descr_B,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 h_beta,
                                                 descr_D,
                                                 nnz_D,
                                                 dDval,
                                                 dDptr,
                                                 dDcol,
                                                 descr_C,
                                                 dCval,
                                                 dCptr,
                                                 dCcol,
                                                 info,
                                                 dbuffer));

        // Copy output from device to CPU
        std::vector<int> hcsr_row_ptr_C(M + 1);
        std::vector<int> hcsr_col_ind_C(nnz_C_gold);
        std::vector<T>   hcsr_val_C_1(nnz_C_gold);
        std::vector<T>   hcsr_val_C_2(nnz_C_gold);

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_row_ptr_C.data(), dCptr, sizeof(int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C.data(), dCcol, sizeof(int) * nnz_C_gold, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1.data(), dCval, sizeof(T) * nnz_C_gold, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemset(dCval, 0, sizeof(T) * nnz_C_gold));

        // Check structure and entries of C
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_1.data());

#ifdef __HIP_PLATFORM_AMD__
        // Device pointer mode
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2(handle,
                                                 M,
                                                 N,
                                                 0,
                                                 (T*)nullptr,
                                                 descr_A,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descr_B,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 dbeta,
                                                 descr_D,
                                                 nnz_D,
                                                 dDval,
                                                 dDptr,
                                                 dDcol,
                                                 descr_C,
                                                 dCval,
                                                 dCptr,
                                                 dCcol,
                                                 info,
                                                 dbuffer));

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2.data(), dCval, sizeof(T) * nnz_C_gold, hipMemcpyDeviceToHost));

        // Check device pointer results
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C_2.data());
#endif
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRGEMM2_B_HPP
