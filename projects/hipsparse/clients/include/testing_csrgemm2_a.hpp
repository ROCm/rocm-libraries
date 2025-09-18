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
#ifndef TESTING_CSRGEMM2_A_HPP
#define TESTING_CSRGEMM2_A_HPP

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
void testing_csrgemm2_a_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int M         = 1;
    int N         = 1;
    int K         = 1;
    int nnz_A     = 1;
    int nnz_B     = 1;
    int safe_size = 1;

    T alpha = 1.0;

    size_t size;
    int    nnz_C;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t           descr_A = unique_ptr_descr_A->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_B(new descr_struct);
    hipsparseMatDescr_t           descr_B = unique_ptr_descr_B->descr;

    std::unique_ptr<descr_struct> unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t           descr_C = unique_ptr_descr_C->descr;

    std::unique_ptr<csrgemm2_struct> unique_ptr_csrgemm2(new csrgemm2_struct);
    csrgemm2Info_t                   info = unique_ptr_csrgemm2->info;

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
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*  dAptr   = (int*)dAptr_managed.get();
    int*  dAcol   = (int*)dAcol_managed.get();
    T*    dAval   = (T*)dAval_managed.get();
    int*  dBptr   = (int*)dBptr_managed.get();
    int*  dBcol   = (int*)dBcol_managed.get();
    T*    dBval   = (T*)dBval_managed.get();
    int*  dCptr   = (int*)dCptr_managed.get();
    int*  dCcol   = (int*)dCcol_managed.get();
    T*    dCval   = (T*)dCval_managed.get();
    void* dbuffer = (void*)dbuffer_managed.get();

    std::vector<int> hcsr_row_ptr_C(M + 1);
    hcsr_row_ptr_C[0] = 0;
    hcsr_row_ptr_C[1] = 1;

    CHECK_HIP_ERROR(
        hipMemcpy(dCptr, hcsr_row_ptr_C.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));

    // Scenario: alpha != 0 and beta == 0
    verify_hipsparse_status_invalid_handle(
        hipsparseXcsrgemm2_bufferSizeExt((hipsparseHandle_t) nullptr,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size));
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         (T*)nullptr,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         (hipsparseMatDescr_t) nullptr,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: descr_A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         (int*)nullptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: dAptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         (int*)nullptr,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: dAcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         (hipsparseMatDescr_t) nullptr,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: descr_B is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         (int*)nullptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: dBptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         (int*)nullptr,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         &size),
        "Error: dBcol is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         (csrgemm2Info_t) nullptr,
                                         &size),
        "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseXcsrgemm2_bufferSizeExt(handle,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         descr_A,
                                         nnz_A,
                                         dAptr,
                                         dAcol,
                                         descr_B,
                                         nnz_B,
                                         dBptr,
                                         dBcol,
                                         (T*)nullptr,
                                         (hipsparseMatDescr_t) nullptr,
                                         0,
                                         (int*)nullptr,
                                         (int*)nullptr,
                                         info,
                                         (size_t*)nullptr),
        "Error: size is nullptr");

    verify_hipsparse_status_invalid_handle(hipsparseXcsrgemm2Nnz((hipsparseHandle_t) nullptr,
                                                                 M,
                                                                 N,
                                                                 K,
                                                                 descr_A,
                                                                 nnz_A,
                                                                 dAptr,
                                                                 dAcol,
                                                                 descr_B,
                                                                 nnz_B,
                                                                 dBptr,
                                                                 dBcol,
                                                                 nullptr,
                                                                 0,
                                                                 nullptr,
                                                                 nullptr,
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C,
                                                                 info,
                                                                 dbuffer));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  (hipsparseMatDescr_t) nullptr,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: descr_A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  (int*)nullptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: dAptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  (int*)nullptr,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: dAcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  (hipsparseMatDescr_t) nullptr,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: descr_B is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  (int*)nullptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: dBptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  (int*)nullptr,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: dBcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  (hipsparseMatDescr_t) nullptr,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: descr_C is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  (int*)nullptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  dbuffer),
                                            "Error: dCptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  (int*)nullptr,
                                                                  info,
                                                                  dbuffer),
                                            "Error: nnz_C is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  (csrgemm2Info_t) nullptr,
                                                                  dbuffer),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2Nnz(handle,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  dAptr,
                                                                  dAcol,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  dBptr,
                                                                  dBcol,
                                                                  nullptr,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  descr_C,
                                                                  dCptr,
                                                                  &nnz_C,
                                                                  info,
                                                                  (void*)nullptr),
                                            "Error: dbuffer is nullptr");

    verify_hipsparse_status_invalid_handle(hipsparseXcsrgemm2((hipsparseHandle_t) nullptr,
                                                              M,
                                                              N,
                                                              K,
                                                              &alpha,
                                                              descr_A,
                                                              nnz_A,
                                                              dAval,
                                                              dAptr,
                                                              dAcol,
                                                              descr_B,
                                                              nnz_B,
                                                              dBval,
                                                              dBptr,
                                                              dBcol,
                                                              (T*)nullptr,
                                                              (hipsparseMatDescr_t) nullptr,
                                                              0,
                                                              (T*)nullptr,
                                                              (int*)nullptr,
                                                              (int*)nullptr,
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol,
                                                              info,
                                                              dbuffer));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               (T*)nullptr,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: alpha is nullptr");

    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: descr_A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               (T*)nullptr,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dAval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               (int*)nullptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dAptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               (int*)nullptr,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dAcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: descr_B is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               (T*)nullptr,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dBval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               (int*)nullptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dBptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               (int*)nullptr,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dBcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: descr_C is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               (T*)nullptr,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dCval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               (int*)nullptr,
                                                               dCcol,
                                                               info,
                                                               dbuffer),
                                            "Error: dCptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               (int*)nullptr,
                                                               info,
                                                               dbuffer),
                                            "Error: dCcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               (csrgemm2Info_t) nullptr,
                                                               dbuffer),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm2(handle,
                                                               M,
                                                               N,
                                                               K,
                                                               &alpha,
                                                               descr_A,
                                                               nnz_A,
                                                               dAval,
                                                               dAptr,
                                                               dAcol,
                                                               descr_B,
                                                               nnz_B,
                                                               dBval,
                                                               dBptr,
                                                               dBcol,
                                                               (T*)nullptr,
                                                               (hipsparseMatDescr_t) nullptr,
                                                               0,
                                                               (T*)nullptr,
                                                               (int*)nullptr,
                                                               (int*)nullptr,
                                                               descr_C,
                                                               dCval,
                                                               dCptr,
                                                               dCcol,
                                                               info,
                                                               (void*)nullptr),
                                            "Error: dbuffer is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_csrgemm2_a(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    int                  M          = argus.M;
    int                  N          = argus.N;
    int                  K          = argus.K;
    hipsparseIndexBase_t idx_base_A = argus.baseA;
    hipsparseIndexBase_t idx_base_B = argus.baseB;
    hipsparseIndexBase_t idx_base_C = argus.baseC;
    std::string          filename   = argus.filename;
    T                    alpha      = make_DataType<T>(argus.alpha);

    T* h_alpha = &alpha;

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
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_A, idx_base_A));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_B, idx_base_B));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr_C, idx_base_C));

    if(M == 0 || N == 0 || K == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr_A;
    std::vector<int> hcsr_col_ind_A;
    std::vector<T>   hcsr_val_A;

    // Read or construct CSR matrix
    int nnz_A = 0;
    if(!generate_csr_matrix(
           filename, M, K, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    std::vector<int> hcsr_row_ptr_B;
    std::vector<int> hcsr_col_ind_B;
    std::vector<T>   hcsr_val_B;
    std::vector<int> hcsr_row_ptr_D;
    std::vector<int> hcsr_col_ind_D;
    std::vector<T>   hcsr_val_D;

    // B = A^T so that we can compute the square of A
    N         = M;
    int nnz_B = nnz_A;

    hcsr_row_ptr_B.resize(K + 1, 0);
    hcsr_col_ind_B.resize(nnz_B);
    hcsr_val_B.resize(nnz_B);

    // B = A^T
    transpose_csr(M,
                  K,
                  nnz_A,
                  hcsr_row_ptr_A.data(),
                  hcsr_col_ind_A.data(),
                  hcsr_val_A.data(),
                  hcsr_row_ptr_B.data(),
                  hcsr_col_ind_B.data(),
                  hcsr_val_B.data(),
                  idx_base_A,
                  idx_base_B);

    // Allocate memory on device
    int one        = 1;
    int safe_K     = std::max(K, one);
    int safe_nnz_A = std::max(nnz_A, one);
    int safe_nnz_B = std::max(nnz_B, one);

    auto dAptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dAcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_nnz_A), device_free};
    auto dAval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_A), device_free};
    auto dBptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_K + 1)), device_free};
    auto dBcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_nnz_B), device_free};
    auto dBval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_nnz_B), device_free};
    auto dCptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dalpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dAptr  = (int*)dAptr_managed.get();
    int* dAcol  = (int*)dAcol_managed.get();
    T*   dAval  = (T*)dAval_managed.get();
    int* dBptr  = (int*)dBptr_managed.get();
    int* dBcol  = (int*)dBcol_managed.get();
    T*   dBval  = (T*)dBval_managed.get();
    int* dCptr  = (int*)dCptr_managed.get();
    T*   dalpha = (T*)dalpha_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dAptr, hcsr_row_ptr_A.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dAcol, hcsr_col_ind_A.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dAval, hcsr_val_A.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dBptr, hcsr_row_ptr_B.data(), sizeof(int) * (K + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dBcol, hcsr_col_ind_B.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dBval, hcsr_val_B.data(), sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Obtain csrgemm2 buffer size
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm2_bufferSizeExt(handle,
                                                           M,
                                                           N,
                                                           K,
                                                           h_alpha,
                                                           descr_A,
                                                           nnz_A,
                                                           dAptr,
                                                           dAcol,
                                                           descr_B,
                                                           nnz_B,
                                                           dBptr,
                                                           dBcol,
                                                           (T*)nullptr,
                                                           descr_D,
                                                           0,
                                                           nullptr,
                                                           nullptr,
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
                                                K,
                                                descr_A,
                                                nnz_A,
                                                dAptr,
                                                dAcol,
                                                descr_B,
                                                nnz_B,
                                                dBptr,
                                                dBcol,
                                                descr_D,
                                                0,
                                                nullptr,
                                                nullptr,
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
                                                    K,
                                                    descr_A,
                                                    nnz_A,
                                                    dAptr,
                                                    dAcol,
                                                    descr_B,
                                                    nnz_B,
                                                    dBptr,
                                                    dBcol,
                                                    descr_D,
                                                    0,
                                                    nullptr,
                                                    nullptr,
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
                                           K,
                                           h_alpha,
                                           hcsr_row_ptr_A.data(),
                                           hcsr_col_ind_A.data(),
                                           hcsr_row_ptr_B.data(),
                                           hcsr_col_ind_B.data(),
                                           (T*)nullptr,
                                           hcsr_row_ptr_D.data(),
                                           hcsr_col_ind_D.data(),
                                           hcsr_row_ptr_C_gold.data(),
                                           idx_base_A,
                                           idx_base_B,
                                           idx_base_C,
                                           HIPSPARSE_INDEX_BASE_ZERO);

        // If nnz_C == 0, we are done
        if(nnz_C_gold == 0)
        {
            return HIPSPARSE_STATUS_SUCCESS;
        }

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        host_csrgemm2(M,
                      N,
                      K,
                      h_alpha,
                      hcsr_row_ptr_A.data(),
                      hcsr_col_ind_A.data(),
                      hcsr_val_A.data(),
                      hcsr_row_ptr_B.data(),
                      hcsr_col_ind_B.data(),
                      hcsr_val_B.data(),
                      (T*)nullptr,
                      hcsr_row_ptr_D.data(),
                      hcsr_col_ind_D.data(),
                      hcsr_val_D.data(),
                      hcsr_row_ptr_C_gold.data(),
                      hcsr_col_ind_C_gold.data(),
                      hcsr_val_C_gold.data(),
                      idx_base_A,
                      idx_base_B,
                      idx_base_C,
                      HIPSPARSE_INDEX_BASE_ZERO);

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
                                                 K,
                                                 h_alpha,
                                                 descr_A,
                                                 nnz_A,
                                                 dAval,
                                                 dAptr,
                                                 dAcol,
                                                 descr_B,
                                                 nnz_B,
                                                 dBval,
                                                 dBptr,
                                                 dBcol,
                                                 (T*)nullptr,
                                                 descr_D,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
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
                                                 K,
                                                 dalpha,
                                                 descr_A,
                                                 nnz_A,
                                                 dAval,
                                                 dAptr,
                                                 dAcol,
                                                 descr_B,
                                                 nnz_B,
                                                 dBval,
                                                 dBptr,
                                                 dBcol,
                                                 (T*)nullptr,
                                                 descr_D,
                                                 0,
                                                 (T*)nullptr,
                                                 nullptr,
                                                 nullptr,
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

#endif // TESTING_CSRGEMM2_A_HPP
