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
#ifndef TESTING_CSRGEMM_HPP
#define TESTING_CSRGEMM_HPP

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
void testing_csrgemm_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  M         = 1;
    int                  N         = 1;
    int                  K         = 1;
    int                  nnz_A     = 1;
    int                  nnz_B     = 1;
    hipsparseOperation_t trans_A   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t trans_B   = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    int                  safe_size = 1;

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

    std::vector<int> hcsr_row_ptr_C(M + 1);
    hcsr_row_ptr_C[0] = 0;
    hcsr_row_ptr_C[1] = 1;

    CHECK_HIP_ERROR(
        hipMemcpy(dCptr, hcsr_row_ptr_C.data(), sizeof(int) * (M + 1), hipMemcpyHostToDevice));

    // testing hipsparseXcsrgemmNnz
    int nnz_C;

    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: dAptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: dAcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: dBptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: dBcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 (int*)nullptr,
                                                                 &nnz_C),
                                            "Error: dCptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 (int*)nullptr),
                                            "Error: nnz_C is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: descr_A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 descr_C,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: descr_B is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemmNnz(handle,
                                                                 trans_A,
                                                                 trans_B,
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
                                                                 (hipsparseMatDescr_t) nullptr,
                                                                 dCptr,
                                                                 &nnz_C),
                                            "Error: descr_C is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrgemmNnz((hipsparseHandle_t) nullptr,
                                                                trans_A,
                                                                trans_B,
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
                                                                descr_C,
                                                                dCptr,
                                                                &nnz_C));
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dAval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dAptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dAcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dBval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dBptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dBcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              (T*)nullptr,
                                                              dCptr,
                                                              dCcol),
                                            "Error: dCval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              (int*)nullptr,
                                                              dCcol),
                                            "Error: dCptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              (int*)nullptr),
                                            "Error: dCcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: descr_A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              descr_C,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: descr_B is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXcsrgemm(handle,
                                                              trans_A,
                                                              trans_B,
                                                              M,
                                                              N,
                                                              K,
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
                                                              (hipsparseMatDescr_t) nullptr,
                                                              dCval,
                                                              dCptr,
                                                              dCcol),
                                            "Error: descr_C is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseXcsrgemm((hipsparseHandle_t) nullptr,
                                                             trans_A,
                                                             trans_B,
                                                             M,
                                                             N,
                                                             K,
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
                                                             descr_C,
                                                             dCval,
                                                             dCptr,
                                                             dCcol));
#endif
}

static int csrgemm_nnz(int                  m,
                       int                  n,
                       int                  k,
                       const int*           csr_row_ptr_A,
                       const int*           csr_col_ind_A,
                       const int*           csr_row_ptr_B,
                       const int*           csr_col_ind_B,
                       int*                 csr_row_ptr_C,
                       hipsparseIndexBase_t idx_base_A,
                       hipsparseIndexBase_t idx_base_B,
                       hipsparseIndexBase_t idx_base_C)
{
    std::vector<int> nnz(n, -1);

    // Index base
    csr_row_ptr_C[0] = idx_base_C;

    // Loop over rows of A
    for(int i = 0; i < m; ++i)
    {
        // Initialize csr row pointer with previous row offset
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i];

        int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
        int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

        // Loop over columns of A
        for(int j = row_begin_A; j < row_end_A; ++j)
        {
            // Current column of A
            int col_A = csr_col_ind_A[j] - idx_base_A;

            int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            // Loop over columns of B in row col_A
            for(int l = row_begin_B; l < row_end_B; ++l)
            {
                // Current column of B
                int col_B = csr_col_ind_B[l] - idx_base_B;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    ++csr_row_ptr_C[i + 1];
                }
            }
        }
    }

    return csr_row_ptr_C[m] - idx_base_C;
}

template <typename T>
static void csrgemm(int                  m,
                    int                  n,
                    int                  k,
                    const int*           csr_row_ptr_A,
                    const int*           csr_col_ind_A,
                    const T*             csr_val_A,
                    const int*           csr_row_ptr_B,
                    const int*           csr_col_ind_B,
                    const T*             csr_val_B,
                    const int*           csr_row_ptr_C,
                    int*                 csr_col_ind_C,
                    T*                   csr_val_C,
                    hipsparseIndexBase_t idx_base_A,
                    hipsparseIndexBase_t idx_base_B,
                    hipsparseIndexBase_t idx_base_C)
{
    std::vector<int> nnz(n, -1);

    // Loop over rows of A
    for(int i = 0; i < m; ++i)
    {
        int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
        int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

        int row_begin_C = csr_row_ptr_C[i] - idx_base_C;
        int row_end_C   = row_begin_C;

        // Loop over columns of A
        for(int j = row_begin_A; j < row_end_A; ++j)
        {
            // Current column of A
            int col_A = csr_col_ind_A[j] - idx_base_A;
            // Current value of A
            T val_A = csr_val_A[j];

            int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
            int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

            // Loop over columns of B in row col_A
            for(int l = row_begin_B; l < row_end_B; ++l)
            {
                // Current column of B
                int col_B = csr_col_ind_B[l] - idx_base_B;
                // Current value of B
                T val_B = csr_val_B[l];

                // Check if a new nnz is generated or if the product is appended
                if(nnz[col_B] < row_begin_C)
                {
                    nnz[col_B]               = row_end_C;
                    csr_col_ind_C[row_end_C] = col_B + idx_base_C;
                    csr_val_C[row_end_C]     = testing_mult(val_A, val_B);
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_B]] = csr_val_C[nnz[col_B]] + testing_mult(val_A, val_B);
                }
            }
        }
    }

    // Sort column indices within each row
    for(int i = 0; i < m; ++i)
    {
        int row_begin = csr_row_ptr_C[i] - idx_base_C;
        int row_end   = csr_row_ptr_C[i + 1] - idx_base_C;

        for(int j = row_begin; j < row_end; ++j)
        {
            for(int jj = row_begin; jj < row_end - 1; ++jj)
            {
                if(csr_col_ind_C[jj] > csr_col_ind_C[jj + 1])
                {
                    // swap elements
                    int ind = csr_col_ind_C[jj];
                    T   val = csr_val_C[jj];

                    csr_col_ind_C[jj] = csr_col_ind_C[jj + 1];
                    csr_val_C[jj]     = csr_val_C[jj + 1];

                    csr_col_ind_C[jj + 1] = ind;
                    csr_val_C[jj + 1]     = val;
                }
            }
        }
    }
}

template <typename T>
hipsparseStatus_t testing_csrgemm(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
    int                  M          = argus.M;
    int                  N          = argus.N;
    int                  K          = argus.K;
    hipsparseOperation_t trans_A    = argus.transA;
    hipsparseOperation_t trans_B    = argus.transB;
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
           filename, M, K, nnz_A, hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, idx_base_A))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // B = A^T so that we can compute the square of A
    N                      = M;
    int              nnz_B = nnz_A;
    std::vector<int> hcsr_row_ptr_B(K + 1, 0);
    std::vector<int> hcsr_col_ind_B(nnz_B);
    std::vector<T>   hcsr_val_B(nnz_B);

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
    auto dAptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};
    auto dAcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_A), device_free};
    auto dAval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_A), device_free};
    auto dBptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (K + 1)), device_free};
    auto dBcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz_B), device_free};
    auto dBval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz_B), device_free};
    auto dCptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};

    int* dAptr = (int*)dAptr_managed.get();
    int* dAcol = (int*)dAcol_managed.get();
    T*   dAval = (T*)dAval_managed.get();
    int* dBptr = (int*)dBptr_managed.get();
    int* dBcol = (int*)dBcol_managed.get();
    T*   dBval = (T*)dBval_managed.get();
    int* dCptr = (int*)dCptr_managed.get();

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

    // csrgemm nnz

    // hipsparse pointer mode host
    int hnnz_C_1;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemmNnz(handle,
                                               trans_A,
                                               trans_B,
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
                                               descr_C,
                                               dCptr,
                                               &hnnz_C_1));

    // Allocate result matrix
    auto dCcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnz_C_1), device_free};
    auto dCval_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * hnnz_C_1), device_free};

    int* dCcol = (int*)dCcol_managed.get();
    T*   dCval = (T*)dCval_managed.get();

    // hipsparse pointer mode device
    auto dnnz_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    int* dnnz_C         = (int*)dnnz_C_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemmNnz(handle,
                                               trans_A,
                                               trans_B,
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
                                               descr_C,
                                               dCptr,
                                               dnnz_C));

    // Copy output from device to CPU
    int hnnz_C_2;
    CHECK_HIP_ERROR(hipMemcpy(&hnnz_C_2, dnnz_C, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        // Compute csrgemm
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm(handle,
                                                trans_A,
                                                trans_B,
                                                M,
                                                N,
                                                K,
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
                                                descr_C,
                                                dCval,
                                                dCptr,
                                                dCcol));

        // Copy output from device to CPU
        std::vector<int> hcsr_row_ptr_C(M + 1);
        std::vector<int> hcsr_col_ind_C(hnnz_C_1);
        std::vector<T>   hcsr_val_C(hnnz_C_1);

        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_row_ptr_C.data(), dCptr, sizeof(int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_col_ind_C.data(), dCcol, sizeof(int) * hnnz_C_1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C.data(), dCval, sizeof(T) * hnnz_C_1, hipMemcpyDeviceToHost));

        // Compute csrgemm host solution
        std::vector<int> hcsr_row_ptr_C_gold(M + 1);

        int nnz_C_gold = csrgemm_nnz(M,
                                     N,
                                     K,
                                     hcsr_row_ptr_A.data(),
                                     hcsr_col_ind_A.data(),
                                     hcsr_row_ptr_B.data(),
                                     hcsr_col_ind_B.data(),
                                     hcsr_row_ptr_C_gold.data(),
                                     idx_base_A,
                                     idx_base_B,
                                     idx_base_C);

        std::vector<int> hcsr_col_ind_C_gold(nnz_C_gold);
        std::vector<T>   hcsr_val_C_gold(nnz_C_gold);

        csrgemm(M,
                N,
                K,
                hcsr_row_ptr_A.data(),
                hcsr_col_ind_A.data(),
                hcsr_val_A.data(),
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
        unit_check_general(1, M + 1, 1, hcsr_row_ptr_C_gold.data(), hcsr_row_ptr_C.data());
        unit_check_general(1, nnz_C_gold, 1, hcsr_col_ind_C_gold.data(), hcsr_col_ind_C.data());
        unit_check_near(1, nnz_C_gold, 1, hcsr_val_C_gold.data(), hcsr_val_C.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm(handle,
                                                    trans_A,
                                                    trans_B,
                                                    M,
                                                    N,
                                                    K,
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
                                                    descr_C,
                                                    dCval,
                                                    dCptr,
                                                    dCcol));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsrgemm(handle,
                                                    trans_A,
                                                    trans_B,
                                                    M,
                                                    N,
                                                    K,
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
                                                    descr_C,
                                                    dCval,
                                                    dCptr,
                                                    dCcol));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = csrgemm_gflop_count<T, int, int>(
            M, hcsr_row_ptr_A.data(), hcsr_col_ind_A.data(), hcsr_row_ptr_B.data(), idx_base_A);
        double gbyte_count = csrgemm_gbyte_count<T, int, int>(M, N, K, nnz_A, nnz_B, hnnz_C_1);

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::transA,
                            hipsparse_operation2string(trans_A),
                            display_key_t::transB,
                            hipsparse_operation2string(trans_B),
                            display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::K,
                            K,
                            display_key_t::nnzA,
                            nnz_A,
                            display_key_t::nnzB,
                            nnz_B,
                            display_key_t::nnzC,
                            hnnz_C_1,
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

#endif // TESTING_CSRGEMM_HPP
