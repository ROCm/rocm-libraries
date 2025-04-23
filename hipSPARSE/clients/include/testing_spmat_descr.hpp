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
#ifndef TESTING_SPMAT_DESCR_HPP
#define TESTING_SPMAT_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hipsparse.h>

#include "hipsparse_arguments.hpp"
#include "utility.hpp"

#include <iostream>

using namespace hipsparse_test;

void testing_spmat_descr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t rows          = 100;
    int64_t cols          = 100;
    int64_t nnz           = 100;
    int64_t ell_cols      = 10;
    int64_t ell_blocksize = 2;

    hipsparseIndexType_t rowType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t colType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t cooType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;
    hipsparseFormat_t    format   = HIPSPARSE_FORMAT_CSR;

    // Allocate memory on device
    auto row_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto col_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto ind_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * 2 * nnz), device_free};
    auto val_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};

    int*   row_data = (int*)row_data_managed.get();
    int*   col_data = (int*)col_data_managed.get();
    int*   ind_data = (int*)ind_data_managed.get();
    float* val_data = (float*)val_data_managed.get();

    hipsparseSpMatDescr_t A;

    // hipsparseCreateCoo
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            nullptr, rows, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCoo(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCoo(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCoo(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCoo(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateCooAoS
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCooAoS(
            nullptr, rows, cols, nnz, ind_data, val_data, cooType, idxBase, dataType),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCooAoS(&A, -1, cols, nnz, ind_data, val_data, cooType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCooAoS(&A, rows, -1, nnz, ind_data, val_data, cooType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCooAoS(&A, rows, cols, -1, ind_data, val_data, cooType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCooAoS(&A, rows, cols, nnz, nullptr, val_data, cooType, idxBase, dataType),
        "Error: ind_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCooAoS(&A, rows, cols, nnz, ind_data, nullptr, cooType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateCsr
    verify_hipsparse_status_invalid_pointer(hipsparseCreateCsr(nullptr,
                                                               rows,
                                                               cols,
                                                               nnz,
                                                               row_data,
                                                               col_data,
                                                               val_data,
                                                               rowType,
                                                               colType,
                                                               idxBase,
                                                               dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCsr(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCsr(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateCsr(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCsr(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCsr(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, colType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateCsr(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, colType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateBlockedEll
    verify_hipsparse_status_invalid_pointer(hipsparseCreateBlockedEll(nullptr,
                                                                      rows,
                                                                      cols,
                                                                      ell_blocksize,
                                                                      ell_cols,
                                                                      col_data,
                                                                      val_data,
                                                                      colType,
                                                                      idxBase,
                                                                      dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateBlockedEll(
            &A, -1, cols, ell_blocksize, ell_cols, col_data, val_data, colType, idxBase, dataType),
        "Error: rows is < 0");

    verify_hipsparse_status_invalid_size(
        hipsparseCreateBlockedEll(
            &A, rows, -1, ell_blocksize, ell_cols, col_data, val_data, colType, idxBase, dataType),
        "Error: cols is < 0");

    verify_hipsparse_status_invalid_size(
        hipsparseCreateBlockedEll(
            &A, rows, cols, -1, ell_cols, col_data, val_data, colType, idxBase, dataType),
        "Error: ell_blocksize is < 0");

    verify_hipsparse_status_invalid_size(
        hipsparseCreateBlockedEll(
            &A, rows, cols, ell_blocksize, -1, col_data, val_data, colType, idxBase, dataType),
        "Error: ell_cols is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateBlockedEll(
            &A, rows, cols, ell_blocksize, ell_cols, nullptr, val_data, colType, idxBase, dataType),
        "Error: ellColInd is nullptr");

    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateBlockedEll(
            &A, rows, cols, ell_blocksize, ell_cols, col_data, nullptr, colType, idxBase, dataType),
        "Error: ellValue is nullptr");

    // hipsparseDestroySpMat
    verify_hipsparse_status_invalid_pointer(hipsparseDestroySpMat(nullptr), "Error: A is nullptr");

    // Create valid descriptors
    hipsparseSpMatDescr_t coo;
    hipsparseSpMatDescr_t coo_aos;
    hipsparseSpMatDescr_t csr;
    hipsparseSpMatDescr_t csc;

    hipsparseSpMatDescr_t bell;
    verify_hipsparse_status_success(hipsparseCreateBlockedEll(&bell,
                                                              rows,
                                                              cols,
                                                              ell_blocksize,
                                                              ell_cols,
                                                              col_data,
                                                              val_data,
                                                              colType,
                                                              idxBase,
                                                              dataType),
                                    "Success");
    verify_hipsparse_status_success(
        hipsparseCreateCoo(
            &coo, rows, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Success");
    verify_hipsparse_status_success(
        hipsparseCreateCooAoS(
            &coo_aos, rows, cols, nnz, ind_data, val_data, cooType, idxBase, dataType),
        "Success");
    verify_hipsparse_status_success(hipsparseCreateCsr(&csr,
                                                       rows,
                                                       cols,
                                                       nnz,
                                                       row_data,
                                                       col_data,
                                                       val_data,
                                                       rowType,
                                                       colType,
                                                       idxBase,
                                                       dataType),
                                    "Success");
    verify_hipsparse_status_success(hipsparseCreateCsc(&csc,
                                                       rows,
                                                       cols,
                                                       nnz,
                                                       col_data,
                                                       row_data,
                                                       val_data,
                                                       colType,
                                                       rowType,
                                                       idxBase,
                                                       dataType),
                                    "Success");

    void* row_ptr;
    void* col_ptr;
    void* ind_ptr;
    void* val_ptr;

    // hipsparseCooGet
    verify_hipsparse_status_invalid_pointer(hipsparseCooGet(nullptr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, nullptr, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, nullptr, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCooGet(coo,
                                                            &rows,
                                                            &cols,
                                                            nullptr,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, nullptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, nullptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, nullptr, &rowType, &idxBase, &dataType),
        "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, nullptr, &idxBase, &dataType),
        "Error: rowType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, nullptr, &dataType),
        "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, nullptr),
        "Error: dataType is nullptr");

    // hipsparseCooAoSGet
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            nullptr, &rows, &cols, &nnz, &ind_ptr, &val_ptr, &cooType, &idxBase, &dataType),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, nullptr, &cols, &nnz, &ind_ptr, &val_ptr, &cooType, &idxBase, &dataType),
        "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, nullptr, &nnz, &ind_ptr, &val_ptr, &cooType, &idxBase, &dataType),
        "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, &cols, nullptr, &ind_ptr, &val_ptr, &cooType, &idxBase, &dataType),
        "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, &cols, &nnz, nullptr, &val_ptr, &cooType, &idxBase, &dataType),
        "Error: ind_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, &cols, &nnz, &ind_ptr, nullptr, &cooType, &idxBase, &dataType),
        "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, &cols, &nnz, &ind_ptr, &val_ptr, nullptr, &idxBase, &dataType),
        "Error: cooType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, &cols, &nnz, &ind_ptr, &val_ptr, &cooType, nullptr, &dataType),
        "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooAoSGet(
            coo_aos, &rows, &cols, &nnz, &ind_ptr, &val_ptr, &cooType, &idxBase, nullptr),
        "Error: dataType is nullptr");

    // hipsparseCsrGet
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(nullptr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            nullptr,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            nullptr,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            nullptr,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            nullptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            nullptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            nullptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            nullptr,
                                                            &colType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: rowType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            nullptr,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: colType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            nullptr,
                                                            &dataType),
                                            "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCsrGet(csr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &row_ptr,
                                                            &col_ptr,
                                                            &val_ptr,
                                                            &rowType,
                                                            &colType,
                                                            &idxBase,
                                                            nullptr),
                                            "Error: dataType is nullptr");

    // hipsparseCscGet
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(nullptr,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            nullptr,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            nullptr,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            nullptr,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            nullptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            nullptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            nullptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            nullptr,
                                                            &rowType,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: colType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            nullptr,
                                                            &idxBase,
                                                            &dataType),
                                            "Error: rowType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            nullptr,
                                                            &dataType),
                                            "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseCscGet(csc,
                                                            &rows,
                                                            &cols,
                                                            &nnz,
                                                            &col_ptr,
                                                            &row_ptr,
                                                            &val_ptr,
                                                            &colType,
                                                            &rowType,
                                                            &idxBase,
                                                            nullptr),
                                            "Error: dataType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(nullptr,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   nullptr,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   nullptr,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   nullptr,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: ell_blocksize is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   nullptr,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: ell_cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   nullptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: ellColInd");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   nullptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: ellValue is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   nullptr,
                                                                   &idxBase,
                                                                   &dataType),
                                            "Error: ellIdxType is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   nullptr,
                                                                   &dataType),
                                            "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseBlockedEllGet(bell,
                                                                   &rows,
                                                                   &cols,
                                                                   &ell_blocksize,
                                                                   &ell_cols,
                                                                   &col_ptr,
                                                                   &val_ptr,
                                                                   &colType,
                                                                   &idxBase,
                                                                   nullptr),
                                            "Error: valueType is nullptr");

    // hipsparseCsrSetPointers
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(nullptr, row_data, col_data, val_data), "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(csr, nullptr, col_data, val_data), "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(csr, row_data, nullptr, val_data), "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetPointers(csr, row_data, col_data, nullptr), "Error: val_data is nullptr");

    // hipsparseSpMatGetSize
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(nullptr, &rows, &cols, &nnz),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(coo, nullptr, &cols, &nnz),
                                            "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(coo, &rows, nullptr, &nnz),
                                            "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetSize(coo, &rows, &cols, nullptr),
                                            "Error: nnz is nullptr");

    // hipsparseSpMatGetFormat
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetFormat(nullptr, &format),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetFormat(coo, nullptr),
                                            "Error: format is nullptr");

    // hipsparseSpMatGetIndexBase
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetIndexBase(nullptr, &idxBase),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetIndexBase(coo, nullptr),
                                            "Error: idxBase is nullptr");

    // hipsparseSpMatGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetValues(nullptr, &val_ptr),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetValues(coo, nullptr),
                                            "Error: val_ptr is nullptr");

    // hipsparseSpMatSetValues
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatSetValues(nullptr, val_ptr),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatSetValues(coo, nullptr),
                                            "Error: val_ptr is nullptr");

    int     batch_count                 = 100;
    int64_t batch_stride                = 100;
    int64_t offsets_batch_stride        = 100;
    int64_t columns_values_batch_stride = 100;

    // hipsparseSpMatGetStridedBatch
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetStridedBatch(nullptr, &batch_count),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetStridedBatch(coo, nullptr),
                                            "Error: batch count is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetStridedBatch(csr, nullptr),
                                            "Error: batch count is nullptr");

    // hipsparseSpMatSetStridedBatch
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatSetStridedBatch(nullptr, batch_count),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseSpMatSetStridedBatch(coo, -1),
                                         "Error: batch count is invalid");
    verify_hipsparse_status_invalid_size(hipsparseSpMatSetStridedBatch(csr, -1),
                                         "Error: batch count is invalid");

    // hipsparseCooSetStridedBatch
    verify_hipsparse_status_invalid_pointer(
        hipsparseCooSetStridedBatch(nullptr, batch_count, batch_stride), "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseCooSetStridedBatch(coo, -1, batch_stride),
                                         "Error: batch count is invalid");
    verify_hipsparse_status_invalid_size(hipsparseCooSetStridedBatch(coo, batch_count, -1),
                                         "Error: batch stride is invalid");
    verify_hipsparse_status_invalid_size(hipsparseCooSetStridedBatch(coo, -1, -1),
                                         "Error: batch count and batch stride is invalid");

    // hipsparseCsrSetStridedBatch
    verify_hipsparse_status_invalid_pointer(
        hipsparseCsrSetStridedBatch(
            nullptr, batch_count, offsets_batch_stride, columns_values_batch_stride),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCsrSetStridedBatch(csr, -1, offsets_batch_stride, columns_values_batch_stride),
        "Error: batch count is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseCsrSetStridedBatch(csr, batch_count, -1, columns_values_batch_stride),
        "Error: batch stride is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseCsrSetStridedBatch(csr, batch_count, offsets_batch_stride, -1),
        "Error: batch stride is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseCsrSetStridedBatch(csr, -1, -1, columns_values_batch_stride),
        "Error: batch count and batch stride is invalid");
    verify_hipsparse_status_invalid_size(hipsparseCsrSetStridedBatch(csr, batch_count, -1, -1),
                                         "Error: batch count and batch stride is invalid");
    verify_hipsparse_status_invalid_size(
        hipsparseCsrSetStridedBatch(csr, -1, offsets_batch_stride, -1),
        "Error: batch count and batch stride is invalid");
    verify_hipsparse_status_invalid_size(hipsparseCsrSetStridedBatch(csr, -1, -1, -1),
                                         "Error: batch count and batch stride is invalid");

    // Destroy valid descriptors
    verify_hipsparse_status_success(hipsparseDestroySpMat(coo), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(coo_aos), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(csr), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(csc), "Success");

    verify_hipsparse_status_success(hipsparseDestroySpMat(bell), "Success");
#endif
}

hipsparseStatus_t testing_spmat_descr(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
    int m    = 2;
    int n    = 2;
    int nnzC = 4;

    // C
    std::vector<int>   hcsr_row_ptrC = {0, 2, 4};
    std::vector<int>   hcsr_col_indC = {0, 2, 0, 1};
    std::vector<float> hcsr_valC     = {5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f};

    int*   dcsr_row_ptrC = NULL;
    int*   dcsr_col_indC = NULL;
    float* dcsr_valC     = NULL;
    CHECK_HIP_ERROR(hipMalloc((void**)&dcsr_row_ptrC, (m + 1) * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc((void**)&dcsr_col_indC, nnzC * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc((void**)&dcsr_valC, nnzC * sizeof(float)));

    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptrC, hcsr_row_ptrC.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_indC, hcsr_col_indC.data(), nnzC * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_valC, hcsr_valC.data(), nnzC * sizeof(float), hipMemcpyHostToDevice));

    hipsparseHandle_t     handle = NULL;
    hipsparseSpMatDescr_t matA, matB, matC, matD, matE, matF, matG, matH, matI, matJ, matK, matL,
        matM, matN, matO;

    CHECK_HIPSPARSE_ERROR(hipsparseCreate(&handle));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matA,
                                             0,
                                             0,
                                             0,
                                             NULL,
                                             NULL,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matB,
                                             m,
                                             n,
                                             0,
                                             NULL,
                                             NULL,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matC,
                                             0,
                                             0,
                                             0,
                                             dcsr_row_ptrC,
                                             NULL,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matD,
                                             m,
                                             n,
                                             0,
                                             dcsr_row_ptrC,
                                             NULL,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matE,
                                             m,
                                             n,
                                             nnzC,
                                             dcsr_row_ptrC,
                                             dcsr_col_indC,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matF,
                                             m,
                                             n,
                                             0,
                                             NULL,
                                             dcsr_col_indC,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matG,
                                             0,
                                             0,
                                             0,
                                             NULL,
                                             dcsr_col_indC,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matH,
                                             m,
                                             n,
                                             0,
                                             NULL,
                                             NULL,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matI,
                                             0,
                                             0,
                                             0,
                                             NULL,
                                             NULL,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matJ,
                                             m,
                                             n,
                                             0,
                                             NULL,
                                             dcsr_col_indC,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matK,
                                             0,
                                             0,
                                             0,
                                             NULL,
                                             dcsr_col_indC,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matL,
                                             m,
                                             n,
                                             0,
                                             dcsr_row_ptrC,
                                             dcsr_col_indC,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matM,
                                             0,
                                             0,
                                             0,
                                             dcsr_row_ptrC,
                                             dcsr_col_indC,
                                             NULL,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matN,
                                             m,
                                             n,
                                             0,
                                             dcsr_row_ptrC,
                                             NULL,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matO,
                                             0,
                                             0,
                                             0,
                                             dcsr_row_ptrC,
                                             NULL,
                                             dcsr_valC,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIP_R_32F));

    // destroy matrix/vector descriptors
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matB));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matC));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matD));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matE));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matF));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matG));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matH));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matI));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matJ));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matK));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matL));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matM));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matN));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matO));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroy(handle));

    CHECK_HIP_ERROR(hipFree(dcsr_row_ptrC));
    CHECK_HIP_ERROR(hipFree(dcsr_col_indC));
    CHECK_HIP_ERROR(hipFree(dcsr_valC));
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_SPMAT_DESCR_HPP
