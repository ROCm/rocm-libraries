/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CONST_SPMAT_DESCR_HPP
#define TESTING_CONST_SPMAT_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hipsparse.h>

#include <iostream>

using namespace hipsparse_test;

void testing_const_spmat_descr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t rows          = 100;
    int64_t cols          = 100;
    int64_t nnz           = 100;
    int64_t ell_cols      = 10;
    int64_t ell_blocksize = 2;

    hipsparseIndexType_t rowType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexType_t colType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;
    hipsparseFormat_t    format   = HIPSPARSE_FORMAT_CSR;

    // Allocate memory on device
    auto row_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto col_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto val_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};

    const int*   row_data = (int*)row_data_managed.get();
    const int*   col_data = (int*)col_data_managed.get();
    const float* val_data = (float*)val_data_managed.get();

    hipsparseConstSpMatDescr_t A;

    // hipsparseCreateConstCoo
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCoo(
            nullptr, rows, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: A is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCoo(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCoo(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCoo(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCoo(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCoo(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCoo(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateConstCsr
    verify_hipsparse_status_invalid_pointer(hipsparseCreateConstCsr(nullptr,
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
        hipsparseCreateConstCsr(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCsr(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCsr(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCsr(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCsr(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, colType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCsr(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, colType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateConstCsc
    verify_hipsparse_status_invalid_pointer(hipsparseCreateConstCsc(nullptr,
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
        hipsparseCreateConstCsc(
            &A, -1, cols, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCsc(
            &A, rows, -1, nnz, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstCsc(
            &A, rows, cols, -1, row_data, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCsc(
            &A, rows, cols, nnz, nullptr, col_data, val_data, rowType, colType, idxBase, dataType),
        "Error: row_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCsc(
            &A, rows, cols, nnz, row_data, nullptr, val_data, rowType, colType, idxBase, dataType),
        "Error: col_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstCsc(
            &A, rows, cols, nnz, row_data, col_data, nullptr, rowType, colType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseCreateConstBlockedEll
    verify_hipsparse_status_invalid_pointer(hipsparseCreateConstBlockedEll(nullptr,
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
        hipsparseCreateConstBlockedEll(
            &A, -1, cols, ell_blocksize, ell_cols, col_data, val_data, colType, idxBase, dataType),
        "Error: rows is < 0");

    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstBlockedEll(
            &A, rows, -1, ell_blocksize, ell_cols, col_data, val_data, colType, idxBase, dataType),
        "Error: cols is < 0");

    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstBlockedEll(
            &A, rows, cols, -1, ell_cols, col_data, val_data, colType, idxBase, dataType),
        "Error: ell_blocksize is < 0");

    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstBlockedEll(
            &A, rows, cols, ell_blocksize, -1, col_data, val_data, colType, idxBase, dataType),
        "Error: ell_cols is < 0");

    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstBlockedEll(
            &A, rows, cols, ell_blocksize, ell_cols, nullptr, val_data, colType, idxBase, dataType),
        "Error: ellColInd is nullptr");

    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstBlockedEll(
            &A, rows, cols, ell_blocksize, ell_cols, col_data, nullptr, colType, idxBase, dataType),
        "Error: ellValue is nullptr");

    // hipsparseDestroySpMat
    verify_hipsparse_status_invalid_pointer(hipsparseDestroySpMat(nullptr), "Error: A is nullptr");

    // Create valid descriptors
    hipsparseConstSpMatDescr_t coo;
    hipsparseConstSpMatDescr_t csr;
    hipsparseConstSpMatDescr_t csc;
    hipsparseConstSpMatDescr_t bell;

    verify_hipsparse_status_success(hipsparseCreateConstBlockedEll(&bell,
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
        hipsparseCreateConstCoo(
            &coo, rows, cols, nnz, row_data, col_data, val_data, rowType, idxBase, dataType),
        "Success");
    verify_hipsparse_status_success(hipsparseCreateConstCsr(&csr,
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
    verify_hipsparse_status_success(hipsparseCreateConstCsc(&csc,
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

    const void* row_ptr;
    const void* col_ptr;
    const void* val_ptr;

    // hipsparseConstCooGet
    verify_hipsparse_status_invalid_pointer(hipsparseConstCooGet(nullptr,
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
        hipsparseConstCooGet(
            coo, nullptr, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstCooGet(
            coo, &rows, nullptr, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseConstCooGet(coo,
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
        hipsparseConstCooGet(
            coo, &rows, &cols, &nnz, nullptr, &col_ptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, nullptr, &val_ptr, &rowType, &idxBase, &dataType),
        "Error: col_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, nullptr, &rowType, &idxBase, &dataType),
        "Error: val_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, nullptr, &idxBase, &dataType),
        "Error: rowType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, nullptr, &dataType),
        "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstCooGet(
            coo, &rows, &cols, &nnz, &row_ptr, &col_ptr, &val_ptr, &rowType, &idxBase, nullptr),
        "Error: dataType is nullptr");

    // hipsparseConstCsrGet
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(nullptr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCsrGet(csr,
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

    // hipsparseConstCsrGet
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(nullptr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstCscGet(csc,
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

    // hipsparseConstBlockedEllGet
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(nullptr,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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
    verify_hipsparse_status_invalid_pointer(hipsparseConstBlockedEllGet(bell,
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

    // hipsparseConstSpMatGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseConstSpMatGetValues(nullptr, &val_ptr),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseConstSpMatGetValues(coo, nullptr),
                                            "Error: val_ptr is nullptr");
    int batch_count = 100;

    // hipsparseSpMatGetStridedBatch
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetStridedBatch(nullptr, &batch_count),
                                            "Error: A is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetStridedBatch(coo, nullptr),
                                            "Error: batch count is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpMatGetStridedBatch(csr, nullptr),
                                            "Error: batch count is nullptr");

    // Destroy valid descriptors
    verify_hipsparse_status_success(hipsparseDestroySpMat(coo), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(csc), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(csr), "Success");
    verify_hipsparse_status_success(hipsparseDestroySpMat(bell), "Success");

#endif
}

#endif // TESTING_CONST_SPMAT_DESCR_HPP
