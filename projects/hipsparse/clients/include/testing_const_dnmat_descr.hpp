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
#ifndef TESTING_CONST_DNMAT_DESCR_HPP
#define TESTING_CONST_DNMAT_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hipsparse.h>

using namespace hipsparse_test;

void testing_const_dnmat_descr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t          rows  = 100;
    int64_t          cols  = 100;
    int64_t          ld    = 100;
    hipsparseOrder_t order = HIPSPARSE_ORDER_ROW;

    hipDataType dataType = HIP_R_32F;

    // Allocate memory on device
    auto val_data_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(float) * rows * cols), device_free};

    const float* val_data = (const float*)val_data_managed.get();

    hipsparseConstDnMatDescr_t x;

    // hipsparseCreateConstDnMat
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstDnMat(nullptr, rows, cols, ld, val_data, dataType, order),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstDnMat(&x, -1, cols, ld, val_data, dataType, order),
        "Error: rows is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstDnMat(&x, rows, -1, ld, val_data, dataType, order),
        "Error: cols is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstDnMat(&x, rows, cols, -1, val_data, dataType, order),
        "Error: ld is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstDnMat(&x, rows, cols, ld, nullptr, dataType, order),
        "Error: val_data is nullptr");

    // hipsparseDestroyDnVec
    verify_hipsparse_status_invalid_pointer(hipsparseDestroyDnMat(nullptr), "Error: x is nullptr");

    // Create valid descriptor
    verify_hipsparse_status_success(
        hipsparseCreateConstDnMat(&x, rows, cols, ld, val_data, dataType, order), "Success");

    // hipsparseConstDnMatGet
    const void* data;

    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(nullptr, &rows, &cols, &ld, &data, &dataType, &order),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(x, nullptr, &cols, &ld, &data, &dataType, &order),
        "Error: rows is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(x, &rows, nullptr, &ld, &data, &dataType, &order),
        "Error: cols is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(x, &rows, &cols, nullptr, &data, &dataType, &order),
        "Error: ld is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(x, &rows, &cols, &ld, nullptr, &dataType, &order),
        "Error: data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(x, &rows, &cols, &ld, &data, nullptr, &order),
        "Error: dataType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstDnMatGet(x, &rows, &cols, &ld, &data, &dataType, nullptr),
        "Error: order is nullptr");

    // hipsparseConstDnMatGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseConstDnMatGetValues(nullptr, &data),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseConstDnMatGetValues(x, nullptr),
                                            "Error: val is nullptr");

    int     batch_count  = 100;
    int64_t batch_stride = 100;

    // hipsparseDnMatGetStridedBatch
    verify_hipsparse_status_invalid_pointer(
        hipsparseDnMatGetStridedBatch(nullptr, &batch_count, &batch_stride), "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseDnMatGetStridedBatch(x, nullptr, &batch_stride), "Error: batch_count is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseDnMatGetStridedBatch(x, &batch_count, nullptr),
                                            "Error: batch_stride is nullptr");

    // Destroy valid descriptor
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "Success");
#endif
}

#endif // TESTING_CONST_DNMAT_DESCR_HPP
