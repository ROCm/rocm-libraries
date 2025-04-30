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
#ifndef TESTING_SPVEC_DESCR_HPP
#define TESTING_SPVEC_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hipsparse.h>

using namespace hipsparse_test;

void testing_spvec_descr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t size = 100;
    int64_t nnz  = 100;

    hipsparseIndexType_t idxType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;

    // Allocate memory on device
    auto idx_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto val_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};

    int*   idx_data = (int*)idx_data_managed.get();
    float* val_data = (float*)val_data_managed.get();

    hipsparseSpVecDescr_t x;

    // hipsparseCreateSpVec
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateSpVec(nullptr, size, nnz, idx_data, val_data, idxType, idxBase, dataType),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateSpVec(&x, -1, nnz, idx_data, val_data, idxType, idxBase, dataType),
        "Error: size is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateSpVec(&x, size, -1, idx_data, val_data, idxType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateSpVec(&x, size, nnz, nullptr, val_data, idxType, idxBase, dataType),
        "Error: idx_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateSpVec(&x, size, nnz, idx_data, nullptr, idxType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseDestroySpVec
    verify_hipsparse_status_invalid_pointer(hipsparseDestroySpVec(nullptr), "Error: x is nullptr");

    // Create valid descriptor
    verify_hipsparse_status_success(
        hipsparseCreateSpVec(&x, size, nnz, idx_data, val_data, idxType, idxBase, dataType),
        "Success");

    // hipsparseSpVecGet
    void* idx;
    void* data;

    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(nullptr, &size, &nnz, &idx, &data, &idxType, &idxBase, &dataType),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, nullptr, &nnz, &idx, &data, &idxType, &idxBase, &dataType),
        "Error: size is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, &size, nullptr, &idx, &data, &idxType, &idxBase, &dataType),
        "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, &size, &nnz, nullptr, &data, &idxType, &idxBase, &dataType),
        "Error: idx is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, &size, &nnz, &idx, nullptr, &idxType, &idxBase, &dataType),
        "Error: val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, &size, &nnz, &idx, &data, nullptr, &idxBase, &dataType),
        "Error: idxType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, &size, &nnz, &idx, &data, &idxType, nullptr, &dataType),
        "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseSpVecGet(x, &size, &nnz, &idx, &data, &idxType, &idxBase, nullptr),
        "Error: dataType is nullptr");

    // hipsparseSpVecGetIndexBase
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecGetIndexBase(nullptr, &idxBase),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecGetIndexBase(x, nullptr),
                                            "Error: idxBase is nullptr");

    // hipsparseSpVecGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecGetValues(nullptr, &data),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecGetValues(x, nullptr),
                                            "Error: val is nullptr");

    // hipsparseSpVecSetValues
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecSetValues(nullptr, data),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecSetValues(x, nullptr),
                                            "Error: val is nullptr");

    // Destroy valid descriptor
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
#endif
}

#endif // TESTING_SPVEC_DESCR_HPP
