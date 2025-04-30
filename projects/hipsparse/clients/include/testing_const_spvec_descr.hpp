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
#ifndef TESTING_CONST_SPVEC_DESCR_HPP
#define TESTING_CONST_SPVEC_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hipsparse.h>

using namespace hipsparse_test;

void testing_const_spvec_descr_bad_arg(void)
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

    const int*   idx_data = (const int*)idx_data_managed.get();
    const float* val_data = (const float*)val_data_managed.get();

    hipsparseConstSpVecDescr_t x;

    // hipsparseCreateConstSpVec
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstSpVec(
            nullptr, size, nnz, idx_data, val_data, idxType, idxBase, dataType),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstSpVec(&x, -1, nnz, idx_data, val_data, idxType, idxBase, dataType),
        "Error: size is < 0");
    verify_hipsparse_status_invalid_size(
        hipsparseCreateConstSpVec(&x, size, -1, idx_data, val_data, idxType, idxBase, dataType),
        "Error: nnz is < 0");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstSpVec(&x, size, nnz, nullptr, val_data, idxType, idxBase, dataType),
        "Error: idx_data is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateConstSpVec(&x, size, nnz, idx_data, nullptr, idxType, idxBase, dataType),
        "Error: val_data is nullptr");

    // hipsparseDestroySpVec
    verify_hipsparse_status_invalid_pointer(hipsparseDestroySpVec(nullptr), "Error: x is nullptr");

    // Create valid descriptor
    verify_hipsparse_status_success(
        hipsparseCreateConstSpVec(&x, size, nnz, idx_data, val_data, idxType, idxBase, dataType),
        "Success");

    // hipsparseConstSpVecGet
    const void* idx;
    const void* data;

    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(nullptr, &size, &nnz, &idx, &data, &idxType, &idxBase, &dataType),
        "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, nullptr, &nnz, &idx, &data, &idxType, &idxBase, &dataType),
        "Error: size is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, &size, nullptr, &idx, &data, &idxType, &idxBase, &dataType),
        "Error: nnz is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, &size, &nnz, nullptr, &data, &idxType, &idxBase, &dataType),
        "Error: idx is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, &size, &nnz, &idx, nullptr, &idxType, &idxBase, &dataType),
        "Error: val is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, &size, &nnz, &idx, &data, nullptr, &idxBase, &dataType),
        "Error: idxType is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, &size, &nnz, &idx, &data, &idxType, nullptr, &dataType),
        "Error: idxBase is nullptr");
    verify_hipsparse_status_invalid_pointer(
        hipsparseConstSpVecGet(x, &size, &nnz, &idx, &data, &idxType, &idxBase, nullptr),
        "Error: dataType is nullptr");

    // hipsparseSpVecGetIndexBase
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecGetIndexBase(nullptr, &idxBase),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseSpVecGetIndexBase(x, nullptr),
                                            "Error: idxBase is nullptr");

    // hipsparseConstSpVecGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseConstSpVecGetValues(nullptr, &data),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseConstSpVecGetValues(x, nullptr),
                                            "Error: val is nullptr");

    // Destroy valid descriptor
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
#endif
}

#endif // TESTING_CONST_SPVEC_DESCR_HPP
