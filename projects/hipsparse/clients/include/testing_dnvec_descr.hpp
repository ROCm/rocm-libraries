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
#ifndef TESTING_DNVEC_DESCR_HPP
#define TESTING_DNVEC_DESCR_HPP

#include "hipsparse_test_unique_ptr.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <hipsparse.h>

using namespace hipsparse_test;

void testing_dnvec_descr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int64_t size = 100;

    hipDataType dataType = HIP_R_32F;

    // Allocate memory on device
    auto val_data_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * size), device_free};

    float* val_data = (float*)val_data_managed.get();

    hipsparseDnVecDescr_t x;

    // hipsparseCreateDnVec
    verify_hipsparse_status_invalid_pointer(hipsparseCreateDnVec(nullptr, size, val_data, dataType),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseCreateDnVec(&x, -1, val_data, dataType),
                                         "Error: size is < 0");
    verify_hipsparse_status_invalid_pointer(hipsparseCreateDnVec(&x, size, nullptr, dataType),
                                            "Error: val_data is nullptr");

    // hipsparseDestroyDnVec
    verify_hipsparse_status_invalid_pointer(hipsparseDestroyDnVec(nullptr), "Error: x is nullptr");

    // Create valid descriptor
    verify_hipsparse_status_success(hipsparseCreateDnVec(&x, size, val_data, dataType), "Success");

    // hipsparseDnVecGet
    void* data;

    verify_hipsparse_status_invalid_pointer(hipsparseDnVecGet(nullptr, &size, &data, &dataType),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecGet(x, nullptr, &data, &dataType),
                                            "Error: size is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecGet(x, &size, nullptr, &dataType),
                                            "Error: val_data is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecGet(x, &size, &data, nullptr),
                                            "Error: dataType is nullptr");

    // hipsparseDnVecGetValues
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecGetValues(nullptr, &data),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecGetValues(x, nullptr),
                                            "Error: val is nullptr");

    // hipsparseDnVecSetValues
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecSetValues(nullptr, data),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseDnVecSetValues(x, nullptr),
                                            "Error: val is nullptr");

    // Destroy valid descriptor
    verify_hipsparse_status_success(hipsparseDestroyDnVec(x), "Success");
#endif
}

#endif // TESTING_DNVEC_DESCR_HPP
