/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_HYB_MAT_INFO_HPP
#define TESTING_HYB_MAT_INFO_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipsparse.h>

#include <limits>

using namespace hipsparse_test;

void testing_hyb_mat_info_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    std::unique_ptr<hyb_struct> unique_ptr_hyb(new hyb_struct);
    hipsparseHybMat_t           hyb = unique_ptr_hyb->hyb;

    // Null HYB handle.
    verify_hipsparse_status_invalid_value(hipsparseHybMatGetInfo(nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: hyb is nullptr");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: hyb is nullptr");

    // nullptr fields: nothing requested / nothing changed.
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
    verify_hipsparse_status_success(hipsparseHybMatSetInfo(hyb,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");

    // Negative or out-of-range sizes and invalid enums are rejected.
    const int                     neg_int       = -1;
    const int64_t                 neg_i64       = -1;
    const int64_t                 big_ell_nnz   = int64_t(std::numeric_limits<int>::max()) + 1;
    const hipsparseHybPartition_t bad_partition = static_cast<hipsparseHybPartition_t>(-1);
    const hipDataType             bad_data_type = static_cast<hipDataType>(-1);

    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 &neg_int,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: m is negative");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 &neg_int,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: n is negative");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &neg_i64,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: ell_nnz is negative");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &neg_int,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: ell_width is negative");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &neg_int,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: coo_nnz is negative");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &bad_partition,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: partition is invalid");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &bad_data_type),
                                          "Error: data_type is invalid");

    // TODO: update this when a later PR widens the stored ELL non-zero count.
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &big_ell_nnz,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: ell_nnz does not fit");
    int64_t get_ell_nnz = -1;
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &get_ell_nnz,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
    int64_t expected_ell_nnz = 0;
    unit_check_general(1, 1, 1, &expected_ell_nnz, &get_ell_nnz);

    // A valid size paired with an invalid enum must not change hyb.
    const int keep_m  = 6;
    const int other_m = 11;
    verify_hipsparse_status_success(hipsparseHybMatSetInfo(hyb,
                                                           &keep_m,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 &other_m,
                                                                 nullptr,
                                                                 &bad_partition,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: partition is invalid");
    int get_m = -1;
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           &get_m,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
    int expected_m = keep_m;
    unit_check_general(1, 1, 1, &expected_m, &get_m);

    // Two non-null array arguments that alias one pointer are rejected, and
    // the caller keeps ownership when the call fails.
    auto alias_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * 4), device_free};
    int* alias_ptr     = static_cast<int*>(alias_managed.get());
    verify_hipsparse_status_invalid_value(hipsparseHybMatSetInfo(hyb,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &alias_ptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &alias_ptr,
                                                                 nullptr,
                                                                 nullptr),
                                          "Error: array arguments alias");

    // Round trip of sizes, partition, data type, ELL column indices and COO
    // arrays. Ownership transfers only after SetInfo succeeds.
    const int                     set_m         = 4;
    const int                     set_n         = 5;
    const hipsparseHybPartition_t set_partition = HIPSPARSE_HYB_PARTITION_USER;
    const int64_t                 set_ell_nnz   = 8;
    const int                     set_ell_width = 2;
    const int                     set_coo_nnz   = 3;
    const hipDataType             set_data_type = HIP_C_64F;

    auto ell_col_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * set_ell_nnz), device_free};
    auto ell_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * set_ell_nnz), device_free};
    auto coo_row_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * set_coo_nnz), device_free};
    auto coo_col_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * set_coo_nnz), device_free};
    auto coo_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * set_coo_nnz), device_free};

    int*  set_ell_col = static_cast<int*>(ell_col_managed.get());
    void* set_ell_val = ell_val_managed.get();
    int*  set_coo_row = static_cast<int*>(coo_row_managed.get());
    int*  set_coo_col = static_cast<int*>(coo_col_managed.get());
    void* set_coo_val = coo_val_managed.get();

    hipsparseStatus_t status = hipsparseHybMatSetInfo(hyb,
                                                      &set_m,
                                                      &set_n,
                                                      &set_partition,
                                                      &set_ell_nnz,
                                                      &set_ell_width,
                                                      &set_ell_col,
                                                      &set_ell_val,
                                                      &set_coo_nnz,
                                                      &set_coo_row,
                                                      &set_coo_col,
                                                      &set_coo_val,
                                                      &set_data_type);
    verify_hipsparse_status_success(status, "Success");
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return;
    }
    ell_col_managed.release();
    ell_val_managed.release();
    coo_row_managed.release();
    coo_col_managed.release();
    coo_val_managed.release();

    int                     got_m;
    int                     got_n;
    hipsparseHybPartition_t got_partition;
    int                     got_ell_width;
    int                     got_coo_nnz;
    const int*              got_ell_col;
    const void*             got_ell_val;
    const int*              got_coo_row;
    const int*              got_coo_col;
    const void*             got_coo_val;
    hipDataType             got_data_type;
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           &got_m,
                                                           &got_n,
                                                           &got_partition,
                                                           &get_ell_nnz,
                                                           &got_ell_width,
                                                           &got_ell_col,
                                                           &got_ell_val,
                                                           &got_coo_nnz,
                                                           &got_coo_row,
                                                           &got_coo_col,
                                                           &got_coo_val,
                                                           &got_data_type),
                                    "Success");
    int expected_n         = set_n;
    int expected_partition = static_cast<int>(set_partition);
    int actual_partition   = static_cast<int>(got_partition);
    int expected_width     = set_ell_width;
    int expected_coo_nnz   = set_coo_nnz;
    int expected_data_type = static_cast<int>(set_data_type);
    int actual_data_type   = static_cast<int>(got_data_type);
    expected_m             = set_m;
    expected_ell_nnz       = set_ell_nnz;
    unit_check_general(1, 1, 1, &expected_m, &got_m);
    unit_check_general(1, 1, 1, &expected_n, &got_n);
    unit_check_general(1, 1, 1, &expected_partition, &actual_partition);
    unit_check_general(1, 1, 1, &expected_ell_nnz, &get_ell_nnz);
    unit_check_general(1, 1, 1, &expected_width, &got_ell_width);
    unit_check_general(1, 1, 1, &expected_coo_nnz, &got_coo_nnz);
    unit_check_general(1, 1, 1, &expected_data_type, &actual_data_type);
    size_t expected_ptr = reinterpret_cast<size_t>(set_ell_col);
    size_t actual_ptr   = reinterpret_cast<size_t>(got_ell_col);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(set_ell_val);
    actual_ptr   = reinterpret_cast<size_t>(got_ell_val);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(set_coo_row);
    actual_ptr   = reinterpret_cast<size_t>(got_coo_row);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(set_coo_col);
    actual_ptr   = reinterpret_cast<size_t>(got_coo_col);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(set_coo_val);
    actual_ptr   = reinterpret_cast<size_t>(got_coo_val);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);

    // Passing a nullptr value releases the array and leaves the field empty.
    void* released_ell_val = nullptr;
    status                 = hipsparseHybMatSetInfo(hyb,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    &released_ell_val,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr);
    verify_hipsparse_status_success(status, "Success");
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return;
    }
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &got_ell_val,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
    expected_ptr = 0;
    actual_ptr   = reinterpret_cast<size_t>(got_ell_val);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);

    // Move the COO row array into the ELL column field. The previous ELL
    // column array is released by SetInfo; the moved pointer is not.
    int* moved_col    = set_coo_row;
    int* released_row = nullptr;
    status            = hipsparseHybMatSetInfo(hyb,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    &moved_col,
                                    nullptr,
                                    nullptr,
                                    &released_row,
                                    nullptr,
                                    nullptr,
                                    nullptr);
    verify_hipsparse_status_success(status, "Success");
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return;
    }
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &got_ell_col,
                                                           nullptr,
                                                           nullptr,
                                                           &got_coo_row,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
    expected_ptr = reinterpret_cast<size_t>(moved_col);
    actual_ptr   = reinterpret_cast<size_t>(got_ell_col);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = 0;
    actual_ptr   = reinterpret_cast<size_t>(got_coo_row);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);

    // Replacing ell_val releases the previous buffer; passing the same pointer
    // again is a no-op. The last one is freed by hipsparseDestroyHybMat.
    auto new_ell_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * set_ell_nnz), device_free};
    void* new_ell_val = new_ell_val_managed.get();
    status            = hipsparseHybMatSetInfo(hyb,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    &new_ell_val,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr);
    verify_hipsparse_status_success(status, "Success");
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return;
    }
    new_ell_val_managed.release();
    verify_hipsparse_status_success(hipsparseHybMatSetInfo(hyb,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &new_ell_val,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr),
                                    "Success");
#endif
}

#endif // TESTING_HYB_MAT_INFO_HPP
