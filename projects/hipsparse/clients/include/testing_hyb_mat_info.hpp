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
#include "utility.hpp"
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
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

    // An ell_nnz above INT_MAX is either rejected or stored exactly, never truncated.
    const hipsparseStatus_t big_status  = hipsparseHybMatSetInfo(hyb,
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
                                                                nullptr);
    int64_t                 get_ell_nnz = -1;
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
    if(big_status == HIPSPARSE_STATUS_SUCCESS)
    {
        ASSERT_EQ(get_ell_nnz, big_ell_nnz);
    }
    else
    {
        verify_hipsparse_status_invalid_value(big_status, "Error: ell_nnz does not fit");
        ASSERT_EQ(get_ell_nnz, 0);
    }

    // Round trip of sizes, partition and data type. The HYB takes ownership of
    // ell_val and frees it on destruction.
    const int                     set_m         = 4;
    const int                     set_n         = 5;
    const hipsparseHybPartition_t set_partition = HIPSPARSE_HYB_PARTITION_USER;
    const int64_t                 set_ell_nnz   = 8;
    const int                     set_ell_width = 2;
    const hipDataType             set_data_type = HIP_C_64F;
    void*                         set_ell_val;
    CHECK_HIP_ERROR(hipMalloc(&set_ell_val, sizeof(hipDoubleComplex) * set_ell_nnz));

    verify_hipsparse_status_success(hipsparseHybMatSetInfo(hyb,
                                                           &set_m,
                                                           &set_n,
                                                           &set_partition,
                                                           &set_ell_nnz,
                                                           &set_ell_width,
                                                           nullptr,
                                                           &set_ell_val,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &set_data_type),
                                    "Success");

    int                     get_m;
    int                     get_n;
    hipsparseHybPartition_t get_partition;
    int                     get_ell_width;
    const void*             get_ell_val;
    hipDataType             get_data_type;
    verify_hipsparse_status_success(hipsparseHybMatGetInfo(hyb,
                                                           &get_m,
                                                           &get_n,
                                                           &get_partition,
                                                           &get_ell_nnz,
                                                           &get_ell_width,
                                                           nullptr,
                                                           &get_ell_val,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &get_data_type),
                                    "Success");
    ASSERT_EQ(get_m, set_m);
    ASSERT_EQ(get_n, set_n);
    ASSERT_EQ(get_partition, set_partition);
    ASSERT_EQ(get_ell_nnz, set_ell_nnz);
    ASSERT_EQ(get_ell_width, set_ell_width);
    ASSERT_EQ(get_ell_val, set_ell_val);
    ASSERT_EQ(get_data_type, set_data_type);

    // Replacing ell_val releases the previous buffer; passing the same pointer
    // again is a no-op. The last one is freed by hipsparseDestroyHybMat.
    void* new_ell_val;
    CHECK_HIP_ERROR(hipMalloc(&new_ell_val, sizeof(hipDoubleComplex) * set_ell_nnz));
    for(int i = 0; i < 2; ++i)
    {
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
    }
#endif
}

#endif // TESTING_HYB_MAT_INFO_HPP
