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

    // A valid empty matrix. Each case below changes some of its fields.
    struct set_info_args
    {
        int                     m           = 0;
        int                     n           = 0;
        hipsparseHybPartition_t partition   = HIPSPARSE_HYB_PARTITION_AUTO;
        int64_t                 ell_nnz     = 0;
        int                     ell_width   = 0;
        int*                    ell_col_ind = nullptr;
        void*                   ell_val     = nullptr;
        int                     coo_nnz     = 0;
        int*                    coo_row_ind = nullptr;
        int*                    coo_col_ind = nullptr;
        void*                   coo_val     = nullptr;
        hipDataType             data_type   = HIP_R_32F;
    };
    auto set_info = [](hipsparseHybMat_t h, const set_info_args& args) {
        return hipsparseHybMatSetInfo(h,
                                      args.m,
                                      args.n,
                                      args.partition,
                                      args.ell_nnz,
                                      args.ell_width,
                                      args.ell_col_ind,
                                      args.ell_val,
                                      args.coo_nnz,
                                      args.coo_row_ind,
                                      args.coo_col_ind,
                                      args.coo_val,
                                      args.data_type);
    };

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
    verify_hipsparse_status_invalid_value(set_info(nullptr, set_info_args{}),
                                          "Error: hyb is nullptr");

    // nullptr outputs: nothing requested. The valid empty matrix is accepted.
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
    verify_hipsparse_status_success(set_info(hyb, set_info_args{}), "Success");

    // Negative or out-of-range sizes and invalid enums are rejected.
    const hipsparseHybPartition_t bad_partition = static_cast<hipsparseHybPartition_t>(-1);
    const hipDataType             bad_data_type = static_cast<hipDataType>(-1);

    set_info_args args;
    args.m = -1;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: m is negative");
    args   = set_info_args{};
    args.n = -1;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: n is negative");
    args         = set_info_args{};
    args.ell_nnz = -1;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: ell_nnz is negative");
    args           = set_info_args{};
    args.ell_width = -1;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: ell_width is negative");
    args         = set_info_args{};
    args.coo_nnz = -1;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: coo_nnz is negative");
    args           = set_info_args{};
    args.partition = bad_partition;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: partition is invalid");
    args           = set_info_args{};
    args.data_type = bad_data_type;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: data_type is invalid");

    // TODO: update this when a later PR widens the stored ELL non-zero count.
    args         = set_info_args{};
    args.ell_nnz = int64_t(std::numeric_limits<int>::max()) + 1;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: ell_nnz does not fit");
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
    const int keep_m = 6;
    args             = set_info_args{};
    args.m           = keep_m;
    verify_hipsparse_status_success(set_info(hyb, args), "Success");
    args.m         = 11;
    args.partition = bad_partition;
    verify_hipsparse_status_invalid_value(set_info(hyb, args), "Error: partition is invalid");
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

    // Round trip of sizes, partition, data type, ELL column indices and COO
    // arrays. The caller keeps ownership: the managed arrays outlive every use
    // of hyb below, and hipsparseDestroyHybMat does not free them.
    set_info_args full;
    full.m         = 4;
    full.n         = 5;
    full.partition = HIPSPARSE_HYB_PARTITION_USER;
    full.ell_nnz   = 8;
    full.ell_width = 2;
    full.coo_nnz   = 3;
    full.data_type = HIP_C_64F;

    auto ell_col_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * full.ell_nnz), device_free};
    auto ell_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * full.ell_nnz), device_free};
    auto coo_row_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * full.coo_nnz), device_free};
    auto coo_col_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * full.coo_nnz), device_free};
    auto coo_val_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(hipDoubleComplex) * full.coo_nnz), device_free};

    full.ell_col_ind = static_cast<int*>(ell_col_managed.get());
    full.ell_val     = ell_val_managed.get();
    full.coo_row_ind = static_cast<int*>(coo_row_managed.get());
    full.coo_col_ind = static_cast<int*>(coo_col_managed.get());
    full.coo_val     = coo_val_managed.get();

    hipsparseStatus_t status = set_info(hyb, full);
    verify_hipsparse_status_success(status, "Success");
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return;
    }

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
    int expected_n         = full.n;
    int expected_partition = static_cast<int>(full.partition);
    int actual_partition   = static_cast<int>(got_partition);
    int expected_width     = full.ell_width;
    int expected_coo_nnz   = full.coo_nnz;
    int expected_data_type = static_cast<int>(full.data_type);
    int actual_data_type   = static_cast<int>(got_data_type);
    expected_m             = full.m;
    expected_ell_nnz       = full.ell_nnz;
    unit_check_general(1, 1, 1, &expected_m, &got_m);
    unit_check_general(1, 1, 1, &expected_n, &got_n);
    unit_check_general(1, 1, 1, &expected_partition, &actual_partition);
    unit_check_general(1, 1, 1, &expected_ell_nnz, &get_ell_nnz);
    unit_check_general(1, 1, 1, &expected_width, &got_ell_width);
    unit_check_general(1, 1, 1, &expected_coo_nnz, &got_coo_nnz);
    unit_check_general(1, 1, 1, &expected_data_type, &actual_data_type);
    size_t expected_ptr = reinterpret_cast<size_t>(full.ell_col_ind);
    size_t actual_ptr   = reinterpret_cast<size_t>(got_ell_col);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(full.ell_val);
    actual_ptr   = reinterpret_cast<size_t>(got_ell_val);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(full.coo_row_ind);
    actual_ptr   = reinterpret_cast<size_t>(got_coo_row);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(full.coo_col_ind);
    actual_ptr   = reinterpret_cast<size_t>(got_coo_col);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = reinterpret_cast<size_t>(full.coo_val);
    actual_ptr   = reinterpret_cast<size_t>(got_coo_val);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);

    // A null ell_val while ell_nnz stays positive is rejected, and the field
    // is unchanged.
    args         = full;
    args.ell_val = nullptr;
    verify_hipsparse_status_invalid_value(set_info(hyb, args),
                                          "Error: ell_val is null while ell_nnz is positive");
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
    expected_ptr = reinterpret_cast<size_t>(full.ell_val);
    actual_ptr   = reinterpret_cast<size_t>(got_ell_val);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);

    // Move the COO row array into the ELL column field and drop the COO part.
    set_info_args moved = full;
    moved.ell_col_ind   = full.coo_row_ind;
    moved.coo_nnz       = 0;
    moved.coo_row_ind   = nullptr;
    moved.coo_col_ind   = nullptr;
    moved.coo_val       = nullptr;
    status              = set_info(hyb, moved);
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
    expected_ptr = reinterpret_cast<size_t>(moved.ell_col_ind);
    actual_ptr   = reinterpret_cast<size_t>(got_ell_col);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);
    expected_ptr = 0;
    actual_ptr   = reinterpret_cast<size_t>(got_coo_row);
    unit_check_general(1, 1, 1, &expected_ptr, &actual_ptr);

    // Replace ell_val, then pass the same pointer again.
    auto new_ell_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(hipDoubleComplex) * moved.ell_nnz), device_free};
    moved.ell_val = new_ell_val_managed.get();
    status        = set_info(hyb, moved);
    verify_hipsparse_status_success(status, "Success");
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        return;
    }
    verify_hipsparse_status_success(set_info(hyb, moved), "Success");

    // Clearing the ELL arrays requires the ELL count to be zero.
    moved.ell_nnz     = 0;
    moved.ell_width   = 0;
    moved.ell_col_ind = nullptr;
    moved.ell_val     = nullptr;
    status            = set_info(hyb, moved);
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
#endif
}

#endif // TESTING_HYB_MAT_INFO_HPP
