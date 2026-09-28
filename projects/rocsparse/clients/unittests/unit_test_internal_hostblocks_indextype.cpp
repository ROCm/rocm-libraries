/*! \file */
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

//
// Unit tests for the rocSPARSE internal determine_I_indextype /
// determine_J_indextype routines
// (library/src/common/rocsparse_determine_indextype.cpp). Split out of
// unit_test_internal_hostblocks.cpp by topic.
//
// These map a sparse-matrix descriptor's storage index types onto the (I, J)
// index roles. For CSR the row-pointer type is I and the column-index type is
// J; CSC swaps them; single-index formats (COO) return the same type for both.
//
// NOTE ON TARGET: host-pure logic, but its include chain requires the HIP
// compile mode, so this file builds into the GPU test binary
// (rocsparse-unit-test-device); the defining TU is compiled in via
// ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES. No kernels are launched.
//
#include "unit_test_utils.hpp"

#include "rocsparse_determine_indextype.hpp" // determine_I/J_indextype

#include <gtest/gtest.h>

namespace
{
    // Small, valid, non-null device allocations so descriptor creation passes its
    // array/size argument checks. Contents are irrelevant to the index-type query.
    struct dummy_device_arrays
    {
        rocsparse_ut::device_vector<char> a{size_t{256}};
        rocsparse_ut::device_vector<char> b{size_t{256}};
        rocsparse_ut::device_vector<char> c{size_t{256}};
    };
}

TEST(internal_hostblocks_indextype, csr)
{
    dummy_device_arrays d;

    // Distinct row-pointer / column-index types so the two roles are separable.
    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr,
                                         d.b.ptr,
                                         d.c.ptr,
                                         rocsparse_indextype_i32, // row_ptr  -> I
                                         rocsparse_indextype_i64, // col_ind  -> J
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i32);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST(internal_hostblocks_indextype, csr_uniform)
{
    dummy_device_arrays d;

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr,
                                         d.b.ptr,
                                         d.c.ptr,
                                         rocsparse_indextype_i64,
                                         rocsparse_indextype_i64,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i64);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST(internal_hostblocks_indextype, csc_swaps_roles)
{
    dummy_device_arrays d;

    // CSC stores col_ptr as the I-role and row_ind as the J-role, i.e. the
    // opposite of CSR.
    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csc_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr, // csc_col_ptr
                                         d.b.ptr, // csc_row_ind
                                         d.c.ptr, // csc_val
                                         rocsparse_indextype_i32, // col_ptr -> I
                                         rocsparse_indextype_i64, // row_ind -> J
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i32);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST(internal_hostblocks_indextype, coo)
{
    dummy_device_arrays d;

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr, // coo_row_ind
                                         d.b.ptr, // coo_col_ind
                                         d.c.ptr, // coo_val
                                         rocsparse_indextype_i64,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i64);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}
