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
// Host-path unit tests for the auxiliary descriptor API (rocsparse_auxiliary.cpp
// and the generic descriptor sources). These are pure host code: descriptor
// create / get / set / copy / destroy plus their argument-validation guards.
// No kernels are launched, so this is high-yield host coverage.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

// ---------------------------------------------------------------------------
// Legacy rocsparse_mat_descr lifecycle + attribute setters/getters.
// ---------------------------------------------------------------------------
TEST(AuxMatDescr, lifecycle_and_attributes)
{
    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    EXPECT_EQ(rocsparse_set_mat_index_base(descr, rocsparse_index_base_one),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_index_base(descr), rocsparse_index_base_one);

    EXPECT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_type(descr), rocsparse_matrix_type_symmetric);

    EXPECT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_upper),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_fill_mode(descr), rocsparse_fill_mode_upper);

    EXPECT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_unit),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_diag_type(descr), rocsparse_diag_type_unit);

    // copy into a second descriptor
    rocsparse_mat_descr dst = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&dst), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_descr(dst, descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_index_base(dst), rocsparse_index_base_one);
    EXPECT_EQ(rocsparse_get_mat_type(dst), rocsparse_matrix_type_symmetric);

    EXPECT_EQ(rocsparse_destroy_mat_descr(dst), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(AuxMatDescr, bad_args)
{
    EXPECT_EQ(rocsparse_create_mat_descr(nullptr), rocsparse_status_invalid_pointer);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    // invalid enum values
    EXPECT_EQ(rocsparse_set_mat_index_base(descr, (rocsparse_index_base)99),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_set_mat_type(descr, (rocsparse_matrix_type)99),
              rocsparse_status_invalid_value);
    // null descriptor
    EXPECT_EQ(rocsparse_set_mat_index_base(nullptr, rocsparse_index_base_zero),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_copy_mat_descr(nullptr, descr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Handle-level queries.
// ---------------------------------------------------------------------------
class AuxHandle : public HandleTest
{
};

TEST_F(AuxHandle, version_and_pointer_mode_and_stream)
{
    int version = 0;
    EXPECT_EQ(rocsparse_get_version(handle, &version), rocsparse_status_success);
    EXPECT_GT(version, 0);

    EXPECT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    rocsparse_pointer_mode mode = rocsparse_pointer_mode_host;
    EXPECT_EQ(rocsparse_get_pointer_mode(handle, &mode), rocsparse_status_success);
    EXPECT_EQ(mode, rocsparse_pointer_mode_device);
    EXPECT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    // default (null) stream
    EXPECT_EQ(rocsparse_set_stream(handle, nullptr), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_get_version(nullptr, &version), rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_get_pointer_mode(handle, nullptr), rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// Generic sparse/dense vector descriptors: create / get / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxSpVec, create_get_destroy)
{
    device_vector<int32_t> ind{std::vector<int32_t>{0, 2, 4}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(ind.ptr && val.ptr);

    rocsparse_spvec_descr x = nullptr;
    ASSERT_EQ(rocsparse_create_spvec_descr(&x,
                                           8,
                                           3,
                                           ind.ptr,
                                           val.ptr,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              size = 0, nnz = 0;
    void*                pind = nullptr;
    void*                pval = nullptr;
    rocsparse_indextype  it   = rocsparse_indextype_i64;
    rocsparse_index_base ib   = rocsparse_index_base_one;
    rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_spvec_get(x, &size, &nnz, &pind, &pval, &it, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(size, 8);
    EXPECT_EQ(nnz, 3);
    EXPECT_EQ(it, rocsparse_indextype_i32);
    EXPECT_EQ(ib, rocsparse_index_base_zero);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    EXPECT_EQ(rocsparse_destroy_spvec_descr(x), rocsparse_status_success);
}

TEST(AuxSpVec, bad_args)
{
    device_vector<int32_t> ind{std::vector<int32_t>{0, 2, 4}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    rocsparse_spvec_descr  x = nullptr;
    // negative size
    EXPECT_EQ(rocsparse_create_spvec_descr(&x,
                                           -1,
                                           3,
                                           ind.ptr,
                                           val.ptr,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    // null descriptor out-param
    EXPECT_EQ(rocsparse_create_spvec_descr(nullptr,
                                           8,
                                           3,
                                           ind.ptr,
                                           val.ptr,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

TEST(AuxDnVec, create_get_destroy)
{
    device_vector<float>  val{std::vector<float>(8, 1.0f)};
    rocsparse_dnvec_descr y = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&y, 8, val.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t            size = 0;
    void*              pv   = nullptr;
    rocsparse_datatype dt   = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_dnvec_get(y, &size, &pv, &dt), rocsparse_status_success);
    EXPECT_EQ(size, 8);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(y), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_create_dnvec_descr(&y, -1, val.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_create_dnvec_descr(nullptr, 8, val.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// Generic sparse matrix descriptors (COO / CSR): create / get / destroy.
// ---------------------------------------------------------------------------
TEST(AuxSpMat, coo_create_get_destroy)
{
    device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(row.ptr && col.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&A,
                                         3,
                                         3,
                                         3,
                                         row.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, nnz = 0;
    void *               pr = nullptr, *pc = nullptr, *pv = nullptr;
    rocsparse_indextype  it = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_coo_get(A, &rows, &cols, &nnz, &pr, &pc, &pv, &it, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(nnz, 3);

    rocsparse_format fmt = rocsparse_format_csr;
    EXPECT_EQ(rocsparse_spmat_get_format(A, &fmt), rocsparse_status_success);
    EXPECT_EQ(fmt, rocsparse_format_coo);

    int64_t gr = 0, gc = 0, gnnz = 0;
    EXPECT_EQ(rocsparse_spmat_get_size(A, &gr, &gc, &gnnz), rocsparse_status_success);
    EXPECT_EQ(gr, 3);

    rocsparse_index_base gib = rocsparse_index_base_one;
    EXPECT_EQ(rocsparse_spmat_get_index_base(A, &gib), rocsparse_status_success);
    EXPECT_EQ(gib, rocsparse_index_base_zero);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxSpMat, csr_create_get_destroy)
{
    device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(ptr.ptr && col.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&A,
                                         3,
                                         3,
                                         3,
                                         ptr.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, nnz = 0;
    void *               prp = nullptr, *pc = nullptr, *pv = nullptr;
    rocsparse_indextype  pit = rocsparse_indextype_i64, cit = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_csr_get(A, &rows, &cols, &nnz, &prp, &pc, &pv, &pit, &cit, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(nnz, 3);
    EXPECT_EQ(pit, rocsparse_indextype_i32);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxSpMat, bad_args)
{
    device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    rocsparse_spmat_descr  A = nullptr;
    // negative dimension
    EXPECT_EQ(rocsparse_create_coo_descr(&A,
                                         -1,
                                         3,
                                         3,
                                         row.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    // null out-param
    EXPECT_EQ(rocsparse_create_coo_descr(nullptr,
                                         3,
                                         3,
                                         3,
                                         row.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// Generic dense matrix descriptor: create / get / destroy.
// ---------------------------------------------------------------------------
TEST(AuxDnMat, create_get_destroy)
{
    device_vector<float>  val{std::vector<float>(3 * 4, 1.0f)};
    rocsparse_dnmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &A, 3, 4, 3, val.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    int64_t            rows = 0, cols = 0, ld = 0;
    void*              pv = nullptr;
    rocsparse_datatype dt = rocsparse_datatype_f64_r;
    rocsparse_order    od = rocsparse_order_row;
    EXPECT_EQ(rocsparse_dnmat_get(A, &rows, &cols, &ld, &pv, &dt, &od), rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(cols, 4);
    EXPECT_EQ(ld, 3);
    EXPECT_EQ(od, rocsparse_order_column);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_create_dnmat_descr(
                  nullptr, 3, 4, 3, val.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// CSC sparse matrix descriptor: create / get / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxCsc, create_get_destroy)
{
    device_vector<int32_t> col_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(col_ptr.ptr && row_ind.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_csc_descr(&A,
                                         3,
                                         3,
                                         3,
                                         col_ptr.ptr,
                                         row_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, nnz = 0;
    void *               pcp = nullptr, *pri = nullptr, *pv = nullptr;
    rocsparse_indextype  cpt = rocsparse_indextype_i64, rit = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_csc_get(A, &rows, &cols, &nnz, &pcp, &pri, &pv, &cpt, &rit, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(cols, 3);
    EXPECT_EQ(nnz, 3);
    EXPECT_EQ(cpt, rocsparse_indextype_i32);
    EXPECT_EQ(rit, rocsparse_indextype_i32);
    EXPECT_EQ(ib, rocsparse_index_base_zero);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    rocsparse_format fmt = rocsparse_format_coo;
    EXPECT_EQ(rocsparse_spmat_get_format(A, &fmt), rocsparse_status_success);
    EXPECT_EQ(fmt, rocsparse_format_csc);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxCsc, bad_args)
{
    device_vector<int32_t> col_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    rocsparse_spmat_descr  A = nullptr;

    EXPECT_EQ(rocsparse_create_csc_descr(&A,
                                         -1,
                                         3,
                                         3,
                                         col_ptr.ptr,
                                         row_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_create_csc_descr(nullptr,
                                         3,
                                         3,
                                         3,
                                         col_ptr.ptr,
                                         row_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_create_csc_descr(&A,
                                         3,
                                         3,
                                         3,
                                         col_ptr.ptr,
                                         row_ind.ptr,
                                         val.ptr,
                                         (rocsparse_indextype)99,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_value);
}

// ---------------------------------------------------------------------------
// ELL sparse matrix descriptor: create / get / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxEll, create_get_destroy)
{
    // rows*ell_width entries
    device_vector<int32_t> ell_col{std::vector<int32_t>(3 * 2, 0)};
    device_vector<float>   ell_val{std::vector<float>(3 * 2, 1.0f)};
    ASSERT_TRUE(ell_col.ptr && ell_val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(&A,
                                         3,
                                         3,
                                         ell_col.ptr,
                                         ell_val.ptr,
                                         2,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, width = 0;
    void *               pc = nullptr, *pv = nullptr;
    rocsparse_indextype  it = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_ell_get(A, &rows, &cols, &pc, &pv, &width, &it, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(cols, 3);
    EXPECT_EQ(width, 2);
    EXPECT_EQ(it, rocsparse_indextype_i32);
    EXPECT_EQ(ib, rocsparse_index_base_zero);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxEll, bad_args)
{
    device_vector<int32_t> ell_col{std::vector<int32_t>(3 * 2, 0)};
    device_vector<float>   ell_val{std::vector<float>(3 * 2, 1.0f)};
    rocsparse_spmat_descr  A = nullptr;

    // negative width
    EXPECT_EQ(rocsparse_create_ell_descr(&A,
                                         3,
                                         3,
                                         ell_col.ptr,
                                         ell_val.ptr,
                                         -1,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    // null out-param
    EXPECT_EQ(rocsparse_create_ell_descr(nullptr,
                                         3,
                                         3,
                                         ell_col.ptr,
                                         ell_val.ptr,
                                         2,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
    // invalid data type enum
    EXPECT_EQ(rocsparse_create_ell_descr(&A,
                                         3,
                                         3,
                                         ell_col.ptr,
                                         ell_val.ptr,
                                         2,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         (rocsparse_datatype)99),
              rocsparse_status_invalid_value);
}

// ---------------------------------------------------------------------------
// Blocked-ELL sparse matrix descriptor: create / get / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxBell, create_get_destroy)
{
    const int64_t          block_dim = 2;
    const int64_t          ell_cols  = 2;
    device_vector<int32_t> ell_col{std::vector<int32_t>(4, 0)};
    device_vector<float>   ell_val{std::vector<float>(4 * block_dim * block_dim, 1.0f)};
    ASSERT_TRUE(ell_col.ptr && ell_val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_bell_descr(&A,
                                          4,
                                          4,
                                          rocsparse_direction_row,
                                          block_dim,
                                          ell_cols,
                                          ell_col.ptr,
                                          ell_val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, bdim = 0, ecols = 0;
    rocsparse_direction  dir = rocsparse_direction_column;
    void *               pc = nullptr, *pv = nullptr;
    rocsparse_indextype  it = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(
        rocsparse_bell_get(A, &rows, &cols, &dir, &bdim, &ecols, &pc, &pv, &it, &ib, &dt),
        rocsparse_status_success);
    EXPECT_EQ(rows, 4);
    EXPECT_EQ(cols, 4);
    EXPECT_EQ(dir, rocsparse_direction_row);
    EXPECT_EQ(bdim, block_dim);
    EXPECT_EQ(it, rocsparse_indextype_i32);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxBell, bad_args)
{
    const int64_t          block_dim = 2;
    device_vector<int32_t> ell_col{std::vector<int32_t>(4, 0)};
    device_vector<float>   ell_val{std::vector<float>(4 * block_dim * block_dim, 1.0f)};
    rocsparse_spmat_descr  A = nullptr;

    // null out-param
    EXPECT_EQ(rocsparse_create_bell_descr(nullptr,
                                          4,
                                          4,
                                          rocsparse_direction_row,
                                          block_dim,
                                          2,
                                          ell_col.ptr,
                                          ell_val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
    // invalid direction enum
    EXPECT_EQ(rocsparse_create_bell_descr(&A,
                                          4,
                                          4,
                                          (rocsparse_direction)99,
                                          block_dim,
                                          2,
                                          ell_col.ptr,
                                          ell_val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_invalid_value);
}

// ---------------------------------------------------------------------------
// BSR sparse matrix descriptor: create / get / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxBsr, create_get_destroy)
{
    const int64_t          block_dim = 2;
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>(2 * block_dim * block_dim, 1.0f)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_bsr_descr(&A,
                                         2,
                                         2,
                                         2,
                                         rocsparse_direction_row,
                                         block_dim,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              brows = 0, bcols = 0, bnnz = 0, bdim = 0;
    rocsparse_direction  dir = rocsparse_direction_column;
    void *               prp = nullptr, *pc = nullptr, *pv = nullptr;
    rocsparse_indextype  pit = rocsparse_indextype_i64, cit = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_bsr_get(A,
                                &brows,
                                &bcols,
                                &bnnz,
                                &dir,
                                &bdim,
                                &prp,
                                &pc,
                                &pv,
                                &pit,
                                &cit,
                                &ib,
                                &dt),
              rocsparse_status_success);
    EXPECT_EQ(brows, 2);
    EXPECT_EQ(bcols, 2);
    EXPECT_EQ(bnnz, 2);
    EXPECT_EQ(dir, rocsparse_direction_row);
    EXPECT_EQ(bdim, block_dim);
    EXPECT_EQ(pit, rocsparse_indextype_i32);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxBsr, bad_args)
{
    const int64_t          block_dim = 2;
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>(2 * block_dim * block_dim, 1.0f)};
    rocsparse_spmat_descr  A = nullptr;

    // negative block dim
    EXPECT_EQ(rocsparse_create_bsr_descr(&A,
                                         2,
                                         2,
                                         2,
                                         rocsparse_direction_row,
                                         -1,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    // null out-param
    EXPECT_EQ(rocsparse_create_bsr_descr(nullptr,
                                         2,
                                         2,
                                         2,
                                         rocsparse_direction_row,
                                         block_dim,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// COO AoS sparse matrix descriptor: create / get / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxCooAos, create_get_destroy)
{
    // <row, col> interleaved indices => 2 * nnz entries
    device_vector<int32_t> ind{std::vector<int32_t>{0, 0, 1, 1, 2, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(ind.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_coo_aos_descr(&A,
                                             3,
                                             3,
                                             3,
                                             ind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, nnz = 0;
    void *               pi = nullptr, *pv = nullptr;
    rocsparse_indextype  it = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_coo_aos_get(A, &rows, &cols, &nnz, &pi, &pv, &it, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(cols, 3);
    EXPECT_EQ(nnz, 3);
    EXPECT_EQ(it, rocsparse_indextype_i32);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    rocsparse_format fmt = rocsparse_format_csr;
    EXPECT_EQ(rocsparse_spmat_get_format(A, &fmt), rocsparse_status_success);
    EXPECT_EQ(fmt, rocsparse_format_coo_aos);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxCooAos, bad_args)
{
    device_vector<int32_t> ind{std::vector<int32_t>{0, 0, 1, 1, 2, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    rocsparse_spmat_descr  A = nullptr;

    EXPECT_EQ(rocsparse_create_coo_aos_descr(&A,
                                             -1,
                                             3,
                                             3,
                                             ind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_create_coo_aos_descr(nullptr,
                                             3,
                                             3,
                                             3,
                                             ind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// const descriptor variants: create + const get + destroy.
// ---------------------------------------------------------------------------
TEST(AuxConst, csr_create_get_destroy)
{
    device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(ptr.ptr && col.ptr && val.ptr);

    rocsparse_const_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_const_csr_descr(&A,
                                               3,
                                               3,
                                               3,
                                               ptr.ptr,
                                               col.ptr,
                                               val.ptr,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows = 0, cols = 0, nnz = 0;
    const void *         prp = nullptr, *pc = nullptr, *pv = nullptr;
    rocsparse_indextype  pit = rocsparse_indextype_i64, cit = rocsparse_indextype_i64;
    rocsparse_index_base ib = rocsparse_index_base_one;
    rocsparse_datatype   dt = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_const_csr_get(A, &rows, &cols, &nnz, &prp, &pc, &pv, &pit, &cit, &ib, &dt),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(nnz, 3);
    EXPECT_EQ(pit, rocsparse_indextype_i32);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST(AuxConst, dnvec_and_dnmat)
{
    device_vector<float>        vv{std::vector<float>(8, 1.0f)};
    rocsparse_const_dnvec_descr y = nullptr;
    ASSERT_EQ(rocsparse_create_const_dnvec_descr(&y, 8, vv.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    int64_t            size = 0;
    const void*        pv   = nullptr;
    rocsparse_datatype dt   = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_const_dnvec_get(y, &size, &pv, &dt), rocsparse_status_success);
    EXPECT_EQ(size, 8);
    EXPECT_EQ(dt, rocsparse_datatype_f32_r);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(y), rocsparse_status_success);

    device_vector<float>        mv{std::vector<float>(3 * 4, 1.0f)};
    rocsparse_const_dnmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &A, 3, 4, 3, mv.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    int64_t         rows = 0, cols = 0, ld = 0;
    const void*     mpv = nullptr;
    rocsparse_order od  = rocsparse_order_row;
    dt                  = rocsparse_datatype_f64_r;
    EXPECT_EQ(rocsparse_const_dnmat_get(A, &rows, &cols, &ld, &mpv, &dt, &od),
              rocsparse_status_success);
    EXPECT_EQ(rows, 3);
    EXPECT_EQ(cols, 4);
    EXPECT_EQ(od, rocsparse_order_column);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Spmat values / nnz accessors + guards.
// ---------------------------------------------------------------------------
TEST(AuxSpMatValues, get_set_values_and_nnz)
{
    device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    device_vector<float>   val2{std::vector<float>{4, 5, 6}};
    ASSERT_TRUE(ptr.ptr && col.ptr && val.ptr && val2.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&A,
                                         3,
                                         3,
                                         3,
                                         ptr.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* pv = nullptr;
    EXPECT_EQ(rocsparse_spmat_get_values(A, &pv), rocsparse_status_success);
    EXPECT_EQ(pv, (void*)val.ptr);

    EXPECT_EQ(rocsparse_spmat_set_values(A, val2.ptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_spmat_get_values(A, &pv), rocsparse_status_success);
    EXPECT_EQ(pv, (void*)val2.ptr);

    int64_t nnz = 0;
    EXPECT_EQ(rocsparse_spmat_get_nnz(A, &nnz), rocsparse_status_success);
    EXPECT_EQ(nnz, 3);

    // bad args
    EXPECT_EQ(rocsparse_spmat_get_values(A, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_get_values(nullptr, &pv), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_set_values(A, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_get_nnz(A, nullptr), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Spmat strided batch get/set + guards.
// ---------------------------------------------------------------------------
TEST(AuxSpMatBatch, get_set_strided_batch)
{
    device_vector<int32_t> row{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(row.ptr && col.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&A,
                                         3,
                                         3,
                                         3,
                                         row.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse_spmat_set_strided_batch(A, 4), rocsparse_status_success);
    rocsparse_int batch_count = 0;
    EXPECT_EQ(rocsparse_spmat_get_strided_batch(A, &batch_count), rocsparse_status_success);
    EXPECT_EQ(batch_count, 4);

    // bad args
    EXPECT_EQ(rocsparse_spmat_set_strided_batch(A, 0), rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_spmat_set_strided_batch(nullptr, 1), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_get_strided_batch(nullptr, &batch_count),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Spmat attribute set/get roundtrip + guards.
// ---------------------------------------------------------------------------
TEST(AuxSpMatAttr, set_get_attribute)
{
    device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1, 2, 3}};
    ASSERT_TRUE(ptr.ptr && col.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&A,
                                         3,
                                         3,
                                         3,
                                         ptr.ptr,
                                         col.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // matrix type roundtrip
    rocsparse_matrix_type mt_in = rocsparse_matrix_type_symmetric;
    EXPECT_EQ(rocsparse_spmat_set_attribute(
                  A, rocsparse_spmat_matrix_type, &mt_in, sizeof(mt_in)),
              rocsparse_status_success);
    rocsparse_matrix_type mt_out = rocsparse_matrix_type_general;
    EXPECT_EQ(rocsparse_spmat_get_attribute(
                  A, rocsparse_spmat_matrix_type, &mt_out, sizeof(mt_out)),
              rocsparse_status_success);
    EXPECT_EQ(mt_out, rocsparse_matrix_type_symmetric);

    // fill mode roundtrip
    rocsparse_fill_mode fm_in = rocsparse_fill_mode_upper;
    EXPECT_EQ(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &fm_in, sizeof(fm_in)),
        rocsparse_status_success);
    rocsparse_fill_mode fm_out = rocsparse_fill_mode_lower;
    EXPECT_EQ(
        rocsparse_spmat_get_attribute(A, rocsparse_spmat_fill_mode, &fm_out, sizeof(fm_out)),
        rocsparse_status_success);
    EXPECT_EQ(fm_out, rocsparse_fill_mode_upper);

    // storage mode roundtrip
    rocsparse_storage_mode sm_in = rocsparse_storage_mode_unsorted;
    EXPECT_EQ(rocsparse_spmat_set_attribute(
                  A, rocsparse_spmat_storage_mode, &sm_in, sizeof(sm_in)),
              rocsparse_status_success);
    rocsparse_storage_mode sm_out = rocsparse_storage_mode_sorted;
    EXPECT_EQ(rocsparse_spmat_get_attribute(
                  A, rocsparse_spmat_storage_mode, &sm_out, sizeof(sm_out)),
              rocsparse_status_success);
    EXPECT_EQ(sm_out, rocsparse_storage_mode_unsorted);

    // bad args
    EXPECT_EQ(rocsparse_spmat_set_attribute(
                  nullptr, rocsparse_spmat_fill_mode, &fm_in, sizeof(fm_in)),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, nullptr, sizeof(fm_in)),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_spmat_set_attribute(
                  A, (rocsparse_spmat_attribute)99, &fm_in, sizeof(fm_in)),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_spmat_get_attribute(A, rocsparse_spmat_fill_mode, &fm_out, 0),
              rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Dense vector values accessors + guards.
// ---------------------------------------------------------------------------
TEST(AuxDnVecValues, get_set_values)
{
    device_vector<float>  val{std::vector<float>(8, 1.0f)};
    device_vector<float>  val2{std::vector<float>(8, 2.0f)};
    rocsparse_dnvec_descr y = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&y, 8, val.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* pv = nullptr;
    EXPECT_EQ(rocsparse_dnvec_get_values(y, &pv), rocsparse_status_success);
    EXPECT_EQ(pv, (void*)val.ptr);

    EXPECT_EQ(rocsparse_dnvec_set_values(y, val2.ptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_dnvec_get_values(y, &pv), rocsparse_status_success);
    EXPECT_EQ(pv, (void*)val2.ptr);

    // bad args
    EXPECT_EQ(rocsparse_dnvec_get_values(y, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dnvec_get_values(nullptr, &pv), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dnvec_set_values(y, nullptr), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(y), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Dense matrix values accessors + guards.
// ---------------------------------------------------------------------------
TEST(AuxDnMatValues, get_set_values)
{
    device_vector<float>  val{std::vector<float>(3 * 4, 1.0f)};
    device_vector<float>  val2{std::vector<float>(3 * 4, 2.0f)};
    rocsparse_dnmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &A, 3, 4, 3, val.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    void* pv = nullptr;
    EXPECT_EQ(rocsparse_dnmat_get_values(A, &pv), rocsparse_status_success);
    EXPECT_EQ(pv, (void*)val.ptr);

    EXPECT_EQ(rocsparse_dnmat_set_values(A, val2.ptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_dnmat_get_values(A, &pv), rocsparse_status_success);
    EXPECT_EQ(pv, (void*)val2.ptr);

    // bad args
    EXPECT_EQ(rocsparse_dnmat_get_values(A, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dnmat_get_values(nullptr, &pv), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dnmat_set_values(A, nullptr), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Dense matrix strided batch get/set + guards.
// ---------------------------------------------------------------------------
TEST(AuxDnMatBatch, get_set_strided_batch)
{
    device_vector<float>  val{std::vector<float>(3 * 4, 1.0f)};
    rocsparse_dnmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &A, 3, 4, 3, val.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    // ld * cols = 3 * 4 = 12 => batch_stride must be >= 12 for batch_count > 1
    EXPECT_EQ(rocsparse_dnmat_set_strided_batch(A, 2, 12), rocsparse_status_success);
    rocsparse_int batch_count  = 0;
    int64_t       batch_stride = 0;
    EXPECT_EQ(rocsparse_dnmat_get_strided_batch(A, &batch_count, &batch_stride),
              rocsparse_status_success);
    EXPECT_EQ(batch_count, 2);
    EXPECT_EQ(batch_stride, 12);

    // bad args
    EXPECT_EQ(rocsparse_dnmat_set_strided_batch(A, 0, 12), rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_dnmat_set_strided_batch(A, 2, -1), rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_dnmat_set_strided_batch(nullptr, 2, 12), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_dnmat_get_strided_batch(nullptr, &batch_count, &batch_stride),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_dnmat_descr(A), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Handle git revision + pointer-mode bad-args.
// ---------------------------------------------------------------------------
TEST_F(AuxHandle, git_rev_and_pointer_mode_bad_args)
{
    char rev[128] = {0};
    EXPECT_EQ(rocsparse_get_git_rev(handle, rev), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_get_git_rev(nullptr, rev), rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_get_git_rev(handle, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_get_pointer_mode(nullptr, nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_set_pointer_mode(nullptr, rocsparse_pointer_mode_host),
              rocsparse_status_invalid_handle);
}

// ---------------------------------------------------------------------------
// Matrix info structure: create / copy / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxMatInfo, create_copy_destroy)
{
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    rocsparse_mat_info dst = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dst), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dst, info), rocsparse_status_success);

    EXPECT_EQ(rocsparse_destroy_mat_info(dst), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_create_mat_info(nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_copy_mat_info(nullptr, nullptr), rocsparse_status_invalid_pointer);
    // destroying a null info structure is a no-op success
    EXPECT_EQ(rocsparse_destroy_mat_info(nullptr), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// Color info structure: create / destroy + guards.
// ---------------------------------------------------------------------------
TEST(AuxColorInfo, create_destroy)
{
    rocsparse_color_info info = nullptr;
    ASSERT_EQ(rocsparse_create_color_info(&info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_color_info(info), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_create_color_info(nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_destroy_color_info(nullptr), rocsparse_status_success);
}
