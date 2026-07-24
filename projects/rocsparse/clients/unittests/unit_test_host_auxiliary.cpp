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
