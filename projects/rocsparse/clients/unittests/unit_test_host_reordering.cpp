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
// Host-path unit tests for the reordering sub-lib (library/src/reordering).
// Drives the public rocsparse_Xcsrcolor C API on a tiny 3x3 diagonal CSR to
// exercise the host dispatch / quick-return / argument-check code paths that
// rocSPARSE host-only line coverage counts.
//
// Notes on pointer residency for csrcolor:
//   * csr_val / csr_row_ptr / csr_col_ind / coloring / reordering  -> DEVICE
//   * fraction_to_color and ncolors                                -> HOST
//     (fraction_to_color[0] and *ncolors are dereferenced on the host inside
//      csrcolor_core, see library/src/reordering/rocsparse_csrcolor.cpp).
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

class Reordering : public HandleTest
{
};

namespace
{
    // Signature of the C csrcolor entry points, parameterized by the value type T
    // and the real base type R of fraction_to_color (float for s/c, double for d/z).
    template <typename T, typename R>
    using csrcolor_fn = rocsparse_status (*)(rocsparse_handle,
                                             rocsparse_int,
                                             rocsparse_int,
                                             const rocsparse_mat_descr,
                                             const T*,
                                             const rocsparse_int*,
                                             const rocsparse_int*,
                                             const R*,
                                             rocsparse_int*,
                                             rocsparse_int*,
                                             rocsparse_int*,
                                             rocsparse_mat_info);

    // Valid tiny call across a supported precision. Optionally requests the
    // reordering permutation to exercise the reordering branch of csrcolor_core.
    template <typename T, typename R>
    void run_csrcolor_valid(rocsparse_handle handle, csrcolor_fn<T, R> fn, bool with_reordering)
    {
        const rocsparse_int m = 3, nnz = 3;

        // 3x3 diagonal (symmetric-pattern) CSR.
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T>             val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
        device_vector<rocsparse_int> coloring{(size_t)m};
        device_vector<rocsparse_int> reordering{(size_t)m};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && coloring.ptr && reordering.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        const R       fraction = static_cast<R>(1.0);
        rocsparse_int ncolors  = 0;

        EXPECT_EQ(fn(handle,
                     m,
                     nnz,
                     descr,
                     val,
                     row_ptr,
                     col_ind,
                     &fraction,
                     &ncolors,
                     coloring,
                     with_reordering ? reordering.ptr : nullptr,
                     info),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_GT(ncolors, 0);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }
}

TEST_F(Reordering, csrcolor_valid_float)
{
    run_csrcolor_valid<float, float>(handle, rocsparse_scsrcolor, false);
}

TEST_F(Reordering, csrcolor_valid_double)
{
    run_csrcolor_valid<double, double>(handle, rocsparse_dcsrcolor, false);
}

TEST_F(Reordering, csrcolor_valid_float_complex)
{
    run_csrcolor_valid<rocsparse_float_complex, float>(handle, rocsparse_ccsrcolor, false);
}

TEST_F(Reordering, csrcolor_valid_double_complex)
{
    run_csrcolor_valid<rocsparse_double_complex, double>(handle, rocsparse_zcsrcolor, false);
}

TEST_F(Reordering, csrcolor_with_reordering)
{
    // Exercise the optional reordering permutation branch (radix sort of colors).
    run_csrcolor_valid<float, float>(handle, rocsparse_scsrcolor, true);
}

TEST_F(Reordering, csrcolor_quick_return)
{
    // m == 0 is a valid quick-return (success) before any pointer checks.
    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const float   fraction = 1.0f;
    rocsparse_int ncolors  = 0;

    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  0,
                                  0,
                                  descr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  &fraction,
                                  &ncolors,
                                  nullptr,
                                  nullptr,
                                  info),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(Reordering, csrcolor_bad_args)
{
    const rocsparse_int m = 3, nnz = 3;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{1, 1, 1}};
    device_vector<rocsparse_int> coloring{(size_t)m};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && coloring.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const float   fraction = 1.0f;
    rocsparse_int ncolors  = 0;

    // null handle -> invalid_handle
    EXPECT_EQ(rocsparse_scsrcolor(nullptr,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_handle);

    // negative m -> invalid_size
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  -1,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_size);

    // negative nnz -> invalid_size
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  -1,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_size);

    // null descr -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  nullptr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null csr_val -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  nullptr,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null csr_row_ptr -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  nullptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null csr_col_ind -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  nullptr,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null fraction_to_color -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  nullptr,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null ncolors -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  nullptr,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null coloring -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  nullptr,
                                  nullptr,
                                  info),
              rocsparse_status_invalid_pointer);

    // null info -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(Reordering, csrcolor_unsupported_descr)
{
    // A non-general matrix type is not implemented for csrcolor; setting an
    // invalid/unsupported descriptor value exercises the descriptor guard.
    const rocsparse_int m = 3, nnz = 3;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{1, 1, 1}};
    device_vector<rocsparse_int> coloring{(size_t)m};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && coloring.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const float   fraction = 1.0f;
    rocsparse_int ncolors  = 0;

    EXPECT_EQ(rocsparse_scsrcolor(handle,
                                  m,
                                  nnz,
                                  descr,
                                  val,
                                  row_ptr,
                                  col_ind,
                                  &fraction,
                                  &ncolors,
                                  coloring,
                                  nullptr,
                                  info),
              rocsparse_status_not_implemented);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}
