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
// Host-path unit tests for the util matrix-validation C API:
//   rocsparse_Xcheck_matrix_{csc,ell,gebsr,gebsc} (+ _buffer_size) and
//   rocsparse_check_matrix_hyb (+ _buffer_size).
//
// These public entry points live in library/src/util/rocsparse_check_matrix_*.
// They were previously uncovered (0%) because the existing suite only drove the
// csr/coo variants. Each test drives the buffer-size query, a valid tiny matrix
// (so the device validation kernel runs and reports rocsparse_data_status), and
// a spread of bad-argument branches (invalid handle / size / enum / null array).
// Inputs are <= 2x2 so the suite never trips the "Insufficient memory" skip.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_index_base   BASE    = rocsparse_index_base_zero;
    constexpr rocsparse_matrix_type  GENERAL = rocsparse_matrix_type_general;
    constexpr rocsparse_fill_mode    LOWER   = rocsparse_fill_mode_lower;
    constexpr rocsparse_storage_mode SORTED  = rocsparse_storage_mode_sorted;
}

// ===========================================================================
// check_matrix_csc  (delegates to check_matrix_csr_core with m/n switched)
// ===========================================================================
class CheckMatrixCsc : public HandleTest
{
};

// check_matrix_csc on a valid 2x2 CSC identity: the buffer-size query and the
// device validation must both succeed and report data_status == success. (The
// query result is a prerequisite of the check, so the two stages stay together.)
TEST_F(CheckMatrixCsc, buffer_size_then_check)
{
    // 2x2 identity in CSC: col_ptr = {0,1,2}, row_ind = {0,1}, val = {1,1}.
    device_vector<int32_t> col_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    ASSERT_TRUE(col_ptr.ptr && row_ind.ptr && val.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(
        rocsparse_scheck_matrix_csc_buffer_size(
            handle, 2, 2, 2, val, col_ptr, row_ind, BASE, GENERAL, LOWER, SORTED, &buffer_size),
        rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_data_status data_status;
    ASSERT_EQ(rocsparse_scheck_matrix_csc(handle,
                                          2,
                                          2,
                                          2,
                                          val,
                                          col_ptr,
                                          row_ind,
                                          BASE,
                                          GENERAL,
                                          LOWER,
                                          SORTED,
                                          &data_status,
                                          tmp.ptr),
              rocsparse_status_success);
    EXPECT_EQ(data_status, rocsparse_data_status_success);
}

// check_matrix_csc bad args: invalid handle / negative size / invalid base enum
// / null value array on the buffer-size query, and null temp buffer on the check.
TEST_F(CheckMatrixCsc, bad_args)
{
    device_vector<int32_t> col_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    size_t                 bs = 0;

    EXPECT_EQ(rocsparse_scheck_matrix_csc_buffer_size(
                  nullptr, 2, 2, 2, val, col_ptr, row_ind, BASE, GENERAL, LOWER, SORTED, &bs),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_scheck_matrix_csc_buffer_size(
                  handle, -1, 2, 2, val, col_ptr, row_ind, BASE, GENERAL, LOWER, SORTED, &bs),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_scheck_matrix_csc_buffer_size(handle,
                                                      2,
                                                      2,
                                                      2,
                                                      val,
                                                      col_ptr,
                                                      row_ind,
                                                      static_cast<rocsparse_index_base>(-1),
                                                      GENERAL,
                                                      LOWER,
                                                      SORTED,
                                                      &bs),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_scheck_matrix_csc_buffer_size(
                  handle, 2, 2, 2, nullptr, col_ptr, row_ind, BASE, GENERAL, LOWER, SORTED, &bs),
              rocsparse_status_invalid_pointer);

    rocsparse_data_status ds;
    EXPECT_EQ(
        rocsparse_scheck_matrix_csc(
            handle, 2, 2, 2, val, col_ptr, row_ind, BASE, GENERAL, LOWER, SORTED, &ds, nullptr),
        rocsparse_status_invalid_pointer);
}

// ===========================================================================
// check_matrix_ell  (own device validation kernel)
// ===========================================================================
class CheckMatrixEll : public HandleTest
{
};

// check_matrix_ell on a valid 2x2 ELL identity (ell_width 1): buffer-size query
// and validation succeed with data_status == success.
TEST_F(CheckMatrixEll, buffer_size_then_check)
{
    // 2x2 identity in ELL: ell_width = 1, val = {1,1}, col_ind = {0,1}.
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    ASSERT_TRUE(col_ind.ptr && val.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scheck_matrix_ell_buffer_size(
                  handle, 2, 2, 1, val, col_ind, BASE, GENERAL, LOWER, SORTED, &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_data_status data_status;
    ASSERT_EQ(
        rocsparse_scheck_matrix_ell(
            handle, 2, 2, 1, val, col_ind, BASE, GENERAL, LOWER, SORTED, &data_status, tmp.ptr),
        rocsparse_status_success);
    EXPECT_EQ(data_status, rocsparse_data_status_success);
}

// check_matrix_ell bad args: invalid handle, negative size, and the ELL-only
// restriction that non-general / invalid matrix_type enums are rejected.
TEST_F(CheckMatrixEll, bad_args)
{
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    size_t                 bs = 0;

    EXPECT_EQ(rocsparse_scheck_matrix_ell_buffer_size(
                  nullptr, 2, 2, 1, val, col_ind, BASE, GENERAL, LOWER, SORTED, &bs),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_scheck_matrix_ell_buffer_size(
                  handle, -1, 2, 1, val, col_ind, BASE, GENERAL, LOWER, SORTED, &bs),
              rocsparse_status_invalid_size);
    // ELL only supports general matrices: a symmetric matrix_type is rejected.
    EXPECT_EQ(rocsparse_scheck_matrix_ell_buffer_size(handle,
                                                      2,
                                                      2,
                                                      1,
                                                      val,
                                                      col_ind,
                                                      BASE,
                                                      rocsparse_matrix_type_symmetric,
                                                      LOWER,
                                                      SORTED,
                                                      &bs),
              rocsparse_status_invalid_value);
    EXPECT_EQ(rocsparse_scheck_matrix_ell_buffer_size(handle,
                                                      2,
                                                      2,
                                                      1,
                                                      val,
                                                      col_ind,
                                                      BASE,
                                                      static_cast<rocsparse_matrix_type>(-1),
                                                      LOWER,
                                                      SORTED,
                                                      &bs),
              rocsparse_status_invalid_value);
}

// ===========================================================================
// check_matrix_gebsr  (own device validation kernel)
// ===========================================================================
class CheckMatrixGebsr : public HandleTest
{
};

// check_matrix_gebsr on a valid 2x2 block-identity (1x1 blocks): buffer-size
// query and validation succeed with data_status == success.
TEST_F(CheckMatrixGebsr, buffer_size_then_check)
{
    // 2x2 block identity, 1x1 blocks: row_ptr = {0,1,2}, col_ind = {0,1}, val = {1,1}.
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    const rocsparse_direction dir         = rocsparse_direction_row;
    size_t                    buffer_size = 0;
    ASSERT_EQ(rocsparse_scheck_matrix_gebsr_buffer_size(handle,
                                                        dir,
                                                        2,
                                                        2,
                                                        2,
                                                        1,
                                                        1,
                                                        val,
                                                        row_ptr,
                                                        col_ind,
                                                        BASE,
                                                        GENERAL,
                                                        LOWER,
                                                        SORTED,
                                                        &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_data_status data_status;
    ASSERT_EQ(rocsparse_scheck_matrix_gebsr(handle,
                                            dir,
                                            2,
                                            2,
                                            2,
                                            1,
                                            1,
                                            val,
                                            row_ptr,
                                            col_ind,
                                            BASE,
                                            GENERAL,
                                            LOWER,
                                            SORTED,
                                            &data_status,
                                            tmp.ptr),
              rocsparse_status_success);
    EXPECT_EQ(data_status, rocsparse_data_status_success);
}

// check_matrix_gebsr bad args: invalid handle, invalid direction enum, negative
// size, and null buffer-size pointer are each rejected.
TEST_F(CheckMatrixGebsr, bad_args)
{
    device_vector<int32_t>    row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t>    col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>      val{std::vector<float>{1.0f, 1.0f}};
    const rocsparse_direction dir = rocsparse_direction_row;
    size_t                    bs  = 0;

    EXPECT_EQ(
        rocsparse_scheck_matrix_gebsr_buffer_size(
            nullptr, dir, 2, 2, 2, 1, 1, val, row_ptr, col_ind, BASE, GENERAL, LOWER, SORTED, &bs),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_scheck_matrix_gebsr_buffer_size(handle,
                                                        static_cast<rocsparse_direction>(-1),
                                                        2,
                                                        2,
                                                        2,
                                                        1,
                                                        1,
                                                        val,
                                                        row_ptr,
                                                        col_ind,
                                                        BASE,
                                                        GENERAL,
                                                        LOWER,
                                                        SORTED,
                                                        &bs),
              rocsparse_status_invalid_value);
    EXPECT_EQ(
        rocsparse_scheck_matrix_gebsr_buffer_size(
            handle, dir, -1, 2, 2, 1, 1, val, row_ptr, col_ind, BASE, GENERAL, LOWER, SORTED, &bs),
        rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_scheck_matrix_gebsr_buffer_size(handle,
                                                        dir,
                                                        2,
                                                        2,
                                                        2,
                                                        1,
                                                        1,
                                                        val,
                                                        row_ptr,
                                                        col_ind,
                                                        BASE,
                                                        GENERAL,
                                                        LOWER,
                                                        SORTED,
                                                        nullptr),
              rocsparse_status_invalid_pointer);
}

// ===========================================================================
// check_matrix_gebsc  (own device validation kernel)
// ===========================================================================
class CheckMatrixGebsc : public HandleTest
{
};

// check_matrix_gebsc on a valid 2x2 block-identity in GEBSC (1x1 blocks):
// buffer-size query and validation succeed with data_status == success.
TEST_F(CheckMatrixGebsc, buffer_size_then_check)
{
    // 2x2 block identity in GEBSC, 1x1 blocks: col_ptr={0,1,2}, row_ind={0,1}.
    device_vector<int32_t> col_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
    ASSERT_TRUE(col_ptr.ptr && row_ind.ptr && val.ptr);

    const rocsparse_direction dir         = rocsparse_direction_row;
    size_t                    buffer_size = 0;
    ASSERT_EQ(rocsparse_scheck_matrix_gebsc_buffer_size(handle,
                                                        dir,
                                                        2,
                                                        2,
                                                        2,
                                                        1,
                                                        1,
                                                        val,
                                                        col_ptr,
                                                        row_ind,
                                                        BASE,
                                                        GENERAL,
                                                        LOWER,
                                                        SORTED,
                                                        &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_data_status data_status;
    ASSERT_EQ(rocsparse_scheck_matrix_gebsc(handle,
                                            dir,
                                            2,
                                            2,
                                            2,
                                            1,
                                            1,
                                            val,
                                            col_ptr,
                                            row_ind,
                                            BASE,
                                            GENERAL,
                                            LOWER,
                                            SORTED,
                                            &data_status,
                                            tmp.ptr),
              rocsparse_status_success);
    EXPECT_EQ(data_status, rocsparse_data_status_success);
}

// check_matrix_gebsc bad args: invalid handle and null buffer-size pointer are
// rejected by the buffer-size query.
TEST_F(CheckMatrixGebsc, bad_args)
{
    device_vector<int32_t>    col_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t>    row_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>      val{std::vector<float>{1.0f, 1.0f}};
    const rocsparse_direction dir = rocsparse_direction_row;
    size_t                    bs  = 0;

    EXPECT_EQ(
        rocsparse_scheck_matrix_gebsc_buffer_size(
            nullptr, dir, 2, 2, 2, 1, 1, val, col_ptr, row_ind, BASE, GENERAL, LOWER, SORTED, &bs),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_scheck_matrix_gebsc_buffer_size(handle,
                                                        dir,
                                                        2,
                                                        2,
                                                        2,
                                                        1,
                                                        1,
                                                        val,
                                                        col_ptr,
                                                        row_ind,
                                                        BASE,
                                                        GENERAL,
                                                        LOWER,
                                                        SORTED,
                                                        nullptr),
              rocsparse_status_invalid_pointer);
}

// ===========================================================================
// check_matrix_hyb  (build a tiny HYB via csr2hyb, then validate it)
// ===========================================================================
class CheckMatrixHyb : public HandleTest
{
};

// check_matrix_hyb: build a HYB from a 3x3 identity CSR (csr2hyb, auto
// partition) then validate it; the buffer-size query and check succeed with
// data_status == success.
TEST_F(CheckMatrixHyb, build_then_check)
{
    // 3x3 identity CSR -> HYB (auto partition).
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    rocsparse_hyb_mat hyb = nullptr;
    ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);

    ASSERT_EQ(rocsparse_scsr2hyb(
                  handle, 3, 3, descr, val, row_ptr, col_ind, hyb, 0, rocsparse_hyb_partition_auto),
              rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_check_matrix_hyb_buffer_size(
                  handle, hyb, BASE, GENERAL, LOWER, SORTED, &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_data_status data_status;
    ASSERT_EQ(rocsparse_check_matrix_hyb(
                  handle, hyb, BASE, GENERAL, LOWER, SORTED, &data_status, tmp.ptr),
              rocsparse_status_success);
    EXPECT_EQ(data_status, rocsparse_data_status_success);

    EXPECT_EQ(rocsparse_destroy_hyb_mat(hyb), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// check_matrix_hyb bad args: invalid handle, null hyb matrix, null buffer-size
// pointer, and null temp buffer on the check are each rejected.
TEST_F(CheckMatrixHyb, bad_args)
{
    rocsparse_hyb_mat hyb = nullptr;
    ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);

    size_t bs = 0;
    EXPECT_EQ(
        rocsparse_check_matrix_hyb_buffer_size(nullptr, hyb, BASE, GENERAL, LOWER, SORTED, &bs),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_check_matrix_hyb_buffer_size(handle, nullptr, BASE, GENERAL, LOWER, SORTED, &bs),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_check_matrix_hyb_buffer_size(handle, hyb, BASE, GENERAL, LOWER, SORTED, nullptr),
        rocsparse_status_invalid_pointer);

    rocsparse_data_status ds;
    EXPECT_EQ(rocsparse_check_matrix_hyb(handle, hyb, BASE, GENERAL, LOWER, SORTED, &ds, nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_hyb_mat(hyb), rocsparse_status_success);
}
