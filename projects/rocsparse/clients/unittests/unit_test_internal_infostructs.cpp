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
// Device-target unit tests for rocSPARSE internal data-structure member
// functions (info structs).
//
// Compiled into the rocsparse-unit-test-device binary. Exercises member
// functions of the internal info/handle structures directly (they are private
// to the library and hidden in the .so, so the matching library/src TUs that
// define them are compiled into this binary -- see CMakeLists.txt,
// ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES). Coverage:
//
//   * rocsparse::numeric_boost           (common/rocsparse_numeric_boost.cpp)
//   * _rocsparse_csrmv_info::clear + adaptive/lrb/nnzsplit clear
//         (header-inline clear; level2/rocsparse_csrmv_{adaptive,lrb,nnzsplit}_info.cpp)
//   * rocsparse csrgemm_info create/copy/destroy (extra/rocsparse_csrgemm_info.cpp)
//   * _rocsparse_mat_info lifecycle via the public C API (rocsparse_create/destroy_mat_info)
//
#include "unit_test_utils.hpp"

// Internal (private) library headers under test. Reachable because the device
// unit-test target adds library/src/include to its include path.
#include "rocsparse_csrgemm_info.hpp"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_numeric_boost.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

using namespace rocsparse_ut;

// ===========================================================================
// harness smoke
// ===========================================================================
TEST(internal_infostructs, harness_smoke)
{
    SUCCEED();
}

// ===========================================================================
// rocsparse::numeric_boost
// ===========================================================================
TEST(internal_infostructs, numeric_boost_default_ctor)
{
    rocsparse::numeric_boost b;
    EXPECT_EQ(b.get_enable(), 0);
    EXPECT_EQ(b.get_tol(), nullptr);
    EXPECT_EQ(b.get_val(), nullptr);
    EXPECT_EQ(b.get_tol_datatype(), rocsparse_datatype_f32_r);
    EXPECT_EQ(b.get_tol_pointer_mode(), static_cast<rocsparse_pointer_mode>(-1));
    EXPECT_EQ(b.get_val_pointer_mode(), static_cast<rocsparse_pointer_mode>(-1));
}

TEST(internal_infostructs, numeric_boost_define)
{
    rocsparse::numeric_boost b;

    const double tol = 1.0e-8;
    const double val = 2.5;

    b.define(1,
             rocsparse_pointer_mode_host,
             rocsparse_datatype_f64_r,
             &tol,
             rocsparse_pointer_mode_device,
             &val);

    EXPECT_EQ(b.get_enable(), 1);
    EXPECT_EQ(b.get_tol_pointer_mode(), rocsparse_pointer_mode_host);
    EXPECT_EQ(b.get_tol_datatype(), rocsparse_datatype_f64_r);
    EXPECT_EQ(b.get_tol(), &tol);
    EXPECT_EQ(b.get_val_pointer_mode(), rocsparse_pointer_mode_device);
    EXPECT_EQ(b.get_val(), &val);
}

TEST(internal_infostructs, numeric_boost_setters)
{
    rocsparse::numeric_boost b;

    const float tol = 3.0f;
    const float val = 4.0f;

    b.set_enable(7);
    b.set_tol(&tol);
    b.set_val(&val);
    b.set_tol_datatype(rocsparse_datatype_f32_c);
    b.set_tol_pointer_mode(rocsparse_pointer_mode_device);
    b.set_val_pointer_mode(rocsparse_pointer_mode_host);

    EXPECT_EQ(b.get_enable(), 7);
    EXPECT_EQ(b.get_tol(), &tol);
    EXPECT_EQ(b.get_val(), &val);
    EXPECT_EQ(b.get_tol_datatype(), rocsparse_datatype_f32_c);
    EXPECT_EQ(b.get_tol_pointer_mode(), rocsparse_pointer_mode_device);
    EXPECT_EQ(b.get_val_pointer_mode(), rocsparse_pointer_mode_host);
}

TEST(internal_infostructs, numeric_boost_copy)
{
    const double tol = 1.0e-6;
    const double val = 9.0;

    rocsparse::numeric_boost src;
    src.define(1,
               rocsparse_pointer_mode_host,
               rocsparse_datatype_f64_r,
               &tol,
               rocsparse_pointer_mode_device,
               &val);

    rocsparse::numeric_boost dst;
    dst.copy(src);

    EXPECT_EQ(dst.get_enable(), src.get_enable());
    EXPECT_EQ(dst.get_tol_pointer_mode(), src.get_tol_pointer_mode());
    EXPECT_EQ(dst.get_val_pointer_mode(), src.get_val_pointer_mode());
    EXPECT_EQ(dst.get_tol_datatype(), src.get_tol_datatype());
    EXPECT_EQ(dst.get_tol(), src.get_tol());
    EXPECT_EQ(dst.get_val(), src.get_val());
}

// ===========================================================================
// _rocsparse_csrmv_info::clear (+ adaptive/lrb/nnzsplit clear)
// ===========================================================================
TEST(internal_infostructs, csrmv_info_clear_resets_fields)
{
    _rocsparse_csrmv_info info;

    // Dirty the scalar bookkeeping fields.
    info.trans    = rocsparse_operation_transpose;
    info.m        = 7;
    info.n        = 8;
    info.nnz      = 42;
    info.max_rows = 3;

    // Dirty the sub-info sizes (device pointers stay null -> clear() only frees
    // nullptrs, which is safe).
    info.adaptive.size = 9;
    info.lrb.size      = 10;
    info.nnzsplit.size = 11;

    info.clear();

    EXPECT_EQ(info.trans, rocsparse_operation_none);
    EXPECT_EQ(info.m, 0);
    EXPECT_EQ(info.n, 0);
    EXPECT_EQ(info.nnz, 0);
    EXPECT_EQ(info.max_rows, 0);
    EXPECT_EQ(info.csr_row_ptr, nullptr);
    EXPECT_EQ(info.csr_col_ind, nullptr);

    EXPECT_EQ(info.adaptive.size, 0u);
    EXPECT_EQ(info.adaptive.row_blocks, nullptr);
    EXPECT_EQ(info.lrb.size, 0u);
    EXPECT_EQ(info.nnzsplit.size, 0u);
}

TEST(internal_infostructs, csrmv_subinfo_clear_null_safe)
{
    _rocsparse_adaptive_info adaptive;
    _rocsparse_lrb_info      lrb;
    _rocsparse_nnzsplit_info nnzsplit;

    adaptive.clear();
    lrb.clear();
    nnzsplit.clear();

    EXPECT_EQ(adaptive.size, 0u);
    EXPECT_EQ(adaptive.row_blocks, nullptr);
    EXPECT_EQ(adaptive.wg_flags, nullptr);
    EXPECT_EQ(adaptive.wg_ids, nullptr);

    EXPECT_EQ(lrb.size, 0u);
    EXPECT_EQ(lrb.wg_flags, nullptr);
    EXPECT_EQ(lrb.rows_bins, nullptr);

    EXPECT_EQ(nnzsplit.size, 0u);
    EXPECT_EQ(nnzsplit.starting_ids, nullptr);
    EXPECT_EQ(nnzsplit.starting_block_ids, nullptr);
}

// ===========================================================================
// rocsparse csrgemm_info create / copy / destroy
// ===========================================================================
TEST(internal_infostructs, csrgemm_info_create_defaults)
{
    rocsparse_csrgemm_info info = nullptr;
    ASSERT_EQ(rocsparse::create_csrgemm_info(&info), rocsparse_status_success);
    ASSERT_NE(info, nullptr);

    EXPECT_EQ(info->buffer_size, 0u);
    EXPECT_FALSE(info->is_initialized);
    EXPECT_TRUE(info->mul);
    EXPECT_TRUE(info->add);

    EXPECT_EQ(rocsparse::destroy_csrgemm_info(info), rocsparse_status_success);
}

TEST(internal_infostructs, csrgemm_info_create_invalid_pointer)
{
    EXPECT_EQ(rocsparse::create_csrgemm_info(nullptr), rocsparse_status_invalid_pointer);
}

TEST(internal_infostructs, csrgemm_info_copy)
{
    rocsparse_csrgemm_info src = nullptr;
    rocsparse_csrgemm_info dst = nullptr;
    ASSERT_EQ(rocsparse::create_csrgemm_info(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse::create_csrgemm_info(&dst), rocsparse_status_success);

    src->buffer_size    = 4096;
    src->is_initialized = true;
    src->mul            = false;
    src->add            = false;

    ASSERT_EQ(rocsparse::copy_csrgemm_info(dst, src), rocsparse_status_success);

    EXPECT_EQ(dst->buffer_size, 4096u);
    EXPECT_TRUE(dst->is_initialized);
    EXPECT_FALSE(dst->mul);
    EXPECT_FALSE(dst->add);

    EXPECT_EQ(rocsparse::destroy_csrgemm_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse::destroy_csrgemm_info(dst), rocsparse_status_success);
}

TEST(internal_infostructs, csrgemm_info_copy_invalid_pointer)
{
    rocsparse_csrgemm_info info = nullptr;
    ASSERT_EQ(rocsparse::create_csrgemm_info(&info), rocsparse_status_success);

    EXPECT_EQ(rocsparse::copy_csrgemm_info(nullptr, info), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::copy_csrgemm_info(info, nullptr), rocsparse_status_invalid_pointer);
    // dest == src is rejected.
    EXPECT_EQ(rocsparse::copy_csrgemm_info(info, info), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse::destroy_csrgemm_info(info), rocsparse_status_success);
}

TEST(internal_infostructs, csrgemm_info_destroy_null_is_success)
{
    EXPECT_EQ(rocsparse::destroy_csrgemm_info(nullptr), rocsparse_status_success);
}

// ===========================================================================
// _rocsparse_mat_info lifecycle via the public C API.
//
// The C++ member routing of _rocsparse_mat_info (get_boost, get/set_csrmv_info,
// get_*_info, clear_*_info) is hidden in the library and its definition drags in
// the whole trm_t / csritsv / bsrmv / sorted_coo2csr dependency graph, so it is
// exercised here through the exported create/destroy C API (which runs the real
// _rocsparse_mat_info constructor and destructor-routing) rather than by
// compiling in common/rocsparse_mat_info.cpp. See the report / findings note.
// ===========================================================================
TEST(internal_infostructs, mat_info_create_destroy_roundtrip)
{
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
}

TEST(internal_infostructs, mat_info_destroy_null_is_success)
{
    EXPECT_EQ(rocsparse_destroy_mat_info(nullptr), rocsparse_status_success);
}
