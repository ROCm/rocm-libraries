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
// Host-path unit tests for the SpSort routines (ROCM-33112):
// rocsparse_spsort_descr_create/destroy/set_input (library/src/conversion/
// rocsparse_spsort_descr.cpp), rocsparse_spsort_buffer_size (rocsparse_spsort_
// buffer_size.cpp) and rocsparse_spsort (rocsparse_spsort.cpp). These drive the
// public C API on tiny COO/CSR/CSC matrices so they exercise the HOST
// dispatch, argument validation and stage-ordering FSM.
//
// There were previously ZERO references to spsort under clients/unittests.
// The integration YAML (clients/tests/test_spsort_*.yaml) never checks the
// analysis-stage buffer size or the stage ordering FSM, so a regression in
// either is otherwise silent; that is what this file is for.
//
// Gated by ROCSPARSE_WITH_SPSORT (CMake option BUILD_WITH_SPSORT, default ON).
//
#include "unit_test_utils.hpp"

#if defined(ROCSPARSE_WITH_SPSORT)

using namespace rocsparse_ut;

namespace
{
    // Creates a SpSort descriptor with the algorithm (and, for COO, the
    // direction) already set, so callers only need to decide the stage. Both
    // rocsparse_spsort_buffer_size and rocsparse_spsort reject a descriptor
    // whose algorithm has not been set (and, for COO sources, whose direction
    // has not been set) with rocsparse_status_invalid_value.
    rocsparse_spsort_descr make_spsort_descr(rocsparse_handle           handle,
                                             rocsparse_spsort_alg       alg,
                                             const rocsparse_direction* dir = nullptr)
    {
        rocsparse_spsort_descr descr = nullptr;
        if(rocsparse_spsort_descr_create(handle, &descr, nullptr) != rocsparse_status_success)
        {
            return nullptr;
        }
        if(rocsparse_spsort_set_input(
               handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr)
           != rocsparse_status_success)
        {
            (void)rocsparse_spsort_descr_destroy(handle, descr, nullptr);
            return nullptr;
        }
        if(dir != nullptr
           && rocsparse_spsort_set_input(
                  handle, descr, rocsparse_spsort_input_direction, dir, sizeof(*dir), nullptr)
                  != rocsparse_status_success)
        {
            (void)rocsparse_spsort_descr_destroy(handle, descr, nullptr);
            return nullptr;
        }
        return descr;
    }

    // Runs one SpSort stage end to end: query the buffer size, allocate it (if
    // non-zero) and call rocsparse_spsort. buffer_size_in_bytes == 0 requires a
    // null buffer (CHECKARG on arg 5/6 of rocsparse_spsort), so a zero-sized
    // buffer is never allocated. Optionally reports the queried buffer size.
    rocsparse_status run_spsort_stage(rocsparse_handle            handle,
                                      rocsparse_spsort_descr      descr,
                                      rocsparse_const_spmat_descr source,
                                      rocsparse_spmat_descr       target,
                                      rocsparse_spsort_stage      stage,
                                      size_t*                     out_buffer_size = nullptr)
    {
        size_t           buffer_size = 0;
        rocsparse_status st          = rocsparse_spsort_buffer_size(
            handle, descr, source, target, stage, &buffer_size, nullptr);
        if(st != rocsparse_status_success)
        {
            return st;
        }
        if(out_buffer_size != nullptr)
        {
            *out_buffer_size = buffer_size;
        }

        if(buffer_size == 0)
        {
            return rocsparse_spsort(handle, descr, source, target, stage, 0, nullptr, nullptr);
        }

        device_vector<char> buffer{buffer_size};
        if(buffer.ptr == nullptr)
        {
            return rocsparse_status_memory_error;
        }
        return rocsparse_spsort(
            handle, descr, source, target, stage, buffer_size, buffer.ptr, nullptr);
    }

    // Runs the full analysis-then-compute sequence required before a SpSort
    // descriptor may be used for the compute stage.
    rocsparse_status run_spsort(rocsparse_handle            handle,
                                rocsparse_spsort_descr      descr,
                                rocsparse_const_spmat_descr source,
                                rocsparse_spmat_descr       target,
                                size_t*                     compute_buffer_size_out = nullptr)
    {
        rocsparse_status st
            = run_spsort_stage(handle, descr, source, target, rocsparse_spsort_stage_analysis);
        if(st != rocsparse_status_success)
        {
            return st;
        }
        return run_spsort_stage(
            handle, descr, source, target, rocsparse_spsort_stage_compute, compute_buffer_size_out);
    }
} // namespace

// Lower-case suite name so the fully-qualified test names contain "spsort"
// verbatim (gtest filters are case-sensitive), matching the CI/CLI convention
// of filtering this file's tests with --gtest_filter='*spsort*'.
class spsort : public HandleTest
{
};

// ===========================================================================
// 1. Analysis-stage buffer_size is 0 for CSR, CSC and COO (library/src/
//    conversion/rocsparse_spsort_buffer_size.cpp lines 47-59).
// ===========================================================================
TEST_F(spsort, analysis_buffer_size_is_zero_csr)
{
    const rocsparse_int m = 2, n = 2, nnz = 2;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    size_t buffer_size = 123;
    EXPECT_EQ(rocsparse_spsort_buffer_size(
                  handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(buffer_size, 0u);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST_F(spsort, analysis_buffer_size_is_zero_csc)
{
    const rocsparse_int m = 2, n = 2, nnz = 2;

    device_vector<rocsparse_int> col_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(col_ptr.ptr && row_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csc_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         col_ptr.ptr,
                                         row_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    size_t buffer_size = 123;
    EXPECT_EQ(rocsparse_spsort_buffer_size(
                  handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(buffer_size, 0u);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST_F(spsort, analysis_buffer_size_is_zero_coo)
{
    const rocsparse_int m = 2, n = 2, nnz = 2;

    // The direction must be set even though the analysis stage itself is a
    // quick return that ignores it: spsort_check_arguments rejects a COO
    // source with an unset direction before the stage is ever dispatched.
    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(row_ind.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ind.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const rocsparse_direction dir   = rocsparse_direction_row;
    rocsparse_spsort_descr    descr = make_spsort_descr(handle, rocsparse_spsort_alg_default, &dir);
    ASSERT_NE(descr, nullptr);

    size_t buffer_size = 123;
    EXPECT_EQ(rocsparse_spsort_buffer_size(
                  handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
              rocsparse_status_success);
    EXPECT_EQ(buffer_size, 0u);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

// ===========================================================================
// 2. Compute-stage buffer_size > 0 when there is something to sort, and the
//    sorted result matches a host sort. CSR is sorted in place; COO is
//    sorted out of place (into a second matrix), covering both styles.
// ===========================================================================

// 2x3 CSR, two rows of two entries each with the column indices scrambled
// within each row:
//   row 0: col = [2, 0], val = [10, 11]  ->  sorted col = [0, 2], val = [11, 10]
//   row 1: col = [2, 1], val = [12, 13]  ->  sorted col = [1, 2], val = [13, 12]
TEST_F(spsort, compute_sorts_csr_in_place)
{
    const rocsparse_int m = 2, n = 3, nnz = 4;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 2, 4}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{2, 0, 2, 1}};
    device_vector<float>         val{std::vector<float>{10.0f, 11.0f, 12.0f, 13.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    size_t compute_buffer_size = 0;
    ASSERT_EQ(run_spsort(handle, descr, mat, mat, &compute_buffer_size), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_GT(compute_buffer_size, 0u);

    const std::vector<rocsparse_int> got_col = to_host(col_ind);
    const std::vector<float>         got_val = to_host(val);
    EXPECT_EQ(got_col, (std::vector<rocsparse_int>{0, 2, 1, 2}));
    EXPECT_EQ(got_val, (std::vector<float>{11.0f, 10.0f, 13.0f, 12.0f}));

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

// 2x2 COO, direction_row, three unsorted entries:
//   (row, col, val) = (1,1,200), (0,0,201), (1,0,202)
// Sorted by row then column:
//   (0,0,201), (1,0,202), (1,1,200)
TEST_F(spsort, compute_sorts_coo_out_of_place)
{
    const rocsparse_int m = 2, n = 2, nnz = 3;

    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{1, 0, 1}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{1, 0, 0}};
    device_vector<float>         val{std::vector<float>{200.0f, 201.0f, 202.0f}};
    ASSERT_TRUE(row_ind.ptr && col_ind.ptr && val.ptr);

    device_vector<rocsparse_int> row_ind_out{(size_t)nnz};
    device_vector<rocsparse_int> col_ind_out{(size_t)nnz};
    device_vector<float>         val_out{(size_t)nnz};
    ASSERT_TRUE(row_ind_out.ptr && col_ind_out.ptr && val_out.ptr);

    rocsparse_spmat_descr source = nullptr, target = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&source,
                                         m,
                                         n,
                                         nnz,
                                         row_ind.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_coo_descr(&target,
                                         m,
                                         n,
                                         nnz,
                                         row_ind_out.ptr,
                                         col_ind_out.ptr,
                                         val_out.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const rocsparse_direction dir   = rocsparse_direction_row;
    rocsparse_spsort_descr    descr = make_spsort_descr(handle, rocsparse_spsort_alg_default, &dir);
    ASSERT_NE(descr, nullptr);

    size_t compute_buffer_size = 0;
    ASSERT_EQ(run_spsort(handle, descr, source, target, &compute_buffer_size),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_GT(compute_buffer_size, 0u);

    const std::vector<rocsparse_int> got_row = to_host(row_ind_out);
    const std::vector<rocsparse_int> got_col = to_host(col_ind_out);
    const std::vector<float>         got_val = to_host(val_out);
    EXPECT_EQ(got_row, (std::vector<rocsparse_int>{0, 1, 1}));
    EXPECT_EQ(got_col, (std::vector<rocsparse_int>{0, 0, 1}));
    EXPECT_EQ(got_val, (std::vector<float>{201.0f, 202.0f, 200.0f}));

    // The unsorted source must be left untouched by an out-of-place sort.
    const std::vector<rocsparse_int> src_row = to_host(row_ind);
    const std::vector<rocsparse_int> src_col = to_host(col_ind);
    EXPECT_EQ(src_row, (std::vector<rocsparse_int>{1, 0, 1}));
    EXPECT_EQ(src_col, (std::vector<rocsparse_int>{1, 0, 0}));

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(source), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(target), rocsparse_status_success);
}

// ===========================================================================
// 3. Status checks: null descr / source / buffer_size pointer, COO with an
//    unset direction, and calling compute before analysis.
// ===========================================================================
TEST_F(spsort, null_pointer_arguments)
{
    const rocsparse_int m = 2, n = 2, nnz = 2;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    size_t buffer_size = 0;

    // Null descr pointer.
    EXPECT_EQ(
        rocsparse_spsort_buffer_size(
            handle, nullptr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);

    // Null source pointer.
    EXPECT_EQ(rocsparse_spsort_buffer_size(handle,
                                           descr,
                                           (rocsparse_const_spmat_descr) nullptr,
                                           mat,
                                           rocsparse_spsort_stage_analysis,
                                           &buffer_size,
                                           nullptr),
              rocsparse_status_invalid_pointer);

    // Null buffer_size_in_bytes pointer.
    EXPECT_EQ(rocsparse_spsort_buffer_size(
                  handle, descr, mat, mat, rocsparse_spsort_stage_analysis, nullptr, nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST_F(spsort, coo_direction_unset_is_invalid_value)
{
    const rocsparse_int m = 2, n = 2, nnz = 2;

    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(row_ind.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ind.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // The algorithm is set, but the direction is left unset (nullptr).
    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_spsort_buffer_size(
                  handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
              rocsparse_status_invalid_value);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

// Calling the compute stage before the analysis stage has ever been executed.
// descr->get_stage() still holds its construction-time sentinel ((rocsparse_
// spsort_stage)-1), so rocsparse_spsort's stage FSM (rocsparse_spsort.cpp,
// around lines 297-327) takes the "analysis must be executed before compute"
// branch and returns rocsparse_status_invalid_value -- the status the code
// actually returns, not rocsparse_status_invalid_handle or any other guess.
TEST_F(spsort, compute_before_analysis_is_invalid_value)
{
    const rocsparse_int m = 2, n = 2, nnz = 2;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    // buffer_size_in_bytes == 0 and buffer == nullptr satisfy the earlier
    // CHECKARGs, so the stage FSM check is reached directly.
    EXPECT_EQ(rocsparse_spsort(
                  handle, descr, mat, mat, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
              rocsparse_status_invalid_value);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

// ===========================================================================
// 4. nnz = 0 and nnz = 1 succeed.
// ===========================================================================
TEST_F(spsort, nnz_zero_succeeds)
{
    const rocsparse_int m = 2, n = 2;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 0, 0}};
    ASSERT_TRUE(row_ptr.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         m,
                                         n,
                                         0,
                                         row_ptr.ptr,
                                         nullptr,
                                         nullptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    EXPECT_EQ(run_spsort(handle, descr, mat, mat), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST_F(spsort, nnz_one_succeeds)
{
    const rocsparse_int m = 1, n = 1, nnz = 1;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0}};
    device_vector<float>         val{std::vector<float>{7.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         m,
                                         n,
                                         nnz,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_spsort_descr descr = make_spsort_descr(handle, rocsparse_spsort_alg_default);
    ASSERT_NE(descr, nullptr);

    EXPECT_EQ(run_spsort(handle, descr, mat, mat), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::vector<rocsparse_int> got_col = to_host(col_ind);
    const std::vector<float>         got_val = to_host(val);
    EXPECT_EQ(got_col, (std::vector<rocsparse_int>{0}));
    EXPECT_EQ(got_val, (std::vector<float>{7.0f}));

    EXPECT_EQ(rocsparse_spsort_descr_destroy(handle, descr, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

#endif // ROCSPARSE_WITH_SPSORT
