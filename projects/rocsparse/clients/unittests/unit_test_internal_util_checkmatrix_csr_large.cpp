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
// AISPARSE-698 regression: the CSR matrix validator used to overflow its own
// launch grid on large matrices and then report the matrix VALID.
//
// library/src/util/rocsparse_check_matrix_csr.cpp launches
// check_matrix_csr_device with one WF_SIZE-wide lane group per row, i.e.
// (wf_size * m) threads. wf_size was an int literal, so with m an int32_t the
// product `wf_size * m` was a SIGNED 32-bit product. It is undefined behaviour,
// and in practice wraps, once m > 2^31 / wf_size. The grid expression
// `(wf_size * m - 1) / block_size + 1` then either went negative, which dim3
// turns into about 4.29e9 blocks that the runtime rejects, or collapsed to a
// handful of blocks. In both cases the rows beyond the first few were never
// looked at, nothing wrote *data_status, and the routine returned
// rocsparse_status_success with data_status == success on corrupt input.
//
// SIZING (why wf_size == 4 and m == 2^29 + 1)
// -------------------------------------------
// The wavefront size is dispatched from the AVERAGE ROW LENGTH,
// avg_row_nnz = nnz / m, so the matrix has to be shaped to select the wf_size
// whose overflow point it reaches:
//
//   wf_size   selected when    first affected m      min nnz to get there
//   ----------------------------------------------------------------------
//         4   avg_row_nnz <=   4     536,870,913                      1
//         8   avg_row_nnz <=   8     268,435,457          1,342,177,285
//        16   avg_row_nnz <=  16     134,217,729          1,207,959,561
//        32   avg_row_nnz <=  32      67,108,865          1,140,850,705
//        64   avg_row_nnz <=  64      33,554,433          1,107,296,289
//       128   avg_row_nnz <= 128      16,777,217          1,090,519,105
//       256   avg_row_nnz >  128       8,388,609          1,082,130,561
//
// "First affected m" is 2^31 / wf_size + 1. Every wf_size is therefore reachable
// through the 32-bit API, but every bucket except wf_size 4 has a nonzero lower
// bound on avg_row_nnz, so reaching it costs about 1.1e9 nonzeros or more: over
// 8 GB for the column and value arrays alone, too much for a default unit test.
// The wf_size 4 bucket needs none, because `avg_row_nnz <= 4` is satisfied by
// avg_row_nnz == 0: an almost empty matrix with a huge row count.
//
// So: m = n = 2^29 + 1 = 536,870,913 rows, nnz = 1, avg_row_nnz = 0 -> wf_size 4,
// and 4 * m = 2^31 + 4 overflows int32_t. The buggy grid is negative, the launch
// is rejected, and no row is examined; the corruption is planted in row m - 1.
//
// FOOTPRINT
// ---------
// Only the row pointer array is large: (m + 1) * sizeof(rocsparse_int) = 2.15 GB
// (4.3 GB with rocsparse_ILP64). It is built on the DEVICE with hipMemset plus a
// single rocsparse_int write, so there is no multi-gigabyte host allocation and
// no host->device copy of the row pointers. The column, value and temporary
// buffers are a handful of bytes. If the allocation cannot be satisfied the test
// fails with a message rather than skipping.
//
// WHAT THIS DOES NOT COVER, AND WHERE THAT IS COVERED
// ---------------------------------------------------
// At m = 2^29 + 1 the corrected grid is 8,388,609 blocks, below the 16,777,215
// that a dispatch can carry, so the grid clamp and the grid-stride loop do not
// engage. They engage at m >= 2^30 for wf_size 4 (16,777,216 blocks, one past the
// 2^32 - 1 work-item limit of the dispatch packet), and the grid-stride loops
// are exercised on small matrices by CheckMatrixCsrForcedClamp below, which
// shrinks maxGridSize[0] so that the same clamp-and-stride code runs for the
// wf_size 4, 32 and 256 kernels.
//
// The library's RETURN_IF_HIPLAUNCHKERNELGGL_ERROR only inspects hipGetLastError()
// when the kernel-launch debug variable is set, so a REJECTED launch does not
// become a failed status -- the routine returns rocsparse_status_success having
// executed nothing, which is indistinguishable from the original bug. The large
// test therefore asserts on the pending HIP error as well as on data_status.
//

#include "unit_test_utils.hpp"

#include "rocsparse.h"
#include "unit_test_grid_clamp.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Smallest row count at which the int32_t (wf_size * m) product overflows for
    // the dispatched wf_size of 4: 4 * (2^29 + 1) = 2^31 + 4. See the SIZING note
    // above.
    constexpr rocsparse_int large_m = (rocsparse_int{1} << 29) + 1; // 536,870,913

    // Exactly one nonzero in the whole matrix, so avg_row_nnz = nnz / m = 0 and
    // the dispatcher selects wf_size 4.
    constexpr rocsparse_int large_nnz = 1;

    constexpr rocsparse_index_base   BASE    = rocsparse_index_base_zero;
    constexpr rocsparse_matrix_type  GENERAL = rocsparse_matrix_type_general;
    constexpr rocsparse_fill_mode    LOWER   = rocsparse_fill_mode_lower;
    constexpr rocsparse_storage_mode SORTED  = rocsparse_storage_mode_sorted;
    constexpr rocsparse_indextype    IT
        = sizeof(rocsparse_int) == 4 ? rocsparse_indextype_i32 : rocsparse_indextype_i64;

    // Destroys the descriptor on every exit path, including an early return from
    // a failed ASSERT_*.
    struct SpmatGuard
    {
        rocsparse_spmat_descr mat = nullptr;

        SpmatGuard() = default;

        ~SpmatGuard()
        {
            if(mat != nullptr)
            {
                rocsparse_destroy_spmat_descr(mat);
            }
        }

        SpmatGuard(const SpmatGuard&) = delete;

        SpmatGuard& operator=(const SpmatGuard&) = delete;
    };
}

class CheckMatrixCsrLargeGrid : public HandleTest
{
};

// A (2^29 + 1)-row CSR matrix whose single nonzero sits in the LAST row. The row
// pointer array is valid throughout (so check_row_ptr_array passes and the
// routine proceeds to the real validation kernel); the column index of that one
// nonzero is what the test flips.
//
// Assertion (b) is the load-bearing one: before AISPARSE-698 the validator
// returned rocsparse_status_success / rocsparse_data_status_success here,
// because the overflowed grid never launched a block that reached row m - 1.
// Assertion (a) runs first on the SAME matrix with a legal column index, so a
// rejection in (b) is attributable to the planted corruption and not to the
// matrix being malformed in some other way.
TEST_F(CheckMatrixCsrLargeGrid, corrupt_high_row_is_rejected)
{
    // row_ptr[i] = 0 for i in [0, m), row_ptr[m] = 1: rows [0, m-1) are empty and
    // row m-1 holds the single nonzero. Allocated and filled on the device.
    // device_vector owns the allocation, so an assertion below cannot leak the
    // 2.15 GB for the rest of the process and starve the other tests in this
    // binary. Its size_t constructor allocates on the device WITHOUT building a
    // host copy, which is what keeps the host footprint at zero.
    const size_t row_ptr_len = static_cast<size_t>(large_m) + 1;

    // Skip, rather than fail with an out-of-memory error, on a device that cannot
    // hold the row pointer array plus the validator's workspace. The 2x margin
    // matches the other large-grid unit tests.
    {
        const size_t need        = row_ptr_len * sizeof(rocsparse_int);
        size_t       free_bytes  = 0;
        size_t       total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess || free_bytes < 2 * need)
        {
            GTEST_SKIP() << "Insufficient device memory: need about " << need << " bytes";
        }
    }

    device_vector<rocsparse_int> d_row_ptr{row_ptr_len};
    ASSERT_NE(d_row_ptr.ptr, nullptr)
        << "could not allocate the " << ((row_ptr_len * sizeof(rocsparse_int)) >> 20)
        << " MB row pointer array for a " << large_m << "-row matrix";
    UT_CHECK_HIP(hipMemset(d_row_ptr.ptr, 0, row_ptr_len * sizeof(rocsparse_int)));

    const rocsparse_int row_ptr_end = large_nnz;
    UT_CHECK_HIP(hipMemcpy(
        d_row_ptr.ptr + large_m, &row_ptr_end, sizeof(rocsparse_int), hipMemcpyHostToDevice));

    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0}};
    device_vector<float>         val{std::vector<float>{1.0f}};
    ASSERT_TRUE(col_ind.ptr && val.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scheck_matrix_csr_buffer_size(handle,
                                                      large_m,
                                                      large_m,
                                                      large_nnz,
                                                      val,
                                                      d_row_ptr.ptr,
                                                      col_ind,
                                                      BASE,
                                                      GENERAL,
                                                      LOWER,
                                                      SORTED,
                                                      &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    const auto check = [&](rocsparse_data_status* out) {
        return rocsparse_scheck_matrix_csr(handle,
                                           large_m,
                                           large_m,
                                           large_nnz,
                                           val,
                                           d_row_ptr.ptr,
                                           col_ind,
                                           BASE,
                                           GENERAL,
                                           LOWER,
                                           SORTED,
                                           out,
                                           tmp.ptr);
    };

    // Discard any error left pending by an earlier test in this binary, so the
    // launch assertions below cannot misattribute somebody else's failure.
    (void)hipGetLastError();

    // (a) Control: column index 0 is in range, so this matrix is valid.
    rocsparse_data_status data_status = rocsparse_data_status_inf;
    ASSERT_EQ(check(&data_status), rocsparse_status_success);
    ASSERT_EQ(hipGetLastError(), hipSuccess)
        << "a validation kernel launch was rejected by the runtime, so the "
           "control matrix was never examined";
    ASSERT_EQ(data_status, rocsparse_data_status_success)
        << "the uncorrupted " << large_m
        << "-row matrix must validate; the corruption check below is only meaningful if it does";

    // (b) Plant an out-of-range column index in row m - 1. Only a grid that
    //     actually covers all m rows can see it.
    const rocsparse_int bad_col = large_m; // == n, i.e. one past the last legal column
    UT_CHECK_HIP(hipMemcpy(col_ind.ptr, &bad_col, sizeof(rocsparse_int), hipMemcpyHostToDevice));

    data_status = rocsparse_data_status_success;
    ASSERT_EQ(check(&data_status), rocsparse_status_success);
    ASSERT_EQ(hipGetLastError(), hipSuccess)
        << "the validation kernel launch was rejected by the runtime, so no row was examined at "
           "all";
    EXPECT_EQ(data_status, rocsparse_data_status_invalid_index)
        << "rocsparse_scheck_matrix_csr accepted a CSR matrix with an "
           "out-of-range column index in row "
        << (static_cast<int64_t>(large_m) - 1)
        << ": the validation grid does not cover every row (AISPARSE-698)";

    // (c) Same corruption through the generic entry point, which reaches the
    //     same check_matrix_csr_core on the CSR path.
    SpmatGuard mat;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat.mat,
                                         large_m,
                                         large_m,
                                         large_nnz,
                                         d_row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         IT,
                                         IT,
                                         BASE,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    size_t                spmat_buffer_size = 0;
    rocsparse_data_status spmat_status      = rocsparse_data_status_success;
    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat.mat,
                                    &spmat_status,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &spmat_buffer_size,
                                    nullptr),
              rocsparse_status_success);

    device_vector<char> spmat_tmp{spmat_buffer_size ? spmat_buffer_size : size_t(1)};
    ASSERT_TRUE(spmat_tmp.ptr);

    spmat_status = rocsparse_data_status_success;
    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat.mat,
                                    &spmat_status,
                                    rocsparse_check_spmat_stage_compute,
                                    &spmat_buffer_size,
                                    spmat_tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipGetLastError(), hipSuccess)
        << "the rocsparse_check_spmat validation kernel launch was rejected by "
           "the runtime, so no row was examined at all";
    EXPECT_EQ(spmat_status, rocsparse_data_status_invalid_index)
        << "rocsparse_check_spmat accepted the same corrupt matrix on the CSR path";
}

// ===========================================================================
// Forced clamp: the whole validator with maxGridSize[0] shrunk.
// ===========================================================================
//
// check_matrix_csr_device, check_row_ptr_array and, for unsorted input,
// shift_offsets_kernel clamp their grids with get_grid_size_x and grid-stride
// over whatever the clamp drops. With maxGridSize[0] at 1, 3 or 7 a 4096-row
// matrix needs several sweeps of every kernel: check_row_ptr_array covers at
// most 7 * 256 rows per sweep, shift_offsets_kernel 7 * 512 offsets and
// check_matrix_csr_device 7 * (256 / wf_size) rows.
//
// Row i holds row_len columns {i, ..., i + row_len - 1} (n = m + row_len - 1),
// ascending for sorted storage and descending for unsorted storage. The row
// length sets avg_row_nnz and with it the wf_size the dispatcher picks:
//
//   row_len     2 -> wf_size   4
//   row_len    20 -> wf_size  32
//   row_len   130 -> wf_size 256
namespace
{
    constexpr rocsparse_int clamp_m = 4096;

    // Row whose offsets are corrupted. It sits in logical block 15 of the
    // 256-thread check_row_ptr_array grid, so only a later sweep reaches it.
    constexpr rocsparse_int bad_row = clamp_m - 16;

    enum class clamp_case
    {
        valid,
        bad_row_ptr, // row_ptr[bad_row + 1] < row_ptr[bad_row]
        // Row m - 2 gives one entry to the last row, which holds
        // {m-1, m+row_len-2, ..., m, m-1}. The duplicate is adjacent only once the
        // columns are sorted, so an unsorted matrix is flagged only if
        // shift_offsets_kernel wrote the last row's offsets.
        split_duplicate_last_row
    };

    rocsparse_data_status run_clamped(rocsparse_handle       handle,
                                      rocsparse_storage_mode storage,
                                      clamp_case             c,
                                      rocsparse_int          row_len)
    {
        const rocsparse_int clamp_n   = clamp_m + row_len - 1;
        const rocsparse_int clamp_nnz = row_len * clamp_m;

        std::vector<rocsparse_int> h_row_ptr(clamp_m + 1);
        std::vector<rocsparse_int> h_col_ind;
        h_col_ind.reserve(clamp_nnz);
        const bool ascending = (storage == rocsparse_storage_mode_sorted);
        for(rocsparse_int i = 0; i < clamp_m; ++i)
        {
            std::vector<rocsparse_int> cols;
            for(rocsparse_int k = 0; k < row_len; ++k)
            {
                cols.push_back(ascending ? i + k : i + row_len - 1 - k);
            }
            if(c == clamp_case::split_duplicate_last_row)
            {
                if(i == clamp_m - 2)
                {
                    cols.pop_back();
                }
                if(i == clamp_m - 1)
                {
                    cols.insert(cols.begin(), clamp_m - 1);
                    cols.back() = clamp_m - 1;
                }
            }
            h_row_ptr[i + 1] = h_row_ptr[i] + static_cast<rocsparse_int>(cols.size());
            h_col_ind.insert(h_col_ind.end(), cols.begin(), cols.end());
        }
        EXPECT_EQ(h_row_ptr[clamp_m], clamp_nnz);

        if(c == clamp_case::bad_row_ptr)
        {
            h_row_ptr[bad_row + 1] = h_row_ptr[bad_row] - 1;
        }
        const std::vector<float> h_val(clamp_nnz, 1.0f);

        device_vector<rocsparse_int> d_row_ptr{h_row_ptr};
        device_vector<rocsparse_int> d_col_ind{h_col_ind};
        device_vector<float>         d_val{h_val};
        EXPECT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr);

        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_scheck_matrix_csr_buffer_size(handle,
                                                          clamp_m,
                                                          clamp_n,
                                                          clamp_nnz,
                                                          d_val.ptr,
                                                          d_row_ptr.ptr,
                                                          d_col_ind.ptr,
                                                          BASE,
                                                          GENERAL,
                                                          LOWER,
                                                          storage,
                                                          &buffer_size),
                  rocsparse_status_success);

        // Zeroed so that offsets shift_offsets_kernel fails to write are
        // deterministic rather than whatever the allocation held.
        device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
        EXPECT_TRUE(tmp.ptr);
        EXPECT_EQ(hipMemset(tmp.ptr, 0, buffer_size), hipSuccess);

        (void)hipGetLastError();

        rocsparse_data_status status = rocsparse_data_status_inf;
        EXPECT_EQ(rocsparse_scheck_matrix_csr(handle,
                                              clamp_m,
                                              clamp_n,
                                              clamp_nnz,
                                              d_val.ptr,
                                              d_row_ptr.ptr,
                                              d_col_ind.ptr,
                                              BASE,
                                              GENERAL,
                                              LOWER,
                                              storage,
                                              &status,
                                              tmp.ptr),
                  rocsparse_status_success);
        EXPECT_EQ(hipGetLastError(), hipSuccess);
        return status;
    }
}

// Parameters: (maxGridSize[0], row_len).
class CheckMatrixCsrForcedClamp : public HandleTest,
                                  public ::testing::WithParamInterface<std::tuple<int, int>>
{
protected:
    int max_grid() const
    {
        return std::get<0>(GetParam());
    }

    int row_len() const
    {
        return std::get<1>(GetParam());
    }
};

TEST_P(CheckMatrixCsrForcedClamp, valid_matrix_passes)
{
    const ScopedMaxGridSizeX clamp(handle, max_grid());
    EXPECT_EQ(run_clamped(handle, rocsparse_storage_mode_sorted, clamp_case::valid, row_len()),
              rocsparse_data_status_success);
    EXPECT_EQ(run_clamped(handle, rocsparse_storage_mode_unsorted, clamp_case::valid, row_len()),
              rocsparse_data_status_success);
}

// check_row_ptr_array returns before the sort and the main kernel run. If it
// missed the bad row, the main kernel would see the overlapping row after it and
// could report a duplicate instead.
TEST_P(CheckMatrixCsrForcedClamp, bad_row_ptr_in_later_block_is_detected)
{
    const ScopedMaxGridSizeX clamp(handle, max_grid());
    EXPECT_EQ(
        run_clamped(handle, rocsparse_storage_mode_sorted, clamp_case::bad_row_ptr, row_len()),
        rocsparse_data_status_invalid_offset_ptr);
    EXPECT_EQ(
        run_clamped(handle, rocsparse_storage_mode_unsorted, clamp_case::bad_row_ptr, row_len()),
        rocsparse_data_status_invalid_offset_ptr);
}

TEST_P(CheckMatrixCsrForcedClamp, unsorted_duplicate_in_last_row_is_detected)
{
    const ScopedMaxGridSizeX clamp(handle, max_grid());
    EXPECT_EQ(run_clamped(handle,
                          rocsparse_storage_mode_unsorted,
                          clamp_case::split_duplicate_last_row,
                          row_len()),
              rocsparse_data_status_duplicate_entry)
        << "shift_offsets_kernel must cover all " << (clamp_m + 1) << " offsets";
}

INSTANTIATE_TEST_SUITE_P(CheckMatrixCsrForcedClamp,
                         CheckMatrixCsrForcedClamp,
                         ::testing::Combine(::testing::Values(1, 3, 7),
                                            ::testing::Values(2, 20, 130)),
                         [](const ::testing::TestParamInfo<std::tuple<int, int>>& info) {
                             return "grid" + std::to_string(std::get<0>(info.param)) + "_len"
                                    + std::to_string(std::get<1>(info.param));
                         });
