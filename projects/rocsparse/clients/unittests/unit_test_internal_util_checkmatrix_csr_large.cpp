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
// (wf_size * m) threads. That product used to be evaluated in J (int32_t in a
// default build), so it wrapped modulo 2^32 and the grid expression
// `(wf_size * m - 1) / block_size + 1` collapsed to ONE BLOCK. A single block of
// 256 threads covers only 256 / wf_size rows; every row above that was never
// looked at, nothing wrote *data_status, and the routine returned
// rocsparse_status_success with data_status == success on corrupt input.
//
// Fixing only the product is not enough. The corrected grid is 16,777,216
// blocks, which a dispatch cannot carry (see A NOTE ON THE GRID CLAMP below), so
// the block count has to be clamped -- and a clamped grid is 64 rows short of m
// at this size. The kernel therefore grid-strides, and the last row is reached
// only on its second sweep.
//
// SIZING (why wf_size == 4 and m == 2^30)
// ---------------------------------------
// The wavefront size is dispatched from the AVERAGE ROW LENGTH,
// avg_row_nnz = nnz / m, so the matrix has to be shaped to select the wf_size
// whose wrap point it reaches:
//
//   wf_size      selected when        wraps at m = 2^32 / wf_size
//   ------------------------------------------------------------
//         4      avg_row_nnz <=   4        1,073,741,824
//         8      avg_row_nnz <=   8          536,870,912
//        ...
//       256      avg_row_nnz >  128           16,777,216
//
// Every bucket except wf_size 4 has a nonzero LOWER bound on avg_row_nnz, so
// reaching its wrap point costs nnz >= (wf_size / 2) * (2^32 / wf_size) = 2^31
// nonzeros. That is both more memory than any current card has (>= 17 GB for
// the column and value arrays alone) and more than the rocsparse_int nnz
// argument of rocsparse_Xcheck_matrix_csr can even express. The wf_size 4
// bucket is the only reachable one, because `avg_row_nnz <= 4` is satisfied by
// avg_row_nnz == 0: an almost empty matrix with a huge row count.
//
// So: m = n = 2^30 = 1,073,741,824 rows, nnz = 1, avg_row_nnz = 0 -> wf_size 4,
// and 4 * 2^30 == 2^32 == 0 (mod 2^32), which is exactly the wrap. The buggy
// grid is 1 block and inspects rows [0, 64); the corruption is planted in row
// m - 1, which only the 16,777,215 blocks that were never launched would reach.
//
// The same m also pins the clamp and the grid-stride loop, because the corrected
// grid for 4 * 2^30 = 2^32 threads is one block past what a dispatch can carry.
//
// FOOTPRINT
// ---------
// Only the row pointer array is large: (m + 1) * sizeof(int32_t) = 4.295 GB,
// which fits comfortably on a 15 GB gfx1201. It is built on the DEVICE with
// hipMemset plus a single 4-byte write, so there is no multi-gigabyte host
// allocation and no host->device copy of the row pointers. The column, value and
// temporary buffers are a handful of bytes.
//
// There is deliberately NO device-memory guard and NO GTEST_SKIP here: guards of
// that kind have repeatedly caused regression tests in this programme to never
// execute anywhere, silently. If the allocation cannot be satisfied this test
// FAILS LOUDLY rather than disappearing.
//

// A NOTE ON THE GRID CLAMP
// ------------------------
// At this size the clamp is load-bearing, and it is not maxGridSize[0] that
// binds. A dispatch may carry at most 2^32 - 1 WORK ITEMS, because the HSA
// packet encodes the grid size in work items in a 32-bit field, so a 256-thread
// block allows only 16,777,215 blocks. The ceil-divide for m = 2^30 at wf_size 4
// asks for exactly 16,777,216 -- one block too many -- and a launch that large
// is rejected with hipErrorInvalidConfiguration.
//
// So this matrix pins the clamp as well as the 64-bit product: with the block
// count clamped to 16,777,215 one sweep of the grid covers
// 16,777,215 * 64 = 1,073,741,760 rows, which is 64 rows SHORT of m, and row
// m - 1 is reached only on the second sweep of the grid-stride loop. The planted
// corruption is invisible unless the 64-bit product, the clamp and the
// grid-stride loop are all three correct.
//
// One more trap this test has to cover explicitly: the library's
// RETURN_IF_HIPLAUNCHKERNELGGL_ERROR only inspects hipGetLastError() when the
// kernel-launch debug variable is set, so a REJECTED launch does not become a
// failed status -- the routine returns rocsparse_status_success having executed
// nothing, which is indistinguishable from the original bug. The test therefore
// asserts on the pending HIP error as well as on data_status.
//
// The second test below additionally drives check_matrix_csr_device directly
// with a deliberately undersized grid. That covers the multi-sweep path for the
// wf_size values the public 32-bit API cannot reach at all (see SIZING), on a
// 4096-row matrix costing a few kilobytes.
//

#include "unit_test_utils.hpp"

#include "check_matrix_csr_device.h"
#include "rocsparse.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>

using namespace rocsparse_ut;

namespace
{
    // Row count at which the int32_t (wf_size * m) product wraps to 0 for the
    // dispatched wf_size of 4. See the SIZING note above.
    constexpr int32_t large_m = int32_t{1} << 30; // 1,073,741,824

    // Exactly one nonzero in the whole matrix, so avg_row_nnz = nnz / m = 0 and
    // the dispatcher selects wf_size 4.
    constexpr int32_t large_nnz = 1;

    constexpr rocsparse_index_base   BASE    = rocsparse_index_base_zero;
    constexpr rocsparse_matrix_type  GENERAL = rocsparse_matrix_type_general;
    constexpr rocsparse_fill_mode    LOWER   = rocsparse_fill_mode_lower;
    constexpr rocsparse_storage_mode SORTED  = rocsparse_storage_mode_sorted;
}

class CheckMatrixCsrLargeGrid : public HandleTest
{
};

// A 2^30-row CSR matrix whose single nonzero sits in the LAST row. The row
// pointer array is valid throughout (so check_row_ptr_array, which has always
// used a correct grid, passes and the routine proceeds to the real validation
// kernel); the column index of that one nonzero is what the test flips.
//
// Assertion (b) is the load-bearing one: before AISPARSE-698 the validator
// returned rocsparse_status_success / rocsparse_data_status_success here,
// because the wrapped grid never launched a block that reached row m - 1.
// Assertion (a) runs first on the SAME matrix with a legal column index, so a
// rejection in (b) is attributable to the planted corruption and not to the
// matrix being malformed in some other way.
TEST_F(CheckMatrixCsrLargeGrid, corrupt_high_row_is_rejected)
{
    // row_ptr[i] = 0 for i in [0, m), row_ptr[m] = 1: rows [0, m-1) are empty and
    // row m-1 holds the single nonzero. Allocated and filled on the device.
    // device_vector owns the allocation, so an assertion below cannot leak the
    // 4.3 GB for the rest of the process and starve the other tests in this
    // binary. Its size_t constructor allocates on the device WITHOUT building a
    // host copy, which is what keeps the host footprint at zero.
    const size_t           row_ptr_len = static_cast<size_t>(large_m) + 1;
    device_vector<int32_t> d_row_ptr{row_ptr_len};
    ASSERT_NE(d_row_ptr.ptr, nullptr)
        << "could not allocate the " << ((row_ptr_len * sizeof(int32_t)) >> 20)
        << " MB row pointer array for a " << large_m << "-row matrix";
    UT_CHECK_HIP(hipMemset(d_row_ptr.ptr, 0, row_ptr_len * sizeof(int32_t)));

    const int32_t row_ptr_end = large_nnz;
    UT_CHECK_HIP(
        hipMemcpy(d_row_ptr.ptr + large_m, &row_ptr_end, sizeof(int32_t), hipMemcpyHostToDevice));

    device_vector<int32_t> col_ind{std::vector<int32_t>{0}};
    device_vector<float>   val{std::vector<float>{1.0f}};
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
        << "the uncorrupted 2^30-row matrix must validate; the corruption check "
           "below is only meaningful if it does";

    // (b) Plant an out-of-range column index in row m - 1. Only a grid that
    //     actually covers all m rows can see it.
    const int32_t bad_col = large_m; // == n, i.e. one past the last legal column
    UT_CHECK_HIP(hipMemcpy(col_ind.ptr, &bad_col, sizeof(int32_t), hipMemcpyHostToDevice));

    data_status = rocsparse_data_status_success;
    ASSERT_EQ(check(&data_status), rocsparse_status_success);
    ASSERT_EQ(hipGetLastError(), hipSuccess)
        << "the validation kernel launch was rejected by the runtime (a grid of "
           "more than 2^32 - 1 work items), so no row was examined at all";
    EXPECT_EQ(data_status, rocsparse_data_status_invalid_index)
        << "rocsparse_scheck_matrix_csr accepted a CSR matrix with an "
           "out-of-range column index in row "
        << (static_cast<int64_t>(large_m) - 1)
        << ": the validation grid does not cover every row (AISPARSE-698)";

    // (c) Same corruption through the generic entry point, which reaches the
    //     same check_matrix_csr_core on the CSR path.
    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         large_m,
                                         large_m,
                                         large_nnz,
                                         d_row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         BASE,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    size_t                spmat_buffer_size = 0;
    rocsparse_data_status spmat_status      = rocsparse_data_status_success;
    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat,
                                    &spmat_status,
                                    rocsparse_check_spmat_stage_buffer_size,
                                    &spmat_buffer_size,
                                    nullptr),
              rocsparse_status_success);

    device_vector<char> spmat_tmp{spmat_buffer_size ? spmat_buffer_size : size_t(1)};
    ASSERT_TRUE(spmat_tmp.ptr);

    spmat_status = rocsparse_data_status_success;
    ASSERT_EQ(rocsparse_check_spmat(handle,
                                    mat,
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

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

// ===========================================================================
// Grid-stride coverage: check_matrix_csr_device driven directly.
// ===========================================================================
//
// The clamp in LAUNCH_CHECK_MATRIX_CSR is not reachable through the 32-bit
// public API on this hardware (see the note at the top of this file), so the
// kernel's grid-stride loop would otherwise only ever be observed completing in
// a single sweep. Launching the kernel directly with a grid far smaller than the
// row count reproduces exactly the situation a clamped grid creates, on a 4096-
// row matrix that costs a few kilobytes.
//
// Without the grid-stride loop a launch of GRID blocks covers only
// GRID * (BLOCKSIZE / WF_SIZE) rows; the corruption is planted in the LAST row,
// which is reachable only on the final sweep.
namespace
{
    constexpr int32_t small_m    = 4096;
    constexpr int32_t small_grid = 2; // deliberately far below what m needs

    // Drive rocsparse::check_matrix_csr_device<256, WF_SIZE> over a 4096-row
    // matrix with one nonzero per row, using `small_grid` blocks. `bad_last_col`
    // selects whether the single nonzero of the last row carries a legal column
    // index or an out-of-range one.
    //
    // The HIP status is returned separately from the data status: every value of
    // rocsparse_data_status is a legitimate result, so none of them can double as
    // an error sentinel without turning a device failure into a wrong answer.
    template <uint32_t WF_SIZE>
    hipError_t run_undersized_grid(bool bad_last_col, rocsparse_data_status* status)
    {
        std::vector<int32_t> h_row_ptr(small_m + 1);
        std::iota(h_row_ptr.begin(), h_row_ptr.end(), 0); // one nonzero per row

        std::vector<int32_t> h_col_ind(small_m);
        std::iota(h_col_ind.begin(), h_col_ind.end(), 0); // diagonal
        if(bad_last_col)
        {
            h_col_ind[small_m - 1] = small_m; // == n, one past the last column
        }

        const std::vector<float> h_val(small_m, 1.0f);

        device_vector<int32_t>               d_row_ptr{h_row_ptr};
        device_vector<int32_t>               d_col_ind{h_col_ind};
        device_vector<float>                 d_val{h_val};
        device_vector<rocsparse_data_status> d_status{size_t{1}};
        if(!d_row_ptr.ptr || !d_col_ind.ptr || !d_val.ptr || !d_status.ptr)
        {
            return hipErrorOutOfMemory;
        }

        // rocsparse_data_status_success == 0, matching how the library primes it.
        const hipError_t memset_status = hipMemset(d_status.ptr, 0, sizeof(rocsparse_data_status));
        if(memset_status != hipSuccess)
        {
            return memset_status;
        }

        // hipGetLastError() is sticky, so drain anything an earlier test left
        // behind before the launch check below reads it.
        (void)hipGetLastError();

        hipLaunchKernelGGL(
            (rocsparse::check_matrix_csr_device<256, WF_SIZE, float, int32_t, int32_t>),
            dim3(small_grid),
            dim3(256),
            0,
            0,
            small_m,
            small_m,
            static_cast<int32_t>(small_m),
            d_val.ptr,
            d_row_ptr.ptr,
            d_col_ind.ptr,
            d_col_ind.ptr,
            BASE,
            GENERAL,
            LOWER,
            SORTED,
            d_status.ptr);

        const hipError_t launch_status = hipGetLastError();
        if(launch_status != hipSuccess)
        {
            return launch_status;
        }

        const hipError_t sync_status = hipDeviceSynchronize();
        if(sync_status != hipSuccess)
        {
            return sync_status;
        }

        *status = to_host(d_status)[0];
        return hipSuccess;
    }

    template <uint32_t WF_SIZE>
    void expect_undersized_grid_covers_last_row()
    {
        const int64_t rows_per_sweep = small_grid * (256 / static_cast<int64_t>(WF_SIZE));
        ASSERT_LT(rows_per_sweep, small_m) << "test grid must be too small for one sweep";

        rocsparse_data_status valid_status = rocsparse_data_status_inf;
        ASSERT_EQ(run_undersized_grid<WF_SIZE>(/*bad_last_col=*/false, &valid_status), hipSuccess)
            << "WF_SIZE=" << WF_SIZE << ": the kernel launch itself failed";
        EXPECT_EQ(valid_status, rocsparse_data_status_success)
            << "WF_SIZE=" << WF_SIZE << ": a valid matrix must not be rejected";

        rocsparse_data_status corrupt_status = rocsparse_data_status_success;
        ASSERT_EQ(run_undersized_grid<WF_SIZE>(/*bad_last_col=*/true, &corrupt_status), hipSuccess)
            << "WF_SIZE=" << WF_SIZE << ": the kernel launch itself failed";
        EXPECT_EQ(corrupt_status, rocsparse_data_status_invalid_index)
            << "WF_SIZE=" << WF_SIZE << ": a grid of " << small_grid << " blocks covers only "
            << rows_per_sweep << " of " << small_m << " rows per sweep, so row " << (small_m - 1)
            << " is reached only by the grid-stride loop (AISPARSE-698)";
    }
}

// One case per end of the dispatched WF_SIZE range plus a middle value. WF_SIZE
// is the lane-group width, not the hardware wavefront, so all three are valid on
// a wave32 part; the kernel has no cross-lane operation.
TEST(CheckMatrixCsrGridStride, undersized_grid_still_covers_every_row_wf4)
{
    expect_undersized_grid_covers_last_row<4>();
}

TEST(CheckMatrixCsrGridStride, undersized_grid_still_covers_every_row_wf32)
{
    expect_undersized_grid_covers_last_row<32>();
}

TEST(CheckMatrixCsrGridStride, undersized_grid_still_covers_every_row_wf256)
{
    expect_undersized_grid_covers_last_row<256>();
}
