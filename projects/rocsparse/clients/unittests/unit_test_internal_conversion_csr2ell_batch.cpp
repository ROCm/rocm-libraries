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
// Regression tests for rocsparse_Xcsr2ell_strided_batched (AISPARSE-683).
//
// FOCUS: the launch put batch_count straight on grid.y, which the hardware caps
// at 65535 (hipDeviceProp_t::maxGridSize[1]), and csr2ell_strided_batched_kernel
// read the batch index as a bare blockIdx.y with no stride. Large batch counts
// therefore died at launch with hipErrorInvalidConfiguration. The launch now
// clamps grid.y through rocsparse::get_batch_grid_size and the kernel
// grid-strides over batch_count.
//
// Both batch counts below exceed the clamp, so the tail batches are reachable
// ONLY through the new grid-stride loop. Asserting the call merely succeeds is
// not enough - that would only prove the grid is legal - so every batch's ELL
// output is read back and compared against a per-batch distinct value. A
// regression in either the clamp or the stride leaves the tail holding the
// pre-fill sentinel.
//
// The matrix is a 2x2 diagonal (m = 2, nnz = 2, ell_width = 1) shared by all
// batches, so 70000 batches need ~1.1 MB of device memory in total. There is
// deliberately NO device-memory guard on these tests: they must run everywhere,
// including the 15 GB gfx1201.
//
// TARGET: rocsparse_csr2ell_strided_batched.cpp is compiled into
// rocsparse-unit-test-device (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES) because
// librocsparse is built with hidden symbol visibility and the
// rocsparse_Xcsr2ell_strided_batched entry points are not declared in any
// public header, so they cannot be reached by linking roc::rocsparse.
//

#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace rocsparse_ut;

// rocsparse_Xcsr2ell_strided_batched is defined (extern "C") in
// library/src/conversion/rocsparse_csr2ell_strided_batched.cpp but declared in no
// header, public or internal. Declare the one overload under test here; the
// defining translation unit is compiled into this binary.
extern "C" rocsparse_status rocsparse_scsr2ell_strided_batched(rocsparse_handle handle,
                                                               rocsparse_int    batch_count,
                                                               rocsparse_int    m,
                                                               const rocsparse_mat_descr csr_descr,
                                                               const float*              csr_val,
                                                               rocsparse_int        csr_val_stride,
                                                               const rocsparse_int* csr_row_ptr,
                                                               const rocsparse_int* csr_col_ind,
                                                               const rocsparse_mat_descr ell_descr,
                                                               rocsparse_int             ell_width,
                                                               float*                    ell_val,
                                                               rocsparse_int  ell_val_stride,
                                                               rocsparse_int* ell_col_ind);

namespace
{
    // The grid.y hardware cap the launch clamps against.
    constexpr int64_t grid_y_cap = 65535;

    // Tiny 2x2 diagonal matrix shared by every batch: one entry per row, so
    // ell_width == 1 and the ELL structure needs no padding.
    constexpr rocsparse_int mat_m         = 2;
    constexpr rocsparse_int mat_nnz       = 2;
    constexpr rocsparse_int mat_ell_width = 1;

    const std::vector<rocsparse_int> csr_row_ptr{0, 1, 2};
    const std::vector<rocsparse_int> csr_col_ind{0, 1};

    // Per-batch, per-row value. Distinct for every (batch, row) pair and exactly
    // representable in float, so a batch written with the wrong stride (or not
    // written at all) is caught rather than aliased onto another batch's value.
    float batch_value(int64_t batch, int64_t row)
    {
        return static_cast<float>(row == 0 ? (batch + 1) : -(batch + 1));
    }

    // RAII for a mat descr.
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };
}

class Csr2EllStridedBatched : public HandleTest
{
protected:
    // Convert `batch_count` instances of the shared diagonal matrix and verify
    // the ELL values of EVERY batch, including those past the grid.y clamp.
    void run(int64_t batch_count)
    {
        // ELL_IND(i, el, m, width) == el * m + i, and ell_width == 1, so the ELL
        // value of row i of batch b lands at b * ell_val_stride + i.
        const rocsparse_int csr_val_stride = mat_nnz;
        const rocsparse_int ell_val_stride = mat_m * mat_ell_width;

        std::vector<float> h_csr_val(static_cast<size_t>(batch_count) * csr_val_stride);
        for(int64_t b = 0; b < batch_count; ++b)
        {
            for(int64_t row = 0; row < mat_m; ++row)
            {
                h_csr_val[static_cast<size_t>(b) * csr_val_stride + row] = batch_value(b, row);
            }
        }

        // Pre-fill the output with a sentinel no batch can legitimately produce,
        // so an entry the kernel never reaches is caught instead of passing on a
        // stale zero.
        const float              sentinel = 12345.0f;
        const std::vector<float> h_ell_val_init(static_cast<size_t>(batch_count) * ell_val_stride,
                                                sentinel);

        device_vector<rocsparse_int> d_csr_row_ptr{csr_row_ptr};
        device_vector<rocsparse_int> d_csr_col_ind{csr_col_ind};
        device_vector<float>         d_csr_val{h_csr_val};
        device_vector<float>         d_ell_val{h_ell_val_init};
        device_vector<rocsparse_int> d_ell_col_ind{
            std::vector<rocsparse_int>(static_cast<size_t>(ell_val_stride), -2)};
        ASSERT_TRUE(d_csr_row_ptr.ptr && d_csr_col_ind.ptr && d_csr_val.ptr && d_ell_val.ptr
                    && d_ell_col_ind.ptr);

        MatDescr csr_descr;
        MatDescr ell_descr;
        ASSERT_TRUE(csr_descr.d && ell_descr.d);

        ASSERT_EQ(rocsparse_scsr2ell_strided_batched(handle,
                                                     static_cast<rocsparse_int>(batch_count),
                                                     mat_m,
                                                     csr_descr.d,
                                                     d_csr_val,
                                                     csr_val_stride,
                                                     d_csr_row_ptr,
                                                     d_csr_col_ind,
                                                     ell_descr.d,
                                                     mat_ell_width,
                                                     d_ell_val,
                                                     ell_val_stride,
                                                     d_ell_col_ind),
                  rocsparse_status_success)
            << "launch rejected for batch_count = " << batch_count
            << " (grid.y cap = " << grid_y_cap << ")";
        UT_CHECK_HIP(hipDeviceSynchronize());

        const std::vector<float> ell_val = to_host(d_ell_val);
        ASSERT_EQ(ell_val.size(), static_cast<size_t>(batch_count) * ell_val_stride);

        // Report the first wrong entry rather than emitting 140000 assertions.
        int64_t bad_batch = -1;
        int64_t bad_row   = -1;
        for(int64_t b = 0; b < batch_count && bad_batch < 0; ++b)
        {
            for(int64_t row = 0; row < mat_m; ++row)
            {
                if(ell_val[static_cast<size_t>(b) * ell_val_stride + row] != batch_value(b, row))
                {
                    bad_batch = b;
                    bad_row   = row;
                    break;
                }
            }
        }
        EXPECT_EQ(bad_batch, -1) << "ELL value not written for batch " << bad_batch << " row "
                                 << bad_row << " (batch_count = " << batch_count
                                 << ", grid.y cap = " << grid_y_cap << "); batches at or above "
                                 << grid_y_cap << " are reachable only through the kernel's "
                                 << "batch grid-stride loop";

        // Spell out the two batches that specifically require the fix, so a
        // failure names them even if the scan above is ever relaxed.
        ASSERT_GT(batch_count, grid_y_cap);
        for(const int64_t b : {grid_y_cap, batch_count - 1})
        {
            EXPECT_EQ(ell_val[static_cast<size_t>(b) * ell_val_stride], batch_value(b, 0))
                << "batch " << b << " past the grid.y clamp was not converted";
        }

        // The column indices are shared by all batches (there is no column-index
        // stride), so every batch writes the same diagonal pattern.
        const std::vector<rocsparse_int> ell_col_ind = to_host(d_ell_col_ind);
        EXPECT_EQ(ell_col_ind[0], 0);
        EXPECT_EQ(ell_col_ind[1], 1);
    }
};

// The smallest batch_count the clamp actually bites on, so blockIdx.y == 0 must
// run exactly two loop iterations and every other block exactly one.
//
// Note this count does NOT necessarily fail to launch on every device: gfx1201
// reports maxGridSize[1] == 65535 but its runtime still accepts grid.y == 65536
// and only rejects 65537 and above. This case is therefore a test of the
// grid-stride loop, not of the launch, which is why the batch values are
// verified rather than just the status.
TEST_F(Csr2EllStridedBatched, grid_stride_at_first_clamped_batch_count)
{
    run(grid_y_cap + 1);
}

// Comfortably past the cap, so the tail [65535, 70000) exercises the stride
// rather than just its first extra step.
TEST_F(Csr2EllStridedBatched, grid_stride_beyond_clamp)
{
    run(70000);
}
