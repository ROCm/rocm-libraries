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
// Host-path unit test for the blocked-ELL SpMM (rocsparse_spmm with
// rocsparse_format_bell -> library/src/level3/rocsparse_bellmm_template_general.cpp).
//
// FOCUS (AISPARSE-667): the general bellmm kernel tiles the dense operand across
// grid.y (one block per BLK_SIZE_Y-wide column panel). The launch CLAMPS grid.y
// with rocsparse::get_grid_size_y, i.e. to the handle's maxGridSize[1], and the
// kernel wrapper grid-strides over the remaining column panels. This test reads
// that limit back from the handle, drives a column count strictly larger than
// limit * 32 (32 being the widest tile the launch can pick, so the clamp engages
// on every architecture) and checks EVERY output column. A
// regression in either the clamp or the panel stride is caught:
//
//   * clamp removed  -> the launch is rejected (hipErrorInvalidConfiguration) and
//                       NO column is computed, so C keeps its prefill value from
//                       index 0.
//   * stride removed -> only the first grid sweep is computed and the column tail
//                       keeps its prefill value.
//   * stride too small (columns visited more than once) -> those columns are
//                       accumulated into more than once and exceed the expectation.
//
// The output comparison is what catches both, not the returned status: an over-large
// grid.y is rejected asynchronously, so rocsparse_spmm still reports success.
//
// The test is cheap on purpose: the defect lives on the COLUMN axis, so the
// sparse operand is a single 2x2 identity block. B and C are therefore 2 x n
// matrices, i.e. roughly 17 MB each in single precision for a grid.y limit of
// 65535 (a device that reports more is capped to 65535 on the handle for the
// duration of the test, see ScopedMaxGridSizeY) -- this runs on any GPU
// (including the 15 GB gfx1201) and is permanently enabled, with no
// device-memory guard that could silently drop it at instantiation.
//
// A = I, alpha = 1 and beta = 1 with C pre-filled to a known value, so C must
// equal B + fill and the check can be an exact comparison. beta = 1 makes every
// update accumulate, so a column visited twice differs from a column visited once.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include "rocsparse_grid.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Largest grid.y limit the test sizes itself for. A device that reports a
    // larger maxGridSize[1] is capped to this on the handle, which keeps B and C
    // small and every value of b_value below exact in single precision.
    constexpr int max_tested_grid_y = 65535;

    // Widest square tile rocsparse::bellmm_general_tile_size can return, so
    // `n` exceeds the clamp on wave32 (where the tile may shrink to 8
    // and the clamp is hit even earlier) as well as on wave64 (fixed 32).
    constexpr int64_t widest_tile = 32;

    // A 2x2 block row of A: one block column, block dimension 2.
    constexpr int64_t block_dim = 2;
    constexpr int64_t mb        = 1;
    constexpr int64_t kb        = 1;
    constexpr int64_t bell_cols = block_dim; // ell_block_width == 1
    constexpr int64_t m         = mb * block_dim;
    constexpr int64_t k         = kb * block_dim;

    // Value C is pre-filled with. Nonzero and distinct from every b_value, so a
    // column that is never visited cannot match the expected b_value + fill.
    constexpr int fill_value = 3;

    // Column-distinguishing value for B[row, col]. Every value, and its sum with
    // fill_value, is an exact integer below 2^24 (a column visited twice, 2 * b +
    // fill, still is), so it round-trips through float without rounding and the
    // output comparison can be exact. Two different (row, col) pairs never
    // collide, so a panel written at the wrong column offset is detected.
    template <typename T>
    T b_value(int64_t row, int64_t col)
    {
        return static_cast<T>(col + 1 + row * 4194304);
    }

    // Expected C[row, col] for A = I, alpha = 1, beta = 1 and C pre-filled with
    // fill_value.
    template <typename T>
    T c_value(int64_t row, int64_t col)
    {
        return b_value<T>(row, col) + static_cast<T>(fill_value);
    }

    // Report the first index whose value differs from the expectation, or -1 if
    // all match. Cheaper, and yields one precise gtest failure, versus wrapping
    // four million EXPECT_EQ macros in a loop.
    template <typename T>
    int64_t first_mismatch(const std::vector<T>& got, int64_t ld, int64_t rows, int64_t cols)
    {
        for(int64_t col = 0; col < cols; ++col)
        {
            for(int64_t row = 0; row < rows; ++row)
            {
                const int64_t idx = row + col * ld;
                if(got[idx] != c_value<T>(row, col))
                {
                    return idx;
                }
            }
        }
        return -1;
    }

    // Temporarily lower handle->properties.maxGridSize[1] to at most `limit`, and
    // restore it on scope exit so the override cannot leak into the next test.
    struct ScopedMaxGridSizeY
    {
        rocsparse_handle handle;
        int              saved;

        ScopedMaxGridSizeY(rocsparse_handle h, int limit)
            : handle(h)
            , saved(h->properties.maxGridSize[1])
        {
            handle->properties.maxGridSize[1] = std::min(saved, limit);
        }

        ~ScopedMaxGridSizeY()
        {
            handle->properties.maxGridSize[1] = saved;
        }

        ScopedMaxGridSizeY(const ScopedMaxGridSizeY&) = delete;

        ScopedMaxGridSizeY& operator=(const ScopedMaxGridSizeY&) = delete;
    };

    struct SpMatDescr
    {
        rocsparse_spmat_descr d = nullptr;
        ~SpMatDescr()
        {
            if(d)
                (void)rocsparse_destroy_spmat_descr(d);
        }
    };

    struct DnMatDescr
    {
        rocsparse_dnmat_descr d = nullptr;
        ~DnMatDescr()
        {
            if(d)
                (void)rocsparse_destroy_dnmat_descr(d);
        }
    };
}

class BellMM : public HandleTest
{
};

template <typename T>
static void run_bellmm_beyond_grid_y_clamp(rocsparse_handle handle)
{
    const ScopedMaxGridSizeY cap(handle, max_tested_grid_y);

    // The grid.y extent the launch clamps to, read through the same helper.
    const int64_t grid_y_max
        = rocsparse::get_grid_size_y(handle, std::numeric_limits<int64_t>::max());

    // One panel past the clamp on the widest tile: the tail columns are reachable
    // only through the kernel's column-panel grid-stride loop. For a grid.y limit
    // of 65535 this is n = 2097160.
    const int64_t n = grid_y_max * widest_tile + widest_tile + 8;
    ASSERT_GT((n - 1) / widest_tile + 1, grid_y_max);

    // A: one 2x2 identity block, values in the row-major layout bellmm expects
    // (val[(block_row * block_dim + r) * bell_cols + ei * block_dim + c]).
    device_vector<int32_t> dA_col_ind{std::vector<int32_t>{0}};
    device_vector<T>       dA_val{
        std::vector<T>{static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)}};

    // B: k x n, column order, ldb = k.
    std::vector<T> hB(static_cast<size_t>(k) * n);
    for(int64_t col = 0; col < n; ++col)
    {
        for(int64_t row = 0; row < k; ++row)
        {
            hB[static_cast<size_t>(row + col * k)] = b_value<T>(row, col);
        }
    }
    device_vector<T> dB(hB);

    // C: m x n, column order, ldc = m. Pre-filled with fill_value and updated with
    // beta = 1, so a column never visited keeps fill_value and a column visited
    // more than once accumulates more than b_value.
    device_vector<T> dC(std::vector<T>(static_cast<size_t>(m) * n, static_cast<T>(fill_value)));

    ASSERT_NE(dA_col_ind.ptr, nullptr);
    ASSERT_NE(dA_val.ptr, nullptr);
    ASSERT_NE(dB.ptr, nullptr) << "could not allocate " << (sizeof(T) * k * n) << " bytes for B";
    ASSERT_NE(dC.ptr, nullptr) << "could not allocate " << (sizeof(T) * m * n) << " bytes for C";

    SpMatDescr mat_A;
    ASSERT_EQ(rocsparse_create_bell_descr(&mat_A.d,
                                          m,
                                          k,
                                          rocsparse_direction_row,
                                          block_dim,
                                          bell_cols,
                                          dA_col_ind.ptr,
                                          dA_val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          dt_of<T>()),
              rocsparse_status_success);

    DnMatDescr mat_B;
    ASSERT_EQ(
        rocsparse_create_dnmat_descr(&mat_B.d, k, n, k, dB.ptr, dt_of<T>(), rocsparse_order_column),
        rocsparse_status_success);

    DnMatDescr mat_C;
    ASSERT_EQ(
        rocsparse_create_dnmat_descr(&mat_C.d, m, n, m, dC.ptr, dt_of<T>(), rocsparse_order_column),
        rocsparse_status_success);

    const T alpha = static_cast<T>(1);
    const T beta  = static_cast<T>(1);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat_A.d,
                             mat_B.d,
                             &beta,
                             mat_C.d,
                             dt_of<T>(),
                             rocsparse_spmm_alg_bell,
                             rocsparse_spmm_stage_buffer_size,
                             &buffer_size,
                             nullptr),
              rocsparse_status_success);

    device_vector<char> dbuffer(buffer_size > 0 ? buffer_size : size_t{1});
    ASSERT_NE(dbuffer.ptr, nullptr);

    ASSERT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat_A.d,
                             mat_B.d,
                             &beta,
                             mat_C.d,
                             dt_of<T>(),
                             rocsparse_spmm_alg_bell,
                             rocsparse_spmm_stage_preprocess,
                             &buffer_size,
                             dbuffer.ptr),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_spmm(handle,
                             rocsparse_operation_none,
                             rocsparse_operation_none,
                             &alpha,
                             mat_A.d,
                             mat_B.d,
                             &beta,
                             mat_C.d,
                             dt_of<T>(),
                             rocsparse_spmm_alg_bell,
                             rocsparse_spmm_stage_compute,
                             &buffer_size,
                             dbuffer.ptr),
              rocsparse_status_success)
        << "bellmm launch failed for n = " << n << " (grid.y clamp = " << grid_y_max << ")";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // A = I, alpha = 1, beta = 1 => C == B + fill_value exactly.
    const std::vector<T> hC  = to_host(dC);
    const int64_t        bad = first_mismatch<T>(hC, m, m, n);
    EXPECT_EQ(bad, -1) << "C[" << (bad % m) << ", " << (bad / m) << "] = " << hC[bad]
                       << ", expected " << c_value<T>(bad % m, bad / m) << " (n = " << n
                       << ", grid.y clamp = " << grid_y_max
                       << "; a column past the clamp is reached only by the kernel's "
                          "column-panel grid-stride loop, and a column visited more than "
                          "once is accumulated into more than once)";
}

TEST_F(BellMM, grid_stride_beyond_grid_y_clamp_f32)
{
    run_bellmm_beyond_grid_y_clamp<float>(handle);
}

TEST_F(BellMM, grid_stride_beyond_grid_y_clamp_f64)
{
    run_bellmm_beyond_grid_y_clamp<double>(handle);
}
