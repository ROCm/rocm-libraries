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
// grid.y (one block per BLK_SIZE_Y-wide column panel). grid.y is limited to 65535
// by hardware, so the launch is CLAMPED there and the kernel wrapper grid-strides
// over the remaining column panels. This test drives a column count strictly
// larger than 65535 * 32 (32 being the widest tile the launch can pick, so the
// clamp engages on every architecture) and checks EVERY output column. A
// regression in either the clamp or the panel stride is caught:
//
//   * clamp removed  -> the launch is rejected (hipErrorInvalidConfiguration) and
//                       NO column is computed, so C keeps its sentinel from index 0.
//   * stride removed -> only the first grid sweep is computed and the column tail
//                       keeps its sentinel value.
//
// The output comparison is what catches both, not the returned status: an over-large
// grid.y is rejected asynchronously, so rocsparse_spmm still reports success.
//
// The test is cheap on purpose: the defect lives on the COLUMN axis, so the
// sparse operand is a single 2x2 identity block. B and C are therefore 2 x n
// matrices, i.e. roughly 17 MB each in single precision -- this runs on any GPU
// (including the 15 GB gfx1201) and is permanently enabled, with no
// device-memory guard that could silently drop it at instantiation.
//
// A = I and alpha/beta = 1/0, so C must equal B bit-for-bit and the check can be
// an exact comparison.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include <cstdint>

using namespace rocsparse_ut;

namespace
{
    // The launch clamps grid.y at this value (rocsparse::bellmm_max_grid_y, kept
    // translation-unit local in the library, hence duplicated here).
    constexpr int64_t bellmm_max_grid_y = 65535;

    // Widest square tile rocsparse::bellmm_general_tile_size can return, so
    // `beyond_clamp` exceeds the clamp on wave32 (where the tile may shrink to 8
    // and the clamp is hit even earlier) as well as on wave64 (fixed 32).
    constexpr int64_t widest_tile = 32;

    // One panel past the clamp on the widest tile: the tail columns are reachable
    // only through the kernel's column-panel grid-stride loop.
    constexpr int64_t beyond_clamp = bellmm_max_grid_y * widest_tile + widest_tile + 8; // 2097160

    // A 2x2 block row of A: one block column, block dimension 2.
    constexpr int64_t block_dim = 2;
    constexpr int64_t mb        = 1;
    constexpr int64_t kb        = 1;
    constexpr int64_t bell_cols = block_dim; // ell_block_width == 1
    constexpr int64_t m         = mb * block_dim;
    constexpr int64_t k         = kb * block_dim;

    // Column-distinguishing value for B[row, col]. Every value is an exact
    // integer below 2^24, so it round-trips through float without rounding and
    // the output comparison can be exact. Two different (row, col) pairs never
    // collide, so a panel written at the wrong column offset is detected.
    template <typename T>
    T b_value(int64_t row, int64_t col)
    {
        return static_cast<T>(col + 1 + row * 4194304);
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
                if(got[idx] != b_value<T>(row, col))
                {
                    return idx;
                }
            }
        }
        return -1;
    }

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
    const int64_t n = beyond_clamp;
    ASSERT_GT(n, bellmm_max_grid_y * widest_tile);

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

    // C: m x n, column order, ldc = m. Pre-filled with a sentinel so a column
    // never visited by the kernel is caught (beta == 0, so every visited element
    // is overwritten).
    device_vector<T> dC(std::vector<T>(static_cast<size_t>(m) * n, static_cast<T>(-1)));

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
    const T beta  = static_cast<T>(0);

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
        << "bellmm launch failed for n = " << n << " (grid.y clamp = " << bellmm_max_grid_y << ")";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // A = I, alpha = 1, beta = 0 => C == B exactly.
    const std::vector<T> hC  = to_host(dC);
    const int64_t        bad = first_mismatch<T>(hC, m, m, n);
    EXPECT_EQ(bad, -1) << "C[" << (bad % m) << ", " << (bad / m) << "] = " << hC[bad]
                       << ", expected " << b_value<T>(bad % m, bad / m) << " (n = " << n
                       << ", grid.y clamp = " << bellmm_max_grid_y
                       << "; a column past the clamp is reached only by the kernel's "
                          "column-panel grid-stride loop)";
}

TEST_F(BellMM, grid_stride_beyond_grid_y_clamp_f32)
{
    run_bellmm_beyond_grid_y_clamp<float>(handle);
}

TEST_F(BellMM, grid_stride_beyond_grid_y_clamp_f64)
{
    run_bellmm_beyond_grid_y_clamp<double>(handle);
}
