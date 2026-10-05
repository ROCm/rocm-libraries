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
// Regression tests for the dense element-count grids of the shared helpers in
// library/src/common/rocsparse_common.cpp (AISPARSE-699):
// rocsparse::valset_2d, scale_array, scale_2d_array and axpby_array_batched.
//
// FOCUS. A dense element count is the product of two index-typed extents. The
// host sized its grid from the 64-bit product -- dim3((int64_t(m) * n - 1) / 256
// + 1) -- but valset_2d_kernel and scale_2d_kernel recomputed the same product in
// the 32-bit template index type for their bounds check (gid >= m * n). At
// m = n = 46341 the int32_t product wraps negative, so the guard discarded EVERY
// thread of a correctly sized grid: coo2dense / coo2dense_aos / ell2dense /
// csx2dense returned an unwritten matrix, and csrmm / coomm / bsrmm / bellmm /
// gebsrmm returned C unscaled. Neither form is a launch error, so the debug
// launch harness cannot see it. The fix forms the count in int64_t once, uses the
// same expression for the kernel bound, widens the flattened index to 64 bits and
// grid-strides over a clamped grid.
//
// TWO CHEAP TRICKS make the real 46341 x 46341 threshold testable. That matrix is
// 8.6 GB in single precision and does not fit the 15 GB gfx1201 alongside its
// readback, so:
//
//   * ld = 1 folds the logical 46341 x 46341 matrix onto m + n - 1 == 92681 real
//     elements. valset_2d and scale_2d_array with scalar 0 both perform an
//     idempotent STORE, so aliasing cannot change the expected result, while the
//     kernel still walks the full 2147488281-element index range. Footprint:
//     ~740 kB including the shadow region below.
//
//   * A SHADOW REGION of 92681 elements is allocated in front of the matrix and
//     checked to be untouched. The element count exceeds INT32_MAX by 4634, so
//     the last 19 blocks of the real grid produce flattened indices past
//     INT32_MAX. A fix that widened the guard but left the index in int32_t wraps
//     those to about -2147478785, which maps to array offsets in [-92680, -46049]
//     -- inside the shadow. This is what makes the two halves of the defect
//     separately visible: widen only the bound and the shadow is written and the
//     tail is missing; widen neither and nothing is written at all.
//
// The grid-stride loops are covered separately by shrinking
// handle->properties.maxGridSize[0] -- the very limit the new code clamps against
// -- to a handful of blocks and running a 200 x 200 matrix (the idiom introduced
// by AISPARSE-702). Reaching the clamp for real needs ~5.5e11 elements.
//
// There is deliberately NO device-memory guard and no size-based skip anywhere in
// this file: every case is well under a megabyte and runs on every GPU.
//
// TARGET: rocsparse_common.cpp is compiled into rocsparse-unit-test-device
// (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES) because librocsparse hides these
// symbols; the tests need a real device, so this is the GPU binary, not the
// host-only rocsparse-unit-test.
//
#include "unit_test_utils.hpp"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit the
// rocsparse_common.cpp launches now clamp grid.x against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

// Internal declarations of the helpers under test.
#include "rocsparse_common.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // Smallest square dense extent whose element count overflows int32_t:
    // 46341 * 46341 == 2147488281 == INT32_MAX + 4634.
    constexpr int32_t overflow_dim = 46341;
    constexpr int64_t overflow_nelm
        = static_cast<int64_t>(overflow_dim) * static_cast<int64_t>(overflow_dim);

    static_assert(overflow_nelm > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                  "the element count must exceed INT32_MAX for these tests to be meaningful");

    // With ld == 1 the logical matrix aliases onto indices [0, m + n - 2].
    constexpr int64_t aliased_count = static_cast<int64_t>(overflow_dim) + overflow_dim - 1;

    // Elements reserved in front of the matrix to catch a 32-bit flattened index
    // (see the file header). The most negative offset such an index can produce is
    // -92680, so aliased_count elements of shadow are sufficient.
    constexpr int64_t shadow_count = aliased_count;

    // Small matrix for the grid-stride cases, and a grid.x limit far below the
    // 157 blocks its element count asks for, so ~98% of the elements are reached
    // only by the stride loop.
    constexpr int32_t small_dim      = 200;
    constexpr int64_t small_nelm     = static_cast<int64_t>(small_dim) * small_dim; // 40000
    constexpr int     clamped_grid_x = 3;

    constexpr int64_t vector_length = 10000;

    // First index whose value differs from `expected`, or -1 if all match. One
    // precise gtest failure instead of tens of thousands of EXPECT_EQ macros.
    template <typename T>
    int64_t first_mismatch(const std::vector<T>& got, T expected, int64_t begin, int64_t end)
    {
        for(int64_t i = begin; i < end; ++i)
        {
            if(got[i] != expected)
            {
                return i;
            }
        }
        return -1;
    }

    // Thin wrappers around the templates under test. The explicit template
    // argument lists contain commas, which a gtest assertion macro cannot see
    // through, and naming the instantiation once keeps the cases readable.
    rocsparse_status valset_2d_f32(rocsparse_handle handle,
                                   int32_t          m,
                                   int32_t          n,
                                   int64_t          ld,
                                   float            value,
                                   float*           array,
                                   rocsparse_order  order)
    {
        return rocsparse::valset_2d<int32_t, float>(handle, m, n, ld, value, array, order);
    }

    rocsparse_status scale_2d_array_f32(rocsparse_handle handle,
                                        int32_t          m,
                                        int32_t          n,
                                        int64_t          ld,
                                        int64_t          batch_count,
                                        int64_t          stride,
                                        const float*     scalar,
                                        float*           array,
                                        rocsparse_order  order)
    {
        return rocsparse::scale_2d_array<int32_t, float, float>(
            handle, m, n, ld, batch_count, stride, scalar, array, order);
    }

    rocsparse_status
        scale_array_f32(rocsparse_handle handle, int64_t length, const float* scalar, float* array)
    {
        return rocsparse::scale_array<int64_t, float, float>(handle, length, scalar, array);
    }

    rocsparse_status axpby_array_batched_f32(rocsparse_handle handle,
                                             int64_t          length,
                                             rocsparse_int    num_extra,
                                             const float*     gamma_device_array,
                                             const float**    x_arrays,
                                             const float*     beta,
                                             float*           y_array)
    {
        return rocsparse::axpby_array_batched<int64_t, float, float, float>(
            handle, length, num_extra, gamma_device_array, x_arrays, beta, y_array);
    }

    using CommonGrids = HandleTest;
}

// ---------------------------------------------------------------------------
// valset_2d: 64-bit element count
// ---------------------------------------------------------------------------

TEST_F(CommonGrids, valset_2d_element_count_is_64bit)
{
    std::vector<float>   h(shadow_count + aliased_count, 0.0f);
    device_vector<float> d(h);
    ASSERT_NE(d.ptr, nullptr);

    float* const matrix = d.ptr + shadow_count;

    ASSERT_EQ(
        valset_2d_f32(
            handle, overflow_dim, overflow_dim, /*ld=*/1, 3.5f, matrix, rocsparse_order_column),
        rocsparse_status_success);
    UT_CHECK_HIP(hipDeviceSynchronize());

    h = to_host(d);

    EXPECT_EQ(first_mismatch(h, 3.5f, shadow_count, shadow_count + aliased_count), -1)
        << "valset_2d left elements of a " << overflow_dim << " x " << overflow_dim
        << " matrix unwritten: the kernel bound is not the 64-bit element count " << overflow_nelm;

    EXPECT_EQ(first_mismatch(h, 0.0f, 0, shadow_count), -1)
        << "valset_2d wrote below the matrix: the flattened element index wrapped "
           "through int32_t past INT32_MAX";
}

TEST_F(CommonGrids, valset_2d_grid_stride_column_order)
{
    std::vector<float>   h(small_nelm, 0.0f);
    device_vector<float> d(h);
    ASSERT_NE(d.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(valset_2d_f32(
                      handle, small_dim, small_dim, small_dim, 7.0f, d.ptr, rocsparse_order_column),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    h = to_host(d);
    EXPECT_EQ(first_mismatch(h, 7.0f, 0, small_nelm), -1)
        << "valset_2d did not grid-stride over a grid.x clamped to " << clamped_grid_x << " blocks";
}

TEST_F(CommonGrids, valset_2d_grid_stride_row_order)
{
    std::vector<float>   h(small_nelm, 0.0f);
    device_vector<float> d(h);
    ASSERT_NE(d.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(valset_2d_f32(
                      handle, small_dim, small_dim, small_dim, 7.0f, d.ptr, rocsparse_order_row),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    h = to_host(d);
    EXPECT_EQ(first_mismatch(h, 7.0f, 0, small_nelm), -1)
        << "valset_2d (row order) did not grid-stride over a clamped grid.x";
}

// ---------------------------------------------------------------------------
// scale_2d_array: 64-bit element count
// ---------------------------------------------------------------------------

TEST_F(CommonGrids, scale_2d_array_element_count_is_64bit)
{
    // scalar 0 makes scale_2d_device an idempotent store, which is what allows the
    // ld == 1 aliasing; a multiply would be applied once per aliasing thread.
    const float scalar = 0.0f;

    std::vector<float>   h(shadow_count + aliased_count, 1.0f);
    device_vector<float> d(h);
    ASSERT_NE(d.ptr, nullptr);

    float* const matrix = d.ptr + shadow_count;

    ASSERT_EQ(scale_2d_array_f32(handle,
                                 overflow_dim,
                                 overflow_dim,
                                 /*ld=*/1,
                                 /*batch_count=*/1,
                                 /*stride=*/0,
                                 &scalar,
                                 matrix,
                                 rocsparse_order_column),
              rocsparse_status_success);
    UT_CHECK_HIP(hipDeviceSynchronize());

    h = to_host(d);

    EXPECT_EQ(first_mismatch(h, 0.0f, shadow_count, shadow_count + aliased_count), -1)
        << "scale_2d_array left elements of a " << overflow_dim << " x " << overflow_dim
        << " matrix unscaled: the kernel bound is not the 64-bit element count " << overflow_nelm;

    EXPECT_EQ(first_mismatch(h, 1.0f, 0, shadow_count), -1)
        << "scale_2d_array wrote below the matrix: the flattened element index "
           "wrapped through int32_t past INT32_MAX";
}

TEST_F(CommonGrids, scale_2d_array_grid_stride_batched)
{
    constexpr int64_t batch_count = 2;
    const float       scalar      = 2.0f;

    std::vector<float>   h(batch_count * small_nelm, 1.0f);
    device_vector<float> d(h);
    ASSERT_NE(d.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(scale_2d_array_f32(handle,
                                     small_dim,
                                     small_dim,
                                     small_dim,
                                     batch_count,
                                     small_nelm,
                                     &scalar,
                                     d.ptr,
                                     rocsparse_order_column),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    h = to_host(d);
    EXPECT_EQ(first_mismatch(h, 2.0f, 0, batch_count * small_nelm), -1)
        << "scale_2d_array did not grid-stride over a grid.x clamped to " << clamped_grid_x
        << " blocks";
}

// ---------------------------------------------------------------------------
// scale_array / axpby_array_batched
//
// Both take a single index-typed length rather than a product, so they were never
// exposed to the overflow above. They were widened and given the same stride loop
// so every kernel in the file shares one indexing idiom; these two cases hold that
// consistency in place.
// ---------------------------------------------------------------------------

TEST_F(CommonGrids, scale_array_grid_stride)
{
    const float scalar = 3.0f;

    std::vector<float>   h(vector_length, 1.0f);
    device_vector<float> d(h);
    ASSERT_NE(d.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(scale_array_f32(handle, vector_length, &scalar, d.ptr), rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    h = to_host(d);
    EXPECT_EQ(first_mismatch(h, 3.0f, 0, vector_length), -1)
        << "scale_array did not grid-stride over a clamped grid.x";
}

TEST_F(CommonGrids, axpby_array_batched_grid_stride)
{
    constexpr rocsparse_int num_extra = 2;
    const float             beta      = 6.0f;

    // gamma and the array of x pointers are read directly by the kernel, so both
    // live in device memory; beta follows the handle pointer mode (host).
    device_vector<float> d_x0(std::vector<float>(vector_length, 2.0f));
    device_vector<float> d_x1(std::vector<float>(vector_length, 5.0f));
    device_vector<float> d_y(std::vector<float>(vector_length, 1.0f));
    device_vector<float> d_gamma(std::vector<float>{3.0f, 4.0f});
    ASSERT_NE(d_x0.ptr, nullptr);
    ASSERT_NE(d_x1.ptr, nullptr);
    ASSERT_NE(d_y.ptr, nullptr);
    ASSERT_NE(d_gamma.ptr, nullptr);

    device_vector<const float*> d_x_arrays(std::vector<const float*>{d_x0.ptr, d_x1.ptr});
    ASSERT_NE(d_x_arrays.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(
            axpby_array_batched_f32(
                handle, vector_length, num_extra, d_gamma.ptr, d_x_arrays.ptr, &beta, d_y.ptr),
            rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    // beta * y + gamma0 * x0 + gamma1 * x1 == 6 * 1 + 3 * 2 + 4 * 5 == 32
    const std::vector<float> got = to_host(d_y);
    EXPECT_EQ(first_mismatch(got, 32.0f, 0, vector_length), -1)
        << "axpby_array_batched did not grid-stride over a clamped grid.x";
}

// ---------------------------------------------------------------------------
// gemmi: the caller must not truncate the element count before the helper sees it
//
// rocsparse_?gemmi's k == 0 and alpha == 0 shortcuts pass the element count of C
// straight to scale_array. That count was computed as rocsparse_int in the caller,
// so it arrived already overflowed and the helper sized a wrong grid from it. The
// overflow threshold itself is NOT testable here: the length is flat (ldc is not
// involved) so the ld == 1 aliasing above does not apply, and 46341 * 46341 floats
// is 8.6 GB. This case pins the corrected call site functionally instead.
// ---------------------------------------------------------------------------

TEST_F(CommonGrids, gemmi_k_zero_scales_c_with_beta)
{
    constexpr rocsparse_int m = 7;
    constexpr rocsparse_int n = 5;

    const float alpha = 1.0f;
    const float beta  = 2.5f;

    std::vector<float> h_C(static_cast<size_t>(m) * n);
    for(size_t i = 0; i < h_C.size(); ++i)
    {
        h_C[i] = static_cast<float>(1 + (i % 4));
    }
    device_vector<float> d_C(h_C);
    ASSERT_NE(d_C.ptr, nullptr);

    device_vector<rocsparse_int> d_row_ptr(std::vector<rocsparse_int>(n + 1, 0));
    ASSERT_NE(d_row_ptr.ptr, nullptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    EXPECT_EQ(rocsparse_sgemmi(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_transpose,
                               m,
                               n,
                               /*k=*/0,
                               /*nnz=*/0,
                               &alpha,
                               nullptr,
                               m,
                               descr,
                               nullptr,
                               d_row_ptr.ptr,
                               nullptr,
                               &beta,
                               d_C.ptr,
                               m),
              rocsparse_status_success);
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<float> got = to_host(d_C);
    for(size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_FLOAT_EQ(got[i], beta * h_C[i]) << "gemmi k == 0 mis-scaled C at " << i;
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}
