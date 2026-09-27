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
// Regression tests for the block-index arithmetic of the three remaining
// common-helper launches hardened by AISPARSE-700:
//
//   rocsparse::valset                            (rocsparse_valset.cpp)
//   rocsparse::singularity_get_async /
//   rocsparse::singularity_get_position_async    (rocsparse_singularity.cpp)
//   rocsparse::dense_transpose_strided_batched /
//   rocsparse::dense_transpose_back_*            (rocsparse_dense_transpose*.cpp)
//
// FOCUS. Each of these sized grid.x from an int64_t count -- correctly -- and
// then formed the element/row index in the kernel as
//
//     hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x
//
// `hipBlockIdx_x` and `BLOCKSIZE` are both unsigned int, so that product is
// evaluated in 32-bit unsigned arithmetic and wraps at 2^32 NO MATTER what type
// the result is assigned to. In valset the result was assigned to an int64_t
// (launch_valset always passes an int64_t length, so the kernel's index template
// parameter was only ever deduced as int64_t) and in the transposes to an
// int64_t in the >INT32_MAX dispatch, and it wrapped anyway. In singularity the
// index was `const auto`, which deduces plain unsigned int. None of the three
// had a grid-stride loop, so there was also nothing to recover the tail past a
// clamped grid.
//
// WHY THESE THRESHOLDS ARE NOT TESTED DIRECTLY. The wrap needs 2^32 elements:
// 17.2 GB for an int32_t valset array, 4 billion matrix rows for the transposes,
// 4 billion batch entries for singularity. None of that fits the 15 GB gfx1201,
// and the smallest machine that could hold it is not the machine CI runs on.
// What IS reachable -- and what the fix actually consists of -- is the pairing of
//
//   (a) a grid.x clamped against handle->properties.maxGridSize[0], and
//   (b) a grid-stride loop that covers the count the clamp dropped.
//
// Every case below shrinks maxGridSize[0] with ScopedMaxGridSizeX so the real
// launch is clamped far below what the count asks for, and then checks that the
// full count was still processed. That exercises the ceiling-division, the
// clamp and the stride loop against the live kernels. Remove the stride loop and
// every case here fails; remove the clamp and the launches are correct only
// while the grid happens to fit.
//
// The 64-bit widening of the index itself is not separately observable at these
// sizes -- it is a precondition for the stride loop to be able to address past
// 2^32 at all -- so it is carried by review and by the -fsanitize=undefined run
// recorded on the ticket, not by a test.
//
// There is deliberately NO device-memory guard and no size-based skip anywhere
// in this file: the largest allocation is 5000 int64_t, so every case runs on
// every GPU.
//
// TARGET: rocsparse_valset.cpp, rocsparse_singularity.cpp and the two
// rocsparse_dense_transpose*.cpp are compiled into rocsparse-unit-test-device
// (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES) because librocsparse hides these
// symbols; the tests launch real kernels, so this is the GPU binary, not the
// host-only rocsparse-unit-test.
//
#include "unit_test_utils.hpp"

#include "unit_test_grid_clamp.hpp"

// Internal declarations of the helpers under test.
#include "rocsparse_dense_transpose.hpp"
#include "rocsparse_dense_transpose_back.hpp"
#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_singularity.hpp"
#include "rocsparse_valset.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // A grid.x limit far below what any count below asks for, so the great
    // majority of the work is reached only by the grid-stride loop.
    constexpr int clamped_grid_x = 3;

    // First index whose value differs from `expected`, or -1 if all match. One
    // precise gtest failure instead of thousands of EXPECT_EQ macros.
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

    using LargeGrids = HandleTest;
}

// ---------------------------------------------------------------------------
// valset
//
// valset_kernel runs 256 threads per block. With the limit shrunk to 3 blocks a
// single grid sweep covers 768 elements, so a length of 10007 needs 14 sweeps
// and 97.7% of the array is written only by the stride loop.
// ---------------------------------------------------------------------------

namespace
{
    constexpr int64_t valset_length = 10007; // deliberately not a multiple of 256

    template <typename T>
    void run_valset_grid_stride(rocsparse_handle handle, rocsparse_indextype indextype)
    {
        device_vector<T> d(std::vector<T>(valset_length, static_cast<T>(-1)));
        ASSERT_NE(d.ptr, nullptr);

        {
            ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
            ASSERT_EQ(rocsparse::valset(handle, valset_length, int64_t{42}, indextype, d.ptr),
                      rocsparse_status_success);
            UT_CHECK_HIP(hipDeviceSynchronize());
        }

        const std::vector<T> got = to_host(d);
        EXPECT_EQ(first_mismatch(got, static_cast<T>(42), 0, valset_length), -1)
            << "valset did not cover the whole length with grid.x clamped to " << clamped_grid_x
            << " blocks (length=" << valset_length << ", 256 threads/block)";
    }
}

TEST_F(LargeGrids, valset_grid_stride_i32)
{
    run_valset_grid_stride<int32_t>(handle, rocsparse_indextype_i32);
}

TEST_F(LargeGrids, valset_grid_stride_i64)
{
    run_valset_grid_stride<int64_t>(handle, rocsparse_indextype_i64);
}

// The ceiling division launch_valset sizes grid.x with is (length - 1) / 256 + 1,
// which is only well behaved for length >= 1. Zero-length valset is reached from
// the conversion routines whenever nnz == 0, so pin it: it must be a successful
// no-op, and in particular must not write element 0 of the array.
TEST_F(LargeGrids, valset_zero_length_is_a_no_op)
{
    device_vector<int32_t> d(std::vector<int32_t>{-1, -1, -1});
    ASSERT_NE(d.ptr, nullptr);

    ASSERT_EQ(rocsparse::valset(handle, int64_t{0}, int64_t{42}, rocsparse_indextype_i32, d.ptr),
              rocsparse_status_success);
    UT_CHECK_HIP(hipDeviceSynchronize());

    const std::vector<int32_t> got = to_host(d);
    EXPECT_EQ(first_mismatch(got, int32_t{-1}, 0, 3), -1)
        << "zero-length valset wrote to the array";
}

// A length exactly on a block boundary and one element past it: pins the
// ceiling division that feeds the clamp, which is the arithmetic the ticket asks
// to have covered.
TEST_F(LargeGrids, valset_block_boundary_lengths)
{
    for(const int64_t length :
        {int64_t{255}, int64_t{256}, int64_t{257}, int64_t{768}, int64_t{769}})
    {
        // One sentinel element past the end catches an off-by-one that rounds the
        // grid up too far without a bound check.
        device_vector<int32_t> d(std::vector<int32_t>(length + 1, -1));
        ASSERT_NE(d.ptr, nullptr);

        {
            ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
            ASSERT_EQ(rocsparse::valset(handle, length, int64_t{9}, rocsparse_indextype_i32, d.ptr),
                      rocsparse_status_success);
            UT_CHECK_HIP(hipDeviceSynchronize());
        }

        const std::vector<int32_t> got = to_host(d);
        EXPECT_EQ(first_mismatch(got, int32_t{9}, 0, length), -1)
            << "valset left an element unwritten at length " << length;
        EXPECT_EQ(got[length], -1) << "valset wrote one past the end at length " << length;
    }
}

// ---------------------------------------------------------------------------
// singularity (device pointer mode)
//
// markers2singularity / markers2position run 1024 threads per block. With the
// limit shrunk to 3 blocks one sweep covers 3072 entries, so a batch of 5000
// needs a second sweep and the tail [3072, 5000) is reached only by the stride
// loop.
//
// The host-pointer path is NOT exercised here and was NOT changed: it stages
// results through the handle buffer and only ever launches one buffer-sized
// chunk at a time, so its grid was never large enough to be at risk.
// ---------------------------------------------------------------------------

namespace
{
    constexpr int64_t singularity_batch_count = 5000;

    // Marker value meaning "no singularity at this batch entry", matching the
    // kernels' `mx`.
    constexpr int32_t no_marker = std::numeric_limits<int32_t>::max();

    // Build an `exact` marker set over the full batch: even entries carry a
    // singular position, odd entries carry none.
    void make_exact_markers(rocsparse::singular_info_t& exact,
                            rocsparse_handle            handle,
                            int32_t                     marker)
    {
        ASSERT_EQ(exact.create_singular_pivot_async(
                      singularity_batch_count, rocsparse_indextype_i32, handle->stream),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipStreamSynchronize(handle->stream));

        std::vector<int32_t> h(singularity_batch_count);
        for(int64_t i = 0; i < singularity_batch_count; ++i)
        {
            h[i] = (i % 2 == 0) ? marker : no_marker;
        }

        UT_CHECK_HIP(hipMemcpy(exact.get_position(),
                               h.data(),
                               sizeof(int32_t) * singularity_batch_count,
                               hipMemcpyHostToDevice));
    }
}

TEST_F(LargeGrids, singularity_get_async_grid_stride)
{
    constexpr int32_t marker = 7;

    rocsparse::singular_info_t exact;
    ASSERT_NO_FATAL_FAILURE(make_exact_markers(exact, handle, marker));

    device_vector<rocsparse_singularity> d_out(std::vector<rocsparse_singularity>(
        singularity_batch_count, static_cast<rocsparse_singularity>(-1)));
    ASSERT_NE(d_out.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse::singularity_get_async(handle,
                                                   singularity_batch_count,
                                                   /*symbolic=*/nullptr,
                                                   &exact,
                                                   /*near=*/nullptr,
                                                   rocsparse_pointer_mode_device,
                                                   d_out.ptr),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    const std::vector<rocsparse_singularity> got = to_host(d_out);
    for(int64_t i = 0; i < singularity_batch_count; ++i)
    {
        const rocsparse_singularity expected
            = (i % 2 == 0) ? rocsparse_singularity_numeric_exact : rocsparse_singularity_none;
        ASSERT_EQ(got[i], expected)
            << "markers2singularity did not classify batch entry " << i
            << " with grid.x clamped to " << clamped_grid_x
            << " blocks (batch_count=" << singularity_batch_count << ", 1024 threads/block)";
    }
}

TEST_F(LargeGrids, singularity_get_position_async_grid_stride)
{
    constexpr int32_t marker = 11;

    rocsparse::singular_info_t exact;
    ASSERT_NO_FATAL_FAILURE(make_exact_markers(exact, handle, marker));

    device_vector<int64_t> d_out(std::vector<int64_t>(singularity_batch_count, -99));
    ASSERT_NE(d_out.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse::singularity_get_position_async(handle,
                                                            singularity_batch_count,
                                                            /*symbolic=*/nullptr,
                                                            &exact,
                                                            /*near=*/nullptr,
                                                            rocsparse_pointer_mode_device,
                                                            rocsparse_indextype_i64,
                                                            d_out.ptr),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    const std::vector<int64_t> got = to_host(d_out);
    for(int64_t i = 0; i < singularity_batch_count; ++i)
    {
        const int64_t expected = (i % 2 == 0) ? marker : -1;
        ASSERT_EQ(got[i], expected)
            << "markers2position did not resolve batch entry " << i << " with grid.x clamped to "
            << clamped_grid_x << " blocks (batch_count=" << singularity_batch_count
            << ", 1024 threads/block)";
    }
}

// ---------------------------------------------------------------------------
// dense_transpose / dense_transpose_back
//
// Both tile 32 rows per block. With the limit shrunk to 3 blocks one sweep
// covers 96 rows, so m = 500 needs 6 sweeps and 81% of the rows are transposed
// only by the stride loop. n is deliberately not a multiple of the 32-wide tile
// so the column tail guards stay exercised.
// ---------------------------------------------------------------------------

namespace
{
    constexpr int64_t t_m = 500;
    constexpr int64_t t_n = 70;

    // A column-major m x n matrix with a value that identifies its position.
    std::vector<float> make_matrix(int64_t rows, int64_t cols, int64_t ld, float bias)
    {
        std::vector<float> a(static_cast<size_t>(ld) * cols, 0.0f);
        for(int64_t j = 0; j < cols; ++j)
        {
            for(int64_t i = 0; i < rows; ++i)
            {
                a[i + ld * j] = bias + static_cast<float>(i) + 1000.0f * static_cast<float>(j);
            }
        }
        return a;
    }
}

// B (n x m, ldb = n) = alpha * A^T, with A m x n, lda = m.
TEST_F(LargeGrids, dense_transpose_grid_stride)
{
    const float alpha = 2.0f;

    const std::vector<float> h_A = make_matrix(t_m, t_n, t_m, 0.0f);
    device_vector<float>     d_A(h_A);
    device_vector<float>     d_B(std::vector<float>(static_cast<size_t>(t_n) * t_m, -1.0f));
    ASSERT_NE(d_A.ptr, nullptr);
    ASSERT_NE(d_B.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse::dense_transpose(handle,
                                             rocsparse_pointer_mode_host,
                                             t_m,
                                             t_n,
                                             rocsparse_datatype_f32_r,
                                             &alpha,
                                             rocsparse_datatype_f32_r,
                                             d_A.ptr,
                                             t_m,
                                             rocsparse_datatype_f32_r,
                                             d_B.ptr,
                                             t_n),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    const std::vector<float> got = to_host(d_B);
    for(int64_t i = 0; i < t_m; ++i)
    {
        for(int64_t j = 0; j < t_n; ++j)
        {
            ASSERT_FLOAT_EQ(got[j + t_n * i], alpha * h_A[i + t_m * j])
                << "dense_transpose missed row " << i << " column " << j
                << " with grid.x clamped to " << clamped_grid_x << " blocks (m=" << t_m
                << ", 32 rows/block)";
        }
    }
}

// The same launch with a batch on grid.y, so the row stride loop has to be
// correct for every batch entry rather than only the first.
//
// Host pointer mode deliberately. In device pointer mode the kernel computes the
// per-batch scalar as `alpha + i * alpha_stride` AFTER dereferencing the pointer,
// i.e. it adds the batch offset to the scalar VALUE instead of indexing the
// array. That is pre-existing, unrelated to the block indexing this ticket is
// about, and untouched here; a test using device pointer mode with batch_count
// > 1 would be asserting on it.
TEST_F(LargeGrids, dense_transpose_strided_batched_grid_stride)
{
    constexpr int64_t batch_count = 3;
    const float       alpha       = 2.0f;

    const int64_t A_stride = t_m * t_n;
    const int64_t B_stride = t_n * t_m;

    std::vector<float> h_A(static_cast<size_t>(A_stride) * batch_count);
    for(int64_t b = 0; b < batch_count; ++b)
    {
        const std::vector<float> block = make_matrix(t_m, t_n, t_m, 100.0f * b);
        std::copy(block.begin(), block.end(), h_A.begin() + b * A_stride);
    }

    device_vector<float> d_A(h_A);
    device_vector<float> d_B(
        std::vector<float>(static_cast<size_t>(B_stride) * batch_count, -1.0f));
    ASSERT_NE(d_A.ptr, nullptr);
    ASSERT_NE(d_B.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse::dense_transpose_strided_batched(handle,
                                                             rocsparse_pointer_mode_host,
                                                             batch_count,
                                                             t_m,
                                                             t_n,
                                                             rocsparse_datatype_f32_r,
                                                             &alpha,
                                                             /*alpha_stride=*/0,
                                                             rocsparse_datatype_f32_r,
                                                             d_A.ptr,
                                                             t_m,
                                                             A_stride,
                                                             rocsparse_datatype_f32_r,
                                                             d_B.ptr,
                                                             t_n,
                                                             B_stride),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    const std::vector<float> got = to_host(d_B);
    for(int64_t b = 0; b < batch_count; ++b)
    {
        for(int64_t i = 0; i < t_m; ++i)
        {
            for(int64_t j = 0; j < t_n; ++j)
            {
                ASSERT_FLOAT_EQ(got[b * B_stride + j + t_n * i],
                                alpha * h_A[b * A_stride + i + t_m * j])
                    << "dense_transpose_strided_batched missed batch " << b << " row " << i
                    << " column " << j << " with grid.x clamped to " << clamped_grid_x << " blocks";
            }
        }
    }
}

// B (m x n, ldb = m) = A^T, with A n x m, lda = n.
TEST_F(LargeGrids, dense_transpose_back_grid_stride)
{
    const std::vector<float> h_A = make_matrix(t_n, t_m, t_n, 0.0f);
    device_vector<float>     d_A(h_A);
    device_vector<float>     d_B(std::vector<float>(static_cast<size_t>(t_m) * t_n, -1.0f));
    ASSERT_NE(d_A.ptr, nullptr);
    ASSERT_NE(d_B.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse::dense_transpose_back(handle,
                                                  t_m,
                                                  t_n,
                                                  rocsparse_datatype_f32_r,
                                                  d_A.ptr,
                                                  t_n,
                                                  rocsparse_datatype_f32_r,
                                                  d_B.ptr,
                                                  t_m),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    const std::vector<float> got = to_host(d_B);
    for(int64_t i = 0; i < t_m; ++i)
    {
        for(int64_t j = 0; j < t_n; ++j)
        {
            ASSERT_FLOAT_EQ(got[i + t_m * j], h_A[j + t_n * i])
                << "dense_transpose_back missed row " << i << " column " << j
                << " with grid.x clamped to " << clamped_grid_x << " blocks (m=" << t_m
                << ", 32 rows/block)";
        }
    }
}
