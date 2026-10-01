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
// rocsparse::gthr_strided_batched_template clamps grid.x with get_grid_size_x
// and grid.y with get_grid_size_y (AISPARSE-649, AISPARSE-654). It launches the
// straight-line kernel when neither clamp binds and the grid-stride kernel
// otherwise. With 512-thread blocks the x clamp binds at (2^32 - 1) / 512 =
// 8,388,607 blocks, about 2^32 nonzeros, and the y clamp at 65,535 batches.
//
// The tests shrink maxGridSize[0] to 1, 3 or 7 blocks, or maxGridSize[1] to 1
// or 2, so a small problem takes the clamped path, and compare against a host
// gather. x_val is poisoned first, so a clamped launch that runs the
// straight-line kernel leaves elements or whole batches ungathered and fails.
// The gaps between batches must stay poisoned.
//

#include "unit_test_utils.hpp"

#include "rocsparse_gthr.hpp"
#include "unit_test_grid_clamp.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // 7 * 512 = 3584 elements per sweep at the largest x limit, so every limit
    // needs at least three sweeps. Not a multiple of 512, so the last block is
    // partial.
    constexpr int64_t NNZ = 10007;

    // Dense length per batch. Indices spread over all of it.
    constexpr int64_t Y_SIZE = 2 * NNZ + 3;

    constexpr int64_t BATCH_COUNT  = 5;
    constexpr int64_t Y_STRIDE     = Y_SIZE + 13;
    constexpr int64_t X_VAL_STRIDE = NNZ + 11;

    // Temporarily shrink the grid.y limit that get_grid_size_y clamps against.
    struct ScopedMaxGridSizeY
    {
        rocsparse_handle handle;
        int              saved;

        ScopedMaxGridSizeY(rocsparse_handle h, int limit)
            : handle(h)
            , saved(h->properties.maxGridSize[1])
        {
            handle->properties.maxGridSize[1] = limit;
        }

        ~ScopedMaxGridSizeY()
        {
            handle->properties.maxGridSize[1] = saved;
        }

        ScopedMaxGridSizeY(const ScopedMaxGridSizeY&) = delete;

        ScopedMaxGridSizeY& operator=(const ScopedMaxGridSizeY&) = delete;
    };

    template <typename T>
    T make_value(int64_t i)
    {
        if constexpr(std::is_same<T, rocsparse_float_complex>{}
                     || std::is_same<T, rocsparse_double_complex>{})
        {
            return T(static_cast<float>(i % 251) + 0.5f, static_cast<float>(i % 241) + 1.0f);
        }
        else
        {
            return static_cast<T>(i % 251) + static_cast<T>(0.5);
        }
    }

    // Distinct from every make_value(i), i >= 0.
    template <typename T>
    T poison()
    {
        return make_value<T>(-7);
    }

    template <typename I, typename T>
    void check_gthr(rocsparse_handle handle, int64_t batch_count, rocsparse_index_base base)
    {
        std::vector<I> h_ind(NNZ);
        for(int64_t i = 0; i < NNZ; ++i)
        {
            h_ind[i] = static_cast<I>((i * 7919) % Y_SIZE + base);
        }

        const int64_t  y_stride = (batch_count > 1) ? Y_STRIDE : 0;
        const int64_t  x_stride = (batch_count > 1) ? X_VAL_STRIDE : 0;
        std::vector<T> h_y(Y_SIZE + y_stride * (batch_count - 1));
        for(size_t i = 0; i < h_y.size(); ++i)
        {
            h_y[i] = make_value<T>(i);
        }
        std::vector<T> h_x(NNZ + x_stride * (batch_count - 1), poison<T>());

        device_vector<I> d_ind{h_ind};
        device_vector<T> d_y{h_y};
        device_vector<T> d_x{h_x};
        ASSERT_TRUE(d_ind.ptr);
        ASSERT_TRUE(d_y.ptr);
        ASSERT_TRUE(d_x.ptr);

        (void)hipGetLastError();
        ASSERT_EQ(
            (rocsparse::gthr_strided_batched_template<I, T>(
                handle, batch_count, NNZ, d_y.ptr, y_stride, d_x.ptr, x_stride, d_ind.ptr, base)),
            rocsparse_status_success);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<T> h_out = to_host(d_x);
        for(int64_t i = 0; i < static_cast<int64_t>(h_out.size()); ++i)
        {
            const int64_t b  = (x_stride > 0) ? i / x_stride : 0;
            const int64_t k  = i - b * x_stride;
            const T expected = (k < NNZ) ? h_y[b * y_stride + static_cast<int64_t>(h_ind[k]) - base]
                                         : poison<T>();
            ASSERT_EQ(h_out[i], expected) << "batch " << b << " element " << k;
        }
    }

    template <typename I>
    void check_all_types(rocsparse_handle handle, int64_t batch_count)
    {
        for(const rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            check_gthr<I, float>(handle, batch_count, base);
            check_gthr<I, double>(handle, batch_count, base);
            check_gthr<I, rocsparse_double_complex>(handle, batch_count, base);
        }
    }
}

// grid.x clamped below the (NNZ - 1) / 512 + 1 = 20 blocks NNZ needs.
class GthrForcedClampX : public HandleTest, public ::testing::WithParamInterface<int>
{
};

TEST_P(GthrForcedClampX, single_matches_host)
{
    const ScopedMaxGridSizeX clamp(handle, GetParam());
    check_all_types<int32_t>(handle, 1);
    check_all_types<int64_t>(handle, 1);
}

TEST_P(GthrForcedClampX, strided_batched_matches_host)
{
    const ScopedMaxGridSizeX clamp(handle, GetParam());
    check_all_types<int32_t>(handle, BATCH_COUNT);
    check_all_types<int64_t>(handle, BATCH_COUNT);
}

INSTANTIATE_TEST_SUITE_P(GthrForcedClampX, GthrForcedClampX, ::testing::Values(1, 3, 7));

// grid.y clamped below BATCH_COUNT while grid.x is not clamped: the clamp on y
// alone must select the grid-stride kernel.
class GthrForcedClampY : public HandleTest, public ::testing::WithParamInterface<int>
{
};

TEST_P(GthrForcedClampY, strided_batched_matches_host)
{
    const ScopedMaxGridSizeY clamp(handle, GetParam());
    check_all_types<int32_t>(handle, BATCH_COUNT);
    check_all_types<int64_t>(handle, BATCH_COUNT);
}

INSTANTIATE_TEST_SUITE_P(GthrForcedClampY, GthrForcedClampY, ::testing::Values(1, 2));

// The unclamped straight-line path.
class GthrUnclamped : public HandleTest
{
};

TEST_F(GthrUnclamped, matches_host)
{
    check_all_types<int32_t>(handle, 1);
    check_all_types<int64_t>(handle, BATCH_COUNT);
}
