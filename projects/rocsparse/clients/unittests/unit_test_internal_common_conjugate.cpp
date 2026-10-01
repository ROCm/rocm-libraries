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
// rocsparse::conjugate and rocsparse::conjugate_strided_batched clamp grid.x
// with get_grid_size_x and grid-stride when the clamp binds (AISPARSE-654). On
// the conjugate-transposed sptrsv path they run over nnz values, which can need
// more than (2^32 - 1) / 256 blocks. The tests shrink maxGridSize[0] to 1, 3 or
// 7 so a small array takes several sweeps, and compare against the host.
//

#include "unit_test_utils.hpp"

#include "rocsparse_conjugate.hpp"
#include "unit_test_grid_clamp.hpp"

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // 7 * 256 = 1792 elements per sweep at the largest limit, so every limit
    // needs at least three sweeps. Not a multiple of 256, so the last block is
    // partial.
    constexpr int64_t LENGTH = 5413;

    template <typename T>
    std::vector<T> make_input(int64_t n)
    {
        std::vector<T> v(n);
        for(int64_t i = 0; i < n; ++i)
        {
            v[i] = T(static_cast<float>(i % 97) + 0.5f, static_cast<float>(i % 89) + 1.0f);
        }
        return v;
    }

    template <typename T>
    void check_conjugate(rocsparse_handle handle)
    {
        const std::vector<T> h_in = make_input<T>(LENGTH);
        device_vector<T>     d{h_in};
        ASSERT_TRUE(d.ptr);

        (void)hipGetLastError();
        ASSERT_EQ(rocsparse::conjugate(handle, LENGTH, d.ptr), rocsparse_status_success);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<T> h_out = to_host(d);
        for(int64_t i = 0; i < LENGTH; ++i)
        {
            ASSERT_EQ(h_out[i], std::conj(h_in[i])) << "element " << i;
        }
    }

    // Batches are separated by a gap that the kernel must leave untouched.
    template <typename T>
    void check_conjugate_strided_batched(rocsparse_handle handle)
    {
        constexpr int64_t batch_count = 3;
        constexpr int64_t stride      = LENGTH + 11;

        const std::vector<T> h_in = make_input<T>(stride * batch_count);
        device_vector<T>     d{h_in};
        ASSERT_TRUE(d.ptr);

        (void)hipGetLastError();
        ASSERT_EQ(rocsparse::conjugate_strided_batched(
                      handle, batch_count, LENGTH, rocsparse::get_datatype<T>(), d.ptr, stride),
                  rocsparse_status_success);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<T> h_out = to_host(d);
        for(int64_t i = 0; i < stride * batch_count; ++i)
        {
            const T expected = (i % stride < LENGTH) ? std::conj(h_in[i]) : h_in[i];
            ASSERT_EQ(h_out[i], expected) << "element " << i;
        }
    }
}

class ConjugateForcedClamp : public HandleTest, public ::testing::WithParamInterface<int>
{
};

TEST_P(ConjugateForcedClamp, single_array_matches_host)
{
    const ScopedMaxGridSizeX clamp(handle, GetParam());
    check_conjugate<rocsparse_float_complex>(handle);
    check_conjugate<rocsparse_double_complex>(handle);
}

TEST_P(ConjugateForcedClamp, strided_batched_matches_host)
{
    const ScopedMaxGridSizeX clamp(handle, GetParam());
    check_conjugate_strided_batched<rocsparse_float_complex>(handle);
    check_conjugate_strided_batched<rocsparse_double_complex>(handle);
}

INSTANTIATE_TEST_SUITE_P(ConjugateForcedClamp, ConjugateForcedClamp, ::testing::Values(1, 3, 7));

// The unclamped straight-line path.
class ConjugateUnclamped : public HandleTest
{
};

TEST_F(ConjugateUnclamped, matches_host)
{
    check_conjugate<rocsparse_float_complex>(handle);
    check_conjugate_strided_batched<rocsparse_double_complex>(handle);
}

namespace
{
    template <typename T>
    void check_conjugate_real_is_noop(rocsparse_handle handle)
    {
        constexpr int64_t batch_count = 3;
        constexpr int64_t stride      = LENGTH + 11;

        std::vector<T> h_in(stride * batch_count);
        for(int64_t i = 0; i < stride * batch_count; ++i)
        {
            h_in[i] = static_cast<T>(i % 97) - static_cast<T>(48.5);
        }
        device_vector<T> d{h_in};
        ASSERT_TRUE(d.ptr);

        // Under capture nothing executes; a launch would show up as a graph node.
        hipStream_t stream;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        ASSERT_EQ(rocsparse_set_stream(handle, stream), rocsparse_status_success);
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed), hipSuccess);

        const rocsparse_status status_single  = rocsparse::conjugate(handle, LENGTH, d.ptr);
        const rocsparse_status status_batched = rocsparse::conjugate_strided_batched(
            handle, batch_count, LENGTH, rocsparse::get_datatype<T>(), d.ptr, stride);

        hipGraph_t graph = nullptr;
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        size_t num_nodes = 0;
        ASSERT_EQ(hipGraphGetNodes(graph, nullptr, &num_nodes), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
        ASSERT_EQ(rocsparse_set_stream(handle, nullptr), rocsparse_status_success);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

        EXPECT_EQ(status_single, rocsparse_status_success);
        EXPECT_EQ(status_batched, rocsparse_status_success);
        EXPECT_EQ(num_nodes, size_t(0));

        const std::vector<T> h_out = to_host(d);
        for(int64_t i = 0; i < stride * batch_count; ++i)
        {
            ASSERT_EQ(h_out[i], h_in[i]) << "element " << i;
        }
    }
}

class ConjugateReal : public HandleTest
{
};

TEST_F(ConjugateReal, is_noop_without_kernel_launch)
{
    check_conjugate_real_is_noop<float>(handle);
    check_conjugate_real_is_noop<double>(handle);
}

TEST_F(ConjugateReal, unsupported_datatype_is_invalid_value)
{
    int32_t dummy = 0;
    EXPECT_EQ(rocsparse::conjugate(handle, 1, rocsparse_datatype_i32_r, &dummy),
              rocsparse_status_invalid_value);
}
