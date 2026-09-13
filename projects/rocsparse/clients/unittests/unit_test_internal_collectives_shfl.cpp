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
// Device (GPU) unit tests for rocSPARSE internal warp SHUFFLES
// (rocsparse::shfl / shfl_up / shfl_down). Split out of
// unit_test_internal_collectives.cpp by collective family. Exercised on one
// wavefront of the device's runtime width via rocsparse_ut::launch_single_warp.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp" // shfl, shfl_up, shfl_down

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <numeric>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_warp;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    // ---- shuffles ----------------------------------------------------------
    template <typename T>
    __global__ void k_shfl(const T* in, T* out, int src)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::shfl(in[lane], src);
    }
    template <typename T>
    __global__ void k_shfl_up(const T* in, T* out, int delta)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::shfl_up(in[lane], delta);
    }
    template <typename T>
    __global__ void k_shfl_down(const T* in, T* out, int delta)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::shfl_down(in[lane], delta);
    }
    // shfl(x, src): every lane reads lane `src`'s value (broadcast). Validated
    // for all lanes of the device's wavefront.
    template <typename T>
    void run_shfl_broadcast(int src)
    {
        const uint32_t wf = require_wavefront_size();
        ASSERT_LT(static_cast<uint32_t>(src), wf);
        std::vector<T> in(wf);
        std::iota(in.begin(), in.end(), T(0)); // lane l holds value l
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_shfl<T>, d_in.ptr, d_out.ptr, src), hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], in[src]);
    }
    // shfl_up(x, delta): lane l reads lane l-delta; out-of-range lanes keep own.
    template <typename T>
    void run_shfl_up(int delta)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<T> in(wf);
        std::iota(in.begin(), in.end(), T(0)); // lane l holds value l
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_shfl_up<T>, d_in.ptr, d_out.ptr, delta), hipSuccess);
        auto h = to_host(d_out);
        for(int l = 0; l < static_cast<int>(wf); ++l)
        {
            const T expected = (l - delta >= 0) ? in[l - delta] : in[l]; // out-of-range -> own
            expect_close(h[l], expected);
        }
    }
    // shfl_down(x, delta): lane l reads lane l+delta; out-of-range lanes keep own.
    template <typename T>
    void run_shfl_down(int delta)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<T> in(wf);
        std::iota(in.begin(), in.end(), T(0)); // lane l holds value l
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_shfl_down<T>, d_in.ptr, d_out.ptr, delta), hipSuccess);
        auto h = to_host(d_out);
        for(int l = 0; l < static_cast<int>(wf); ++l)
        {
            const T expected = (l + delta < static_cast<int>(wf)) ? in[l + delta] : in[l];
            expect_close(h[l], expected);
        }
    }
} // namespace

TEST(internal_collectives_shfl, broadcast_i32)
{
    run_shfl_broadcast<int32_t>(5);
}
TEST(internal_collectives_shfl, broadcast_i64)
{
    run_shfl_broadcast<int64_t>(11);
}
TEST(internal_collectives_shfl, broadcast_f32)
{
    run_shfl_broadcast<float>(7);
}
TEST(internal_collectives_shfl, broadcast_f64)
{
    run_shfl_broadcast<double>(23);
}
TEST(internal_collectives_shfl_up, i32)
{
    run_shfl_up<int32_t>(2);
}
TEST(internal_collectives_shfl_up, f32)
{
    run_shfl_up<float>(1);
}
TEST(internal_collectives_shfl_up, f64)
{
    run_shfl_up<double>(4);
}
TEST(internal_collectives_shfl_down, i32)
{
    run_shfl_down<int32_t>(3);
}
TEST(internal_collectives_shfl_down, i64)
{
    run_shfl_down<int64_t>(1);
}
TEST(internal_collectives_shfl_down, f32)
{
    run_shfl_down<float>(5);
}
