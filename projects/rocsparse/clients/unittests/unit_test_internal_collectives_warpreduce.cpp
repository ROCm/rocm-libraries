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
// Device (GPU) unit tests for rocSPARSE internal WAVEFRONT reductions
// (rocsparse::wfreduce_sum / _max / _min / wfreduce_partial_sum). Split out of
// unit_test_internal_collectives.cpp by collective family. The warp
// collectives are templated on the wavefront size (32 and 64); both are
// instantiated and rocsparse_ut::launch_warp_by_size dispatches to the one
// matching the device's runtime wavefront on exactly one wavefront, so no
// path is skipped or hard-coded.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp" // wfreduce_*

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_warp_by_size;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    // ---- wavefront reductions (templated on the wavefront size WFSIZE) ------
    template <uint32_t WFSIZE, typename T>
    __global__ void k_wfreduce_sum(const T* in, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::wfreduce_sum<WFSIZE>(in[lane]);
    }
    template <uint32_t WFSIZE, typename T>
    __global__ void k_wfreduce_max(const T* in, T* out)
    {
        const int lane = threadIdx.x;
        T         v    = in[lane];
        rocsparse::wfreduce_max<WFSIZE>(&v);
        out[lane] = v;
    }
    template <uint32_t WFSIZE, typename T>
    __global__ void k_wfreduce_min(const T* in, T* out)
    {
        const int lane = threadIdx.x;
        T         v    = in[lane];
        rocsparse::wfreduce_min<WFSIZE>(&v);
        out[lane] = v;
    }
    template <uint32_t WFSIZE, uint32_t SUB, typename T>
    __global__ void k_wfreduce_partial_sum(const T* in, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::wfreduce_partial_sum<WFSIZE, SUB>(in[lane]);
    }
    // wfreduce_sum<WFSIZE>(x): all-reduce sum across one wavefront; every lane
    // returns the total. Runs on the device's own wavefront width (32 or 64) by
    // dispatching to the matching instantiation; the host reference is the exact
    // sum of the wf lane values produced by `gen`.
    template <typename T, typename Gen>
    void run_wfreduce_sum(Gen gen)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<T> in(wf);
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = gen(l);
        T ref = T(0);
        for(auto v : in)
            ref = ref + v;
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(
            launch_warp_by_size(k_wfreduce_sum<32, T>, k_wfreduce_sum<64, T>, d_in.ptr, d_out.ptr),
            hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], ref);
    }
    // wfreduce_max<WFSIZE>(&v): all-reduce max across one wavefront; every lane
    // ends holding the maximum. Host reference is std::max_element over the wf
    // lane values from `gen`.
    template <typename T, typename Gen>
    void run_wfreduce_max(Gen gen)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<T> in(wf);
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = gen(l);
        T                ref = *std::max_element(in.begin(), in.end());
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(
            launch_warp_by_size(k_wfreduce_max<32, T>, k_wfreduce_max<64, T>, d_in.ptr, d_out.ptr),
            hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], ref);
    }
    // wfreduce_min<WFSIZE>(&v): all-reduce min across one wavefront.
    template <typename T, typename Gen>
    void run_wfreduce_min(Gen gen)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<T> in(wf);
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = gen(l);
        T                ref = *std::min_element(in.begin(), in.end());
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(
            launch_warp_by_size(k_wfreduce_min<32, T>, k_wfreduce_min<64, T>, d_in.ptr, d_out.ptr),
            hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], ref);
    }
    // Distinct, non-monotone exact values across the wavefront so max/min land on
    // interior lanes, not simply lane 0 or the last lane. Backed by a deterministic
    // shuffle of 0..63 (covers both 32- and 64-wide wavefronts) instead of a
    // modulo pattern; lane l reads entry l (l < wavefront size <= 64).
    template <typename T>
    T perm_wf_value(uint32_t l)
    {
        static const std::vector<T> tbl = [] {
            std::vector<T> v(64);
            std::iota(v.begin(), v.end(), T(0));
            std::mt19937 rng(0x1234567u);
            std::shuffle(v.begin(), v.end(), rng);
            return v;
        }();
        return tbl[l];
    }
    // wfreduce_partial_sum<WFSIZE, SUB>(x): xor-butterfly that sums within each
    // SUB-lane sub-group. Host reference mirrors the exact butterfly (halving the
    // stride from wf/2 down to SUB), so every lane is checked, not just one.
    template <uint32_t SUB, typename T, typename Gen>
    void run_wfreduce_partial_sum(Gen gen)
    {
        const uint32_t wf = require_wavefront_size();
        ASSERT_LE(SUB, wf);
        std::vector<T> in(wf);
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = gen(l);
        std::vector<T> cur = in;
        for(int i = static_cast<int>(wf) >> 1; i >= static_cast<int>(SUB); i >>= 1)
        {
            std::vector<T> nxt(wf);
            for(uint32_t l = 0; l < wf; ++l)
                nxt[l] = cur[l] + cur[l ^ static_cast<uint32_t>(i)];
            cur = nxt;
        }
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_warp_by_size(k_wfreduce_partial_sum<32, SUB, T>,
                                      k_wfreduce_partial_sum<64, SUB, T>,
                                      d_in.ptr,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], cur[l]);
    }
} // namespace

// Input: exact small integers per lane. Expected: every lane == sum over the
// wavefront. One case per element type the routine supports.
TEST(internal_collectives_wfreduce_sum, i32)
{
    run_wfreduce_sum<int32_t>([](uint32_t l) { return static_cast<int32_t>((l % 7) + 1); });
}
TEST(internal_collectives_wfreduce_sum, i64)
{
    run_wfreduce_sum<int64_t>([](uint32_t l) { return static_cast<int64_t>(l) + 1; });
}
TEST(internal_collectives_wfreduce_sum, f32)
{
    run_wfreduce_sum<float>([](uint32_t l) { return static_cast<float>((l % 5) + 1); });
}
TEST(internal_collectives_wfreduce_sum, f64)
{
    run_wfreduce_sum<double>([](uint32_t l) { return static_cast<double>((l % 9) + 1); });
}
TEST(internal_collectives_wfreduce_sum, complex_f32)
{
    run_wfreduce_sum<rocsparse_float_complex>([](uint32_t l) {
        return rocsparse_float_complex(static_cast<float>((l % 5) + 1),
                                       static_cast<float>((l % 3) + 1));
    });
}
TEST(internal_collectives_wfreduce_max, f32)
{
    run_wfreduce_max<float>(perm_wf_value<float>);
}
TEST(internal_collectives_wfreduce_max, f64)
{
    run_wfreduce_max<double>(perm_wf_value<double>);
}
TEST(internal_collectives_wfreduce_max, i32)
{
    run_wfreduce_max<int32_t>(perm_wf_value<int32_t>);
}
TEST(internal_collectives_wfreduce_max, i64)
{
    run_wfreduce_max<int64_t>(perm_wf_value<int64_t>);
}
// wfreduce_min only has int32 / int64 overloads (no float/double); cover both.
TEST(internal_collectives_wfreduce_min, i32)
{
    run_wfreduce_min<int32_t>(perm_wf_value<int32_t>);
}
TEST(internal_collectives_wfreduce_min, i64)
{
    run_wfreduce_min<int64_t>(perm_wf_value<int64_t>);
}
TEST(internal_collectives_wfreduce_partial_sum, i32_sub16)
{
    run_wfreduce_partial_sum<16, int32_t>([](uint32_t l) { return static_cast<int32_t>(l) + 1; });
}
TEST(internal_collectives_wfreduce_partial_sum, i32_sub8)
{
    run_wfreduce_partial_sum<8, int32_t>(
        [](uint32_t l) { return static_cast<int32_t>((l % 7) + 1); });
}
TEST(internal_collectives_wfreduce_partial_sum, f32_sub8)
{
    run_wfreduce_partial_sum<8, float>([](uint32_t l) { return static_cast<float>((l % 5) + 1); });
}
// SUB == 32 leaves at most one butterfly step on wave64 and none on wave32.
TEST(internal_collectives_wfreduce_partial_sum, i32_sub32)
{
    run_wfreduce_partial_sum<32, int32_t>(
        [](uint32_t l) { return static_cast<int32_t>(l) * 2 - 5; });
}
