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
// Device (GPU) unit tests for the residual rocSPARSE internal WAVEFRONT
// reductions not covered by the collectives TU: rocsparse::wfreduce_sum_mask
// and the f64 overload of rocsparse::wfreduce_partial_sum. Split out of
// unit_test_internal_collective_extras.cpp by family. Tests run on the
// device's runtime wavefront width; wavefront-size-templated blocks are
// instantiated for both 32 and 64 and dispatched via launch_warp_by_size.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collective_extras_common.hpp"

#include "rocsparse_common.hpp" // wfreduce_sum_mask, wfreduce_partial_sum

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_warp;
using rocsparse_ut::launch_warp_by_size;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collective_extras;

namespace
{
    // wfreduce_sum_mask<T>(x, mask): every active lane returns the sum of x over
    // the lanes whose bit is set in `mask`. Not templated on the wavefront size;
    // it acts on the active wavefront, so the wrapper simply launches one
    // wavefront of the device's width. `mask` bits beyond the wavefront width are
    // ignored by both the routine (no such lanes) and the host reference.
    template <typename T>
    __global__ void k_wfreduce_sum_mask(const T* in, unsigned long long int mask, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::wfreduce_sum_mask<T>(in[lane], mask);
    }

    // Runs on the device's runtime wavefront width. `gen(l)` fills lane l; the
    // host reference sums the generated values over the lanes selected by `mask`.
    template <typename T, typename Gen>
    void run_wfreduce_sum_mask(Gen gen, unsigned long long int mask)
    {
        const uint32_t wf = require_wavefront_size();
        // Restrict the active-lane mask to lanes that actually exist at this
        // wavefront width: bits for non-existent lanes would make the routine
        // shuffle from out-of-range lanes. This lets callers pass a "logical"
        // pattern (e.g. ~0 for all lanes, 0x5555... for even lanes) that adapts
        // to both 32- and 64-wide wavefronts.
        const unsigned long long int lane_mask = (wf >= 64) ? mask : (mask & 0xFFFFFFFFULL);
        std::vector<T>               in(wf);
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = gen(l);
        T ref = T(0);
        for(uint32_t l = 0; l < wf; ++l)
        {
            if(lane_mask & (1ULL << l))
                ref += in[l];
        }
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_wfreduce_sum_mask<T>, d_in.ptr, lane_mask, d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            EXPECT_EQ(h[l], ref);
    }
    // wfreduce_partial_sum<WFSIZE, SUB>(x): xor-butterfly summing within each
    // SUB-lane sub-group. Templated on the wavefront size, so BOTH the 32- and
    // 64-lane instantiations are referenced and dispatched at runtime.
    template <uint32_t WFSIZE, uint32_t SUB, typename T>
    __global__ void k_wfreduce_partial_sum(const T* in, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::wfreduce_partial_sum<WFSIZE, SUB>(in[lane]);
    }

    // Host reference mirrors the exact xor-butterfly (stride wf/2 down to SUB) at
    // the device's runtime wavefront width; every lane's result is checked.
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
            EXPECT_DOUBLE_EQ(h[l], cur[l]);
    }
} // namespace

// All lanes active (~0 selects every lane at either wavefront width). Expected:
// every lane holds the full-wavefront sum.
TEST(internal_collective_extras_wfreduce_sum_mask, all_lanes_i32)
{
    run_wfreduce_sum_mask<int32_t>([](uint32_t l) { return static_cast<int32_t>(l) + 1; },
                                   0xFFFFFFFFFFFFFFFFULL);
}
// Even lanes active (0x5555... at either width). Expected: sum over even lanes.
TEST(internal_collective_extras_wfreduce_sum_mask, even_lanes_i32)
{
    run_wfreduce_sum_mask<int32_t>([](uint32_t l) { return static_cast<int32_t>((l % 5) + 1); },
                                   0x5555555555555555ULL);
}
// A sparse set of active lanes (all < 32, so valid at both widths). Expected:
// sum over exactly lanes 3, 7, 11, 19, 31.
TEST(internal_collective_extras_wfreduce_sum_mask, sparse_lanes_i64)
{
    const unsigned long long int mask
        = (1ULL << 3) | (1ULL << 7) | (1ULL << 11) | (1ULL << 19) | (1ULL << 31);
    run_wfreduce_sum_mask<int64_t>([](uint32_t l) { return static_cast<int64_t>(l) * 3 + 1; },
                                   mask);
}
// f64 overload of the partial sum (i32/f32 covered in the collectives TU).
// Expected: each lane holds the sum over its SUB-lane sub-group.
TEST(internal_collective_extras_wfreduce_partial_sum, f64_sub16)
{
    run_wfreduce_partial_sum<16, double>(
        [](uint32_t l) { return static_cast<double>((l % 5) + 1); });
}
TEST(internal_collective_extras_wfreduce_partial_sum, f64_sub8)
{
    run_wfreduce_partial_sum<8, double>(
        [](uint32_t l) { return static_cast<double>((l % 7) + 1); });
}
// SUB == 32: on wave32 no butterfly step runs (identity); on wave64 one step runs.
TEST(internal_collective_extras_wfreduce_partial_sum, f64_sub32)
{
    run_wfreduce_partial_sum<32, double>([](uint32_t l) { return static_cast<double>(l) - 7.0; });
}
