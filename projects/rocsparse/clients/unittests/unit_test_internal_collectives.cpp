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
// Device (GPU) unit tests for rocSPARSE internal block/warp collectives.
//
// Compiled into the rocsparse-unit-test-device binary (links hip::device); must
// run on a GPU. Pattern: write a thin __global__ wrapper around an internal
// ROCSPARSE_DEVICE_ILF collective, launch it on one block/warp via
// rocsparse_ut::launch_single_block / launch_single_warp, read the result back
// with rocsparse_ut::to_host and assert on it.
//
// Wavefront-size policy: the warp collectives are templated on the wavefront
// size. Every wavefront-collective wrapper below is instantiated for BOTH 32 and
// 64. rocsparse_ut::launch_warp_by_size (see unit_test_utils.hpp) launches the
// instantiation that matches the device's runtime wavefront size on exactly one
// wavefront, so the 32-lane path is exercised on wave32 parts (e.g. gfx1201) and
// the 64-lane path on wave64 parts (e.g. gfx94x/gfx950). No wavefront path is
// skipped and none is hard-coded; both are always compiled.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp" // blockreduce_*, wfreduce_*, wfsegmented_reduce, popc, shfl_*, assign_ilu0_boost_value
#include "rocsparse_dichotomic_search.hpp" // rocsparse::dichotomic_search
#include "rocsparse_primitives.hpp" // rocsparse::primitives::double_buffer (host-side)

// segmented_blockreduce lives in the level2 device header. The device unit-test
// target only puts library/src/{include,level1,level3} on the include path, so
// we reach the level2 header with a source-relative include (this TU lives in
// clients/unittests/). This keeps the addition local to this file and avoids a
// shared CMakeLists.txt include-dir change.
#include "../../library/src/level2/coomv_device.h" // rocsparse::segmented_blockreduce

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::device_warp_size;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::launch_single_warp;
using rocsparse_ut::launch_warp_by_size;
using rocsparse_ut::to_host;

namespace
{
    // ---- type-aware "close enough" comparison ------------------------------
    void expect_close(float a, float b)
    {
        EXPECT_FLOAT_EQ(a, b);
    }
    void expect_close(double a, double b)
    {
        EXPECT_DOUBLE_EQ(a, b);
    }
    void expect_close(int32_t a, int32_t b)
    {
        EXPECT_EQ(a, b);
    }
    void expect_close(int64_t a, int64_t b)
    {
        EXPECT_EQ(a, b);
    }
    void expect_close(uint32_t a, uint32_t b)
    {
        EXPECT_EQ(a, b);
    }
    void expect_close(const rocsparse_float_complex& a, const rocsparse_float_complex& b)
    {
        EXPECT_FLOAT_EQ(std::real(a), std::real(b));
        EXPECT_FLOAT_EQ(std::imag(a), std::imag(b));
    }
    void expect_close(const rocsparse_double_complex& a, const rocsparse_double_complex& b)
    {
        EXPECT_DOUBLE_EQ(std::real(a), std::real(b));
        EXPECT_DOUBLE_EQ(std::imag(a), std::imag(b));
    }

    // Returns the active device wavefront size and asserts it is one of the two
    // supported values so a misconfigured device fails loudly instead of
    // silently skipping. Used by every warp-collective test below.
    uint32_t require_wavefront_size()
    {
        const int ws = device_warp_size();
        EXPECT_TRUE(ws == 32 || ws == 64) << "unsupported device wavefront size: " << ws;
        return static_cast<uint32_t>(ws);
    }

    // ========================================================================
    // __global__ wrappers around the internal device collectives
    // ========================================================================

    // ---- block reductions (rocsparse_common.hpp) ---------------------------
    template <uint32_t BS, typename T>
    __global__ void k_blockreduce_sum(const T* in, T* out)
    {
        __shared__ T s[BS];
        const int    tid = threadIdx.x;
        s[tid]           = in[tid];
        __syncthreads();
        rocsparse::blockreduce_sum<BS>(tid, s);
        if(tid == 0)
        {
            out[0] = s[0];
        }
    }
    template <uint32_t BS, typename T>
    __global__ void k_blockreduce_max(const T* in, T* out)
    {
        __shared__ T s[BS];
        const int    tid = threadIdx.x;
        s[tid]           = in[tid];
        __syncthreads();
        rocsparse::blockreduce_max<BS>(tid, s);
        if(tid == 0)
        {
            out[0] = s[0];
        }
    }
    template <uint32_t BS, typename T>
    __global__ void k_blockreduce_min(const T* in, T* out)
    {
        __shared__ T s[BS];
        const int    tid = threadIdx.x;
        s[tid]           = in[tid];
        __syncthreads();
        rocsparse::blockreduce_min<BS>(tid, s);
        if(tid == 0)
        {
            out[0] = s[0];
        }
    }

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

    // ---- segmented wavefront reduce ----------------------------------------
    template <uint32_t WFSIZE, typename R, typename T>
    __global__ void k_wfsegmented_reduce(const R* row, const T* val, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::wfsegmented_reduce<WFSIZE>(row[lane], val[lane]);
    }

    // ---- segmented block reduce (coomv_device.h) ---------------------------
    template <uint32_t BS, typename I, typename T>
    __global__ void k_segmented_blockreduce(const I* rin, const T* vin, T* vout)
    {
        __shared__ I sr[BS];
        __shared__ T sv[BS];
        const int    tid = threadIdx.x;
        sr[tid]          = rin[tid];
        sv[tid]          = vin[tid];
        __syncthreads();
        rocsparse::segmented_blockreduce<BS>(sr, sv);
        vout[tid] = sv[tid];
    }

    // ---- dichotomic search -------------------------------------------------
    template <typename I, typename J>
    __global__ void k_dichotomic(J left, J right, const I* vals, I max_val, const I* arr, J* out)
    {
        const int tid = threadIdx.x;
        out[tid]      = rocsparse::dichotomic_search<I, J>(left, right, vals[tid], max_val, arr);
    }

    // ---- popc (inclusive bit-scan) -----------------------------------------
    template <uint32_t WFSZ>
    __global__ void k_popc(uint64_t mask, uint32_t* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::popc<WFSZ>(mask, static_cast<uint32_t>(lane));
    }

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

    // ---- ilu0 boost --------------------------------------------------------
    template <typename T>
    __global__ void k_assign_boost(T value, T boost, T* out)
    {
        out[0] = rocsparse::assign_ilu0_boost_value<T>(value, boost);
    }

    // ========================================================================
    // host-side helpers computing golden references
    // ========================================================================

    template <uint32_t BS, typename T>
    void run_blockreduce_sum(const std::vector<T>& in)
    {
        ASSERT_EQ(in.size(), size_t{BS});
        T ref = T(0);
        for(auto v : in)
        {
            ref = ref + v;
        }
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_blockreduce_sum<BS, T>, BS, d_in.ptr, d_out.ptr),
                  hipSuccess);
        expect_close(to_host(d_out)[0], ref);
    }
    template <uint32_t BS, typename T>
    void run_blockreduce_max(const std::vector<T>& in)
    {
        ASSERT_EQ(in.size(), size_t{BS});
        T                ref = *std::max_element(in.begin(), in.end());
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_blockreduce_max<BS, T>, BS, d_in.ptr, d_out.ptr),
                  hipSuccess);
        expect_close(to_host(d_out)[0], ref);
    }
    template <uint32_t BS, typename T>
    void run_blockreduce_min(const std::vector<T>& in)
    {
        ASSERT_EQ(in.size(), size_t{BS});
        T                ref = *std::min_element(in.begin(), in.end());
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_blockreduce_min<BS, T>, BS, d_in.ptr, d_out.ptr),
                  hipSuccess);
        expect_close(to_host(d_out)[0], ref);
    }

    // Build a 256-element well-conditioned integer-valued vector: a deterministic
    // shuffle of 0..255 (distinct, all exact in every supported type) rather than
    // a modulo pattern, so max/min land on interior lanes, not lane 0 / the last.
    template <typename T>
    std::vector<T> perm256()
    {
        std::vector<T> v(256);
        std::iota(v.begin(), v.end(), T(0));
        std::mt19937 rng(0x9E3779B9u);
        std::shuffle(v.begin(), v.end(), rng);
        return v;
    }
} // namespace

// ===========================================================================
// block reductions
// ===========================================================================
// blockreduce_sum<BS>(tid, shared): in-place tree sum of BS shared values; lane
// 0 holds the total. Input: exact small integers -> exact float/double sums.
TEST(internal_collectives_blockreduce, sum_float)
{
    std::vector<float> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = static_cast<float>((i % 17) + 1); // small exact integers
    run_blockreduce_sum<256, float>(in);
}
TEST(internal_collectives_blockreduce, sum_double)
{
    std::vector<double> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = static_cast<double>((i % 29) + 1);
    run_blockreduce_sum<256, double>(in);
}
TEST(internal_collectives_blockreduce, sum_int32)
{
    std::vector<int32_t> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = (i % 13) + 1;
    run_blockreduce_sum<256, int32_t>(in);
}
TEST(internal_collectives_blockreduce, sum_int64)
{
    std::vector<int64_t> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = static_cast<int64_t>(i) + 1;
    run_blockreduce_sum<256, int64_t>(in);
}
// blockreduce_max<BS>: lane 0 holds the maximum of BS shared values.
TEST(internal_collectives_blockreduce, max_float)
{
    run_blockreduce_max<256, float>(perm256<float>());
}
TEST(internal_collectives_blockreduce, max_double)
{
    run_blockreduce_max<256, double>(perm256<double>());
}
TEST(internal_collectives_blockreduce, max_int32)
{
    run_blockreduce_max<256, int32_t>(perm256<int32_t>());
}
// blockreduce_min<BS>: lane 0 holds the minimum of BS shared values.
TEST(internal_collectives_blockreduce, min_float)
{
    run_blockreduce_min<256, float>(perm256<float>());
}
TEST(internal_collectives_blockreduce, min_int32)
{
    run_blockreduce_min<256, int32_t>(perm256<int32_t>());
}
TEST(internal_collectives_blockreduce, min_int64)
{
    run_blockreduce_min<256, int64_t>(perm256<int64_t>());
}
// Non-power-of-two block size still reduces correctly (guards the i+stride<BS logic).
TEST(internal_collectives_blockreduce, sum_int32_bs192)
{
    std::vector<int32_t> in(192);
    for(int i = 0; i < 192; ++i)
        in[i] = (i % 11) + 1;
    run_blockreduce_sum<192, int32_t>(in);
}

// ===========================================================================
// wavefront sum (value-returning: every lane holds the full-warp sum)
// ===========================================================================
namespace
{
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

// ===========================================================================
// wavefront max / min (pointer, in-place: every lane holds the extremum)
// ===========================================================================
namespace
{
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
} // namespace

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

// ===========================================================================
// wavefront partial sum (reduces within SUB_WF_SIZE sub-warps via xor-butterfly)
// ===========================================================================
namespace
{
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

// ===========================================================================
// segmented wavefront reduce (in-segment inclusive scan by row key, via shfl_up)
// ===========================================================================
namespace
{
    // wfsegmented_reduce<WFSIZE>(row, val): inclusive scan of val restricted to
    // runs of equal row key. Host reference mirrors the exact shfl_up scan, so
    // every lane's result is validated.
    template <typename R, typename T, typename RowGen, typename ValGen>
    void run_wfsegmented_reduce(RowGen rowgen, ValGen valgen)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<R> row(wf);
        std::vector<T> val(wf);
        for(uint32_t l = 0; l < wf; ++l)
        {
            row[l] = rowgen(l);
            val[l] = valgen(l);
        }
        std::vector<T> v = val;
        for(uint32_t j = 1; j < wf; j <<= 1)
        {
            std::vector<T> nv = v;
            for(uint32_t l = j; l < wf; ++l)
            {
                if(row[l] == row[l - j])
                    nv[l] = v[l] + v[l - j];
            }
            v = nv;
        }
        device_vector<R> d_row(row);
        device_vector<T> d_val(val), d_out(size_t{wf});
        ASSERT_NE(d_row.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_warp_by_size(k_wfsegmented_reduce<32, R, T>,
                                      k_wfsegmented_reduce<64, R, T>,
                                      d_row.ptr,
                                      d_val.ptr,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], v[l]);
    }

    // Contiguous runs of equal row keys of varying lengths; valid for wf 32 or 64
    // (the final run simply extends on wider wavefronts).
    template <typename R>
    R seg_row_key(uint32_t l)
    {
        if(l < 5)
            return static_cast<R>(0);
        if(l < 6)
            return static_cast<R>(1);
        if(l < 14)
            return static_cast<R>(2);
        if(l < 20)
            return static_cast<R>(3);
        return static_cast<R>(4);
    }
} // namespace

TEST(internal_collectives_wfsegmented_reduce, row_i32_val_i32)
{
    run_wfsegmented_reduce<int32_t, int32_t>(
        seg_row_key<int32_t>, [](uint32_t l) { return static_cast<int32_t>((l % 4) + 1); });
}
TEST(internal_collectives_wfsegmented_reduce, row_i32_val_f32)
{
    run_wfsegmented_reduce<int32_t, float>(
        seg_row_key<int32_t>, [](uint32_t l) { return static_cast<float>((l % 4) + 1); });
}
TEST(internal_collectives_wfsegmented_reduce, row_i64_val_f64)
{
    run_wfsegmented_reduce<int64_t, double>(
        seg_row_key<int64_t>, [](uint32_t l) { return static_cast<double>((l % 4) + 1); });
}

// ===========================================================================
// segmented block reduce (coomv_device.h)
// ===========================================================================
namespace
{
    // segmented_blockreduce<BS>(row, val): block-wide Hillis-Steele inclusive
    // scan restricted to equal-row-key runs. Host reference mirrors the exact
    // scan; every one of the BS outputs is checked.
    template <uint32_t BS, typename I, typename T>
    void run_segmented_blockreduce(const std::vector<I>& row, const std::vector<T>& val)
    {
        ASSERT_EQ(row.size(), size_t{BS});
        ASSERT_EQ(val.size(), size_t{BS});
        std::vector<T> v = val;
        for(uint32_t j = 1; j < BS; j <<= 1)
        {
            std::vector<T> add(BS, T(0));
            for(uint32_t t = j; t < BS; ++t)
            {
                if(row[t] == row[t - j])
                    add[t] = v[t - j];
            }
            for(uint32_t t = 0; t < BS; ++t)
                v[t] = v[t] + add[t];
        }
        device_vector<I> d_row(row);
        device_vector<T> d_val(val), d_out(size_t{BS});
        ASSERT_NE(d_row.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(
                      k_segmented_blockreduce<BS, I, T>, BS, d_row.ptr, d_val.ptr, d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t t = 0; t < BS; ++t)
            expect_close(h[t], v[t]);
    }

    template <typename I>
    std::vector<I> seg_rows64()
    {
        std::vector<I> r(64);
        for(uint32_t t = 0; t < 64; ++t)
            r[t] = static_cast<I>(t / 5); // runs of length 5 (last run shorter)
        return r;
    }
} // namespace

TEST(internal_collectives_segmented_blockreduce, row_i32_val_f32)
{
    std::vector<float> val(64);
    for(uint32_t t = 0; t < 64; ++t)
        val[t] = static_cast<float>((t % 3) + 1);
    run_segmented_blockreduce<64, int32_t, float>(seg_rows64<int32_t>(), val);
}
TEST(internal_collectives_segmented_blockreduce, row_i64_val_f64)
{
    std::vector<double> val(64);
    for(uint32_t t = 0; t < 64; ++t)
        val[t] = static_cast<double>((t % 3) + 1);
    run_segmented_blockreduce<64, int64_t, double>(seg_rows64<int64_t>(), val);
}

// ===========================================================================
// dichotomic search
// ===========================================================================
namespace
{
    template <typename I, typename J>
    J host_dichotomic(J left, J right, I val, I max_val, const std::vector<I>& arr)
    {
        if(val < max_val)
        {
            while(left < right)
            {
                const J mid = (left + right) / 2;
                if(arr[mid + 1] <= val)
                    left = mid + 1;
                else
                    right = mid;
            }
            return left;
        }
        return static_cast<J>(0);
    }

    // dichotomic_search<I,J>: for each query value, the segment index in a sorted
    // "row offset" array. Host reference computes the same for every query.
    template <typename I, typename J>
    void run_dichotomic()
    {
        // "Row offset" style array: sorted, includes a zero-length segment (7==7).
        const std::vector<I> arr{0, 3, 3, 7, 12, 20};
        const J              n       = 5; // number of segments (arr has n+1 entries)
        const I              max_val = arr[n];
        std::vector<I>       vals;
        for(I v = 0; v <= max_val + 1; ++v)
            vals.push_back(v);
        const size_t nq = vals.size();

        std::vector<J> ref(nq);
        for(size_t q = 0; q < nq; ++q)
            ref[q] = host_dichotomic<I, J>(J(0), n, vals[q], max_val, arr);

        device_vector<I> d_arr(arr), d_vals(vals);
        device_vector<J> d_out(nq);
        ASSERT_NE(d_arr.ptr, nullptr);
        ASSERT_NE(d_vals.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_dichotomic<I, J>,
                                      static_cast<unsigned int>(nq),
                                      J(0),
                                      n,
                                      d_vals.ptr,
                                      max_val,
                                      d_arr.ptr,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(size_t q = 0; q < nq; ++q)
            expect_close(h[q], ref[q]);
    }
} // namespace

TEST(internal_collectives_dichotomic_search, i32_i32)
{
    run_dichotomic<int32_t, int32_t>();
}
TEST(internal_collectives_dichotomic_search, i64_i64)
{
    run_dichotomic<int64_t, int64_t>();
}

// ===========================================================================
// popc (inclusive population-count scan over the low (lid+1) bits)
// ===========================================================================
namespace
{
    // popc<WFSIZE>(mask, lid): number of set bits of `mask` in bit positions
    // [0, lid]. Host reference uses __builtin_popcountll over the same low-bit
    // window; validated for every lane of the device's wavefront.
    void run_popc()
    {
        const uint32_t wf   = require_wavefront_size();
        const uint64_t mask = 0xB6D1E4A5F0C3927Bull; // 64-bit pattern (low 32 used on wf32)
        device_vector<uint32_t> d_out(size_t{wf});
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_warp_by_size(k_popc<32>, k_popc<64>, mask, d_out.ptr), hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t lid = 0; lid < wf; ++lid)
        {
            const uint64_t lowmask  = (lid >= 63) ? ~0ull : ((1ull << (lid + 1)) - 1);
            const uint32_t expected = static_cast<uint32_t>(__builtin_popcountll(mask & lowmask));
            expect_close(h[lid], expected);
        }
    }
} // namespace

TEST(internal_collectives_popc, inclusive_scan)
{
    run_popc();
}

// ===========================================================================
// shuffles
// ===========================================================================
namespace
{
    template <typename T>
    T lane_value(uint32_t l)
    {
        return static_cast<T>(l * 3 + 1);
    }

    // shfl(x, src): every lane reads lane `src`'s value (broadcast). Validated
    // for all lanes of the device's wavefront.
    template <typename T>
    void run_shfl_broadcast(int src)
    {
        const uint32_t wf = require_wavefront_size();
        ASSERT_LT(static_cast<uint32_t>(src), wf);
        std::vector<T> in(wf);
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = lane_value<T>(l);
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
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = lane_value<T>(l);
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
        for(uint32_t l = 0; l < wf; ++l)
            in[l] = lane_value<T>(l);
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

// ===========================================================================
// assign_ilu0_boost_value
// ===========================================================================
//
// PR #10290 (merged to upstream/develop) makes ROCSPARSE_ENABLE_ILU0_BOOST_SIGN
// default ON, so ROCSPARSE_WITH_ILU0_BOOST_SIGN is defined in this build and the
// sign/inertia-preserving path is the default. In that path the routine returns
// the boost *magnitude* carried along the sign/phase of the original pivot:
//
//   real:    (|value| > 0) ? copysign(|boost|, value) : |boost|
//   complex: (|value| > 0) ? |boost| * value / |value| : |boost|
//
// i.e. a negative boost can never flip the pivot's sign (inertia is preserved),
// and a zero pivot receives the pure boost magnitude. The sign-aware result is
// therefore the DEFAULT expectation below; a verbatim-boost expectation is only
// retained under an explicit #ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN fallback for
// a hypothetical no-sign build.
namespace
{
    template <typename T>
    void run_assign_boost(T value, T boost, T expected)
    {
        device_vector<T> d_out(size_t{1});
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_assign_boost<T>, 1u, value, boost, d_out.ptr), hipSuccess);
        expect_close(to_host(d_out)[0], expected);
    }
} // namespace

// --- Primary, same-direction cases (identical under both configurations) -----
TEST(internal_collectives_assign_ilu0_boost, f64)
{
    run_assign_boost<double>(2.0, 0.25, 0.25);
}
TEST(internal_collectives_assign_ilu0_boost, f32)
{
    run_assign_boost<float>(2.0f, 0.25f, 0.25f);
}
TEST(internal_collectives_assign_ilu0_boost, complex_f32)
{
    run_assign_boost<rocsparse_float_complex>(rocsparse_float_complex(2.0f, 0.0f),
                                              rocsparse_float_complex(0.25f, 0.0f),
                                              rocsparse_float_complex(0.25f, 0.0f));
}
TEST(internal_collectives_assign_ilu0_boost, complex_f64)
{
    run_assign_boost<rocsparse_double_complex>(rocsparse_double_complex(2.0, 0.0),
                                               rocsparse_double_complex(0.25, 0.0),
                                               rocsparse_double_complex(0.25, 0.0));
}

// --- Opposite-direction cases: sign-aware follows the pivot sign -------------
// value=-4.0, boost=0.25 -> copysign(|0.25|, -4.0) == -0.25 (sign-aware default).
TEST(internal_collectives_assign_ilu0_boost, f64_negative_value)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<double>(-4.0, 0.25, 0.25); // no-sign fallback: verbatim boost
#else
    run_assign_boost<double>(-4.0, 0.25, -0.25);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, f32_negative_value)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<float>(-4.0f, 0.25f, 0.25f);
#else
    run_assign_boost<float>(-4.0f, 0.25f, -0.25f);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, complex_f32_negative_value)
{
    const rocsparse_float_complex value(-4.0f, 0.0f);
    const rocsparse_float_complex boost(0.25f, 0.0f);
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<rocsparse_float_complex>(value, boost, rocsparse_float_complex(0.25f, 0.0f));
#else
    // |0.25| * (-4,0)/|(-4,0)| == (-0.25, 0)
    run_assign_boost<rocsparse_float_complex>(value, boost, rocsparse_float_complex(-0.25f, 0.0f));
#endif
}
TEST(internal_collectives_assign_ilu0_boost, complex_f64_negative_value)
{
    const rocsparse_double_complex value(-4.0, 0.0);
    const rocsparse_double_complex boost(0.25, 0.0);
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(0.25, 0.0));
#else
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(-0.25, 0.0));
#endif
}

// --- Negative boost: magnitude used, so the pivot sign is never flipped ------
// value=4.0, boost=-0.25 -> copysign(|-0.25|, 4.0) == +0.25 (sign-aware default).
TEST(internal_collectives_assign_ilu0_boost, f64_negative_boost_uses_magnitude)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<double>(4.0, -0.25, -0.25); // no-sign fallback: verbatim boost
#else
    run_assign_boost<double>(4.0, -0.25, 0.25);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, f32_negative_boost_uses_magnitude)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<float>(4.0f, -0.25f, -0.25f);
#else
    run_assign_boost<float>(4.0f, -0.25f, 0.25f);
#endif
}

// --- Zero pivot: |value| == 0 returns the pure boost magnitude |boost| -------
TEST(internal_collectives_assign_ilu0_boost, f64_zero_value_returns_abs_boost)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<double>(0.0, -0.25, -0.25); // no-sign fallback: verbatim boost
#else
    run_assign_boost<double>(0.0, -0.25, 0.25);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, f32_zero_value_returns_abs_boost)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<float>(0.0f, -0.25f, -0.25f);
#else
    run_assign_boost<float>(0.0f, -0.25f, 0.25f);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, complex_f64_zero_value_returns_abs_boost)
{
    // |value| == 0 -> returns |boost| as a pure-real complex.
    const rocsparse_double_complex value(0.0, 0.0);
    const rocsparse_double_complex boost(0.0, -0.25); // |boost| == 0.25
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(0.0, -0.25));
#else
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(0.25, 0.0));
#endif
}

// ===========================================================================
// host-side primitives::double_buffer
// ===========================================================================
// double_buffer<T>: current()/alternate() selectors with swap(); a pure host
// utility (no device launch).
TEST(internal_collectives_double_buffer, select_and_swap)
{
    int                                       a = 0, b = 0;
    rocsparse::primitives::double_buffer<int> db(&a, &b);
    EXPECT_EQ(db.current(), &a);
    EXPECT_EQ(db.alternate(), &b);
    db.swap();
    EXPECT_EQ(db.current(), &b);
    EXPECT_EQ(db.alternate(), &a);
    db.swap();
    EXPECT_EQ(db.current(), &a);
    EXPECT_EQ(db.alternate(), &b);
}
TEST(internal_collectives_double_buffer, default_ctor_is_null)
{
    rocsparse::primitives::double_buffer<double> db;
    EXPECT_EQ(db.current(), nullptr);
    EXPECT_EQ(db.alternate(), nullptr);
}
TEST(internal_collectives_double_buffer, void_ctor)
{
    int                                       a = 0, b = 0;
    rocsparse::primitives::double_buffer<int> db(static_cast<void*>(&a), static_cast<void*>(&b));
    EXPECT_EQ(db.current(), &a);
    EXPECT_EQ(db.alternate(), &b);
}
