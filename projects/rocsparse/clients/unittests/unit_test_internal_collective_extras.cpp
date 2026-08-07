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
// Device (GPU) unit tests for the RESIDUAL in-scope rocSPARSE internal device
// building blocks not already covered by unit_test_internal_collectives.cpp.
//
// Compiled into the rocsparse-unit-test-device binary (links hip::device); must
// run on a GPU. Same proven mechanism (#1) as unit_test_internal_collectives.cpp:
// a thin __global__ wrapper calls the internal ROCSPARSE_DEVICE_ILF routine and
// stores its result into a device buffer, the wrapper is launched on ONE
// block/warp via rocsparse_ut::launch_single_block / launch_single_warp, and the
// result is read back with rocsparse_ut::to_host and asserted on.
//
// Wavefront-size policy: tests run on the device's own wavefront width (32 or
// 64) obtained from rocsparse_ut::device_warp_size(); nothing is hard-coded to
// 32 and no wavefront is skipped. Wavefront-size-templated building blocks (e.g.
// wfreduce_partial_sum<WFSIZE, SUB>) are instantiated for BOTH 32 and 64 and
// dispatched at runtime via rocsparse_ut::launch_warp_by_size, so the wave64
// path is compiled and validated by CI's wave64 parts (gfx94x/gfx950). Routines
// that already act on the active wavefront (wfreduce_sum_mask, atomic_add_by_CAS)
// simply launch one wavefront of the device's width.
//
// Routines covered here (all header-inline / device):
//   * wfreduce_sum_mask                     (rocsparse_common.hpp)
//   * wfreduce_partial_sum  f64 overload    (rocsparse_common.hpp)
//   * atomic_add / atomic_min / atomic_max / atomic_cas / atomic_load /
//     atomic_store / atomic_add_check / atomic_add_by_CAS (rocsparse_common.hpp)
//   * elementwise min / max int/uint/float/double overloads (rocsparse_common.hpp)
//   * nontemporal_load / nontemporal_store family (rocsparse_common.hpp)
//   * coo2csr lower_bound<I,J>              (conversion/coo2csr_device.h)
//   * csrgemm insert_key / insert_pair      (extra/csrgemm_device.h)
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp"

// coo2csr lower_bound and csrgemm hash helpers live in the conversion/extra
// device headers. The device unit-test target only puts library/src/{include,
// level1,level3} on the include path, so we reach these headers with a
// source-relative include (this TU lives in clients/unittests/). This keeps the
// addition local to this file and avoids a shared CMakeLists.txt include-dir
// change (same pattern as coomv_device.h in unit_test_internal_collectives.cpp).
#include "../../library/src/conversion/coo2csr_device.h" // rocsparse::lower_bound
#include "../../library/src/extra/csrgemm_device.h" // rocsparse::insert_key / insert_pair

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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
    // Active device wavefront size (32 or 64); asserted so a misconfigured device
    // fails loudly instead of silently skipping. Warp tests below size their data
    // and reference to this runtime value.
    uint32_t require_wavefront_size()
    {
        const int ws = device_warp_size();
        EXPECT_TRUE(ws == 32 || ws == 64) << "unsupported device wavefront size: " << ws;
        return static_cast<uint32_t>(ws);
    }
}

// ===========================================================================
// wfreduce_sum_mask : every active lane returns the sum of `sum` over the lanes
// whose bit is set in active_mask (broadcast from the first active lane).
// ===========================================================================
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

// ===========================================================================
// wfreduce_partial_sum : f64 overload (i32/f32/i32-sub already covered).
// Reduces within SUB_WF_SIZE sub-warps via xor-butterfly.
// ===========================================================================
namespace
{
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

// ===========================================================================
// atomic_add : all threads of a block add their value into a single cell. The
// order of the atomic updates is nondeterministic, but the final accumulated
// total is deterministic (integer / small-exact-integer operands), so the test
// is race-free by construction. (Named "atomic_add", not "..._race": no data
// race is expected -- the atomic serializes the concurrent updates.)
// ===========================================================================
namespace
{
    template <typename T>
    __global__ void k_atomic_add(const T* vals, T* accumulator)
    {
        rocsparse::atomic_add(accumulator, vals[threadIdx.x]);
    }

    template <typename T>
    void run_atomic_add(const std::vector<T>& vals)
    {
        const unsigned int n   = static_cast<unsigned int>(vals.size());
        T                  ref = T(0);
        for(auto v : vals)
            ref += v;
        std::vector<T>   zero(1, T(0));
        device_vector<T> d_vals(vals), d_accumulator(zero);
        ASSERT_NE(d_vals.ptr, nullptr);
        ASSERT_NE(d_accumulator.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_add<T>, n, d_vals.ptr, d_accumulator.ptr),
                  hipSuccess);
        EXPECT_EQ(to_host(d_accumulator)[0], ref);
    }
} // namespace

TEST(internal_collective_extras_atomic_add, i32)
{
    std::vector<int32_t> vals(256);
    for(int i = 0; i < 256; ++i)
        vals[i] = (i % 7) + 1;
    run_atomic_add<int32_t>(vals);
}
TEST(internal_collective_extras_atomic_add, u32)
{
    std::vector<uint32_t> vals(256);
    for(int i = 0; i < 256; ++i)
        vals[i] = static_cast<uint32_t>((i % 11) + 1);
    run_atomic_add<uint32_t>(vals);
}
TEST(internal_collective_extras_atomic_add, i64)
{
    std::vector<int64_t> vals(256);
    for(int i = 0; i < 256; ++i)
        vals[i] = static_cast<int64_t>((i % 13) + 1);
    run_atomic_add<int64_t>(vals);
}
TEST(internal_collective_extras_atomic_add, f32)
{
    // Small exact integers keep the accumulated float sum exact regardless of the
    // (nondeterministic) update order.
    std::vector<float> vals(256);
    for(int i = 0; i < 256; ++i)
        vals[i] = static_cast<float>((i % 4) + 1);
    run_atomic_add<float>(vals);
}
TEST(internal_collective_extras_atomic_add, f64)
{
    std::vector<double> vals(256);
    for(int i = 0; i < 256; ++i)
        vals[i] = static_cast<double>((i % 4) + 1);
    run_atomic_add<double>(vals);
}

// ===========================================================================
// atomic_min / atomic_max : all threads of a block reduce into a single cell via
// atomics. The update order is nondeterministic but the final extremum is
// deterministic, so the test is race-free (no "..._race" naming).
// ===========================================================================
namespace
{
    template <typename T>
    __global__ void k_atomic_min(const T* vals, T* accumulator)
    {
        rocsparse::atomic_min(accumulator, vals[threadIdx.x]);
    }
    template <typename T>
    __global__ void k_atomic_max(const T* vals, T* accumulator)
    {
        rocsparse::atomic_max(accumulator, vals[threadIdx.x]);
    }

    template <typename T>
    void run_atomic_min(const std::vector<T>& vals, T init)
    {
        const unsigned int n   = static_cast<unsigned int>(vals.size());
        T                  ref = init;
        for(auto v : vals)
            ref = std::min(ref, v);
        std::vector<T>   init_cell(1, init);
        device_vector<T> d_vals(vals), d_accumulator(init_cell);
        ASSERT_NE(d_vals.ptr, nullptr);
        ASSERT_NE(d_accumulator.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_min<T>, n, d_vals.ptr, d_accumulator.ptr),
                  hipSuccess);
        EXPECT_EQ(to_host(d_accumulator)[0], ref);
    }
    template <typename T>
    void run_atomic_max(const std::vector<T>& vals, T init)
    {
        const unsigned int n   = static_cast<unsigned int>(vals.size());
        T                  ref = init;
        for(auto v : vals)
            ref = std::max(ref, v);
        std::vector<T>   init_cell(1, init);
        device_vector<T> d_vals(vals), d_accumulator(init_cell);
        ASSERT_NE(d_vals.ptr, nullptr);
        ASSERT_NE(d_accumulator.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_max<T>, n, d_vals.ptr, d_accumulator.ptr),
                  hipSuccess);
        EXPECT_EQ(to_host(d_accumulator)[0], ref);
    }

    // n genuinely-distinct values (a deterministic shuffle of 0..n-1), so the
    // min/max see a non-sorted permutation rather than a mod-collision pattern.
    // Values are non-negative, matching how the library uses these (unsigned)
    // atomics on indices.
    template <typename T>
    std::vector<T> distinct_vals(int n)
    {
        std::vector<T> v(n);
        std::iota(v.begin(), v.end(), T(0));
        std::mt19937 rng(0x9E3779B9u);
        std::shuffle(v.begin(), v.end(), rng);
        return v;
    }
} // namespace

TEST(internal_collective_extras_atomic_min, i32)
{
    run_atomic_min<int32_t>(distinct_vals<int32_t>(256), 1000000);
}
TEST(internal_collective_extras_atomic_min, u32)
{
    run_atomic_min<uint32_t>(distinct_vals<uint32_t>(256), 1000000u);
}
TEST(internal_collective_extras_atomic_min, i64)
{
    run_atomic_min<int64_t>(distinct_vals<int64_t>(256), 1000000);
}
TEST(internal_collective_extras_atomic_max, i32)
{
    run_atomic_max<int32_t>(distinct_vals<int32_t>(256), -1);
}
TEST(internal_collective_extras_atomic_max, u32)
{
    run_atomic_max<uint32_t>(distinct_vals<uint32_t>(256), 0u);
}
TEST(internal_collective_extras_atomic_max, i64)
{
    // NOTE: atomic_max<int64_t> casts the address to uint64_t* and does an
    // UNSIGNED atomicMax (likewise atomic_min<int64_t>). It therefore only
    // matches the signed maximum when every operand is non-negative, which is
    // how the library uses it (on non-negative indices). We keep this test
    // well-conditioned with a non-negative init and non-negative values so the
    // unsigned and signed maxima coincide.
    run_atomic_max<int64_t>(distinct_vals<int64_t>(256), 0);
}

// ===========================================================================
// atomic_cas : deterministic single-thread exercise of both branches
// (swap-on-match and no-swap-on-mismatch), returning the prior value.
// ===========================================================================
namespace
{
    template <typename T>
    __global__ void k_atomic_cas(T* cell, T* olds)
    {
        // Start value is 5.
        olds[0] = rocsparse::atomic_cas(cell, static_cast<T>(5), static_cast<T>(9)); // match -> 9
        olds[1] = rocsparse::atomic_cas(
            cell, static_cast<T>(5), static_cast<T>(7)); // mismatch -> no-op
    }

    template <typename T>
    void run_atomic_cas()
    {
        std::vector<T>   init(1, static_cast<T>(5));
        device_vector<T> d_cell(init), d_olds(size_t{2});
        ASSERT_NE(d_cell.ptr, nullptr);
        ASSERT_NE(d_olds.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_cas<T>, 1u, d_cell.ptr, d_olds.ptr), hipSuccess);
        auto olds = to_host(d_olds);
        EXPECT_EQ(olds[0], static_cast<T>(5)); // returned prior value on match
        EXPECT_EQ(olds[1], static_cast<T>(9)); // returned current value on mismatch
        EXPECT_EQ(to_host(d_cell)[0], static_cast<T>(9)); // swapped once, second was a no-op
    }
} // namespace

TEST(internal_collective_extras_atomic_cas, i32)
{
    run_atomic_cas<int32_t>();
}
TEST(internal_collective_extras_atomic_cas, u32)
{
    run_atomic_cas<uint32_t>();
}
TEST(internal_collective_extras_atomic_cas, i64)
{
    run_atomic_cas<int64_t>();
}

// ===========================================================================
// atomic_load / atomic_store : relaxed round-trip.
// ===========================================================================
namespace
{
    template <typename T>
    __global__ void k_atomic_load_store(const T* in, T* out)
    {
        T tmp;
        rocsparse::atomic_store(&tmp, in[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        out[0] = rocsparse::atomic_load(&tmp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    template <typename T>
    void run_atomic_load_store(T value)
    {
        std::vector<T>   in(1, value);
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_load_store<T>, 1u, d_in.ptr, d_out.ptr), hipSuccess);
        EXPECT_EQ(to_host(d_out)[0], value);
    }
} // namespace

TEST(internal_collective_extras_atomic_load_store, i32)
{
    run_atomic_load_store<int32_t>(-12345);
}
TEST(internal_collective_extras_atomic_load_store, i64)
{
    run_atomic_load_store<int64_t>(9876543210LL);
}
TEST(internal_collective_extras_atomic_load_store, f32)
{
    run_atomic_load_store<float>(3.5f);
}
TEST(internal_collective_extras_atomic_load_store, f64)
{
    run_atomic_load_store<double>(-2.25);
}

// ===========================================================================
// atomic_add_check : skips the add when val == 0, else behaves like atomic_add
// and returns the prior value. Also exercises the (base_ptr, idx, size, val)
// index overload.
// ===========================================================================
namespace
{
    template <typename T>
    __global__ void k_atomic_add_check(T* cell, T* olds)
    {
        olds[0] = rocsparse::atomic_add_check(cell, static_cast<T>(0)); // no-op, returns *cell
        olds[1] = rocsparse::atomic_add_check(cell, static_cast<T>(5)); // adds 5, returns prior
    }
    template <typename T>
    __global__ void k_atomic_add_check_idx(T* base, T* olds)
    {
        // base has >= 3 elements; operate on index 2, size 3.
        olds[0] = rocsparse::atomic_add_check(base, int64_t{2}, int64_t{3}, static_cast<T>(0));
        olds[1] = rocsparse::atomic_add_check(base, int64_t{2}, int64_t{3}, static_cast<T>(4));
    }

    template <typename T>
    void run_atomic_add_check()
    {
        std::vector<T>   init(1, static_cast<T>(10));
        device_vector<T> d_cell(init), d_olds(size_t{2});
        ASSERT_NE(d_cell.ptr, nullptr);
        ASSERT_NE(d_olds.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_add_check<T>, 1u, d_cell.ptr, d_olds.ptr),
                  hipSuccess);
        auto olds = to_host(d_olds);
        EXPECT_EQ(olds[0], static_cast<T>(10)); // zero add returns current value
        EXPECT_EQ(olds[1], static_cast<T>(10)); // nonzero add returns prior value
        EXPECT_EQ(to_host(d_cell)[0], static_cast<T>(15));
    }

    template <typename T>
    void run_atomic_add_check_idx()
    {
        std::vector<T>   init{static_cast<T>(1), static_cast<T>(2), static_cast<T>(10)};
        device_vector<T> d_base(init), d_olds(size_t{2});
        ASSERT_NE(d_base.ptr, nullptr);
        ASSERT_NE(d_olds.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_atomic_add_check_idx<T>, 1u, d_base.ptr, d_olds.ptr),
                  hipSuccess);
        auto olds = to_host(d_olds);
        auto base = to_host(d_base);
        EXPECT_EQ(olds[0], static_cast<T>(10));
        EXPECT_EQ(olds[1], static_cast<T>(10));
        EXPECT_EQ(base[2], static_cast<T>(14));
    }
} // namespace

TEST(internal_collective_extras_atomic_add_check, i32)
{
    run_atomic_add_check<int32_t>();
}
TEST(internal_collective_extras_atomic_add_check, f32)
{
    run_atomic_add_check<float>();
}
TEST(internal_collective_extras_atomic_add_check, f64)
{
    run_atomic_add_check<double>();
}
TEST(internal_collective_extras_atomic_add_check, i32_index_overload)
{
    run_atomic_add_check_idx<int32_t>();
}
TEST(internal_collective_extras_atomic_add_check, f32_index_overload)
{
    run_atomic_add_check_idx<float>();
}

// ===========================================================================
// atomic_add_by_CAS : half / bfloat16 packed-CAS adder. One wavefront of threads
// each add 1.0 onto a single element; the accumulated sum is deterministic
// (small exact integers), so the test is race-free. Launches on the device's
// runtime wavefront width, so the expected total is that width.
//   * even-sized array, index 0        -> paired-CAS path
//   * odd-sized array, last element     -> spinlock + wfreduce_sum_mask path
// ===========================================================================
namespace
{
    __global__ void k_atomic_add_by_CAS_half(half* base, int64_t idx, int64_t size)
    {
        rocsparse::atomic_add_by_CAS(base, idx, __float2half(1.0f), size);
    }
    __global__ void k_atomic_add_by_CAS_bf16(rocsparse_bfloat16* base, int64_t idx, int64_t size)
    {
        rocsparse::atomic_add_by_CAS(base, idx, static_cast<rocsparse_bfloat16>(1.0f), size);
    }

    // Read one half element back as float.
    float read_half(const half* d_ptr, int64_t idx)
    {
        std::vector<uint16_t> h(static_cast<size_t>(idx) + 1);
        (void)hipMemcpy(h.data(), d_ptr, h.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
        _Float16 v;
        std::memcpy(&v, &h[idx], sizeof(v));
        return static_cast<float>(v);
    }
} // namespace

TEST(internal_collective_extras_atomic_add_by_CAS, half_even_paired)
{
    const int64_t           size = 4; // even
    const unsigned int      n = require_wavefront_size(); // one wavefront adds 1.0 onto element 0
    std::vector<uint16_t>   zero(static_cast<size_t>(size), 0);
    device_vector<uint16_t> d_raw(zero);
    ASSERT_NE(d_raw.ptr, nullptr);
    half* d_base = reinterpret_cast<half*>(d_raw.ptr);
    ASSERT_EQ(launch_single_block(k_atomic_add_by_CAS_half, n, d_base, int64_t{0}, size),
              hipSuccess);
    EXPECT_FLOAT_EQ(read_half(d_base, 0), static_cast<float>(n));
}
TEST(internal_collective_extras_atomic_add_by_CAS, half_odd_last_spinlock)
{
    const int64_t           size = 5; // odd -> last element takes the spinlock path
    const unsigned int      n    = require_wavefront_size();
    std::vector<uint16_t>   zero(static_cast<size_t>(size), 0);
    device_vector<uint16_t> d_raw(zero);
    ASSERT_NE(d_raw.ptr, nullptr);
    half* d_base = reinterpret_cast<half*>(d_raw.ptr);
    ASSERT_EQ(launch_single_block(k_atomic_add_by_CAS_half, n, d_base, int64_t{size - 1}, size),
              hipSuccess);
    EXPECT_FLOAT_EQ(read_half(d_base, size - 1), static_cast<float>(n));
}
TEST(internal_collective_extras_atomic_add_by_CAS, bfloat16_even_paired)
{
    const int64_t           size = 4;
    const unsigned int      n    = require_wavefront_size();
    std::vector<uint16_t>   zero(static_cast<size_t>(size), 0);
    device_vector<uint16_t> d_raw(zero);
    ASSERT_NE(d_raw.ptr, nullptr);
    rocsparse_bfloat16* d_base = reinterpret_cast<rocsparse_bfloat16*>(d_raw.ptr);
    ASSERT_EQ(launch_single_block(k_atomic_add_by_CAS_bf16, n, d_base, int64_t{0}, size),
              hipSuccess);
    std::vector<uint16_t> hraw(static_cast<size_t>(size));
    (void)hipMemcpy(hraw.data(), d_raw.ptr, hraw.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
    rocsparse_bfloat16 bf;
    bf.data = hraw[0];
    EXPECT_FLOAT_EQ(static_cast<float>(bf), static_cast<float>(n));
}

// ===========================================================================
// elementwise min / max (host+device __forceinline). These are __device__
// __host__, so we call them directly on the host.
// ===========================================================================
TEST(internal_collective_extras_minmax, int32)
{
    EXPECT_EQ(rocsparse::min(int32_t{-3}, int32_t{5}), int32_t{-3});
    EXPECT_EQ(rocsparse::max(int32_t{-3}, int32_t{5}), int32_t{5});
}
TEST(internal_collective_extras_minmax, int64)
{
    EXPECT_EQ(rocsparse::min(int64_t{7}, int64_t{-100}), int64_t{-100});
    EXPECT_EQ(rocsparse::max(int64_t{7}, int64_t{-100}), int64_t{7});
}
TEST(internal_collective_extras_minmax, uint32)
{
    EXPECT_EQ(rocsparse::min(uint32_t{3}, uint32_t{5}), uint32_t{3});
    EXPECT_EQ(rocsparse::max(uint32_t{3}, uint32_t{5}), uint32_t{5});
}
TEST(internal_collective_extras_minmax, uint64)
{
    EXPECT_EQ(rocsparse::min(uint64_t{300}, uint64_t{5}), uint64_t{5});
    EXPECT_EQ(rocsparse::max(uint64_t{300}, uint64_t{5}), uint64_t{300});
}
TEST(internal_collective_extras_minmax, float)
{
    EXPECT_FLOAT_EQ(rocsparse::min(-3.5f, 5.25f), -3.5f);
    EXPECT_FLOAT_EQ(rocsparse::max(-3.5f, 5.25f), 5.25f);
}
TEST(internal_collective_extras_minmax, double)
{
    EXPECT_DOUBLE_EQ(rocsparse::min(-3.5, 5.25), -3.5);
    EXPECT_DOUBLE_EQ(rocsparse::max(-3.5, 5.25), 5.25);
}

// ===========================================================================
// nontemporal_load / nontemporal_store round-trip via device wrappers.
// ===========================================================================
namespace
{
    template <typename T>
    __global__ void k_nontemporal_roundtrip(const T* in, T* out)
    {
        const int i = threadIdx.x;
        T         v = rocsparse::nontemporal_load(in + i);
        rocsparse::nontemporal_store(v, out + i);
    }

    template <typename T>
    void run_nontemporal_roundtrip(const std::vector<T>& in)
    {
        const unsigned int n = static_cast<unsigned int>(in.size());
        device_vector<T>   d_in(in), d_out(in.size());
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_nontemporal_roundtrip<T>, n, d_in.ptr, d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(size_t i = 0; i < in.size(); ++i)
            EXPECT_EQ(h[i], in[i]);
    }
} // namespace

TEST(internal_collective_extras_nontemporal, i32)
{
    std::vector<int32_t> in{-5, 0, 7, 42, -1000};
    run_nontemporal_roundtrip<int32_t>(in);
}
TEST(internal_collective_extras_nontemporal, i64)
{
    std::vector<int64_t> in{-5, 0, 7, 42, 9999999999LL};
    run_nontemporal_roundtrip<int64_t>(in);
}
TEST(internal_collective_extras_nontemporal, f32)
{
    std::vector<float> in{-5.5f, 0.0f, 7.25f, 42.125f};
    run_nontemporal_roundtrip<float>(in);
}
TEST(internal_collective_extras_nontemporal, f64)
{
    std::vector<double> in{-5.5, 0.0, 7.25, 42.125};
    run_nontemporal_roundtrip<double>(in);
}
namespace
{
    void expect_eq_complex(const rocsparse_float_complex& a, const rocsparse_float_complex& b)
    {
        EXPECT_FLOAT_EQ(std::real(a), std::real(b));
        EXPECT_FLOAT_EQ(std::imag(a), std::imag(b));
    }
} // namespace
TEST(internal_collective_extras_nontemporal, complex_f32)
{
    std::vector<rocsparse_float_complex>   in{{1.0f, 2.0f}, {-3.0f, 0.5f}, {0.0f, -7.0f}};
    device_vector<rocsparse_float_complex> d_in(in), d_out(in.size());
    ASSERT_NE(d_in.ptr, nullptr);
    ASSERT_NE(d_out.ptr, nullptr);
    ASSERT_EQ(launch_single_block(k_nontemporal_roundtrip<rocsparse_float_complex>,
                                  static_cast<unsigned int>(in.size()),
                                  d_in.ptr,
                                  d_out.ptr),
              hipSuccess);
    auto h = to_host(d_out);
    for(size_t i = 0; i < in.size(); ++i)
        expect_eq_complex(h[i], in[i]);
}

// ===========================================================================
// coo2csr lower_bound<I,J> : binary lower-bound over a sorted array.
// ===========================================================================
namespace
{
    template <typename I, typename J>
    __global__ void k_lower_bound(const J* arr, const J* keys, I low, I high, I* out)
    {
        const int t = threadIdx.x;
        out[t]      = rocsparse::lower_bound<I, J>(arr, keys[t], low, high);
    }

    template <typename I, typename J>
    void run_lower_bound()
    {
        // Sorted, with duplicates and gaps to exercise <-vs-<= boundary logic.
        const std::vector<J> arr{0, 2, 2, 2, 5, 9, 9, 14};
        const I              high = static_cast<I>(arr.size());
        std::vector<J>       keys;
        for(J k = -1; k <= 16; ++k)
            keys.push_back(k);
        const size_t nq = keys.size();

        std::vector<I> ref(nq);
        for(size_t q = 0; q < nq; ++q)
            ref[q]
                = static_cast<I>(std::lower_bound(arr.begin(), arr.end(), keys[q]) - arr.begin());

        device_vector<J> d_arr(arr), d_keys(keys);
        device_vector<I> d_out(nq);
        ASSERT_NE(d_arr.ptr, nullptr);
        ASSERT_NE(d_keys.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_lower_bound<I, J>,
                                      static_cast<unsigned int>(nq),
                                      d_arr.ptr,
                                      d_keys.ptr,
                                      static_cast<I>(0),
                                      high,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(size_t q = 0; q < nq; ++q)
            EXPECT_EQ(h[q], ref[q]) << "key=" << keys[q];
    }
} // namespace

TEST(internal_collective_extras_lower_bound, i32_j32)
{
    run_lower_bound<int32_t, int32_t>();
}
TEST(internal_collective_extras_lower_bound, i64_j32)
{
    run_lower_bound<int64_t, int32_t>();
}
TEST(internal_collective_extras_lower_bound, i64_j64)
{
    run_lower_bound<int64_t, int64_t>();
}

// ===========================================================================
// csrgemm insert_key / insert_pair : single-block linear-probing hash table.
// ===========================================================================
namespace
{
    constexpr uint32_t HASHVAL  = 7u;
    constexpr uint32_t HASHSIZE = 32u; // power of two

    template <typename I>
    __global__ void k_insert_key(const I* keys, I* table, int32_t* inserted)
    {
        const int t   = threadIdx.x;
        bool      ins = rocsparse::insert_key<HASHVAL, HASHSIZE, I>(keys[t], table);
        inserted[t]   = ins ? 1 : 0;
    }

    template <typename I, typename T>
    __global__ void k_insert_pair(const I* keys, const T* vals, I* table, T* data)
    {
        const int t = threadIdx.x;
        rocsparse::insert_pair<HASHVAL, HASHSIZE, I, T>(
            keys[t], vals[t], table, data, static_cast<I>(-1));
    }
} // namespace

TEST(internal_collective_extras_insert_key, distinct_and_dedupe)
{
    using I = int32_t;
    // 24 insertions with duplicates -> 12 distinct keys (each appears twice).
    std::vector<I> keys;
    for(int rep = 0; rep < 2; ++rep)
        for(int k = 0; k < 12; ++k)
            keys.push_back(k * 3 + 1); // distinct keys {1,4,7,...,34}
    const unsigned int n = static_cast<unsigned int>(keys.size());

    std::vector<I>         table(HASHSIZE, static_cast<I>(-1));
    device_vector<I>       d_keys(keys), d_table(table);
    device_vector<int32_t> d_ins(size_t{n});
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_table.ptr, nullptr);
    ASSERT_NE(d_ins.ptr, nullptr);
    ASSERT_EQ(launch_single_block(k_insert_key<I>, n, d_keys.ptr, d_table.ptr, d_ins.ptr),
              hipSuccess);

    auto h_ins   = to_host(d_ins);
    auto h_table = to_host(d_table);

    // Exactly 12 insertions must have reported "true" (one per distinct key).
    int true_count = 0;
    for(auto v : h_ins)
        true_count += v;
    EXPECT_EQ(true_count, 12);

    // The table must contain exactly the 12 distinct keys (order/slot arbitrary).
    std::vector<I> present;
    for(auto v : h_table)
        if(v != static_cast<I>(-1))
            present.push_back(v);
    std::sort(present.begin(), present.end());
    std::vector<I> expected;
    for(int k = 0; k < 12; ++k)
        expected.push_back(k * 3 + 1);
    EXPECT_EQ(present, expected);
}

TEST(internal_collective_extras_insert_pair, accumulate_by_key)
{
    using I = int32_t;
    using T = float;
    // Keys 0..7, each appearing 4 times, with per-insertion value 1.0 -> each
    // distinct key accumulates 4.0.
    std::vector<I> keys;
    std::vector<T> vals;
    for(int rep = 0; rep < 4; ++rep)
        for(int k = 0; k < 8; ++k)
        {
            keys.push_back(k);
            vals.push_back(1.0f);
        }
    const unsigned int n = static_cast<unsigned int>(keys.size());

    std::vector<I>   table(HASHSIZE, static_cast<I>(-1));
    std::vector<T>   data(HASHSIZE, T(0));
    device_vector<I> d_keys(keys), d_table(table);
    device_vector<T> d_vals(vals), d_data(data);
    ASSERT_NE(d_keys.ptr, nullptr);
    ASSERT_NE(d_vals.ptr, nullptr);
    ASSERT_NE(d_table.ptr, nullptr);
    ASSERT_NE(d_data.ptr, nullptr);
    ASSERT_EQ(launch_single_block(
                  k_insert_pair<I, T>, n, d_keys.ptr, d_vals.ptr, d_table.ptr, d_data.ptr),
              hipSuccess);

    auto h_table = to_host(d_table);
    auto h_data  = to_host(d_data);

    // Map each occupied slot's key to its accumulated data value.
    std::vector<float> acc_by_key(8, -1.0f);
    int                occupied = 0;
    for(uint32_t s = 0; s < HASHSIZE; ++s)
    {
        if(h_table[s] != static_cast<I>(-1))
        {
            ++occupied;
            const I key = h_table[s];
            ASSERT_GE(key, 0);
            ASSERT_LT(key, 8);
            acc_by_key[key] = h_data[s];
        }
    }
    EXPECT_EQ(occupied, 8);
    for(int k = 0; k < 8; ++k)
        EXPECT_FLOAT_EQ(acc_by_key[k], 4.0f) << "key=" << k;
}
