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
// Device (GPU) unit tests for rocSPARSE internal ATOMIC building blocks:
// atomic_add / atomic_min / atomic_max / atomic_cas / atomic_load /
// atomic_store / atomic_add_check / atomic_add_by_CAS. Split out of
// unit_test_internal_collective_extras.cpp by family. Operands are chosen so
// the final accumulated result is deterministic despite nondeterministic
// atomic ordering, so the tests are race-free by construction.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collective_extras_common.hpp"

#include "rocsparse_common.hpp" // atomic_* building blocks

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collective_extras;

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
