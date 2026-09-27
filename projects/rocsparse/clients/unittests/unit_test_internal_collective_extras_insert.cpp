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
// Device (GPU) unit tests for rocSPARSE internal csrgemm hash-table inserts
// (rocsparse::insert_key / insert_pair single-block linear-probing hash).
// Split out of unit_test_internal_collective_extras.cpp by family.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp"

// The csrgemm hash helpers live in the extra device header. The device
// unit-test target only puts library/src/{include,level1,level3} on the
// include path, so we reach it with a source-relative include (this TU lives
// in clients/unittests/). This keeps the addition local to this file and
// avoids a shared CMakeLists.txt include-dir change.
#include "../../library/src/extra/csrgemm_device.h" // rocsparse::insert_key / insert_pair

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::to_host;

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
