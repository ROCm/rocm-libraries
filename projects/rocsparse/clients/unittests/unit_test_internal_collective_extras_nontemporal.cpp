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
// Device (GPU) unit tests for rocSPARSE internal NONTEMPORAL load / store
// (rocsparse::nontemporal_load / nontemporal_store round-trip). Split out of
// unit_test_internal_collective_extras.cpp by family.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp" // nontemporal_load, nontemporal_store

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::to_host;

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
    void expect_eq_complex(const rocsparse_float_complex& a, const rocsparse_float_complex& b)
    {
        EXPECT_FLOAT_EQ(std::real(a), std::real(b));
        EXPECT_FLOAT_EQ(std::imag(a), std::imag(b));
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
