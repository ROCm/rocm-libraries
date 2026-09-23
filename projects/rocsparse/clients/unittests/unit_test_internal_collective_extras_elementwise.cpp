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
// Device (GPU) unit tests for rocSPARSE internal ELEMENTWISE min / max
// (rocsparse::min / rocsparse::max int/uint/float/double overloads). Split out
// of unit_test_internal_collective_extras.cpp by family. These are
// __device__ __host__ __forceinline, so the tests call them directly on the
// host.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp" // rocsparse::min / rocsparse::max

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>

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
