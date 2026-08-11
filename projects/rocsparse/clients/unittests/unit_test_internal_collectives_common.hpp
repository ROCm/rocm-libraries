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
// Shared helpers for the internal device-collective unit tests, which are split
// by collective family across the unit_test_internal_collectives_*.cpp TUs
// (blockreduce, warpreduce, segmented, search, shfl, boost). Provides the
// type-aware "close enough" numeric comparison and the runtime wavefront-size
// accessor used by the warp-collective tests. Header-only (inline) so every
// topical TU that includes it stays self-contained; the helpers live in a named
// namespace so they never clash with the sibling extras-leaf helpers when both
// sets of TUs are linked into the same rocsparse-unit-test-device binary.
//
#pragma once

#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp" // rocsparse complex types / std::real, std::imag support

#include <gtest/gtest.h>

#include <cstdint>

namespace rocsparse_ut_collectives
{
    // ---- type-aware "close enough" comparison ------------------------------
    inline void expect_close(float a, float b)
    {
        EXPECT_FLOAT_EQ(a, b);
    }
    inline void expect_close(double a, double b)
    {
        EXPECT_DOUBLE_EQ(a, b);
    }
    inline void expect_close(int32_t a, int32_t b)
    {
        EXPECT_EQ(a, b);
    }
    inline void expect_close(int64_t a, int64_t b)
    {
        EXPECT_EQ(a, b);
    }
    inline void expect_close(uint32_t a, uint32_t b)
    {
        EXPECT_EQ(a, b);
    }
    inline void expect_close(const rocsparse_float_complex& a, const rocsparse_float_complex& b)
    {
        EXPECT_FLOAT_EQ(std::real(a), std::real(b));
        EXPECT_FLOAT_EQ(std::imag(a), std::imag(b));
    }
    inline void expect_close(const rocsparse_double_complex& a, const rocsparse_double_complex& b)
    {
        EXPECT_DOUBLE_EQ(std::real(a), std::real(b));
        EXPECT_DOUBLE_EQ(std::imag(a), std::imag(b));
    }

    // Returns the active device wavefront size and asserts it is one of the two
    // supported values so a misconfigured device fails loudly instead of
    // silently skipping. Used by every warp-collective test.
    inline uint32_t require_wavefront_size()
    {
        const int ws = rocsparse_ut::device_warp_size();
        EXPECT_TRUE(ws == 32 || ws == 64) << "unsupported device wavefront size: " << ws;
        return static_cast<uint32_t>(ws);
    }
} // namespace rocsparse_ut_collectives
