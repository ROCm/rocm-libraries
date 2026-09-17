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
// Shared helper for the residual internal device-collective unit tests, split by
// family across the unit_test_internal_collective_extras_*.cpp TUs (wfreduce,
// atomics, elementwise, nontemporal, lower_bound, insert). Provides the runtime
// wavefront-size accessor used by the warp-oriented tests. Header-only (inline)
// so every topical TU that includes it stays self-contained; the helper lives in
// a named namespace so it never clashes with the sibling collectives-leaf helper
// of the same name when both sets of TUs are linked into the same
// rocsparse-unit-test-device binary.
//
#pragma once

#include "unit_test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace rocsparse_ut_collective_extras
{
    // Active device wavefront size (32 or 64); asserted so a misconfigured device
    // fails loudly instead of silently skipping. Warp tests size their data and
    // reference to this runtime value.
    inline uint32_t require_wavefront_size()
    {
        const int ws = rocsparse_ut::device_warp_size();
        EXPECT_TRUE(ws == 32 || ws == 64) << "unsupported device wavefront size: " << ws;
        return static_cast<uint32_t>(ws);
    }
} // namespace rocsparse_ut_collective_extras
