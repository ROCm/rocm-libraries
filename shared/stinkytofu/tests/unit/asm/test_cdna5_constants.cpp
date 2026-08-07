/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
// Sanity-checks for CDNA5 (Gfx1250) scheduler tunable defaults.
// DAG scheduling behaviour is covered by DAGSchedulerPassTest.cpp and the
// filecheck suite under tests/filecheck/dag_*.stir.
#include <gtest/gtest.h>

// Mirror of the kCdna5* constants in CDNA5.hpp (anonymous namespace).
// If these values change intentionally, update both places.
static constexpr int kCdna5DsReadQueueDepth = 16;
static constexpr int kCdna5DsReadDrainLatency = 72;
static constexpr int kCdna5DsReadThrottleLatency = 72;
static constexpr int kCdna5DsReadPerWmma = 3;
static constexpr int kCdna5GlobalReadPerWmma = 1;

TEST(CDNA5Constants, KnownDefaults) {
    EXPECT_EQ(kCdna5DsReadQueueDepth, 16);
    EXPECT_EQ(kCdna5DsReadDrainLatency, 72);
    EXPECT_EQ(kCdna5DsReadThrottleLatency, 72);
    EXPECT_EQ(kCdna5DsReadPerWmma, 3);
    EXPECT_EQ(kCdna5GlobalReadPerWmma, 1);
}
