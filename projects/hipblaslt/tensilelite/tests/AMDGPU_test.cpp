/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>

using namespace TensileLite;

// ---------------------------------------------------------------------------
// Verify that the RDNA2 gfx1031/1032/1034/1035 entries are present in the
// AMDGPU::Processor enum and round-trip correctly through toProcessor/toString.
// These were previously missing, causing "Enum not found!" at kernel load time
// on consumer RDNA2 GPUs (e.g. RX 6650 XT = gfx1032). See issue #1202.
// ---------------------------------------------------------------------------

TEST(AMDGPU_Processor, ToString_gfx1031)
{
    EXPECT_EQ(AMDGPU::toString(AMDGPU::Processor::gfx1031), "gfx1031");
}

TEST(AMDGPU_Processor, ToString_gfx1032)
{
    EXPECT_EQ(AMDGPU::toString(AMDGPU::Processor::gfx1032), "gfx1032");
}

TEST(AMDGPU_Processor, ToString_gfx1034)
{
    EXPECT_EQ(AMDGPU::toString(AMDGPU::Processor::gfx1034), "gfx1034");
}

TEST(AMDGPU_Processor, ToString_gfx1035)
{
    EXPECT_EQ(AMDGPU::toString(AMDGPU::Processor::gfx1035), "gfx1035");
}

TEST(AMDGPU_Processor, ToProcessor_gfx1031)
{
    EXPECT_EQ(AMDGPU::toProcessor("gfx1031"), AMDGPU::Processor::gfx1031);
}

TEST(AMDGPU_Processor, ToProcessor_gfx1032)
{
    EXPECT_EQ(AMDGPU::toProcessor("gfx1032"), AMDGPU::Processor::gfx1032);
}

TEST(AMDGPU_Processor, ToProcessor_gfx1034)
{
    EXPECT_EQ(AMDGPU::toProcessor("gfx1034"), AMDGPU::Processor::gfx1034);
}

TEST(AMDGPU_Processor, ToProcessor_gfx1035)
{
    EXPECT_EQ(AMDGPU::toProcessor("gfx1035"), AMDGPU::Processor::gfx1035);
}

// Round-trip: toProcessor -> toString should give back the original name.
TEST(AMDGPU_Processor, RoundTrip_gfx1032)
{
    auto proc = AMDGPU::toProcessor("gfx1032");
    EXPECT_EQ(AMDGPU::toString(proc), "gfx1032");
}

// The full arch name string (as reported by hip) should still match.
TEST(AMDGPU_Processor, ToProcessor_matches_within_longer_string)
{
    // hipDeviceGetName may return strings like "gfx1032" or
    // "amd:gfx1032:sramecc-" — toProcessor uses substring search.
    EXPECT_EQ(AMDGPU::toProcessor("gfx1032:sramecc-"), AMDGPU::Processor::gfx1032);
}
