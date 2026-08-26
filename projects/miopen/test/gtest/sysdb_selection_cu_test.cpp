/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <miopen/handle.hpp>
#include <gtest/gtest.h>

// Unit tests for miopen::GetSysDbSelectionCu, which maps a device's real
// compute-unit count to the CU count used when picking a *system* database by
// closest-CU match. Low-CU gfx942 parts (e.g. MI308) ship no dedicated system
// DB, so they are redirected to the high-CU (304) gfx942 system database.
// The redirect must be scoped to gfx942 and must not perturb any other device.

namespace {

// The redirect threshold and target baked into GetSysDbSelectionCu.
constexpr int gfx942_low_cu_threshold = 80;
constexpr int gfx942_sysdb_target_cu  = 304;

} // namespace

TEST(CPU_SysDbSelectionCu_NONE, Gfx942LowCuRedirectsToHighCuDb)
{
    // A low-CU gfx942 part (MI308-class) must borrow the high-CU gfx942 DB.
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", 1), gfx942_sysdb_target_cu);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", 20), gfx942_sysdb_target_cu);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", 64), gfx942_sysdb_target_cu);
}

TEST(CPU_SysDbSelectionCu_NONE, Gfx942ThresholdBoundary)
{
    // At and below the threshold the redirect applies; one CU above it does not.
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", gfx942_low_cu_threshold),
              gfx942_sysdb_target_cu);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", gfx942_low_cu_threshold + 1),
              gfx942_low_cu_threshold + 1);
}

TEST(CPU_SysDbSelectionCu_NONE, Gfx942HighCuUnchanged)
{
    // High-CU gfx942 parts already have a matching DB; the value passes through,
    // including the 304-CU part itself and a typical 228-CU (MI300X-class) part.
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", 228), 228);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx942", gfx942_sysdb_target_cu),
              gfx942_sysdb_target_cu);
}

TEST(CPU_SysDbSelectionCu_NONE, OtherArchesArePassthrough)
{
    // The redirect is scoped to gfx942 only; every other db_id keeps its real
    // CU count, even when that count is at or below the gfx942 threshold.
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx90a", 64), 64);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx950", 32), 32);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx1100", gfx942_low_cu_threshold),
              gfx942_low_cu_threshold);
    EXPECT_EQ(miopen::GetSysDbSelectionCu("gfx908", 120), 120);
}

TEST(CPU_SysDbSelectionCu_NONE, EmptyDbIdIsPassthrough)
{
    // A missing/empty db_id must not accidentally trigger the gfx942 redirect.
    EXPECT_EQ(miopen::GetSysDbSelectionCu("", 40), 40);
}
