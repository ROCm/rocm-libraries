// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <gtest/gtest.h>

#include <limits>

namespace {
using miopen::CheckWorkgroupCount;
using miopen::GetMaxWorkgroupsYZ;

constexpr std::array<size_t, 3> ldims{8, 4, 1};

// Y and Z workgroup counts; one workgroup in X.
std::array<size_t, 3> Grid(size_t y_workgroups, size_t z_workgroups)
{
    return {8, 4 * y_workgroups, z_workgroups};
}
} // namespace

TEST(CPU_hipoc_workgroup_limit_NONE, arch_limits)
{
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1201"), 65536u);
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1200"), 65536u);
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1250"), 65535u);
    // Feature suffixes reported by gcnArchName.
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1250-strict"), 65535u);
    EXPECT_EQ(GetMaxWorkgroupsYZ("gfx1201:sramecc+:xnack-"), 65536u);
    for(const char* arch : {"gfx942", "gfx950", "gfx1100", "gfx1151", "gfx90a"})
        EXPECT_EQ(GetMaxWorkgroupsYZ(arch), std::numeric_limits<size_t>::max()) << arch;
}

TEST(CPU_hipoc_workgroup_limit_NONE, gfx1201_boundary)
{
    const auto limit = GetMaxWorkgroupsYZ("gfx1201");
    EXPECT_NO_THROW(CheckWorkgroupCount(Grid(65536, 1), ldims, "k", limit));
    EXPECT_NO_THROW(CheckWorkgroupCount(Grid(1, 65536), ldims, "k", limit));
    EXPECT_THROW(CheckWorkgroupCount(Grid(65537, 1), ldims, "k", limit), miopen::Exception);
    EXPECT_THROW(CheckWorkgroupCount(Grid(1, 65537), ldims, "k", limit), miopen::Exception);
}

TEST(CPU_hipoc_workgroup_limit_NONE, gfx1250_boundary)
{
    // 65536 workgroups already run nothing on gfx1250.
    const auto limit = GetMaxWorkgroupsYZ("gfx1250");
    EXPECT_NO_THROW(CheckWorkgroupCount(Grid(65535, 1), ldims, "k", limit));
    EXPECT_THROW(CheckWorkgroupCount(Grid(65536, 1), ldims, "k", limit), miopen::Exception);
    EXPECT_THROW(CheckWorkgroupCount(Grid(1, 65536), ldims, "k", limit), miopen::Exception);
}

TEST(CPU_hipoc_workgroup_limit_NONE, unlimited_arch_and_x_dimension)
{
    const auto limit = GetMaxWorkgroupsYZ("gfx942");
    EXPECT_NO_THROW(CheckWorkgroupCount(Grid(1000000, 1000000), ldims, "k", limit));
    // X is not limited.
    EXPECT_NO_THROW(
        CheckWorkgroupCount({8 * 1000000, 4, 1}, ldims, "k", GetMaxWorkgroupsYZ("gfx1250")));
}

TEST(CPU_hipoc_workgroup_limit_NONE, partial_workgroup_rounds_up)
{
    // 65536 full workgroups plus one work-item is 65537 workgroups.
    const auto limit = GetMaxWorkgroupsYZ("gfx1201");
    EXPECT_THROW(CheckWorkgroupCount({8, 4 * 65536 + 1, 1}, ldims, "k", limit), miopen::Exception);
}
