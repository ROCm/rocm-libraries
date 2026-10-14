// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

TEST(TestKpackArchSelection, ADecoratedDevicePicksABareEntry)
{
    const std::vector<std::string> arches{"gfx942", "gfx90a"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx90a:sramecc+:xnack-");

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(*selected, "gfx90a");
}

TEST(TestKpackArchSelection, ADecoratedDevicePicksADecoratedEntry)
{
    const std::vector<std::string> arches{"gfx942", "gfx90a:xnack-"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx90a:xnack-");

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(*selected, "gfx90a:xnack-");
}

TEST(TestKpackArchSelection, TheFirstMatchingEntryWins)
{
    const std::vector<std::string> arches{"gfx90a", "gfx90a:xnack-"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx90a:xnack-");

    // The address, not only the text: the result points into the list it was given.
    EXPECT_EQ(selected, arches.data());
}

// A plan keeps the decorated device arch for this reason: the stripped name cannot select
// a decorated entry.
TEST(TestKpackArchSelection, AStrippedDeviceNameCannotSelectADecoratedEntry)
{
    const std::vector<std::string> arches{"gfx90a:xnack-"};

    EXPECT_EQ(selectKpackArchiveArch(arches, "gfx90a"), nullptr);
    EXPECT_NE(selectKpackArchiveArch(arches, "gfx90a:xnack-"), nullptr);
}

TEST(TestKpackArchSelection, AFamilyStemDoesNotMatch)
{
    const std::vector<std::string> arches{"gfx90", "gfx90a"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx90a:xnack-");

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(*selected, "gfx90a");
}

TEST(TestKpackArchSelection, NoEntryForTheDeviceSelectsNothing)
{
    const std::vector<std::string> arches{"gfx942", "gfx1100"};

    EXPECT_EQ(selectKpackArchiveArch(arches, "gfx90a:sramecc+:xnack-"), nullptr);
    EXPECT_EQ(selectKpackArchiveArch({}, "gfx90a"), nullptr);
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
