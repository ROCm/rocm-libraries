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

TEST(TestKpackArchSelection, AnExplicitEntryBeatsAGenericEntry)
{
    const std::vector<std::string> arches{"gfx11-generic", "gfx1151"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx1151");

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(*selected, "gfx1151");
}

TEST(TestKpackArchSelection, AGenericEntryServesAMemberDevice)
{
    const std::vector<std::string> arches{"gfx942", "gfx11-generic"};

    const std::string* member = selectKpackArchiveArch(arches, "gfx1151:xnack-");
    ASSERT_NE(member, nullptr);
    EXPECT_EQ(*member, "gfx11-generic");

    const std::string* other = selectKpackArchiveArch(arches, "gfx942");
    ASSERT_NE(other, nullptr);
    EXPECT_EQ(*other, "gfx942");
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
