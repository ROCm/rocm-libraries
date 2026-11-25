// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

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

// The opt-out: exact entries are still chosen, generic ones never.
TEST(TestKpackArchSelection, AnExplicitEntryIsChosenWithGenericEntriesRefused)
{
    const std::vector<std::string> arches{"gfx11-generic", "gfx1151"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx1151:xnack-", false);

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(*selected, "gfx1151");
}

TEST(TestKpackArchSelection, AGenericEntryIsChosenWhenAllowed)
{
    const std::vector<std::string> arches{"gfx1100", "gfx11-generic"};

    const std::string* selected = selectKpackArchiveArch(arches, "gfx1101", true);

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(*selected, "gfx11-generic");
}

TEST(TestKpackArchSelection, AGenericEntryIsRefusedWhenDisabled)
{
    const std::vector<std::string> arches{"gfx1100", "gfx11-generic", "gfx12-generic"};

    EXPECT_EQ(selectKpackArchiveArch(arches, "gfx1101", false), nullptr);
    EXPECT_EQ(selectKpackArchiveArch(arches, "gfx1201", false), nullptr);
    // Not a generic refusal for the exact entry.
    ASSERT_NE(selectKpackArchiveArch(arches, "gfx1100", false), nullptr);
}

TEST(TestKpackArchSelection, TheEnvironmentVariableRefusesGenericEntries)
{
    const std::vector<std::string> arches{"gfx1100", "gfx11-generic"};

    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter unset(
            DISABLE_GENERIC_CODE_OBJECTS_ENV);
        EXPECT_FALSE(genericCodeObjectsDisabled());
        const std::string* selected = selectKpackArchiveArch(arches, "gfx1151");
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(*selected, "gfx11-generic");
    }
    for(const char* on : {"1", "true", "ON", " yes "})
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disable(
            DISABLE_GENERIC_CODE_OBJECTS_ENV, on);
        SCOPED_TRACE(on);
        EXPECT_TRUE(genericCodeObjectsDisabled());
        EXPECT_EQ(selectKpackArchiveArch(arches, "gfx1151"), nullptr);
        const std::string* exact = selectKpackArchiveArch(arches, "gfx1100");
        ASSERT_NE(exact, nullptr);
        EXPECT_EQ(*exact, "gfx1100");
    }
    for(const char* off : {"0", "false", ""})
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter keep(
            DISABLE_GENERIC_CODE_OBJECTS_ENV, off);
        SCOPED_TRACE(off);
        EXPECT_FALSE(genericCodeObjectsDisabled());
        EXPECT_NE(selectKpackArchiveArch(arches, "gfx1151"), nullptr);
    }
}

} // namespace
} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
