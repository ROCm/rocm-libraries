// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/ArchMatch.hpp>

using hipdnn_test_sdk::utilities::archMatches;
using hipdnn_test_sdk::utilities::ArchMatchMode;

// ---------------------------------------------------------------------------
// Strict mode — arch-locking semantics.
// Candidate must be the base-arch prefix of the device string, terminated by
// ':' or end-of-string. Used by the golden-data arch guard.
// ---------------------------------------------------------------------------

TEST(TestArchMatchStrict, MatchesBareArchExactly)
{
    EXPECT_TRUE(archMatches("gfx942", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestArchMatchStrict, MatchesBaseArchAgainstFeatureSuffix)
{
    EXPECT_TRUE(archMatches("gfx942:sramecc+:xnack-", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestArchMatchStrict, MatchesFullFeatureStringExactly)
{
    EXPECT_TRUE(
        archMatches("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-", ArchMatchMode::PREFIX));
}

TEST(TestArchMatchStrict, RejectsPartialArchName)
{
    // "gfx94" is a prefix of "gfx942" but not a complete base arch: the next
    // char is '2', not ':'.
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx94", ArchMatchMode::PREFIX));
}

TEST(TestArchMatchStrict, RejectsDifferentArch)
{
    EXPECT_FALSE(archMatches("gfx1100", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestArchMatchStrict, RejectsDifferingFeatureFlags)
{
    EXPECT_FALSE(
        archMatches("gfx942:sramecc-:xnack-", "gfx942:sramecc+:xnack-", ArchMatchMode::PREFIX));
}

// ---------------------------------------------------------------------------
// Loose mode — test-skip family semantics.
// Candidate is any literal substring of the device string. Used by the
// test-skip system so one entry (e.g. "gfx10") covers an arch family.
// ---------------------------------------------------------------------------

TEST(TestArchMatchLoose, MatchesBaseArchAgainstFeatureSuffix)
{
    EXPECT_TRUE(archMatches("gfx942:sramecc+:xnack-", "gfx942", ArchMatchMode::SUBSTRING));
}

TEST(TestArchMatchLoose, MatchesFamilyPrefix)
{
    // "gfx10" is meant to cover the whole gfx10xx family.
    EXPECT_TRUE(archMatches("gfx1030", "gfx10", ArchMatchMode::SUBSTRING));
    EXPECT_TRUE(archMatches("gfx1100", "gfx11", ArchMatchMode::SUBSTRING));
}

TEST(TestArchMatchLoose, RejectsNonSubstring)
{
    // A more-qualified candidate is a literal substring search: "gfx942:xnack-"
    // does not appear in "gfx942:sramecc+:xnack-".
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx942:xnack-", ArchMatchMode::SUBSTRING));
}

TEST(TestArchMatchLoose, RejectsDifferentArch)
{
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx1100", ArchMatchMode::SUBSTRING));
}

TEST(TestArchMatchLoose, FailsAgainstEmptyDeviceArch)
{
    // Empty device arch (could not be queried) must not match a real candidate.
    EXPECT_FALSE(archMatches("", "gfx942", ArchMatchMode::SUBSTRING));
}
