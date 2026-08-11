// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>

using hipdnn_plugin_sdk::archMatches;
using hipdnn_plugin_sdk::ArchMatchMode;

// ---------------------------------------------------------------------------
// PREFIX mode — exact base-arch gate.
// Candidate must be the base-arch prefix of the device string, terminated by
// ':' or end-of-string.
// ---------------------------------------------------------------------------

TEST(TestPluginArchMatchPrefix, MatchesBareArchExactly)
{
    EXPECT_TRUE(archMatches("gfx942", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, MatchesBaseArchAgainstFeatureSuffix)
{
    EXPECT_TRUE(archMatches("gfx942:sramecc+:xnack-", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, MatchesFullFeatureStringExactly)
{
    EXPECT_TRUE(
        archMatches("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, RejectsPartialArchName)
{
    // "gfx94" is a prefix of "gfx942" but not a complete base arch: the next
    // char is '2', not ':'.
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx94", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, RejectsDifferentArch)
{
    EXPECT_FALSE(archMatches("gfx1100", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, RejectsDifferingFeatureFlags)
{
    EXPECT_FALSE(
        archMatches("gfx942:sramecc-:xnack-", "gfx942:sramecc+:xnack-", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, CandidateLongerThanDeviceRejected)
{
    EXPECT_FALSE(archMatches("gfx94", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, EmptyDeviceArchRejectsRealCandidate)
{
    EXPECT_FALSE(archMatches("", "gfx942", ArchMatchMode::PREFIX));
}

TEST(TestPluginArchMatchPrefix, FamilyStemDoesNotMatchWiderArch)
{
    // Documents the intended limitation: a bare family stem cannot be expressed
    // with PREFIX. "gfx115" does NOT match "gfx1150" because the next char is
    // '0', not ':'. Family matching must use SUBSTRING (see below).
    EXPECT_FALSE(archMatches("gfx1150", "gfx115", ArchMatchMode::PREFIX));
}

// ---------------------------------------------------------------------------
// SUBSTRING mode — arch-family gate.
// Candidate is any literal substring of the device string.
// ---------------------------------------------------------------------------

TEST(TestPluginArchMatchSubstring, MatchesBaseArchAgainstFeatureSuffix)
{
    EXPECT_TRUE(archMatches("gfx942:sramecc+:xnack-", "gfx942", ArchMatchMode::SUBSTRING));
}

TEST(TestPluginArchMatchSubstring, MatchesFamilyStem)
{
    EXPECT_TRUE(archMatches("gfx1030", "gfx10", ArchMatchMode::SUBSTRING));
    EXPECT_TRUE(archMatches("gfx1100", "gfx11", ArchMatchMode::SUBSTRING));
}

TEST(TestPluginArchMatchSubstring, RejectsNonSubstring)
{
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx942:xnack-", ArchMatchMode::SUBSTRING));
}

TEST(TestPluginArchMatchSubstring, RejectsDifferentArch)
{
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx1100", ArchMatchMode::SUBSTRING));
}

TEST(TestPluginArchMatchSubstring, FailsAgainstEmptyDeviceArch)
{
    EXPECT_FALSE(archMatches("", "gfx942", ArchMatchMode::SUBSTRING));
}
