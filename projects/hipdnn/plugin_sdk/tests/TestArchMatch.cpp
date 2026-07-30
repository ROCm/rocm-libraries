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

TEST(PluginArchMatchPrefix, MatchesBareArchExactly)
{
    EXPECT_TRUE(archMatches("gfx942", "gfx942", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, MatchesBaseArchAgainstFeatureSuffix)
{
    EXPECT_TRUE(archMatches("gfx942:sramecc+:xnack-", "gfx942", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, MatchesFullFeatureStringExactly)
{
    EXPECT_TRUE(
        archMatches("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, RejectsPartialArchName)
{
    // "gfx94" is a prefix of "gfx942" but not a complete base arch: the next
    // char is '2', not ':'.
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx94", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, RejectsDifferentArch)
{
    EXPECT_FALSE(archMatches("gfx1100", "gfx942", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, RejectsDifferingFeatureFlags)
{
    EXPECT_FALSE(
        archMatches("gfx942:sramecc-:xnack-", "gfx942:sramecc+:xnack-", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, CandidateLongerThanDeviceRejected)
{
    EXPECT_FALSE(archMatches("gfx94", "gfx942", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, EmptyDeviceArchRejectsRealCandidate)
{
    EXPECT_FALSE(archMatches("", "gfx942", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchPrefix, FamilyStemDoesNotMatchWiderArch)
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

TEST(PluginArchMatchSubstring, MatchesBaseArchAgainstFeatureSuffix)
{
    EXPECT_TRUE(archMatches("gfx942:sramecc+:xnack-", "gfx942", ArchMatchMode::SUBSTRING));
}

TEST(PluginArchMatchSubstring, MatchesFamilyStem)
{
    EXPECT_TRUE(archMatches("gfx1030", "gfx10", ArchMatchMode::SUBSTRING));
    EXPECT_TRUE(archMatches("gfx1100", "gfx11", ArchMatchMode::SUBSTRING));
}

TEST(PluginArchMatchSubstring, RejectsNonSubstring)
{
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx942:xnack-", ArchMatchMode::SUBSTRING));
}

TEST(PluginArchMatchSubstring, RejectsDifferentArch)
{
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx1100", ArchMatchMode::SUBSTRING));
}

TEST(PluginArchMatchSubstring, FailsAgainstEmptyDeviceArch)
{
    EXPECT_FALSE(archMatches("", "gfx942", ArchMatchMode::SUBSTRING));
}

// ---------------------------------------------------------------------------
// Provider workaround scenarios — the exact calls the providers make, matched
// against raw gcnArchName strings (suffix intact), so the whole path is covered.
// ---------------------------------------------------------------------------

TEST(PluginArchMatchWorkarounds, Issue5409Gfx90aExactBasePrefix)
{
    // MIOpen #5409 gates on exactly gfx90a.
    EXPECT_TRUE(archMatches("gfx90a:sramecc+:xnack-", "gfx90a", ArchMatchMode::PREFIX));
    EXPECT_TRUE(archMatches("gfx90a", "gfx90a", ArchMatchMode::PREFIX));
    EXPECT_FALSE(archMatches("gfx942:sramecc+:xnack-", "gfx90a", ArchMatchMode::PREFIX));
}

TEST(PluginArchMatchWorkarounds, Issue9962Gfx115xFamilySubstring)
{
    // hipBLASLt #9962 gates on the gfx115x family.
    EXPECT_TRUE(archMatches("gfx1150:xnack-", "gfx115", ArchMatchMode::SUBSTRING));
    EXPECT_TRUE(archMatches("gfx1151", "gfx115", ArchMatchMode::SUBSTRING));
    EXPECT_FALSE(archMatches("gfx1100", "gfx115", ArchMatchMode::SUBSTRING));
}
