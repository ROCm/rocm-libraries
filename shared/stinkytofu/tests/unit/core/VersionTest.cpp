// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// versionsMatch() is exact-string comparison, no semver-range logic — it's the
// whole "no cross-version support" contract used by both PassBuilder::loadPlugin()
// and rocisa's import-time check. Tested directly with literal strings here so
// mismatch behavior is covered without needing to build stinkytofu twice —
// building a second, differently-versioned stinkytofu is not something a single
// CI run can do; two arbitrary strings are enough to exercise the same logic.

#include <gtest/gtest.h>

#include "stinkytofu/Version.h"

using namespace stinkytofu;

TEST(VersionTest, IdenticalStringsMatch) {
    EXPECT_TRUE(versionsMatch("1.2.3-abc1234", "1.2.3-abc1234"));
}

TEST(VersionTest, DifferentHashDoesNotMatch) {
    EXPECT_FALSE(versionsMatch("1.2.3-abc1234", "1.2.3-def5678"));
}

TEST(VersionTest, DifferentVersionNumberDoesNotMatch) {
    // Same commit hash, different X.Y.Z — still rejected. There is no
    // "compatible enough" case; this is deliberately not a semver comparison.
    EXPECT_FALSE(versionsMatch("1.2.3-abc1234", "1.2.4-abc1234"));
}

TEST(VersionTest, DirtySuffixMakesItsOwnVersion) {
    // A local, uncommitted change reports a different runtime version than a
    // clean build of the same commit — correctly treated as a mismatch, since
    // the dirty build genuinely contains different code.
    EXPECT_FALSE(versionsMatch("1.2.3-abc1234", "1.2.3-abc1234-dirty"));
}

TEST(VersionTest, RuntimeVersionIsSelfConsistent) {
    // getRuntimeVersion() must match this TU's own compile-time
    // STINKYTOFU_FULL_VERSION within a single build — the real-world checks in
    // loadPlugin()/init_stinkytofu() rely on exactly this property holding
    // whenever there is in fact only one stinkytofu in play.
    EXPECT_TRUE(versionsMatch(STINKYTOFU_FULL_VERSION, getRuntimeVersion()));
}
