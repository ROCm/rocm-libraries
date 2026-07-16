// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/input-init/SynthesisConfig.hpp"

using namespace hipdnn_integration_tests;

// ── setDefault (declaration functions) ──────────────────────────────────────

TEST(TestSynthesisConfig, SetDefaultWritesWhenEmpty)
{
    SynthesisConfig config;
    config.setDefault(1, FillSpec::free(-1.0f, 1.0f));

    const auto fill = config.fill(1);
    EXPECT_EQ(fill.kind, FillSpec::Kind::FREE);
    EXPECT_FLOAT_EQ(fill.lo, -1.0f);
    EXPECT_FLOAT_EQ(fill.hi, 1.0f);
}

TEST(TestSynthesisConfig, SetDefaultDoesNotOverwrite)
{
    SynthesisConfig config;
    config.setDefault(1, FillSpec::free(-1.0f, 1.0f));
    config.setDefault(1, FillSpec::free(-99.0f, 99.0f));

    const auto fill = config.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -1.0f);
    EXPECT_FLOAT_EQ(fill.hi, 1.0f);
}

// ── set (metadata / test code) ──────────────────────────────────────────────

TEST(TestSynthesisConfig, SetOverwritesDefault)
{
    SynthesisConfig config;
    config.setDefault(1, FillSpec::free(-1.0f, 1.0f));
    config.set(1, FillSpec::free(-5.0f, 5.0f));

    const auto fill = config.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -5.0f);
    EXPECT_FLOAT_EQ(fill.hi, 5.0f);
}

TEST(TestSynthesisConfig, SetOverwritesSet)
{
    SynthesisConfig config;
    config.set(1, FillSpec::free(-5.0f, 5.0f));
    config.set(1, FillSpec::free(-10.0f, 10.0f));

    const auto fill = config.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -10.0f);
    EXPECT_FLOAT_EQ(fill.hi, 10.0f);
}

TEST(TestSynthesisConfig, SetDefaultDoesNotOverwriteSet)
{
    SynthesisConfig config;
    config.set(1, FillSpec::free(-5.0f, 5.0f));
    config.setDefault(1, FillSpec::free(-99.0f, 99.0f));

    const auto fill = config.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -5.0f);
    EXPECT_FLOAT_EQ(fill.hi, 5.0f);
}

// ── Three-tier precedence (the real contract) ───────────────────────────────

TEST(TestSynthesisConfig, ThreeTierPrecedence)
{
    SynthesisConfig config;

    // 1. Metadata sets a range (runs first via setBundle)
    config.set(1, FillSpec::free(-1.0f, 1.0f));

    // 2. Declaration function tries to set a default (emplace, should lose)
    config.setDefault(1, FillSpec::free(-99.0f, 99.0f));

    // 3. Test body overwrites with its own range (runs after metadata)
    config.set(1, FillSpec::free(-10.0f, 10.0f));

    const auto fill = config.fill(1);
    EXPECT_EQ(fill.kind, FillSpec::Kind::FREE);
    EXPECT_FLOAT_EQ(fill.lo, -10.0f);
    EXPECT_FLOAT_EQ(fill.hi, 10.0f);
}

// ── get returns default-constructed FillSpec for unknown uid ────────────────

TEST(TestSynthesisConfig, GetUnknownUidReturnsDefault)
{
    const SynthesisConfig config;
    const auto fill = config.fill(999);

    EXPECT_EQ(fill.kind, FillSpec::Kind::FREE);
    EXPECT_FLOAT_EQ(fill.lo, -1.0f);
    EXPECT_FLOAT_EQ(fill.hi, 1.0f);
}

// ── unfilled only checks ownedUids ──────────────────────────────────────────

TEST(TestSynthesisConfig, UnfilledReportsStructuredAndDerived)
{
    SynthesisConfig config;
    config.set(1, FillSpec::free(-1.0f, 1.0f));
    config.set(2, FillSpec::structured());
    config.set(3, FillSpec::derived());
    config.set(4, FillSpec::fixed(0.5f));

    const auto missing = config.unfilled({1, 2, 3, 4});
    EXPECT_EQ(missing.size(), 2u);
    EXPECT_NE(std::find(missing.begin(), missing.end(), 2), missing.end());
    EXPECT_NE(std::find(missing.begin(), missing.end(), 3), missing.end());
}

TEST(TestSynthesisConfig, UnfilledIgnoresNonOwnedUids)
{
    SynthesisConfig config;
    config.set(1, FillSpec::free(-1.0f, 1.0f));
    config.set(2, FillSpec::structured());

    // Only check uid 1 — uid 2 (structured) is not owned, should be ignored
    const auto missing = config.unfilled({1});
    EXPECT_TRUE(missing.empty());
}

// ── seed resolution ─────────────────────────────────────────────────────────

TEST(TestSynthesisConfig, ResolveSeedPerTensor)
{
    SynthesisConfig config;
    config.setSeed(1, 100);

    EXPECT_EQ(config.resolveSeed(1), 100u);
    EXPECT_EQ(config.resolveSeed(2), std::nullopt);
}

TEST(TestSynthesisConfig, GlobalSeedDefaultValue)
{
    const SynthesisConfig config;
    EXPECT_EQ(config.globalSeed(), 42u);
}

TEST(TestSynthesisConfig, GlobalSeedCanBeSet)
{
    SynthesisConfig config;
    config.setGlobalSeed(123);
    EXPECT_EQ(config.globalSeed(), 123u);
}

// ── seed() does not block setDefault() (regression for the footgun) ─────────

TEST(TestSynthesisConfig, SeedThenSetDefaultStillShowsUnfilled)
{
    SynthesisConfig config;
    config.setSeed(1, 100);
    config.setDefault(1, FillSpec::derived());

    const auto missing = config.unfilled({1});
    EXPECT_EQ(missing.size(), 1u);
    EXPECT_EQ(missing[0], 1);
}

// ── JSON round-trip ─────────────────────────────────────────────────────────

TEST(TestSynthesisConfig, ToJsonAndLoadFromJsonRoundTrip)
{
    SynthesisConfig original;
    original.set(1, FillSpec::free(-2.0f, 2.0f));
    original.set(2, FillSpec::fixed(0.5f));
    original.set(3, FillSpec::structured());
    original.setSeed(1, 42);

    const auto json = original.toJson();

    // Parse back into uid→json map (same format as BundleMetadata::inputs)
    std::unordered_map<int64_t, nlohmann::json> inputMap;
    for(const auto& [key, val] : json.items())
    {
        inputMap[std::stoll(key)] = val;
    }

    SynthesisConfig loaded;
    loaded.loadFromJson(inputMap);

    // Verify fills
    const auto f1 = loaded.fill(1);
    EXPECT_EQ(f1.kind, FillSpec::Kind::FREE);
    EXPECT_FLOAT_EQ(f1.lo, -2.0f);
    EXPECT_FLOAT_EQ(f1.hi, 2.0f);

    const auto f2 = loaded.fill(2);
    EXPECT_EQ(f2.kind, FillSpec::Kind::FIXED);
    EXPECT_FLOAT_EQ(f2.value, 0.5f);

    const auto f3 = loaded.fill(3);
    EXPECT_EQ(f3.kind, FillSpec::Kind::STRUCTURED);

    // Verify seed survived
    EXPECT_EQ(loaded.resolveSeed(1), 42u);
    EXPECT_EQ(loaded.resolveSeed(2), std::nullopt);
}

TEST(TestSynthesisConfig, SeedOnlyTensorSurvivesRoundTrip)
{
    SynthesisConfig original;
    original.set(1, FillSpec::free(-1.0f, 1.0f));
    original.setSeed(1, 10);
    original.setSeed(2, 20);

    const auto json = original.toJson();

    std::unordered_map<int64_t, nlohmann::json> inputMap;
    for(const auto& [key, val] : json.items())
    {
        inputMap[std::stoll(key)] = val;
    }

    SynthesisConfig loaded;
    loaded.loadFromJson(inputMap);

    EXPECT_EQ(loaded.resolveSeed(1), 10u);
    EXPECT_EQ(loaded.resolveSeed(2), 20u);
    EXPECT_EQ(loaded.fills().count(1), 1u);
    EXPECT_EQ(loaded.fills().count(2), 0u);
}
