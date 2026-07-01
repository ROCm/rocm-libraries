// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/input_init/SynthesisConfig.hpp"

using namespace hipdnn_integration_tests;

// ── setDefault (declaration functions) ──────────────────────────────────────

TEST(TestSynthesisConfig, SetDefaultWritesWhenEmpty)
{
    SynthesisConfig config;
    config.setDefault(1, TensorInit::free(-1.0f, 1.0f));

    const auto init = config.get(1);
    EXPECT_EQ(init.kind, TensorInit::Kind::FREE);
    EXPECT_FLOAT_EQ(init.lo, -1.0f);
    EXPECT_FLOAT_EQ(init.hi, 1.0f);
}

TEST(TestSynthesisConfig, SetDefaultDoesNotOverwrite)
{
    SynthesisConfig config;
    config.setDefault(1, TensorInit::free(-1.0f, 1.0f));
    config.setDefault(1, TensorInit::free(-99.0f, 99.0f));

    const auto init = config.get(1);
    EXPECT_FLOAT_EQ(init.lo, -1.0f);
    EXPECT_FLOAT_EQ(init.hi, 1.0f);
}

// ── set (metadata / test code) ──────────────────────────────────────────────

TEST(TestSynthesisConfig, SetOverwritesDefault)
{
    SynthesisConfig config;
    config.setDefault(1, TensorInit::free(-1.0f, 1.0f));
    config.set(1, TensorInit::free(-5.0f, 5.0f));

    const auto init = config.get(1);
    EXPECT_FLOAT_EQ(init.lo, -5.0f);
    EXPECT_FLOAT_EQ(init.hi, 5.0f);
}

TEST(TestSynthesisConfig, SetOverwritesSet)
{
    SynthesisConfig config;
    config.set(1, TensorInit::free(-5.0f, 5.0f));
    config.set(1, TensorInit::free(-10.0f, 10.0f));

    const auto init = config.get(1);
    EXPECT_FLOAT_EQ(init.lo, -10.0f);
    EXPECT_FLOAT_EQ(init.hi, 10.0f);
}

TEST(TestSynthesisConfig, SetDefaultDoesNotOverwriteSet)
{
    SynthesisConfig config;
    config.set(1, TensorInit::free(-5.0f, 5.0f));
    config.setDefault(1, TensorInit::free(-99.0f, 99.0f));

    const auto init = config.get(1);
    EXPECT_FLOAT_EQ(init.lo, -5.0f);
    EXPECT_FLOAT_EQ(init.hi, 5.0f);
}

// ── Three-tier precedence (the real contract) ───────────────────────────────

TEST(TestSynthesisConfig, ThreeTierPrecedence)
{
    SynthesisConfig config;

    // 1. Metadata sets a range (runs first via setBundle)
    config.set(1, TensorInit::free(-1.0f, 1.0f));

    // 2. Declaration function tries to set a default (emplace, should lose)
    config.setDefault(1, TensorInit::free(-99.0f, 99.0f));

    // 3. Test body overwrites with its own range (runs after metadata)
    config.set(1, TensorInit::free(-10.0f, 10.0f));

    const auto init = config.get(1);
    EXPECT_EQ(init.kind, TensorInit::Kind::FREE);
    EXPECT_FLOAT_EQ(init.lo, -10.0f);
    EXPECT_FLOAT_EQ(init.hi, 10.0f);
}

// ── get returns default-constructed TensorInit for unknown uid ──────────────

TEST(TestSynthesisConfig, GetUnknownUidReturnsDefault)
{
    SynthesisConfig config;
    const auto init = config.get(999);

    EXPECT_EQ(init.kind, TensorInit::Kind::FREE);
    EXPECT_FLOAT_EQ(init.lo, -1.0f);
    EXPECT_FLOAT_EQ(init.hi, 1.0f);
}

// ── unfilled only checks ownedUids ──────────────────────────────────────────

TEST(TestSynthesisConfig, UnfilledReportsStructuredAndDerived)
{
    SynthesisConfig config;
    config.set(1, TensorInit::free(-1.0f, 1.0f));
    config.set(2, TensorInit::structured());
    config.set(3, TensorInit::derived());
    config.set(4, TensorInit::fixed(0.5f));

    const auto missing = config.unfilled({1, 2, 3, 4});
    EXPECT_EQ(missing.size(), 2u);
    EXPECT_NE(std::find(missing.begin(), missing.end(), 2), missing.end());
    EXPECT_NE(std::find(missing.begin(), missing.end(), 3), missing.end());
}

TEST(TestSynthesisConfig, UnfilledIgnoresNonOwnedUids)
{
    SynthesisConfig config;
    config.set(1, TensorInit::free(-1.0f, 1.0f));
    config.set(2, TensorInit::structured());

    // Only check uid 1 — uid 2 (structured) is not owned, should be ignored
    const auto missing = config.unfilled({1});
    EXPECT_TRUE(missing.empty());
}

// ── seed resolution ─────────────────────────────────────────────────────────

TEST(TestSynthesisConfig, ResolveSeedPerTensor)
{
    SynthesisConfig config;
    config.seed(1, 100);

    EXPECT_EQ(config.resolveSeed(1), 100u);
    EXPECT_EQ(config.resolveSeed(2), std::nullopt);
}

TEST(TestSynthesisConfig, FallbackSeedUsedWhenNoPerTensor)
{
    SynthesisConfig config;
    config.fallbackSeed(42);

    EXPECT_EQ(config.resolveSeed(1), 42u);
}

TEST(TestSynthesisConfig, PerTensorSeedBeatssFallback)
{
    SynthesisConfig config;
    config.fallbackSeed(42);
    config.seed(1, 100);

    EXPECT_EQ(config.resolveSeed(1), 100u);
    EXPECT_EQ(config.resolveSeed(2), 42u);
}
