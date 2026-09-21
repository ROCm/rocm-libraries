// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/input-init/InputFillRecipes.hpp"

using namespace hipdnn_integration_tests;

// ── setDefault (declaration functions) ──────────────────────────────────────

TEST(TestInputFillRecipes, SetDefaultWritesWhenEmpty)
{
    InputFillRecipes recipes;
    recipes.setDefault(1, FillRecipe::free(-1.0f, 1.0f));

    const auto fill = recipes.fill(1);
    EXPECT_EQ(fill.kind, FillRecipe::Kind::FREE);
    EXPECT_FLOAT_EQ(fill.lo, -1.0f);
    EXPECT_FLOAT_EQ(fill.hi, 1.0f);
}

TEST(TestInputFillRecipes, SetDefaultDoesNotOverwrite)
{
    InputFillRecipes recipes;
    recipes.setDefault(1, FillRecipe::free(-1.0f, 1.0f));
    recipes.setDefault(1, FillRecipe::free(-99.0f, 99.0f));

    const auto fill = recipes.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -1.0f);
    EXPECT_FLOAT_EQ(fill.hi, 1.0f);
}

// ── set (metadata / test code) ──────────────────────────────────────────────

TEST(TestInputFillRecipes, SetOverwritesDefault)
{
    InputFillRecipes recipes;
    recipes.setDefault(1, FillRecipe::free(-1.0f, 1.0f));
    recipes.set(1, FillRecipe::free(-5.0f, 5.0f));

    const auto fill = recipes.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -5.0f);
    EXPECT_FLOAT_EQ(fill.hi, 5.0f);
}

TEST(TestInputFillRecipes, SetOverwritesSet)
{
    InputFillRecipes recipes;
    recipes.set(1, FillRecipe::free(-5.0f, 5.0f));
    recipes.set(1, FillRecipe::free(-10.0f, 10.0f));

    const auto fill = recipes.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -10.0f);
    EXPECT_FLOAT_EQ(fill.hi, 10.0f);
}

TEST(TestInputFillRecipes, SetDefaultDoesNotOverwriteSet)
{
    InputFillRecipes recipes;
    recipes.set(1, FillRecipe::free(-5.0f, 5.0f));
    recipes.setDefault(1, FillRecipe::free(-99.0f, 99.0f));

    const auto fill = recipes.fill(1);
    EXPECT_FLOAT_EQ(fill.lo, -5.0f);
    EXPECT_FLOAT_EQ(fill.hi, 5.0f);
}

// ── Three-tier precedence (the real contract) ───────────────────────────────

TEST(TestInputFillRecipes, ThreeTierPrecedence)
{
    InputFillRecipes recipes;

    // 1. Metadata sets a range (runs first via setBundle)
    recipes.set(1, FillRecipe::free(-1.0f, 1.0f));

    // 2. Declaration function tries to set a default (emplace, should lose)
    recipes.setDefault(1, FillRecipe::free(-99.0f, 99.0f));

    // 3. Test body overwrites with its own range (runs after metadata)
    recipes.set(1, FillRecipe::free(-10.0f, 10.0f));

    const auto fill = recipes.fill(1);
    EXPECT_EQ(fill.kind, FillRecipe::Kind::FREE);
    EXPECT_FLOAT_EQ(fill.lo, -10.0f);
    EXPECT_FLOAT_EQ(fill.hi, 10.0f);
}

// ── get returns default-constructed FillRecipe for unknown uid ────────────────

TEST(TestInputFillRecipes, GetUnknownUidReturnsDefault)
{
    const InputFillRecipes recipes;
    const auto fill = recipes.fill(999);

    EXPECT_EQ(fill.kind, FillRecipe::Kind::FREE);
    EXPECT_FLOAT_EQ(fill.lo, -1.0f);
    EXPECT_FLOAT_EQ(fill.hi, 1.0f);
}

// ── seed resolution ─────────────────────────────────────────────────────────

TEST(TestInputFillRecipes, ResolveSeedPerTensor)
{
    InputFillRecipes recipes;
    recipes.setSeed(1, 100);

    EXPECT_EQ(recipes.resolveSeed(1), 100u);
    EXPECT_EQ(recipes.resolveSeed(2), std::nullopt);
}

TEST(TestInputFillRecipes, GlobalSeedDefaultValue)
{
    const InputFillRecipes recipes;
    EXPECT_EQ(recipes.globalSeed(), 42u);
}

TEST(TestInputFillRecipes, GlobalSeedCanBeSet)
{
    InputFillRecipes recipes;
    recipes.setGlobalSeed(123);
    EXPECT_EQ(recipes.globalSeed(), 123u);
}

// ── seed() does not block setDefault() (regression for the footgun) ─────────

// ── JSON round-trip ─────────────────────────────────────────────────────────

TEST(TestInputFillRecipes, ToJsonAndLoadFromJsonRoundTrip)
{
    InputFillRecipes original;
    original.set(1, FillRecipe::free(-2.0f, 2.0f));
    original.set(2, FillRecipe::fixed(0.5f));
    original.setSeed(1, 42);

    const auto json = original.toJson();

    // Parse back into uid→json map (same format as BundleMetadata::inputs)
    std::unordered_map<int64_t, nlohmann::json> inputMap;
    for(const auto& [key, val] : json.items())
    {
        inputMap[std::stoll(key)] = val;
    }

    InputFillRecipes loaded;
    loaded.loadFromJson(inputMap);

    // Verify fills
    const auto f1 = loaded.fill(1);
    EXPECT_EQ(f1.kind, FillRecipe::Kind::FREE);
    EXPECT_FLOAT_EQ(f1.lo, -2.0f);
    EXPECT_FLOAT_EQ(f1.hi, 2.0f);

    const auto f2 = loaded.fill(2);
    EXPECT_EQ(f2.kind, FillRecipe::Kind::FIXED);
    EXPECT_FLOAT_EQ(f2.value, 0.5f);

    // Verify seed survived
    EXPECT_EQ(loaded.resolveSeed(1), 42u);
    EXPECT_EQ(loaded.resolveSeed(2), std::nullopt);
}

TEST(TestInputFillRecipes, SeedOnlyTensorSurvivesRoundTrip)
{
    InputFillRecipes original;
    original.set(1, FillRecipe::free(-1.0f, 1.0f));
    original.setSeed(1, 10);
    original.setSeed(2, 20);

    const auto json = original.toJson();

    std::unordered_map<int64_t, nlohmann::json> inputMap;
    for(const auto& [key, val] : json.items())
    {
        inputMap[std::stoll(key)] = val;
    }

    InputFillRecipes loaded;
    loaded.loadFromJson(inputMap);

    EXPECT_EQ(loaded.resolveSeed(1), 10u);
    EXPECT_EQ(loaded.resolveSeed(2), 20u);
    EXPECT_EQ(loaded.fills().count(1), 1u);
    EXPECT_EQ(loaded.fills().count(2), 0u);
}
