// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "harness/input-init/InputFillRecipes.hpp"

using namespace hipdnn_integration_tests;

namespace
{

std::unordered_map<int64_t, nlohmann::json> toInputMap(const nlohmann::json& inputs)
{
    std::unordered_map<int64_t, nlohmann::json> inputMap;
    for(const auto& [key, val] : inputs.items())
    {
        inputMap[std::stoll(key)] = val;
    }
    return inputMap;
}

// Writes `original` as metadata text and reads it back, as capture and load do.
InputFillRecipes roundTrip(const InputFillRecipes& original)
{
    InputFillRecipes loaded;
    loaded.loadFromJson(toInputMap(nlohmann::json::parse(original.toJson().dump())));
    return loaded;
}

FillRecipe parseRecipe(const char* specText)
{
    auto spec = parseInputFillSpec(nlohmann::json::parse(specText));
    if(!spec.recipe.has_value())
    {
        throw std::logic_error("spec has no recipe");
    }
    return std::move(*spec.recipe);
}

} // namespace

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

// ── setTypeDefault (data-type tier) ─────────────────────────────────────────

// The data-type tier sits below set() and setDefault() whichever is written
// first, so a default derived from fill(x) mid-pass sees x's final range.
TEST(TestInputFillRecipes, TypeDefaultLosesToOpDefaultInEitherOrder)
{
    const auto type = FillRecipe::free(-6.0f, 6.0f);
    const auto op = FillRecipe::free(-2.0f, 2.0f);

    InputFillRecipes typeFirst;
    typeFirst.setTypeDefault(1, type);
    EXPECT_EQ(typeFirst.fill(1), type);
    typeFirst.setDefault(1, op);
    EXPECT_EQ(typeFirst.fill(1), op);

    InputFillRecipes opFirst;
    opFirst.setDefault(1, op);
    opFirst.setTypeDefault(1, type);
    EXPECT_EQ(opFirst.fill(1), op);
}

TEST(TestInputFillRecipes, TypeDefaultLosesToSet)
{
    InputFillRecipes recipes;
    recipes.setTypeDefault(1, FillRecipe::free(-6.0f, 6.0f));
    recipes.set(1, FillRecipe::free(-3.0f, 3.0f));

    EXPECT_EQ(recipes.fill(1), FillRecipe::free(-3.0f, 3.0f));
}

// fills() and toJson() report what fill() resolves, across both tiers.
TEST(TestInputFillRecipes, FillsAndJsonResolveAcrossTiers)
{
    InputFillRecipes recipes;
    recipes.setTypeDefault(1, FillRecipe::free(-6.0f, 6.0f));
    recipes.setTypeDefault(2, FillRecipe::free(-6.0f, 6.0f));
    recipes.setDefault(2, FillRecipe::free(-2.0f, 2.0f));

    const auto fills = recipes.fills();
    ASSERT_EQ(fills.size(), 2u);
    EXPECT_EQ(fills.at(1), FillRecipe::free(-6.0f, 6.0f));
    EXPECT_EQ(fills.at(2), FillRecipe::free(-2.0f, 2.0f));

    const auto json = recipes.toJson();
    EXPECT_FLOAT_EQ(json.at("1").at("hi").get<float>(), 6.0f);
    EXPECT_FLOAT_EQ(json.at("2").at("hi").get<float>(), 2.0f);
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

// ── parseInputFillSpec ──────────────────────────────────────────────────────

TEST(TestInputFillRecipes, LoadFromJsonThrowsOnUnknownKind)
{
    InputFillRecipes recipes;
    const std::unordered_map<int64_t, nlohmann::json> inputs
        = {{7, nlohmann::json::parse(R"({"kind": "vaules", "values": [1]})")}};

    try
    {
        recipes.loadFromJson(inputs);
        FAIL() << "expected std::invalid_argument";
    }
    catch(const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("vaules"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(recipes.fills().empty());
}

TEST(TestInputFillRecipes, ParseInputFillSpecReadsFreeAndFixed)
{
    EXPECT_EQ(parseRecipe(R"({"kind": "free", "lo": -2, "hi": 3.5})"),
              FillRecipe::free(-2.0f, 3.5f));
    EXPECT_EQ(parseRecipe(R"({"kind": "fixed", "value": 0.25})"), FillRecipe::fixed(0.25f));
}

TEST(TestInputFillRecipes, ParseInputFillSpecWithoutKindSetsOnlyASeed)
{
    const auto spec = parseInputFillSpec(nlohmann::json::parse(R"({"seed": 9})"));
    EXPECT_FALSE(spec.recipe.has_value());
    EXPECT_EQ(spec.seed, 9u);
}

TEST(TestInputFillRecipes, ParseInputFillSpecRejectsAMalformedSpec)
{
    for(const auto* text : {R"("free")",
                            R"([1, 2])",
                            R"({"kind": 1})",
                            R"({"kind": null})",
                            R"({"kind": "FREE"})",
                            R"({"kind": ""})",
                            R"({"sedd": 9})",
                            R"({"kind": "fixed", "value": 1, "mistake": true})"})
    {
        EXPECT_THROW(parseInputFillSpec(nlohmann::json::parse(text)), std::invalid_argument)
            << text;
    }
}

// ── VALUES kind ─────────────────────────────────────────────────────────────

TEST(TestInputFillRecipes, ValuesOfIntegersAreStoredAsInt64)
{
    const auto recipe = parseRecipe(R"({"kind": "values", "values": [64, -32, 0]})");

    EXPECT_EQ(recipe.kind, FillRecipe::Kind::VALUES);
    EXPECT_TRUE(recipe.repeat);
    ASSERT_NE(recipe.list, nullptr);
    EXPECT_EQ(std::get<std::vector<int64_t>>(*recipe.list), (std::vector<int64_t>{64, -32, 0}));
}

TEST(TestInputFillRecipes, ValuesWithAnyNonIntegerAreStoredAsDouble)
{
    const auto recipe
        = parseRecipe(R"({"kind": "values", "values": [1, 2.5, -3], "repeat": false})");

    EXPECT_FALSE(recipe.repeat);
    ASSERT_NE(recipe.list, nullptr);
    EXPECT_EQ(std::get<std::vector<double>>(*recipe.list), (std::vector<double>{1.0, 2.5, -3.0}));
}

TEST(TestInputFillRecipes, ValuesAcceptASeed)
{
    const auto spec = parseInputFillSpec(
        nlohmann::json::parse(R"({"kind": "values", "values": [1], "seed": 4})"));
    ASSERT_TRUE(spec.recipe.has_value());
    EXPECT_EQ(*spec.recipe, FillRecipe::values(std::vector<int64_t>{1}));
    EXPECT_EQ(spec.seed, 4u);
}

TEST(TestInputFillRecipes, ValuesRejectAMalformedSpec)
{
    for(const auto* text : {
            R"({"kind": "values"})",
            R"({"kind": "values", "values": []})",
            R"({"kind": "values", "values": 3})",
            R"({"kind": "values", "values": ["64"]})",
            R"({"kind": "values", "values": [true]})",
            R"({"kind": "values", "values": [null]})",
            R"({"kind": "values", "values": [[1, 2]]})",
            R"({"kind": "values", "values": [{"v": 1}]})",
            R"({"kind": "values", "values": [1], "repeat": "true"})",
            R"({"kind": "values", "values": [1], "repeat": 1})",
            R"({"kind": "values", "values": [1], "lo": 0})",
            R"({"kind": "values", "values": [1], "repet": false})",
            R"({"kind": "values", "values": [9223372036854775808]})",
            R"({"kind": "values", "values": [18446744073709551615]})",
        })
    {
        EXPECT_THROW(parseInputFillSpec(nlohmann::json::parse(text)), std::invalid_argument)
            << text;
    }
}

TEST(TestInputFillRecipes, ValuesKeepTheExtremesOfInt64)
{
    const auto recipe = parseRecipe(
        R"({"kind": "values", "values": [9223372036854775807, -9223372036854775808]})");
    ASSERT_NE(recipe.list, nullptr);
    EXPECT_EQ(std::get<std::vector<int64_t>>(*recipe.list),
              (std::vector<int64_t>{std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::min()}));
}

TEST(TestInputFillRecipes, ValuesRoundTripThroughJson)
{
    constexpr int64_t K_TWO_TO_53_PLUS_ONE = (int64_t{1} << 53) + 1;

    InputFillRecipes original;
    original.set(1, FillRecipe::values(std::vector<int64_t>{64, 32}));
    original.set(2, FillRecipe::values(std::vector<int64_t>{K_TWO_TO_53_PLUS_ONE, -7}, false));
    original.set(3, FillRecipe::values(std::vector<double>{1.0, 2.5}, false));
    original.setSeed(1, 11);

    const auto json = original.toJson();
    EXPECT_EQ(json.at("1").at("kind").get<std::string>(), "values");
    EXPECT_TRUE(json.at("1").at("values").at(0).is_number_integer());
    EXPECT_TRUE(json.at("3").at("values").at(0).is_number_float());

    const auto loaded = roundTrip(original);
    EXPECT_EQ(loaded.fill(1), original.fill(1));
    EXPECT_EQ(loaded.fill(2), original.fill(2));
    EXPECT_EQ(loaded.fill(3), original.fill(3));
    EXPECT_EQ(loaded.resolveSeed(1), 11u);
    EXPECT_EQ(std::get<std::vector<int64_t>>(*loaded.fill(2).list).front(), K_TWO_TO_53_PLUS_ONE);
}

TEST(TestInputFillRecipes, RequiredFormatVersionIsTwoOnlyWithValues)
{
    InputFillRecipes recipes;
    EXPECT_EQ(requiredFormatVersion(recipes), 1);

    recipes.set(1, FillRecipe::free(-2.0f, 2.0f));
    recipes.set(2, FillRecipe::fixed(1.0f));
    recipes.setSeed(3, 5);
    EXPECT_EQ(requiredFormatVersion(recipes), 1);

    recipes.setTypeDefault(4, FillRecipe::values(std::vector<int64_t>{1}));
    EXPECT_EQ(requiredFormatVersion(recipes), 2);
}

// ── FillRecipe value semantics with a VALUES list ───────────────────────────

TEST(TestInputFillRecipes, ValuesRejectsAnEmptyList)
{
    EXPECT_THROW(FillRecipe::values(std::vector<int64_t>{}, /*repeat=*/true),
                 std::invalid_argument);
    EXPECT_THROW(FillRecipe::values(std::vector<double>{}, /*repeat=*/true), std::invalid_argument);
}

TEST(TestInputFillRecipes, ValuesRecipesCompareByContent)
{
    const auto a = FillRecipe::values(std::vector<int64_t>{1, 2});
    const auto b = FillRecipe::values(std::vector<int64_t>{1, 2});
    ASSERT_NE(a.list, b.list);
    EXPECT_EQ(a, b);

    EXPECT_NE(a, FillRecipe::values(std::vector<int64_t>{1, 3}));
    EXPECT_NE(a, FillRecipe::values(std::vector<int64_t>{1, 2}, false));
    EXPECT_NE(a, FillRecipe::values(std::vector<double>{1.0, 2.0}));
    EXPECT_NE(a, FillRecipe{});

    FillRecipe noList = a;
    noList.list = nullptr;
    FillRecipe otherNoList = b;
    otherNoList.list = nullptr;
    EXPECT_EQ(noList, otherNoList);
    EXPECT_NE(noList, a);
}

TEST(TestInputFillRecipes, ValuesRecipesCopyAndMove)
{
    const auto original = FillRecipe::values(std::vector<int64_t>{5, 6, 7}, false);

    FillRecipe copied(original);
    EXPECT_EQ(copied, original);
    copied.repeat = true;
    EXPECT_NE(copied, original);
    EXPECT_FALSE(original.repeat);

    FillRecipe copyAssigned;
    copyAssigned = original;
    EXPECT_EQ(copyAssigned, original);

    FillRecipe source = original;
    const FillRecipe moved(std::move(source));
    EXPECT_EQ(moved, original);
    source = FillRecipe::fixed(1.0f);
    EXPECT_EQ(source, FillRecipe::fixed(1.0f));

    FillRecipe moveSource = original;
    FillRecipe moveAssigned;
    moveAssigned = std::move(moveSource);
    EXPECT_EQ(moveAssigned, original);
    moveSource = original;
    EXPECT_EQ(moveSource, original);

    InputFillRecipes recipes;
    recipes.set(1, original);
    const auto fromRecipes = recipes.fill(1);
    EXPECT_EQ(fromRecipes, original);
}
