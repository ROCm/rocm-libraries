// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "harness/input-init/InputFillRecipes.hpp"

using namespace hipdnn_integration_tests;

namespace
{

using InputMap = std::unordered_map<int64_t, nlohmann::json>;

InputMap toInputMap(const nlohmann::json& json)
{
    InputMap inputMap;
    for(const auto& [key, val] : json.items())
    {
        inputMap[std::stoll(key)] = val;
    }
    return inputMap;
}

// The recipe loadFromJson makes of one entry, filed under uid 7.
FillRecipe loadOne(const std::string& spec)
{
    InputMap inputs;
    inputs[7] = nlohmann::json::parse(spec);
    InputFillRecipes recipes;
    recipes.loadFromJson(inputs);
    return recipes.fill(7);
}

// What loadFromJson says about one entry filed under uid 7, or "" when it loads.
std::string loadError(const std::string& spec, const std::string& where = "case X")
{
    InputMap inputs;
    inputs[7] = nlohmann::json::parse(spec);
    InputFillRecipes recipes;
    try
    {
        recipes.loadFromJson(inputs, where);
    }
    catch(const FillRecipeError& e)
    {
        return e.what();
    }
    return {};
}

bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
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

// ── New kinds: JSON ─────────────────────────────────────────────────────────

// A kind that serializes under the wrong name, or loses a parameter on the way out or back,
// changes what a recaptured bundle fills with. Defaults and non-default parameters both.
TEST(TestInputFillRecipes, EveryKindRoundTripsThroughJson)
{
    using Special = FillRecipe::SpecialValue;
    const std::vector<FillRecipe> recipes
        = {FillRecipe::free(-2.0f, 2.0f),
           FillRecipe::fixed(0.5f),
           FillRecipe::nonzero(-3.0f, 5.0f),
           FillRecipe::zeros(),
           FillRecipe::large(100.0f),
           FillRecipe::subnormal(),
           FillRecipe::nonFinite(Special::QUIET_NAN, 0.25f),
           FillRecipe::nonFinite(Special::POS_INF, 1.0f, -2.0f, 3.0f),
           FillRecipe::nonFinite(Special::NEG_INF, 0.0f)};

    InputFillRecipes original;
    for(std::size_t i = 0; i < recipes.size(); ++i)
    {
        const auto uid = static_cast<int64_t>(i + 1);
        original.set(uid, recipes[i]);
        if(i % 2 == 1)
        {
            original.setSeed(uid, static_cast<unsigned int>(100 + i));
        }
    }

    InputFillRecipes loaded;
    loaded.loadFromJson(toInputMap(original.toJson()));

    for(std::size_t i = 0; i < recipes.size(); ++i)
    {
        const auto uid = static_cast<int64_t>(i + 1);
        SCOPED_TRACE(InputFillRecipes::kindToString(recipes[i].kind));
        EXPECT_EQ(loaded.fill(uid), recipes[i]);
        EXPECT_EQ(loaded.resolveSeed(uid),
                  i % 2 == 1 ? std::optional<unsigned int>(static_cast<unsigned int>(100 + i))
                             : std::nullopt);
    }
}

// The JSON a space renders is this schema, field for field; a kind that wrote extra or
// differently named fields would not be read by the harness it was meant for.
TEST(TestInputFillRecipes, JsonCarriesOnlyTheFieldsOfEachKind)
{
    using Special = FillRecipe::SpecialValue;
    struct Case
    {
        FillRecipe recipe;
        const char* json;
    };
    const std::vector<Case> cases = {
        {FillRecipe::free(-1.0f, 2.0f), R"({"kind":"free","lo":-1,"hi":2})"},
        {FillRecipe::fixed(0.5f), R"({"kind":"fixed","value":0.5})"},
        {FillRecipe::nonzero(-1.0f, 1.0f), R"({"kind":"nonzero","lo":-1,"hi":1})"},
        {FillRecipe::zeros(), R"({"kind":"zeros"})"},
        {FillRecipe::large(100.0f), R"({"kind":"large","scale":100})"},
        {FillRecipe::subnormal(), R"({"kind":"subnormal"})"},
        {FillRecipe::nonFinite(Special::QUIET_NAN, 0.25f),
         R"({"kind":"special","special":"nan","fraction":0.25,"lo":-1,"hi":1})"},
        {FillRecipe::nonFinite(Special::POS_INF, 0.5f, 2.0f, 3.0f),
         R"({"kind":"special","special":"inf","fraction":0.5,"lo":2,"hi":3})"},
        {FillRecipe::nonFinite(Special::NEG_INF, 1.0f),
         R"({"kind":"special","special":"-inf","fraction":1,"lo":-1,"hi":1})"},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.json);
        InputFillRecipes recipes;
        recipes.set(1, c.recipe);
        EXPECT_EQ(recipes.toJson().at("1"), nlohmann::json::parse(c.json));
    }
}

// The documented per-kind JSON, as a space writes it, loads to the recipe it names; absent
// optional fields take the recipe defaults, and free and fixed read as they always did.
TEST(TestInputFillRecipes, DocumentedJsonLoadsToTheRecipeItNames)
{
    using Special = FillRecipe::SpecialValue;
    EXPECT_EQ(loadOne(R"({"kind":"nonzero","lo":-1,"hi":1})"), FillRecipe::nonzero(-1.0f, 1.0f));
    EXPECT_EQ(loadOne(R"({"kind":"nonzero"})"), FillRecipe::nonzero(-1.0f, 1.0f));
    EXPECT_EQ(loadOne(R"({"kind":"zeros"})"), FillRecipe::zeros());
    EXPECT_EQ(loadOne(R"({"kind":"large","scale":100})"), FillRecipe::large(100.0f));
    EXPECT_EQ(loadOne(R"({"kind":"subnormal"})"), FillRecipe::subnormal());
    EXPECT_EQ(loadOne(R"({"kind":"special","special":"nan","fraction":0.25})"),
              FillRecipe::nonFinite(Special::QUIET_NAN, 0.25f));
    EXPECT_EQ(loadOne(R"({"kind":"special","special":"-inf","fraction":1,"lo":2,"hi":3})"),
              FillRecipe::nonFinite(Special::NEG_INF, 1.0f, 2.0f, 3.0f));

    EXPECT_EQ(loadOne(R"({"kind":"free","lo":-2,"hi":2,"seed":5})"), FillRecipe::free(-2.0f, 2.0f));
    EXPECT_EQ(loadOne(R"({"kind":"free"})"), FillRecipe{});
    EXPECT_EQ(loadOne(R"({"kind":"fixed","value":0.5})"), FillRecipe::fixed(0.5f));
}

// Each setter writes its own kind and parameters; one that wrote FREE would fill an override
// like any other tensor.
TEST(TestInputFillRecipes, SettersWriteTheirKind)
{
    InputFillRecipes recipes;
    recipes.setNonzero(1, -2.0f, 3.0f)
        .setZeros(2)
        .setLarge(3, 50.0f)
        .setSubnormal(4)
        .setNonFinite(5, FillRecipe::SpecialValue::NEG_INF, 0.5f, -4.0f, 4.0f);

    EXPECT_EQ(recipes.fill(1), FillRecipe::nonzero(-2.0f, 3.0f));
    EXPECT_EQ(recipes.fill(2), FillRecipe::zeros());
    EXPECT_EQ(recipes.fill(3), FillRecipe::large(50.0f));
    EXPECT_EQ(recipes.fill(4), FillRecipe::subnormal());
    EXPECT_EQ(recipes.fill(5),
              FillRecipe::nonFinite(FillRecipe::SpecialValue::NEG_INF, 0.5f, -4.0f, 4.0f));
}

// A kind added without a name would serialize as another kind; the names are the JSON's.
TEST(TestInputFillRecipes, KindNamesAreTheJsonNames)
{
    using Kind = FillRecipe::Kind;
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::FREE), "free");
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::FIXED), "fixed");
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::NONZERO), "nonzero");
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::ZEROS), "zeros");
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::LARGE), "large");
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::SUBNORMAL), "subnormal");
    EXPECT_STREQ(InputFillRecipes::kindToString(Kind::SPECIAL), "special");
}

// ── New kinds: strict parsing ───────────────────────────────────────────────

// A typo in `kind` used to fall back to FREE, so a "nonzero" case misspelled "non_zero" ran
// as a plain random fill and passed. It is an error that names the uid, the field and where
// the metadata came from; the old constant fill is `fixed`, and the message says so.
TEST(TestInputFillRecipes, UnknownKindIsAnErrorNamingTheUidAndTheKind)
{
    for(const std::string name : {"constant", "non_zero", "Free", "", "nan"})
    {
        SCOPED_TRACE(name);
        std::string spec = R"({"kind":")";
        spec += name;
        spec += R"("})";
        const auto message = loadError(spec);
        EXPECT_TRUE(contains(message, "uid 7")) << message;
        EXPECT_TRUE(contains(message, "case X")) << message;
        std::string expected = "unknown fill kind \"";
        expected += name;
        expected += '"';
        EXPECT_TRUE(contains(message, expected)) << message;
    }
    EXPECT_TRUE(contains(loadError(R"({"kind":"constant"})"), "\"fixed\""));
}

// The error is the recipe parser's own exception, which the metadata parser re-raises as a
// BundleMetadataError, so a bad case is reported as malformed metadata at load time.
TEST(TestInputFillRecipes, StrictErrorsAreFillRecipeErrors)
{
    InputMap inputs;
    inputs[1] = nlohmann::json::parse(R"({"kind":"bogus"})");
    InputFillRecipes recipes;

    EXPECT_THROW(recipes.loadFromJson(inputs), FillRecipeError);
    EXPECT_THROW(recipes.loadFromJson(inputs), std::runtime_error);
}

// A wrong-typed field the kind reads was silently dropped, leaving the default in its place;
// every one is now an error naming uid and field.
TEST(TestInputFillRecipes, WrongFieldTypesAreErrors)
{
    struct Case
    {
        const char* json;
        const char* field;
    };
    const std::vector<Case> cases = {
        {R"({"kind":"free","lo":"x"})", "\"lo\""},
        {R"({"kind":"free","hi":true})", "\"hi\""},
        {R"({"kind":"fixed","value":null})", "\"value\""},
        {R"({"kind":"large","scale":"big"})", "\"scale\""},
        {R"({"kind":"special","special":"nan","fraction":"half"})", "\"fraction\""},
        {R"({"kind":"nonzero","hi":"x"})", "\"hi\""},
        {R"({"kind":5})", "\"kind\""},
        {R"({"kind":null})", "\"kind\""},
        {R"({"kind":"free","lo":1e300})", "\"lo\""},
        {R"({"kind":"free","hi":-1e300})", "\"hi\""},
        {R"({"kind":"free","seed":-1})", "\"seed\""},
        {R"({"kind":"free","seed":1.5})", "\"seed\""},
        {R"({"kind":"free","seed":"7"})", "\"seed\""},
        {R"({"kind":"free","seed":4294967296})", "\"seed\""},
        {R"({"seed":-3})", "\"seed\""},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.json);
        const auto message = loadError(c.json);
        EXPECT_TRUE(contains(message, "uid 7")) << message;
        EXPECT_TRUE(contains(message, c.field)) << message;
    }

    for(const char* notAnObject : {R"("free")", "[1, 2]", "3", "null"})
    {
        SCOPED_TRACE(notAnObject);
        const auto message = loadError(notAnObject);
        EXPECT_TRUE(contains(message, "uid 7")) << message;
        EXPECT_TRUE(contains(message, "JSON object")) << message;
    }
}

// An unsigned 32-bit seed, the top one included, loads: the strictness must not reject what
// the harness itself writes.
TEST(TestInputFillRecipes, SeedsUpToTheUnsignedLimitLoad)
{
    InputMap inputs;
    inputs[1] = nlohmann::json::parse(R"({"seed":4294967295})");
    inputs[2] = nlohmann::json::parse(R"({"seed":0})");
    InputFillRecipes recipes;
    recipes.loadFromJson(inputs);

    EXPECT_EQ(recipes.resolveSeed(1), 4294967295u);
    EXPECT_EQ(recipes.resolveSeed(2), 0u);
}

// "special" names which non-finite value to plant and "fraction" how much of the tensor: a
// missing, misspelled or out-of-range one would plant nothing, or the wrong value, silently.
TEST(TestInputFillRecipes, SpecialFieldsAreValidated)
{
    struct Case
    {
        const char* json;
        const char* inMessage;
    };
    const std::vector<Case> cases = {
        {R"({"kind":"special","fraction":0.5})", "requires \"special\""},
        {R"({"kind":"special","special":1,"fraction":0.5})", "requires \"special\""},
        {R"({"kind":"special","special":"NaN","fraction":0.5})", "\"special\" must be one of"},
        {R"({"kind":"special","special":"+inf","fraction":0.5})", "\"special\" must be one of"},
        {R"({"kind":"special","special":"","fraction":0.5})", "\"special\" must be one of"},
        {R"({"kind":"special","special":"nan"})", "requires \"fraction\""},
        {R"({"kind":"special","special":"nan","fraction":1.5})", "\"fraction\" must be in [0, 1]"},
        {R"({"kind":"special","special":"nan","fraction":-0.1})", "\"fraction\" must be in [0, 1]"},
        {R"({"kind":"special","special":"nan","fraction":0.5,"lo":2,"hi":1})", "lo <= hi"},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.json);
        const auto message = loadError(c.json);
        EXPECT_TRUE(contains(message, "uid 7")) << message;
        EXPECT_TRUE(contains(message, c.inMessage)) << message;
    }

    // The bounds of the range are valid: a fraction of 0 or 1, and each spelling.
    for(const char* special : {"nan", "inf", "-inf"})
    {
        for(const char* fraction : {"0", "1"})
        {
            std::string spec = R"({"kind":"special","special":")";
            spec += special;
            spec += R"(","fraction":)";
            spec += fraction;
            spec += "}";
            EXPECT_EQ(loadError(spec), "") << spec;
        }
    }
}

// large needs a scale to be large relative to, and a positive one; zero or negative has no
// magnitude range.
TEST(TestInputFillRecipes, LargeNeedsAPositiveScale)
{
    EXPECT_TRUE(contains(loadError(R"({"kind":"large"})"), "requires \"scale\""));
    EXPECT_TRUE(contains(loadError(R"({"kind":"large","scale":0})"), "\"scale\" must be > 0"));
    EXPECT_TRUE(contains(loadError(R"({"kind":"large","scale":-3})"), "\"scale\" must be > 0"));
    EXPECT_EQ(loadError(R"({"kind":"large","scale":1e-30})"), "");
}

// A nonzero range has to be ordered for a draw to exist; a degenerate lo == hi is left for the
// fill to judge against the data type.
TEST(TestInputFillRecipes, NonzeroNeedsAnOrderedRange)
{
    EXPECT_TRUE(contains(loadError(R"({"kind":"nonzero","lo":2,"hi":-2})"), "lo <= hi"));
    EXPECT_EQ(loadError(R"({"kind":"nonzero","lo":3,"hi":3})"), "");
}

// One bad entry must leave the recipes as they were: applying the good entries first would
// leave a half-loaded case that then runs with some of its overrides missing.
TEST(TestInputFillRecipes, RejectedEntryLoadsNothing)
{
    InputMap inputs;
    inputs[1] = nlohmann::json::parse(R"({"kind":"zeros","seed":3})");
    inputs[2] = nlohmann::json::parse(R"({"kind":"large","scale":10})");
    inputs[3] = nlohmann::json::parse(R"({"kind":"special","special":"nan","fraction":2})");
    InputFillRecipes recipes;
    recipes.setFixed(9, 1.0f);

    EXPECT_THROW(recipes.loadFromJson(inputs), FillRecipeError);

    EXPECT_EQ(recipes.fills().size(), 1u);
    EXPECT_EQ(recipes.fill(9), FillRecipe::fixed(1.0f));
    EXPECT_EQ(recipes.resolveSeed(1), std::nullopt);
}

// A space renders every leaf (kind, lo, hi, value, scale) for every domain, so an entry carries
// fields its kind does not read. They must be ignored, whatever they hold, and the entry must
// behave per its kind: an extra field that was read (or rejected) would change the fill or
// fail a case that is well formed.
TEST(TestInputFillRecipes, FieldsTheKindDoesNotReadAreIgnored)
{
    EXPECT_EQ(loadOne(R"({"kind":"free","lo":-2,"hi":2,"value":7,"scale":-1,"fraction":9})"),
              FillRecipe::free(-2.0f, 2.0f));
    EXPECT_EQ(loadOne(R"({"kind":"fixed","value":0.5,"lo":5,"hi":-5,"scale":0})"),
              FillRecipe::fixed(0.5f));
    EXPECT_EQ(loadOne(R"({"kind":"zeros","lo":-2,"hi":-3,"value":1,"scale":0,"fraction":-1})"),
              FillRecipe::zeros());
    EXPECT_EQ(loadOne(R"({"kind":"zeros","lo":"x","value":null,"scale":"big"})"),
              FillRecipe::zeros());
    EXPECT_EQ(loadOne(R"({"kind":"subnormal","lo":5,"hi":1,"value":2,"scale":3})"),
              FillRecipe::subnormal());
    EXPECT_EQ(loadOne(R"({"kind":"large","scale":10,"lo":5,"hi":1,"value":2,"fraction":3})"),
              FillRecipe::large(10.0f));
    EXPECT_EQ(loadOne(R"({"kind":"nonzero","lo":-1,"hi":2,"value":3,"scale":0,"special":"x"})"),
              FillRecipe::nonzero(-1.0f, 2.0f));
}
