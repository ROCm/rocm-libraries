// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <hipdnn_plugin_sdk/ingestor/JsonDataSource.hpp>
#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>

namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

using json = nlohmann::json;
using V = jexpr::Value;

static_assert(!std::is_constructible_v<jexpr::JsonDataSource, json, char>,
              "JsonDataSource uses the shared variable sigil and has no custom sigil API");

// ---------------------------------------------------------------------------
// JsonDataSource: the sample nlohmann::json-backed data source.
// ---------------------------------------------------------------------------
TEST(TestJsonDataSource, GetResolvesPathsAndSubscripts)
{
    const jexpr::JsonDataSource src{json{{"q", {{"dims", {8, 16}}}},
                                         {"rows", {{{"name", "a0"}}, {{"name", "a1"}}}},
                                         {"grid", {{1, 2}, {3, 4}}}}};
    EXPECT_EQ(src.getData("q.dims"), V(V::Array{V(8), V(16)}));
    EXPECT_EQ(src.getData("q.dims[1]"), V(16));
    EXPECT_EQ(src.getData("q.dims.0"), V(8)); // dot-form index against an array
    EXPECT_EQ(src.getData("rows[1].name"), V("a1"));
    EXPECT_EQ(src.getData("grid[0][1]"), V(2));
    EXPECT_EQ(src.getData(""), V()); // an empty path names nothing
    // Unresolved paths read as null.
    EXPECT_EQ(src.getData("q.nope"), V());
    EXPECT_EQ(src.getData("q.dims[9]"), V());
    EXPECT_EQ(src.getData("q.dims[x]"), V());
    EXPECT_EQ(src.getData("q.dims[99999999999999999999]"), V()); // index too long to name a slot
    EXPECT_EQ(src.getData("q.dims["), V()); // malformed subscript
}

TEST(TestJsonDataSource, UsesFixedVariableSigil)
{
    const jexpr::JsonDataSource src{json{{"q", {{"dims", {8, 16}}}}}};
    EXPECT_EQ(src.getData("$q.dims[0]"), src.getData("q.dims[0]"));
    EXPECT_EQ(src.getData("$q.dims[0]"), V(8));
}

TEST(TestJsonDataSource, GetRejectsLeadingDotPaths)
{
    const jexpr::JsonDataSource src{json{{"q", 1}}};
    EXPECT_EQ(src.getData(".q"), V());
    EXPECT_EQ(src.getData("$.q"), V());
}

TEST(TestJsonDataSource, GetDeclinesUnsignedIntegersOutsideInt64Range)
{
    constexpr auto MAX_INT64 = std::numeric_limits<std::int64_t>::max();
    const auto maxUnsigned = static_cast<std::uint64_t>(MAX_INT64);
    const jexpr::JsonDataSource src{json{{"max", maxUnsigned},
                                         {"tooLarge", maxUnsigned + 1U},
                                         {"arr", json::array({maxUnsigned + 1U})}}};

    EXPECT_EQ(src.getData("max"), V(MAX_INT64));
    EXPECT_EQ(src.getData("tooLarge"), V());
    EXPECT_EQ(src.getData("arr[0]"), V());
}

namespace
{
/// A document whose `deep` member is an array nested `levels` deep, with 1 at
/// the centre. `levels == 1` is `{"deep": [1]}`.
json nestedArrayDocument(std::size_t levels)
{
    json inner = 1;
    for(std::size_t i = 0; i < levels; ++i)
    {
        inner = json::array({inner});
    }
    return json{{"deep", inner}};
}

/// Follow `levels` subscripts down that document.
std::string nestedArrayPath(std::size_t levels)
{
    std::string path = "deep";
    for(std::size_t i = 0; i < levels; ++i)
    {
        path += "[0]";
    }
    return path;
}
} // namespace

TEST(TestJsonDataSource, GetDeclinesDocumentsNestedPastTheValueDepthBound)
{
    // A rule is depth-checked at compile time; a document is not, and both are
    // read off disk. Every consumer of a Value recurses over it, so a document
    // deeper than the bound would overflow the stack rather than report a
    // value this source could not represent.
    //
    // The root Value counts as level 1, so a document with MAX_VALUE_DEPTH - 1
    // array levels beneath `deep` is the deepest that resolves.
    constexpr std::size_t DEEPEST = jexpr::MAX_VALUE_DEPTH - 1;

    const jexpr::JsonDataSource atBound{nestedArrayDocument(DEEPEST)};
    EXPECT_FALSE(atBound.getData("deep").containsUnresolved());
    EXPECT_EQ(atBound.getData(nestedArrayPath(DEEPEST)), V(1));

    // One level deeper, the array at the bound reads as null instead of
    // recursing, and null is contagious, so the whole value is unresolved.
    const jexpr::JsonDataSource pastBound{nestedArrayDocument(DEEPEST + 1)};
    EXPECT_TRUE(pastBound.getData("deep").containsUnresolved());

    // A path that walks past the bound in the *document* still resolves,
    // because the path walk is iterative: only the Value built at the end is
    // bounded, and that element is shallow.
    EXPECT_EQ(pastBound.getData(nestedArrayPath(DEEPEST + 1)), V(1));
}

TEST(TestJsonDataSource, OverDeepDocumentsDeclineRatherThanWidenACriterion)
{
    // The decline must survive negation. If the over-deep read answered a
    // plain false, a `!` around it would turn "never read" into a pass.
    const jexpr::JsonDataSource src{nestedArrayDocument(jexpr::MAX_VALUE_DEPTH)};
    const auto ev
        = [&src](const json& rule) { return jexpr::compile<jexpr::JsonDataSource>(rule)(src); };

    const json equality = json({{"==", json::array({"$deep", json::array({1})})}});
    EXPECT_TRUE(ev(equality).isNull());
    EXPECT_TRUE(ev(json({{"!", json::array({equality})}})).isNull());

    // Neither presence operator claims it either: a value with a hole in it is
    // neither wholly supplied nor wholly absent.
    EXPECT_EQ(ev(json({{"present", json::array({"$deep"})}})), V(false));
    EXPECT_EQ(ev(json({{"not_present", json::array({"$deep"})}})), V(false));
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
