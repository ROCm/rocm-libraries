// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <limits>
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
    constexpr auto maxInt64 = std::numeric_limits<std::int64_t>::max();
    const auto maxUnsigned = static_cast<std::uint64_t>(maxInt64);
    const jexpr::JsonDataSource src{json{{"max", maxUnsigned},
                                         {"tooLarge", maxUnsigned + 1U},
                                         {"arr", json::array({maxUnsigned + 1U})}}};

    EXPECT_EQ(src.getData("max"), V(maxInt64));
    EXPECT_EQ(src.getData("tooLarge"), V());
    EXPECT_EQ(src.getData("arr[0]"), V());
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
