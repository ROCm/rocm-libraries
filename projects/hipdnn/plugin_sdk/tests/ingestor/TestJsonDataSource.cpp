// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <stdexcept>

#include <hipdnn_plugin_sdk/ingestor/JsonDataSource.hpp>
#include <hipdnn_plugin_sdk/ingestor/JsonExpression.hpp>

namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

using json = nlohmann::json;
using V = jexpr::Value;

// ---------------------------------------------------------------------------
// JsonDataSource: the sample nlohmann::json-backed data source (getData/setData).
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
    EXPECT_EQ(src.getData(""), V()); // whole document is an object -> null
    // unresolved paths read as null
    EXPECT_EQ(src.getData("q.nope"), V());
    EXPECT_EQ(src.getData("q.dims[9]"), V());
    EXPECT_EQ(src.getData("q.dims[x]"), V());
    EXPECT_EQ(src.getData("q.dims["), V()); // malformed subscript
}

TEST(TestJsonDataSource, GetStripsOptionalSigil)
{
    const jexpr::JsonDataSource src{json{{"q", {{"dims", {8, 16}}}}}};
    EXPECT_EQ(src.getData("$q.dims[0]"), src.getData("q.dims[0]"));
    EXPECT_EQ(src.getData("$q.dims[0]"), V(8));
}

TEST(TestJsonDataSource, SetScalarCreatesNestedObjects)
{
    jexpr::JsonDataSource src;
    src.setData("q.dims", V(4));
    EXPECT_EQ(src.getData("q.dims"), V(4));
    EXPECT_EQ(src.document(), (json{{"q", {{"dims", 4}}}}));
}

TEST(TestJsonDataSource, SetSubscriptCreatesAndExtendsArray)
{
    jexpr::JsonDataSource src;
    // [N] forces array creation; gaps fill with null.
    src.setData("q.dims[2]", V(7));
    EXPECT_EQ(src.document(), (json{{"q", {{"dims", {nullptr, nullptr, 7}}}}}));
    EXPECT_EQ(src.getData("q.dims[2]"), V(7));
    EXPECT_EQ(src.getData("q.dims[0]"), V()); // gap reads as null
}

TEST(TestJsonDataSource, SetOverwritesExistingArrayIndex)
{
    jexpr::JsonDataSource src{json{{"q", {{"dims", {8, 16}}}}}};
    src.setData("$q.dims[0]", V(2)); // the motivating example
    EXPECT_EQ(src.document(), (json{{"q", {{"dims", {2, 16}}}}}));
    EXPECT_EQ(src.getData("q.dims[0]"), V(2));
}

TEST(TestJsonDataSource, SetWholeDocument)
{
    jexpr::JsonDataSource src{json{{"x", 1}}};
    src.setData("", V(V::Array{V(1), V(2)}));
    EXPECT_EQ(src.document(), (json::array({1, 2})));
    EXPECT_EQ(src.getData("[1]"), V(2));
}

TEST(TestJsonDataSource, SetRoundTripsValueKinds)
{
    jexpr::JsonDataSource src;
    src.setData("b", V(true));
    src.setData("s", V("amd"));
    src.setData("d", V(1.5));
    src.setData("a", V(V::Array{V(1), V("two"), V(false)}));
    EXPECT_EQ(src.getData("b"), V(true));
    EXPECT_EQ(src.getData("s"), V("amd"));
    EXPECT_EQ(src.getData("d"), V(1.5));
    EXPECT_EQ(src.getData("a"), V(V::Array{V(1), V("two"), V(false)}));
}

TEST(TestJsonDataSource, SetThenEvaluateReflectsChange)
{
    jexpr::JsonDataSource src{json{{"q", {{"dims", {8, 16}}}}}};
    const auto expr = jexpr::compile<jexpr::JsonDataSource>(
        json({{"*", json::array({"$q.dims[0]", "$q.dims[1]"})}}));
    EXPECT_EQ(expr(src), V(128));
    src.setData("$q.dims[0]", V(2));
    EXPECT_EQ(expr(src), V(32));
}

TEST(TestJsonDataSource, SetRejectsMalformedPaths)
{
    jexpr::JsonDataSource src{json{{"q", {{"dims", {8, 16}}}}}};
    EXPECT_THROW(src.setData("q.dims[0", V(1)), std::invalid_argument); // missing ']'
    EXPECT_THROW(src.setData("q.dims[x]", V(1)), std::invalid_argument); // non-numeric index
    EXPECT_THROW(src.setData("q.dims.nope", V(1)), std::invalid_argument); // string key on array
}

TEST(TestJsonDataSource, SetRejectsAnOutOfBoundsIndex)
{
    // A `[N]` subscript grows the array to N, so an unbounded index turns a
    // typo in a descriptor path into an allocation of arbitrary size.
    jexpr::JsonDataSource src;
    EXPECT_THROW(src.setData("q[100000000]", V(1)), std::invalid_argument);
    EXPECT_THROW(src.setData("q[99999999999999999999]", V(1)), std::invalid_argument);
    // Nothing was written on the way to the rejection.
    EXPECT_EQ(src.getData("q"), V());

    // An index inside the bound still works, and getData resolves an
    // out-of-bounds index to null exactly as it does past the end.
    src.setData("q[3]", V(7));
    EXPECT_EQ(src.getData("q[3]"), V(7));
    EXPECT_EQ(src.getData("q[100000000]"), V());
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
