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
    // Text between a ']' and the next separator is not a key: accepting it
    // would have setData silently create one the caller never wrote.
    EXPECT_THROW(src.setData("q.dims[0]bogus", V(1)), std::invalid_argument);
    EXPECT_THROW(src.setData("q..dims", V(1)), std::invalid_argument); // empty segment
    EXPECT_THROW(src.setData("q.", V(1)), std::invalid_argument); // trailing separator
    // strtol would accept both of these as 3; an index is digits only.
    EXPECT_THROW(src.setData("q.dims[ 1]", V(1)), std::invalid_argument);
    EXPECT_THROW(src.setData("q.dims[+1]", V(1)), std::invalid_argument);
    // Every rejection above left the document exactly as it was.
    EXPECT_EQ(src.document(), json({{"q", {{"dims", {8, 16}}}}}));
}

TEST(TestJsonDataSource, SetRejectsAnOutOfBoundsIndex)
{
    // A `[N]` subscript grows the array to N, so an unbounded index turns a
    // typo in a descriptor path into an allocation of arbitrary size.
    jexpr::JsonDataSource src;
    EXPECT_THROW(src.setData("q[100000000]", V(1)), std::invalid_argument);
    EXPECT_THROW(src.setData("q[99999999999999999999]", V(1)), std::invalid_argument);
    // Asserted on the document, not through getData: getData maps an absent
    // key and a JSON null to the same Value, so it cannot tell "never written"
    // from "written as null" and would pass against a partial write.
    EXPECT_TRUE(src.document().is_null());

    // The bound itself, pinned from both sides.
    jexpr::JsonDataSource bound;
    EXPECT_NO_THROW(bound.setData("q[1048575]", V(1))); // MAX_ARRAY_INDEX - 1
    EXPECT_THROW(bound.setData("r[1048576]", V(1)), std::invalid_argument);

    // An index inside the bound still works, and getData resolves an
    // out-of-bounds index to null exactly as it does past the end.
    src.setData("q[3]", V(7));
    EXPECT_EQ(src.getData("q[3]"), V(7));
    EXPECT_EQ(src.getData("q[100000000]"), V());
}

TEST(TestJsonDataSource, SetIsAllOrNothingOnARejectedPath)
{
    // The write walk creates each intermediate container as it descends, so
    // validating inline would let a rejected path destroy data it had already
    // passed over. The whole path is checked first; a throwing setData must
    // leave the document byte-identical.
    jexpr::JsonDataSource src{json{{"q", 5}}};
    EXPECT_THROW(src.setData("q.dims[999999999]", V(1)), std::invalid_argument);
    EXPECT_EQ(src.document(), json({{"q", 5}})); // the 5 is still a 5

    jexpr::JsonDataSource nested{json{{"a", {{"b", {{"c", 1}}}}}}};
    EXPECT_THROW(nested.setData("a.b.c.d[999999999]", V(2)), std::invalid_argument);
    EXPECT_EQ(nested.document(), json({{"a", {{"b", {{"c", 1}}}}}}));

    // A rejected index deep in an existing array leaves that array alone.
    jexpr::JsonDataSource arr{json{{"q", {{"dims", {8, 16}}}}}};
    EXPECT_THROW(arr.setData("q.dims[0].x[999999999]", V(3)), std::invalid_argument);
    EXPECT_EQ(arr.document(), json({{"q", {{"dims", {8, 16}}}}}));

    // ...while a valid write through the same shape still succeeds.
    arr.setData("q.dims[0]", V(99));
    EXPECT_EQ(arr.document(), json({{"q", {{"dims", {99, 16}}}}}));
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
