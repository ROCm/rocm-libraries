// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <optional>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestMatchContext.cpp
 * @brief Unit tests for MatchContext.hpp: the catalog cache key's equality and hash, and
 *        tryGetGraphId()'s optional-identity contract.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

// ---------------------------------------------------------------------------
// CatalogKey::operator==
// ---------------------------------------------------------------------------

TEST(TestIngestorMatchContext, CatalogKeysWithEqualFieldsCompareEqual)
{
    const CatalogKey first{makeGraphId(1), 0};
    const CatalogKey second{makeGraphId(1), 0};

    EXPECT_TRUE(first == second);
}

struct CatalogKeyInequalityCase
{
    std::string name;
    CatalogKey key;
};

class TestIngestorMatchContextCatalogKeyInequality
    : public ::testing::TestWithParam<CatalogKeyInequalityCase>
{
};

TEST_P(TestIngestorMatchContextCatalogKeyInequality, KeysDifferingInOneFieldCompareUnequal)
{
    const CatalogKey reference{makeGraphId(1), 0};

    EXPECT_FALSE(reference == GetParam().key);
}

INSTANTIATE_TEST_SUITE_P(
    OneFieldAtATime,
    TestIngestorMatchContextCatalogKeyInequality,
    ::testing::Values(CatalogKeyInequalityCase{"DifferentGraphId", CatalogKey{makeGraphId(2), 0}},
                      CatalogKeyInequalityCase{"DifferentDeviceId", CatalogKey{makeGraphId(1), 1}}),
    [](const ::testing::TestParamInfo<CatalogKeyInequalityCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// CatalogKeyHash
// ---------------------------------------------------------------------------

TEST(TestIngestorMatchContext, CatalogKeyHashIsConsistentForEqualKeys)
{
    const CatalogKey first{makeGraphId(3), 2};
    const CatalogKey second{makeGraphId(3), 2};
    const CatalogKeyHash hash;

    // A hash function must agree with operator== on which keys are "the same": unequal
    // hashes for equal keys would break the LruCache's unordered_map lookup silently.
    EXPECT_EQ(hash(first), hash(second));
}

TEST(TestIngestorMatchContext, CatalogKeyHashDistinguishesDifferentDeviceIds)
{
    const CatalogKeyHash hash;
    const CatalogKey onDeviceZero{makeGraphId(4), 0};
    const CatalogKey onDeviceOne{makeGraphId(4), 1};

    // Not a correctness requirement (collisions are legal), but this hash's own doc
    // claims it folds the device ordinal in, and a hash that silently ignored deviceId
    // would still "work" via bucket collision while quietly regressing lookup cost.
    EXPECT_NE(hash(onDeviceZero), hash(onDeviceOne));
}

// ---------------------------------------------------------------------------
// tryGetGraphId
// ---------------------------------------------------------------------------

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsTheGraphsIdentity)
{
    const TestGraph graph(makeGraphId(0x42));

    const auto id = tryGetGraphId(graph);

    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(*id, makeGraphId(0x42));
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsNulloptForAGraphWithNoIdentity)
{
    // A legacy or unfinalized graph: there is no key to memoize under, and callers must
    // treat this as "cannot cache", not as an error.
    const TestGraph graph;

    EXPECT_EQ(tryGetGraphId(graph), std::nullopt);
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsNulloptForANilId)
{
    // A present-but-nil id must not read as a valid, cacheable key.
    const TestGraph graph(makeNilGraphId());

    EXPECT_EQ(tryGetGraphId(graph), std::nullopt);
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsNulloptForANonV4Id)
{
    // Any non-v4 id is as unable to promise per-graph uniqueness as nil.
    const TestGraph graph(makeNonV4GraphId(0x55));

    EXPECT_EQ(tryGetGraphId(graph), std::nullopt);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
