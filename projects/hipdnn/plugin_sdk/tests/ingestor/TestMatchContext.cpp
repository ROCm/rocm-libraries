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
    ::testing::Values(
        CatalogKeyInequalityCase{"DifferentGraphId", CatalogKey{makeGraphId(2), 0}},
        CatalogKeyInequalityCase{"DifferentDeviceId", CatalogKey{makeGraphId(1), 1}},
        // RFC 0019 §9.2 keys the cached ranking on an inventory generation counter so a
        // ranking cannot outlive the descriptor set that produced it. This build has no
        // in-process counter -- engines load once at startup -- so the engine descriptor
        // set's own revision carries that duty, and a bumped revision must not read an
        // entry the previous one ranked. A patch bump counts: the engine's authored
        // semantics are what the revision tracks, and this key cannot tell which part of a
        // bump changed behaviour.
        CatalogKeyInequalityCase{"DifferentEngineMajorVersion",
                                 CatalogKey{makeGraphId(1), 0, {1, 0, 0}}},
        CatalogKeyInequalityCase{"DifferentEnginePatchVersion",
                                 CatalogKey{makeGraphId(1), 0, {0, 0, 1}}},
        // RFC 0019 §11.4: an order ranked for one metric is not an order for another, so
        // a `time` request must never read the entry a `tflops` request cached.
        CatalogKeyInequalityCase{"DifferentRankingMetric",
                                 CatalogKey{makeGraphId(1), 0, {}, "time"}}),
    [](const ::testing::TestParamInfo<CatalogKeyInequalityCase>& info) { return info.param.name; });

TEST(TestIngestorMatchContext, CatalogKeyHashIsConsistentForEqualKeys)
{
    const CatalogKey first{makeGraphId(3), 2};
    const CatalogKey second{makeGraphId(3), 2};
    const CatalogKeyHash hash;

    EXPECT_EQ(hash(first), hash(second));
}

TEST(TestIngestorMatchContext, CatalogKeyHashDistinguishesDifferentDeviceIds)
{
    const CatalogKeyHash hash;
    const CatalogKey onDeviceZero{makeGraphId(4), 0};
    const CatalogKey onDeviceOne{makeGraphId(4), 1};

    EXPECT_NE(hash(onDeviceZero), hash(onDeviceOne));
}

/// Equality alone would still let a bumped revision land in the previous one's bucket and
/// be rejected only on the subsequent comparison; that is correct but wastes the bucket. The
/// real reason to assert it, though, is that folding a version into a hash invites packing
/// the three components into one number, and every cheap packing collides two versions that
/// must stay apart -- here 1.10.0 against 2.0.0 under a `major * 10 + minor` fold.
TEST(TestIngestorMatchContext, CatalogKeyHashDistinguishesEngineVersionsThatPackAlike)
{
    const CatalogKeyHash hash;
    const CatalogKey onOneTen{makeGraphId(5), 0, {1, 10, 0}};
    const CatalogKey onTwoZero{makeGraphId(5), 0, {2, 0, 0}};

    EXPECT_NE(hash(onOneTen), hash(onTwoZero));
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsTheGraphsIdentity)
{
    const TestGraph graph(makeGraphId(0x42));

    const auto id = tryGetGraphId(graph);

    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(*id, makeGraphId(0x42));
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsNulloptForAGraphWithNoIdentity)
{
    // No key to memoize under; callers must treat this as "cannot cache", not an error.
    const TestGraph graph;

    EXPECT_EQ(tryGetGraphId(graph), std::nullopt);
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsNulloptForANilId)
{
    // Present-but-nil must not read as a valid, cacheable key.
    const TestGraph graph(makeNilGraphId());

    EXPECT_EQ(tryGetGraphId(graph), std::nullopt);
}

TEST(TestIngestorMatchContext, TryGetGraphIdReturnsNulloptForANonV4Id)
{
    const TestGraph graph(makeNonV4GraphId(0x55));

    EXPECT_EQ(tryGetGraphId(graph), std::nullopt);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
