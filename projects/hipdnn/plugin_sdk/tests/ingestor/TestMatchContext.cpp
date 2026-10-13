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
 *        tryGetGraphId()'s optional-identity contract, and the graph half of the key.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

TEST(TestIngestorMatchContext, CatalogKeysWithEqualFieldsCompareEqual)
{
    const CatalogKey first{CatalogGraphKey{makeGraphId(1)}, 0};
    const CatalogKey second{CatalogGraphKey{makeGraphId(1)}, 0};

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
    const CatalogKey reference{CatalogGraphKey{makeGraphId(1)}, 0};

    EXPECT_FALSE(reference == GetParam().key);
}

INSTANTIATE_TEST_SUITE_P(
    OneFieldAtATime,
    TestIngestorMatchContextCatalogKeyInequality,
    ::testing::Values(
        CatalogKeyInequalityCase{"DifferentGraphId",
                                 CatalogKey{CatalogGraphKey{makeGraphId(2)}, 0}},
        CatalogKeyInequalityCase{"DifferentDeviceId",
                                 CatalogKey{CatalogGraphKey{makeGraphId(1)}, 1}},
        // RFC 0019 §9.2: the engine descriptor revision stands in for the inventory
        // generation counter, so any bump, patch included, must miss.
        CatalogKeyInequalityCase{"DifferentEngineMajorVersion",
                                 CatalogKey{CatalogGraphKey{makeGraphId(1)}, 0, {1, 0, 0}}},
        CatalogKeyInequalityCase{"DifferentEnginePatchVersion",
                                 CatalogKey{CatalogGraphKey{makeGraphId(1)}, 0, {0, 0, 1}}},
        // RFC 0019 §11.4: an order ranked for one metric is not an order for another, so
        // a `time` request must never read the entry a `tflops` request cached.
        CatalogKeyInequalityCase{"DifferentRankingMetric",
                                 CatalogKey{CatalogGraphKey{makeGraphId(1)}, 0, {}, "time"}}),
    [](const ::testing::TestParamInfo<CatalogKeyInequalityCase>& info) { return info.param.name; });

TEST(TestIngestorMatchContext, CatalogKeyHashIsConsistentForEqualKeys)
{
    const CatalogKey first{CatalogGraphKey{makeGraphId(3)}, 2};
    const CatalogKey second{CatalogGraphKey{makeGraphId(3)}, 2};
    const CatalogKeyHash hash;

    EXPECT_EQ(hash(first), hash(second));
}

TEST(TestIngestorMatchContext, CatalogKeyHashDistinguishesDifferentDeviceIds)
{
    const CatalogKeyHash hash;
    const CatalogKey onDeviceZero{CatalogGraphKey{makeGraphId(4)}, 0};
    const CatalogKey onDeviceOne{CatalogGraphKey{makeGraphId(4)}, 1};

    EXPECT_NE(hash(onDeviceZero), hash(onDeviceOne));
}

/// A cheap packing such as `major * 10 + minor` would collide 1.10.0 with 2.0.0.
TEST(TestIngestorMatchContext, CatalogKeyHashDistinguishesEngineVersionsThatPackAlike)
{
    const CatalogKeyHash hash;
    const CatalogKey onOneTen{CatalogGraphKey{makeGraphId(5)}, 0, {1, 10, 0}};
    const CatalogKey onTwoZero{CatalogGraphKey{makeGraphId(5)}, 0, {2, 0, 0}};

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

TEST(TestIngestorMatchContext, CatalogGraphKeyUsesAV4IdWhenTheGraphHasOne)
{
    const TestGraph graph(makeGraphId(0x42));

    const auto key = CatalogGraphKey::tryMake(graph);

    ASSERT_TRUE(key.has_value());
    EXPECT_FALSE(key->isByContent());
    EXPECT_TRUE(*key == CatalogGraphKey{makeGraphId(0x42)});
}

/// A corpus graph's content-derived id is not v4; its bytes still identify it.
TEST(TestIngestorMatchContext, CatalogGraphKeyFallsBackToBytesForANonV4Id)
{
    const TestGraph first(makeNonV4GraphId(0x55));
    const TestGraph second(makeNonV4GraphId(0x55));

    const auto firstKey = CatalogGraphKey::tryMake(first);
    const auto secondKey = CatalogGraphKey::tryMake(second);

    ASSERT_TRUE(firstKey.has_value());
    ASSERT_TRUE(secondKey.has_value());
    EXPECT_TRUE(firstKey->isByContent());
    EXPECT_TRUE(*firstKey == *secondKey);
    EXPECT_EQ(firstKey->hash(), secondKey->hash());
}

TEST(TestIngestorMatchContext, CatalogGraphKeysOfGraphsWithDifferentBytesDiffer)
{
    const TestGraph first(makeNonV4GraphId(0x55));
    const TestGraph otherId(makeNonV4GraphId(0x56));
    const TestGraph otherFloor(makeNonV4GraphId(0x55),
                               hipdnn_data_sdk::utilities::Version{1, 2, 0});
    const TestGraph noId;

    const auto key = CatalogGraphKey::tryMake(first);
    ASSERT_TRUE(key.has_value());
    EXPECT_FALSE(*key == *CatalogGraphKey::tryMake(otherId));
    EXPECT_FALSE(*key == *CatalogGraphKey::tryMake(otherFloor));
    EXPECT_FALSE(*key == *CatalogGraphKey::tryMake(noId));
}

TEST(TestIngestorMatchContext, CatalogGraphKeyByBytesNeverEqualsAKeyById)
{
    const TestGraph graph(makeNilGraphId());

    const auto key = CatalogGraphKey::tryMake(graph);

    ASSERT_TRUE(key.has_value());
    EXPECT_TRUE(key->isByContent());
    EXPECT_FALSE(*key == CatalogGraphKey{makeNilGraphId()});
}

/// No v4 id and no bytes: nothing identifies the graph, so it is never cached.
TEST(TestIngestorMatchContext, CatalogGraphKeyIsAbsentForAGraphWithNeitherIdNorBytes)
{
    const TestGraphWithoutBytes graph;

    EXPECT_EQ(CatalogGraphKey::tryMake(graph), std::nullopt);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
