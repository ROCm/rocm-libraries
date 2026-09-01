// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <functional>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/LruCache.hpp>

/**
 * @file TestLruCache.cpp
 * @brief Tests for the bounded, thread-safe LRU cache LruCache.hpp declares.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

TEST(TestIngestorLruCache, ReturnsCachedValueWithinCapacity)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");

    const auto found = cache.get(1);

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, "one");
}

TEST(TestIngestorLruCache, ReportsMissForAbsentKey)
{
    LruCache<int, std::string> cache(2);

    EXPECT_FALSE(cache.get(42).has_value());
}

TEST(TestIngestorLruCache, RejectsZeroCapacity)
{
    using IntCache = LruCache<int, int>;
    EXPECT_THROW(IntCache(0), std::invalid_argument);
}

struct LruEvictionCase
{
    std::string name;
    std::function<void(LruCache<int, std::string>&)> setup;
    int expectedEvicted;
    int expectedSurvivor;
};

class TestIngestorLruCacheEvictionOrder : public ::testing::TestWithParam<LruEvictionCase>
{
};

TEST_P(TestIngestorLruCacheEvictionOrder, EvictsByRecencyNotInsertionOrder)
{
    const auto& testCase = GetParam();
    LruCache<int, std::string> cache(2);
    testCase.setup(cache);

    cache.put(3, "three");

    EXPECT_FALSE(cache.get(testCase.expectedEvicted).has_value());
    EXPECT_TRUE(cache.get(testCase.expectedSurvivor).has_value());
    EXPECT_TRUE(cache.get(3).has_value());
    EXPECT_EQ(cache.size(), 2U);
}

INSTANTIATE_TEST_SUITE_P(EvictionOrder,
                         TestIngestorLruCacheEvictionOrder,
                         ::testing::Values(LruEvictionCase{"PlainInsertionOrderEvictsOldest",
                                                           [](LruCache<int, std::string>& cache) {
                                                               cache.put(1, "one");
                                                               cache.put(2, "two");
                                                           },
                                                           /*expectedEvicted=*/1,
                                                           /*expectedSurvivor=*/2},
                                           LruEvictionCase{"ReadingAnEntryProtectsItFromEviction",
                                                           [](LruCache<int, std::string>& cache) {
                                                               cache.put(1, "one");
                                                               cache.put(2, "two");
                                                               ASSERT_TRUE(
                                                                   cache.get(1).has_value());
                                                           },
                                                           /*expectedEvicted=*/2,
                                                           /*expectedSurvivor=*/1},
                                           LruEvictionCase{"OverwritingAKeyRefreshesItsRecency",
                                                           [](LruCache<int, std::string>& cache) {
                                                               cache.put(1, "one");
                                                               cache.put(2, "two");
                                                               cache.put(1, "uno");
                                                           },
                                                           /*expectedEvicted=*/2,
                                                           /*expectedSurvivor=*/1}),
                         [](const ::testing::TestParamInfo<LruEvictionCase>& info) {
                             return info.param.name;
                         });

TEST(TestIngestorLruCache, OverwritingAKeyDoesNotGrowTheCache)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(1, "uno");

    const auto found = cache.get(1);

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, "uno");
    EXPECT_EQ(cache.size(), 1U);
}

TEST(TestIngestorLruCache, PutIfAbsentInsertsWhenTheKeyIsMissing)
{
    LruCache<int, std::string> cache(2);

    EXPECT_TRUE(cache.putIfAbsent(1, "one"));

    const auto found = cache.get(1);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, "one");
}

TEST(TestIngestorLruCache, PutIfAbsentLeavesAnExistingValueAlone)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");

    EXPECT_FALSE(cache.putIfAbsent(1, "uno"));

    const auto found = cache.get(1);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, "one");
    EXPECT_EQ(cache.size(), 1U);
}

TEST(TestIngestorLruCache, PutIfAbsentEvictsTheLeastRecentlyUsedWhenItInserts)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");

    EXPECT_TRUE(cache.putIfAbsent(3, "three"));

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_FALSE(cache.get(1).has_value());
    EXPECT_TRUE(cache.get(3).has_value());
}

TEST(TestIngestorLruCache, PutIfAbsentRefreshesRecencyOnAKeyItDidNotWrite)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    cache.put(2, "two");

    EXPECT_FALSE(cache.putIfAbsent(1, "uno"));
    cache.put(3, "three");

    EXPECT_TRUE(cache.get(1).has_value());
    EXPECT_FALSE(cache.get(2).has_value());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
