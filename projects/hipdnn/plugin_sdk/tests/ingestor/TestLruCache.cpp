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
 * @brief Unit tests for the bounded, thread-safe LRU cache LruCache.hpp declares.
 *
 * Concurrent access is covered separately by TestKernelIngestorConcurrency.cpp; these
 * tests are single-threaded and pin the eviction and recency contract itself.
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
    // A zero-capacity cache would evict every entry as it was inserted, which is always
    // a caller bug rather than a way to disable caching.
    using IntCache = LruCache<int, int>;
    EXPECT_THROW(IntCache(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Eviction order: which entry a capacity-crossing insert evicts depends on recency,
// not on insertion order, and a get() counts as a touch just as a put() does.
// ---------------------------------------------------------------------------

struct LruEvictionCase
{
    std::string name;
    /// Populates the cache and touches whatever entries the case needs, leaving it one
    /// insert away from a capacity-2 overflow.
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

    // The insert that crosses capacity: key 3 must always survive, since it is always
    // the most recently used entry regardless of which case set it up this way.
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
                                                               // Touching 1 makes 2 the least recently used.
                                                               ASSERT_TRUE(
                                                                   cache.get(1).has_value());
                                                           },
                                                           /*expectedEvicted=*/2,
                                                           /*expectedSurvivor=*/1},
                                           LruEvictionCase{"OverwritingAKeyRefreshesItsRecency",
                                                           [](LruCache<int, std::string>& cache) {
                                                               cache.put(1, "one");
                                                               cache.put(2, "two");
                                                               // Overwriting 1 marks it most-recently-used, same as a get().
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

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
