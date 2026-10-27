// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

/// visit() with a visitor that only reports the hit.
bool visitKey(LruCache<int, std::string>& cache, int key)
{
    return cache.visit(key, [](const std::string& /*value*/) { return true; }).has_value();
}

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
                                                           /*expectedSurvivor=*/1},
                                           LruEvictionCase{"VisitingAnEntryProtectsItFromEviction",
                                                           [](LruCache<int, std::string>& cache) {
                                                               cache.put(1, "one");
                                                               cache.put(2, "two");
                                                               ASSERT_TRUE(visitKey(cache, 1));
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

TEST(TestIngestorLruCache, VisitReturnsTheVisitorsResultOnAHit)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");

    const auto length = cache.visit(1, [](const std::string& value) { return value.size(); });

    ASSERT_TRUE(length.has_value());
    EXPECT_EQ(*length, 3U);
}

TEST(TestIngestorLruCache, VisitReportsAMissWithoutCallingTheVisitor)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "one");
    bool visited = false;

    const auto result = cache.visit(42, [&visited](const std::string& /*value*/) {
        visited = true;
        return 0;
    });

    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(visited);
}

/// Counts copy constructions through a shared counter; moves are free, so only a real
/// copy of the cached value moves the count. The cache never copy-assigns a value.
struct CopyCountingValue
{
    CopyCountingValue(std::shared_ptr<int> copyCount, int payloadValue)
        : copies(std::move(copyCount))
        , payload(payloadValue)
    {
    }

    CopyCountingValue(const CopyCountingValue& other)
        : copies(other.copies)
        , payload(other.payload)
    {
        ++*copies;
    }

    CopyCountingValue& operator=(const CopyCountingValue&) = delete;

    CopyCountingValue(CopyCountingValue&&) noexcept = default;
    CopyCountingValue& operator=(CopyCountingValue&&) noexcept = default;
    ~CopyCountingValue() = default;

    std::shared_ptr<int> copies;
    int payload;
};

/// The point of visit() over get(): a hot-path probe of a large cached value (a whole
/// catalog) must read it in place rather than copy it out.
TEST(TestIngestorLruCache, VisitReadsTheCachedValueWithoutCopyingIt)
{
    const auto copies = std::make_shared<int>(0);
    LruCache<int, CopyCountingValue> cache(2);
    cache.put(1, CopyCountingValue(copies, 7));
    *copies = 0;

    const auto payload
        = cache.visit(1, [](const CopyCountingValue& value) { return value.payload; });

    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(*payload, 7);
    EXPECT_EQ(*copies, 0);
}

using Batch = std::vector<std::pair<int, std::string>>;

/// Key 1's newer value comes first and gets evicted; its older duplicate must still not be
/// admitted. A miss is acceptable.
TEST(TestIngestorLruCache, MergeAbsentNeverAdmitsAnOlderDuplicate)
{
    LruCache<int, std::string> cache(2);

    cache.mergeAbsent(Batch{{1, "newer"}, {3, "three"}, {2, "two"}, {1, "older"}});

    const auto found = cache.get(1);
    ASSERT_TRUE(found.has_value()) << "the newest entries are the resident ones";
    EXPECT_EQ(*found, "newer");
}

/// A key already cached is newer than the whole batch: it may be evicted, never replaced.
TEST(TestIngestorLruCache, MergeAbsentNeverReplacesAKeyThatWasPresent)
{
    LruCache<int, std::string> cache(2);
    cache.put(1, "in memory");

    cache.mergeAbsent(Batch{{2, "two"}, {3, "three"}, {1, "from batch"}});

    const auto found = cache.get(1);
    EXPECT_TRUE(!found.has_value() || *found == "in memory");
}

TEST(TestIngestorLruCache, MergeAbsentKeepsTheEarliestBatchEntriesWhenItOverflows)
{
    LruCache<int, std::string> cache(2);

    cache.mergeAbsent(Batch{{1, "one"}, {2, "two"}, {3, "three"}});

    EXPECT_EQ(cache.size(), 2U);
    EXPECT_TRUE(cache.get(1).has_value());
    EXPECT_TRUE(cache.get(2).has_value());
    EXPECT_FALSE(cache.get(3).has_value());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
