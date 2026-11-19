// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include <Tensile/ReadMostlyMap.hpp>

using TensileLite::InsertOnceCache;
using TensileLite::ReadMostlyMap;

namespace
{
    // Every key gets this hash, which ReadMostlyMap mixes to 0xffff: each probe
    // starts in the last slot of the table and wraps to the first.
    struct CollidingHash
    {
        size_t operator()(int) const
        {
            return 0xa0b118f49355e94cull;
        }
    };

    struct StringViewHash
    {
        size_t operator()(std::string_view s) const
        {
            return std::hash<std::string_view>{}(s);
        }
    };

    // Counts the values constructed, so a test can see whether a node was built.
    struct Counted
    {
        static int constructed;

        explicit Counted(int v)
            : value(v)
        {
            ++constructed;
        }

        int value;
    };

    int Counted::constructed = 0;
}

TEST(ReadMostlyMapTest, GrowsThroughSeveralDoublings)
{
    ReadMostlyMap<int, int> map;
    EXPECT_EQ(map.find(0), nullptr);

    for(int i = 0; i < 1000; ++i)
        map.insert(i, i * 3);

    for(int i = 0; i < 1000; ++i)
    {
        ASSERT_NE(map.find(i), nullptr) << i;
        EXPECT_EQ(*map.find(i), i * 3);
    }
    EXPECT_EQ(map.find(1000), nullptr);
}

TEST(ReadMostlyMapTest, InsertKeepsTheFirstValue)
{
    ReadMostlyMap<int, Counted> map;
    Counted::constructed = 0;

    Counted& first = map.insert(1, 10);
    EXPECT_EQ(Counted::constructed, 1);

    Counted& second = map.insert(1, 20);
    EXPECT_EQ(&second, &first);
    EXPECT_EQ(second.value, 10);
    EXPECT_EQ(Counted::constructed, 1) << "a present key builds no node";
    EXPECT_EQ(map.find(1), &first);
}

TEST(ReadMostlyMapTest, CollidingKeysWrapAround)
{
    ReadMostlyMap<int, int, CollidingHash> map;
    for(int i = 0; i < 40; ++i)
        map.insert(i, i + 100);
    EXPECT_EQ(map.insert(7, 7), 107);

    for(int i = 0; i < 40; ++i)
    {
        ASSERT_NE(map.find(i), nullptr) << i;
        EXPECT_EQ(*map.find(i), i + 100);
    }
    EXPECT_EQ(map.find(40), nullptr);
}

TEST(ReadMostlyMapTest, HeterogeneousFind)
{
    ReadMostlyMap<std::string, int, StringViewHash, std::equal_to<>> map;
    map.insert(std::string_view("kernel"), 5);

    ASSERT_NE(map.find(std::string_view("kernel")), nullptr);
    EXPECT_EQ(*map.find(std::string_view("kernel")), 5);
    EXPECT_EQ(*map.find(std::string("kernel")), 5);
    EXPECT_EQ(map.find(std::string_view("other")), nullptr);
}

TEST(ReadMostlyMapTest, ForEachValueVisitsEveryEntry)
{
    ReadMostlyMap<int, int> map;
    for(int i = 0; i < 100; ++i)
        map.insert(i, i);

    int sum = 0;
    map.forEachValue([&](int& v) {
        sum += v;
        v = -1;
    });
    EXPECT_EQ(sum, 99 * 100 / 2);
    for(int i = 0; i < 100; ++i)
        EXPECT_EQ(*map.find(i), -1);
}

TEST(ReadMostlyMapTest, ConcurrentReadersSeeWholeValues)
{
    constexpr int keys = 2000;

    ReadMostlyMap<int, std::vector<int>> map;
    std::mutex                           writer;
    std::atomic<bool>                    done{false};
    std::atomic<bool>                    torn{false};

    std::vector<std::thread> readers;
    for(int t = 0; t < 4; ++t)
        readers.emplace_back([&] {
            while(!done.load())
                for(int k = 0; k < keys; ++k)
                    if(auto const* v = map.find(k))
                        if(v->size() != 2 || (*v)[0] != k || (*v)[1] != -k)
                            torn = true;
        });

    for(int round = 0; round < 3; ++round)
        for(int k = 0; k < keys; ++k)
        {
            std::lock_guard<std::mutex> lock(writer);
            map.insert(k, std::vector<int>{k, -k});
        }

    done = true;
    for(auto& r : readers)
        r.join();

    EXPECT_FALSE(torn.load());
    for(int k = 0; k < keys; ++k)
        ASSERT_NE(map.find(k), nullptr) << k;
}

// An atomic value is updated and reset in place while readers run, the way a
// cache invalidates its entries without replacing them.
TEST(ReadMostlyMapTest, ConcurrentReadersSeeAtomicValuesUpdatedInPlace)
{
    constexpr int keys = 500;

    ReadMostlyMap<int, std::atomic<int>> map;
    std::mutex                           writer;
    std::atomic<bool>                    done{false};
    std::atomic<bool>                    bad{false};

    std::vector<std::thread> readers;
    for(int t = 0; t < 4; ++t)
        readers.emplace_back([&] {
            while(!done.load())
                for(int k = 0; k < keys; ++k)
                    if(auto const* v = map.find(k))
                    {
                        int value = v->load(std::memory_order_acquire);
                        if(value != 0 && value != k + 1 && value != -(k + 1))
                            bad = true;
                    }
        });

    for(int round = 0; round < 20; ++round)
    {
        for(int k = 0; k < keys; ++k)
        {
            int const                   value = round % 2 ? -(k + 1) : k + 1;
            std::lock_guard<std::mutex> lock(writer);
            map.insert(k, value).store(value, std::memory_order_release);
        }
        std::lock_guard<std::mutex> lock(writer);
        map.forEachValue([](std::atomic<int>& v) { v.store(0, std::memory_order_release); });
    }

    done = true;
    for(auto& r : readers)
        r.join();

    EXPECT_FALSE(bad.load());
    for(int k = 0; k < keys; ++k)
    {
        ASSERT_NE(map.find(k), nullptr) << k;
        EXPECT_EQ(map.find(k)->load(), 0);
    }
}

TEST(InsertOnceCacheTest, KeepsTheFirstValueAndReportsMisses)
{
    InsertOnceCache<std::tuple<int, int>, int> cache(std::make_tuple(-1, -1));
    EXPECT_EQ(cache.find(3), std::make_tuple(-1, -1));

    cache.add(std::make_tuple(1, 2), 3);
    cache.add(std::make_tuple(4, 5), 3);
    EXPECT_EQ(cache.find(3), std::make_tuple(1, 2));
    EXPECT_EQ(cache.find(4), std::make_tuple(-1, -1));
}
