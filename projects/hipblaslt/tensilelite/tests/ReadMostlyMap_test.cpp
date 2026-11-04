// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <Tensile/ReadMostlyMap.hpp>

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
}

TEST(ReadMostlyMapTest, GrowsThroughSeveralDoublings)
{
    ReadMostlyMap<int, int> map;
    EXPECT_EQ(map.find(0), nullptr);

    for(int i = 0; i < 1000; ++i)
        map.insertOrAssign(i, i * 3);

    for(int i = 0; i < 1000; ++i)
    {
        ASSERT_NE(map.find(i), nullptr) << i;
        EXPECT_EQ(*map.find(i), i * 3);
    }
    EXPECT_EQ(map.find(1000), nullptr);
}

TEST(ReadMostlyMapTest, CollidingKeysWrapAround)
{
    ReadMostlyMap<int, int, CollidingHash> map;
    for(int i = 0; i < 40; ++i)
        map.insertOrAssign(i, i + 100);
    map.insertOrAssign(7, 7);

    for(int i = 0; i < 40; ++i)
    {
        ASSERT_NE(map.find(i), nullptr) << i;
        EXPECT_EQ(*map.find(i), i == 7 ? 7 : i + 100);
    }
    EXPECT_EQ(map.find(40), nullptr);
}

TEST(ReadMostlyMapTest, OverwriteKeepsOldValueValid)
{
    ReadMostlyMap<int, std::string> map;
    map.insertOrAssign(1, "old");
    std::string const* old = map.find(1);

    map.insertOrAssign(1, "new");
    EXPECT_EQ(*old, "old");
    EXPECT_EQ(*map.find(1), "new");
}

TEST(ReadMostlyMapTest, ClearThenReinsert)
{
    ReadMostlyMap<int, std::string> map;
    map.insertOrAssign(1, "first");
    std::string const* old = map.find(1);

    map.clear();
    EXPECT_EQ(map.find(1), nullptr);
    EXPECT_EQ(*old, "first");

    map.insertOrAssign(1, "second");
    EXPECT_EQ(*map.find(1), "second");
}

TEST(ReadMostlyMapTest, HeterogeneousFind)
{
    ReadMostlyMap<std::string, int, StringViewHash, std::equal_to<>> map;
    map.insertOrAssign("kernel", 5);

    ASSERT_NE(map.find(std::string_view("kernel")), nullptr);
    EXPECT_EQ(*map.find(std::string_view("kernel")), 5);
    EXPECT_EQ(map.find(std::string_view("other")), nullptr);
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
                        if(v->size() != 2 || (*v)[0] != k || ((*v)[1] != k && (*v)[1] != -k))
                            torn = true;
        });

    for(int round = 0; round < 3; ++round)
        for(int k = 0; k < keys; ++k)
        {
            std::lock_guard<std::mutex> lock(writer);
            map.insertOrAssign(k, {k, round % 2 ? -k : k});
        }

    done = true;
    for(auto& r : readers)
        r.join();

    EXPECT_FALSE(torn.load());
    for(int k = 0; k < keys; ++k)
        ASSERT_NE(map.find(k), nullptr) << k;
}
