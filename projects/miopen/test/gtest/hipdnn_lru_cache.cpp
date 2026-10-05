// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for src/private/lru_cache.hpp, which bounds the hipDNN plan cache.
// No hipDNN install and no GPU are needed.

#include <gtest/gtest.h>

#include "../../src/private/lru_cache.hpp"

namespace {

using Cache = miopen::wrapper::LruCache<int, int>;

} // namespace

TEST(CPU_HipdnnLruCache_NONE, SizeNeverExceedsCapacity)
{
    Cache cache(3);
    for(int key = 0; key < 10; ++key)
    {
        cache.Insert(key, key);
        EXPECT_LE(cache.Size(), 3u);
    }
    EXPECT_EQ(cache.Size(), 3u);
    EXPECT_EQ(cache.Find(6), nullptr);
    ASSERT_NE(cache.Find(9), nullptr);
    EXPECT_EQ(*cache.Find(9), 9);
}

TEST(CPU_HipdnnLruCache_NONE, FindKeepsAnEntryFromBeingEvicted)
{
    Cache cache(2);
    cache.Insert(1, 10);
    cache.Insert(2, 20);
    ASSERT_NE(cache.Find(1), nullptr);
    cache.Insert(3, 30);
    EXPECT_NE(cache.Find(1), nullptr);
    EXPECT_EQ(cache.Find(2), nullptr);
    EXPECT_NE(cache.Find(3), nullptr);
}

TEST(CPU_HipdnnLruCache_NONE, InsertingAnExistingKeyReplacesItsValue)
{
    Cache cache(2);
    cache.Insert(1, 10);
    cache.Insert(1, 11);
    EXPECT_EQ(cache.Size(), 1u);
    ASSERT_NE(cache.Find(1), nullptr);
    EXPECT_EQ(*cache.Find(1), 11);
}

TEST(CPU_HipdnnLruCache_NONE, EraseIfRemovesOnlyMatchingEntries)
{
    Cache cache(4);
    for(int key = 0; key < 4; ++key)
        cache.Insert(key, key);
    cache.EraseIf([](int key) { return key % 2 == 0; });
    EXPECT_EQ(cache.Size(), 2u);
    EXPECT_EQ(cache.Find(0), nullptr);
    EXPECT_EQ(cache.Find(2), nullptr);
    EXPECT_NE(cache.Find(1), nullptr);
    EXPECT_NE(cache.Find(3), nullptr);
}
