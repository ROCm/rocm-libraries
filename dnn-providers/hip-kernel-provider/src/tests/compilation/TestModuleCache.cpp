// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "compilation/ModuleCache.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

// Trivial value type for testing — a shared_ptr to an int.
using TestValue = std::shared_ptr<int>;

// A mock derived class that uses the CRTP base.  The load function
// returns a TestValue constructed from the key, or nullptr if the key
// starts with "FAIL".
class MockModuleCache : public hip_kernel_provider::compilation::
                            ModuleCache<MockModuleCache, TestValue, const std::string&>
{
public:
    static std::string makeKey(const std::string& name)
    {
        return name;
    }

    static TestValue load(const std::string& name)
    {
        if(name.rfind("FAIL", 0) == 0)
        {
            return nullptr;
        }
        // Use the key length as a distinguishable payload.
        return std::make_shared<int>(static_cast<int>(name.size()));
    }
};

/// Counts CountingModuleCache::load() calls. A free function holding the counter rather
/// than a static data member, so the fake keeps the all-static shape the CRTP base asks
/// for and the counter is still reachable from load().
std::atomic<int>& loadCount()
{
    static std::atomic<int> s_loads{0};
    return s_loads;
}

// A second fake whose load() is counted and deliberately slow. A cache that dropped its
// lock would let several threads miss the same key at once, and that shows up here as a
// counted, repeatable second load rather than as a rare torn read of the map.
class CountingModuleCache : public hip_kernel_provider::compilation::
                                ModuleCache<CountingModuleCache, TestValue, const std::string&>
{
public:
    static std::string makeKey(const std::string& name)
    {
        return name;
    }

    static TestValue load(const std::string& name)
    {
        loadCount().fetch_add(1, std::memory_order_relaxed);
        // Wide enough that unsynchronized threads released together overlap inside here
        // on any scheduler; getOrLoad holds its lock across this call.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return std::make_shared<int>(static_cast<int>(name.size()));
    }
};

/// Releases every thread at the same instant, so the calls under test actually overlap
/// instead of being serialized by thread start-up cost.
void waitForAll(std::atomic<int>& arrived, int expected)
{
    arrived.fetch_add(1, std::memory_order_acq_rel);
    while(arrived.load(std::memory_order_acquire) < expected)
    {
        std::this_thread::yield();
    }
}

TEST(TestModuleCache, EmptyOnConstruction)
{
    const MockModuleCache cache;
    EXPECT_EQ(cache.size(), 0u);
}

TEST(TestModuleCache, GetOrLoadCallsLoadOnMiss)
{
    MockModuleCache cache;
    auto result = cache.getOrLoad("kernel_a");

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, static_cast<int>(std::string("kernel_a").size()));
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_TRUE(cache.contains("kernel_a"));
}

TEST(TestModuleCache, GetOrLoadReturnsCachedOnHit)
{
    MockModuleCache cache;
    auto first = cache.getOrLoad("kernel_b");
    auto second = cache.getOrLoad("kernel_b");

    // Must be the exact same object (pointer equality).
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(cache.size(), 1u);
}

TEST(TestModuleCache, FailedLoadNotCached)
{
    MockModuleCache cache;
    auto result = cache.getOrLoad("FAIL_kernel");

    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.contains("FAIL_kernel"));
}

TEST(TestModuleCache, ContainsAndSizeTracking)
{
    MockModuleCache cache;
    EXPECT_FALSE(cache.contains("x"));

    cache.getOrLoad("x");
    cache.getOrLoad("y");
    cache.getOrLoad("FAIL_z");

    EXPECT_TRUE(cache.contains("x"));
    EXPECT_TRUE(cache.contains("y"));
    EXPECT_FALSE(cache.contains("FAIL_z"));
    EXPECT_EQ(cache.size(), 2u);
}

TEST(TestModuleCache, SeparateInstancesAreIsolated)
{
    MockModuleCache cacheA;
    const MockModuleCache cacheB;

    cacheA.getOrLoad("shared_key");

    EXPECT_EQ(cacheA.size(), 1u);
    EXPECT_EQ(cacheB.size(), 0u);
    EXPECT_TRUE(cacheA.contains("shared_key"));
    EXPECT_FALSE(cacheB.contains("shared_key"));
}

// Concurrency. getOrLoad() holds one mutex across the lookup, the load, and the insert,
// which is what makes a module cache shareable between the threads of one process.

TEST(TestModuleCache, ConcurrentGetOrLoadOfOneKeyLoadsExactlyOnce)
{
    constexpr int THREAD_COUNT = 8;

    CountingModuleCache cache;
    loadCount().store(0, std::memory_order_relaxed);

    std::atomic<int> arrived{0};
    // Sized up front and written one element per thread, so the threads never touch the
    // same object and the only shared state under test is the cache itself.
    std::vector<TestValue> results(THREAD_COUNT);

    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(int index = 0; index < THREAD_COUNT; ++index)
    {
        threads.emplace_back([&cache, &arrived, &results, index]() {
            waitForAll(arrived, THREAD_COUNT);
            results[static_cast<size_t>(index)] = cache.getOrLoad("contended_key");
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    // One load, not one per racing thread: the losers must observe the winner's entry.
    // Two loads of one key would mean two hipModule_t for one code object, which is the
    // cost this cache exists to avoid.
    EXPECT_EQ(loadCount().load(std::memory_order_relaxed), 1);

    ASSERT_NE(results.front(), nullptr);
    for(const auto& result : results)
    {
        EXPECT_EQ(result.get(), results.front().get())
            << "a racing caller was handed a different object for the same key";
    }
    EXPECT_EQ(cache.size(), 1u);
}

TEST(TestModuleCache, ConcurrentGetOrLoadOfDistinctKeysKeepsSizeExact)
{
    constexpr int THREAD_COUNT = 8;
    constexpr int KEYS_PER_THREAD = 32;

    MockModuleCache cache;
    std::atomic<int> arrived{0};
    std::atomic<int> mismatches{0};

    // Every thread inserts its own keys, so the count is known exactly; enough of them to
    // rehash the map repeatedly while other threads are reading it.
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(int index = 0; index < THREAD_COUNT; ++index)
    {
        threads.emplace_back([&cache, &arrived, &mismatches, index]() {
            waitForAll(arrived, THREAD_COUNT);
            for(int key = 0; key < KEYS_PER_THREAD; ++key)
            {
                const std::string name
                    = "thread" + std::to_string(index) + "_key" + std::to_string(key);
                const auto value = cache.getOrLoad(name);
                if(value == nullptr || *value != static_cast<int>(name.size()))
                {
                    mismatches.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0)
        << "a concurrent insert handed back a value that does not belong to its key";
    EXPECT_EQ(cache.size(), static_cast<size_t>(THREAD_COUNT * KEYS_PER_THREAD));
}

} // namespace
