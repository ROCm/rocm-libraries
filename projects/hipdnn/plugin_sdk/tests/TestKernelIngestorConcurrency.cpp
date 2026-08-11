// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/LruCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelIngestorConcurrency.cpp
 * @brief The ingestor's shared state under concurrent use.
 *
 * One engine, and therefore one catalog cache, is shared by every handle in the process,
 * and hipDNN may drive several handles from several threads at once. These tests pin the
 * two properties that arrangement depends on: the shared state stays internally
 * consistent under concurrent access, and the answers it gives do not depend on which
 * thread asked or on what any other thread was doing.
 *
 * They are worth running under thread and address sanitizers, where a data race in the
 * cache would be reported directly rather than inferred from a corrupted result.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

constexpr int THREAD_COUNT = 8;
constexpr int ITERATIONS_PER_THREAD = 200;

/// Runs @p body on several threads at once, releasing them together so their work
/// genuinely overlaps rather than serializing by startup order.
template <typename Body>
void runConcurrently(int threadCount, Body body)
{
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));

    for(int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back([&go, &body, i]() {
            while(!go.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }
            body(i);
        });
    }

    go.store(true, std::memory_order_release);
    for(auto& thread : threads)
    {
        thread.join();
    }
}

// ---------------------------------------------------------------------------
// LruCache
// ---------------------------------------------------------------------------

TEST(TestIngestorCacheConcurrency, SurvivesConcurrentReadsAndWrites)
{
    // Capacity far below the key range, so eviction runs constantly and readers are
    // touching entries writers are simultaneously retiring.
    LruCache<int, int> cache(16);

    runConcurrently(THREAD_COUNT, [&cache](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const int key = (thread * ITERATIONS_PER_THREAD + i) % 128;
            cache.put(key, key * 2);

            // A hit must carry the value stored under that key; a miss is legitimate
            // because another thread may have evicted it in between.
            if(const auto found = cache.get(key); found.has_value())
            {
                EXPECT_EQ(*found, key * 2);
            }
        }
    });

    EXPECT_LE(cache.size(), cache.capacity());
}

TEST(TestIngestorCacheConcurrency, NeverExceedsCapacityUnderContention)
{
    LruCache<int, int> cache(4);

    runConcurrently(THREAD_COUNT, [&cache](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            cache.put(thread * ITERATIONS_PER_THREAD + i, i);
        }
    });

    // The bound is what makes this cache safe to leave running in a long-lived process:
    // concurrent insertion must not be able to grow it without limit.
    EXPECT_LE(cache.size(), cache.capacity());
}

// ---------------------------------------------------------------------------
// KernelIngestorStateManager
// ---------------------------------------------------------------------------

TEST(TestIngestorStateManagerConcurrency, ServesOneGraphFromManyThreadsConsistently)
{
    const ScopedTestSymbols symbols;
    const auto manager = makeTestStateManager();
    const TestGraph graph(makeGraphId(0x21));
    const auto properties = testDeviceProperties();

    // Every thread asks the same question of the same shared cache, the way several
    // handles bound to one device would.
    runConcurrently(THREAD_COUNT, [&](int) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const MatchContext context{graph, 0, properties};
            const auto definitions = manager->unsortedDefinitions(context);
            ASSERT_EQ(definitions.size(), 2U);
        }
    });
}

TEST(TestIngestorStateManagerConcurrency, RanksConsistentlyFromManyThreads)
{
    const ScopedTestSymbols symbols;
    const auto manager = makeTestStateManager();
    const TestGraph graph(makeGraphId(0x22));
    const auto properties = testDeviceProperties();

    // Ranking writes the sorted catalog back into the cache, so this is the path where
    // concurrent readers and a writer meet. Every thread must still see the same winner.
    runConcurrently(THREAD_COUNT, [&](int) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const MatchContext context{graph, 0, properties};
            const auto ranked = manager->sortedDefinitions(context);
            ASSERT_EQ(ranked.size(), 2U);
            EXPECT_EQ(ranked.front().getIntMetadata(std::string(BLOCK_SIZE)), 256);
        }
    });
}

TEST(TestIngestorStateManagerConcurrency, KeepsPerDeviceCatalogsDistinct)
{
    const ScopedTestSymbols symbols;
    const auto manager = makeTestStateManager();
    const TestGraph graph(makeGraphId(0x23));
    const auto properties = testDeviceProperties();

    // One graph, several devices, all at once: the catalog is keyed on both, so a
    // device's answer must never be served from another device's entry.
    runConcurrently(THREAD_COUNT, [&](int thread) {
        const DeviceId deviceId = thread % 4;
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const MatchContext context{graph, deviceId, properties};
            const auto definitions = manager->unsortedDefinitions(context);
            ASSERT_EQ(definitions.size(), 2U);
        }
    });
}

TEST(TestIngestorStateManagerConcurrency, ServesUncacheableGraphsConcurrently)
{
    const ScopedTestSymbols symbols;
    const auto manager = makeTestStateManager();
    // No identity, so every call rematches instead of reading the cache. That path runs
    // the matchers concurrently rather than serializing behind a cached result.
    const TestGraph graph;
    const auto properties = testDeviceProperties();

    runConcurrently(THREAD_COUNT, [&](int) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const MatchContext context{graph, 0, properties};
            ASSERT_EQ(manager->unsortedDefinitions(context).size(), 2U);
        }
    });
}

TEST(TestIngestorStateManagerConcurrency, EvictsUnderConcurrentDistinctGraphs)
{
    const ScopedTestSymbols symbols;
    // Capacity 2 against many distinct graphs, so entries are evicted while other
    // threads are reading and writing them.
    const auto manager = makeTestStateManager(2);
    const auto properties = testDeviceProperties();

    runConcurrently(THREAD_COUNT, [&](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const TestGraph graph(makeGraphId(static_cast<uint8_t>((thread * 7 + i) % 32)));
            const MatchContext context{graph, 0, properties};

            // Eviction costs a rematch and never a wrong answer, so the result is the
            // same whether this call hit the cache or rebuilt the catalog.
            ASSERT_EQ(manager->unsortedDefinitions(context).size(), 2U);
        }
    });
}

// ---------------------------------------------------------------------------
// NativeRegistry
// ---------------------------------------------------------------------------

TEST(TestIngestorRegistryConcurrency, ResolvesFromManyThreads)
{
    const ScopedTestSymbols symbols;

    // Resolution happens on the applicability path, which several threads reach at once.
    runConcurrently(THREAD_COUNT, [](int) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            EXPECT_NE(GraphMatcherRegistry::resolve(std::string(GRAPH_MATCH_SYMBOL)), nullptr);
            EXPECT_NE(ScoreRegistry::resolve(std::string(SCORE_SYMBOL)), nullptr);
        }
    });
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
