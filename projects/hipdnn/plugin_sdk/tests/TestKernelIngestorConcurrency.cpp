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

#include "ingestor/KernelIngestorTestFixtures.hpp"

/**
 * @file TestKernelIngestorConcurrency.cpp
 * @brief Concurrency tests for the ingestor's shared per-process catalog cache: internal
 *        consistency and thread/order independence under concurrent access.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

constexpr int THREAD_COUNT = 8;
constexpr int ITERATIONS_PER_THREAD = 200;

/// Runs @p body on threadCount threads, releasing them together so they truly overlap.
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
    // Capacity far below the key range: eviction runs constantly, so readers race
    // evicting writers on the same entries.
    LruCache<int, int> cache(16);

    runConcurrently(THREAD_COUNT, [&cache](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const int key = (thread * ITERATIONS_PER_THREAD + i) % 128;
            cache.put(key, key * 2);

            // Hit must match; a miss is legal (raced with eviction).
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

    // Concurrent insertion must not grow the cache past capacity.
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

    // Same graph and device from every thread: shared-cache read contention.
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

    // Ranking writes the sorted catalog back: readers and the writer race here.
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

    // Same graph, distinct devices concurrently: catalog is keyed on both and must not
    // cross-serve.
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
    // No graph identity: every call rematches, exercising concurrent matcher execution
    // instead of the cache.
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
    // Capacity 2 against many distinct graphs: constant eviction races readers/writers.
    const auto manager = makeTestStateManager(2);
    const auto properties = testDeviceProperties();

    runConcurrently(THREAD_COUNT, [&](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const TestGraph graph(makeGraphId(static_cast<uint8_t>((thread * 7 + i) % 32)));
            const MatchContext context{graph, 0, properties};

            // Eviction only costs a rematch, never a wrong answer.
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

    // Exercises concurrent resolve() from the applicability path.
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
