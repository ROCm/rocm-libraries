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

/// Counts scorer calls, so a test can tell a served ranking from a recomputed one.
std::atomic<int>& scoreCalls()
{
    static std::atomic<int> s_scoreCalls{0};
    return s_scoreCalls;
}

double countingScoreByBlockSize(const KernelDefinition& kernel, const MatchContext& context)
{
    scoreCalls().fetch_add(1, std::memory_order_relaxed);
    return scoreByBlockSize(kernel, context);
}

/// Set by the thread taking the unsorted path, so the barrier below can hold it until
/// the sorting thread has installed its ranking.
thread_local bool holdUntilRanked = false;

TEST(TestIngestorStateManagerConcurrency, ARankingSurvivesAConcurrentUnsortedAccess)
{
    // D3, and the only interleaving that reaches it. Both threads must miss one key
    // and be inside buildCatalog() together -- if either finishes first the other takes
    // the cache hit and returns before it can write -- and the unsorted writer must
    // land *after* the ranking is installed. Single-threaded the defect is unreachable,
    // which is why the state-manager suite only covers the half of D3 it can see.
    //
    // Both conditions are forced rather than raced for, so this fails every run against
    // the defect instead of occasionally.
    constexpr const char* BARRIER_GRAPH_SYMBOL = "test.d3.barrier_graph_match";
    constexpr const char* COUNTING_SCORE_SYMBOL = "test.d3.counting_score";

    static std::atomic<int> s_arrived{0};
    static std::atomic<bool> s_ranked{false};
    s_arrived.store(0);
    s_ranked.store(false);
    scoreCalls().store(0);

    GraphMatcherRegistry::registerSymbol(
        BARRIER_GRAPH_SYMBOL, [](const MatchContext&, BoundTokens&) -> bool {
            // Neither thread leaves until both are here: this is what makes them miss
            // the cache together rather than one serving the other.
            s_arrived.fetch_add(1, std::memory_order_acq_rel);
            while(s_arrived.load(std::memory_order_acquire) < 2)
            {
                std::this_thread::yield();
            }

            // The unsorted thread then waits for the ranking to be installed, so its
            // own write is the one that lands last -- the case that used to clobber.
            while(holdUntilRanked && !s_ranked.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }
            return true;
        });
    KernelMatcherRegistry::registerSymbol(KERNEL_MATCH_SYMBOL, &acceptFloatKernels);
    ScoreRegistry::registerSymbol(COUNTING_SCORE_SYMBOL, &countingScoreByBlockSize);

    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "d3 schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "d3 pack";
    pack.matcherIds = {GRAPH_MATCHER_ID, KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_256_float", 256, "FLOAT")};

    const KernelIngestorStateManager<int> manager(
        std::move(schema),
        std::vector<MatchDescriptor>{
            {GRAPH_MATCHER_ID, "barrier graph scoped", MatchScope::GRAPH, BARRIER_GRAPH_SYMBOL},
            {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}},
        std::vector<DispatchDescriptor>{{DISPATCH_ID, "d3 dispatch", "test.d3.dispatch"}},
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(COUNTING_SCORE_SYMBOL));

    const TestGraph graph(makeGraphId(0x5E));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    runConcurrently(2, [&](int thread) {
        if(thread == 0)
        {
            EXPECT_TRUE(manager.sortedCatalog(context).isSorted);
            s_ranked.store(true, std::memory_order_release);
        }
        else
        {
            holdUntilRanked = true;
            static_cast<void>(manager.unsortedCatalog(context));
            holdUntilRanked = false;
        }
    });

    // Asserting isSorted here would prove nothing: sortedCatalog() ranks on demand, so
    // it reads true whether or not the cached entry survived. Whether the scorer runs
    // again is the observable difference -- and re-ranking every query is exactly the
    // thrash D3 describes.
    const auto callsAfterRace = scoreCalls().load(std::memory_order_relaxed);
    static_cast<void>(manager.sortedCatalog(context));

    EXPECT_EQ(scoreCalls().load(std::memory_order_relaxed), callsAfterRace)
        << "the cached ranking was discarded, so this query had to rank again";

    GraphMatcherRegistry::unregisterSymbol(BARRIER_GRAPH_SYMBOL);
    KernelMatcherRegistry::unregisterSymbol(KERNEL_MATCH_SYMBOL);
    ScoreRegistry::unregisterSymbol(COUNTING_SCORE_SYMBOL);
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
