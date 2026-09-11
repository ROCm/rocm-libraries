// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
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

/// Ceiling on every wait in this file: bounds a hang to a named assertion instead of a
/// killed binary with no diagnosis. Named MAX_WAIT because winerror.h makes
/// WAIT_TIMEOUT a macro.
constexpr auto MAX_WAIT = std::chrono::seconds(30);

class Gate
{
public:
    bool wait()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _cv.wait_for(lock, MAX_WAIT, [this] { return _open; });
    }

    void open()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _open = true;
        }
        _cv.notify_all();
    }

private:
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _open = false;
};

template <typename Body>
void runConcurrently(int threadCount, Body body)
{
    Gate start;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));

    for(int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back([&start, &body, i]() {
            EXPECT_TRUE(start.wait()) << "thread " << i << " timed out waiting to start";
            body(i);
        });
    }

    start.open();
    for(auto& thread : threads)
    {
        thread.join();
    }
}

TEST(TestIngestorCacheConcurrency, SurvivesConcurrentReadsAndWrites)
{
    LruCache<int, int> cache(16);

    runConcurrently(THREAD_COUNT, [&cache](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const int key = (thread * ITERATIONS_PER_THREAD + i) % 128;
            cache.put(key, key * 2);

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

    EXPECT_LE(cache.size(), cache.capacity());
}

TEST(TestIngestorStateManagerConcurrency, ServesOneGraphFromManyThreadsConsistently)
{
    const ScopedTestSymbols symbols;
    const auto manager = makeTestStateManager();
    const TestGraph graph(makeGraphId(0x21));
    const auto properties = testDeviceProperties();

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
    const auto manager = makeTestStateManager(2);
    const auto properties = testDeviceProperties();

    runConcurrently(THREAD_COUNT, [&](int thread) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            const TestGraph graph(makeGraphId(static_cast<uint8_t>((thread * 7 + i) % 32)));
            const MatchContext context{graph, 0, properties};

            ASSERT_EQ(manager->unsortedDefinitions(context).size(), 2U);
        }
    });
}

std::atomic<int>& scoreCalls()
{
    static std::atomic<int> s_scoreCalls{0};
    return s_scoreCalls;
}

double countingScoreByBlockSize(const MatchContext& context,
                                const BoundTokens& bound,
                                const KernelDefinition& kernel)
{
    scoreCalls().fetch_add(1, std::memory_order_relaxed);
    return scoreByBlockSize(context, bound, kernel);
}

thread_local bool holdUntilRanked = false;

/// A two-thread rendezvous: release condition is another thread arriving, not an
/// external open(); plus a one-way "ranking installed" signal.
class RankingBarrier
{
public:
    bool arriveAndWait()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        ++_arrived;
        _cv.notify_all();
        return _cv.wait_for(lock, MAX_WAIT, [this] { return _arrived >= 2; });
    }

    bool waitForRanked()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _cv.wait_for(lock, MAX_WAIT, [this] { return _ranked; });
    }

    void markRanked()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _ranked = true;
        }
        _cv.notify_all();
    }

    /// True if any wait timed out, meaning the forced interleaving never happened.
    bool timedOut() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _timedOut;
    }

    void recordTimeout()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _timedOut = true;
    }

private:
    mutable std::mutex _mutex;
    std::condition_variable _cv;
    int _arrived = 0;
    bool _ranked = false;
    bool _timedOut = false;
};

/// Lets the plain-function-pointer matcher find the barrier; set before, cleared after.
RankingBarrier*& rankingBarrier()
{
    static RankingBarrier* s_barrier = nullptr;
    return s_barrier;
}

TEST(TestIngestorStateManagerConcurrency, ARankingSurvivesAConcurrentUnsortedAccess)
{
    // Waits are bounded so a regression hangs CI by name instead of silently.
    constexpr const char* BARRIER_GRAPH_SYMBOL = "test.d3.barrier_graph_match";
    constexpr const char* COUNTING_SCORE_SYMBOL = "test.d3.counting_score";

    RankingBarrier barrier;
    rankingBarrier() = &barrier;
    scoreCalls().store(0);

    GraphMatchRegistry::registerSymbol(BARRIER_GRAPH_SYMBOL,
                                       [](const MatchContext&) -> std::optional<BoundTokens> {
                                           if(!rankingBarrier()->arriveAndWait())
                                           {
                                               rankingBarrier()->recordTimeout();
                                               return BoundTokens{};
                                           }

                                           if(holdUntilRanked && !rankingBarrier()->waitForRanked())
                                           {
                                               rankingBarrier()->recordTimeout();
                                           }
                                           return BoundTokens{};
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
    pack.matcherIds = {KERNEL_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_256_float", 256, "FLOAT")};

    const KernelIngestorStateManager<TestHandle> manager(
        std::move(schema),
        std::vector<MatchDescriptor>{
            {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, KERNEL_MATCH_SYMBOL}},
        makeTestDispatches<TestHandle>(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(COUNTING_SCORE_SYMBOL),
        BARRIER_GRAPH_SYMBOL);

    const TestGraph graph(makeGraphId(0x5E));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    runConcurrently(2, [&](int thread) {
        if(thread == 0)
        {
            EXPECT_TRUE(manager.sortedCatalog(context).isSorted);
            barrier.markRanked();
        }
        else
        {
            holdUntilRanked = true;
            static_cast<void>(manager.unsortedCatalog(context));
            holdUntilRanked = false;
        }
    });

    ASSERT_FALSE(barrier.timedOut())
        << "the forced interleaving did not happen, so this run proves nothing about D3";

    const auto callsAfterRace = scoreCalls().load(std::memory_order_relaxed);
    static_cast<void>(manager.sortedCatalog(context));

    EXPECT_EQ(scoreCalls().load(std::memory_order_relaxed), callsAfterRace)
        << "the cached ranking was discarded, so this query had to rank again";

    rankingBarrier() = nullptr;
    GraphMatchRegistry::unregisterSymbol(BARRIER_GRAPH_SYMBOL);
    KernelMatcherRegistry::unregisterSymbol(KERNEL_MATCH_SYMBOL);
    ScoreRegistry::unregisterSymbol(COUNTING_SCORE_SYMBOL);
}

TEST(TestIngestorRegistryConcurrency, ResolvesFromManyThreads)
{
    const ScopedTestSymbols symbols;

    runConcurrently(THREAD_COUNT, [](int) {
        for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
        {
            EXPECT_NE(GraphMatchRegistry::resolve(std::string(GRAPH_MATCH_SYMBOL)), nullptr);
            EXPECT_NE(ScoreRegistry::resolve(std::string(SCORE_SYMBOL)), nullptr);
        }
    });
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
