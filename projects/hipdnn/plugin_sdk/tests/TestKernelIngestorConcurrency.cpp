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

/// Ceiling on every wait in this file.
///
/// The waits here are on thread scheduling, not on work: the whole suite runs in ~20ms,
/// and ~20ms under deliberate CPU oversubscription. 30s is roughly three orders of
/// magnitude of headroom, so reaching it means a thread never ran at all, which is a
/// deadlock rather than a slow machine. Kept far below ctest's 1500s default so a
/// genuine deadlock fails as a named assertion here rather than as a killed binary with
/// no diagnosis.
constexpr auto WAIT_TIMEOUT = std::chrono::seconds(30);

/// A one-shot gate several threads block on until it is opened.
///
/// Blocking rather than spinning. A spin-wait keeps every waiting thread on a core,
/// competing with the threads doing the work under test and, on a small CI runner, with
/// the thread trying to open the gate. Blocked threads leave the runqueue, so the
/// opener runs immediately and the release is closer to simultaneous.
class Gate
{
public:
    /// @return false on timeout, meaning the gate was never opened.
    bool wait()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _cv.wait_for(lock, WAIT_TIMEOUT, [this] { return _open; });
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

/// Runs @p body on threadCount threads, releasing them together so they truly overlap.
template <typename Body>
void runConcurrently(int threadCount, Body body)
{
    Gate start;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(threadCount));

    for(int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back([&start, &body, i]() {
            // On timeout, run anyway: the test then fails on its own assertions rather
            // than deadlocking in join() below.
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

/// A two-thread rendezvous plus a one-way "ranking installed" signal, both bounded.
///
/// Not the Gate above: this is reached from inside a graph matcher called by library
/// code, and its release condition is another thread arriving at the same line rather
/// than an external open(). std::barrier would express the rendezvous directly but is
/// C++20; this file is C++17.
class RankingBarrier
{
public:
    /// Blocks until both threads have arrived. @return false on timeout.
    bool arriveAndWait()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        ++_arrived;
        _cv.notify_all();
        return _cv.wait_for(lock, WAIT_TIMEOUT, [this] { return _arrived >= 2; });
    }

    /// Blocks until markRanked(). @return false on timeout.
    bool waitForRanked()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _cv.wait_for(lock, WAIT_TIMEOUT, [this] { return _ranked; });
    }

    void markRanked()
    {
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _ranked = true;
        }
        _cv.notify_all();
    }

    /// True if any wait timed out. The test asserts on this: a timeout means the
    /// interleaving never happened, and the scorer-count assertion below would then
    /// pass for the wrong reason.
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

/// The barrier the registered matcher below reaches. A GraphMatcherFn is a plain
/// function pointer with no state of its own, so the matcher needs a way to find it;
/// the test sets this before registering and clears it after.
RankingBarrier*& rankingBarrier()
{
    static RankingBarrier* s_barrier = nullptr;
    return s_barrier;
}

TEST(TestIngestorStateManagerConcurrency, ARankingSurvivesAConcurrentUnsortedAccess)
{
    // D3, and the only interleaving that reaches it. Both threads must miss one key and
    // be inside buildCatalog() together, and the unsorted writer must land after the
    // ranking is installed. Single-threaded the defect is unreachable, which is why the
    // state-manager suite covers only the half of D3 it can see.
    //
    // Both conditions are forced rather than raced for, so this fails every run against
    // the defect instead of occasionally. Every wait is bounded: an unbounded one turns
    // a future refactor that stops both threads reaching the matcher into a hung CI job
    // rather than a failing test.
    constexpr const char* BARRIER_GRAPH_SYMBOL = "test.d3.barrier_graph_match";
    constexpr const char* COUNTING_SCORE_SYMBOL = "test.d3.counting_score";

    RankingBarrier barrier;
    rankingBarrier() = &barrier;
    scoreCalls().store(0);

    GraphMatcherRegistry::registerSymbol(
        BARRIER_GRAPH_SYMBOL, [](const MatchContext&, BoundTokens&) -> bool {
            // Neither thread leaves until both are here: this is what makes them miss
            // the cache together rather than one serving the other.
            if(!rankingBarrier()->arriveAndWait())
            {
                rankingBarrier()->recordTimeout();
                return true;
            }

            // The unsorted thread then waits for the ranking to be installed, so its
            // own write is the one that lands last, the case that used to clobber.
            if(holdUntilRanked && !rankingBarrier()->waitForRanked())
            {
                rankingBarrier()->recordTimeout();
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
        makeTestDispatches<TestHandle>(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(COUNTING_SCORE_SYMBOL));

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

    // Checked before the real assertion, and separately from it. A timed-out barrier
    // means the interleaving never happened, and without the ordering it forces the
    // scorer-count check below passes for the wrong reason. Silence there would read as
    // proof the defect is fixed.
    ASSERT_FALSE(barrier.timedOut())
        << "the forced interleaving did not happen, so this run proves nothing about D3";

    // Asserting isSorted here would prove nothing: sortedCatalog() ranks on demand, so
    // it reads true whether or not the cached entry survived. Whether the scorer runs
    // again is the observable difference, and re-ranking every query is exactly the
    // thrash D3 describes.
    const auto callsAfterRace = scoreCalls().load(std::memory_order_relaxed);
    static_cast<void>(manager.sortedCatalog(context));

    EXPECT_EQ(scoreCalls().load(std::memory_order_relaxed), callsAfterRace)
        << "the cached ranking was discarded, so this query had to rank again";

    rankingBarrier() = nullptr;
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
