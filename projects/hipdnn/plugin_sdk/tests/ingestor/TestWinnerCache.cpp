// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "ContentCarryingTestGraph.hpp"
#include "KernelIngestorTestFixtures.hpp"

namespace hipdnn_plugin_sdk::ingestor::testing
{
namespace
{

using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;

KernelDefinition definitionFor(uint8_t kernel, uint8_t pack = 0xF0, uint8_t dispatch = 0xD0)
{
    KernelDefinition definition;
    definition.kernelId = testId(kernel);
    definition.packId = testId(pack);
    definition.dispatchId = testId(dispatch);
    return definition;
}

RankedEntry entryFor(const KernelDefinition& definition, double timeMs)
{
    return RankedEntry{definition.kernelId, definition.packId, definition.dispatchId, timeMs};
}

TEST(TestIngestorWinnerCache, ARecordCoveringEveryCandidateIsCovered)
{
    const auto first = definitionFor(0x01);
    const auto second = definitionFor(0x02);
    const WinnerRecord record{entryFor(second, 1.0), entryFor(first, 2.0)};

    EXPECT_TRUE(recordCovers(record, {first, second}));
}

TEST(TestIngestorWinnerCache, ARecordMissingACandidateIsNotCovered)
{
    const auto first = definitionFor(0x01);
    const auto second = definitionFor(0x02);
    const WinnerRecord record{entryFor(first, 2.0)};

    EXPECT_FALSE(recordCovers(record, {first, second}));
}

// The asymmetry is the point: a wider record still covers the candidate set. Treating
// extra entries as a failure would re-benchmark on every narrowed knob filter.
TEST(TestIngestorWinnerCache, ARecordWiderThanTheCandidateSetStillCoversIt)
{
    const auto first = definitionFor(0x01);
    const auto second = definitionFor(0x02);
    const WinnerRecord record{entryFor(first, 1.0), entryFor(second, 2.0)};

    EXPECT_TRUE(recordCovers(record, {first}));
}

TEST(TestIngestorWinnerCache, AnEmptyCandidateSetIsVacuouslyCovered)
{
    EXPECT_TRUE(recordCovers(WinnerRecord{entryFor(definitionFor(0x01), 1.0)}, {}));
}

TEST(TestIngestorWinnerCache, AnEmptyRecordCoversNothing)
{
    EXPECT_FALSE(recordCovers(WinnerRecord{}, {definitionFor(0x01)}));
}

TEST(TestIngestorWinnerCache, OrderByRecordPutsCandidatesIntoMeasuredOrder)
{
    const auto slow = definitionFor(0x01);
    const auto fast = definitionFor(0x02);
    const WinnerRecord record{entryFor(fast, 0.5), entryFor(slow, 5.0)};

    const auto ordered = orderByRecord(record, {slow, fast});

    ASSERT_EQ(ordered.size(), 2U);
    EXPECT_EQ(ordered[0].kernelId, fast.kernelId) << "the measured winner must come first";
    EXPECT_EQ(ordered[1].kernelId, slow.kernelId);
}

// Coverage asks "was this kernel measured"; agreement asks "is it still the same kernel".
// A pack replaced between runs can leave the id intact while the kernel behind it moved.
TEST(TestIngestorWinnerCache, OrderByRecordSkipsAnEntryWhosePackNoLongerAgrees)
{
    const auto current = definitionFor(0x01, 0xA1);
    const WinnerRecord record{entryFor(definitionFor(0x01, 0xB2), 1.0)};

    EXPECT_TRUE(orderByRecord(record, {current}).empty())
        << "a kernel id that now resolves to a different pack is a different kernel";
}

TEST(TestIngestorWinnerCache, OrderByRecordSkipsAnEntryWhoseDispatchNoLongerAgrees)
{
    const auto current = definitionFor(0x01, 0xF0, 0xD1);
    const WinnerRecord record{entryFor(definitionFor(0x01, 0xF0, 0xD2), 1.0)};

    EXPECT_TRUE(orderByRecord(record, {current}).empty());
}

TEST(TestIngestorWinnerCache, OrderByRecordDropsRecordEntriesAbsentFromTheCandidates)
{
    const auto present = definitionFor(0x01);
    const WinnerRecord record{entryFor(definitionFor(0x02), 0.5), entryFor(present, 1.0)};

    const auto ordered = orderByRecord(record, {present});

    ASSERT_EQ(ordered.size(), 1U);
    EXPECT_EQ(ordered[0].kernelId, present.kernelId);
}

// Rank 0's kernelId is present in the candidates, but its pack no longer agrees; the
// entry must be skipped (not treated as terminal), letting rank 1 take over. Distinct
// from the single-entry skip tests above, which only prove exhaustion.
TEST(TestIngestorWinnerCache, OrderByRecordFallsThroughToRankOneWhenRankZeroPackIsStale)
{
    const auto staleRankZero = definitionFor(0x01, 0xA1);
    const auto validRankOne = definitionFor(0x02);
    const WinnerRecord record{entryFor(definitionFor(0x01, 0xB2), 1.0),
                              entryFor(validRankOne, 2.0)};

    const auto ordered = orderByRecord(record, {staleRankZero, validRankOne});

    ASSERT_EQ(ordered.size(), 1U);
    EXPECT_EQ(ordered[0].kernelId, validRankOne.kernelId)
        << "a stale rank 0 must fall through to rank 1, not empty the result";
}

// Same fall-through, but the disagreement is on dispatchId instead of packId.
TEST(TestIngestorWinnerCache, OrderByRecordFallsThroughToRankOneWhenRankZeroDispatchIsStale)
{
    const auto staleRankZero = definitionFor(0x01, 0xF0, 0xD1);
    const auto validRankOne = definitionFor(0x02);
    const WinnerRecord record{entryFor(definitionFor(0x01, 0xF0, 0xD2), 1.0),
                              entryFor(validRankOne, 2.0)};

    const auto ordered = orderByRecord(record, {staleRankZero, validRankOne});

    ASSERT_EQ(ordered.size(), 1U);
    EXPECT_EQ(ordered[0].kernelId, validRankOne.kernelId)
        << "a stale rank 0 must fall through to rank 1, not empty the result";
}

TEST(TestIngestorWinnerCache, KeysDifferingOnlyInDeviceAreDistinct)
{
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    DeviceProperties first;
    first.gcnArchName = "gfx942";
    DeviceProperties second;
    second.gcnArchName = "gfx950";

    const WinnerKey firstKey{GraphContentKey{graph}, DeviceKey{first}};
    const WinnerKey secondKey{GraphContentKey{graph}, DeviceKey{second}};

    EXPECT_NE(firstKey, secondKey);
    EXPECT_NE(WinnerKeyHash{}(firstKey), WinnerKeyHash{}(secondKey));
}

TEST(TestIngestorWinnerCache, KeysDifferingOnlyInGraphAreDistinct)
{
    ContentCarryingTestGraph::Spec narrow;
    narrow.tensors[0].dims = {4, 8};
    ContentCarryingTestGraph::Spec wide;
    wide.tensors[0].dims = {4, 16};

    DeviceProperties properties;
    properties.gcnArchName = "gfx942";

    const WinnerKey firstKey{GraphContentKey{ContentCarryingTestGraph{narrow}},
                             DeviceKey{properties}};
    const WinnerKey secondKey{GraphContentKey{ContentCarryingTestGraph{wide}},
                              DeviceKey{properties}};

    EXPECT_NE(firstKey, secondKey);
}

TEST(TestIngestorWinnerCache, EqualGraphAndDeviceProduceEqualKeys)
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";

    const WinnerKey firstKey{GraphContentKey{ContentCarryingTestGraph{}}, DeviceKey{properties}};
    const WinnerKey secondKey{GraphContentKey{ContentCarryingTestGraph{}}, DeviceKey{properties}};

    EXPECT_EQ(firstKey, secondKey);
    EXPECT_EQ(WinnerKeyHash{}(firstKey), WinnerKeyHash{}(secondKey));
}

// ---------------------------------------------------------------------------
// The cache inside KernelIngestorStateManager: no eviction, the soft threshold,
// Check 1, and thread safety (D8, D13, D16)
// ---------------------------------------------------------------------------

/// Builds a distinct key per index without needing a distinct graph: the device half is
/// enough to separate them, and it keeps the loop cheap.
WinnerKey keyForIndex(const ContentCarryingTestGraph& graph, int index)
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    properties.multiProcessorCount = index;
    return WinnerKey{GraphContentKey{graph}, DeviceKey{properties}};
}

/// THE NO-EVICTION REGRESSION GUARD: the winner cache sits beside an LruCache in the
/// same class, so "tidying" it into that neighbour is a live temptation. Evicting a
/// winner costs a GPU sweep, not a rematch.
TEST(TestIngestorWinnerCacheStateManager, TheEarliestEntrySurvivesFarPastAnyPlausibleLruCapacity)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto first = keyForIndex(graph, 0);
    const WinnerRecord record{entryFor(definitionFor(0x01), 1.0)};

    manager->recordWinner(first, record);

    // Well past DEFAULT_CATALOG_CACHE_CAPACITY (256), which is what an LruCache here
    // would have been sized at.
    for(int index = 1; index <= 1000; ++index)
    {
        manager->recordWinner(keyForIndex(graph, index), record);
    }

    EXPECT_EQ(manager->winnerCacheSize(), 1001U);
    EXPECT_TRUE(manager->winnerFor(first).has_value())
        << "the first entry must still be served after 1000 later insertions";
}

TEST(TestIngestorWinnerCacheStateManager, RecordingTheSameKeyTwiceReplacesRatherThanAccumulates)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto key = keyForIndex(graph, 7);

    manager->recordWinner(key, WinnerRecord{entryFor(definitionFor(0x01), 5.0)});
    manager->recordWinner(
        key, WinnerRecord{entryFor(definitionFor(0x02), 1.0), entryFor(definitionFor(0x03), 2.0)});

    EXPECT_EQ(manager->winnerCacheSize(), 1U);
    const auto stored = manager->winnerFor(key);
    ASSERT_TRUE(stored.has_value());
    ASSERT_EQ(stored->size(), 2U) << "the later, wider sweep must win";
    EXPECT_EQ(stored->front().kernelId, testId(0x02));
}

TEST(TestIngestorWinnerCacheStateManager, AnEmptyRecordIsNotStored)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};

    manager->recordWinner(keyForIndex(graph, 1), WinnerRecord{});

    EXPECT_EQ(manager->winnerCacheSize(), 0U)
        << "an all-unusable sweep has no ranking; storing one would read as a covered hit";
}

TEST(TestIngestorWinnerCacheStateManager, AMissReturnsNulloptRatherThanAnEmptyRecord)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};

    EXPECT_FALSE(manager->winnerFor(keyForIndex(graph, 99)).has_value());
}

/// D8's soft threshold has no observable effect other than its log line, so the log
/// assertion is the only possible test of it.
TEST(TestIngestorWinnerCacheStateManager, TheGrowthWarningFiresOnceAndOnlyPastTheThreshold)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const WinnerRecord record{entryFor(definitionFor(0x01), 1.0)};

    // Fill to exactly the threshold: indices 0..threshold-1 is `threshold` entries, and
    // the warning fires only once size exceeds it.
    const auto threshold = StateManager::WINNER_CACHE_WARNING_THRESHOLD;
    for(size_t index = 0; index < threshold; ++index)
    {
        manager->recordWinner(keyForIndex(graph, static_cast<int>(index)), record);
    }
    ASSERT_EQ(manager->winnerCacheSize(), threshold);
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "past the soft threshold"))
        << "the warning must not fire at exactly the threshold:\n"
        << recorder.getRecordedLogsAsString();

    manager->recordWinner(keyForIndex(graph, static_cast<int>(threshold)), record);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "past the soft threshold"))
        << recorder.getRecordedLogsAsString();

    // And it stays quiet afterwards rather than re-logging on every later insert.
    const auto afterFirstWarning = recorder.getRecordedLogsAsString();
    manager->recordWinner(keyForIndex(graph, static_cast<int>(threshold) + 2), record);
    EXPECT_EQ(recorder.getRecordedLogsAsString(), afterFirstWarning)
        << "the growth warning is reported once, not per insertion";
}

/// Check 1: a record covering the WHOLE catalog orders it, and rank() is never
/// consulted. The heuristic is rigged to invert the order to make this observable.
TEST(TestIngestorWinnerCacheStateManager, ACoveringRecordOrdersTheCatalogWithoutTheHeuristic)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0xE1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto catalog = manager->sortedDefinitions(context);
    ASSERT_GE(catalog.size(), 2U) << "this test needs at least two candidates to reorder";

    // Record the heuristic's order reversed, then assert selection follows the record.
    WinnerRecord record;
    double time = 1.0;
    for(auto entry = catalog.rbegin(); entry != catalog.rend(); ++entry)
    {
        record.push_back(entryFor(*entry, time));
        time += 1.0;
    }

    const auto freshManager = makeStateManager();
    freshManager->recordWinner(WinnerKey{GraphContentKey{graph}, DeviceKey{properties}}, record);
    const auto ordered = freshManager->sortedDefinitions(context);

    ASSERT_EQ(ordered.size(), catalog.size());
    EXPECT_EQ(ordered.front().kernelId, catalog.back().kernelId)
        << "a covering record must decide the order, not the heuristic";
}

/// The production sequence, on ONE manager: sort (heuristic, memoized), then record,
/// then sort again. A benchmark sweep always writes after buildPlan already sorted and
/// cached the catalog, so a memoized heuristic order must yield to a later measurement.
TEST(TestIngestorWinnerCacheStateManager, ARecordAdoptedAfterTheCatalogWasAlreadySorted)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const TestGraph graph(makeGraphId(0xE7));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    // First sort: no record exists, so this is the heuristic order, and it is memoized.
    const auto heuristicOrder = manager->sortedDefinitions(context);
    ASSERT_GE(heuristicOrder.size(), 2U) << "this test needs at least two candidates";

    // The sweep finishes and writes a record that reverses that order.
    WinnerRecord reversed;
    double time = 1.0;
    for(auto entry = heuristicOrder.rbegin(); entry != heuristicOrder.rend(); ++entry)
    {
        reversed.push_back(entryFor(*entry, time));
        time += 1.0;
    }
    manager->recordWinner(WinnerKey{GraphContentKey{graph}, DeviceKey{properties}}, reversed);

    // Second sort, same manager: the measured order must now win.
    const auto measuredOrder = manager->sortedDefinitions(context);

    ASSERT_EQ(measuredOrder.size(), heuristicOrder.size());
    EXPECT_EQ(measuredOrder.front().kernelId, heuristicOrder.back().kernelId)
        << "a memoized heuristic order must yield to a measurement that arrives later";

    // And a third call is stable: once ordered from a record, it stays that way.
    const auto thirdCall = manager->sortedDefinitions(context);
    EXPECT_EQ(thirdCall.front().kernelId, measuredOrder.front().kernelId);
}

/// Check 1 fails on a partial record: the heuristic still ranks, because interleaving
/// measured entries with unmeasured ones would invent an order nobody took.
TEST(TestIngestorWinnerCacheStateManager, APartialRecordLeavesTheHeuristicOrderIntact)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto reference = makeStateManager();
    const TestGraph graph(makeGraphId(0xE2));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto heuristicOrder = reference->sortedDefinitions(context);
    ASSERT_GE(heuristicOrder.size(), 2U);

    // Record only the LAST candidate, so coverage fails.
    const auto manager = makeStateManager();
    manager->recordWinner(WinnerKey{GraphContentKey{graph}, DeviceKey{properties}},
                          WinnerRecord{entryFor(heuristicOrder.back(), 0.1)});

    const auto ordered = manager->sortedDefinitions(context);

    ASSERT_EQ(ordered.size(), heuristicOrder.size());
    EXPECT_EQ(ordered.front().kernelId, heuristicOrder.front().kernelId)
        << "an uncovering record must not reorder anything";
}

/// D8 puts the cache under its own mutex. Concurrent readers and writers must neither
/// race nor lose entries; this is the guard for someone removing the lock as "unneeded".
TEST(TestIngestorWinnerCacheStateManager, ConcurrentWritersAndReadersKeepEveryEntry)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const WinnerRecord record{entryFor(definitionFor(0x01), 1.0)};

    constexpr int THREADS = 8;
    constexpr int PER_THREAD = 250;
    std::vector<std::thread> threads;
    threads.reserve(THREADS);

    for(int thread = 0; thread < THREADS; ++thread)
    {
        threads.emplace_back([&manager, &graph, &record, thread]() {
            for(int index = 0; index < PER_THREAD; ++index)
            {
                const int unique = thread * PER_THREAD + index;
                manager->recordWinner(keyForIndex(graph, unique), record);
                // Interleave reads so writers and readers genuinely overlap.
                (void)manager->winnerFor(keyForIndex(graph, unique));
                (void)manager->winnerCacheSize();
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(manager->winnerCacheSize(), static_cast<size_t>(THREADS * PER_THREAD))
        << "every distinct key must survive concurrent insertion";
    for(int unique = 0; unique < THREADS * PER_THREAD; ++unique)
    {
        ASSERT_TRUE(manager->winnerFor(keyForIndex(graph, unique)).has_value())
            << "entry " << unique << " was lost under concurrency";
    }
}

/// Readers and writers contend on ONE key -- disjoint keys would make this a crash
/// canary rather than a lock test. `winnerFor` returns a copy taken under the lock, so
/// every read must be one whole record, never a mixture of two.
TEST(TestIngestorWinnerCacheStateManager, ConcurrentReadsOfOneKeyNeverSeeATornRecord)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto contended = keyForIndex(graph, 0);

    // Two records of different lengths and contents. A torn read would show a length or
    // an id belonging to neither.
    const WinnerRecord shortRecord{entryFor(definitionFor(0x01), 1.0)};
    const WinnerRecord longRecord{entryFor(definitionFor(0x02), 1.0),
                                  entryFor(definitionFor(0x03), 2.0),
                                  entryFor(definitionFor(0x04), 3.0)};
    manager->recordWinner(contended, shortRecord);

    std::atomic<bool> torn{false};
    std::atomic<int> reads{0};
    std::vector<std::thread> threads;
    threads.reserve(8);

    for(int writer = 0; writer < 4; ++writer)
    {
        threads.emplace_back([&, writer]() {
            for(int index = 0; index < 500; ++index)
            {
                manager->recordWinner(contended,
                                      (writer + index) % 2 == 0 ? shortRecord : longRecord);
            }
        });
    }
    for(int reader = 0; reader < 4; ++reader)
    {
        threads.emplace_back([&]() {
            for(int index = 0; index < 500; ++index)
            {
                const auto seen = manager->winnerFor(contended);
                if(!seen.has_value())
                {
                    torn = true;
                    continue;
                }
                reads.fetch_add(1, std::memory_order_relaxed);

                const bool isShort = seen->size() == 1 && seen->front().kernelId == testId(0x01);
                const bool isLong = seen->size() == 3 && seen->front().kernelId == testId(0x02)
                                    && seen->back().kernelId == testId(0x04);
                if(!isShort && !isLong)
                {
                    torn = true;
                }
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_FALSE(torn) << "every read must be one whole record, never a mixture";
    EXPECT_EQ(reads.load(), 2000) << "the key exists throughout, so no read may miss";
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
