// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>
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
// The cache inside KernelIngestorStateManager: no eviction, the soft growth-warning
// threshold, ordering a covering record, and thread safety
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

/// A key whose graph yields no bytes matches nothing on lookup, so storing under one
/// would strand an entry the cache can never serve or evict.
TEST(TestIngestorWinnerCacheStateManager, ARecordUnderAnUnusableGraphKeyIsNotStored)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();

    // Supplies no bytes(), taking IGraph's default: valid or not, it cannot be keyed.
    class BytelessGraph : public hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph
    {
    public:
        const hipdnn_flatbuffers_sdk::data_objects::Graph& getGraph() const override
        {
            throw std::logic_error("BytelessGraph carries no graph");
        }
        bool isValid() const override
        {
            return false;
        }
        uint32_t nodeCount() const override
        {
            return 0;
        }
        bool hasOnlySupportedAttributes(
            std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes> /*supported*/)
            const override
        {
            return false;
        }
        const hipdnn_flatbuffers_sdk::data_objects::Node& getNode(uint32_t /*index*/) const override
        {
            throw std::logic_error("BytelessGraph carries no nodes");
        }
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper&
            getNodeWrapper(uint32_t /*index*/) const override
        {
            throw std::logic_error("BytelessGraph carries no nodes");
        }
        const std::vector<
            std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper>>&
            nodeWrappers() const override
        {
            throw std::logic_error("BytelessGraph carries no nodes");
        }
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            getTensorMap() const override
        {
            throw std::logic_error("BytelessGraph carries no tensors");
        }
    };

    const BytelessGraph graph;
    DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    properties.multiProcessorCount = 304;
    const WinnerKey key{GraphContentKey{graph}, DeviceKey{properties}};
    ASSERT_FALSE(key.graph.isUsable()) << "the fixture must actually be unkeyable";

    manager->recordWinner(key, WinnerRecord{entryFor(definitionFor(0x01), 1.0)});

    EXPECT_EQ(manager->winnerCacheSize(), 0U)
        << "an unkeyable graph never matches on lookup, so the entry would be unreachable";
}

TEST(TestIngestorWinnerCacheStateManager, AMissReturnsNulloptRatherThanAnEmptyRecord)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};

    EXPECT_FALSE(manager->winnerFor(keyForIndex(graph, 99)).has_value());
}

/// The soft threshold has no observable effect other than its log line, so the log
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

/// A record covering the WHOLE catalog orders it, and rank() is never consulted. The
/// heuristic is rigged to invert the order to make this observable.
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

/// A partial record leaves the heuristic order intact: interleaving measured entries
/// with unmeasured ones would invent an order nobody took.
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

/// The cache sits under its own mutex. Concurrent readers and writers must neither race
/// nor lose entries; this is the guard for someone removing the lock as "unneeded".
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

/// A value no other live ScopedCacheDir shares, in this process or any other: a
/// steady-clock reading separates concurrent processes, the counter separates two
/// guards within one process. Not getpid(): <unistd.h> does not exist on Windows.
inline std::string uniqueSuffix()
{
    static std::atomic<uint64_t> s_counter{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::to_string(static_cast<uint64_t>(stamp)) + "_"
           + std::to_string(s_counter.fetch_add(1));
}

/// Points HIPDNN_CACHE_DIR at a directory of this test's own for the guard's lifetime and
/// restores the caller's value after.
///
/// Also neutralizes HIPDNN_DISABLE_CACHE for the same window. An ambient truthy value in
/// the runner's environment makes cacheRoot() return empty before it reads
/// HIPDNN_CACHE_DIR at all, which silently turns every disk test below into a test of the
/// in-memory fallback.
class ScopedCacheDir
{
public:
    explicit ScopedCacheDir(const std::string& name)
        : _previous(hipdnn_data_sdk::utilities::getEnv(CACHE_DIR_ENV))
        , _previousDisable(hipdnn_data_sdk::utilities::getEnv(DISABLE_CACHE_ENV))
        , _path(std::filesystem::temp_directory_path()
                / ("hipdnn_winner_cache_test_" + uniqueSuffix() + "_" + name))
    {
        std::filesystem::remove_all(_path);
        std::filesystem::create_directories(_path);
        const std::string value = _path.string();
        hipdnn_data_sdk::utilities::setEnv(CACHE_DIR_ENV, value.c_str());
        hipdnn_data_sdk::utilities::unsetEnv(DISABLE_CACHE_ENV);
    }

    ScopedCacheDir(const ScopedCacheDir&) = delete;
    ScopedCacheDir& operator=(const ScopedCacheDir&) = delete;
    ScopedCacheDir(ScopedCacheDir&&) = delete;
    ScopedCacheDir& operator=(ScopedCacheDir&&) = delete;

    ~ScopedCacheDir()
    {
        if(_previous.empty())
        {
            hipdnn_data_sdk::utilities::unsetEnv(CACHE_DIR_ENV);
        }
        else
        {
            hipdnn_data_sdk::utilities::setEnv(CACHE_DIR_ENV, _previous.c_str());
        }
        if(!_previousDisable.empty())
        {
            hipdnn_data_sdk::utilities::setEnv(DISABLE_CACHE_ENV, _previousDisable.c_str());
        }
        std::error_code ignored;
        std::filesystem::remove_all(_path, ignored);
    }

    const std::filesystem::path& path() const
    {
        return _path;
    }

private:
    static constexpr const char* CACHE_DIR_ENV = "HIPDNN_CACHE_DIR";
    static constexpr const char* DISABLE_CACHE_ENV = "HIPDNN_DISABLE_CACHE";

    std::string _previous;
    std::string _previousDisable;
    std::filesystem::path _path;
};

/// A device whose arch carries the feature suffix a real gfx942 reports.
DeviceProperties suffixedDeviceProperties(int multiProcessorCount = 304)
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx942:sramecc+:xnack-";
    properties.warpSize = 64;
    properties.multiProcessorCount = multiProcessorCount;
    return properties;
}

WinnerKey keyFor(const ContentCarryingTestGraph& graph, const DeviceProperties& properties)
{
    return WinnerKey{GraphContentKey{graph}, DeviceKey{properties}};
}

/// A ranking a manager can record, distinct per @p kernel so two records do not alias.
WinnerRecord recordFor(uint8_t kernel, double timeMs)
{
    return WinnerRecord{entryFor(definitionFor(kernel), timeMs)};
}

/// Proves the codec plus the read-once path across two manager lifetimes; the
/// cross-process case is a separate ctest-driven pair below.
TEST(TestIngestorWinnerCacheStateManager, ARecordSurvivesIntoAFreshManagerThroughTheShard)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("cross_instance");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);

    {
        const auto writer = makeNamedStateManager("test:CrossInstance");
        writer->recordWinner(key, recordFor(0x11, 1.5));
    }

    const auto reader = makeNamedStateManager("test:CrossInstance");
    const auto served = reader->winnerFor(key);

    ASSERT_TRUE(served.has_value())
        << "a fresh manager did not read back the record the previous one wrote";
    ASSERT_EQ(served->size(), 1U);
    EXPECT_EQ(served->front().kernelId, testId(0x11));
}

/// gcnArchName is raw, driver-supplied and suffixed. ALMIOPEN-2451's design record
/// requires the arch directory to be the stripped base target id and to stay readable, so
/// a user can find and delete one arch's cache by eye: `gfx942`, not `gfx942-<hash>` and
/// not an opaque digest. Run on Linux CI, where a Windows-only break in the ':' handling
/// would otherwise never surface.
///
/// Falsifying mutation: wrap the arch in sanitizeForPath(). The exact-equality check below
/// fails on the appended hash suffix.
TEST(TestIngestorWinnerCache, TheArchShardComponentIsTheReadableBaseTargetId)
{
    const ScopedCacheDir cacheDir("arch_component");

    const auto path = winnerCacheShardPath("test:ArchComponent", "gfx942:sramecc+:xnack-");
    ASSERT_FALSE(path.empty());

    EXPECT_EQ(path.parent_path().filename().string(), "gfx942")
        << "the arch directory must be the readable stripped base id, verbatim";

    // The engine component is sanitized: a conforming scoped name contains a colon, which
    // is illegal on Windows, and an engine name has no restricted alphabet to lean on.
    const auto engineComponent = path.parent_path().parent_path().filename().string();
    EXPECT_EQ(engineComponent.find(':'), std::string::npos);
}

/// The arch is validated, not reshaped: a string that is not a plain path component is a
/// driver anomaly, and declining the disk cache is safer than folding separators into a
/// name that reads like a different arch. Callers already treat an empty path as
/// in-memory-only.
///
/// Falsifying mutation: drop the isPlainArchComponent() guard in winnerCacheShardPath().
/// Every case below then yields a non-empty path, and the first walks out of the tree.
TEST(TestIngestorWinnerCache, AnArchThatIsNotAPlainComponentDeclinesTheShard)
{
    const ScopedCacheDir cacheDir("arch_reject");

    for(const std::string_view hostile :
        {"../../../evil", "gfx942/../../escape", "gfx942\\evil", ".", "..", "", "gfx942 evil"})
    {
        EXPECT_TRUE(winnerCacheShardPath("test:ArchComponent", hostile).empty())
            << "an arch that is not a plain path component produced a shard path: \"" << hostile
            << "\"";
    }
}

/// The shard is per-arch but the key is per-device: two parts reporting the same arch with
/// different compute-unit counts do not share a measurement, yet do share a file.
TEST(TestIngestorWinnerCacheStateManager, TwoDevicesOnOneArchShareAShardAndStayDistinctRecords)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("coarse_shard");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto fewerUnits = suffixedDeviceProperties(228);
    const auto moreUnits = suffixedDeviceProperties(304);
    ASSERT_NE(DeviceKey{fewerUnits}, DeviceKey{moreUnits});

    {
        const auto manager = makeNamedStateManager("test:CoarseShard");
        manager->recordWinner(keyFor(graph, fewerUnits), recordFor(0x21, 1.0));
        manager->recordWinner(keyFor(graph, moreUnits), recordFor(0x22, 2.0));
    }

    const auto path = winnerCacheShardPath("test:CoarseShard", fewerUnits.gcnArchName);
    ASSERT_EQ(path, winnerCacheShardPath("test:CoarseShard", moreUnits.gcnArchName));
    ASSERT_TRUE(std::filesystem::exists(path));

    const auto reader = makeNamedStateManager("test:CoarseShard");
    const auto forFewerUnits = reader->winnerFor(keyFor(graph, fewerUnits));
    const auto forMoreUnits = reader->winnerFor(keyFor(graph, moreUnits));
    ASSERT_TRUE(forFewerUnits.has_value());
    ASSERT_TRUE(forMoreUnits.has_value());
    EXPECT_EQ(forFewerUnits->front().kernelId, testId(0x21));
    EXPECT_EQ(forMoreUnits->front().kernelId, testId(0x22));
}

/// A shard written by a different build is declined rather than misread, and says so once.
TEST(TestIngestorWinnerCacheStateManager, AVersionMismatchedShardIsDeclinedAndLogged)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("version_mismatch");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();

    const auto path = winnerCacheShardPath("test:VersionMismatch", properties.gcnArchName);
    ASSERT_FALSE(path.empty());
    std::filesystem::create_directories(path.parent_path());
    {
        std::ofstream out(path);
        out << "not-the-version-this-build-writes\n";
    }

    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    const auto manager = makeNamedStateManager("test:VersionMismatch");

    EXPECT_FALSE(manager->winnerFor(keyFor(graph, properties)).has_value())
        << "a version-mismatched shard must decline, not serve";
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "version mismatch"))
        << "the decline must say why, rather than failing silently:\n"
        << recorder.getRecordedLogsAsString();
}

/// One unparseable line costs itself and nothing else -- a shard is not poisoned by a
/// record a newer build wrote or a partial write left behind. The second good record
/// lives AFTER the garbage line specifically: an abort-on-first-bad `readAllLines()`
/// (e.g. `break` instead of `continue`/skip on a decode failure) would still pass the
/// first assertion below, since that record predates the garbage -- it can only be
/// caught by a record the reader must continue PAST the bad line to reach.
///
/// Falsifying mutation: change `readAllLines()`'s malformed-line handling from "skip
/// this line, keep scanning" to "stop at the first line `parseLine` rejects" (e.g. add
/// `break;` where the `if(auto parsed = parseLine(...))` fails). `laterKey`'s record
/// below is then never decoded.
TEST(TestIngestorWinnerCacheStateManager, AMalformedLineCostsOnlyItself)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("malformed_line");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);
    // A second, distinct key (same shard: same engine/arch, different device), whose
    // record is written after the garbage line.
    const auto laterProperties = suffixedDeviceProperties(400);
    const auto laterKey = keyFor(graph, laterProperties);

    {
        const auto writer = makeNamedStateManager("test:MalformedLine");
        writer->recordWinner(key, recordFor(0x31, 1.0));
    }

    const auto path = winnerCacheShardPath("test:MalformedLine", properties.gcnArchName);
    ASSERT_TRUE(std::filesystem::exists(path));
    {
        std::ofstream out(path, std::ios::app);
        out << "{not valid json at all\n";
        out << encodeWinnerRecordLine(laterKey, recordFor(0x32, 2.0)) << "\n";
    }

    const auto reader = makeNamedStateManager("test:MalformedLine");
    const auto served = reader->winnerFor(key);
    ASSERT_TRUE(served.has_value()) << "the good line before the garbage one must still load";
    EXPECT_EQ(served->front().kernelId, testId(0x31));

    const auto servedAfterGarbage = reader->winnerFor(laterKey);
    ASSERT_TRUE(servedAfterGarbage.has_value())
        << "the good line AFTER the garbage one must still load -- an implementation "
           "that aborts on the first bad line loses it";
    EXPECT_EQ(servedAfterGarbage->front().kernelId, testId(0x32));
}

/// A manager built without an engine name has no shard path to compose, so it neither
/// reads nor writes even with a writable cache directory configured.
TEST(TestIngestorWinnerCacheStateManager, AnUnnamedEngineTouchesNoFile)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("unnamed_engine");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);

    const auto manager = makeStateManager();
    manager->recordWinner(key, recordFor(0x41, 1.0));

    EXPECT_TRUE(manager->winnerFor(key).has_value());

    EXPECT_TRUE(std::filesystem::is_empty(cacheDir.path()))
        << "an unnamed engine wrote into the cache directory; every directly-constructed "
           "test manager would then share one shard";
}

/// B1.1 -- the in-process half of A2's fix: N threads call `recordWinner()` on ONE
/// manager, half racing a shared key. Counts RAW LINES in the shard file directly,
/// never through a manager: a manager-mediated count can never see a duplicate append,
/// because duplicates collapse last-wins when merged into `_winnerCache`, and
/// `readAllLines()` (used by `winnerFor`) also only ever returns the merged view, not a
/// per-line tally. Every racing thread on the shared key writes a byte-identical
/// `(kernelId, packId, dispatchId)` sequence (`timeMs` varies), matching
/// `rankedIdsEqual()`'s ignore-timeMs comparison -- otherwise `writeBackToShard()` would
/// legitimately append a superseding line per thread and the expected count would not
/// be fixed.
///
/// Falsifying mutation: make the in-process mutex in `acquireLineStoreLock()` a no-op,
/// leaving only the file-level fcntl lock. POSIX fcntl locks are (process, inode)-scoped,
/// not per-descriptor or per-thread: with one process-wide registry entry per shard,
/// every thread of this test shares the SAME native descriptor, so the OS lock alone
/// cannot serialize them against each other, and the shared key's raw line count exceeds
/// 1 (observed 2-35+ extra lines in a standalone repro of the exact critical section).
///
/// NOT `_winnerCacheMutex`: `recordWinner()` calls `writeBackToShard()` BEFORE taking
/// it, so the shard's own lock -- taken by `lockLineStore()` and held across the
/// read-then-append -- is what this oracle depends on. Deleting `_winnerCacheMutex`
/// races the `_winnerCache` map instead and leaves the line counts below untouched.
///
/// This guards only the in-process (mutex) half of A2; the cross-process (file-lock)
/// half, which this in-process test structurally cannot observe, is covered by B1.2 in
/// TestLineStore.cpp/LineStoreLockHelper.cpp.
TEST(TestIngestorWinnerCacheStateManager, ConcurrentRecordWinnerOnOneKeyAppendsExactlyOneLine)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("concurrent_record_winner");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto sharedKey = keyFor(graph, properties);

    constexpr int THREADS = 8;
    constexpr int ITERATIONS_PER_THREAD = 20;

    const auto manager = makeNamedStateManager("test:ConcurrentRecordWinner");

    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for(int t = 0; t < THREADS; ++t)
    {
        threads.emplace_back([&manager, &graph, sharedKey, t]() {
            for(int i = 0; i < ITERATIONS_PER_THREAD; ++i)
            {
                if(t % 2 == 0)
                {
                    // Half the threads race the SAME key with the SAME ranked ids: the
                    // only sequence rankedIdsEqual() ever treats as unchanged, so every
                    // write after the first must adopt rather than append.
                    manager->recordWinner(sharedKey, recordFor(0x71, 1.0 + i));
                }
                else
                {
                    // The other half touch distinct keys, so a shard-wide lock (rather
                    // than a per-key one) is exercised without perturbing the shared
                    // key's expected count.
                    DeviceProperties distinctProperties = suffixedDeviceProperties();
                    distinctProperties.multiProcessorCount = 1000 + t;
                    manager->recordWinner(keyFor(graph, distinctProperties),
                                          recordFor(static_cast<uint8_t>(0x80 + t), 1.0 + i));
                }
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    const auto path = winnerCacheShardPath("test:ConcurrentRecordWinner", properties.gcnArchName);
    ASSERT_TRUE(std::filesystem::exists(path));
    std::ifstream shardFile(path);
    ASSERT_TRUE(shardFile.is_open());

    std::vector<std::string> rawLines;
    for(std::string line; std::getline(shardFile, line);)
    {
        rawLines.push_back(line);
    }

    ASSERT_FALSE(rawLines.empty());
    // Exactly one version line: a broken lock lets two threads both take the
    // create-if-absent branch and both write one (LineStore.hpp's openLineStore()).
    int versionLines = 0;
    int sharedKeyRecordLines = 0;
    const std::string sharedKernelIdText = toString(testId(0x71));
    for(const auto& line : rawLines)
    {
        if(line == winnerCacheVersion())
        {
            ++versionLines;
        }
        else if(line.find(sharedKernelIdText) != std::string::npos)
        {
            ++sharedKeyRecordLines;
        }
    }
    EXPECT_EQ(versionLines, 1) << "a broken lock let two threads both stamp the version line";
    EXPECT_EQ(sharedKeyRecordLines, 1)
        << "the shared key's ranked ids never changed, so exactly one line for it may "
           "exist; a broken lock lets racing threads both observe \"absent/unchanged\" "
           "and both append -- got "
        << sharedKeyRecordLines << " raw lines:\n"
        << [&rawLines] {
               std::string joined;
               for(const auto& l : rawLines)
               {
                   joined += l;
                   joined += '\n';
               }
               return joined;
           }();

    // A separate CONTENT assertion, not the counter above: a fresh manager reads back
    // exactly the shared record, proving the merged view is also correct.
    const auto freshManager = makeNamedStateManager("test:ConcurrentRecordWinner");
    const auto served = freshManager->winnerFor(sharedKey);
    ASSERT_TRUE(served.has_value());
    ASSERT_EQ(served->size(), 1U);
    EXPECT_EQ(served->front().kernelId, testId(0x71));
}

/// B1.3 -- refresh semantics for A1: a superseding record widens the shard rather than
/// replacing the earlier line, and the winning value is the ranking last observed by
/// disk order.
///
/// Falsifying mutation: revert `writeBackToShard()`'s `rankedIdsEqual()` gate to the
/// pre-A1 "adopt whenever ANY on-disk record for the key exists" behavior (drop the
/// `&& rankedIdsEqual(*onDiskWinner, record)` check). The genuinely different
/// 4-kernel write then adopts the stale 3-kernel line instead of appending: the shard
/// stays at 2 raw lines (version + one record) instead of 3.
TEST(TestIngestorWinnerCacheStateManager, ASupersedingRecordWidensTheShardAndWins)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("superseding_record");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);

    const WinnerRecord threeKernelRecord{entryFor(definitionFor(0x81), 1.0),
                                         entryFor(definitionFor(0x82), 2.0),
                                         entryFor(definitionFor(0x83), 3.0)};
    const WinnerRecord fourKernelRecord{entryFor(definitionFor(0x81), 1.1),
                                        entryFor(definitionFor(0x82), 2.1),
                                        entryFor(definitionFor(0x83), 3.1),
                                        entryFor(definitionFor(0x84), 4.1)};

    {
        const auto writer = makeNamedStateManager("test:SupersedingRecord");
        writer->recordWinner(key, threeKernelRecord);
        writer->recordWinner(key, fourKernelRecord);
    }

    const auto path = winnerCacheShardPath("test:SupersedingRecord", properties.gcnArchName);
    ASSERT_TRUE(std::filesystem::exists(path));
    std::ifstream shardFile(path);
    ASSERT_TRUE(shardFile.is_open());
    std::vector<std::string> rawLines;
    for(std::string line; std::getline(shardFile, line);)
    {
        rawLines.push_back(line);
    }
    // version line + one line per genuinely distinct ranking.
    ASSERT_EQ(rawLines.size(), 3U) << "the shard must hold BOTH the 3-entry and the "
                                      "4-entry line, not just the latest";

    const auto reader = makeNamedStateManager("test:SupersedingRecord");
    const auto served = reader->winnerFor(key);
    ASSERT_TRUE(served.has_value());
    EXPECT_EQ(served->size(), 4U) << "winnerFor() must return the 4-entry (last-written) "
                                     "record, not the earlier 3-entry one";
}

/// B1.3 -- writing the SAME ranking twice adopts the on-disk copy both times and never
/// grows the shard.
///
/// Falsifying mutation: delete the `rankedIdsEqual()` early-return in
/// `writeBackToShard()` (always append). The shard then gains a line on the identical
/// second `recordWinner()` call, and `rawLines.size()` below observes 2 record lines
/// instead of 1.
TEST(TestIngestorWinnerCacheStateManager, WritingTheIdenticalRecordTwiceAppendsNoLine)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("identical_record_twice");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);
    const WinnerRecord record{entryFor(definitionFor(0x91), 1.0),
                              entryFor(definitionFor(0x92), 2.0)};

    const auto manager = makeNamedStateManager("test:IdenticalRecordTwice");
    manager->recordWinner(key, record);
    // Same ranked ids, different timeMs -- rankedIdsEqual() ignores timeMs, so this must
    // adopt, not append.
    manager->recordWinner(
        key, WinnerRecord{entryFor(definitionFor(0x91), 9.0), entryFor(definitionFor(0x92), 9.0)});

    const auto path = winnerCacheShardPath("test:IdenticalRecordTwice", properties.gcnArchName);
    ASSERT_TRUE(std::filesystem::exists(path));
    std::ifstream shardFile(path);
    ASSERT_TRUE(shardFile.is_open());
    std::vector<std::string> rawLines;
    for(std::string line; std::getline(shardFile, line);)
    {
        rawLines.push_back(line);
    }
    ASSERT_EQ(rawLines.size(), 2U) << "version line plus exactly one record line; the "
                                      "unchanged second write must not append";
}

/// B1.3 -- on the ADOPT path (on-disk ranking already matches by `rankedIdsEqual()`,
/// which ignores `timeMs`), the value that reaches `_winnerCache` must be the ON-DISK
/// record `writeBackToShard()` returned, not the caller's argument re-stored under its
/// own name. `timeMs` is the only field that can tell them apart here, since
/// `rankedIdsEqual()` treats them as the same ranking by design.
///
/// Falsifying mutation: in `recordWinner()`, change
/// `_winnerCache[key] = std::move(adopted);` to `_winnerCache[key] = record;` (the
/// caller's original argument). The disk's original `timeMs` (1.0) would then be
/// replaced by the second caller's `timeMs` (99.0) even though nothing was appended --
/// silently reintroducing whatever the second call's measurement was, on a ranking the
/// shard says is unchanged.
TEST(TestIngestorWinnerCacheStateManager, TheValueStoredInMemoryAfterAdoptionIsTheOnDiskRecord)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("in_memory_after_adopt");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);

    {
        const auto writer = makeNamedStateManager("test:InMemoryAfterAdopt");
        writer->recordWinner(key, recordFor(0xA1, 1.0)); // disk's timeMs stays 1.0
    }

    // A second, independent manager: its in-memory cache does not already hold this
    // key, so the only way `timeMs` reaches it is through writeBackToShard()'s return.
    // Same ranked ids as the disk line, different timeMs -- rankedIdsEqual() calls this
    // unchanged, so the disk's original entry (timeMs 1.0) must be what is adopted.
    const auto second = makeNamedStateManager("test:InMemoryAfterAdopt");
    second->recordWinner(key, recordFor(0xA1, 99.0));

    const auto inMemory = second->winnerFor(key);
    ASSERT_TRUE(inMemory.has_value());
    ASSERT_EQ(inMemory->size(), 1U);
    EXPECT_EQ(inMemory->front().kernelId, testId(0xA1));
    EXPECT_DOUBLE_EQ(inMemory->front().timeMs, 1.0)
        << "_winnerCache must hold the ON-DISK record writeBackToShard() returned on "
           "adoption, not the caller's own argument re-stored under its own timeMs";
}

/// B1.3 -- a shard already holding TWO lines for one key: the write-back comparison
/// must be made against the LAST one. This is what makes last-match scanning
/// load-bearing; with a first-match scan the comparison would run against the FIRST
/// (stale) line and this test fails -- see the mutation note on
/// ASupersedingRecordWidensTheShardAndWins above.
TEST(TestIngestorWinnerCacheStateManager, TwoLinesOneKeyComparesAgainstTheLastLine)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("two_lines_last_wins");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const auto key = keyFor(graph, properties);

    const WinnerRecord firstRecord{entryFor(definitionFor(0xB1), 1.0)};
    const WinnerRecord secondRecord{entryFor(definitionFor(0xB2), 1.0)};

    {
        const auto writer = makeNamedStateManager("test:TwoLinesLastWins");
        writer->recordWinner(key, firstRecord); // line 1: 0xB1
        writer->recordWinner(key, secondRecord); // line 2: 0xB2 (genuinely different)
    }

    // A third write repeating the LAST line's ranking. A last-match scan adopts (no
    // append); a first-match scan would compare against 0xB1, see a difference, and
    // append a third line.
    {
        const auto writer = makeNamedStateManager("test:TwoLinesLastWins");
        writer->recordWinner(key, WinnerRecord{entryFor(definitionFor(0xB2), 9.0)});
    }

    const auto path = winnerCacheShardPath("test:TwoLinesLastWins", properties.gcnArchName);
    ASSERT_TRUE(std::filesystem::exists(path));
    std::ifstream shardFile(path);
    ASSERT_TRUE(shardFile.is_open());
    std::vector<std::string> rawLines;
    for(std::string line; std::getline(shardFile, line);)
    {
        rawLines.push_back(line);
    }
    ASSERT_EQ(rawLines.size(), 3U) << "version line plus exactly two record lines; a "
                                      "first-match comparison would wrongly append a third";
}

/// The pair below is driven by ctest, not gtest: the writer runs in one process, exits,
/// and the reader runs in a second process against the same shard. Both are DISABLED_ by
/// default so a plain developer run skips them; ctest passes
/// --gtest_also_run_disabled_tests.
///
/// This test pins that `mightHaveWinnerFor()` (not a bare in-memory-empty check) is what
/// lets a cold manager reach the shard: reverting it to a `winnerCacheSize() == 0` guard
/// makes only this test fail.
TEST(TestIngestorWinnerCacheStateManager, AColdManagerOrdersACatalogFromTheShardAlone)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedCacheDir cacheDir("cold_order");
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto properties = suffixedDeviceProperties();
    const MatchContext context{graph, 0, properties};

    // Record the heuristic's order reversed, covering the whole catalog, so a served
    // record is distinguishable from a heuristic ranking.
    std::vector<DescriptorId> reversedIds;
    {
        const auto writer = makeNamedStateManager("test:ColdOrder");
        const auto catalog = writer->sortedDefinitions(context);
        ASSERT_GE(catalog.size(), 2U) << "need at least two kernels for order to be observable";

        WinnerRecord record;
        double timeMs = 1.0;
        for(auto kernel = catalog.rbegin(); kernel != catalog.rend(); ++kernel)
        {
            record.push_back(entryFor(*kernel, timeMs));
            reversedIds.push_back(kernel->kernelId);
            timeMs += 1.0;
        }
        writer->recordWinner(WinnerKey{GraphContentKey{graph}, DeviceKey{properties}}, record);
    }

    // A manager that has recorded nothing: its in-memory cache is empty, so the only
    // way to the measured order is through the shard.
    const auto reader = makeNamedStateManager("test:ColdOrder");
    ASSERT_EQ(reader->winnerCacheSize(), 0U) << "the reader must start cold";

    const auto ordered = reader->sortedDefinitions(context);

    ASSERT_EQ(ordered.size(), reversedIds.size());
    for(size_t i = 0; i < reversedIds.size(); ++i)
    {
        EXPECT_EQ(ordered[i].kernelId, reversedIds[i])
            << "entry " << i << " is not in the measured order the shard holds";
    }
}

constexpr const char* CROSS_PROCESS_ENGINE = "test:CrossProcess";

/// The fixed key both halves derive, so the reader looks for exactly what the writer
/// laid down.
WinnerKey crossProcessKey()
{
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    return WinnerKey{GraphContentKey{graph}, DeviceKey{suffixedDeviceProperties()}};
}

TEST(TestIngestorWinnerCacheCrossProcess, DISABLED_WriterLeavesARecordOnDisk)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_CACHE_DIR").empty())
        << "the ctest registration must supply a shared cache directory";

    const auto manager = makeNamedStateManager(CROSS_PROCESS_ENGINE);
    manager->recordWinner(crossProcessKey(), recordFor(0x51, 4.25));

    const auto path
        = winnerCacheShardPath(CROSS_PROCESS_ENGINE, suffixedDeviceProperties().gcnArchName);
    ASSERT_FALSE(path.empty());
    EXPECT_TRUE(std::filesystem::exists(path))
        << "the writer process left no shard for the reader process to find";
}

TEST(TestIngestorWinnerCacheCrossProcess, DISABLED_ReaderServesTheRecordTheWriterLeft)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_CACHE_DIR").empty())
        << "the ctest registration must supply a shared cache directory";

    // Nothing recorded in this process, so a hit can only come from the writer's file.
    const auto manager = makeNamedStateManager(CROSS_PROCESS_ENGINE);
    const auto served = manager->winnerFor(crossProcessKey());

    ASSERT_TRUE(served.has_value())
        << "a record written by a separate process was not served in this one";
    ASSERT_EQ(served->size(), 1U);
    EXPECT_EQ(served->front().kernelId, testId(0x51));
    EXPECT_DOUBLE_EQ(served->front().timeMs, 4.25);
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
