// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/StagedApplicability.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "KernelIngestorTestFixtures.hpp"
#include "flatbuffer_utilities/ContentCarryingTestGraph.hpp"

/**
 * @file TestSlottedStagedApplicability.cpp
 * @brief The staged-applicability contract of TestStagedApplicability.cpp, held by a two-slot
 *        engine: begin plus finish decides what isApplicable() decides, with the same matcher
 *        work and logs and the same cache left behind (verdict and per-slot catalogs), a
 *        prediction summed over slots, and finishes that are safe on worker threads.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::ContentCarryingTestGraph;
using hipdnn_test_sdk::utilities::SharedLogRecorder;

using Manager = KernelIngestorStateManager<StubHandle>;
using SlottedStagedEngine = GenericEngine<StubHandle, StubSettings, StubContext>;

constexpr const char* GRAPH_SYMBOL = "test.slotted_staged.graph";
constexpr const char* KERNEL_SYMBOL = "test.slotted_staged.kernel";
constexpr const char* REJECT_KERNEL_SYMBOL = "test.slotted_staged.reject_kernel";
constexpr const char* WS_BYTES = "ws.bytes";
/// testDeviceProperties() reports gfx000, so a pack or kernel narrowed to this never runs.
constexpr const char* UNAVAILABLE_ARCH = "gfx942";
constexpr int ENGINE_COUNT = 8;

const hipdnn_data_sdk::utilities::Version NEWER_THAN_BASELINE{
    hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION};

BoundTokens workspaceTokens(int64_t scratchOffset)
{
    return {{WS_BYTES, int64_t{1000}}, {"ws.scratch_offset", scratchOffset}};
}

std::optional<BoundTokens> acceptSlottedGraph(const MatchContext& /*context*/)
{
    ++counters().graphMatchCalls;
    return workspaceTokens(1024);
}

/// Publishes a scratch offset that is not the shared size rounded up to its alignment.
std::optional<BoundTokens> badTokenGraph(const MatchContext& /*context*/)
{
    ++counters().graphMatchCalls;
    return workspaceTokens(1000);
}

std::optional<BoundTokens> throwingGraph(const MatchContext& /*context*/)
{
    ++counters().graphMatchCalls;
    throw std::runtime_error("graph_match failed while deciding applicability");
}

bool rejectEveryKernel(const MatchContext& /*context*/,
                       const BoundTokens& /*bound*/,
                       const KernelDefinition& /*kernel*/)
{
    ++counters().kernelCalls;
    return false;
}

bool throwingKernel(const MatchContext& /*context*/,
                    const BoundTokens& /*bound*/,
                    const KernelDefinition& /*kernel*/)
{
    ++counters().kernelCalls;
    throw std::runtime_error("a kernel matcher failed while deciding applicability");
}

std::atomic<int>& graphMatchesAcrossThreads()
{
    static std::atomic<int> s_calls{0};
    return s_calls;
}

std::optional<BoundTokens> atomicCountingGraph(const MatchContext& /*context*/)
{
    graphMatchesAcrossThreads().fetch_add(1, std::memory_order_relaxed);
    return workspaceTokens(1024);
}

bool viewReadingKernels(const MatchContext& context,
                        const BoundTokens& bound,
                        const KernelDefinition& kernel)
{
    const bool intact
        = context.graph.getTensorMap().size() == 2U && context.graph.nodeWrappers().size() == 1U;
    return intact && acceptFloatKernels(context, bound, kernel);
}

/// One slot's pack: the three kernels of makePack(), with ids of their own so no id repeats
/// across slots. @p arch narrows the pack; @p excludeSecondKernel narrows one kernel.
KernelDescriptorPack slotPack(uint8_t seed,
                              const std::string& slot,
                              const std::vector<std::string>& arch = {},
                              bool excludeSecondKernel = false)
{
    auto pack = makePack({KERNEL_MATCHER_ID}, arch);
    pack.id = testId(seed);
    pack.name = slot + " pack";
    pack.slot = slot;
    pack.kernels
        = {makeKernel(testId(static_cast<uint8_t>(seed + 1)), slot + "_64_float", 64, "FLOAT"),
           makeKernel(testId(static_cast<uint8_t>(seed + 2)), slot + "_256_float", 256, "FLOAT"),
           makeKernel(testId(static_cast<uint8_t>(seed + 3)), slot + "_64_half", 64, "HALF")};
    if(excludeSecondKernel)
    {
        pack.kernels[1].arch = {UNAVAILABLE_ARCH};
    }
    return pack;
}

struct Packs
{
    std::vector<KernelDescriptorPack> first;
    std::vector<KernelDescriptorPack> second;
};

Packs defaultPacks()
{
    return {{slotPack(0x50, "first")}, {slotPack(0x60, "second")}};
}

/// A two-slot engine over @p packs, matching kernels through @p kernelSymbol. Twins built by two
/// calls with the same arguments decide every graph identically. @p manager, when given,
/// receives the engine's state manager.
std::unique_ptr<SlottedStagedEngine> makeSlottedEngine(const IDeviceResolver<StubHandle>& resolver,
                                                       Packs packs,
                                                       const std::string& kernelSymbol
                                                       = KERNEL_SYMBOL,
                                                       const Manager** manager = nullptr)
{
    std::vector<MatchDescriptor> matchers{
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, kernelSymbol}};
    std::vector<KernelSlotInput> slots;
    slots.push_back({"first",
                     makeSchema(),
                     std::move(packs.first),
                     std::make_shared<UnrankedKernelHeuristic>()});
    slots.push_back({"second",
                     makeSchema(),
                     std::move(packs.second),
                     std::make_shared<UnrankedKernelHeuristic>()});

    auto engine = makeEngineWithKnobs({});
    engine.graphMatchNativeSymbol = GRAPH_SYMBOL;
    engine.sharedWorkspaceToken = WS_BYTES;
    for(const char* name : {"first", "second"})
    {
        SlotDescriptor described;
        described.name = name;
        described.metadataSchemaId = SCHEMA_ID;
        described.knobs = {BLOCK_SIZE};
        engine.slots.push_back(std::move(described));
    }

    auto owned = std::make_unique<Manager>(std::move(slots),
                                           std::move(matchers),
                                           makeStubDispatches(),
                                           GRAPH_SYMBOL,
                                           WS_BYTES,
                                           "engine 'slotted staged fixture'");
    if(manager != nullptr)
    {
        *manager = owned.get();
    }
    return std::make_unique<SlottedStagedEngine>(std::move(engine), std::move(owned), resolver);
}

std::unique_ptr<SlottedStagedEngine> makeSlottedEngine(const IDeviceResolver<StubHandle>& resolver)
{
    return makeSlottedEngine(resolver, defaultPacks());
}

struct StagedOutcome
{
    bool decidedInBegin = false;
    bool verdict = false;
};

/// begin, then finish if begin left work pending: what a dispatcher does per engine.
StagedOutcome stage(const IStagedApplicability<StubHandle>& engine,
                    StubHandle& handle,
                    const IGraph& graph,
                    ApplicabilityScope& scope)
{
    StagedApplicability staged = engine.beginApplicability(handle, graph, scope);
    if(staged.decided)
    {
        return {true, staged.verdict};
    }
    EXPECT_TRUE(staged.pending.valid()) << "an undecided begin must leave work to finish";
    return {false, engine.finishApplicability(std::move(staged.pending))};
}

struct CallRecord
{
    bool verdict = false;
    int graphMatchCalls = 0;
    int kernelCalls = 0;
    std::vector<std::string> logs;
};

template <typename Call>
CallRecord record(SharedLogRecorder& recorder, Call&& call)
{
    counters().reset();
    recorder.clearLogs();
    CallRecord result;
    result.verdict = call();
    result.graphMatchCalls = counters().graphMatchCalls;
    result.kernelCalls = counters().kernelCalls;
    for(const auto& log : recorder.getRecordedLogs())
    {
        result.logs.push_back(std::to_string(static_cast<int>(log.severity)) + " " + log.message);
    }
    return result;
}

enum class GraphKind
{
    KEYED,
    UNKEYABLE,
    NEWER_SCHEMA,
};

std::unique_ptr<IGraph> makeGraph(GraphKind kind)
{
    switch(kind)
    {
    case GraphKind::UNKEYABLE:
        return std::make_unique<BytelessTestGraph>(makeGraphId(0xB1));
    case GraphKind::NEWER_SCHEMA:
        return std::make_unique<TestGraph>(makeGraphId(0xB2), NEWER_THAN_BASELINE);
    case GraphKind::KEYED:
    default:
        break;
    }
    return std::make_unique<TestGraph>(makeGraphId(0xB0));
}

struct EquivalenceCase
{
    std::string name;
    GraphMatchFn graphMatch;
    KernelMatcherFn kernelMatch;
    /// Arch of the first and second slot's pack.
    std::vector<std::string> firstArch;
    std::vector<std::string> secondArch;
    GraphKind graph;
    /// Answer once through isApplicable() first, so the staged call is a cache hit.
    bool warm;
    bool expectedVerdict;
    bool decidedInBegin;
    /// graph_match calls a second isApplicable() makes: 0 when the first call left the
    /// verdict cached (or never reaches graph_match), 1 when it must rematch.
    int secondCallGraphMatches;
};

class TestSlottedStagedApplicabilityEquivalence : public ::testing::TestWithParam<EquivalenceCase>
{
};

/// The contract every dispatcher relies on, for an engine with slots: splitting the decision
/// changes nothing observable. Same verdict, same matcher work, same logs, and the same cache
/// left behind, read back through a second plain call on each engine.
TEST_P(TestSlottedStagedApplicabilityEquivalence, MatchesIsApplicable)
{
    const auto& param = GetParam();
    const ScopedSymbols symbols(GRAPH_SYMBOL, param.graphMatch, KERNEL_SYMBOL, param.kernelMatch);
    const StubDeviceResolver resolver;
    const auto packsOf = [&param] {
        return Packs{{slotPack(0x50, "first", param.firstArch)},
                     {slotPack(0x60, "second", param.secondArch)}};
    };
    const auto staged = makeSlottedEngine(resolver, packsOf());
    const auto twin = makeSlottedEngine(resolver, packsOf());
    const auto graph = makeGraph(param.graph);
    StubHandle handle;

    if(param.warm)
    {
        ASSERT_EQ(staged->isApplicable(handle, *graph), param.expectedVerdict);
        ASSERT_EQ(twin->isApplicable(handle, *graph), param.expectedVerdict);
    }

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    const auto expected = record(recorder, [&] { return twin->isApplicable(handle, *graph); });
    ApplicabilityScope scope;
    StagedOutcome outcome;
    const auto actual = record(recorder, [&] {
        outcome = stage(*staged, handle, *graph, scope);
        return outcome.verdict;
    });

    EXPECT_EQ(expected.verdict, param.expectedVerdict);
    EXPECT_EQ(actual.verdict, expected.verdict);
    EXPECT_EQ(outcome.decidedInBegin, param.decidedInBegin);
    EXPECT_EQ(actual.graphMatchCalls, expected.graphMatchCalls);
    EXPECT_EQ(actual.kernelCalls, expected.kernelCalls);
    EXPECT_EQ(actual.logs, expected.logs);

    const auto twinAgain = record(recorder, [&] { return twin->isApplicable(handle, *graph); });
    const auto stagedAgain = record(recorder, [&] { return staged->isApplicable(handle, *graph); });

    EXPECT_EQ(twinAgain.graphMatchCalls, param.secondCallGraphMatches);
    EXPECT_EQ(stagedAgain.verdict, twinAgain.verdict);
    EXPECT_EQ(stagedAgain.graphMatchCalls, twinAgain.graphMatchCalls)
        << "the staged call left a different cache entry than isApplicable() does";
    EXPECT_EQ(stagedAgain.kernelCalls, twinAgain.kernelCalls);
    EXPECT_EQ(stagedAgain.logs, twinAgain.logs);
}

INSTANTIATE_TEST_SUITE_P(
    Cases,
    TestSlottedStagedApplicabilityEquivalence,
    ::testing::Values(
        EquivalenceCase{"CacheHit",
                        acceptSlottedGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::KEYED,
                        /*warm=*/true,
                        /*expectedVerdict=*/true,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/0},
        EquivalenceCase{"MissThatApplies",
                        acceptSlottedGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::KEYED,
                        false,
                        true,
                        false,
                        0},
        // Passes graph_match, so it pays the full variant loop to decline; the decline must
        // still be cached.
        EquivalenceCase{"MissWhereNoKernelMatches",
                        acceptSlottedGraph,
                        rejectEveryKernel,
                        {},
                        {},
                        GraphKind::KEYED,
                        false,
                        false,
                        false,
                        0},
        EquivalenceCase{"GraphMatchDecline",
                        rejectGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::KEYED,
                        false,
                        false,
                        true,
                        0},
        // An invalid `shared_workspace` token is settled by the graph-level stage alone.
        EquivalenceCase{"BadSharedWorkspaceToken",
                        badTokenGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::KEYED,
                        false,
                        false,
                        true,
                        0},
        EquivalenceCase{"SchemaFloorDecline",
                        acceptSlottedGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::NEWER_SCHEMA,
                        false,
                        false,
                        true,
                        0},
        // No content key: matched in full on the caller and never memoized.
        EquivalenceCase{"UnkeyableGraph",
                        acceptSlottedGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::UNKEYABLE,
                        false,
                        true,
                        true,
                        1},
        // One slot has no pack for the device: graph_match still runs and the other slot's
        // catalog is built, so the decline is settled only by the finish.
        EquivalenceCase{"ASlotWithNoPackForTheDevice",
                        acceptSlottedGraph,
                        countingFloatKernels,
                        {},
                        {UNAVAILABLE_ARCH},
                        GraphKind::KEYED,
                        false,
                        false,
                        false,
                        0},
        EquivalenceCase{"EverySlotWithoutAPackForTheDevice",
                        acceptSlottedGraph,
                        countingFloatKernels,
                        {UNAVAILABLE_ARCH},
                        {UNAVAILABLE_ARCH},
                        GraphKind::KEYED,
                        false,
                        false,
                        true,
                        0},
        // An error caches nothing, so the next call rematches.
        EquivalenceCase{"GraphMatchThrows",
                        throwingGraph,
                        countingFloatKernels,
                        {},
                        {},
                        GraphKind::KEYED,
                        false,
                        false,
                        true,
                        1},
        // Thrown in finish: declined and logged there, never propagated.
        EquivalenceCase{"KernelMatcherThrows",
                        acceptSlottedGraph,
                        throwingKernel,
                        {},
                        {},
                        GraphKind::KEYED,
                        false,
                        false,
                        false,
                        1}),
    [](const ::testing::TestParamInfo<EquivalenceCase>& info) { return info.param.name; });

struct PredictionCase
{
    std::string name;
    /// A factory, not a value: descriptor ids are dynamically initialised inline variables, so
    /// packs are built when the case runs, not at static init.
    Packs (*makePacks)();
    size_t expectedVariants;
    /// A slot without a pack for the device leaves the engine declining, though the other slot's
    /// variants are still examined.
    bool expectedVerdict = true;
};

class TestSlottedStagedApplicabilityPrediction : public ::testing::TestWithParam<PredictionCase>
{
};

/// Counts arch-eligible variants of every slot past graph_match, before any kernel matcher runs:
/// makePack()'s HALF kernel, which no matcher here admits, still counts.
TEST_P(TestSlottedStagedApplicabilityPrediction, SumsArchEligibleVariantsOverEverySlot)
{
    const auto& param = GetParam();
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, acceptSlottedGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto engine = makeSlottedEngine(resolver, param.makePacks());
    const TestGraph graph(makeGraphId(0xB3));
    StubHandle handle;
    ApplicabilityScope scope;

    auto staged = engine->beginApplicability(handle, graph, scope);

    ASSERT_FALSE(staged.decided);
    ASSERT_TRUE(staged.pending.valid());
    EXPECT_EQ(staged.pending.predictedVariants(), param.expectedVariants);
    EXPECT_EQ(engine->finishApplicability(std::move(staged.pending)), param.expectedVerdict);
}

INSTANTIATE_TEST_SUITE_P(
    Cases,
    TestSlottedStagedApplicabilityPrediction,
    ::testing::Values(
        PredictionCase{"EveryKernelOfEverySlotEligible", [] { return defaultPacks(); }, 6},
        PredictionCase{"OneKernelArchExcludedInOneSlot",
                       [] {
                           return Packs{
                               {slotPack(0x50, "first")},
                               {slotPack(0x60, "second", {}, /*excludeSecondKernel=*/true)}};
                       },
                       5},
        PredictionCase{"ArchExcludedPackOfOneSlotAddsNothing",
                       [] {
                           return Packs{{slotPack(0x50, "first", {UNAVAILABLE_ARCH}),
                                         slotPack(0x70, "first")},
                                        {slotPack(0x60, "second")}};
                       },
                       6},
        PredictionCase{"ASlotWithNoPackForTheDeviceAddsNothing",
                       [] {
                           return Packs{{slotPack(0x50, "first")},
                                        {slotPack(0x60, "second", {UNAVAILABLE_ARCH})}};
                       },
                       3,
                       /*expectedVerdict=*/false}),
    [](const ::testing::TestParamInfo<PredictionCase>& info) { return info.param.name; });

/// A program record is valid only if the manager has seen every slot's catalog for the graph.
/// A staged-only warm-up, with no plain isApplicable or catalog access, must leave them.
TEST(TestSlottedStagedApplicability, AStagedOnlyWarmUpLeavesEverySlotCatalogForProgramRecords)
{
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, acceptSlottedGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const Manager* manager = nullptr;
    const auto engine = makeSlottedEngine(resolver, defaultPacks(), KERNEL_SYMBOL, &manager);
    const ContentCarryingTestGraph graph;
    const auto properties = testDeviceProperties();
    StubHandle handle;

    const WinnerKey base{GraphContentKey{graph}, DeviceKey{properties}};
    const auto ranking = [](uint8_t pack, uint8_t first, uint8_t second) {
        return WinnerRecord{
            RankedEntry{testId(first), testId(pack), DISPATCH_ID, 1.0},
            RankedEntry{testId(second), testId(pack), DISPATCH_ID, 2.0},
        };
    };
    // FLOAT kernels only: the HALF one is rejected by the matcher.
    const std::vector<std::pair<std::string, WinnerRecord>> slotRecords{
        {"first", ranking(0x50, 0x51, 0x52)}, {"second", ranking(0x60, 0x61, 0x62)}};
    const ProgramRecord program{{{"first", testId(0x51)}, {"second", testId(0x61)}}, 3.0};
    manager->recordSlotted(
        base, slotRecords, program, WinnerWriteCause::FRESH_MISS, WinnerWriteCause::FRESH_MISS);
    EXPECT_FALSE(manager->programFor(base).has_value()) << "no catalog has been seen yet";

    ApplicabilityScope scope;
    const auto outcome = stage(*engine, handle, graph, scope);
    ASSERT_TRUE(outcome.verdict);

    EXPECT_TRUE(manager->programFor(base).has_value())
        << "a staged finish must remember each slot's catalog";
}

/// Every slot's catalog is cached by a staged finish, so a plain catalog read afterwards
/// matches nothing again.
TEST(TestSlottedStagedApplicability, AStagedFinishCachesEverySlotCatalog)
{
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, acceptSlottedGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const Manager* manager = nullptr;
    const auto engine = makeSlottedEngine(resolver, defaultPacks(), KERNEL_SYMBOL, &manager);
    const TestGraph graph(makeGraphId(0xB8));
    const auto properties = testDeviceProperties();
    StubHandle handle;
    ApplicabilityScope scope;
    ASSERT_TRUE(stage(*engine, handle, graph, scope).verdict);

    counters().reset();
    const MatchContext context{graph, 0, properties};
    for(size_t slot = 0; slot < manager->slotCount(); ++slot)
    {
        EXPECT_EQ(manager->unsortedCatalog(slot, context).entries.size(), 2U) << "slot " << slot;
    }
    EXPECT_EQ(counters().graphMatchCalls, 0);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestSlottedStagedApplicability, FinishDeclinesAPendingBegunByAnotherEngine)
{
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, acceptSlottedGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto owner = makeSlottedEngine(resolver);
    const auto stranger = makeSlottedEngine(resolver);
    const TestGraph graph(makeGraphId(0xB4));
    StubHandle handle;
    ApplicabilityScope scope;

    auto staged = owner->beginApplicability(handle, graph, scope);
    ASSERT_FALSE(staged.decided);

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    EXPECT_FALSE(stranger->finishApplicability(std::move(staged.pending)));
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U);
}

/// The dispatcher's shape: every begin on the caller, prewarm, then each engine's finish on its
/// own thread, all released together. Even engines admit their FLOAT kernels and odd ones admit
/// none, so a verdict landing on the wrong engine shows.
TEST(TestSlottedStagedApplicability, ThreadedFinishesMatchSequentialVerdicts)
{
    graphMatchesAcrossThreads().store(0);
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, atomicCountingGraph, KERNEL_SYMBOL, viewReadingKernels);
    KernelMatcherRegistry::registerSymbol(REJECT_KERNEL_SYMBOL, &rejectEveryKernel);
    const StubDeviceResolver resolver;

    std::vector<std::unique_ptr<SlottedStagedEngine>> engines;
    std::vector<std::unique_ptr<SlottedStagedEngine>> twins;
    for(int i = 0; i < ENGINE_COUNT; ++i)
    {
        const std::string symbol = i % 2 == 0 ? KERNEL_SYMBOL : REJECT_KERNEL_SYMBOL;
        engines.push_back(makeSlottedEngine(resolver, defaultPacks(), symbol));
        twins.push_back(makeSlottedEngine(resolver, defaultPacks(), symbol));
    }

    // Two wrappers over one buffer: the sequential reference must not warm the views the
    // threaded run reads, or the race this case exists to expose disappears.
    const ContentCarryingTestGraph source;
    const auto blob = source.bytes();
    const GraphWrapper referenceGraph(blob.data, blob.size);
    const GraphWrapper sharedGraph(blob.data, blob.size);
    StubHandle handle;

    std::array<bool, ENGINE_COUNT> expected{};
    for(size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = twins[i]->isApplicable(handle, referenceGraph);
        ASSERT_EQ(expected[i], i % 2 == 0) << "engine " << i;
    }

    ApplicabilityScope scope;
    std::vector<PendingApplicability> pending;
    pending.reserve(engines.size());
    for(const auto& engine : engines)
    {
        auto staged = engine->beginApplicability(handle, sharedGraph, scope);
        ASSERT_FALSE(staged.decided);
        pending.push_back(std::move(staged.pending));
    }

    prewarmGraph(sharedGraph);

    std::array<bool, ENGINE_COUNT> verdicts{};
    runConcurrently(ENGINE_COUNT, [&](int i) {
        const auto index = static_cast<size_t>(i);
        verdicts[index] = engines[index]->finishApplicability(std::move(pending[index]));
    });
    KernelMatcherRegistry::unregisterSymbol(REJECT_KERNEL_SYMBOL);

    for(size_t i = 0; i < verdicts.size(); ++i)
    {
        EXPECT_EQ(verdicts[i], expected[i]) << "engine " << i;
    }

    const int matchesBefore = graphMatchesAcrossThreads().load();
    for(size_t i = 0; i < engines.size(); ++i)
    {
        EXPECT_EQ(engines[i]->isApplicable(handle, sharedGraph), expected[i]) << "engine " << i;
    }
    EXPECT_EQ(graphMatchesAcrossThreads().load(), matchesBefore)
        << "every threaded finish must leave its engine's verdict cached";
}

/// Concurrent calls on one engine: several callers' pendings for the same key finish at once
/// and race to store the same slot catalogs and verdict.
TEST(TestSlottedStagedApplicability, ConcurrentFinishesOnOneEngineAgree)
{
    graphMatchesAcrossThreads().store(0);
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, atomicCountingGraph, KERNEL_SYMBOL, viewReadingKernels);
    const StubDeviceResolver resolver;
    const auto engine = makeSlottedEngine(resolver);

    const ContentCarryingTestGraph source;
    const auto blob = source.bytes();
    const GraphWrapper graph(blob.data, blob.size);
    StubHandle handle;

    std::vector<std::unique_ptr<ApplicabilityScope>> scopes;
    std::vector<PendingApplicability> pending;
    pending.reserve(ENGINE_COUNT);
    for(int i = 0; i < ENGINE_COUNT; ++i)
    {
        scopes.push_back(std::make_unique<ApplicabilityScope>());
        auto staged = engine->beginApplicability(handle, graph, *scopes.back());
        ASSERT_FALSE(staged.decided) << "caller " << i;
        pending.push_back(std::move(staged.pending));
    }

    prewarmGraph(graph);

    std::array<bool, ENGINE_COUNT> verdicts{};
    runConcurrently(ENGINE_COUNT, [&](int i) {
        const auto index = static_cast<size_t>(i);
        verdicts[index] = engine->finishApplicability(std::move(pending[index]));
    });

    for(size_t i = 0; i < verdicts.size(); ++i)
    {
        EXPECT_TRUE(verdicts[i]) << "caller " << i;
    }

    const int matchesBefore = graphMatchesAcrossThreads().load();
    EXPECT_TRUE(engine->isApplicable(handle, graph));
    EXPECT_EQ(graphMatchesAcrossThreads().load(), matchesBefore);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
