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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/StagedApplicability.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "KernelIngestorTestFixtures.hpp"
#include "flatbuffer_utilities/ContentCarryingTestGraph.hpp"

/**
 * @file TestStagedApplicability.cpp
 * @brief Tests for StagedApplicability.hpp as GenericEngine implements it: begin plus
 *        finish decides exactly what isApplicable() decides, leaves the same cache and
 *        logs, shares one key build per scope, and finishes safely on other threads.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::ContentCarryingTestGraph;
using hipdnn_test_sdk::utilities::SharedLogRecorder;

using StagedEngine = GenericEngine<StubHandle, StubSettings, StubContext>;
using TensorMap
    = std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>;

static_assert(std::is_base_of_v<IStagedApplicability<StubHandle>, StagedEngine>);
static_assert(std::is_default_constructible_v<ApplicabilityScope>);
static_assert(!std::is_copy_constructible_v<ApplicabilityScope>);
static_assert(!std::is_copy_assignable_v<ApplicabilityScope>);
static_assert(!std::is_copy_constructible_v<PendingApplicability>);
static_assert(!std::is_copy_assignable_v<PendingApplicability>);
static_assert(std::is_nothrow_move_constructible_v<PendingApplicability>);

constexpr const char* GRAPH_SYMBOL = "test.graph";
constexpr const char* KERNEL_SYMBOL = "test.kernel";
/// testDeviceProperties() reports gfx000, so a pack or kernel narrowed to this never runs.
constexpr const char* UNAVAILABLE_ARCH = "gfx942";

const hipdnn_data_sdk::utilities::Version NEWER_THAN_BASELINE{
    hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION};

/// One engine over @p packs, matching kernels through @p kernelSymbol. Twins built by
/// two calls with the same arguments decide every graph identically.
std::unique_ptr<StagedEngine> makeEngine(const IDeviceResolver<StubHandle>& resolver,
                                         std::vector<KernelDescriptorPack> packs,
                                         const std::string& kernelSymbol = KERNEL_SYMBOL)
{
    std::vector<MatchDescriptor> matchers{
        {KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, kernelSymbol}};
    auto manager = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        makeSchema(),
        std::move(matchers),
        makeStubDispatches(),
        std::move(packs),
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        GRAPH_SYMBOL,
        "engine 'staged fixture'");
    return std::make_unique<StagedEngine>(
        makeEngineWithKnobs({BLOCK_SIZE}), std::move(manager), resolver);
}

std::unique_ptr<StagedEngine> makeEngine(const IDeviceResolver<StubHandle>& resolver)
{
    return makeEngine(resolver, {makePack({KERNEL_MATCHER_ID})});
}

/// A second pack with its own ids: two FLOAT kernels and one HALF, like makePack(), but
/// with block sizes makePack() does not use. The (block_size, dtype) tuple is the
/// catalog key and must be unique across every kernel that reaches a device.
KernelDescriptorPack makeSecondPack(const std::vector<std::string>& arch = {})
{
    auto pack = makePack({KERNEL_MATCHER_ID}, arch);
    pack.id = testId(0xE7);
    pack.name = "second test pack";
    pack.kernels = {makeKernel(testId(0x74), "second_128_float", 128, "FLOAT"),
                    makeKernel(testId(0x75), "second_512_float", 512, "FLOAT"),
                    makeKernel(testId(0x76), "second_128_half", 128, "HALF")};
    return pack;
}

struct StagedOutcome
{
    bool decidedInBegin = false;
    bool verdict = false;
};

/// begin, then finish if begin left work pending: what a dispatcher does per engine.
/// Goes through the interface, since that is all a dispatcher sees.
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

bool rejectEveryKernel(const MatchContext& /*context*/,
                       const BoundTokens& /*bound*/,
                       const KernelDefinition& /*kernel*/)
{
    ++counters().kernelCalls;
    return false;
}

std::optional<BoundTokens> throwingGraphMatch(const MatchContext& /*context*/)
{
    ++counters().graphMatchCalls;
    throw std::runtime_error("graph_match failed while deciding applicability");
}

bool throwingKernelMatch(const MatchContext& /*context*/,
                         const BoundTokens& /*bound*/,
                         const KernelDefinition& /*kernel*/)
{
    ++counters().kernelCalls;
    throw std::runtime_error("a kernel matcher failed while deciding applicability");
}

// ---------------------------------------------------------------------------
// Equivalence: begin (+ finish) against isApplicable() on an identical twin
// ---------------------------------------------------------------------------

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
    std::vector<std::string> packArch;
    GraphKind graph;
    /// Answer once through isApplicable() first, so the staged call is a cache hit.
    bool warm;
    bool expectedVerdict;
    bool decidedInBegin;
    /// graph_match calls a second isApplicable() makes: 0 when the first call left the
    /// verdict cached (or never reaches graph_match), 1 when it must rematch.
    int secondCallGraphMatches;
};

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

class TestIngestorStagedApplicabilityEquivalence : public ::testing::TestWithParam<EquivalenceCase>
{
};

/// The contract every dispatcher relies on: splitting the decision changes nothing
/// observable. Same verdict, same matcher work, same logs, and the same cache left
/// behind, read back through a second plain call on each engine.
TEST_P(TestIngestorStagedApplicabilityEquivalence, MatchesIsApplicable)
{
    const auto& param = GetParam();
    const ScopedSymbols symbols(GRAPH_SYMBOL, param.graphMatch, KERNEL_SYMBOL, param.kernelMatch);
    const StubDeviceResolver resolver;
    const auto staged = makeEngine(resolver, {makePack({KERNEL_MATCHER_ID}, param.packArch)});
    const auto twin = makeEngine(resolver, {makePack({KERNEL_MATCHER_ID}, param.packArch)});
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
    TestIngestorStagedApplicabilityEquivalence,
    ::testing::Values(
        EquivalenceCase{"CacheHit",
                        acceptGraph,
                        countingFloatKernels,
                        {},
                        GraphKind::KEYED,
                        /*warm=*/true,
                        /*expectedVerdict=*/true,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/0},
        EquivalenceCase{"MissThatApplies",
                        acceptGraph,
                        countingFloatKernels,
                        {},
                        GraphKind::KEYED,
                        /*warm=*/false,
                        /*expectedVerdict=*/true,
                        /*decidedInBegin=*/false,
                        /*secondCallGraphMatches=*/0},
        // Passes graph_match, so it pays the full variant loop to decline; the decline
        // must still be cached.
        EquivalenceCase{"MissWhereNoKernelMatches",
                        acceptGraph,
                        rejectEveryKernel,
                        {},
                        GraphKind::KEYED,
                        /*warm=*/false,
                        /*expectedVerdict=*/false,
                        /*decidedInBegin=*/false,
                        /*secondCallGraphMatches=*/0},
        EquivalenceCase{"GraphMatchDecline",
                        rejectGraph,
                        countingFloatKernels,
                        {},
                        GraphKind::KEYED,
                        /*warm=*/false,
                        /*expectedVerdict=*/false,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/0},
        EquivalenceCase{"SchemaFloorDecline",
                        acceptGraph,
                        countingFloatKernels,
                        {},
                        GraphKind::NEWER_SCHEMA,
                        /*warm=*/false,
                        /*expectedVerdict=*/false,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/0},
        // No content key: matched in full on the caller and never memoized.
        EquivalenceCase{"UnkeyableGraph",
                        acceptGraph,
                        countingFloatKernels,
                        {},
                        GraphKind::UNKEYABLE,
                        /*warm=*/false,
                        /*expectedVerdict=*/true,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/1},
        EquivalenceCase{"ArchDecline",
                        acceptGraph,
                        countingFloatKernels,
                        {UNAVAILABLE_ARCH},
                        GraphKind::KEYED,
                        /*warm=*/false,
                        /*expectedVerdict=*/false,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/0},
        // An error caches nothing, so the next call rematches.
        EquivalenceCase{"GraphMatchThrows",
                        throwingGraphMatch,
                        countingFloatKernels,
                        {},
                        GraphKind::KEYED,
                        /*warm=*/false,
                        /*expectedVerdict=*/false,
                        /*decidedInBegin=*/true,
                        /*secondCallGraphMatches=*/1},
        // Thrown in finish: declined and logged there, never propagated.
        EquivalenceCase{"KernelMatcherThrows",
                        acceptGraph,
                        throwingKernelMatch,
                        {},
                        GraphKind::KEYED,
                        /*warm=*/false,
                        /*expectedVerdict=*/false,
                        /*decidedInBegin=*/false,
                        /*secondCallGraphMatches=*/1}),
    [](const ::testing::TestParamInfo<EquivalenceCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// predictedVariants: the threading gate's cost estimate
// ---------------------------------------------------------------------------

struct PredictionCase
{
    std::string name;
    /// A factory, not a value: descriptor ids are dynamically initialised inline
    /// variables, so packs are built when the case runs, not at static init.
    std::vector<KernelDescriptorPack> (*makePacks)();
    size_t expectedVariants;
};

class TestIngestorStagedApplicabilityPrediction : public ::testing::TestWithParam<PredictionCase>
{
};

/// Counts arch-eligible variants of the packs past graph_match, before any kernel
/// matcher runs: makePack()'s HALF kernel, which no matcher here admits, still counts.
TEST_P(TestIngestorStagedApplicabilityPrediction, CountsArchEligibleVariants)
{
    const auto& param = GetParam();
    const ScopedSymbols symbols(GRAPH_SYMBOL, acceptGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto engine = makeEngine(resolver, param.makePacks());
    const TestGraph graph(makeGraphId(0xB3));
    StubHandle handle;
    ApplicabilityScope scope;

    auto staged = engine->beginApplicability(handle, graph, scope);

    ASSERT_FALSE(staged.decided);
    ASSERT_TRUE(staged.pending.valid());
    EXPECT_EQ(staged.pending.predictedVariants(), param.expectedVariants);
    EXPECT_TRUE(engine->finishApplicability(std::move(staged.pending)));
}

INSTANTIATE_TEST_SUITE_P(
    Cases,
    TestIngestorStagedApplicabilityPrediction,
    ::testing::Values(PredictionCase{"EveryKernelEligible",
                                     [] {
                                         return std::vector<KernelDescriptorPack>{
                                             makePack({KERNEL_MATCHER_ID})};
                                     },
                                     3},
                      PredictionCase{"OneKernelArchExcluded",
                                     [] {
                                         auto pack = makePack({KERNEL_MATCHER_ID});
                                         pack.kernels[1].arch = {UNAVAILABLE_ARCH};
                                         return std::vector<KernelDescriptorPack>{std::move(pack)};
                                     },
                                     2},
                      PredictionCase{"ArchExcludedPackAddsNothing",
                                     [] {
                                         return std::vector<KernelDescriptorPack>{
                                             makePack({KERNEL_MATCHER_ID}, {UNAVAILABLE_ARCH}),
                                             makeSecondPack()};
                                     },
                                     3},
                      PredictionCase{"EveryEligiblePackAdds",
                                     [] {
                                         return std::vector<KernelDescriptorPack>{
                                             makePack({KERNEL_MATCHER_ID}), makeSecondPack()};
                                     },
                                     6}),
    [](const ::testing::TestParamInfo<PredictionCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Finishing a pending this engine did not begin
// ---------------------------------------------------------------------------

TEST(TestIngestorStagedApplicability, FinishDeclinesAPendingBegunByAnotherEngine)
{
    const ScopedSymbols symbols(GRAPH_SYMBOL, acceptGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto owner = makeEngine(resolver);
    const auto stranger = makeEngine(resolver);
    const TestGraph graph(makeGraphId(0xB4));
    StubHandle handle;
    ApplicabilityScope scope;

    auto staged = owner->beginApplicability(handle, graph, scope);
    ASSERT_FALSE(staged.decided);

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    EXPECT_FALSE(stranger->finishApplicability(std::move(staged.pending)));
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U);

    // The stranger cached nothing from the foreign stage: its own call matches afresh.
    counters().reset();
    EXPECT_TRUE(stranger->isApplicable(handle, graph));
    EXPECT_EQ(counters().graphMatchCalls, 1);
}

TEST(TestIngestorStagedApplicability, FinishDeclinesAnEmptyPending)
{
    const ScopedSymbols symbols(GRAPH_SYMBOL, acceptGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto engine = makeEngine(resolver);
    PendingApplicability empty;

    EXPECT_FALSE(empty.valid());
    EXPECT_EQ(empty.predictedVariants(), 0U);

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    EXPECT_FALSE(engine->finishApplicability(std::move(empty)));
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U);
}

// ---------------------------------------------------------------------------
// One scope, one key build
// ---------------------------------------------------------------------------

/// Stages @p engines on one fresh scope the way a dispatcher does (every begin, then
/// every pending finish), expecting each to apply, and returns how often that read the
/// graph's bytes.
unsigned bytesReadStaging(const std::vector<const StagedEngine*>& engines,
                          StubHandle& handle,
                          const IGraph& graph,
                          std::atomic_uint& bytesCalls)
{
    ApplicabilityScope scope;
    bytesCalls.store(0);

    std::vector<StagedApplicability> staged;
    staged.reserve(engines.size());
    for(const auto* engine : engines)
    {
        staged.push_back(engine->beginApplicability(handle, graph, scope));
    }
    for(size_t i = 0; i < engines.size(); ++i)
    {
        const bool verdict = staged[i].decided
                                 ? staged[i].verdict
                                 : engines[i]->finishApplicability(std::move(staged[i].pending));
        EXPECT_TRUE(verdict) << "engine " << i;
    }
    return bytesCalls.load();
}

TEST(TestIngestorStagedApplicability, ColdEnginesSharingAScopeKeyTheGraphOnce)
{
    const ScopedSymbols symbols(GRAPH_SYMBOL, acceptGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto probe = makeEngine(resolver);
    const auto first = makeEngine(resolver);
    const auto second = makeEngine(resolver);
    const auto third = makeEngine(resolver);
    const auto bytesCalls = std::make_shared<std::atomic_uint>(0);
    const CountingBytesGraph graph(bytesCalls);
    StubHandle handle;

    const unsigned oneKeyBuild = bytesReadStaging({probe.get()}, handle, graph, *bytesCalls);
    ASSERT_GT(oneKeyBuild, 0U);

    EXPECT_EQ(
        bytesReadStaging({first.get(), second.get(), third.get()}, handle, graph, *bytesCalls),
        oneKeyBuild)
        << "three engines on one scope must key the graph once, not once each";
}

TEST(TestIngestorStagedApplicability, WarmEnginesSharingAScopeKeyTheGraphOnce)
{
    const ScopedSymbols symbols(GRAPH_SYMBOL, acceptGraph, KERNEL_SYMBOL, countingFloatKernels);
    const StubDeviceResolver resolver;
    const auto probe = makeEngine(resolver);
    const auto first = makeEngine(resolver);
    const auto second = makeEngine(resolver);
    const auto third = makeEngine(resolver);
    const auto bytesCalls = std::make_shared<std::atomic_uint>(0);
    const CountingBytesGraph graph(bytesCalls);
    StubHandle handle;

    for(const auto* engine : {probe.get(), first.get(), second.get(), third.get()})
    {
        ASSERT_TRUE(engine->isApplicable(handle, graph));
    }

    const unsigned oneKeyBuild = bytesReadStaging({probe.get()}, handle, graph, *bytesCalls);
    ASSERT_GT(oneKeyBuild, 0U);

    EXPECT_EQ(
        bytesReadStaging({first.get(), second.get(), third.get()}, handle, graph, *bytesCalls),
        oneKeyBuild)
        << "three cache hits on one scope must key the graph once, not once each";
}

// ---------------------------------------------------------------------------
// prewarmGraph
// ---------------------------------------------------------------------------

/// TestGraph's nodeWrappers() throws; this one also counts tensor-map reads.
class TensorMapCountingGraph : public TestGraph
{
public:
    using TestGraph::TestGraph;

    const TensorMap& getTensorMap() const override
    {
        ++_tensorMapReads;
        return TestGraph::getTensorMap();
    }

    int tensorMapReads() const
    {
        return _tensorMapReads;
    }

private:
    mutable int _tensorMapReads = 0;
};

/// A graph whose tensor map throws and whose node wrappers count reads.
class NodeWrappersCountingGraph : public TestGraph
{
public:
    using TestGraph::TestGraph;

    const TensorMap& getTensorMap() const override
    {
        throw std::logic_error("NodeWrappersCountingGraph has no tensor map");
    }

    const std::vector<std::unique_ptr<INodeWrapper>>& nodeWrappers() const override
    {
        ++_nodeWrapperReads;
        return _nodeWrappers;
    }

    int nodeWrapperReads() const
    {
        return _nodeWrapperReads;
    }

private:
    mutable int _nodeWrapperReads = 0;
    std::vector<std::unique_ptr<INodeWrapper>> _nodeWrappers;
};

/// One view failing must neither escape (prewarmGraph is noexcept: an escape would
/// terminate) nor stop the other view from being built before workers read it.
TEST(TestIngestorStagedApplicability, PrewarmBuildsTheTensorMapEvenWhenNodeWrappersThrow)
{
    const TensorMapCountingGraph graph(makeGraphId(0xB5));

    prewarmGraph(graph);

    EXPECT_EQ(graph.tensorMapReads(), 1);
}

TEST(TestIngestorStagedApplicability, PrewarmBuildsNodeWrappersEvenWhenTheTensorMapThrows)
{
    const NodeWrappersCountingGraph graph(makeGraphId(0xB6));

    prewarmGraph(graph);

    EXPECT_EQ(graph.nodeWrapperReads(), 1);
}

// ---------------------------------------------------------------------------
// Concurrency: finishes on worker threads (TSAN target)
// ---------------------------------------------------------------------------

constexpr int ENGINE_COUNT = 8;
constexpr const char* REJECT_KERNEL_SYMBOL = "test.kernel.reject";
/// ContentCarryingTestGraph's default spec: two tensors, one node.
constexpr size_t GRAPH_TENSORS = 2;
constexpr size_t GRAPH_NODES = 1;

/// counters() is plain ints, so the threaded cases count graph_match atomically.
std::atomic<int>& graphMatchesAcrossThreads()
{
    static std::atomic<int> s_calls{0};
    return s_calls;
}

std::optional<BoundTokens> atomicCountingAcceptGraph(const MatchContext& /*context*/)
{
    graphMatchesAcrossThreads().fetch_add(1, std::memory_order_relaxed);
    return BoundTokens{};
}

/// Reads the shared graph's lazily built views from whichever thread runs the finish.
/// GraphWrapper builds them on first access without a lock, so unwarmed views read from
/// several threads at once are a data race, and a torn build shows as a wrong size.
bool graphViewsIntact(const MatchContext& context)
{
    return context.graph.getTensorMap().size() == GRAPH_TENSORS
           && context.graph.nodeWrappers().size() == GRAPH_NODES;
}

bool viewReadingFloatKernels(const MatchContext& context,
                             const BoundTokens& bound,
                             const KernelDefinition& kernel)
{
    return graphViewsIntact(context) && acceptFloatKernels(context, bound, kernel);
}

bool viewReadingRejectKernels(const MatchContext& context,
                              const BoundTokens& /*bound*/,
                              const KernelDefinition& /*kernel*/)
{
    static_cast<void>(graphViewsIntact(context));
    return false;
}

hipdnn_data_sdk::utilities::ScopedResource<const char*> scopedKernelMatcher(const char* symbol,
                                                                            KernelMatcherFn matcher)
{
    KernelMatcherRegistry::registerSymbol(symbol, matcher);
    return {symbol,
            [](const char* registered) { KernelMatcherRegistry::unregisterSymbol(registered); }};
}

/// The dispatcher's shape: every begin on the caller, prewarm, then each engine's finish
/// on its own thread, all released together. Even engines admit their FLOAT kernels and
/// odd ones admit none, so a verdict landing on the wrong engine shows.
TEST(TestIngestorStagedApplicabilityConcurrency, ThreadedFinishesMatchSequentialVerdicts)
{
    graphMatchesAcrossThreads().store(0);
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, atomicCountingAcceptGraph, KERNEL_SYMBOL, viewReadingFloatKernels);
    const auto rejectMatcher = scopedKernelMatcher(REJECT_KERNEL_SYMBOL, viewReadingRejectKernels);
    const StubDeviceResolver resolver;

    std::vector<std::unique_ptr<StagedEngine>> engines;
    std::vector<std::unique_ptr<StagedEngine>> twins;
    for(int i = 0; i < ENGINE_COUNT; ++i)
    {
        const std::string symbol = i % 2 == 0 ? KERNEL_SYMBOL : REJECT_KERNEL_SYMBOL;
        engines.push_back(makeEngine(resolver, {makePack({KERNEL_MATCHER_ID})}, symbol));
        twins.push_back(makeEngine(resolver, {makePack({KERNEL_MATCHER_ID})}, symbol));
    }

    // Two wrappers over one buffer: the sequential reference must not warm the views
    // the threaded run reads, or the race this case exists to expose disappears.
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

/// Concurrent calls on one engine: several callers' pendings for the same key finish at
/// once and race to store the same catalog.
TEST(TestIngestorStagedApplicabilityConcurrency, ConcurrentFinishesOnOneEngineAgree)
{
    graphMatchesAcrossThreads().store(0);
    const ScopedSymbols symbols(
        GRAPH_SYMBOL, atomicCountingAcceptGraph, KERNEL_SYMBOL, viewReadingFloatKernels);
    const StubDeviceResolver resolver;
    const auto engine = makeEngine(resolver);

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
