// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "KernelIngestorTestFixtures.hpp"
#include "flatbuffer_utilities/ContentCarryingTestGraph.hpp"

/**
 * @file TestSlottedStateManager.cpp
 * @brief KernelIngestorStateManager over a three-slot engine (prep / dkdv / dq, RFC 0021):
 *        per-slot catalogs, one shared graph match, per-slot uniqueness, applicability and the
 *        shared-workspace token checks, the `ws.` feature reserve, and engine identity.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::ContentCarryingTestGraph;

using Manager = KernelIngestorStateManager<StubHandle>;

// Catalogs hold pointers into each slot's definitions, which stay valid only while the manager
// stays where it was built.
static_assert(!std::is_copy_constructible_v<Manager> && !std::is_copy_assignable_v<Manager>
                  && !std::is_move_constructible_v<Manager> && !std::is_move_assignable_v<Manager>,
              "the state manager must be pinned: its catalogs point into its own storage");

constexpr const char* GRAPH_MATCH = "hipdnn.kernel_ingestor.test.slotted.graph_match";
constexpr const char* WS_BYTES = "ws.bytes";
constexpr const char* BLOCK_M = "block_m";
constexpr const char* BLOCK_N = "block_n";

/// What the fixture's native graph match does, and how often it ran.
struct GraphMatchState
{
    int calls = 0;
    bool declines = false;
    BoundTokens tokens;
};

GraphMatchState& graphMatchState()
{
    static GraphMatchState s_state;
    return s_state;
}

std::optional<BoundTokens> slottedGraphMatch(const MatchContext& /*context*/)
{
    auto& state = graphMatchState();
    ++state.calls;
    if(state.declines)
    {
        return std::nullopt;
    }
    return state.tokens;
}

/// A shared region of @p shared bytes published the way RFC 0021 §6 describes: its size, the
/// scratch offset (that size rounded up to 256 B) and an offset token per shared object.
BoundTokens validTokens(int64_t shared = 1000)
{
    return {{WS_BYTES, shared},
            {"ws.scratch_offset", int64_t{1024}},
            {"ws.delta.offset", int64_t{0}},
            {"ws.lse.offset", int64_t{512}}};
}

/// RAII: registers the fixture's graph match and resets what it publishes.
class ScopedSlottedMatch
{
public:
    ScopedSlottedMatch()
    {
        graphMatchState() = GraphMatchState{};
        graphMatchState().tokens = validTokens();
        GraphMatchRegistry::registerSymbol(GRAPH_MATCH, &slottedGraphMatch);
    }

    ~ScopedSlottedMatch()
    {
        GraphMatchRegistry::unregisterSymbol(GRAPH_MATCH);
    }

    ScopedSlottedMatch(const ScopedSlottedMatch&) = delete;
    ScopedSlottedMatch& operator=(const ScopedSlottedMatch&) = delete;
};

MetadataSchema slotSchema(uint8_t seed, const std::string& name)
{
    return {
        testId(seed),
        name,
        {{BLOCK_M, MetadataType::INT, std::nullopt}, {BLOCK_N, MetadataType::INT, std::nullopt}}};
}

struct KernelSpec
{
    uint8_t seed;
    const char* name;
    int64_t blockM;
    int64_t blockN;
};

KernelDescriptorPack slotPack(uint8_t packSeed,
                              const std::string& slot,
                              const std::vector<KernelSpec>& kernels,
                              const std::vector<std::string>& arch = {})
{
    std::vector<KernelDescriptor> descriptors;
    descriptors.reserve(kernels.size());
    for(const auto& spec : kernels)
    {
        descriptors.push_back(makeSlotKernel(
            spec.seed,
            spec.name,
            {{BLOCK_M, MetadataValue{spec.blockM}}, {BLOCK_N, MetadataValue{spec.blockN}}}));
    }
    return makeSlotPack(packSeed, slot, std::move(descriptors), arch);
}

std::vector<KernelDescriptorPack> prepPacks()
{
    return {slotPack(0x10, "prep", {{0x11, "prep_small", 64, 1}, {0x12, "prep_large", 128, 1}})};
}

/// (64, 64) here and in dqPacks: one tuple in two slots.
std::vector<KernelDescriptorPack> dkdvPacks()
{
    return {slotPack(
        0x20,
        "dkdv",
        {{0x21, "dkdv_a", 64, 64}, {0x22, "dkdv_b", 128, 64}, {0x23, "dkdv_c", 128, 128}})};
}

std::vector<KernelDescriptorPack> dqPacks()
{
    return {slotPack(0x30, "dq", {{0x31, "dq_a", 64, 64}, {0x32, "dq_b", 128, 64}})};
}

KernelSlotInput
    slotInput(const std::string& name, uint8_t schemaSeed, std::vector<KernelDescriptorPack> packs)
{
    return {name,
            slotSchema(schemaSeed, name + " kmd"),
            std::move(packs),
            std::make_shared<UnrankedKernelHeuristic>()};
}

std::vector<KernelSlotInput> defaultSlots()
{
    std::vector<KernelSlotInput> slots;
    slots.push_back(slotInput("prep", 0xA1, prepPacks()));
    slots.push_back(slotInput("dkdv", 0xA2, dkdvPacks()));
    slots.push_back(slotInput("dq", 0xA3, dqPacks()));
    return slots;
}

std::unique_ptr<Manager> makeManager(std::vector<KernelSlotInput> slots = defaultSlots(),
                                     const std::string& sharedWorkspaceToken = WS_BYTES,
                                     size_t cacheCapacity = Manager::DEFAULT_CATALOG_CACHE_CAPACITY)
{
    return std::make_unique<Manager>(std::move(slots),
                                     std::vector<MatchDescriptor>{},
                                     makeStubDispatches(),
                                     GRAPH_MATCH,
                                     sharedWorkspaceToken,
                                     "engine 'slotted fixture'",
                                     cacheCapacity);
}

std::vector<std::string> namesOf(const Catalog& catalog)
{
    std::vector<std::string> names;
    names.reserve(catalog.entries.size());
    for(const auto& entry : catalog.entries)
    {
        names.push_back(entry->name);
    }
    return names;
}

using Names = std::vector<std::string>;

TEST(TestSlottedStateManager, EachSlotHoldsItsOwnCatalogInKernelIdOrder)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    ASSERT_TRUE(manager->isSlotted());

    EXPECT_EQ(namesOf(manager->enumerableCatalog(0, context)), (Names{"prep_small", "prep_large"}));
    EXPECT_EQ(namesOf(manager->enumerableCatalog(1, context)),
              (Names{"dkdv_a", "dkdv_b", "dkdv_c"}));
    EXPECT_EQ(namesOf(manager->enumerableCatalog(2, context)), (Names{"dq_a", "dq_b"}));
    // Declared order, before any ranker: pack order, then kernel order within it.
    EXPECT_EQ(namesOf(manager->unsortedCatalog(1, context)), (Names{"dkdv_a", "dkdv_b", "dkdv_c"}));

    const auto verdict = manager->slotProgramVerdict(context);
    EXPECT_TRUE(verdict.applicable) << verdict.declineReason;
    EXPECT_TRUE(manager->isApplicable(context));
    EXPECT_EQ(verdict.sharedWorkspaceBytes, 1000U);

    EXPECT_THROW(manager->unsortedCatalog(3, context), std::out_of_range);
    EXPECT_THROW(manager->slotName(3), std::out_of_range);
}

TEST(TestSlottedStateManager, OneGraphMatchServesEveryAccessorOfACatalogBuild)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager(defaultSlots(), WS_BYTES, 4);
    const auto properties = testDeviceProperties();

    const TestGraph graph(makeGraphId(1));
    const MatchContext context{graph, 0, properties};
    ASSERT_TRUE(manager->slotProgramVerdict(context).applicable);
    for(size_t slot = 0; slot < manager->slotCount(); ++slot)
    {
        (void)manager->unsortedCatalog(slot, context);
        (void)manager->sortedCatalog(slot, context);
        (void)manager->enumerableCatalog(slot, context);
    }
    EXPECT_EQ(graphMatchState().calls, 1) << "every slot's catalog is built from one match";

    // Every slot's catalog carries the tokens that one match published.
    for(size_t slot = 0; slot < manager->slotCount(); ++slot)
    {
        EXPECT_EQ(manager->unsortedCatalog(slot, context).bound, validTokens()) << "slot " << slot;
    }

    // Another graph is another build: the catalog key follows the graph's content, not its id.
    ContentCarryingTestGraph::Spec otherSpec;
    otherSpec.tensors[0].dims = {16, 8};
    const ContentCarryingTestGraph other{otherSpec};
    const MatchContext otherContext{other, 0, properties};
    (void)manager->unsortedCatalog(2, otherContext);
    EXPECT_EQ(graphMatchState().calls, 2);
    (void)manager->unsortedCatalog(0, otherContext);
    (void)manager->unsortedCatalog(1, otherContext);
    EXPECT_EQ(graphMatchState().calls, 2);
    EXPECT_EQ(namesOf(manager->unsortedCatalog(1, context)), (Names{"dkdv_a", "dkdv_b", "dkdv_c"}))
        << "the rebuilt catalog is the slot's own";
    EXPECT_EQ(graphMatchState().calls, 3) << "the evicted entry rebuilds, once";
}

TEST(TestSlottedStateManager, AGraphWithNoContentKeyStillMatchesOncePerBuildNotOncePerSlot)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const auto properties = testDeviceProperties();
    const BytelessTestGraph graph;
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);

    EXPECT_EQ(graphMatchState().calls, 1);
}

TEST(TestSlottedStateManager, AnEqualTupleInTwoSlotsIsAcceptedInBoth)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    // dkdv_a and dq_a both carry (block_m, block_n) = (64, 64).
    const auto dkdv = manager->unsortedCatalog(1, context);
    const auto dq = manager->unsortedCatalog(2, context);

    ASSERT_FALSE(dkdv.entries.empty());
    ASSERT_FALSE(dq.entries.empty());
    EXPECT_EQ(dkdv.entries.front()->getIntMetadata(BLOCK_M), 64);
    EXPECT_EQ(dq.entries.front()->getIntMetadata(BLOCK_M), 64);
    EXPECT_EQ(dkdv.entries.front()->getIntMetadata(BLOCK_N), 64);
    EXPECT_EQ(dq.entries.front()->getIntMetadata(BLOCK_N), 64);
    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);
}

TEST(TestSlottedStateManager, ADuplicateTupleInOneSlotDropsOnlyThePackThatRepeatsIt)
{
    const ScopedSlottedMatch symbols;
    auto dkdv = dkdvPacks();
    // The first kernel is new, the second repeats dkdv_a's tuple: the whole pack goes.
    dkdv.push_back(slotPack(0x24, "dkdv", {{0x25, "y_new", 256, 256}, {0x26, "y_repeat", 64, 64}}));
    // The dropped pack's claim on (256, 256) is released, so a later pack may take it.
    dkdv.push_back(slotPack(0x27, "dkdv", {{0x28, "z_late", 256, 256}}));
    std::vector<KernelSlotInput> slots;
    slots.push_back(slotInput("prep", 0xA1, prepPacks()));
    slots.push_back(slotInput("dkdv", 0xA2, std::move(dkdv)));
    slots.push_back(slotInput("dq", 0xA3, dqPacks()));

    const auto manager = makeManager(std::move(slots));
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_EQ(namesOf(manager->enumerableCatalog(1, context)),
              (Names{"dkdv_a", "dkdv_b", "dkdv_c", "z_late"}));
    EXPECT_EQ(namesOf(manager->enumerableCatalog(0, context)), (Names{"prep_small", "prep_large"}));
    EXPECT_EQ(namesOf(manager->enumerableCatalog(2, context)), (Names{"dq_a", "dq_b"}));
    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);
}

TEST(TestSlottedStateManager, ADuplicateKernelIdInOneSlotDropsThatPack)
{
    const ScopedSlottedMatch symbols;
    auto dkdv = dkdvPacks();
    // Kernel id 0x21 is dkdv_a's, declared again on arches both reach.
    dkdv.push_back(slotPack(0x24, "dkdv", {{0x21, "dkdv_again", 512, 512}}));
    std::vector<KernelSlotInput> slots;
    slots.push_back(slotInput("prep", 0xA1, prepPacks()));
    slots.push_back(slotInput("dkdv", 0xA2, std::move(dkdv)));
    slots.push_back(slotInput("dq", 0xA3, dqPacks()));

    const auto manager = makeManager(std::move(slots));
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_EQ(namesOf(manager->enumerableCatalog(1, context)),
              (Names{"dkdv_a", "dkdv_b", "dkdv_c"}));
}

TEST(TestSlottedStateManager, AnUnslottedEngineIsOneImplicitSlot)
{
    const ScopedSlottedMatch symbols;
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const Manager manager(slotSchema(0xA2, "kmd"),
                          {},
                          makeStubDispatches(),
                          {slotPack(0x24, "", {{0x25, "one", 64, 64}, {0x26, "two", 128, 64}})},
                          std::make_shared<UnrankedKernelHeuristic>(),
                          GRAPH_MATCH);
    EXPECT_FALSE(manager.isSlotted());
    EXPECT_EQ(manager.slotCount(), 1U);
    EXPECT_EQ(namesOf(manager.unsortedCatalog(context)), (Names{"one", "two"}));
    EXPECT_EQ(namesOf(manager.unsortedCatalog(0, context)), (Names{"one", "two"}));
    EXPECT_EQ(graphMatchState().calls, 1) << "the implicit slot shares the unslotted cache";
}

/// A slot's KMD with a STRING field `variant` besides the int fields, over kernels carrying
/// @p variants in order.
KernelSlotInput variantSlot(const std::string& name,
                            uint8_t schemaSeed,
                            uint8_t packSeed,
                            const std::vector<std::string>& variants)
{
    auto schema = slotSchema(schemaSeed, name + " kmd");
    schema.fields.push_back({"variant", MetadataType::STRING, std::nullopt});
    std::vector<KernelSpec> specs;
    specs.reserve(variants.size());
    for(size_t i = 0; i < variants.size(); ++i)
    {
        specs.push_back({static_cast<uint8_t>(packSeed + 1 + i), "kernel", 64, int64_t(i)});
    }
    auto pack = slotPack(packSeed, name, specs);
    for(size_t i = 0; i < variants.size(); ++i)
    {
        pack.kernels[i].metadata["variant"] = MetadataValue{variants[i]};
    }
    return {
        name, std::move(schema), {std::move(pack)}, std::make_shared<UnrankedKernelHeuristic>()};
}

TEST(TestSlottedStateManager, EachSlotReadsItsOwnOrdinalDomainForANonIntField)
{
    const ScopedSlottedMatch symbols;
    std::vector<KernelSlotInput> slots;
    // The domain of a non-int field is its observed values in sorted order, per slot.
    slots.push_back(variantSlot("prep", 0xA1, 0x10, {"q", "p"}));
    slots.push_back(variantSlot("dkdv", 0xA2, 0x20, {"b", "a", "c"}));
    slots.push_back(slotInput("dq", 0xA3, dqPacks()));
    const auto manager = makeManager(std::move(slots));
    const MetadataValue a{std::string("a")};
    const MetadataValue b{std::string("b")};
    const MetadataValue p{std::string("p")};
    const MetadataValue q{std::string("q")};

    EXPECT_EQ(manager->knobOrdinal(0, "variant", p), std::optional<int64_t>(0));
    EXPECT_EQ(manager->knobOrdinal(0, "variant", q), std::optional<int64_t>(1));
    EXPECT_EQ(manager->knobOrdinal(1, "variant", a), std::optional<int64_t>(0));
    EXPECT_EQ(manager->knobOrdinal(1, "variant", b), std::optional<int64_t>(1));
    // A value of another slot's domain is not in this one.
    EXPECT_EQ(manager->knobOrdinal(0, "variant", b), std::nullopt);
    EXPECT_EQ(manager->knobOrdinal(1, "variant", p), std::nullopt);

    EXPECT_TRUE(manager->isOrdinalKnob(0, "variant"));
    EXPECT_TRUE(manager->isOrdinalKnob(1, "variant"));
    EXPECT_FALSE(manager->isOrdinalKnob(2, "variant")) << "dq declares no such field";
    // An int field is addressed by its own value.
    EXPECT_FALSE(manager->isOrdinalKnob(1, BLOCK_M));
    EXPECT_EQ(manager->knobOrdinal(1, BLOCK_M, MetadataValue{int64_t{128}}),
              std::optional<int64_t>(128));
}

TEST(TestSlottedStateManager, ASlotlessAccessorOfASlottedEngineIsALogicError)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_THROW(manager->unsortedCatalog(context), std::logic_error);
    EXPECT_THROW(manager->sortedCatalog(context), std::logic_error);
    EXPECT_THROW(manager->enumerableCatalog(context), std::logic_error);
    EXPECT_THROW(manager->unsortedDefinitions(context), std::logic_error);
    EXPECT_THROW(manager->metadataSchema(), std::logic_error);
    EXPECT_EQ(manager->calibratedModelId("time", "gfx942"), "");
}

TEST(TestSlottedStateManager, ASlotWithNoPackDeclinesTheEngine)
{
    const ScopedSlottedMatch symbols;
    std::vector<KernelSlotInput> slots;
    slots.push_back(slotInput("prep", 0xA1, prepPacks()));
    slots.push_back(slotInput("dkdv", 0xA2, dkdvPacks()));
    slots.push_back(slotInput("dq", 0xA3, {}));
    const auto manager = makeManager(std::move(slots));
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto verdict = manager->slotProgramVerdict(context);

    EXPECT_FALSE(verdict.applicable);
    EXPECT_FALSE(manager->isApplicable(context));
    EXPECT_EQ(verdict.declineReason, "slot 'dq' has no applicable kernel");
    EXPECT_TRUE(manager->unsortedCatalog(2, context).entries.empty());
    EXPECT_FALSE(manager->unsortedCatalog(1, context).entries.empty())
        << "the other slots still report what applies to them";
}

TEST(TestSlottedStateManager, ASlotWhoseOnlyPackServesAnotherArchDeclinesTheEngine)
{
    const ScopedSlottedMatch symbols;
    std::vector<KernelSlotInput> slots;
    slots.push_back(slotInput("prep", 0xA1, prepPacks()));
    slots.push_back(slotInput("dkdv", 0xA2, dkdvPacks()));
    slots.push_back(
        slotInput("dq", 0xA3, {slotPack(0x30, "dq", {{0x31, "dq_a", 64, 64}}, {"gfx900"})}));
    const auto manager = makeManager(std::move(slots));
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties(); // gfx000
    const MatchContext context{graph, 0, properties};

    const auto verdict = manager->slotProgramVerdict(context);

    EXPECT_FALSE(verdict.applicable);
    EXPECT_EQ(verdict.declineReason, "slot 'dq' has no applicable kernel");
}

TEST(TestSlottedStateManager, NoDeviceDeclinesWithoutMatching)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, NO_DEVICE, properties};

    EXPECT_FALSE(manager->slotProgramVerdict(context).applicable);
    EXPECT_TRUE(manager->unsortedCatalog(0, context).entries.empty());
    EXPECT_EQ(graphMatchState().calls, 0);
}

TEST(TestSlottedStateManager, AGraphMatchDeclineEmptiesEverySlotAfterOneCall)
{
    const ScopedSlottedMatch symbols;
    graphMatchState().declines = true;
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto verdict = manager->slotProgramVerdict(context);

    EXPECT_FALSE(verdict.applicable);
    EXPECT_EQ(verdict.declineReason, "graph_match declined the graph");
    for(size_t slot = 0; slot < manager->slotCount(); ++slot)
    {
        EXPECT_TRUE(manager->unsortedCatalog(slot, context).entries.empty());
    }
    EXPECT_EQ(graphMatchState().calls, 1);
}

struct SharedWorkspaceCase
{
    const char* name;
    BoundTokens tokens;
    const char* declineReason;
};

TEST(TestSlottedStateManager, ABadSharedWorkspaceDeclinesWithAMessageNamingTheToken)
{
    const auto without = [](const std::string& token) {
        auto tokens = validTokens();
        tokens.erase(token);
        return tokens;
    };
    const auto with = [](const std::string& token, MetadataValue value) {
        auto tokens = validTokens();
        tokens.insert_or_assign(token, std::move(value));
        return tokens;
    };
    const std::vector<SharedWorkspaceCase> cases = {
        {"MissingToken",
         without(WS_BYTES),
         "shared_workspace token 'ws.bytes' was not published by the graph match"},
        {"StringToken",
         with(WS_BYTES, MetadataValue{std::string{"1000"}}),
         "shared_workspace token 'ws.bytes' is not an integer"},
        {"DoubleToken",
         with(WS_BYTES, MetadataValue{1000.0}),
         "shared_workspace token 'ws.bytes' is not an integer"},
        {"NegativeToken",
         with(WS_BYTES, MetadataValue{int64_t{-1}}),
         "shared_workspace token 'ws.bytes' is negative (-1)"},
        {"ScratchOffsetNotPublished",
         without("ws.scratch_offset"),
         "ws.scratch_offset was not published by the graph match"},
        {"ScratchOffsetNotTheAlignedSize",
         with("ws.scratch_offset", MetadataValue{int64_t{1000}}),
         "ws.scratch_offset (1000) does not equal shared_workspace token 'ws.bytes' value 1000 "
         "rounded up to 256 bytes (1024)"},
        {"ScratchOffsetNotAnInteger",
         with("ws.scratch_offset", MetadataValue{std::string{"1024"}}),
         "ws.scratch_offset is not an integer"},
        {"NegativeOffsetToken",
         with("ws.delta.offset", MetadataValue{int64_t{-5}}),
         "offset token 'ws.delta.offset' is not a non-negative integer"},
        {"NonIntegerOffsetToken",
         with("ws.lse.offset", MetadataValue{std::string{"x"}}),
         "offset token 'ws.lse.offset' is not a non-negative integer"},
        {"TheFirstBadOffsetTokenByNameIsReported",
         [&] {
             auto tokens = with("ws.b.offset", MetadataValue{std::string{"x"}});
             tokens["ws.a.offset"] = MetadataValue{int64_t{-1}};
             return tokens;
         }(),
         "offset token 'ws.a.offset' is not a non-negative integer"},
    };
    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const ScopedSlottedMatch symbols;
        graphMatchState().tokens = testCase.tokens;
        const auto manager = makeManager();
        const TestGraph graph(makeGraphId(1));
        const auto properties = testDeviceProperties();
        const MatchContext context{graph, 0, properties};

        const auto verdict = manager->slotProgramVerdict(context);

        EXPECT_FALSE(verdict.applicable);
        EXPECT_EQ(verdict.declineReason, testCase.declineReason);
        EXPECT_EQ(verdict.sharedWorkspaceBytes, 0U);
        for(size_t slot = 0; slot < manager->slotCount(); ++slot)
        {
            EXPECT_TRUE(manager->unsortedCatalog(slot, context).entries.empty())
                << "slot " << slot << " served a kernel of a declined engine";
        }
        EXPECT_EQ(graphMatchState().calls, 1);
    }
}

TEST(TestSlottedStateManager, ASharedRegionIsValidatedAgainstItsAlignedSize)
{
    EXPECT_EQ(alignSharedWorkspace(0), 0U);
    EXPECT_EQ(alignSharedWorkspace(1), 256U);
    EXPECT_EQ(alignSharedWorkspace(256), 256U);
    EXPECT_EQ(alignSharedWorkspace(257), 512U);

    struct Region
    {
        int64_t shared;
        int64_t scratchOffset;
    };
    for(const auto& region : {Region{0, 0}, Region{256, 256}, Region{257, 512}, Region{1000, 1024}})
    {
        SCOPED_TRACE("shared " + std::to_string(region.shared));
        const ScopedSlottedMatch symbols;
        auto& tokens = graphMatchState().tokens;
        tokens.clear();
        tokens[WS_BYTES] = region.shared;
        tokens["ws.scratch_offset"] = region.scratchOffset;
        const auto manager = makeManager();
        const TestGraph graph(makeGraphId(1));
        const auto properties = testDeviceProperties();
        const MatchContext context{graph, 0, properties};

        const auto verdict = manager->slotProgramVerdict(context);

        EXPECT_TRUE(verdict.applicable) << verdict.declineReason;
        EXPECT_EQ(verdict.sharedWorkspaceBytes, static_cast<size_t>(region.shared));
    }
}

TEST(TestSlottedStateManager, AnEngineDeclaringNoSharedWorkspaceValidatesNoWorkspaceTokens)
{
    const ScopedSlottedMatch symbols;
    graphMatchState().tokens.clear();
    graphMatchState().tokens["ws.scratch_offset"] = int64_t{7};
    const auto manager = makeManager(defaultSlots(), /*sharedWorkspaceToken=*/"");
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto verdict = manager->slotProgramVerdict(context);

    EXPECT_TRUE(verdict.applicable) << verdict.declineReason;
    EXPECT_EQ(verdict.sharedWorkspaceBytes, 0U);
}

TEST(TestSlottedStateManager, ASlotListThatRepeatsANameOrIsEmptyIsRejected)
{
    const ScopedSlottedMatch symbols;
    std::vector<KernelSlotInput> repeated;
    repeated.push_back(slotInput("prep", 0xA1, prepPacks()));
    repeated.push_back(slotInput("prep", 0xA2, dkdvPacks()));

    EXPECT_THROW(makeManager(std::move(repeated)), std::invalid_argument);
    EXPECT_THROW(makeManager({}), std::invalid_argument);
}

TEST(TestSlottedStateManager, ACatalogKeyDistinguishesTheSlotItsCatalogBelongsTo)
{
    const ContentCarryingTestGraph graph{ContentCarryingTestGraph::Spec{}};
    const auto makeKey = [&graph] {
        return CatalogKey{
            GraphContentKey{graph}, CatalogGraphFields::of(graph.getGraph()), {1, 2}, 0};
    };
    const CatalogKey unslotted = makeKey();
    CatalogKey slotted = unslotted;
    slotted.slot = "dq";
    CatalogKey other = unslotted;
    other.slot = "dkdv";

    EXPECT_FALSE(unslotted == slotted);
    EXPECT_FALSE(slotted == other);
    CatalogKey sameSlot = unslotted;
    sameSlot.slot = "dq";
    EXPECT_TRUE(slotted == sameSlot);
    EXPECT_NE(CatalogKeyHash{}(slotted), CatalogKeyHash{}(other));
}

// ----- ws.* reserve ------------------------------------------------------------------------

std::vector<std::vector<double>>& capturedRows()
{
    static std::vector<std::vector<double>> s_rows;
    return s_rows;
}

double captureRow(const double* features, size_t count)
{
    capturedRows().emplace_back(features, features + count);
    return 1.0;
}

constexpr const char* CAPTURE_SYMBOL = "hipdnn.kernel_ingestor.test.slotted.capture_row";

/// A ranker whose model reads `$ws.bytes` and the slot's `block_m`.
HeuristicDescriptor wsReadingRanker()
{
    const std::vector<nlohmann::json> signature = {"$ws.bytes", "$kernel.block_m"};
    HeuristicDescriptor ranker;
    ranker.id = testId(0xC1);
    ranker.name = "ws reader";
    ranker.adapter = UhdAdapter::NATIVE;
    ranker.nativeSymbol = CAPTURE_SYMBOL;
    ranker.featuresSignature = signature;
    ranker.featuresHash = hipdnn_plugin_sdk::uhd::FeatureExtractor::computeHash(signature);
    return ranker;
}

DescriptorSet makeSlottedSet()
{
    DescriptorSet set;
    set.engine.id = testId(0xE0);
    set.engine.name = "test:slotted";
    set.engine.graphMatchNativeSymbol = GRAPH_MATCH;
    set.engine.sharedWorkspaceToken = WS_BYTES;
    set.dispatches = makeStubDispatches();
    struct SlotSpec
    {
        const char* name;
        uint8_t schemaSeed;
        std::vector<KernelDescriptorPack> packs;
        std::vector<std::string> knobs;
    };
    const std::array<SlotSpec, 3> slots{{{"prep", 0xA1, prepPacks(), {}},
                                         {"dkdv", 0xA2, dkdvPacks(), {BLOCK_M, BLOCK_N}},
                                         {"dq", 0xA3, dqPacks(), {BLOCK_M, BLOCK_N}}}};
    for(const auto& slot : slots)
    {
        SlotDescriptor described;
        described.name = slot.name;
        described.metadataSchemaId = testId(slot.schemaSeed);
        described.knobs = slot.knobs;
        set.engine.slots.push_back(std::move(described));
        SlotDescriptorSet resolved;
        resolved.name = slot.name;
        resolved.schema = slotSchema(slot.schemaSeed, std::string(slot.name) + " kmd");
        resolved.packs = slot.packs;
        set.slots.push_back(std::move(resolved));
    }
    return set;
}

TEST(TestSlottedStateManager, WorkspaceTokensAreNotProblemFeaturesOfASlottedEngine)
{
    const MatmulTestGraph matmul(8, 8, 8);
    const auto properties = testDeviceProperties();
    const MatchContext context{matmul.graph(), 0, properties};
    const BoundTokens bound{{WS_BYTES, int64_t{1000}}, {"attention.scale", int64_t{3}}};

    const auto unslotted
        = hipdnn_plugin_sdk::ingestor::detail::catalogProblemFeatures(context, bound).toJson();
    const auto slotted = hipdnn_plugin_sdk::ingestor::detail::catalogProblemFeatures(
                             context, bound, /*reserveWorkspace=*/true)
                             .toJson();

    EXPECT_TRUE(unslotted.contains(WS_BYTES)) << "an unslotted engine's ws.* stays a feature";
    EXPECT_FALSE(slotted.contains(WS_BYTES));
    EXPECT_TRUE(slotted.contains("attention.scale")) << "shared runtime args stay features";
    EXPECT_TRUE(slotted.contains("graph.flops"));
    EXPECT_TRUE(hipdnn_plugin_sdk::ingestor::detail::isReservedFeatureName("$ws.bytes", true));
    EXPECT_FALSE(hipdnn_plugin_sdk::ingestor::detail::isReservedFeatureName("$ws.bytes"));
    EXPECT_FALSE(hipdnn_plugin_sdk::ingestor::detail::isReservedFeatureName("wsx.bytes", true));

    using hipdnn_plugin_sdk::ingestor::detail::isWorkspaceToken;
    EXPECT_TRUE(isWorkspaceToken("ws.bytes"));
    EXPECT_TRUE(isWorkspaceToken("$ws.bytes"));
    EXPECT_FALSE(isWorkspaceToken("wsx.bytes"));
    EXPECT_FALSE(isWorkspaceToken("$other.rows.m")) << "the root is a prefix, not a substring";
    EXPECT_FALSE(isWorkspaceToken("$"));
    EXPECT_FALSE(isWorkspaceToken(""));
}

TEST(TestSlottedStateManager, ASlotRankerNeverSeesWorkspaceTokensButAnUnslottedOneStill)
{
    // The slotted engine is named, so its winner cache is read from a shard under the cache root.
    const ScopedCacheDir cacheDir("slot_ws_tokens");
    const ScopedSlottedMatch symbols;
    const hipdnn_plugin_sdk::uhd::ScopedNativeScorer capture(CAPTURE_SYMBOL, &captureRow);
    const MatmulTestGraph matmul(8, 8, 8);
    const auto properties = testDeviceProperties();
    const MatchContext context{matmul.graph(), 0, properties};

    // Slotted: the model's `$ws.bytes` is unbound, so ranking degrades to declared order and
    // the model never scores.
    auto slottedSet = makeSlottedSet();
    slottedSet.slots[1].heuristic = wsReadingRanker();
    const auto slotted = makeStateManager<StubHandle>(std::move(slottedSet), GRAPH_MATCH);
    capturedRows().clear();
    const auto ranked = slotted->sortedCatalog(1, context);
    EXPECT_TRUE(capturedRows().empty());
    EXPECT_EQ(namesOf(ranked), (Names{"dkdv_a", "dkdv_b", "dkdv_c"}));

    // Unslotted: the same model reads the token the graph match published.
    DescriptorSet unslottedSet;
    unslottedSet.engine.graphMatchNativeSymbol = GRAPH_MATCH;
    unslottedSet.engine.knobs = {BLOCK_M, BLOCK_N};
    unslottedSet.schema = slotSchema(0xA2, "dkdv kmd");
    unslottedSet.dispatches = makeStubDispatches();
    unslottedSet.heuristic = wsReadingRanker();
    unslottedSet.packs = dkdvPacks();
    unslottedSet.packs.front().slot.clear();
    // No engine name: no on-disk winner cache is consulted.
    const auto unslotted = makeStateManager<StubHandle>(std::move(unslottedSet), GRAPH_MATCH);
    capturedRows().clear();
    (void)unslotted->sortedCatalog(context);
    ASSERT_EQ(capturedRows().size(), 3U) << "an unslotted engine's ranker must still bind ws.*";
    EXPECT_DOUBLE_EQ(capturedRows().front().front(), 1000.0);
}

constexpr const char* DKDV_SCORE = "hipdnn.kernel_ingestor.test.slotted.dkdv_score";

/// Larger tiles first: the opposite of the kernel-id order every slot's kernels are declared in.
double scoreByTileExtent(const MatchContext& /*context*/,
                         const BoundTokens& /*bound*/,
                         const KernelDefinition& kernel)
{
    return static_cast<double>(1000 * kernel.getIntMetadata(BLOCK_M)
                               + kernel.getIntMetadata(BLOCK_N));
}

TEST(TestSlottedStateManager, ASlotsRankerOrdersOnlyThatSlot)
{
    const ScopedSlottedMatch symbols;
    ScoreRegistry::registerSymbol(DKDV_SCORE, &scoreByTileExtent);
    const hipdnn_data_sdk::utilities::ScopedResource<const char*> scorer{
        DKDV_SCORE, [](const char* symbol) { ScoreRegistry::unregisterSymbol(symbol); }};
    auto slots = defaultSlots();
    slots[1].heuristic = std::make_shared<NativeKernelHeuristic>(DKDV_SCORE);
    const auto manager = makeManager(std::move(slots));
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_EQ(namesOf(manager->sortedCatalog(1, context)), (Names{"dkdv_c", "dkdv_b", "dkdv_a"}));
    EXPECT_EQ(namesOf(manager->sortedCatalog(0, context)), (Names{"prep_small", "prep_large"}))
        << "prep keeps its own, unranked order";
    EXPECT_EQ(namesOf(manager->sortedCatalog(2, context)), (Names{"dq_a", "dq_b"}))
        << "dq keeps its own, unranked order";
}

// ----- identity ----------------------------------------------------------------------------

HeuristicDescriptor slotRanker(uint8_t seed, const std::string& modelHash)
{
    HeuristicDescriptor ranker;
    ranker.id = testId(seed);
    ranker.name = "slot ranker";
    ranker.adapter = UhdAdapter::TREE_DATA;
    ranker.modelArtifactPath = "model.bin";
    ranker.modelHash = modelHash;
    return ranker;
}

DescriptorSet makeRankedSlottedSet()
{
    auto set = makeSlottedSet();
    set.slots[1].heuristic = slotRanker(0xC1, std::string(64, '1'));
    set.slots[2].heuristic = slotRanker(0xC2, std::string(64, '2'));
    return set;
}

TEST(TestSlottedStateManager, ASlottedEngineIdentityCoversEverySlotRanker)
{
    const auto set = makeRankedSlottedSet();

    const auto identity = engineIdentity(set);

    EXPECT_EQ(identity.name, "test:slotted");
    EXPECT_EQ(identity.uhdId, toString(set.engine.id))
        << "a slotted engine has no ranker of its own";
    EXPECT_EQ(identity.modelHash.size(), 64U);
    EXPECT_TRUE(identity.contentIdentified);
    EXPECT_EQ(engineIdentity(makeRankedSlottedSet()).modelHash, identity.modelHash)
        << "the hash is a function of the descriptors alone";
}

TEST(TestSlottedStateManager, ASlottedEngineWithNoSlotRankerHashesItsSlotNamesAndOrder)
{
    const auto set = makeSlottedSet();
    const auto hash = engineModelHash(set);

    EXPECT_EQ(hash.size(), 64U);
    EXPECT_TRUE(engineIdentity(set).contentIdentified);

    auto renamed = set;
    renamed.slots[1].name += "_renamed";
    EXPECT_NE(engineModelHash(renamed), hash);

    auto reordered = set;
    std::swap(reordered.slots[1], reordered.slots[2]);
    EXPECT_NE(engineModelHash(reordered), hash);
}

TEST(TestSlottedStateManager, TheModelHashChangesWithAnySlotRankerAndWithSlotOrder)
{
    const auto base = makeRankedSlottedSet();
    const auto baseHash = engineModelHash(base);

    auto retrained = base;
    retrained.slots[2].heuristic->modelHash = std::string(64, '3');
    EXPECT_NE(engineModelHash(retrained), baseHash);

    auto swapped = base;
    std::swap(swapped.slots[1], swapped.slots[2]);
    EXPECT_NE(engineModelHash(swapped), baseHash);
    EXPECT_NE(engineSelectorRevision(swapped), engineSelectorRevision(base));

    // The same ranker in another slot is another engine.
    auto moved = base;
    moved.slots[0].heuristic = moved.slots[1].heuristic;
    moved.slots[1].heuristic.reset();
    EXPECT_NE(engineModelHash(moved), baseHash);
}

TEST(TestSlottedStateManager, ASlotRankerNamingAnArtifactWithoutADigestIsNotContentIdentified)
{
    auto set = makeRankedSlottedSet();
    set.slots[2].heuristic->modelHash.clear();

    EXPECT_FALSE(engineContentIdentified(set));
    EXPECT_FALSE(engineIdentity(set).contentIdentified);

    set.slots[2].heuristic->modelHash = std::string(64, 'a');
    EXPECT_TRUE(engineContentIdentified(set));

    // A slot without a ranker never disables persistence.
    set.slots[0].heuristic.reset();
    EXPECT_TRUE(engineContentIdentified(set));

    // The per-metric rankers are checked too.
    set.slots[0].heuristicsByMetric["tflops"]["default"] = slotRanker(0xC3, "");
    EXPECT_FALSE(engineContentIdentified(set));
}

TEST(TestSlottedStateManager, TheSelectorRevisionCoversSlotNamesKnobsAndRankers)
{
    const auto base = makeRankedSlottedSet();
    const auto revision = engineSelectorRevision(base);
    EXPECT_EQ(revision.rfind("generic-untuned-v1/", 0), 0U);
    EXPECT_EQ(engineSelectorRevision(makeRankedSlottedSet()), revision);

    auto renamed = base;
    renamed.slots[1].name = "dkdv2";
    EXPECT_NE(engineSelectorRevision(renamed), revision);

    auto reknobbed = base;
    reknobbed.engine.slots[1].knobs = {BLOCK_M};
    EXPECT_NE(engineSelectorRevision(reknobbed), revision);

    auto reranked = base;
    reranked.slots[2].heuristic->modelHash = std::string(64, '3');
    EXPECT_NE(engineSelectorRevision(reranked), revision);

    auto retokened = base;
    retokened.engine.sharedWorkspaceToken = "ws.other";
    EXPECT_NE(engineSelectorRevision(retokened), revision);
}

TEST(TestSlottedStateManager, SlottedProvenanceNamesEverySlotKmdInsteadOfOne)
{
    const auto provenance = enginePredictionProvenance(makeRankedSlottedSet());

    EXPECT_FALSE(provenance.contains("kmd"));
    ASSERT_TRUE(provenance.contains("kmds"));
    const auto& kmds = provenance.at("kmds");
    ASSERT_EQ(kmds.size(), 3U);
    EXPECT_EQ(kmds[0].at("slot"), "prep");
    EXPECT_EQ(kmds[1].at("slot"), "dkdv");
    EXPECT_EQ(kmds[2].at("slot"), "dq");
    EXPECT_EQ(kmds[1].at("id"), toString(testId(0xA2)));
    EXPECT_EQ(kmds[1].at("revision"), "1.0");
}

// ----- engine construction -----------------------------------------------------------------

TEST(TestSlottedStateManager, TheEngineChecksEachSlotsKnobsAgainstThatSlotsSchema)
{
    const ScopedSlottedMatch symbols;
    const StubDeviceResolver resolver;

    EXPECT_NO_THROW(
        (makeEngine<StubHandle, StubSettings, StubContext>(makeSlottedSet(), resolver)));

    auto bad = makeSlottedSet();
    bad.engine.slots[2].knobs = {BLOCK_M, "no_such_field"};
    try
    {
        (void)makeEngine<StubHandle, StubSettings, StubContext>(std::move(bad), resolver);
        FAIL() << "expected std::invalid_argument";
    }
    catch(const std::invalid_argument& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("slot 'dq'"), std::string::npos) << message;
        EXPECT_NE(message.find("'no_such_field'"), std::string::npos) << message;
    }
}

// ----- multi-launch kernels in a slot (UKD `sources`) --------------------------------------

constexpr const char* SINGLE_DISPATCH_SYMBOL = "hipdnn.kernel_ingestor.test.slotted.single_launch";
constexpr const char* MULTI_DISPATCH_SYMBOL = "hipdnn.kernel_ingestor.test.slotted.multi_launch";

/// A native handler that launches the named stages; the checks under test never call prepare or
/// launch.
class LaunchesHandler : public NoopDispatchHandler<StubHandle>
{
public:
    explicit LaunchesHandler(std::set<std::string> launches)
        : _launches(std::move(launches))
    {
    }

    std::set<std::string> launchNames() const override
    {
        return _launches;
    }

private:
    std::set<std::string> _launches;
};

/// Two dispatches, one dispatch descriptor per launch set: slots with different launch sets need
/// different dispatch descriptors.
class ScopedLaunchDispatches
{
public:
    ScopedLaunchDispatches() = default;

    static DescriptorId singleId()
    {
        return testId(0xD1);
    }

    static DescriptorId multiId()
    {
        return testId(0xD2);
    }

    static std::vector<DispatchDescriptor> descriptors()
    {
        return {{singleId(), "single-launch dispatch", SINGLE_DISPATCH_SYMBOL},
                {multiId(), "multi-launch dispatch", MULTI_DISPATCH_SYMBOL}};
    }

private:
    LaunchesHandler _single{{}};
    LaunchesHandler _multi{{"decode", "merge"}};
    ScopedDispatchRegistration<StubHandle> _singleRegistration{SINGLE_DISPATCH_SYMBOL, _single};
    ScopedDispatchRegistration<StubHandle> _multiRegistration{MULTI_DISPATCH_SYMBOL, _multi};
};

/// A kernel whose launches are @p launches, each a kpack entry of its own symbol.
KernelDescriptor multiLaunchKernel(uint8_t seed,
                                   const std::string& name,
                                   int64_t blockM,
                                   int64_t blockN,
                                   const std::vector<std::string>& launches = {"decode", "merge"})
{
    auto kernel = makeSlotKernel(
        seed, name, {{BLOCK_M, MetadataValue{blockM}}, {BLOCK_N, MetadataValue{blockN}}});
    kernel.source = KernelSource{};
    for(const auto& launch : launches)
    {
        auto symbol = name;
        symbol += "/";
        symbol += launch;
        kernel.sources.emplace(launch,
                               makeKpackSource("kpack/test.kpack", symbol, launch + "_kernel"));
    }
    return kernel;
}

/// A single-launch kernel in the same KMD.
KernelDescriptor
    singleLaunchKernel(uint8_t seed, const std::string& name, int64_t blockM, int64_t blockN)
{
    return makeSlotKernel(
        seed, name, {{BLOCK_M, MetadataValue{blockM}}, {BLOCK_N, MetadataValue{blockN}}});
}

KernelDescriptorPack launchPack(uint8_t seed,
                                const std::string& slot,
                                const DescriptorId& dispatch,
                                std::vector<KernelDescriptor> kernels)
{
    auto pack = makeSlotPack(seed, slot, std::move(kernels));
    pack.dispatchId = dispatch;
    return pack;
}

/// prep (single-launch handler) and dkdv (a handler launching decode and merge) with @p dkdv.
std::unique_ptr<Manager> makeLaunchManager(std::vector<KernelDescriptor> prep,
                                           std::vector<KernelDescriptor> dkdv)
{
    std::vector<KernelSlotInput> slots;
    slots.push_back(
        slotInput("prep",
                  0xA1,
                  {launchPack(0x10, "prep", ScopedLaunchDispatches::singleId(), std::move(prep))}));
    slots.push_back(
        slotInput("dkdv",
                  0xA2,
                  {launchPack(0x20, "dkdv", ScopedLaunchDispatches::multiId(), std::move(dkdv))}));
    return std::make_unique<Manager>(std::move(slots),
                                     std::vector<MatchDescriptor>{},
                                     ScopedLaunchDispatches::descriptors(),
                                     GRAPH_MATCH,
                                     WS_BYTES,
                                     "engine 'launch fixture'");
}

std::vector<KernelDescriptor> twoPrepKernels()
{
    return {singleLaunchKernel(0x11, "prep_small", 64, 1),
            singleLaunchKernel(0x12, "prep_large", 128, 1)};
}

TEST(TestSlottedStateManager, AMultiLaunchSlotBesideASingleLaunchSlotIsApplicable)
{
    const ScopedSlottedMatch symbols;
    const ScopedLaunchDispatches dispatches;
    const auto manager = makeLaunchManager(
        twoPrepKernels(),
        {multiLaunchKernel(0x21, "dkdv_a", 64, 64), multiLaunchKernel(0x22, "dkdv_b", 128, 64)});
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto verdict = manager->slotProgramVerdict(context);

    EXPECT_TRUE(verdict.applicable) << verdict.declineReason;
    const auto dkdv = manager->unsortedCatalog(1, context);
    ASSERT_EQ(dkdv.entries.size(), 2U);
    for(const auto* kernel : dkdv.entries)
    {
        EXPECT_EQ(kernel->sources.size(), 2U) << kernel->name << " lost its launches";
    }
    EXPECT_EQ(namesOf(manager->unsortedCatalog(0, context)).size(), 2U);
}

TEST(TestSlottedStateManager, AKernelWhoseLaunchNamesDifferFromItsHandlersIsDroppedFromThatSlotOnly)
{
    const ScopedSlottedMatch symbols;
    const ScopedLaunchDispatches dispatches;
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    const auto manager
        = makeLaunchManager(twoPrepKernels(),
                            {multiLaunchKernel(0x21, "dkdv_good", 64, 64),
                             multiLaunchKernel(0x22, "dkdv_renamed", 128, 64, {"decode", "other"}),
                             singleLaunchKernel(0x23, "dkdv_single", 128, 128)});
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);

    EXPECT_EQ(namesOf(manager->unsortedCatalog(1, context)),
              (std::vector<std::string>{"dkdv_good"}));
    EXPECT_EQ(namesOf(manager->unsortedCatalog(0, context)).size(), 2U)
        << "the other slot's single-launch kernels are unaffected";
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "kernel 'dkdv_renamed'"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "kernel 'dkdv_single'"));
}

/// The launch set is the slot's: a single-launch kernel is as unlaunchable by a two-launch
/// handler as a renamed one, and a slot left with nothing declines the engine while the other
/// slots keep their catalogs.
TEST(TestSlottedStateManager, ASlotWhoseEveryKernelMismatchesItsHandlerDeclinesTheEngine)
{
    const ScopedSlottedMatch symbols;
    const ScopedLaunchDispatches dispatches;
    const auto manager
        = makeLaunchManager(twoPrepKernels(), {singleLaunchKernel(0x21, "dkdv_single", 64, 64)});
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto verdict = manager->slotProgramVerdict(context);

    EXPECT_FALSE(verdict.applicable);
    EXPECT_NE(verdict.declineReason.find("slot 'dkdv'"), std::string::npos)
        << verdict.declineReason;
    EXPECT_EQ(manager->unsortedCatalog(0, context).entries.size(), 2U);
}

TEST(TestSlottedStateManager, AnUnadaptableLaunchDropsOnlyItsKernel)
{
    const ScopedSlottedMatch symbols;
    const ScopedLaunchDispatches dispatches;
    auto oneNot = multiLaunchKernel(0x22, "dkdv_one_not", 128, 64);
    oneNot.sources.at("merge").kind = KernelSourceKind::HSACO_FILE;
    const auto manager = makeLaunchManager(twoPrepKernels(),
                                           {multiLaunchKernel(0x21, "dkdv_all", 64, 64), oneNot});
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);

    EXPECT_EQ(namesOf(manager->unsortedCatalog(1, context)),
              (std::vector<std::string>{"dkdv_all"}));
    EXPECT_EQ(manager->unsortedCatalog(0, context).entries.size(), 2U);
}

/// A kernel dropped for its launch set claims nothing, so a sibling sharing its metadata tuple
/// is not refused as a duplicate (which in a slotted engine would drop the whole pack).
TEST(TestSlottedStateManager, AKernelDroppedForItsLaunchSetClaimsNoMetadataTuple)
{
    const ScopedSlottedMatch symbols;
    const ScopedLaunchDispatches dispatches;
    const auto manager
        = makeLaunchManager(twoPrepKernels(),
                            {multiLaunchKernel(0x21, "dkdv_renamed", 64, 64, {"decode", "other"}),
                             multiLaunchKernel(0x22, "dkdv_good", 64, 64)});
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);
    EXPECT_EQ(namesOf(manager->unsortedCatalog(1, context)),
              (std::vector<std::string>{"dkdv_good"}));
}

/// `sources` is part of the selector identity of a slotted engine's kernels, so a launch's code
/// object changing makes it a different selector.
TEST(TestSlottedStateManager, ASlotKernelsSourcesChangeTheEngineIdentity)
{
    auto base = makeSlottedSet();
    auto& dkdvKernel = base.slots[1].packs.front().kernels.front();
    dkdvKernel = multiLaunchKernel(dkdvKernel.id.front(), "dkdv_a", 64, 64);
    const auto revision = engineSelectorRevision(base);
    EXPECT_EQ(engineSelectorRevision(base), revision);

    auto resymbolled = base;
    resymbolled.slots[1].packs.front().kernels.front().sources.at("merge").symbol = "other_kernel";
    EXPECT_NE(engineSelectorRevision(resymbolled), revision);

    auto rehashed = base;
    rehashed.slots[1].packs.front().kernels.front().sources.at("decode").sha256
        = std::string(64, 'b');
    EXPECT_NE(engineSelectorRevision(rehashed), revision);

    auto launchRenamed = base;
    auto& sources = launchRenamed.slots[1].packs.front().kernels.front().sources;
    sources.emplace("renamed", sources.at("merge"));
    sources.erase("merge");
    EXPECT_NE(engineSelectorRevision(launchRenamed), revision);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
