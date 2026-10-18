// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include "KernelIngestorTestFixtures.hpp"

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

using Manager = KernelIngestorStateManager<StubHandle>;

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
    KernelDescriptorPack pack;
    pack.id = testId(packSeed);
    pack.name = slot + " pack";
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.arch = arch;
    pack.slot = slot;
    for(const auto& spec : kernels)
    {
        KernelDescriptor kernel;
        kernel.id = testId(spec.seed);
        kernel.name = spec.name;
        kernel.source.sourceFile = "Test.cpp";
        kernel.source.entryPoint = spec.name;
        kernel.metadata
            = {{BLOCK_M, MetadataValue{spec.blockM}}, {BLOCK_N, MetadataValue{spec.blockN}}};
        pack.kernels.push_back(std::move(kernel));
    }
    return pack;
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
        names.push_back(entry.name);
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
}

TEST(TestSlottedStateManager, AnOutOfRangeSlotIsRejected)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_THROW(manager->unsortedCatalog(3, context), std::out_of_range);
    EXPECT_THROW(manager->slotName(3), std::out_of_range);
}

TEST(TestSlottedStateManager, OneGraphMatchServesEveryAccessorOfACatalogBuild)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
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

    // Another graph is another build.
    const TestGraph other(makeGraphId(2));
    const MatchContext otherContext{other, 0, properties};
    (void)manager->unsortedCatalog(2, otherContext);
    EXPECT_EQ(graphMatchState().calls, 2);
    (void)manager->unsortedCatalog(0, otherContext);
    (void)manager->unsortedCatalog(1, otherContext);
    EXPECT_EQ(graphMatchState().calls, 2);
}

TEST(TestSlottedStateManager, AGraphWithNoIdentityStillMatchesOncePerBuildNotOncePerSlot)
{
    const ScopedSlottedMatch symbols;
    const auto manager = makeManager();
    const auto properties = testDeviceProperties();
    const TestGraph graph;
    const MatchContext context{graph, 0, properties};

    EXPECT_TRUE(manager->slotProgramVerdict(context).applicable);

    EXPECT_EQ(graphMatchState().calls, 1);
}

TEST(TestSlottedStateManager, AnEvictedSlotCatalogRebuildsFromOneMatch)
{
    const ScopedSlottedMatch symbols;
    // Room for one graph's three slot catalogs, not a second graph's.
    const auto manager = makeManager(defaultSlots(), WS_BYTES, 4);
    const auto properties = testDeviceProperties();
    const TestGraph first(makeGraphId(1));
    const TestGraph second(makeGraphId(2));
    const MatchContext firstContext{first, 0, properties};
    const MatchContext secondContext{second, 0, properties};

    (void)manager->unsortedCatalog(0, firstContext);
    (void)manager->unsortedCatalog(0, secondContext);
    ASSERT_EQ(graphMatchState().calls, 2);

    EXPECT_EQ(namesOf(manager->unsortedCatalog(1, firstContext)),
              (Names{"dkdv_a", "dkdv_b", "dkdv_c"}));
    EXPECT_EQ(graphMatchState().calls, 3) << "the evicted entry rebuilds, once";
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
    EXPECT_EQ(dkdv.entries.front().getIntMetadata(BLOCK_M), 64);
    EXPECT_EQ(dq.entries.front().getIntMetadata(BLOCK_M), 64);
    EXPECT_EQ(dkdv.entries.front().getIntMetadata(BLOCK_N), 64);
    EXPECT_EQ(dq.entries.front().getIntMetadata(BLOCK_N), 64);
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

TEST(TestSlottedStateManager, AnUnslottedEngineIsOneImplicitSlotAndKeepsItsThrow)
{
    const ScopedSlottedMatch symbols;
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    // Same two kernels, same tuple: an unslotted engine still refuses to construct.
    auto repeated = slotPack(0x24, "", {{0x25, "one", 64, 64}, {0x26, "two", 64, 64}});
    EXPECT_THROW(Manager(slotSchema(0xA2, "kmd"),
                         {},
                         makeStubDispatches(),
                         {repeated},
                         std::make_shared<UnrankedKernelHeuristic>(),
                         GRAPH_MATCH),
                 std::invalid_argument);

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

TEST(TestSlottedStateManager, TheFirstOfSeveralBadOffsetTokensByNameIsReported)
{
    const ScopedSlottedMatch symbols;
    graphMatchState().tokens["ws.b.offset"] = MetadataValue{std::string{"x"}};
    graphMatchState().tokens["ws.a.offset"] = MetadataValue{int64_t{-1}};
    const auto manager = makeManager();
    const TestGraph graph(makeGraphId(1));
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    EXPECT_EQ(manager->slotProgramVerdict(context).declineReason,
              "offset token 'ws.a.offset' is not a non-negative integer");
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
    const CatalogKey unslotted{makeGraphId(1), 0};
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
    // An unslotted key hashes as it always has: the empty slot adds nothing.
    EXPECT_EQ(CatalogKeyHash{}(unslotted), CatalogKeyHash{}(CatalogKey{makeGraphId(1), 0}));
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
}

TEST(TestSlottedStateManager, ASlotRankerNeverSeesWorkspaceTokensButAnUnslottedOneStill)
{
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

TEST(TestSlottedStateManager, ASlottedEngineWithNoSlotRankerHasNoModelHash)
{
    const auto set = makeSlottedSet();

    EXPECT_EQ(engineModelHash(set), "");
    EXPECT_TRUE(engineIdentity(set).contentIdentified);
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

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
