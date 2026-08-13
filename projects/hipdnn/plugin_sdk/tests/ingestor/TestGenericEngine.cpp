// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestGenericEngine.cpp
 * @brief Unit tests for GenericEngine.hpp: knob validation at construction and
 *        IEngine overrides delegating to the plan builder and state manager.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

using StubEngine = GenericEngine<StubHandle, StubSettings, StubContext>;

// GenericEngine holds its plan builder by reference to its own members, so it is
// non-movable and non-copyable.
static_assert(!std::is_move_constructible_v<StubEngine>);
static_assert(!std::is_move_assignable_v<StubEngine>);
static_assert(!std::is_copy_constructible_v<StubEngine>);
static_assert(!std::is_copy_assignable_v<StubEngine>);

// ---------------------------------------------------------------------------
// Construction: knob validation against the metadata schema.
// ---------------------------------------------------------------------------

TEST(TestIngestorGenericEngine, AcceptsAKnobNamingADeclaredMetadataField)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    EXPECT_NO_THROW(
        (StubEngine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver)));
}

TEST(TestIngestorGenericEngine, RejectsAKnobNamingNoMetadataField)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    // A knob naming no metadata field would otherwise be silently dropped.
    EXPECT_THROW(
        (StubEngine(makeEngineWithKnobs({"no_such_field"}), makeStubStateManager(), resolver)),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// IEngine overrides: each delegates to the plan builder or the state manager.
// ---------------------------------------------------------------------------

TEST(TestIngestorGenericEngine, IdHashesTheUedNameIntoHipdnnsEngineIdSpace)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    // Pins that construction produces a stable, non-zero id.
    EXPECT_NE(engine.id(), 0);
}

TEST(TestIngestorGenericEngine, IsApplicableTrueWhenTheStateManagerHasASurvivingKernel)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x60));

    EXPECT_TRUE(engine.isApplicable(handle, graph));
}

TEST(TestIngestorGenericEngine, IsApplicableFalseWhenNoMatcherAccepts)
{
    // A distinct symbol avoids colliding with ScopedTestSymbols' matcher running
    // concurrently in another test.
    constexpr const char* REJECT_SYMBOL = "hipdnn.kernel_ingestor.test.generic_engine.reject";
    GraphMatcherRegistry::registerSymbol(REJECT_SYMBOL, &rejectGraph);
    ScoreRegistry::registerSymbol(SCORE_SYMBOL, &scoreByBlockSize);

    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.matcherIds = {GRAPH_MATCHER_ID};
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};

    auto stateManager = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{
            {GRAPH_MATCHER_ID, "graph scoped", MatchScope::GRAPH, REJECT_SYMBOL}},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), std::move(stateManager), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x61));

    EXPECT_FALSE(engine.isApplicable(handle, graph));

    GraphMatcherRegistry::unregisterSymbol(REJECT_SYMBOL);
    ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
}

TEST(TestIngestorGenericEngine, GetDetailsReportsTheEnginesKnobs)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x62));
    hipdnnPluginConstData_t details{};

    engine.getDetails(handle, graph, details);

    ASSERT_NE(details.ptr, nullptr);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    ASSERT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), engine.id());
    ASSERT_EQ(wrapper.knobCount(), 1U);
    EXPECT_EQ(wrapper.getKnobByName(BLOCK_SIZE).knobId(), BLOCK_SIZE);
}

TEST(TestIngestorGenericEngine, GetMaxWorkspaceSizeDelegatesToThePlanBuilder)
{
    // Asserted positively rather than by an expected throw. These once proved
    // delegation by leaving the dispatch symbol unregistered, which stopped working
    // when dispatch symbols began resolving at construction; and a test that passes
    // because something upstream threw cannot tell delegation from any other failure.
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x63));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);

    // The stub manager ships one 64-block kernel, and the handler reports block_size as
    // its workspace, so this value can only come from the plan builder having run.
    EXPECT_EQ(engine.getMaxWorkspaceSize(handle, graph, emptyConfig), 64U);
}

TEST(TestIngestorGenericEngine, InitializeExecutionContextDelegatesToThePlanBuilder)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x64));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);
    StubContext context;

    engine.initializeExecutionContext(handle, graph, emptyConfig, context);

    // A plan reached the context, which only the plan builder can put there.
    EXPECT_TRUE(context.hasPlan());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
