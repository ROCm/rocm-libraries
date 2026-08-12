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
 * @brief Unit tests for GenericEngine.hpp: engine-level knob validation at construction,
 *        and every IEngine override GenericEngine implements by delegating to its plan
 *        builder and state manager.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

using StubEngine = GenericEngine<StubHandle, StubSettings, StubContext>;

// GenericEngine holds a plan builder bound by reference to its own members (see the
// class doc); relocating it would dangle those references.
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

    // A knob is only a name: the field it points at supplies the type, the default and
    // the legal values, so a knob matching no field can never be reported or honoured.
    // Left unchecked it is silently dropped, which reads to a caller exactly like a knob
    // the engine chose not to expose.
    EXPECT_THROW(
        (StubEngine(makeEngineWithKnobs({"no_such_field"}), makeStubStateManager(), resolver)),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// IEngine overrides: each one delegates to the plan builder or the state manager, so
// these tests exercise that delegation end to end rather than re-testing the builder.
// ---------------------------------------------------------------------------

TEST(TestIngestorGenericEngine, IdHashesTheUedNameIntoHipdnnsEngineIdSpace)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    // The id is derived, not stored verbatim, so this only pins that construction
    // succeeds in producing a stable, non-zero id -- an engine with no id could never
    // be looked up by the host.
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
    // A distinct symbol so this test's registration does not collide with
    // ScopedTestSymbols' GRAPH_MATCH_SYMBOL running concurrently in another test.
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
        std::vector<DispatchDescriptor>{
            {DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}},
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
    // The single-kernel stub state manager has no dispatch symbol registered, so the
    // plan builder's own workspace query (through getDispatchDetails) throws -- proving
    // the call actually reached the plan builder rather than returning a stub zero.
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x63));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);

    EXPECT_THROW(engine.getMaxWorkspaceSize(handle, graph, emptyConfig), std::runtime_error);
}

TEST(TestIngestorGenericEngine, InitializeExecutionContextDelegatesToThePlanBuilder)
{
    // Same reasoning as the workspace test: no dispatch handler is registered for the
    // stub state manager's dispatch symbol, so buildPlan()'s own lookup throws. That
    // failure mode is only reachable if initializeExecutionContext() actually calls
    // through to the plan builder.
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x64));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);
    StubContext context;

    EXPECT_THROW(engine.initializeExecutionContext(handle, graph, emptyConfig, context),
                 std::runtime_error);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
