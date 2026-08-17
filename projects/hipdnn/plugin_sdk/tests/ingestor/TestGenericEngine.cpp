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

static_assert(!std::is_move_constructible_v<StubEngine>);
static_assert(!std::is_move_assignable_v<StubEngine>);
static_assert(!std::is_copy_constructible_v<StubEngine>);
static_assert(!std::is_copy_assignable_v<StubEngine>);

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

    EXPECT_THROW(
        (StubEngine(makeEngineWithKnobs({"no_such_field"}), makeStubStateManager(), resolver)),
        std::invalid_argument);
}

TEST(TestIngestorGenericEngine, IdHashesTheUedNameIntoHipdnnsEngineIdSpace)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

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
    // Distinct symbol avoids colliding with ScopedTestSymbols' matcher elsewhere.
    constexpr const char* REJECT_SYMBOL = "hipdnn.kernel_ingestor.test.generic_engine.reject";
    const auto rejectMatcher = scopedGraphMatcher(REJECT_SYMBOL, &rejectGraph);
    const ScopedBlockSizeScore scorer;

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
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x63));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);

    // Stub manager ships one 64-block kernel; only the plan builder having run
    // explains this value.
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

    EXPECT_TRUE(context.hasPlan());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
