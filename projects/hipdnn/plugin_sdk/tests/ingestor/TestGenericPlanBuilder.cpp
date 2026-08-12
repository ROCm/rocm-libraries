// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "IngestorMocks.hpp"
#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestGenericPlanBuilder.cpp
 * @brief Unit tests for GenericPlanBuilder.hpp: applicability, workspace sizing, plan
 *        building, knob filtering, and per-handle device resolution.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using ::testing::_;
using ::testing::Ref;
using ::testing::Return;
using ::testing::ReturnRef;

// ---------------------------------------------------------------------------
// isApplicable / getMaxWorkspaceSize / buildPlan / getCustomKnobs, using a
// two-survivor catalog (TestHandle = int).
// ---------------------------------------------------------------------------

/// Workspace requirement equals the kernel's block_size, so getMaxWorkspaceSize can assert
/// a real max instead of a shared constant.
class WorkspaceEqualsBlockSizeHandler : public IKernelDispatchHandler<TestHandle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return static_cast<size_t>(kernel.getIntMetadata(BLOCK_SIZE));
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const TestHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
};

struct KnobFilterSettings
{
    KnobFilter ingestorKnobFilter;
};

struct KnobFilterContext
{
    void setExecutionSettings(const KnobFilterSettings& /*settings*/) {}

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
    }

    /// Safe without RTTI: setPlan() only ever receives a GenericPlan<TestHandle> here.
    const GenericPlan<TestHandle>& plan() const
    {
        return static_cast<const GenericPlan<TestHandle>&>(*_plan);
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> _plan;
};

using TestPlanBuilder = GenericPlanBuilder<TestHandle, KnobFilterSettings, KnobFilterContext>;

hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper
    makeEmptyEngineConfig(flatbuffers::FlatBufferBuilder& builder)
{
    builder.Finish(
        hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, ENGINE_ID.front()));
    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
        builder.GetBufferPointer(), builder.GetSize());
}

TEST(TestIngestorGenericPlanBuilder, IsApplicableTrueWhenTheCatalogHasASurvivor)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x90));

    EXPECT_TRUE(builder.isApplicable(0, graph));
}

TEST(TestIngestorGenericPlanBuilder, IsApplicableFalseWhenTheGraphMatcherRejects)
{
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x91));

    EXPECT_FALSE(builder.isApplicable(0, graph));
}

TEST(TestIngestorGenericPlanBuilder, GetMaxWorkspaceSizeTakesTheMaxAcrossSurvivors)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x92));
    // Two survivors (block_size 64, 256): expects the max, not the first entry or a sum.
    EXPECT_EQ(builder.getMaxWorkspaceSize(0, graph, KnobFilterSettings{}), 256U);
}

TEST(TestIngestorGenericPlanBuilder, BuildPlanThrowsInternalErrorOnAnEmptyRankedCatalog)
{
    // Applicability accepted a graph with no matching kernel at all (not a knob issue);
    // this must carry a different status/message than the knob-filtering case below.
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0x93));
    KnobFilterContext context;

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(ex.getMessage(),
                  "engine '" + engine.name + "' accepted this graph but has no applicable kernel");
    }
}

TEST(TestIngestorGenericPlanBuilder, GetMaxWorkspaceSizeThrowsInternalErrorOnAnEmptyCatalog)
{
    // Must agree with buildPlan()'s empty-catalog case above, not silently return 0.
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x9A));

    try
    {
        builder.getMaxWorkspaceSize(0, graph, KnobFilterSettings{});
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(ex.getMessage(),
                  "engine '" + engine.name + "' accepted this graph but has no applicable kernel");
    }
}

TEST(TestIngestorGenericPlanBuilder, GetCustomKnobsReportsMinMaxStepAndRankedDefault)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x94));
    const auto knobs = builder.getCustomKnobs(0, graph);

    ASSERT_EQ(knobs.size(), 1U);
    const auto& knob = knobs.front();
    EXPECT_EQ(knob.knob_id, BLOCK_SIZE);
    ASSERT_TRUE(knob.constraint.AsIntConstraint() != nullptr);
    const auto& constraint = *knob.constraint.AsIntConstraint();
    EXPECT_EQ(constraint.min_value, 64);
    EXPECT_EQ(constraint.max_value, 256);
    EXPECT_EQ(constraint.step, 1);
    ASSERT_TRUE(knob.default_value.AsIntValue() != nullptr);
    // The heuristic ranks 256 first, so leaving the knob alone must reproduce that pick.
    EXPECT_EQ(knob.default_value.AsIntValue()->value, 256);
}

TEST(TestIngestorGenericPlanBuilder, GetCustomKnobsSkipsFieldsWithNoIntegerValues)
{
    // dtype takes only string values, so it can never become an integer knob and must be
    // skipped.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE, DTYPE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x95));
    const auto knobs = builder.getCustomKnobs(0, graph);

    ASSERT_EQ(knobs.size(), 1U);
    EXPECT_EQ(knobs.front().knob_id, BLOCK_SIZE);
}

// ---------------------------------------------------------------------------
// Knob filtering: setting a knob must actually restrict kernel selection.
// ---------------------------------------------------------------------------

TEST(TestIngestorGenericPlanBuilder, HonorsAnExplicitKnobSettingOverTheHeuristicDefault)
{
    // The heuristic would pick 256 by default; setting block_size=64 must override it.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 64);

    const TestGraph graph(makeGraphId(30));
    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64);
}

TEST(TestIngestorGenericPlanBuilder, UnsatisfiableKnobValueThrowsInvalidValueNamingItAndTheValue)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    // No surviving kernel carries block_size 999: the catalog matches (two FLOAT
    // kernels), but knob filtering excludes both.
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 999);

    const TestGraph graph(makeGraphId(31));
    KnobFilterContext context;

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(ex.getMessage().find(BLOCK_SIZE), std::string::npos);
        EXPECT_NE(ex.getMessage().find("999"), std::string::npos);
        // Survivor count distinguishes "knobs excluded everything" from "graph matched nothing".
        EXPECT_NE(ex.getMessage().find("2 kernel(s) matched the graph before knob filtering"),
                  std::string::npos);
    }
}

TEST(TestIngestorGenericPlanBuilder, GetMaxWorkspaceSizeHonorsTheSameKnobFilterAsBuildPlan)
{
    // getMaxWorkspaceSize and buildPlan must apply the same knob filter: block_size=64
    // caps the max at 64, not the unfiltered catalog's 256.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 64);
    const TestGraph graph(makeGraphId(0x96));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_EQ(builder.getMaxWorkspaceSize(0, graph, settings), 64U);
}

TEST(TestIngestorGenericPlanBuilder,
     GetMaxWorkspaceSizeThrowsInvalidValueWhenTheFilterIsUnsatisfiable)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 999);
    const TestGraph graph(makeGraphId(0x97));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_THROW(builder.getMaxWorkspaceSize(0, graph, settings),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestIngestorGenericPlanBuilder,
     EmptyCatalogBeforeFilteringStillThrowsInternalErrorWithItsOriginalMessage)
{
    const ScopedSymbols symbols("test.graph", rejectGraph, "test.kernel", countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    // An empty engine config exercises the no-kernels-at-all path, not knob filtering.
    const auto engineConfig = makeEmptyEngineConfig(fbb);

    const TestGraph graph(makeGraphId(32));
    KnobFilterContext context;

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(ex.getMessage(),
                  "engine '" + engine.name + "' accepted this graph but has no applicable kernel");
    }
}

// ---------------------------------------------------------------------------
// readKnobFilter(): rejecting a non-integer setting for an engine-exposed knob.
// ---------------------------------------------------------------------------

TEST(TestIngestorGenericPlanBuilder, InitializeExecutionSettingsRejectsAFloatValuedKnobSetting)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeFloatKnobEngineConfig(fbb, BLOCK_SIZE, 64.0);
    const TestGraph graph(makeGraphId(0x9B));
    KnobFilterSettings settings;

    try
    {
        builder.initializeExecutionSettings(0, graph, engineConfig, settings);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_EQ(ex.getMessage(),
                  "engine '" + engine.name + "' knob '" + BLOCK_SIZE
                      + "' must be set to an integer value");
    }
}

TEST(TestIngestorGenericPlanBuilder, InitializeExecutionSettingsRejectsAStringValuedKnobSetting)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeStringKnobEngineConfig(fbb, BLOCK_SIZE, "fast");
    const TestGraph graph(makeGraphId(0x9C));
    KnobFilterSettings settings;

    try
    {
        builder.initializeExecutionSettings(0, graph, engineConfig, settings);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_EQ(ex.getMessage(),
                  "engine '" + engine.name + "' knob '" + BLOCK_SIZE
                      + "' must be set to an integer value");
    }
}

// ---------------------------------------------------------------------------
// contextFor(): a mocked device resolver proves per-handle device resolution folds into
// the MatchContext (uses StubHandle/StubSettings/StubContext to match IngestorMocks.hpp).
// ---------------------------------------------------------------------------

constexpr DeviceId DEVICE_FOR_HANDLE_A = 42;
constexpr DeviceId DEVICE_FOR_HANDLE_B = 7;
constexpr const char* DEVICE_GATED_MATCH_SYMBOL
    = "hipdnn.kernel_ingestor.test.generic_plan_builder.device_gated";

bool acceptsOnlyDeviceA(const MatchContext& context, BoundTokens& /*bound*/)
{
    return context.deviceId == DEVICE_FOR_HANDLE_A;
}

TEST(TestIngestorGenericPlanBuilder, ContextForFoldsPerHandleDeviceResolutionIntoTheMatchContext)
{
    GraphMatcherRegistry::registerSymbol(DEVICE_GATED_MATCH_SYMBOL, &acceptsOnlyDeviceA);
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

    const KernelIngestorStateManager<StubHandle> manager(
        std::move(schema),
        std::vector<MatchDescriptor>{
            {GRAPH_MATCHER_ID, "device gated", MatchScope::GRAPH, DEVICE_GATED_MATCH_SYMBOL}},
        std::vector<DispatchDescriptor>{
            {DISPATCH_ID, "test dispatch", "hipdnn.kernel_ingestor.test.dispatch"}},
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL));

    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const MockDeviceResolver resolver;
    const auto properties = testDeviceProperties();
    const StubHandle handleA;
    const StubHandle handleB;

    EXPECT_CALL(resolver, deviceId(Ref(handleA))).WillRepeatedly(Return(DEVICE_FOR_HANDLE_A));
    EXPECT_CALL(resolver, deviceId(Ref(handleB))).WillRepeatedly(Return(DEVICE_FOR_HANDLE_B));
    EXPECT_CALL(resolver, deviceProperties(_)).WillRepeatedly(ReturnRef(properties));

    const GenericPlanBuilder<StubHandle, StubSettings, StubContext> builder(
        engine, manager, resolver);
    const TestGraph graph(makeGraphId(0x98));

    // Same builder and graph, two handles differing only in resolved device: isApplicable
    // can differ only if contextFor() asks the resolver per handle.
    EXPECT_TRUE(builder.isApplicable(handleA, graph));
    EXPECT_FALSE(builder.isApplicable(handleB, graph));

    GraphMatcherRegistry::unregisterSymbol(DEVICE_GATED_MATCH_SYMBOL);
    ScoreRegistry::unregisterSymbol(SCORE_SYMBOL);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
