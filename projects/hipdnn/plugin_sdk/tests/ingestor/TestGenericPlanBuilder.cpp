// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
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
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

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
    IngestorSettings ingestorSettings;
};

struct KnobFilterContext
{
    void setExecutionSettings(const KnobFilterSettings& settings)
    {
        _settings = settings;
    }

    const KnobFilterSettings& executionSettings() const
    {
        return _settings;
    }

    /// buildPlan() calls this with a GenericPlan on the benchmarking-off branch and a
    /// BenchmarkPlan on the benchmarking-on one. Only the former may be narrowed below,
    /// so record which arrived. RTTI is off in this build, so the type cannot be
    /// recovered from the plan itself.
    void setPlan(std::unique_ptr<GenericPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
        _planIsGeneric = true;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
        _planIsGeneric = false;
    }

    /// @throws if buildPlan() produced anything but a plain GenericPlan. Narrowing a
    /// BenchmarkPlan here would be UB; a stray HIPDNN_FORCE_BENCHMARKING is the way that
    /// happens, and it must fail the test rather than corrupt it.
    const GenericPlan<TestHandle>& plan() const
    {
        if(!_planIsGeneric)
        {
            throw std::runtime_error(
                "expected a plain GenericPlan; buildPlan() produced a different plan type "
                "(benchmarking unexpectedly on?)");
        }
        return static_cast<const GenericPlan<TestHandle>&>(*_plan);
    }

private:
    KnobFilterSettings _settings;
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> _plan;
    bool _planIsGeneric = false;
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

/// Fails the way HandleDeviceResolver does when hipGetDeviceProperties fails.
class ThrowingDeviceResolver : public IDeviceResolver<TestHandle>
{
public:
    DeviceId deviceId(const TestHandle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId deviceId) const override
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "hipGetDeviceProperties failed for device "
                                                           + std::to_string(deviceId));
    }
};

std::optional<BoundTokens> throwingGraphMatcher(const MatchContext& /*context*/)
{
    throw std::runtime_error("a matcher threw while deciding applicability");
}

// isApplicable is called in a loop over every engine, so a throw here would deny the
// caller the engines that would have answered. Both legs of the body can throw.
TEST(TestIngestorGenericPlanBuilder, IsApplicableDeclinesWhenTheDeviceResolverThrows)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const ThrowingDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x92));

    EXPECT_FALSE(builder.isApplicable(0, graph));
}

TEST(TestIngestorGenericPlanBuilder, IsApplicableDeclinesWhenAMatcherThrows)
{
    const ScopedSymbols symbols(
        "test.graph", throwingGraphMatcher, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x93));

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
    EXPECT_EQ(builder.getMaxWorkspaceSize(0, graph, KnobFilterSettings{}), 256U);
}

TEST(TestIngestorGenericPlanBuilder, BuildPlanThrowsInternalErrorOnAnEmptyRankedCatalog)
{
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
    EXPECT_EQ(knob.default_value.AsIntValue()->value, 256);
}

TEST(TestIngestorGenericPlanBuilder, GetCustomKnobsSkipsFieldsWithNoIntegerValues)
{
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

TEST(TestIngestorGenericPlanBuilder, HonorsAnExplicitKnobSettingOverTheHeuristicDefault)
{
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

    // The sequence GenericEngine::initializeExecutionContext runs: populate the settings
    // from the config, store them on the context, then build. buildPlan reads the filter
    // from the context, so a test that skips the middle step is exercising a state the
    // engine never produces.
    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
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

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 999);

    const TestGraph graph(makeGraphId(31));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    context.setExecutionSettings(settings);

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
        EXPECT_NE(ex.getMessage().find("2 kernel(s) matched the graph before knob filtering"),
                  std::string::npos);
    }
}

TEST(TestIngestorGenericPlanBuilder, GetMaxWorkspaceSizeHonorsTheSameKnobFilterAsBuildPlan)
{
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

constexpr DeviceId DEVICE_FOR_HANDLE_A = 42;
constexpr DeviceId DEVICE_FOR_HANDLE_B = 7;
constexpr const char* DEVICE_GATED_MATCH_SYMBOL
    = "hipdnn.kernel_ingestor.test.generic_plan_builder.device_gated";

std::optional<BoundTokens> acceptsOnlyDeviceA(const MatchContext& context)
{
    if(context.deviceId != DEVICE_FOR_HANDLE_A)
    {
        return std::nullopt;
    }
    return BoundTokens{};
}

TEST(TestIngestorGenericPlanBuilder, ContextForFoldsPerHandleDeviceResolutionIntoTheMatchContext)
{
    GraphMatchRegistry::registerSymbol(DEVICE_GATED_MATCH_SYMBOL, &acceptsOnlyDeviceA);
    const ScopedBlockSizeScore scorer;

    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};

    const KernelIngestorStateManager<StubHandle> manager(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        DEVICE_GATED_MATCH_SYMBOL);

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

    EXPECT_TRUE(builder.isApplicable(handleA, graph));
    EXPECT_FALSE(builder.isApplicable(handleB, graph));

    GraphMatchRegistry::unregisterSymbol(DEVICE_GATED_MATCH_SYMBOL);
}

/// A graph schema floor an engine declaring the baseline cannot serve.
const hipdnn_data_sdk::utilities::Version NEWER_THAN_BASELINE{
    hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION};
const hipdnn_data_sdk::utilities::Version BASELINE{
    hipdnn_plugin_sdk::K_ENGINE_PLUGIN_API_VERSION_BASELINE};

TEST(TestIngestorGenericPlanBuilder, EngineDecliningTheGraphsSchemaNeverMatches)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE}, BASELINE);
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xA0), NEWER_THAN_BASELINE);

    EXPECT_FALSE(builder.isApplicable(0, graph));
    EXPECT_EQ(counters().graphMatchCalls, 0);
    EXPECT_EQ(counters().kernelCalls, 0);
}

TEST(TestIngestorGenericPlanBuilder, EngineDeclaringTheGraphsSchemaMatchesNormally)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE}, NEWER_THAN_BASELINE);
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xA1), NEWER_THAN_BASELINE);

    EXPECT_TRUE(builder.isApplicable(0, graph));
    EXPECT_EQ(counters().graphMatchCalls, 1);
}

TEST(TestIngestorGenericPlanBuilder, EngineNewerThanTheGraphNeedsMatchesNormally)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE}, NEWER_THAN_BASELINE);
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xA2), BASELINE);

    EXPECT_TRUE(builder.isApplicable(0, graph));
}

TEST(TestIngestorGenericPlanBuilder, AnUnstampedGraphReadsAsTheBaselineAndMatches)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xA3));

    EXPECT_TRUE(builder.isApplicable(0, graph));
}

// ---------------------------------------------------------------------------
// The benchmarking knob composes with the knob filter and buildPlan()
// ---------------------------------------------------------------------------

/// Every case below asserts what the knob alone decides, so a runner carrying a stray
/// HIPDNN_FORCE_BENCHMARKING must not be able to flip the result. The guard clears the
/// variable for the case's duration and restores whatever was there.
class TestIngestorGenericPlanBuilderBenchmarking : public ::testing::Test
{
private:
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _forceBenchmarkingGuard{
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME};
};

TEST_F(TestIngestorGenericPlanBuilderBenchmarking,
       BenchmarkingKnobSetToOneLeavesTheFilterEmptyWhileEnablingTheFlag)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    const TestGraph graph(makeGraphId(0xB0));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    // Not a metadata field: it must never narrow the catalog through readKnobFilter.
    EXPECT_TRUE(settings.ingestorSettings.knobFilter.empty());
    EXPECT_TRUE(settings.ingestorSettings.benchmarkingEnabled);
}

TEST_F(TestIngestorGenericPlanBuilderBenchmarking, BenchmarkingKnobSetToZeroLeavesTheFlagFalse)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 0);
    const TestGraph graph(makeGraphId(0xB1));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_TRUE(settings.ingestorSettings.knobFilter.empty());
    EXPECT_FALSE(settings.ingestorSettings.benchmarkingEnabled);
}

TEST_F(TestIngestorGenericPlanBuilderBenchmarking,
       NonIntBenchmarkingKnobSettingThrowsInvalidValueNamingTheKnob)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeStringKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, "fast");
    const TestGraph graph(makeGraphId(0xB2));
    KnobFilterSettings settings;

    try
    {
        builder.initializeExecutionSettings(0, graph, engineConfig, settings);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(ex.getMessage().find(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME),
                  std::string::npos);
    }
}

TEST_F(TestIngestorGenericPlanBuilderBenchmarking, AnInvalidEngineConfigLeavesBenchmarkingFalse)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);
    const TestGraph graph(makeGraphId(0xB3));
    KnobFilterSettings settings;

    builder.initializeExecutionSettings(0, graph, invalidConfig, settings);

    EXPECT_FALSE(settings.ingestorSettings.benchmarkingEnabled);
}

/// Three kernels, no matchers, so all three survive to become benchmarking candidates.
/// ScopedConstantScore ties every kernel at 1.0, so rank() falls through to priority
/// and kernel_64 becomes the ranked front. The catalog's workspace max (256, from
/// kernel_256) stays strictly larger than the front's own (64), which is what makes
/// "front alone" and "max across all candidates" distinguishable.
std::unique_ptr<KernelIngestorStateManager<TestHandle>> makeThreeKernelWorkspaceStateManager()
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeKernel(testId(0x70), "kernel_64", 64, "FLOAT", /*priority=*/10),
                    makeKernel(testId(0x71), "kernel_128", 128, "FLOAT", /*priority=*/0),
                    makeKernel(testId(0x72), "kernel_256", 256, "FLOAT", /*priority=*/0)};

    ensureNoopDispatchRegistered<TestHandle>("test.dispatch");
    return std::make_unique<KernelIngestorStateManager<TestHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        std::vector<DispatchDescriptor>{{DISPATCH_ID, "test dispatch", "test.dispatch"}},
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(CONSTANT_SCORE_SYMBOL),
        "test.graph");
}

/// KnobFilterContext narrows its stored plan to GenericPlan, which the benchmarking-on
/// case cannot: buildPlan() hands it a BenchmarkPlan. This one keeps the IPlan.
struct BenchmarkContext
{
    void setExecutionSettings(const KnobFilterSettings& settings)
    {
        _settings = settings;
    }

    const KnobFilterSettings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
    }

    const hipdnn_plugin_sdk::IPlan<TestHandle>& plan() const
    {
        return *_plan;
    }

private:
    KnobFilterSettings _settings;
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> _plan;
};

using BenchmarkPlanBuilder = GenericPlanBuilder<TestHandle, KnobFilterSettings, BenchmarkContext>;

/// With benchmarking on, the plan the context receives reports the workspace max over
/// all three knob-filtered candidates (256, kernel_256's), not the ranked front's own
/// 64.
TEST_F(TestIngestorGenericPlanBuilderBenchmarking,
       BuildPlanWithBenchmarkingOnSizesForTheMaxAcrossAllCandidates)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    const TestGraph graph(makeGraphId(0xB4));
    const TestHandle handle;

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    EXPECT_EQ(context.plan().getWorkspaceSize(handle), 256U);
}

/// With benchmarking unset, buildPlan() takes the single-plan branch: the plan is sized
/// for the ranked front alone (64), not the catalog max (256) the case above reports
/// for the same catalog and knob filter.
TEST_F(TestIngestorGenericPlanBuilderBenchmarking,
       BuildPlanWithBenchmarkingOffSizesForTheRankedFrontAlone)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xB5));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().getWorkspaceSize(0), 64U);
    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64);
}

// ---------------------------------------------------------------------------
// HIPDNN_FORCE_BENCHMARKING composes as value_or, never an OR
// ---------------------------------------------------------------------------

/// Unset and never touched vs. unset via ScopedEnvironmentVariableSetter's one-arg
/// clearing form must produce identical IngestorSettings, independent of how the test
/// harness represents "unset".
TEST(TestIngestorGenericPlanBuilderOverride, UnsetOverrideWithNoKnobChangesNothing)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME);
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xC0));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_FALSE(settings.ingestorSettings.benchmarkingEnabled);
    EXPECT_TRUE(settings.ingestorSettings.knobFilter.empty());
}

/// With the knob set to 1, an unset override must not change the outcome.
TEST(TestIngestorGenericPlanBuilderOverride, UnsetOverrideWithKnobSetToOneStillWins)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME);
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    const TestGraph graph(makeGraphId(0xC1));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_TRUE(settings.ingestorSettings.benchmarkingEnabled);
}

/// Forces on with no knob at all: the plain-execute path with no autotune.
TEST(TestIngestorGenericPlanBuilderOverride, OverrideOnForcesBenchmarkingWithNoKnobSet)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "1");
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xC2));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_TRUE(settings.ingestorSettings.benchmarkingEnabled);
}

/// Forces on even with an invalid IEngineConfig: the override is consulted outside the
/// config's early return, and an invalid config is what a plain hipdnnExecute presents.
TEST(TestIngestorGenericPlanBuilderOverride, OverrideOnForcesBenchmarkingWithAnInvalidConfig)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "true");
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);
    const TestGraph graph(makeGraphId(0xC3));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, invalidConfig, settings);

    EXPECT_TRUE(settings.ingestorSettings.benchmarkingEnabled);
}

/// `0` forces off even when the knob asked for on, which an `||` composition could not
/// express: a false override term never clears a true knob term.
TEST(TestIngestorGenericPlanBuilderOverride, OverrideOffForcesBenchmarkingFalseEvenWithKnobSetToOne)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "0");
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    const TestGraph graph(makeGraphId(0xC4));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_FALSE(settings.ingestorSettings.benchmarkingEnabled);
}

/// An unrecognized value degrades to unset, not to on -- a typo must never silently
/// turn benchmarking on.
TEST(TestIngestorGenericPlanBuilderOverride, UnrecognizedOverrideValueBehavesAsUnset)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "onn");
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xC5));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_FALSE(settings.ingestorSettings.benchmarkingEnabled);
}

/// The override never populates the knob filter -- it is a plain on/off, not a
/// metadata-narrowing setting, exactly like the knob it overrides.
TEST(TestIngestorGenericPlanBuilderOverride, OverrideNeverPopulatesTheKnobFilter)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "1");
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, BLOCK_SIZE, 64);
    const TestGraph graph(makeGraphId(0xC6));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    EXPECT_TRUE(settings.ingestorSettings.benchmarkingEnabled);
    EXPECT_EQ(settings.ingestorSettings.knobFilter.size(), 1U);
    EXPECT_EQ(settings.ingestorSettings.knobFilter.count(BLOCK_SIZE), 1U);
    EXPECT_EQ(settings.ingestorSettings.knobFilter.count(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME),
              0U);
}

/// Read per call, never cached: flipping the variable between two
/// initializeExecutionSettings() calls on the same builder must be reflected in each.
TEST(TestIngestorGenericPlanBuilderOverride, TheOverrideIsReReadOnEveryCallNotCached)
{
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "1");
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xC7));

    KnobFilterSettings firstCall;
    builder.initializeExecutionSettings(0, graph, engineConfig, firstCall);
    EXPECT_TRUE(firstCall.ingestorSettings.benchmarkingEnabled);

    guard.setValue("0");

    KnobFilterSettings secondCall;
    builder.initializeExecutionSettings(0, graph, engineConfig, secondCall);
    EXPECT_FALSE(secondCall.ingestorSettings.benchmarkingEnabled);
}

// ---------------------------------------------------------------------------
// Exhaustive vocabulary parsing over the full accepted set
// ---------------------------------------------------------------------------

struct OverrideVariantCase
{
    std::string envValue;
    std::optional<bool> expected;
};

class TestIngestorBenchmarkingOverrideVariants
    : public ::testing::TestWithParam<OverrideVariantCase>
{
};

TEST_P(TestIngestorBenchmarkingOverrideVariants, ParsesAsExpected)
{
    const auto& testCase = GetParam();
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, testCase.envValue);

    EXPECT_EQ(hipdnn_plugin_sdk::benchmarkingOverrideFromEnv(), testCase.expected);
}

std::string mixedCase(const std::string& spelling)
{
    // Alternates case per character (e.g. "enabled" -> "EnAbLeD"), a deterministic
    // "mixed" transform distinct from all-lower and all-upper.
    std::string mixed = spelling;
    for(size_t index = 0; index < mixed.size(); ++index)
    {
        mixed[index] = (index % 2 == 0) ? static_cast<char>(std::toupper(mixed[index]))
                                        : static_cast<char>(std::tolower(mixed[index]));
    }
    return mixed;
}

std::string allUpper(const std::string& spelling)
{
    std::string upper = spelling;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });
    return upper;
}

std::vector<OverrideVariantCase> everyCasingOf(const std::string& spelling, bool expected)
{
    return {{spelling, expected},
            {allUpper(spelling), expected},
            {mixedCase(spelling), expected},
            {" " + spelling + " ", expected}};
}

std::vector<OverrideVariantCase> allAcceptedVariantCases()
{
    std::vector<OverrideVariantCase> cases;
    for(const auto& trueSpelling : {std::string("1"),
                                    std::string("true"),
                                    std::string("on"),
                                    std::string("yes"),
                                    std::string("enable"),
                                    std::string("enabled")})
    {
        for(auto& variant : everyCasingOf(trueSpelling, true))
        {
            cases.push_back(std::move(variant));
        }
    }
    for(const auto& falseSpelling : {std::string("0"),
                                     std::string("false"),
                                     std::string("off"),
                                     std::string("no"),
                                     std::string("disable"),
                                     std::string("disabled")})
    {
        for(auto& variant : everyCasingOf(falseSpelling, false))
        {
            cases.push_back(std::move(variant));
        }
    }
    // Near-misses that must resolve to nullopt, never to on.
    for(const auto& nearMiss : {std::string("2"),
                                std::string("-1"),
                                std::string("onn"),
                                std::string("tru"),
                                std::string("y"),
                                std::string("n"),
                                std::string(""),
                                std::string("   ")})
    {
        cases.push_back({nearMiss, std::nullopt});
    }
    return cases;
}

std::string variantCaseName(const ::testing::TestParamInfo<OverrideVariantCase>& info)
{
    std::string name = "Case" + std::to_string(info.index) + "_";
    for(const char character : info.param.envValue)
    {
        name += std::isalnum(static_cast<unsigned char>(character)) != 0 ? std::string(1, character)
                                                                         : std::string("_");
    }
    if(name.back() == '_' && info.param.envValue.empty())
    {
        name += "Empty";
    }
    return name;
}

INSTANTIATE_TEST_SUITE_P(EveryAcceptedSpellingAndNearMiss,
                         TestIngestorBenchmarkingOverrideVariants,
                         ::testing::ValuesIn(allAcceptedVariantCases()),
                         variantCaseName);

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
