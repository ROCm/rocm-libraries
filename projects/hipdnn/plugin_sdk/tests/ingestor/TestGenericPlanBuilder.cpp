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
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
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

/// Reports a resolved device (deviceId != NO_DEVICE) whose properties are unresolvable
/// by arch, distinct from ThrowingDeviceResolver's outright query failure above.
class UnresolvedArchDeviceResolver : public IDeviceResolver<TestHandle>
{
public:
    DeviceId deviceId(const TestHandle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties;
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

// contextFor's D10 guard rejects a device with an empty gcnArchName; isApplicable's
// existing catch-all must turn that throw into a decline with a logged error.
TEST(TestIngestorGenericPlanBuilder, IsApplicableDeclinesWhenTheDeviceArchIsUnresolved)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const UnresolvedArchDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x94));

    EXPECT_FALSE(builder.isApplicable(0, graph));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, engine.name))
        << recorder.getRecordedLogsAsString();
}

// The other side of the same guard: a caller that needs a plan gets a loud failure,
// not a plan built for a device nobody identified.
TEST(TestIngestorGenericPlanBuilder, BuildPlanThrowsInternalErrorWhenTheDeviceArchIsUnresolved)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const UnresolvedArchDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0x95));
    KnobFilterContext context;

    // The status, not merely the type: D10 specifies INTERNAL_ERROR.
    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
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

/// Supplies deterministic sample times through BenchmarkPlan's D11 seam, mirroring
/// TestBenchmarkPlan.cpp's subclass on the stream-capable handle the builder tests use.
class DeterministicStreamBenchmarkPlan : public BenchmarkPlan<StreamCapableHandle>
{
public:
    DeterministicStreamBenchmarkPlan(std::vector<Candidate> candidates,
                                     const StreamCapableHandle& handle,
                                     std::vector<std::optional<double>> times,
                                     RecordRankingFn recordRanking)
        : BenchmarkPlan<StreamCapableHandle>(
              std::move(candidates), handle, std::move(recordRanking))
        , _times(std::move(times))
    {
    }

protected:
    std::optional<double> sampleCandidate(size_t index,
                                          const StreamCapableHandle& /*handle*/,
                                          const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                                          uint32_t /*numDeviceBuffers*/,
                                          void* /*workspace*/) const override
    {
        return index < _times.size() ? _times[index] : std::nullopt;
    }

private:
    std::vector<std::optional<double>> _times;
};

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
// Check 2: the winner-cache coverage gate and ranked walk at the lookup site
// ---------------------------------------------------------------------------

/// Workspace size is the observable proxy for "which kernel was selected": the heuristic
/// ties every score and priority puts kernel_64 at the front, so a record naming
/// kernel_256 proves the cache decided, with no timing involved.
WinnerKey winnerKeyFor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                       const DeviceProperties& properties)
{
    return WinnerKey{GraphContentKey{graph}, DeviceKey{properties}};
}

RankedEntry rankedEntryFor(const KernelDefinition& kernel, double timeMs)
{
    return RankedEntry{kernel.kernelId, kernel.packId, kernel.dispatchId, timeMs};
}

/// Every kernel the manager admits for this graph, in catalog order.
template <typename THandle>
std::vector<KernelDefinition>
    catalogFor(const KernelIngestorStateManager<THandle>& manager,
               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
               const DeviceProperties& properties)
{
    return manager.sortedDefinitions(MatchContext{graph, 0, properties});
}

TEST(TestIngestorGenericPlanBuilder, ACoveringRecordServesItsRankedFrontWithoutBenchmarking)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<TestHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xD1));
    const auto properties = testDeviceProperties();

    // Rank kernel_256 first, the opposite of the heuristic's choice.
    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    WinnerRecord record;
    for(const auto& kernel : catalog)
    {
        const auto blockSize = kernel.getIntMetadata(BLOCK_SIZE);
        record.push_back(rankedEntryFor(kernel, blockSize == 256 ? 0.1 : 9.0));
    }
    std::stable_sort(record.begin(), record.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timeMs < rhs.timeMs;
    });
    manager->recordWinner(winnerKeyFor(graph, properties), record);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 256)
        << "the measured winner must beat the heuristic front";
}

/// D7's mirror, half one: benchmark wide, then run narrow. The record is wider than the
/// catalog -- it carries an extra entry for a kernel this engine does not admit -- while
/// still covering all three live candidates, so it must be served (kernel_256), not the
/// heuristic front (kernel_64).
TEST(TestIngestorGenericPlanBuilder, ARecordWiderThanTheFilteredSetIsStillServed)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<TestHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xD2));
    const auto properties = testDeviceProperties();

    WinnerRecord record;
    for(const auto& kernel : catalogFor(*manager, graph, properties))
    {
        record.push_back(
            rankedEntryFor(kernel, kernel.getIntMetadata(BLOCK_SIZE) == 256 ? 0.1 : 9.0));
    }
    ASSERT_EQ(record.size(), 3U);

    // A prior wider run's leftover entry for a kernel this engine does not admit. Must
    // be skipped without failing coverage.
    record.push_back(RankedEntry{testId(0xAA), testId(0xF0), testId(0xD0), 0.05});
    std::stable_sort(record.begin(), record.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timeMs < rhs.timeMs;
    });
    manager->recordWinner(winnerKeyFor(graph, properties), record);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 256)
        << "a record wider than the filtered set still covers it and must be served; "
           "64 would mean the cache was skipped and the heuristic front used";
}

/// D7's mirror, half two: a partial record with benchmarking OFF must serve the best
/// covered entry rather than decline -- those entries were genuinely measured.
TEST(TestIngestorGenericPlanBuilder, APartialRecordWithBenchmarkingOffStillServesWhatItCovers)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<TestHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xD3));
    const auto properties = testDeviceProperties();

    WinnerRecord record;
    for(const auto& kernel : catalogFor(*manager, graph, properties))
    {
        if(kernel.getIntMetadata(BLOCK_SIZE) == 256)
        {
            record.push_back(rankedEntryFor(kernel, 0.1));
        }
    }
    ASSERT_EQ(record.size(), 1U);
    manager->recordWinner(winnerKeyFor(graph, properties), record);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 256)
        << "a real measurement beats the heuristic guess even under partial coverage";
}

/// A record whose entries no longer resolve is not an error: selection falls back to the
/// heuristic front rather than serving a kernel the record no longer describes.
TEST(TestIngestorGenericPlanBuilder, AWhollyStaleRecordFallsBackToNormalSelection)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<TestHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xD4));
    const auto properties = testDeviceProperties();

    // Right kernel ids, wrong pack: every entry fails the staleness cross-check.
    WinnerRecord record;
    for(const auto& kernel : catalogFor(*manager, graph, properties))
    {
        auto entry = rankedEntryFor(kernel, 0.1);
        entry.packId = testId(0xEE);
        record.push_back(entry);
    }
    manager->recordWinner(winnerKeyFor(graph, properties), record);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64)
        << "a stale record must degrade to today's behaviour, never throw or serve blind";
}

/// A record keyed on a different device must never be served here. This is why the key
/// folds the whole DeviceProperties struct rather than the arch string alone.
TEST(TestIngestorGenericPlanBuilder, ARecordForAnotherDeviceIsNotServed)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<TestHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xD5));
    const auto properties = testDeviceProperties();

    auto otherDevice = properties;
    otherDevice.multiProcessorCount = properties.multiProcessorCount + 1;

    WinnerRecord record;
    for(const auto& kernel : catalogFor(*manager, graph, properties))
    {
        if(kernel.getIntMetadata(BLOCK_SIZE) == 256)
        {
            record.push_back(rankedEntryFor(kernel, 0.1));
        }
    }
    manager->recordWinner(winnerKeyFor(graph, otherDevice), record);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64)
        << "a record measured on another device must not decide this one's kernel";
}

/// D7's narrow-then-wide half: a record written under a narrow knob filter does NOT
/// cover a later unfiltered run, so that run must re-benchmark rather than serve the
/// best of a subset it never fully measured. Observable without timing: benchmarking
/// sizes for the max over all candidates (256), a served hit sizes for the one kernel
/// chosen (128 or less).
TEST(TestIngestorGenericPlanBuilder, ANarrowRecordDoesNotCoverAWiderRunAndTriggersReBenchmarking)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const StreamWorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<StreamCapableHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<StreamCapableHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const StreamDeviceResolver resolver;
    const StreamPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xD6));
    const auto properties = testDeviceProperties();
    const StreamCapableHandle handle;

    // A prior narrow run measured ONLY kernel_128.
    const auto catalog = manager->sortedDefinitions(MatchContext{graph, 0, properties});
    WinnerRecord narrow;
    for(const auto& kernel : catalog)
    {
        if(kernel.getIntMetadata(BLOCK_SIZE) == 128)
        {
            narrow.push_back(rankedEntryFor(kernel, 0.1));
        }
    }
    ASSERT_EQ(narrow.size(), 1U);
    manager->recordWinner(winnerKeyFor(graph, properties), narrow);

    // Now a WIDE run, unfiltered, with benchmarking on.
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    StreamSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    StreamContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    EXPECT_EQ(context.plan().getWorkspaceSize(handle), 256U)
        << "an uncovering record with benchmarking on must be ignored and the whole "
           "filtered set re-benchmarked, not served from the narrow subset";
}

/// Two buildPlan calls for the same graph and device: the first populates the cache by
/// benchmarking, the second is served from it with no BenchmarkPlan built -- the shape
/// an EXHAUSTIVE autotune() run takes, minus autotune itself.
TEST(TestIngestorGenericPlanBuilder, ASecondBuildPlanIsServedFromTheFirstRunsRanking)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const StreamWorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<StreamCapableHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<StreamCapableHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const StreamDeviceResolver resolver;
    const StreamPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xD7));
    const auto properties = testDeviceProperties();
    const StreamCapableHandle handle;

    // Stand in for the priming sweep's write-back: a ranking naming kernel_256, which
    // the heuristic (tied scores, priority order) would never choose.
    const auto catalog = manager->sortedDefinitions(MatchContext{graph, 0, properties});
    ASSERT_EQ(catalog.size(), 3U);
    // Rank kernel_128 first -- the MIDDLE workspace, so a served hit (128) and a
    // re-benchmark (256, the max) are distinguishable by workspace alone.
    WinnerRecord ranking;
    for(const auto& kernel : catalog)
    {
        ranking.push_back(
            rankedEntryFor(kernel, kernel.getIntMetadata(BLOCK_SIZE) == 128 ? 0.1 : 9.0));
    }
    std::stable_sort(ranking.begin(), ranking.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timeMs < rhs.timeMs;
    });
    manager->recordWinner(winnerKeyFor(graph, properties), ranking);

    // The post-priming plan: benchmarking still ON, exactly as autotune leaves it.
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    StreamSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    StreamContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    // Workspace is the whole discriminator here: StreamContext exposes only IPlan, which
    // has no kernel accessor, and 128-vs-256 already separates the two paths cleanly.
    EXPECT_EQ(context.plan().getWorkspaceSize(handle), 128U)
        << "a served hit sizes for the one chosen kernel; 256 would mean a BenchmarkPlan "
           "was built and the priming sweep's ranking was thrown away";
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
// The real write-back path, end to end (D1 as revised by D6)
// ---------------------------------------------------------------------------

/// Substitutes a deterministic sampling plan through the builder's own seam, so the
/// callback and key under test are the ones `buildPlan` actually captured. The real
/// sampler needs hipEvents, which a device-less runner never provides.
class DeterministicStreamPlanBuilder : public StreamPlanBuilder
{
public:
    using StreamPlanBuilder::StreamPlanBuilder;

protected:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<StreamCapableHandle>> makeBenchmarkPlan(
        std::vector<BenchmarkPlan<StreamCapableHandle>::Candidate> candidates,
        const StreamCapableHandle& handle,
        BenchmarkPlan<StreamCapableHandle>::RecordRankingFn recordRanking) const override
    {
        // Time descending by index, so the LAST candidate wins -- the opposite of the
        // heuristic front, making a served record observable.
        std::vector<std::optional<double>> times;
        times.reserve(candidates.size());
        for(size_t index = 0; index < candidates.size(); ++index)
        {
            times.emplace_back(static_cast<double>(candidates.size() - index));
        }
        return std::make_unique<DeterministicStreamBenchmarkPlan>(
            std::move(candidates), handle, std::move(times), std::move(recordRanking));
    }
};

TEST(TestIngestorGenericPlanBuilder, SamplingWritesTheRankingBackThroughTheBuildersOwnCallback)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const StreamWorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<StreamCapableHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager<StreamCapableHandle>();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const StreamDeviceResolver resolver;
    const DeterministicStreamPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xD8));
    const auto properties = testDeviceProperties();
    const StreamCapableHandle handle;

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    StreamSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    StreamContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    ASSERT_EQ(manager->winnerCacheSize(), 0U) << "nothing is recorded until execute() samples";

    std::vector<std::byte> workspace(context.plan().getWorkspaceSize(handle));
    context.plan().execute(handle, nullptr, 0U, workspace.data());

    // The key the builder computed internally must be the one the record landed under.
    const auto stored = manager->winnerFor(winnerKeyFor(graph, properties));
    ASSERT_TRUE(stored.has_value())
        << "the callback buildPlan captured must have written the ranking back";
    ASSERT_EQ(stored->size(), 3U) << "every usable candidate belongs in the record";
    // Rank 0 must be the LAST candidate in catalog order (the deterministic sampler's
    // fastest), the opposite of the heuristic front. Compared against ids directly, not
    // sortedDefinitions, which now returns the record's own order.
    EXPECT_EQ(stored->front().kernelId, testId(0x72))
        << "the fastest sampled candidate must rank first, not the heuristic front";
    EXPECT_EQ(stored->back().kernelId, testId(0x70)) << "and the slowest must rank last";
}

} // namespace
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
