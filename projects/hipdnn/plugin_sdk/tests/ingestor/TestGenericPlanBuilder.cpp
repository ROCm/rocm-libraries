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
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
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

#include "ContentCarryingTestGraph.hpp"
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
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey;
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

/// prepare() returns nullptr for one specific block size, modeling a kernel whose
/// launch state cannot be resolved this run; every other kernel prepares normally.
/// Shares WorkspaceEqualsBlockSizeHandler's workspace-bytes contract so the served
/// kernel is identifiable by workspace size alone.
class NullPrepareForBlockSizeHandler : public IKernelDispatchHandler<TestHandle>
{
public:
    explicit NullPrepareForBlockSizeHandler(int64_t blockSizeToFail)
        : _blockSizeToFail(blockSizeToFail)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return static_cast<size_t>(kernel.getIntMetadata(BLOCK_SIZE));
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        if(kernel.getIntMetadata(BLOCK_SIZE) == _blockSizeToFail)
        {
            return nullptr;
        }
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const TestHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }

private:
    int64_t _blockSizeToFail;
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

// contextFor's device-arch guard rejects a device with an empty gcnArchName;
// isApplicable's existing catch-all must turn that throw into a decline with a logged
// error.
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

    // The status, not the type: an unresolved device arch reports INTERNAL_ERROR.
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

// The device-arch guard also gates getMaxWorkspaceSize, which reaches contextFor before
// either catalog lookup.
TEST(TestIngestorGenericPlanBuilder,
     GetMaxWorkspaceSizeThrowsInternalErrorWhenTheDeviceArchIsUnresolved)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const UnresolvedArchDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x99));

    try
    {
        builder.getMaxWorkspaceSize(0, graph, KnobFilterSettings{});
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
}

// The device-arch guard also gates getCustomKnobs, which reaches contextFor before
// sortedDefinitions.
TEST(TestIngestorGenericPlanBuilder, GetCustomKnobsThrowsInternalErrorWhenTheDeviceArchIsUnresolved)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const UnresolvedArchDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x9D));

    try
    {
        builder.getCustomKnobs(0, graph);
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

/// Deterministic sample times keyed by workspace size rather than candidate identity:
/// the constructor injects this Timer before `buildPlan` constructs any candidate, so
/// the closure cannot capture their pointers in advance the way an override on the
/// constructed candidate list could. WorkspaceEqualsBlockSizeHandler makes workspace
/// size (== block_size) a stable, distinguishing proxy for which kernel is which.
inline BenchmarkPlan<TestHandle>::Timer
    makeWorkspaceKeyedTimer(std::unordered_map<size_t, double> timesByWorkspaceSize)
{
    return
        [times = std::move(timesByWorkspaceSize)](const hipdnn_plugin_sdk::IPlan<TestHandle>& plan,
                                                  const TestHandle& handle,
                                                  const hipdnnPluginDeviceBuffer_t*,
                                                  uint32_t,
                                                  void*) -> std::optional<double> {
            const auto found = times.find(plan.getWorkspaceSize(handle));
            return found != times.end() ? std::optional<double>(found->second) : std::nullopt;
        };
}

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

/// prepare() fails for one kernel and succeeds for the rest: the shape of a code object
/// that cannot be loaded -- a kpack archive missing from the install, a symbol the module
/// does not export -- which is a property of that one kernel, not of its pack.
class FailsToPrepareOneBlockSizeHandler : public IKernelDispatchHandler<TestHandle>
{
public:
    explicit FailsToPrepareOneBlockSizeHandler(int64_t failingBlockSize)
        : _failingBlockSize(failingBlockSize)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        return static_cast<size_t>(kernel.getIntMetadata(BLOCK_SIZE));
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        if(kernel.getIntMetadata(BLOCK_SIZE) == _failingBlockSize)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "the archive holds no code object for this kernel");
        }
        return std::make_unique<PreparedDispatch>();
    }

    void launch(const TestHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }

private:
    int64_t _failingBlockSize;
};

/// Constructing a GenericPlan runs prepare(), so the ranked front is where an unloadable
/// code object surfaces -- after applicability already promised the graph. kernel_64 fails,
/// kernel_128 takes the plan, and the WARN is the only record that a slower kernel is
/// running.
TEST(TestIngestorGenericPlanBuilder, FallsBackWhenTheFrontCandidateCannotPrepare)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const FailsToPrepareOneBlockSizeHandler handler(64);
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xB6));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    // 128, not 64: priority puts kernel_64 at the front, and the next candidate served.
    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 128);
    EXPECT_EQ(context.plan().getWorkspaceSize(0), 128U);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, toString(testId(0x70))))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN,
                                          "the archive holds no code object for this "
                                          "kernel"));
}

// ---------------------------------------------------------------------------
// The winner-cache coverage gate and ranked walk at the buildPlan() lookup site
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
    const auto manager = makeThreeKernelWorkspaceStateManager();
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

/// The catalog-sort coverage rule governs every `sortedCatalog` consumer, including
/// `getCustomKnobs`, whose `choices.front()` becomes the knob's advertised default.
/// Same rig as `TestWinnerCache.cpp`'s `ACoveringRecordOrdersTheCatalogWithoutTheHeuristic`:
/// record the heuristic's order reversed, covering the WHOLE catalog, and assert the
/// advertised default is the measured front, not the heuristic's own front (kernel_64,
/// via ScopedConstantScore's tie plus priority).
TEST(TestIngestorGenericPlanBuilder, GetCustomKnobsAdvertisesTheMeasuredDefaultUnderACoveringRecord)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xDA));
    const auto properties = testDeviceProperties();

    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    ASSERT_EQ(catalog.front().getIntMetadata(BLOCK_SIZE), 64)
        << "this test needs the heuristic front to differ from the record's";

    WinnerRecord record;
    double time = 1.0;
    for(auto kernel = catalog.rbegin(); kernel != catalog.rend(); ++kernel)
    {
        record.push_back(rankedEntryFor(*kernel, time));
        time += 1.0;
    }
    manager->recordWinner(winnerKeyFor(graph, properties), record);

    const auto knobs = builder.getCustomKnobs(0, graph);

    ASSERT_EQ(knobs.size(), 1U);
    const auto& knob = knobs.front();
    EXPECT_EQ(knob.knob_id, BLOCK_SIZE);
    ASSERT_TRUE(knob.default_value.AsIntValue() != nullptr);
    EXPECT_EQ(knob.default_value.AsIntValue()->value, 256)
        << "the advertised default must be the measured front (kernel_256, the record's "
           "first entry), not the heuristic front (kernel_64)";
}

/// Benchmark wide, then run narrow. The record is wider than the catalog -- it carries
/// an extra entry for a kernel this engine does not admit -- while still covering all
/// three live candidates, so it must be served (kernel_256), not the heuristic front
/// (kernel_64).
TEST(TestIngestorGenericPlanBuilder, ARecordWiderThanTheFilteredSetIsStillServed)
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

/// A partial record with benchmarking OFF must fall back to the heuristic front rather
/// than serve the entry it covers: those entries were measured only against each
/// other, never against the candidates the record excludes, so a partial record can
/// only ever reorder a fully-measured set, not override the heuristic with an
/// unraced kernel.
TEST(TestIngestorGenericPlanBuilder, APartialRecordWithBenchmarkingOffFallsBackToTheHeuristic)
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

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64)
        << "256 would mean an unraced measurement overrode the heuristic's own pick";
}

/// A record whose entries no longer resolve is not an error: selection falls back to the
/// heuristic front rather than serving a kernel the record no longer describes.
TEST(TestIngestorGenericPlanBuilder, AWhollyStaleRecordFallsBackToNormalSelection)
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

/// Regression for the bug where the lookup-site coverage test (`recordCovers`, by
/// `kernelId` alone) and its ordering (`orderByRecord`, by the full `(kernelId, packId,
/// dispatchId)` triple) could disagree: a record covering every candidate by
/// `kernelId`, but with one entry's `packId` stale, must trigger a re-benchmark
/// (workspace sizes for the max across all three candidates) rather than serve the
/// subset that still resolves. Benchmarking ON, unlike the sibling wholly-stale test,
/// since a stale-with-benchmarking-off record already falls back correctly.
TEST(TestIngestorGenericPlanBuilder, APartiallyStaleRecordWithBenchmarkingOnTriggersReBenchmarking)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xDB));
    const auto properties = testDeviceProperties();
    const TestHandle handle;

    // Every kernelId is covered, but kernel_128's packId no longer agrees with the
    // catalog's current one.
    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    WinnerRecord record;
    for(const auto& kernel : catalog)
    {
        auto entry = rankedEntryFor(kernel, kernel.getIntMetadata(BLOCK_SIZE) == 128 ? 0.1 : 9.0);
        if(kernel.getIntMetadata(BLOCK_SIZE) == 128)
        {
            entry.packId = testId(0xEE);
        }
        record.push_back(entry);
    }
    std::stable_sort(record.begin(), record.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timeMs < rhs.timeMs;
    });
    manager->recordWinner(winnerKeyFor(graph, properties), record);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    EXPECT_EQ(context.plan().getWorkspaceSize(handle), 256U)
        << "covered by kernelId with one stale packId must decline the whole record and "
           "re-benchmark, not serve the ranked subset that still resolves by identity "
           "alone";
}

/// Regression for the ranked walk existing to be skipped: constructing a GenericPlan
/// throws when the dispatch handler's prepare() returns nullptr (GenericPlan.hpp), so a
/// covering record whose rank-0 kernel can no longer be prepared must fall through to
/// rank 1 rather than propagate that throw -- a warm cache must not be stricter than an
/// empty one.
TEST(TestIngestorGenericPlanBuilder, ARecordWhoseRankZeroKernelFailsToPrepareFallsThroughToRankOne)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const NullPrepareForBlockSizeHandler handler(/*blockSizeToFail=*/64);
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xDC));
    const auto properties = testDeviceProperties();

    // A fully-covering record ranking kernel_64 (which cannot prepare) first and
    // kernel_128 second.
    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    // Ranks kernel_64 (which cannot prepare) first, then kernel_128, then kernel_256.
    const std::map<int64_t, double> timeByBlockSize{{64, 0.1}, {128, 0.2}, {256, 0.3}};
    WinnerRecord record;
    for(const auto& kernel : catalog)
    {
        record.push_back(
            rankedEntryFor(kernel, timeByBlockSize.at(kernel.getIntMetadata(BLOCK_SIZE))));
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

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 128)
        << "rank 0 (kernel_64) cannot prepare; the plan must build from rank 1 instead "
           "of throwing";
}

/// A record keyed on a different device must never be served here. This is why the key
/// folds the whole DeviceProperties struct rather than the arch string alone.
TEST(TestIngestorGenericPlanBuilder, ARecordForAnotherDeviceIsNotServed)
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

/// The graph-half counterpart of the device test above. `TestGraph` carries no content,
/// so every instance keys alike and cannot express this; the content-carrying fixture
/// makes the two graphs genuinely different computations.
TEST(TestIngestorGenericPlanBuilder, ARecordForAnotherGraphIsNotServed)
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
    const auto properties = testDeviceProperties();

    using Spec = ContentCarryingTestGraph::Spec;
    const ContentCarryingTestGraph graph{Spec{}};
    Spec wider;
    wider.tensors[0].dims = {4, 16};
    const ContentCarryingTestGraph otherGraph{wider};

    WinnerRecord record;
    for(const auto& kernel : catalogFor(*manager, graph, properties))
    {
        if(kernel.getIntMetadata(BLOCK_SIZE) == 256)
        {
            record.push_back(rankedEntryFor(kernel, 0.1));
        }
    }
    manager->recordWinner(winnerKeyFor(otherGraph, properties), record);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64)
        << "a record measured for another graph must not decide this one's kernel";
}

/// The narrow-then-wide half: a record written under a narrow knob filter does NOT
/// cover a later unfiltered run, so that run must re-benchmark rather than serve the
/// best of a subset it never fully measured. Observable without timing: benchmarking
/// sizes for the max over all candidates (256), a served hit sizes for the one kernel
/// chosen (128 or less).
TEST(TestIngestorGenericPlanBuilder, ANarrowRecordDoesNotCoverAWiderRunAndTriggersReBenchmarking)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xD6));
    const auto properties = testDeviceProperties();
    const TestHandle handle;

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

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
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
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xD7));
    const auto properties = testDeviceProperties();
    const TestHandle handle;

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

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    // Workspace is the whole discriminator here: BenchmarkContext exposes only IPlan, which
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
// The real write-back path, end to end
// ---------------------------------------------------------------------------

/// Times descending by block size (== workspace size in this fixture, unique per
/// kernel): kernel_256 -- the LAST of the three in catalog order, the opposite of the
/// heuristic front -- samples fastest, making a served record observable. Injected
/// through the constructor's Timer parameter, so `buildPlan` runs the real write-back
/// factory rather than a test double that re-implements it.
inline BenchmarkPlan<TestHandle>::Timer makeThreeKernelDescendingTimer()
{
    return makeWorkspaceKeyedTimer({{64, 3.0}, {128, 2.0}, {256, 1.0}});
}

TEST(TestIngestorGenericPlanBuilder, SamplingWritesTheRankingBackThroughTheBuildersOwnCallback)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(
        engine, *manager, resolver, makeThreeKernelDescendingTimer());

    const TestGraph graph(makeGraphId(0xD8));
    const auto properties = testDeviceProperties();
    const TestHandle handle;

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
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

/// A ranking sampled under one numbering must be served for the same graph numbered
/// differently. Key-level equality is pinned in TestGraphContentKey.cpp; this adds that
/// the record `recordWinner` wrote under the first key is found by `winnerFor` under the
/// second, and orders the catalog.
TEST(TestIngestorGenericPlanBuilder, ARecordSampledUnderOneNumberingIsServedForAnother)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(
        engine, *manager, resolver, makeThreeKernelDescendingTimer());

    using Spec = ContentCarryingTestGraph::Spec;
    Spec low;
    low.tensors
        = {ContentCarryingTestGraph::TensorSpec{1}, ContentCarryingTestGraph::TensorSpec{2}};
    low.nodes[0].in0TensorUid = 1;
    low.nodes[0].out0TensorUid = 2;

    Spec high;
    high.tensors
        = {ContentCarryingTestGraph::TensorSpec{1000}, ContentCarryingTestGraph::TensorSpec{2000}};
    high.nodes[0].in0TensorUid = 1000;
    high.nodes[0].out0TensorUid = 2000;

    const ContentCarryingTestGraph sampled{low};
    const ContentCarryingTestGraph renumbered{high};
    const auto properties = testDeviceProperties();
    const TestHandle handle;

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, sampled, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext sampledContext;
    sampledContext.setExecutionSettings(settings);
    builder.buildPlan(handle, sampled, engineConfig, sampledContext);
    std::vector<std::byte> workspace(sampledContext.plan().getWorkspaceSize(handle));
    sampledContext.plan().execute(handle, nullptr, 0U, workspace.data());

    ASSERT_EQ(manager->winnerCacheSize(), 1U) << "sampling must have recorded one ranking";
    const auto stored = manager->winnerFor(winnerKeyFor(renumbered, properties));
    ASSERT_TRUE(stored.has_value())
        << "the renumbered graph must find the record the sampled graph wrote";
    EXPECT_EQ(stored->front().kernelId, testId(0x72))
        << "and it must be the sampled ranking, not the heuristic order";

    // The served path: the record orders the catalog for the renumbered graph, so its
    // plan is built from the measured front rather than re-sampled.
    const auto ordered = manager->sortedDefinitions(MatchContext{renumbered, 0, properties});
    ASSERT_FALSE(ordered.empty());
    EXPECT_EQ(ordered.front().kernelId, testId(0x72))
        << "the renumbered graph must be ordered by the record it hit";
}

/// The write-back side of the narrow-then-wide re-benchmark
/// (`ANarrowRecordDoesNotCoverAWiderRunAndTriggersReBenchmarking` proves the
/// re-benchmark itself but never inspects the record afterward): the wide run must not
/// just re-benchmark, it must write the superset ranking back, so the next run is fully
/// covered.
TEST(TestIngestorGenericPlanBuilder,
     ANarrowRecordThatTriggersReBenchmarkingWritesBackASupersetRecord)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(
        engine, *manager, resolver, makeThreeKernelDescendingTimer());

    const TestGraph graph(makeGraphId(0xD9));
    const auto properties = testDeviceProperties();
    const TestHandle handle;

    // A prior narrow run measured ONLY kernel_128, exactly as the sibling narrow test.
    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
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

    // Now a WIDE run, unfiltered, with benchmarking on, so sampling produces real
    // usable candidates and write-back actually fires.
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, engineConfig, context);

    std::vector<std::byte> workspace(context.plan().getWorkspaceSize(handle));
    context.plan().execute(handle, nullptr, 0U, workspace.data());

    const auto stored = manager->winnerFor(winnerKeyFor(graph, properties));
    ASSERT_TRUE(stored.has_value())
        << "the wide re-benchmark must write its ranking back, not merely discard it";
    EXPECT_TRUE(recordCovers(*stored, catalog))
        << "the post-run record must cover every kernel in the wide filtered set, not "
           "just kernel_128 the narrow record held";
    EXPECT_EQ(stored->size(), 3U)
        << "the superset write-back must carry all three benchmarked candidates";
}

} // namespace
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
