// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
#include <hipdnn_plugin_sdk/OutputAgreement.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkValidation.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "IngestorMocks.hpp"
#include "KernelIngestorTestFixtures.hpp"
#include "flatbuffer_utilities/ContentCarryingTestGraph.hpp"

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
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::testing::ContentCarryingTestGraph;

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

/// Admits every kernel, so the catalog holds both dtypes.
inline bool acceptEveryKernel(const MatchContext& /*context*/,
                              const BoundTokens& /*bound*/,
                              const KernelDefinition& /*kernel*/)
{
    return true;
}

TEST(TestIngestorGenericPlanBuilder, GetCustomKnobsAdvertisesANonIntegerFieldAsAnOrdinal)
{
    // The two block_size=64 kernels differ only in dtype, so it must be advertised.
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", acceptEveryKernel);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE, DTYPE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0x95));
    const auto knobs = builder.getCustomKnobs(0, graph);

    ASSERT_EQ(knobs.size(), 2U);
    const auto dtype = std::find_if(
        knobs.begin(), knobs.end(), [](const auto& knob) { return knob.knob_id == DTYPE; });
    ASSERT_NE(dtype, knobs.end());
    ASSERT_TRUE(dtype->constraint.AsIntConstraint() != nullptr);
    // Ordinals number the engine's distinct values: 0 is "FLOAT", 1 is "HALF".
    auto advertised = dtype->constraint.AsIntConstraint()->valid_values;
    std::sort(advertised.begin(), advertised.end());
    EXPECT_EQ(advertised, (std::vector<int64_t>{0, 1}));
    // The caller is told these are indices; 0 and 1 are not dtypes.
    EXPECT_NE(dtype->description.find("ordinal"), std::string::npos);
}

TEST(TestIngestorGenericPlanBuilder, AnOrdinalPinSelectsTheKernelCarryingThatValue)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", acceptEveryKernel);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE, DTYPE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    // The catalog is {64 FLOAT, 256 FLOAT, 64 HALF} and the score is the block size, so
    // unpinned the winner is 256 FLOAT; block_size alone cannot reach 64 HALF.
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeIntKnobEngineConfig(fbb, DTYPE, 1);
    const TestGraph graph(makeGraphId(0x96));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getStringMetadata(DTYPE), "HALF");
    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64);
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
            if(found != times.end())
            {
                return found->second;
            }
            return std::nullopt;
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

/// The catalog cache keys on graph content, so every catalog lookup reads the graph's
/// bytes. With benchmarking off and the winner shard already latched by the first build,
/// the winner-cache path must build no GraphContentKey of its own: the second build reads
/// bytes exactly as often as one catalog lookup does (sortedCatalog's cache hit is one
/// catalogFor call and stores nothing back).
TEST(TestIngestorGenericPlanBuilder,
     ASecondColdMissWithBenchmarkingOffReadsGraphBytesOnlyForTheCatalogKey)
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
    const auto bytesCalls = std::make_shared<std::atomic_uint>(0);
    const CountingBytesGraph graph(bytesCalls);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    // What one catalog lookup costs in bytes reads, measured on the same warm entry the
    // second build hits; a cache hit changes no state the build depends on.
    const auto properties = testDeviceProperties();
    bytesCalls->store(0);
    static_cast<void>(manager->unsortedCatalog(MatchContext{graph, 0, properties}));
    const unsigned catalogLookupReads = bytesCalls->load();

    bytesCalls->store(0);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 64);
    EXPECT_EQ(bytesCalls->load(), catalogLookupReads)
        << "with benchmarking off, a second build must read graph bytes only for its one "
           "catalog lookup; the winner-cache path must not key the graph";
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

/// Rejects the front candidate the way a malformed descriptor does, rather than the way a
/// kernel that merely does not fit does. INVALID_VALUE is the status buildPlan() keys its
/// rethrow on.
class MalformedAtBlockSizeHandler : public IKernelDispatchHandler<TestHandle>
{
public:
    explicit MalformedAtBlockSizeHandler(int64_t malformedBlockSize)
        : _malformedBlockSize(malformedBlockSize)
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
        if(kernel.getIntMetadata(BLOCK_SIZE) == _malformedBlockSize)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "kernel source names a library outside the descriptor's directory");
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
    int64_t _malformedBlockSize;
};

/// Fails every candidate, naming the block size in each reason, so the reasons the error
/// gathers can be told apart from one another.
class FailsToPrepareEveryBlockSizeHandler : public IKernelDispatchHandler<TestHandle>
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
                                              const KernelDefinition& kernel) const override
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "no code object at block size " + std::to_string(kernel.getIntMetadata(BLOCK_SIZE)));
    }

    void launch(const TestHandle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
    }
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

TEST(TestIngestorGenericPlanBuilder, RethrowsAMalformedDescriptorInsteadOfServingTheNextKernel)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const MalformedAtBlockSizeHandler handler(64);
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xB7));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected a malformed descriptor to be reported, not skipped";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        // The author's own status and wording, not the aggregate one: falling past this
        // kernel would have served kernel_128 and hidden the fault entirely.
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("outside the descriptor's directory"),
                  std::string::npos)
            << error.what();
    }
}

TEST(TestIngestorGenericPlanBuilder, NamesEveryCandidateReasonWhenNothingCanBeBuilt)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const FailsToPrepareEveryBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xB8));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected the exhausted candidate walk to be reported";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        EXPECT_NE(what.find("3 applicable kernel(s)"), std::string::npos) << what;
        // Every candidate's reason, not just the last: the per-kernel WARN lines that
        // carry the same information are suppressed at the default log level, so this
        // message is the only place the cause survives.
        EXPECT_NE(what.find("block size 64"), std::string::npos) << what;
        EXPECT_NE(what.find("block size 128"), std::string::npos) << what;
        EXPECT_NE(what.find("block size 256"), std::string::npos) << what;
        EXPECT_NE(what.find(toString(testId(0x70))), std::string::npos) << what;
        EXPECT_NE(what.find(toString(testId(0x72))), std::string::npos) << what;
    }
}

TEST(TestIngestorGenericPlanBuilder, RethrowsAMalformedDescriptorWhileBenchmarking)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const MalformedAtBlockSizeHandler handler(64);
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(engine, *manager, resolver);
    const TestHandle handle;

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    // The only difference from RethrowsAMalformedDescriptorInsteadOfServingTheNextKernel:
    // whether a malformed descriptor is reported or absorbed must not follow a tuning setting.
    const TestGraph graph(makeGraphId(0xB9));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
    context.setExecutionSettings(settings);

    try
    {
        builder.buildPlan(handle, graph, engineConfig, context);
        FAIL() << "expected a malformed descriptor to be reported, not dropped from the "
                  "benchmarking set";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("outside the descriptor's directory"),
                  std::string::npos)
            << error.what();
    }
}

TEST(TestIngestorGenericPlanBuilder, NamesEveryCandidateReasonWhenBenchmarkingBuildsNothing)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const FailsToPrepareEveryBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(engine, *manager, resolver);
    const TestHandle handle;

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig
        = makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    // The benchmarking half of NamesEveryCandidateReasonWhenNothingCanBeBuilt.
    const TestGraph graph(makeGraphId(0xBA));

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, engineConfig, settings);
    ASSERT_TRUE(settings.ingestorSettings.benchmarkingEnabled);

    BenchmarkContext context;
    context.setExecutionSettings(settings);

    try
    {
        builder.buildPlan(handle, graph, engineConfig, context);
        FAIL() << "expected the exhausted candidate set to be reported";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        const std::string what = error.what();
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
        EXPECT_NE(what.find("3 applicable kernel(s)"), std::string::npos) << what;
        EXPECT_NE(what.find("block size 64"), std::string::npos) << what;
        EXPECT_NE(what.find("block size 128"), std::string::npos) << what;
        EXPECT_NE(what.find("block size 256"), std::string::npos) << what;
    }
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
std::vector<const KernelDefinition*>
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
    for(const KernelDefinition* kernel : catalog)
    {
        const auto blockSize = kernel->getIntMetadata(BLOCK_SIZE);
        record.entries.push_back(rankedEntryFor(*kernel, blockSize == 256 ? 0.1 : 9.0));
    }
    std::stable_sort(record.entries.begin(),
                     record.entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.timeMs < rhs.timeMs; });
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(0, graph, engineConfig, context);

    EXPECT_EQ(context.plan().kernel().getIntMetadata(BLOCK_SIZE), 256)
        << "the measured winner must beat the heuristic front";
}

/// The catalog key already holds the graph's verified content, so ordering from a record
/// reuses it: a sort that finds a record reads graph bytes exactly as often as one catalog
/// lookup does, not once more for a winner key of its own.
TEST(TestIngestorGenericPlanBuilder, OrderingFromARecordReadsGraphBytesOnlyForTheCatalogKey)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto bytesCalls = std::make_shared<std::atomic_uint>(0);
    const CountingBytesGraph graph(bytesCalls);
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    WinnerRecord record;
    for(const KernelDefinition* kernel : catalog)
    {
        record.entries.push_back(
            rankedEntryFor(*kernel, 1.0 + static_cast<double>(record.entries.size())));
    }
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

    bytesCalls->store(0);
    static_cast<void>(manager->unsortedCatalog(context));
    const unsigned catalogLookupReads = bytesCalls->load();
    ASSERT_GT(catalogLookupReads, 0U);

    bytesCalls->store(0);
    const auto sorted = manager->sortedCatalog(context);

    ASSERT_NE(sorted.measuredRecord, nullptr) << "the record must have ordered this catalog";
    EXPECT_EQ(bytesCalls->load(), catalogLookupReads)
        << "ordering from a record must reuse the catalog key's graph content, not key the "
           "graph again";
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
    ASSERT_EQ(catalog.front()->getIntMetadata(BLOCK_SIZE), 64)
        << "this test needs the heuristic front to differ from the record's";

    WinnerRecord record;
    double time = 1.0;
    for(auto kernel = catalog.rbegin(); kernel != catalog.rend(); ++kernel)
    {
        record.entries.push_back(rankedEntryFor(**kernel, time));
        time += 1.0;
    }
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalogFor(*manager, graph, properties))
    {
        record.entries.push_back(
            rankedEntryFor(*kernel, kernel->getIntMetadata(BLOCK_SIZE) == 256 ? 0.1 : 9.0));
    }
    ASSERT_EQ(record.entries.size(), 3U);

    // A prior wider run's leftover entry for a kernel this engine does not admit. Must
    // be skipped without failing coverage.
    record.entries.push_back(RankedEntry{testId(0xAA), testId(0xF0), testId(0xD0), 0.05});
    std::stable_sort(record.entries.begin(),
                     record.entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.timeMs < rhs.timeMs; });
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalogFor(*manager, graph, properties))
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) == 256)
        {
            record.entries.push_back(rankedEntryFor(*kernel, 0.1));
        }
    }
    ASSERT_EQ(record.entries.size(), 1U);
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalogFor(*manager, graph, properties))
    {
        auto entry = rankedEntryFor(*kernel, 0.1);
        entry.packId = testId(0xEE);
        record.entries.push_back(entry);
    }
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalog)
    {
        auto entry = rankedEntryFor(*kernel, kernel->getIntMetadata(BLOCK_SIZE) == 128 ? 0.1 : 9.0);
        if(kernel->getIntMetadata(BLOCK_SIZE) == 128)
        {
            entry.packId = testId(0xEE);
        }
        record.entries.push_back(entry);
    }
    std::stable_sort(record.entries.begin(),
                     record.entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.timeMs < rhs.timeMs; });
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalog)
    {
        record.entries.push_back(
            rankedEntryFor(*kernel, timeByBlockSize.at(kernel->getIntMetadata(BLOCK_SIZE))));
    }
    std::stable_sort(record.entries.begin(),
                     record.entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.timeMs < rhs.timeMs; });
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

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

/// The other half of the walk above: carrying past a kernel that cannot be prepared is
/// what the ranked walk is for, but a malformed descriptor is the author's mistake and
/// stops the build. The same descriptor reaching the empty-cache walk already throws, so
/// absorbing it here would make the diagnosis depend on whether a record happened to
/// exist.
TEST(TestIngestorGenericPlanBuilder, ARecordWhoseRankZeroKernelIsMalformedRethrows)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const MalformedAtBlockSizeHandler handler(64);
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);
    const TestGraph graph(makeGraphId(0xDD));
    const auto properties = testDeviceProperties();

    // Fully covering, so the ranked walk runs at all, and ranking kernel_64 -- the
    // malformed one -- first. kernel_128 and kernel_256 both prepare, so a walk that
    // absorbed the throw would serve kernel_128 and report nothing.
    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    const std::map<int64_t, double> timeByBlockSize{{64, 0.1}, {128, 0.2}, {256, 0.3}};
    WinnerRecord record;
    for(const KernelDefinition* kernel : catalog)
    {
        record.entries.push_back(
            rankedEntryFor(*kernel, timeByBlockSize.at(kernel->getIntMetadata(BLOCK_SIZE))));
    }
    std::stable_sort(record.entries.begin(),
                     record.entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.timeMs < rhs.timeMs; });
    manager->recordWinner(winnerKeyFor(graph, properties), record, WinnerWriteCause::FRESH_MISS);

    KnobFilterSettings settings;
    builder.initializeExecutionSettings(0, graph, engineConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    KnobFilterContext context;
    context.setExecutionSettings(settings);

    try
    {
        builder.buildPlan(0, graph, engineConfig, context);
        FAIL() << "expected a malformed descriptor at rank 0 to be reported, not skipped "
                  "in favour of rank 1";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        // The author's own status and wording, as on the cache-free walks.
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("outside the descriptor's directory"),
                  std::string::npos)
            << error.what();
    }
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
    for(const KernelDefinition* kernel : catalogFor(*manager, graph, properties))
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) == 256)
        {
            record.entries.push_back(rankedEntryFor(*kernel, 0.1));
        }
    }
    manager->recordWinner(winnerKeyFor(graph, otherDevice), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalogFor(*manager, graph, properties))
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) == 256)
        {
            record.entries.push_back(rankedEntryFor(*kernel, 0.1));
        }
    }
    manager->recordWinner(
        winnerKeyFor(otherGraph, properties), record, WinnerWriteCause::FRESH_MISS);

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
    for(const KernelDefinition* kernel : catalog)
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) == 128)
        {
            narrow.entries.push_back(rankedEntryFor(*kernel, 0.1));
        }
    }
    ASSERT_EQ(narrow.entries.size(), 1U);
    manager->recordWinner(winnerKeyFor(graph, properties), narrow, WinnerWriteCause::FRESH_MISS);

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

/// RFC 0019 §5 step 8: coverage is decided against the canonical candidate set, before any
/// knob filter, so a record covering only a narrowed run is refused for it too. Narrowed by
/// the workspace limit because a block_size pin always leaves a single kernel.
TEST(TestIngestorGenericPlanBuilder, APartiallyCoveringRecordIsRefusedByTheNarrowedRunToo)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);

    const TestGraph graph(makeGraphId(0xD8));
    const auto properties = testDeviceProperties();

    // The record covers only the two small kernels and ranks kernel_128 first, against the
    // heuristic's order, so the returned plan's workspace shows which order was used.
    const auto catalog = catalogFor(*manager, graph, properties);
    ASSERT_EQ(catalog.size(), 3U);
    ASSERT_EQ(catalog.front()->getIntMetadata(BLOCK_SIZE), 64)
        << "this test needs the heuristic front to differ from the record's";
    WinnerRecord partial;
    for(const KernelDefinition* kernel : catalog)
    {
        const auto blockSize = kernel->getIntMetadata(BLOCK_SIZE);
        if(blockSize == 128)
        {
            partial.entries.push_back(rankedEntryFor(*kernel, 0.1));
        }
    }
    for(const KernelDefinition* kernel : catalog)
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) == 64)
        {
            partial.entries.push_back(rankedEntryFor(*kernel, 9.0));
        }
    }
    ASSERT_EQ(partial.entries.size(), 2U)
        << "the record must cover the narrowed set and nothing more";
    manager->recordWinner(winnerKeyFor(graph, properties), partial, WinnerWriteCause::FRESH_MISS);

    flatbuffers::FlatBufferBuilder wideBuilder;
    const auto wideConfig = makeEmptyEngineConfig(wideBuilder);
    KnobFilterSettings wideSettings;
    builder.initializeExecutionSettings(0, graph, wideConfig, wideSettings);
    ASSERT_FALSE(wideSettings.ingestorSettings.workspaceLimit.has_value());
    KnobFilterContext wideRun;
    wideRun.setExecutionSettings(wideSettings);
    builder.buildPlan(0, graph, wideConfig, wideRun);

    // 200 bytes admits kernel_64 and kernel_128 and excludes kernel_256, leaving exactly
    // the set the record covers.
    flatbuffers::FlatBufferBuilder narrowBuilder;
    const auto narrowConfig = makeIntKnobEngineConfig(
        narrowBuilder, hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, 200);
    KnobFilterSettings narrowSettings;
    builder.initializeExecutionSettings(0, graph, narrowConfig, narrowSettings);
    ASSERT_EQ(narrowSettings.ingestorSettings.workspaceLimit, 200);
    KnobFilterContext narrowRun;
    narrowRun.setExecutionSettings(narrowSettings);
    builder.buildPlan(0, graph, narrowConfig, narrowRun);

    EXPECT_EQ(wideRun.plan().kernel().getIntMetadata(BLOCK_SIZE), 64)
        << "a record that does not cover the whole catalog cannot order it";
    EXPECT_EQ(narrowRun.plan().kernel().getIntMetadata(BLOCK_SIZE), 64)
        << "the narrowed run must read the same order source as the wide one: the record "
           "covers what survived the limit, but orderability is the full catalog's question";
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
    for(const KernelDefinition* kernel : catalog)
    {
        ranking.entries.push_back(
            rankedEntryFor(*kernel, kernel->getIntMetadata(BLOCK_SIZE) == 128 ? 0.1 : 9.0));
    }
    std::stable_sort(ranking.entries.begin(),
                     ranking.entries.end(),
                     [](const auto& lhs, const auto& rhs) { return lhs.timeMs < rhs.timeMs; });
    manager->recordWinner(winnerKeyFor(graph, properties), ranking, WinnerWriteCause::FRESH_MISS);

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
    ASSERT_EQ(stored->entries.size(), 3U) << "every usable candidate belongs in the record";
    // Rank 0 must be the LAST candidate in catalog order (the deterministic sampler's
    // fastest), the opposite of the heuristic front. Compared against ids directly, not
    // sortedDefinitions, which now returns the record's own order.
    EXPECT_EQ(stored->entries.front().kernelId, testId(0x72))
        << "the fastest sampled candidate must rank first, not the heuristic front";
    EXPECT_EQ(stored->entries.back().kernelId, testId(0x70)) << "and the slowest must rank last";
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
    EXPECT_EQ(stored->entries.front().kernelId, testId(0x72))
        << "and it must be the sampled ranking, not the heuristic order";

    // The served path: the record orders the catalog for the renumbered graph, so its
    // plan is built from the measured front rather than re-sampled.
    const auto ordered = manager->sortedDefinitions(MatchContext{renumbered, 0, properties});
    ASSERT_FALSE(ordered.empty());
    EXPECT_EQ(ordered.front()->kernelId, testId(0x72))
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
    for(const KernelDefinition* kernel : catalog)
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) == 128)
        {
            narrow.entries.push_back(rankedEntryFor(*kernel, 0.1));
        }
    }
    ASSERT_EQ(narrow.entries.size(), 1U);
    manager->recordWinner(winnerKeyFor(graph, properties), narrow, WinnerWriteCause::FRESH_MISS);

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
    EXPECT_EQ(stored->entries.size(), 3U)
        << "the superset write-back must carry all three benchmarked candidates";
}

/// One arch-keyed UHD serves every board of that arch, so each row carries the device's
/// facts, written through the same deviceFeatureValues() the extractor binds.
TEST(TestIngestorGenericPlanBuilderBenchmarkRecord, EveryCandidateRowCarriesTheDeviceFacts)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
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
    std::vector<std::byte> workspace(context.plan().getWorkspaceSize(handle));
    context.plan().execute(handle, nullptr, 0U, workspace.data());

    nlohmann::json row;
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        const auto start = recorded.message.find('{');
        if(start == std::string::npos)
        {
            continue;
        }
        auto parsed = nlohmann::json::parse(recorded.message.substr(start), nullptr, false);
        if(!parsed.is_discarded() && parsed.contains("event")
           && parsed["event"] == "ingestor.benchmark.candidate")
        {
            row = std::move(parsed);
            break;
        }
    }
    ASSERT_FALSE(row.is_null()) << "no candidate record was logged at all";

    const auto properties = testDeviceProperties();
    ASSERT_TRUE(row.contains("device.cu_count"));
    EXPECT_EQ(row["device.cu_count"].get<int64_t>(), properties.multiProcessorCount);
    ASSERT_TRUE(row.contains("device.total_global_mem"));
    EXPECT_EQ(row["device.total_global_mem"].get<int64_t>(),
              static_cast<int64_t>(properties.totalGlobalMem));
    ASSERT_TRUE(row.contains("device.peak_memory_bandwidth"));
    EXPECT_DOUBLE_EQ(row["device.peak_memory_bandwidth"].get<double>(),
                     peakMemoryBandwidth(properties));

    // The device identity stays envelope: a model splitting on it memorises the fleet.
    EXPECT_FALSE(row.contains("device.arch"));
    EXPECT_FALSE(row.contains("device.gcn_arch_name"));
}

TEST(TestIngestorCandidateEnumeration, SparsePagesEnrollExactlyTheirReportedCandidate)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter benchmarking(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "0");
    const ScopedSymbols symbols(
        "test.graph",
        [](const MatchContext& context) {
            auto bound = acceptGraph(context);
            (*bound)["$attention.dims"] = std::vector<int64_t>{32, 128};
            (*bound)["$attention.dims[0]"] = int64_t{99};
            return bound;
        },
        "test.kernel",
        countingFloatKernels);
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    auto schema = makeSchema();
    schema.fields.push_back({"vector_width", MetadataType::INT, MetadataValue{int64_t{1}}});
    auto pack = makePack({KERNEL_MATCHER_ID});
    pack.kernels[1].metadata["vector_width"] = int64_t{4};
    // A device-inapplicable tuple must not appear, even though all its knob values exist.
    auto otherDevice = makeKernel(testId(0x70), "other_device", 128, "FLOAT");
    otherDevice.arch = {"gfx999"};
    pack.kernels.push_back(std::move(otherDevice));
    const StateManager manager(
        std::move(schema),
        {{KERNEL_MATCHER_ID, "kernel scoped", MatchScope::KERNEL, "test.kernel"}},
        makeTestDispatches(),
        {std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        "test.graph");
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE, "vector_width"});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, manager, resolver);
    const TestGraph graph(makeGraphId(0xDA));
    flatbuffers::FlatBufferBuilder empty;
    const auto all = makeEmptyEngineConfig(empty);
    const auto first = builder.enumerateCandidates(0, graph, all, 0, 1);
    const auto second = builder.enumerateCandidates(0, graph, all, 1, 1);
    ASSERT_EQ(first.total_count, 2U); // Not the four-point Cartesian product.
    ASSERT_EQ(first.candidates.size(), 1U);
    ASSERT_EQ(second.candidates.size(), 1U);
    EXPECT_EQ(first.candidates.front()->id, toString(testId(0x64)));
    EXPECT_EQ(second.candidates.front()->id, toString(testId(0x65)));
    EXPECT_EQ(first.graph_id, second.graph_id);
    EXPECT_EQ(first.device_id, second.device_id);
    EXPECT_EQ(nlohmann::json::parse(first.problem_features).at("test.bound_token"),
              BOUND_TOKEN_VALUE);
    EXPECT_FALSE(nlohmann::json::parse(first.problem_features).contains("q.test.bound_token"));
    const auto problemFeatures = nlohmann::json::parse(first.problem_features);
    EXPECT_EQ(problemFeatures.at("attention.dims[0]"), 99);
    EXPECT_EQ(problemFeatures.at("attention.dims[1]"), 128);

    for(const auto* page : {&first, &second})
    {
        const auto& candidate = *page->candidates.front();
        hipdnn_flatbuffers_sdk::data_objects::EngineConfigT enrolled;
        enrolled.engine_id = ENGINE_ID.front();
        for(const auto& knob : candidate.knob_settings)
        {
            enrolled.knobs.push_back(
                std::make_unique<hipdnn_flatbuffers_sdk::data_objects::KnobSettingT>(*knob));
        }
        flatbuffers::FlatBufferBuilder serialized;
        serialized.Finish(
            hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(serialized, &enrolled));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configuration(
            serialized.GetBufferPointer(), serialized.GetSize());
        KnobFilterSettings settings;
        builder.initializeExecutionSettings(0, graph, configuration, settings);
        KnobFilterContext context;
        context.setExecutionSettings(settings);
        builder.buildPlan(0, graph, configuration, context);
        EXPECT_EQ(toString(context.plan().kernel().kernelId), candidate.id);
    }
    flatbuffers::FlatBufferBuilder scopedBuffer;
    const auto scope = makeIntKnobEngineConfig(scopedBuffer, "vector_width", 4);
    const auto scoped = builder.enumerateCandidates(0, graph, scope, 0, 10000);
    ASSERT_EQ(scoped.total_count, 1U);
    EXPECT_EQ(scoped.candidates.front()->id, second.candidates.front()->id);
    EXPECT_THROW(builder.enumerateCandidates(0, graph, all, 3, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestIngestorCandidateEnumeration, RejectsAmbiguityBeforeReturningTheFirstPage)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);
    const TestGraph graph(makeGraphId(0xDB));
    flatbuffers::FlatBufferBuilder serialized;
    const auto all = makeEmptyEngineConfig(serialized);
    EXPECT_THROW(builder.enumerateCandidates(0, graph, all, 0, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestIngestorCandidateEnumeration, EmptyScopedCatalogIsNotUnsupportedOrAnInvalidScope)
{
    const ScopedSymbols symbols("test.graph", acceptGraph, "test.kernel", countingFloatKernels);
    const auto manager = makeStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const TestPlanBuilder builder(engine, *manager, resolver);
    const TestGraph graph(makeGraphId(0xDC));
    flatbuffers::FlatBufferBuilder absentBuffer;
    const auto absent = makeIntKnobEngineConfig(absentBuffer, BLOCK_SIZE, 777);
    const auto empty = builder.enumerateCandidates(0, graph, absent, 0, 1);
    EXPECT_EQ(empty.total_count, 0U);
    EXPECT_TRUE(empty.candidates.empty());
    flatbuffers::FlatBufferBuilder invalidBuffer;
    const auto invalid = makeIntKnobEngineConfig(invalidBuffer, "unknown_knob", 1);
    EXPECT_THROW(builder.enumerateCandidates(0, graph, invalid, 0, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// One problem feature set for collection and for the live ranker
// ---------------------------------------------------------------------------

/// Every value shape a matcher can publish, plus two names in namespaces the engine owns
/// that no matcher may claim: the corpus and the ranker must both carry the canonical value.
std::optional<BoundTokens> bindEveryTokenShape(const MatchContext& context)
{
    auto bound = acceptGraph(context);
    (*bound)["$attention.dims"] = std::vector<int64_t>{32, 128};
    (*bound)["$attention.causal"] = true;
    (*bound)["$attention.scale"] = 0.125;
    (*bound)["$attention.layout"] = std::string("bshd");
    (*bound)["$graph.flops"] = 1.0;
    (*bound)["device.cu_count"] = int64_t{1};
    return bound;
}

/// The rows the live ranker hands its scorer, in catalog order.
std::vector<std::vector<double>>& capturedRankerRows()
{
    static std::vector<std::vector<double>> s_rows;
    return s_rows;
}

double captureRankerRow(const double* features, size_t count)
{
    capturedRankerRows().emplace_back(features, features + count);
    return 1.0;
}

/// The sweep record's envelope: identity and measurement, never a feature.
bool isSweepEnvelope(const std::string& key)
{
    static const std::set<std::string> s_envelope = {"event",
                                                     "benchmark",
                                                     "device",
                                                     "kernel",
                                                     "pack",
                                                     "dispatch",
                                                     "status",
                                                     "reason",
                                                     "min_ms",
                                                     "avg_ms",
                                                     "stddev_ms",
                                                     "robust_mean_ms",
                                                     "iters"};
    return s_envelope.count(key) != 0;
}

TEST(TestIngestorCatalogFeatureParity, TheLiveRankerReadsWhatEnumerationAndTheSweepPublish)
{
    // A `sort_kernel_catalog` model is fitted on enumeration pages and sweep rows and scored
    // through the live ranker, so all three must bind every name to the same value.
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const ScopedSymbols symbols(
        "test.graph", bindEveryTokenShape, "test.kernel", countingFloatKernels);
    const ScopedConstantScore constantScore;
    const WorkspaceEqualsBlockSizeHandler handler;
    const ScopedDispatchRegistration<TestHandle> dispatch("test.dispatch", handler);
    const auto manager = makeThreeKernelWorkspaceStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const TestDeviceResolver resolver;
    const BenchmarkPlanBuilder builder(
        engine, *manager, resolver, makeThreeKernelDescendingTimer());
    const MatmulTestGraph matmul(16, 32, 8);
    const auto& graph = matmul.graph();
    const TestHandle handle;

    // Collection, first half: the enumeration page.
    flatbuffers::FlatBufferBuilder emptyBuffer;
    const auto page
        = builder.enumerateCandidates(handle, graph, makeEmptyEngineConfig(emptyBuffer), 0, 100);
    ASSERT_EQ(page.candidates.size(), 3U);
    const auto problem = nlohmann::json::parse(page.problem_features);
    const auto device = nlohmann::json::parse(page.device_features);
    EXPECT_DOUBLE_EQ(problem.at("graph.flops").get<double>(), 2.0 * 16 * 32 * 8)
        << "a matmul page must carry the work model's graph.flops, not a matcher's";
    EXPECT_EQ(device.at("device.cu_count"), testDeviceProperties().multiProcessorCount)
        << "a matcher's device.cu_count replaced the device fact";
    EXPECT_EQ(problem.at("attention.dims"), nlohmann::json::array({32, 128}));
    for(const auto& entry : problem.items())
    {
        EXPECT_NE(entry.key().rfind("device.", 0), 0U) << entry.key() << " is not a problem fact";
    }
    for(const auto& entry : device.items())
    {
        EXPECT_EQ(entry.key().rfind("device.", 0), 0U) << entry.key() << " is not a device fact";
    }
    std::map<std::string, nlohmann::json> published; // candidate id -> its full feature row
    for(const auto& candidate : page.candidates)
    {
        auto row = problem;
        row.update(device);
        row.update(nlohmann::json::parse(candidate->kernel_features));
        published[candidate->id] = std::move(row);
    }

    // Collection, second half: the sweep rows.
    flatbuffers::FlatBufferBuilder benchmarkBuffer;
    const auto benchmarking
        = makeIntKnobEngineConfig(benchmarkBuffer, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1);
    KnobFilterSettings settings;
    builder.initializeExecutionSettings(handle, graph, benchmarking, settings);
    BenchmarkContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(handle, graph, benchmarking, context);
    std::vector<std::byte> workspace(context.plan().getWorkspaceSize(handle));
    context.plan().execute(handle, nullptr, 0U, workspace.data());
    size_t sweepRows = 0;
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        const auto start = recorded.message.find('{');
        const auto row
            = start == std::string::npos
                  ? nlohmann::json()
                  : nlohmann::json::parse(recorded.message.substr(start), nullptr, false);
        if(!row.is_object() || row.value("event", "") != "ingestor.benchmark.candidate")
        {
            continue;
        }
        ++sweepRows;
        const auto& expected = published.at(row.at("kernel").get<std::string>());
        for(const auto& entry : expected.items())
        {
            EXPECT_EQ(row.value(entry.key(), nlohmann::json()), entry.value())
                << "sweep row disagrees with the page on " << entry.key();
        }
        for(const auto& entry : row.items())
        {
            EXPECT_TRUE(expected.contains(entry.key()) || isSweepEnvelope(entry.key()))
                << "sweep row publishes " << entry.key() << ", which the page does not";
        }
    }
    EXPECT_EQ(sweepRows, 3U);

    // Runtime: a model reading every scalar the corpus carries. A list has no scalar binding;
    // its indexed elements are read instead.
    std::vector<nlohmann::json> signature;
    std::vector<std::string> names;
    hipdnn_plugin_sdk::uhd::CategoricalEncoding encoding;
    for(const auto& entry : published.begin()->second.items())
    {
        if(entry.value().is_array())
        {
            continue;
        }
        names.push_back(entry.key());
        signature.emplace_back("$" + entry.key());
        for(const auto& row : published)
        {
            if(const auto& value = row.second.at(entry.key()); value.is_string())
            {
                encoding["$" + entry.key()].emplace(value.get<std::string>(),
                                                    encoding["$" + entry.key()].size());
            }
        }
    }
    constexpr const char* CAPTURE_SYMBOL = "hipdnn.kernel_ingestor.test.capture_ranker_row";
    const hipdnn_plugin_sdk::uhd::ScopedNativeScorer capture(CAPTURE_SYMBOL, &captureRankerRow);
    HeuristicDescriptor model;
    model.id = testId(0xAB);
    model.name = "feature parity probe";
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = CAPTURE_SYMBOL;
    model.featuresSignature = signature;
    model.categoricalEncoding = encoding;
    model.featuresHash = hipdnn_plugin_sdk::uhd::FeatureExtractor::computeHash(signature, encoding);
    const auto ranker = UhdKernelHeuristic::tryCreate(
        model, "parity probe", {BLOCK_SIZE, DTYPE}, {BLOCK_SIZE, DTYPE});
    ASSERT_NE(ranker, nullptr);

    const auto properties = testDeviceProperties();
    const MatchContext match{graph, 0, properties};
    const auto catalog = manager->unsortedCatalog(match);
    capturedRankerRows().clear();
    (void)ranker->rankScored(catalog, match);
    ASSERT_EQ(capturedRankerRows().size(), catalog.entries.size())
        << "the ranker could not bind every published name, so its model never scored";
    for(size_t index = 0; index < catalog.entries.size(); ++index)
    {
        const auto& expected = published.at(toString(catalog.entries[index]->kernelId));
        const auto& row = capturedRankerRows()[index];
        ASSERT_EQ(row.size(), names.size());
        for(size_t slot = 0; slot < names.size(); ++slot)
        {
            const auto& value = expected.at(names[slot]);
            const double want = [&] {
                if(value.is_string())
                {
                    return static_cast<double>(
                        encoding.at("$" + names[slot]).at(value.get<std::string>()));
                }
                if(value.is_boolean())
                {
                    return value.get<bool>() ? 1.0 : 0.0;
                }
                return value.get<double>();
            }();
            EXPECT_DOUBLE_EQ(row[slot], want)
                << "the ranker reads " << names[slot] << " differently";
        }
    }
}

// ---------------------------------------------------------------------------
// Output validation through the builder, end to end
// ---------------------------------------------------------------------------

constexpr int64_t VALIDATION_INPUT_UID = 1;
constexpr int64_t VALIDATION_OUTPUT_UID = 2;
constexpr size_t VALIDATION_ELEMENTS = 4;
using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
constexpr size_t VALIDATION_OUTPUT_BYTES = VALIDATION_ELEMENTS * sizeof(float);

/// The three kernels of the validation catalog in heuristic order, identified by block size
/// (which is also the workspace size the keyed timer reads): index 0 is ranked first.
constexpr std::array<int64_t, 3> KERNEL_BLOCKS = {64, 128, 256};

DescriptorId validationKernelId(size_t index)
{
    return testId(static_cast<uint8_t>(0x70 + index));
}

std::vector<uint8_t> floatImage(std::initializer_list<float> values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    size_t offset = 0;
    for(const float value : values)
    {
        std::memcpy(bytes.data() + offset, &value, sizeof(float));
        offset += sizeof(float);
    }
    return bytes;
}

/// What a correct kernel writes. @p variant moves one element by a single ulp, which the float
/// tolerance accepts but a byte compare does not, so a buffer tells which correct kernel wrote it.
std::vector<uint8_t> correctImage(size_t variant = 0)
{
    return floatImage({1.0F + static_cast<float>(variant) * 1.0e-7F, 2.0F, 3.0F, 4.0F});
}

std::vector<uint8_t> wrongImage()
{
    return floatImage({9.0F, 9.0F, 9.0F, 9.0F});
}

class ScriptedPrepared : public PreparedDispatch
{
public:
    explicit ScriptedPrepared(int64_t size)
        : blockSize(size)
    {
    }

    int64_t blockSize;
};

/// Stands in for a real dispatch: a launch writes the scripted bytes into the buffer bound to
/// the output uid, so what a kernel "computes" is visible through the plan the builder hands
/// out. The workspace size is the kernel's block size.
class ScriptedWritingHandler : public IKernelDispatchHandler<TestHandle>
{
public:
    struct Behavior
    {
        /// Written to the output buffer on every launch; nullopt writes nothing.
        std::optional<std::vector<uint8_t>> output;
        /// False makes prepare() decline, so the kernel cannot become a candidate.
        bool buildable = true;
    };

    void script(int64_t blockSize, Behavior behavior)
    {
        _behaviors[blockSize] = std::move(behavior);
    }

    int launches(int64_t blockSize) const
    {
        const auto found = _launches.find(blockSize);
        return found == _launches.end() ? 0 : found->second;
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
        const auto blockSize = kernel.getIntMetadata(BLOCK_SIZE);
        const auto found = _behaviors.find(blockSize);
        if(found != _behaviors.end() && !found->second.buildable)
        {
            return nullptr;
        }
        return std::make_unique<ScriptedPrepared>(blockSize);
    }

    void launch(const TestHandle& /*handle*/,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto blockSize = static_cast<const ScriptedPrepared&>(prepared).blockSize;
        ++_launches[blockSize];
        const auto found = _behaviors.find(blockSize);
        if(found == _behaviors.end() || !found->second.output.has_value())
        {
            return;
        }
        void* output = hipdnn_plugin_sdk::ingestor::detail::findDeviceBuffer(
            deviceBuffers, numDeviceBuffers, VALIDATION_OUTPUT_UID);
        if(output != nullptr)
        {
            std::memcpy(output, found->second.output->data(), found->second.output->size());
        }
    }

private:
    std::map<int64_t, Behavior> _behaviors;
    mutable std::map<int64_t, int> _launches;
};

/// How often the host stand-ins for the device hooks ran.
struct ValidationHookCounters
{
    int fills = 0;
    int snapshots = 0;
};

/// Host memset and memcpy in place of the device operations; the buffers are host memory.
BenchmarkValidationHooks hostValidationHooks(ValidationHookCounters& counters)
{
    BenchmarkValidationHooks hooks;
    hooks.fill = [&counters](hipStream_t,
                             const hipdnnPluginDeviceBuffer_t* buffers,
                             uint32_t count,
                             const std::vector<OutputBufferSpec>& specs) {
        ++counters.fills;
        for(const auto& spec : specs)
        {
            void* ptr
                = hipdnn_plugin_sdk::ingestor::detail::findDeviceBuffer(buffers, count, spec.uid);
            if(ptr != nullptr)
            {
                std::memset(ptr,
                            hipdnn_plugin_sdk::output_agreement::fillByte(spec.dataType),
                            spec.byteSize);
            }
        }
    };
    hooks.snapshot = [&counters](hipStream_t,
                                 const hipdnnPluginDeviceBuffer_t* buffers,
                                 uint32_t count,
                                 const std::vector<OutputBufferSpec>& specs,
                                 OutputImages& images) {
        ++counters.snapshots;
        images.assign(specs.size(), {});
        for(size_t i = 0; i < specs.size(); ++i)
        {
            const void* ptr = hipdnn_plugin_sdk::ingestor::detail::findDeviceBuffer(
                buffers, count, specs[i].uid);
            if(ptr == nullptr)
            {
                return false;
            }
            images[i].resize(specs[i].byteSize);
            std::memcpy(images[i].data(), ptr, specs[i].byteSize);
        }
        return true;
    };
    return hooks;
}

/// The input and output buffers of one execute, in host memory. The output can be made to
/// share the input's memory.
class HostBuffers
{
public:
    HostBuffers()
        : _input(VALIDATION_OUTPUT_BYTES, 0x00)
        , _output(VALIDATION_OUTPUT_BYTES, 0x00)
    {
    }

    void aliasOutputToInput()
    {
        _aliased = true;
    }

    const std::vector<uint8_t>& output() const
    {
        return _aliased ? _input : _output;
    }

    std::array<hipdnnPluginDeviceBuffer_t, 2> bindings()
    {
        return {{{VALIDATION_INPUT_UID, _input.data()},
                 {VALIDATION_OUTPUT_UID, _aliased ? _input.data() : _output.data()}}};
    }

private:
    std::vector<uint8_t> _input;
    std::vector<uint8_t> _output;
    bool _aliased = false;
};

/// Keeps the IPlan buildPlan() hands out, and which kind it was: a GenericPlan is a plan served
/// from a stored record or the heuristic, anything else is a benchmark sweep.
struct ValidationContext
{
    void setExecutionSettings(const KnobFilterSettings& settings)
    {
        _settings = settings;
    }

    const KnobFilterSettings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<GenericPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
        _servedWithoutBenchmark = true;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> plan)
    {
        _plan = std::move(plan);
        _servedWithoutBenchmark = false;
    }

    bool servedWithoutBenchmark() const
    {
        return _servedWithoutBenchmark;
    }

    const hipdnn_plugin_sdk::IPlan<TestHandle>& plan() const
    {
        return *_plan;
    }

    /// The block size of the kernel a non-benchmark plan serves.
    int64_t servedBlockSize() const
    {
        return static_cast<const GenericPlan<TestHandle>&>(*_plan).kernel().getIntMetadata(
            BLOCK_SIZE);
    }

private:
    KnobFilterSettings _settings;
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<TestHandle>> _plan;
    bool _servedWithoutBenchmark = false;
};

using ValidationBuilder = GenericPlanBuilder<TestHandle, KnobFilterSettings, ValidationContext>;

/// A private HIPDNN_CACHE_DIR for one test, with the disable switch cleared, so the winner shard
/// the builder writes can be read back as text.
class ScopedShardDir
{
public:
    ScopedShardDir()
        : _path(std::filesystem::temp_directory_path()
                / ("hipdnn_builder_validation_" + uniqueDirectoryName()))
        , _cacheDir("HIPDNN_CACHE_DIR", _path.string())
        , _cacheDisabled("HIPDNN_DISABLE_CACHE")
    {
        std::filesystem::create_directories(_path);
    }

    ScopedShardDir(const ScopedShardDir&) = delete;
    ScopedShardDir& operator=(const ScopedShardDir&) = delete;
    ScopedShardDir(ScopedShardDir&&) = delete;
    ScopedShardDir& operator=(ScopedShardDir&&) = delete;

    ~ScopedShardDir()
    {
        std::error_code ignored;
        std::filesystem::remove_all(_path, ignored);
    }

private:
    /// A steady-clock reading separates concurrent processes, the counter separates two
    /// directories of one process.
    static std::string uniqueDirectoryName()
    {
        static std::atomic<uint64_t> s_counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::to_string(static_cast<uint64_t>(stamp)) + "_"
               + std::to_string(s_counter.fetch_add(1));
    }

    std::filesystem::path _path;
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _cacheDir;
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _cacheDisabled;
};

std::vector<std::string> fileLines(const std::filesystem::path& path)
{
    std::vector<std::string> lines;
    std::ifstream stream(path);
    for(std::string line; std::getline(stream, line);)
    {
        lines.push_back(line);
    }
    return lines;
}

std::vector<DescriptorId> entryIdsOf(const WinnerRecord& record)
{
    std::vector<DescriptorId> ids;
    ids.reserve(record.entries.size());
    for(const auto& entry : record.entries)
    {
        ids.push_back(entry.kernelId);
    }
    return ids;
}

std::vector<std::string> logsAt(const hipdnn_test_sdk::utilities::SharedLogRecorder& recorder,
                                hipdnnSeverity_t severity)
{
    std::vector<std::string> messages;
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        if(recorded.severity == severity)
        {
            messages.push_back(recorded.message);
        }
    }
    return messages;
}

/// True when one message contains every needle.
bool anyLogContainsAll(const std::vector<std::string>& messages,
                       std::initializer_list<std::string> needles)
{
    return std::any_of(messages.begin(), messages.end(), [&needles](const std::string& message) {
        return std::all_of(needles.begin(), needles.end(), [&message](const std::string& needle) {
            return message.find(needle) != std::string::npos;
        });
    });
}

/// One pointwise node reading the input uid and writing the output uid.
ContentCarryingTestGraph::Spec validationGraphSpec()
{
    ContentCarryingTestGraph::Spec spec;
    spec.name = "validation_graph";
    const std::vector<int64_t> dims{static_cast<int64_t>(VALIDATION_ELEMENTS)};
    const std::vector<int64_t> strides{1};
    spec.tensors = {
        ContentCarryingTestGraph::TensorSpec{VALIDATION_INPUT_UID, DataType::FLOAT, dims, strides},
        ContentCarryingTestGraph::TensorSpec{
            VALIDATION_OUTPUT_UID, DataType::FLOAT, dims, strides}};
    spec.nodes[0].in0TensorUid = VALIDATION_INPUT_UID;
    spec.nodes[0].out0TensorUid = VALIDATION_OUTPUT_UID;
    return spec;
}

/// The kernels the validation tests build: a graph with one pointwise node, an engine of up to
/// three kernels whose launches write scripted bytes, host buffers and host hooks, a private
/// winner cache directory, and the environment switch under the test's control.
class TestIngestorGenericPlanBuilderValidation : public ::testing::Test
{
protected:
    /// HIPDNN_VALIDATE_BENCHMARKING on or off for the rest of the test.
    void validateEnvironment(bool on)
    {
        _validateEnvironment.setValue(on ? "1" : "0");
    }

    void script(size_t kernelIndex,
                std::optional<std::vector<uint8_t>> output,
                bool buildable = true)
    {
        _handler.script(KERNEL_BLOCKS[kernelIndex],
                        ScriptedWritingHandler::Behavior{std::move(output), buildable});
    }

    /// Milliseconds per kernel, in catalog order; a kernel without an entry cannot be timed.
    static BenchmarkPlan<TestHandle>::Timer timesMs(std::initializer_list<double> perKernel)
    {
        std::unordered_map<size_t, double> byWorkspace;
        size_t index = 0;
        for(const double time : perKernel)
        {
            byWorkspace[static_cast<size_t>(KERNEL_BLOCKS[index++])] = time;
        }
        return makeWorkspaceKeyedTimer(std::move(byWorkspace));
    }

    BenchmarkValidationHooks hooks()
    {
        return hostValidationHooks(_hookCounters);
    }

    /// A state manager over the first @p kernelCount kernels whose winner shard is named by
    /// @p engineName. Two managers of one name share a shard, as two processes would.
    static std::unique_ptr<StateManager> manager(const std::string& engineName,
                                                 size_t kernelCount = 3)
    {
        KernelDescriptorPack pack;
        pack.id = PACK_ID;
        pack.name = "test pack";
        pack.engineId = ENGINE_ID;
        pack.dispatchId = DISPATCH_ID;
        const std::vector<KernelDescriptor> kernels
            = {makeKernel(validationKernelId(0), "kernel_64", 64, "FLOAT", /*priority=*/30),
               makeKernel(validationKernelId(1), "kernel_128", 128, "FLOAT", /*priority=*/20),
               makeKernel(validationKernelId(2), "kernel_256", 256, "FLOAT", /*priority=*/10)};
        pack.kernels.assign(kernels.begin(),
                            kernels.begin() + static_cast<std::ptrdiff_t>(kernelCount));

        ensureNoopDispatchRegistered<TestHandle>("test.dispatch");
        return std::make_unique<StateManager>(
            makeSchema(),
            std::vector<MatchDescriptor>{},
            std::vector<DispatchDescriptor>{{DISPATCH_ID, "test dispatch", "test.dispatch"}},
            std::vector<KernelDescriptorPack>{std::move(pack)},
            std::make_shared<NativeKernelHeuristic>(CONSTANT_SCORE_SYMBOL),
            "test.graph",
            "engine 'validation'",
            StateManager::DEFAULT_CATALOG_CACHE_CAPACITY,
            EngineIdentity{engineName});
    }

    static std::filesystem::path shardPath(const std::string& engineName)
    {
        return winnerCacheShardPath(EngineIdentity{engineName}, testDeviceProperties().gcnArchName);
    }

    static std::vector<std::string> shardLines(const std::string& engineName)
    {
        return fileLines(shardPath(engineName));
    }

    static std::string shardText(const std::string& engineName)
    {
        std::string text;
        for(const auto& line : shardLines(engineName))
        {
            text += line + "\n";
        }
        return text;
    }

    /// Digest of the kernels @p manager admits for the graph, before anything is recorded.
    CatalogDigest catalogDigestOf(const StateManager& stateManager) const
    {
        return catalogDigest(
            stateManager.unsortedDefinitions(MatchContext{_graph, 0, testDeviceProperties()}));
    }

    WinnerKey winnerKey() const
    {
        return winnerKeyFor(_graph, testDeviceProperties());
    }

    /// Ranked entries for the kernels at @p order (indexes into KERNEL_BLOCKS), best first.
    std::vector<RankedEntry> entriesFor(const StateManager& stateManager,
                                        std::initializer_list<size_t> order) const
    {
        const auto catalog = catalogFor(stateManager, _graph, testDeviceProperties());
        std::vector<RankedEntry> entries;
        for(const size_t index : order)
        {
            const auto kernel = std::find_if(
                catalog.begin(), catalog.end(), [index](const KernelDefinition* candidate) {
                    return candidate->kernelId == validationKernelId(index);
                });
            if(kernel == catalog.end())
            {
                ADD_FAILURE() << "kernel " << index << " is not in the catalog";
                continue;
            }
            entries.push_back(rankedEntryFor(**kernel, 1.0 + static_cast<double>(entries.size())));
        }
        return entries;
    }

    /// initializeExecutionSettings() and buildPlan(), benchmarking requested through its knob
    /// or left off.
    void
        build(const ValidationBuilder& builder, bool benchmarking, ValidationContext& context) const
    {
        flatbuffers::FlatBufferBuilder fbb;
        const auto engineConfig
            = benchmarking
                  ? makeIntKnobEngineConfig(fbb, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1)
                  : makeEmptyEngineConfig(fbb);
        KnobFilterSettings settings;
        builder.initializeExecutionSettings(_handle, _graph, engineConfig, settings);
        context.setExecutionSettings(settings);
        builder.buildPlan(_handle, _graph, engineConfig, context);
    }

    /// One execute() of the built plan over @p buffers, which a benchmark plan resolves on.
    void execute(const ValidationContext& context, HostBuffers& buffers) const
    {
        auto bindings = buffers.bindings();
        std::vector<std::byte> workspace(context.plan().getWorkspaceSize(_handle));
        context.plan().execute(
            _handle, bindings.data(), static_cast<uint32_t>(bindings.size()), workspace.data());
    }

    const ScopedSymbols _symbols{"test.graph", acceptGraph, "test.kernel", countingFloatKernels};
    const ScopedConstantScore _constantScore;
    ScriptedWritingHandler _handler;
    const ScopedDispatchRegistration<TestHandle> _dispatch{"test.dispatch", _handler};
    ValidationHookCounters _hookCounters;
    const ContentCarryingTestGraph _graph{validationGraphSpec()};
    const TestHandle _handle;
    const TestDeviceResolver _resolver;
    const EngineDescriptor _engine = makeEngineWithKnobs({BLOCK_SIZE});

private:
    // Each guard clears its variable for the test and restores it after; the shard directory
    // goes first so the winner cache the managers open lives in it.
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _forceBenchmarking{
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME};
    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _validateEnvironment{
        hipdnn_plugin_sdk::VALIDATE_BENCHMARKING_ENV_NAME};
    ScopedShardDir _shardDir;
};

TEST_F(TestIngestorGenericPlanBuilderValidation,
       InitializeExecutionSettingsReadsTheValidateBenchmarkingEnvironment)
{
    const auto stateManager = manager("test:ValidationSettings");
    const ValidationBuilder builder(_engine, *stateManager, _resolver);
    flatbuffers::FlatBufferBuilder fbb;
    const auto engineConfig = makeEmptyEngineConfig(fbb);

    KnobFilterSettings unset;
    builder.initializeExecutionSettings(_handle, _graph, engineConfig, unset);
    EXPECT_FALSE(unset.ingestorSettings.validateBenchmarking);

    validateEnvironment(true);
    KnobFilterSettings on;
    builder.initializeExecutionSettings(_handle, _graph, engineConfig, on);
    EXPECT_TRUE(on.ingestorSettings.validateBenchmarking);

    validateEnvironment(false);
    KnobFilterSettings off;
    builder.initializeExecutionSettings(_handle, _graph, engineConfig, off);
    EXPECT_FALSE(off.ingestorSettings.validateBenchmarking);
}

TEST_F(TestIngestorGenericPlanBuilderValidation, ModeOffPassesNoValidationAndCallsNoHooks)
{
    validateEnvironment(false);
    script(0, correctImage(0));
    script(1, correctImage(1));
    script(2, wrongImage());
    const auto stateManager = manager("test:ValidationModeOff");
    // The wrong kernel is the fastest, which is what mode off lets win.
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);
    ASSERT_FALSE(context.servedWithoutBenchmark());
    HostBuffers buffers;
    execute(context, buffers);

    EXPECT_EQ(_hookCounters.fills, 0);
    EXPECT_EQ(_hookCounters.snapshots, 0);
    EXPECT_EQ(buffers.output(), wrongImage()) << "the fastest candidate wins without validation";
    // One warmup launch per candidate and one launch of the winner; validation adds none.
    EXPECT_EQ(_handler.launches(KERNEL_BLOCKS[0]), BENCHMARK_WARMUP_RUNS);
    EXPECT_EQ(_handler.launches(KERNEL_BLOCKS[1]), BENCHMARK_WARMUP_RUNS);
    EXPECT_EQ(_handler.launches(KERNEL_BLOCKS[2]), BENCHMARK_WARMUP_RUNS + 1);
    const auto stored = stateManager->winnerFor(winnerKey());
    ASSERT_TRUE(stored.has_value());
    EXPECT_FALSE(stored->validated.has_value()) << "a record written without validation is legacy";
    EXPECT_EQ(stored->entries.size(), 3U);
}

TEST_F(TestIngestorGenericPlanBuilderValidation, ValidateBenchmarkingWithBenchmarkingOffIsANoOp)
{
    validateEnvironment(true);

    // No record: the heuristic front is served.
    const auto plainManager = manager("test:ValidationBenchmarkingOffPlain");
    const ValidationBuilder plainBuilder(_engine, *plainManager, _resolver, timesMs({}), hooks());
    ValidationContext heuristic;
    build(plainBuilder, /*benchmarking=*/false, heuristic);
    ASSERT_TRUE(heuristic.servedWithoutBenchmark());
    EXPECT_EQ(heuristic.servedBlockSize(), KERNEL_BLOCKS[0]);

    // A legacy record is still served: the variable only matters while benchmarking.
    const auto stateManager = manager("test:ValidationBenchmarkingOff");
    const ValidationBuilder builder(_engine, *stateManager, _resolver, timesMs({}), hooks());
    stateManager->recordWinner(winnerKey(),
                               WinnerRecord{entriesFor(*stateManager, {2, 1, 0}), std::nullopt},
                               WinnerWriteCause::FRESH_MISS);
    ValidationContext recorded;
    build(builder, /*benchmarking=*/false, recorded);

    ASSERT_TRUE(recorded.servedWithoutBenchmark());
    EXPECT_EQ(recorded.servedBlockSize(), KERNEL_BLOCKS[2]);
    EXPECT_EQ(_hookCounters.fills, 0);
    EXPECT_EQ(_hookCounters.snapshots, 0);
}

TEST_F(TestIngestorGenericPlanBuilderValidation, ACacheHitNeverValidates)
{
    validateEnvironment(true);
    const auto stateManager = manager("test:ValidationCacheHit");
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());
    const WinnerRecord validated{entriesFor(*stateManager, {1, 0, 2}),
                                 catalogDigestOf(*stateManager)};
    stateManager->recordWinner(winnerKey(), validated, WinnerWriteCause::FRESH_MISS);

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);

    ASSERT_TRUE(context.servedWithoutBenchmark());
    EXPECT_EQ(context.servedBlockSize(), KERNEL_BLOCKS[1]);
    EXPECT_EQ(_hookCounters.fills, 0);
    EXPECT_EQ(_hookCounters.snapshots, 0);
    EXPECT_EQ(_handler.launches(KERNEL_BLOCKS[0]) + _handler.launches(KERNEL_BLOCKS[1])
                  + _handler.launches(KERNEL_BLOCKS[2]),
              0);
}

TEST_F(TestIngestorGenericPlanBuilderValidation,
       ModeOnBenchmarkRecordsOnlyPassingKernelsInProcessAndOnDisk)
{
    validateEnvironment(true);
    script(0, correctImage(0));
    script(1, correctImage(1));
    script(2, wrongImage());
    const std::string engineName = "test:ValidationRecords";
    const auto stateManager = manager(engineName);
    const auto expectedDigest = catalogDigestOf(*stateManager);
    // The wrong kernel is the fastest, so only validation keeps it from winning.
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);
    ASSERT_FALSE(context.servedWithoutBenchmark());
    HostBuffers buffers;
    execute(context, buffers);

    EXPECT_EQ(buffers.output(), correctImage(1)) << "the fastest passing kernel wins";
    const auto stored = stateManager->winnerFor(winnerKey());
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(entryIdsOf(*stored),
              (std::vector<DescriptorId>{validationKernelId(1), validationKernelId(0)}))
        << "only passing kernels are recorded, fastest first";
    EXPECT_EQ(stored->validated, std::optional<CatalogDigest>(expectedDigest))
        << "the digest covers every candidate, the failed one included";
    const auto shard = shardText(engineName);
    EXPECT_EQ(shard.find(toString(validationKernelId(2))), std::string::npos)
        << "the failed kernel never reaches the disk";
    EXPECT_NE(shard.find("\"validated\""), std::string::npos);
}

TEST_F(TestIngestorGenericPlanBuilderValidation,
       AnUntouchedKernelIsInTheDigestSoTheNextBuilderServes)
{
    validateEnvironment(true);
    // The heuristic-first kernel is the fastest and writes nothing.
    script(0, std::nullopt);
    script(1, correctImage(1));
    script(2, correctImage(2));
    const std::string engineName = "test:ValidationUntouched";
    const auto stateManager = manager(engineName);
    const auto expectedDigest = catalogDigestOf(*stateManager);
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({1.0, 2.0, 3.0}), hooks());

    ValidationContext first;
    build(builder, /*benchmarking=*/true, first);
    ASSERT_FALSE(first.servedWithoutBenchmark());
    HostBuffers buffers;
    execute(first, buffers);

    EXPECT_EQ(buffers.output(), correctImage(1));
    const auto stored = stateManager->winnerFor(winnerKey());
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(entryIdsOf(*stored),
              (std::vector<DescriptorId>{validationKernelId(1), validationKernelId(2)}));
    EXPECT_EQ(stored->validated, std::optional<CatalogDigest>(expectedDigest));

    // A later process reads the shard and serves the record without sweeping again.
    const auto nextProcess = manager(engineName);
    const ValidationBuilder nextBuilder(
        _engine, *nextProcess, _resolver, timesMs({1.0, 2.0, 3.0}), hooks());
    ValidationContext second;
    build(nextBuilder, /*benchmarking=*/true, second);

    ASSERT_TRUE(second.servedWithoutBenchmark());
    EXPECT_EQ(second.servedBlockSize(), KERNEL_BLOCKS[1]);
}

TEST_F(TestIngestorGenericPlanBuilderValidation, ALegacyRecordUnderModeOnReBenchmarksOnceThenServes)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    validateEnvironment(true);
    script(0, correctImage(0));
    script(1, correctImage(1));
    script(2, correctImage(2));
    const std::string engineName = "test:ValidationLegacy";
    {
        // An earlier run, or an older binary, left a record that covers the whole catalog.
        const auto earlier = manager(engineName);
        earlier->recordWinner(winnerKey(),
                              WinnerRecord{entriesFor(*earlier, {0, 1, 2}), std::nullopt},
                              WinnerWriteCause::FRESH_MISS);
    }
    ASSERT_EQ(shardLines(engineName).size(), 2U) << "the version line and the legacy record";

    const auto stateManager = manager(engineName);
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());
    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);

    ASSERT_FALSE(context.servedWithoutBenchmark()) << "an unvalidated record is not served";
    EXPECT_TRUE(anyLogContainsAll(logsAt(recorder, HIPDNN_SEV_INFO),
                                  {"does not fully cover", "unvalidated record"}));
    HostBuffers buffers;
    execute(context, buffers);
    EXPECT_EQ(shardLines(engineName).size(), 3U) << "exactly one line is appended";

    const auto nextProcess = manager(engineName);
    const ValidationBuilder nextBuilder(
        _engine, *nextProcess, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());
    ValidationContext second;
    build(nextBuilder, /*benchmarking=*/true, second);
    ASSERT_TRUE(second.servedWithoutBenchmark());
    EXPECT_EQ(second.servedBlockSize(), KERNEL_BLOCKS[2]) << "the re-benchmarked winner";
}

TEST_F(TestIngestorGenericPlanBuilderValidation, NoPassingCandidateThroughTheBuilderRecordsNothing)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    validateEnvironment(true);
    // No kernel writes anything, so every output is still the fill pattern.
    script(0, std::nullopt);
    script(1, std::nullopt);
    script(2, std::nullopt);
    const std::string engineName = "test:ValidationNoPass";
    const auto stateManager = manager(engineName);
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);
    HostBuffers buffers;
    execute(context, buffers);

    EXPECT_EQ(stateManager->winnerCacheSize(), 0U);
    EXPECT_FALSE(stateManager->winnerFor(winnerKey()).has_value());
    EXPECT_LE(shardLines(engineName).size(), 1U) << "no record line is written";
    EXPECT_TRUE(
        anyLogContainsAll(logsAt(recorder, HIPDNN_SEV_ERROR), {toString(validationKernelId(0))}))
        << "the error names the kernel that is served instead";
}

TEST_F(TestIngestorGenericPlanBuilderValidation,
       AllTimingUnusableThroughTheBuilderServesAPassingCandidateAndRecordsNothing)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    validateEnvironment(true);
    script(0, wrongImage());
    script(1, correctImage(1));
    script(2, correctImage(2));
    const std::string engineName = "test:ValidationNoTiming";
    const auto stateManager = manager(engineName);
    // The timer reports nothing for any kernel.
    const ValidationBuilder builder(_engine, *stateManager, _resolver, timesMs({}), hooks());

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);
    HostBuffers buffers;
    execute(context, buffers);

    EXPECT_EQ(buffers.output(), correctImage(1))
        << "the first passing candidate is served, never the wrong heuristic-first one";
    EXPECT_EQ(stateManager->winnerCacheSize(), 0U);
    EXPECT_LE(shardLines(engineName).size(), 1U) << "no record line is written";
    EXPECT_FALSE(logsAt(recorder, HIPDNN_SEV_ERROR).empty());
}

TEST_F(TestIngestorGenericPlanBuilderValidation, TheTwoCandidateCaseThroughTheBuilder)
{
    script(0, correctImage(0));
    script(1, wrongImage());
    // The correct kernel is heuristic-first and slower; the wrong one is faster.
    for(const bool validationOn : {true, false})
    {
        SCOPED_TRACE(validationOn ? "validation on" : "validation off");
        validateEnvironment(validationOn);
        const auto stateManager
            = manager(validationOn ? "test:ValidationTwoOn" : "test:ValidationTwoOff", 2);
        const ValidationBuilder builder(
            _engine, *stateManager, _resolver, timesMs({5.0, 2.0}), hooks());

        ValidationContext context;
        build(builder, /*benchmarking=*/true, context);
        HostBuffers buffers;
        execute(context, buffers);

        EXPECT_EQ(buffers.output(), validationOn ? correctImage(0) : wrongImage());
    }
}

TEST_F(TestIngestorGenericPlanBuilderValidation, TheMajorityCaseThroughTheBuilder)
{
    // The wrong kernel is heuristic-first and the fastest; two correct kernels are behind it.
    script(0, wrongImage());
    script(1, correctImage(1));
    script(2, correctImage(2));
    for(const bool validationOn : {true, false})
    {
        SCOPED_TRACE(validationOn ? "validation on" : "validation off");
        validateEnvironment(validationOn);
        const auto stateManager
            = manager(validationOn ? "test:ValidationMajorityOn" : "test:ValidationMajorityOff");
        const ValidationBuilder builder(
            _engine, *stateManager, _resolver, timesMs({1.0, 2.0, 3.0}), hooks());

        ValidationContext context;
        build(builder, /*benchmarking=*/true, context);
        HostBuffers buffers;
        execute(context, buffers);

        EXPECT_EQ(buffers.output(), validationOn ? correctImage(1) : wrongImage());
    }
}

TEST_F(TestIngestorGenericPlanBuilderValidation,
       AnUnbuildableCandidateIsOutsideTheDigestSoTheRecordStaysPartial)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    validateEnvironment(true);
    script(0, correctImage(0));
    script(1, correctImage(1));
    script(2, correctImage(2), /*buildable=*/false);
    const std::string engineName = "test:ValidationUnbuildable";
    const auto stateManager = manager(engineName);
    const auto fullDigest = catalogDigestOf(*stateManager);
    std::vector<const KernelDefinition*> buildable;
    for(const KernelDefinition* kernel : catalogFor(*stateManager, _graph, testDeviceProperties()))
    {
        if(kernel->getIntMetadata(BLOCK_SIZE) != KERNEL_BLOCKS[2])
        {
            buildable.push_back(kernel);
        }
    }
    ASSERT_EQ(buildable.size(), 2U);
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({2.0, 1.0}), hooks());

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);
    HostBuffers buffers;
    execute(context, buffers);

    const auto stored = stateManager->winnerFor(winnerKey());
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->validated, std::optional<CatalogDigest>(catalogDigest(buildable)))
        << "the digest covers the two candidates that were swept";
    EXPECT_NE(stored->validated, std::optional<CatalogDigest>(fullDigest));

    // The record does not cover the unbuildable kernel, so the next process sweeps again.
    const auto nextProcess = manager(engineName);
    const ValidationBuilder nextBuilder(
        _engine, *nextProcess, _resolver, timesMs({2.0, 1.0}), hooks());
    ValidationContext second;
    build(nextBuilder, /*benchmarking=*/true, second);
    EXPECT_FALSE(second.servedWithoutBenchmark());
    EXPECT_TRUE(anyLogContainsAll(logsAt(recorder, HIPDNN_SEV_INFO), {"does not fully cover"}));
}

TEST_F(TestIngestorGenericPlanBuilderValidation, AliasedBuffersThroughTheBuilderSkipValidation)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    validateEnvironment(true);
    script(0, correctImage(0));
    script(1, correctImage(1));
    script(2, correctImage(2));
    const std::string engineName = "test:ValidationAliased";
    const auto stateManager = manager(engineName);
    const auto expectedDigest = catalogDigestOf(*stateManager);
    const ValidationBuilder builder(
        _engine, *stateManager, _resolver, timesMs({3.0, 2.0, 1.0}), hooks());

    ValidationContext context;
    build(builder, /*benchmarking=*/true, context);
    HostBuffers buffers;
    buffers.aliasOutputToInput();
    execute(context, buffers);

    // Filling the output would have destroyed the input it shares memory with.
    EXPECT_EQ(_hookCounters.fills, 0);
    EXPECT_EQ(_hookCounters.snapshots, 0);
    EXPECT_TRUE(anyLogContainsAll(logsAt(recorder, HIPDNN_SEV_WARN), {"aliases input"}));
    const auto stored = stateManager->winnerFor(winnerKey());
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->validated, std::optional<CatalogDigest>(expectedDigest))
        << "an inconclusive resolve is still recorded as validated";
    EXPECT_EQ(stored->entries.size(), 3U);
}

} // namespace
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
