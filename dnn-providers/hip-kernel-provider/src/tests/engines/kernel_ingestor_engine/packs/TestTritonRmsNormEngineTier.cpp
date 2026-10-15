// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: if this file lands on an ungated source list
// it compiles to an empty translation unit instead of a suite reading an unstaged shard.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/TritonRmsNormTestGraphs.hpp"

/**
 * @file TestTritonRmsNormEngineTier.cpp
 * @brief Which graphs the hipkernel:TritonRmsNorm engine itself accepts: the engine's
 *        schema gate in front of the pack's matchers, over the staged descriptor set.
 *
 * Part of hip_kernel_provider_census_tests, registered per shipped arch in
 * src/tests/CMakeLists.txt. The engine is the provider's own plan builder over the staged
 * set, with a device resolver reporting the census entry's arch, so no device is needed.
 * Each graph carries the minimum engine API version the backend stamps on it, computed by
 * the backend's own function, so a graph newer than the engine's schema is declined here
 * as it would be in a real plan build.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using data_objects::NormFwdPhase;

/// The arch the census entry names. Supplied by the entry, not detected from a device.
std::string expectedArch()
{
    auto arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
    if(arch.empty())
    {
        throw std::runtime_error("HIPDNN_TEST_EXPECTED_ARCH is unset, so this census has no "
                                 "arch to resolve the device to");
    }
    return arch;
}

/// A fixed, device-less machine identity reporting @p arch. The state manager skips every
/// pack not built for the device's arch, so this must be the staged shard's.
class FixedArchDeviceResolver : public IDeviceResolver<Handle>
{
public:
    explicit FixedArchDeviceResolver(std::string arch)
    {
        _properties.gcnArchName = std::move(arch);
        _properties.warpSize = 64;
    }

    DeviceId deviceId(const Handle& /*handle*/) const override
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

struct TierSettings
{
    IngestorSettings ingestorSettings;
};

/// The minimum a GenericPlanBuilder asks of its context. The provider's real Context is
/// not used because it needs a live Container and handle; applicability reads neither.
class TierContext
{
public:
    const TierSettings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> plan)
    {
        _plan = std::move(plan);
    }

private:
    TierSettings _settings;
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> _plan;
};

using TierPlanBuilder = GenericPlanBuilder<Handle, TierSettings, TierContext>;

/// @p options's graph stamped with the minimum engine API version it requires, from the
/// same graph-level flags the backend derives when it builds a graph.
flatbuffers::FlatBufferBuilder stampedGraph(TritonRmsNormGraphOptions options)
{
    const auto unstamped = buildTritonRmsNormGraph(options);
    const auto* graph = data_objects::GetGraph(unstamped.GetBufferPointer());

    // The schema defaults, which a tensor that sets nothing carries.
    const data_objects::TensorAttributesT defaults;
    bool runtimePassByValue = false;
    bool ragged = false;
    bool nonDefaultAlignment = false;
    bool raggedOffsetMultiplier = false;
    for(const auto* tensor : *graph->tensors())
    {
        runtimePassByValue = runtimePassByValue || tensor->is_runtime_pass_by_value();
        ragged = ragged || tensor->ragged_offset_tensor_uid().has_value();
        nonDefaultAlignment = nonDefaultAlignment || tensor->alignment() != defaults.alignment;
        raggedOffsetMultiplier
            = raggedOffsetMultiplier
              || tensor->ragged_offset_multiplier() != defaults.ragged_offset_multiplier;
    }

    options.minRequiredEngineApiVersion = hipdnn_plugin_sdk::toEngineApiVersion(
        hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(graph->is_override_shape_enabled(),
                                                                runtimePassByValue,
                                                                ragged,
                                                                nonDefaultAlignment,
                                                                raggedOffsetMultiplier));
    return buildTritonRmsNormGraph(options);
}

const DescriptorSet& registeredSet()
{
    registerNativeIngestorSymbols();
    return loadedSet(TRITON_RMSNORM.engineName);
}

/// The engine as the provider builds it over the staged set: a state manager and the
/// generic plan builder, with the census entry's arch.
class TestTritonRmsNormEngineTier : public ::testing::Test
{
protected:
    TestTritonRmsNormEngineTier()
        : _set(registeredSet())
        , _stateManager(_set.schema,
                        _set.matchers,
                        _set.dispatches,
                        _set.packs,
                        makeKernelHeuristic(_set.heuristic),
                        _set.engine.graphMatchNativeSymbol)
        , _resolver(expectedArch())
        , _builder(_set.engine, _stateManager, _resolver)
    {
    }

    bool isApplicable(const TritonRmsNormGraphOptions& options) const
    {
        const auto buffer = stampedGraph(options);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            buffer.GetBufferPointer(), buffer.GetSize());
        const Handle handle;
        return _builder.isApplicable(handle, graph);
    }

    /// The widest `block_n` any staged kernel covers.
    int64_t widestStagedBlock() const
    {
        int64_t widest = 0;
        for(const auto& pack : _set.packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                const auto* blockN
                    = tryGetMetadataField<int64_t>(kernel.metadata, TRITON_BLOCK_N_FIELD);
                if(blockN != nullptr)
                {
                    widest = std::max(widest, *blockN);
                }
            }
        }
        return widest;
    }

private:
    const DescriptorSet& _set;
    KernelIngestorStateManager<Handle> _stateManager;
    FixedArchDeviceResolver _resolver;
    TierPlanBuilder _builder;
};

TEST_F(TestTritonRmsNormEngineTier, AcceptsTheDefaultGraphWithABakedEpsilon)
{
    EXPECT_TRUE(isApplicable({}));
}

TEST_F(TestTritonRmsNormEngineTier, AcceptsAGraphWithARuntimeEpsilon)
{
    // A runtime pass-by-value scalar raises the graph's floor to 1.2.0, which the engine's
    // schema version must reach for the matchers to see the graph at all.
    TritonRmsNormGraphOptions options;
    options.runtimeEpsilon = true;
    EXPECT_TRUE(isApplicable(options));
}

TEST_F(TestTritonRmsNormEngineTier, DeclinesAGraphDeclaringAlignment32)
{
    // The kernel matcher admits 32 on a 16-aligned kernel, but a non-default alignment
    // raises the floor to 1.3.0, above the engine's schema.
    TritonRmsNormGraphOptions options;
    options.xAlignment = 32;
    EXPECT_FALSE(isApplicable(options));
}

TEST_F(TestTritonRmsNormEngineTier, DeclinesARaggedX)
{
    TritonRmsNormGraphOptions options;
    options.xRagged = true;
    EXPECT_FALSE(isApplicable(options));
}

TEST_F(TestTritonRmsNormEngineTier, DeclinesAGraphWithOverrideShapesEnabled)
{
    // Override shapes raise the floor only to 1.1.0, so the graph reaches the matchers,
    // and the graph matcher's own refusal is what declines it.
    TritonRmsNormGraphOptions options;
    options.overrideShapeEnabled = true;
    EXPECT_FALSE(isApplicable(options));
}

TEST_F(TestTritonRmsNormEngineTier, DeclinesATrainingGraph)
{
    TritonRmsNormGraphOptions options;
    options.forwardPhase = NormFwdPhase::TRAINING;
    EXPECT_FALSE(isApplicable(options));
}

TEST_F(TestTritonRmsNormEngineTier, DeclinesARowWiderThanTheWidestStagedBlock)
{
    // The graph matches; no kernel does.
    const auto widest = widestStagedBlock();
    ASSERT_GT(widest, 0);

    TritonRmsNormGraphOptions options;
    options.xDims = {8, widest + 1};
    EXPECT_FALSE(isApplicable(options));
}

TEST_F(TestTritonRmsNormEngineTier, DeclinesAFloatGraph)
{
    TritonRmsNormGraphOptions options;
    options.dataType = DataType::FLOAT;
    EXPECT_FALSE(isApplicable(options));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
