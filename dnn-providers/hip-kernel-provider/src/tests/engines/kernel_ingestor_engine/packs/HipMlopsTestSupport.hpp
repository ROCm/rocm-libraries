// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file HipMlopsTestSupport.hpp
 * @brief What the HIP MLOps family tests share (ALMIOPEN-3066): the engines' names, their
 *        graph matches, their knobs and the ranking behind them, device buffers, and a run of
 *        a graph through the provider as a plan build and execute would, knobs pinned or not.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops
{

/// One of the HIP MLOps engines on the ingestor, one per primary operation: the name its UED
/// declares, the symbol of its graph match, and how many packs it ships.
struct MlopsEngine
{
    std::string_view name;
    std::string_view graphMatcher;
    std::size_t packCount;
};

inline constexpr MlopsEngine RESAMPLE_FWD_ENGINE{
    "hipkernel:ResampleFwd", "hipkernel.resample_fwd.graph_match", 1};
inline constexpr MlopsEngine RESAMPLE_BWD_ENGINE{
    "hipkernel:ResampleBwd", "hipkernel.resample_bwd.graph_match", 1};
inline constexpr MlopsEngine LAYERNORM_FWD_ENGINE{
    "hipkernel:LayernormFwd", "hipkernel.layernorm_fwd.graph_match", 1};
inline constexpr MlopsEngine LAYERNORM_BWD_ENGINE{
    "hipkernel:LayernormBwd", "hipkernel.layernorm_bwd.graph_match", 1};
inline constexpr MlopsEngine RMS_NORM_FWD_ENGINE{
    "hipkernel:RmsNormFwd", "hipkernel.rms_norm_fwd.graph_match", 1};
inline constexpr MlopsEngine RMS_NORM_BWD_ENGINE{
    "hipkernel:RmsNormBwd", "hipkernel.rms_norm_bwd.graph_match", 1};
/// Two packs: an inverse variance the graph supplies, or a variance and epsilon.
inline constexpr MlopsEngine BATCHNORM_INFERENCE_ENGINE{
    "hipkernel:BatchnormInference", "hipkernel.batchnorm_inference.graph_match", 2};
inline constexpr MlopsEngine BATCHNORM_TRAINING_ENGINE{
    "hipkernel:BatchnormTraining", "hipkernel.batchnorm_training.graph_match", 1};
inline constexpr MlopsEngine BATCHNORM_BACKWARD_ENGINE{
    "hipkernel:BatchnormBackward", "hipkernel.batchnorm_backward.graph_match", 1};

inline constexpr std::array<MlopsEngine, 9> ENGINES = {RESAMPLE_FWD_ENGINE,
                                                       RESAMPLE_BWD_ENGINE,
                                                       LAYERNORM_FWD_ENGINE,
                                                       LAYERNORM_BWD_ENGINE,
                                                       RMS_NORM_FWD_ENGINE,
                                                       RMS_NORM_BWD_ENGINE,
                                                       BATCHNORM_INFERENCE_ENGINE,
                                                       BATCHNORM_TRAINING_ENGINE,
                                                       BATCHNORM_BACKWARD_ENGINE};

/// The built-in engine the hand-written HIP MLOps engine registers, not built in this mode.
constexpr std::string_view BUILTIN_NAME = "HIP_MLOPS_ENGINE";
constexpr std::string_view OPERATION_TOKEN = "hip_mlops.operation";

/// @p engine's graph match on @p builder's graph.
inline std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    matchGraph(const MlopsEngine& engine, flatbuffers::FlatBufferBuilder builder)
{
    registerNativeIngestorSymbols();
    const GraphFixture fixture(std::move(builder));
    return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(engine.graphMatcher))(fixture.context());
}

/// The names of the HIP MLOps engines whose graph match accepts @p builder's graph: the one
/// engine that serves it, or none.
inline std::vector<std::string_view> acceptingEngines(flatbuffers::FlatBufferBuilder builder)
{
    registerNativeIngestorSymbols();
    const GraphFixture fixture(std::move(builder));
    std::vector<std::string_view> accepting;
    for(const auto& engine : ENGINES)
    {
        if(hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
               std::string(engine.graphMatcher))(fixture.context())
               .has_value())
        {
            accepting.push_back(engine.name);
        }
    }
    return accepting;
}

/// @p engine is loaded with its packs and offered under its own name and id, and the
/// hand-written engine's id is offered by nobody.
inline void expectOfferedUnderItsOwnNameAndId(const MlopsEngine& engine)
{
    SCOPED_TRACE(std::string(engine.name));
    const auto* set = findLoadedSet(engine.name);
    ASSERT_NE(set, nullptr) << "no descriptor set loaded";
    EXPECT_EQ(distinctPackIdCount(*set), engine.packCount);

    core::Container container;
    const auto ids = container.getEngineManager().getAllEngineIds();
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(std::string(engine.name));
    EXPECT_THAT(ids, ::testing::Contains(engineId));
    EXPECT_THAT(ids,
                ::testing::Not(::testing::Contains(
                    hipdnn_data_sdk::utilities::engineNameToId(std::string(BUILTIN_NAME)))));

    const char* advertised = nullptr;
    ASSERT_EQ(core::Container::getEngineName(engineId, &advertised), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(std::string(advertised), engine.name);
}

/// The operation a graph match bound, or nullopt when it declined.
inline std::optional<std::string>
    boundOperation(const std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>& bound)
{
    if(!bound.has_value())
    {
        return std::nullopt;
    }
    const auto entry = bound->find(std::string(OPERATION_TOKEN));
    if(entry == bound->end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&entry->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

// ---------------------------------------------------------------------------
// Knobs: one kernel per performance configuration
// ---------------------------------------------------------------------------

/// A knob's name and the value pinned, or a kernel's value, for each knob of an engine.
using KnobSettings = std::vector<std::pair<std::string, int64_t>>;

/// The symbol @p engine registers as `<its namespace>.<suffix>`, e.g. its kernel_match.
inline std::string engineSymbol(const MlopsEngine& engine, std::string_view suffix)
{
    const std::string graphMatcher(engine.graphMatcher);
    return graphMatcher.substr(0, graphMatcher.rfind('.') + 1) + std::string(suffix);
}

/// The knobs @p engine's UED lists.
inline std::vector<std::string> engineKnobs(const MlopsEngine& engine)
{
    return loadedSet(engine.name).engine.knobs;
}

/// @p engine's kernels as its matchers read them, once each (the packer copies a pack per
/// arch, each copy with the authored kernel ids).
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>
    loadedKernels(const MlopsEngine& engine)
{
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    std::set<hipdnn_plugin_sdk::ingestor::DescriptorId> seen;
    for(const auto& pack : loadedSet(engine.name).packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            if(!seen.insert(kernel.id).second)
            {
                continue;
            }
            hipdnn_plugin_sdk::ingestor::KernelDefinition definition;
            definition.kernelId = kernel.id;
            definition.packId = pack.id;
            definition.dispatchId = pack.dispatchId;
            definition.source = kernel.source;
            definition.metadata = kernel.metadata;
            definition.priority = kernel.priority;
            definition.arch = kernel.arch.empty() ? pack.arch : kernel.arch;
            definition.originDirectory = kernel.originDirectory;
            definition.name = kernel.name;
            definition.treeRoot = kernel.treeRoot;
            kernels.push_back(std::move(definition));
        }
    }
    return kernels;
}

/// @p kernel's value of each of @p knobs, in that order.
inline KnobSettings knobTuple(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                              const std::vector<std::string>& knobs)
{
    KnobSettings tuple;
    tuple.reserve(knobs.size());
    for(const auto& knob : knobs)
    {
        tuple.emplace_back(knob, kernel.getIntMetadata(knob));
    }
    return tuple;
}

/// The kernels of @p engine its kernel matcher admits for @p builder's graph on a device of
/// @p properties, best first by its scorer: the catalog its plans and knobs are drawn from,
/// in its order. Empty when the graph match declines. Expects every score distinct, so the
/// order is the scorer's alone.
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>
    rankedKernels(const MlopsEngine& engine,
                  flatbuffers::FlatBufferBuilder builder,
                  hipdnn_plugin_sdk::ingestor::DeviceProperties properties = testDeviceProperties())
{
    using namespace hipdnn_plugin_sdk::ingestor;
    registerNativeIngestorSymbols();
    const GraphFixture fixture(std::move(builder), std::move(properties));
    const auto context = fixture.context();
    const auto bound = GraphMatchRegistry::resolve(std::string(engine.graphMatcher))(context);
    if(!bound.has_value())
    {
        return {};
    }

    const auto kernelMatcher = KernelMatcherRegistry::resolve(engineSymbol(engine, "kernel_match"));
    const auto scorer = ScoreRegistry::resolve(engineSymbol(engine, "score"));
    std::vector<std::pair<double, KernelDefinition>> scored;
    for(auto& kernel : loadedKernels(engine))
    {
        if(kernelMatcher(context, *bound, kernel))
        {
            const double score = scorer(context, *bound, kernel);
            scored.emplace_back(score, std::move(kernel));
        }
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first;
    });

    std::vector<KernelDefinition> ranked;
    for(size_t i = 0; i < scored.size(); ++i)
    {
        if(i > 0)
        {
            EXPECT_NE(scored[i - 1].first, scored[i].first)
                << scored[i - 1].second.name << " and " << scored[i].second.name << " tie";
        }
        ranked.push_back(std::move(scored[i].second));
    }
    return ranked;
}

/// testDeviceProperties() with @p multiProcessorCount CUs, for the heuristics that size for
/// the device.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties
    devicePropertiesWithComputeUnits(int multiProcessorCount)
{
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = multiProcessorCount;
    return properties;
}

/// The workspace @p engine's dispatch handler reports for @p kernel on @p builder's graph on a
/// device of @p properties: what a plan of that kernel allocates.
inline size_t workspaceBytesOf(const MlopsEngine& engine,
                               flatbuffers::FlatBufferBuilder builder,
                               const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                               hipdnn_plugin_sdk::ingestor::DeviceProperties properties)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    registerNativeIngestorSymbols();
    const GraphFixture fixture(std::move(builder), std::move(properties));
    const auto context = fixture.context();
    const auto bound = GraphMatchRegistry::resolve(std::string(engine.graphMatcher))(context);
    EXPECT_TRUE(bound.has_value());
    const auto* handler = DispatchRegistry<Handle>::resolve(engineSymbol(engine, "dispatch"));
    return handler->workspaceBytes(context, bound.value_or(BoundTokens{}), kernel);
}

/// The distinct values of integer field @p knob across @p ranked, best first: the values the
/// engine advertises for the knob, its default first.
inline std::vector<int64_t>
    knobValues(const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& ranked,
               const std::string& knob)
{
    std::vector<int64_t> values;
    for(const auto& kernel : ranked)
    {
        const auto value = kernel.getIntMetadata(knob);
        if(std::find(values.begin(), values.end(), value) == values.end())
        {
            values.push_back(value);
        }
    }
    return values;
}

/// The knob tuple of the kernel a plan pinning @p pins builds: the best ranked of @p ranked
/// carrying every pinned value, or nullopt when none does.
inline std::optional<KnobSettings>
    pinnedTuple(const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& ranked,
                const std::vector<std::string>& knobs,
                const KnobSettings& pins)
{
    for(const auto& kernel : ranked)
    {
        if(std::all_of(pins.begin(), pins.end(), [&kernel](const auto& pin) {
               return kernel.getIntMetadata(pin.first) == pin.second;
           }))
        {
            return knobTuple(kernel, knobs);
        }
    }
    return std::nullopt;
}

/// @p ranked holds no two kernels with one knob tuple, so a tuple addresses one kernel.
inline void
    expectUniqueTuples(const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& ranked,
                       const std::vector<std::string>& knobs)
{
    std::set<KnobSettings> tuples;
    for(const auto& kernel : ranked)
    {
        EXPECT_TRUE(tuples.insert(knobTuple(kernel, knobs)).second)
            << kernel.name << " shares its knob tuple";
    }
}

/// An engine configuration of @p engineId pinning @p knobs.
class PinnedEngineConfig
{
public:
    PinnedEngineConfig(int64_t engineId, const KnobSettings& knobs)
        : _wrapper(build(_builder, engineId, knobs))
    {
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& get() const
    {
        return _wrapper;
    }

private:
    static hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper
        build(flatbuffers::FlatBufferBuilder& builder, int64_t engineId, const KnobSettings& knobs)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        std::vector<flatbuffers::Offset<KnobSetting>> settings;
        settings.reserve(knobs.size());
        for(const auto& [name, value] : knobs)
        {
            settings.push_back(CreateKnobSettingDirect(builder,
                                                       name.c_str(),
                                                       KnobValue::IntValue,
                                                       CreateIntValue(builder, value).Union()));
        }
        builder.Finish(CreateEngineConfig(builder, engineId, builder.CreateVector(settings)));
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
            builder.GetBufferPointer(), builder.GetSize());
    }

    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper _wrapper;
};

/// What @p engine advertises for @p builder's graph through the provider: each knob's name,
/// default and valid values, in its order.
struct AdvertisedKnob
{
    std::string name;
    int64_t defaultValue;
    std::vector<int64_t> validValues;
};

inline std::vector<AdvertisedKnob> advertisedKnobs(const MlopsEngine& engine,
                                                   const flatbuffers::FlatBufferBuilder& builder)
{
    core::Container container;
    Handle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    hipdnnPluginConstData_t details{};
    container.getEngineManager().getEngineDetails(
        handle,
        graph,
        hipdnn_data_sdk::utilities::engineNameToId(std::string(engine.name)),
        details);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    std::vector<AdvertisedKnob> knobs;
    for(const auto& name : engineKnobs(engine))
    {
        const auto& knob = wrapper.getKnobByName(name);
        const auto* valid = knob.constraintAs<hipdnn_flatbuffers_sdk::data_objects::IntConstraint>()
                                .valid_values();
        knobs.push_back(
            {name,
             knob.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>().value(),
             valid == nullptr ? std::vector<int64_t>{}
                              : std::vector<int64_t>(valid->begin(), valid->end())});
    }
    return knobs;
}

/// The kernels of @p engine admitted for @p builder's graph on this device, best first, as
/// the provider's engine ranks them: the configurations a device test can pin.
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>
    rankedKernelsOnThisDevice(const MlopsEngine& engine, flatbuffers::FlatBufferBuilder builder)
{
    return rankedKernels(engine, std::move(builder), currentDeviceProperties());
}

inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// The packed channel-last strides of @p dims, given as N, C, then the spatial dimensions:
/// NLC, NHWC or NDHWC.
inline std::vector<int64_t> channelLastStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t stride = dims[1];
    for(size_t i = dims.size(); i-- > 2;)
    {
        strides[i] = stride;
        stride *= dims[i];
    }
    strides[0] = stride;
    return strides;
}

inline size_t elementCount(const std::vector<int64_t>& dims)
{
    size_t count = 1;
    for(const auto dim : dims)
    {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

/// A device buffer of @p count elements of T, freed on scope exit. Without host values it
/// starts as 0xFF bytes, so an element the kernel never writes shows.
template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count, const std::vector<T>& host = {})
        : _count(count)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_ptr, count * sizeof(T)));
        if(!host.empty())
        {
            EXPECT_EQ(hipSuccess,
                      hipMemcpy(_ptr, host.data(), count * sizeof(T), hipMemcpyHostToDevice));
        }
        else
        {
            EXPECT_EQ(hipSuccess, hipMemset(_ptr, 0xFF, count * sizeof(T)));
        }
    }
    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_ptr));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const
    {
        return _ptr;
    }
    std::vector<T> read() const
    {
        std::vector<T> host(_count);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(host.data(), _ptr, _count * sizeof(T), hipMemcpyDeviceToHost));
        return host;
    }

private:
    void* _ptr = nullptr;
    size_t _count;
};

/// Smoothly varying values in [-2, 2]; @p phase tells one tensor's apart from another's.
inline std::vector<float> deterministicValues(size_t count, float phase)
{
    std::vector<float> values(count);
    for(size_t i = 0; i < count; ++i)
    {
        values[i] = 2.0f * std::sin(0.71f * static_cast<float>(i) + phase);
    }
    return values;
}

/// The workspace sizes a run saw: the engine's before the plan, and the plan's own.
struct EngineRun
{
    size_t maxWorkspaceBytes = 0;
    size_t workspaceBytes = 0;
};

/// Builds @p engine's plan for @p builder's graph through the provider, as a plan build
/// would, with @p knobs pinned (none: the engine's own choice), and executes it on @p buffers
/// with a workspace of the plan's size.
inline void executeThroughTheEngine(const MlopsEngine& engine,
                                    const flatbuffers::FlatBufferBuilder& builder,
                                    const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
                                    EngineRun* run = nullptr,
                                    const KnobSettings& knobs = {})
{
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(std::string(engine.name));
    core::Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto applicable = engineManager.getApplicableEngineIds(handle, graph);
    ASSERT_THAT(applicable, ::testing::Contains(engineId));

    hipdnn_test_sdk::utilities::MockEngineConfig unpinned;
    EXPECT_CALL(unpinned, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(unpinned, engineId()).WillRepeatedly(::testing::Return(engineId));
    const PinnedEngineConfig pinned(engineId, knobs);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig
        = knobs.empty()
              ? static_cast<const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&>(
                    unpinned)
              : pinned.get();

    EngineRun sizes;
    ASSERT_NO_THROW(sizes.maxWorkspaceBytes
                    = engineManager.getMaxWorkspaceSize(handle, graph, engineConfig));

    Context context;
    ASSERT_NO_THROW(engineManager.initializeExecutionContext(handle, graph, engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());
    sizes.workspaceBytes = context.plan().getWorkspaceSize(handle);

    std::optional<DeviceBuffer<std::byte>> workspace;
    if(sizes.workspaceBytes > 0)
    {
        workspace.emplace(sizes.workspaceBytes);
    }
    ASSERT_NO_THROW(context.plan().execute(handle,
                                           buffers.data(),
                                           static_cast<uint32_t>(buffers.size()),
                                           workspace.has_value() ? workspace->get() : nullptr));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));

    if(run != nullptr)
    {
        *run = sizes;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops
