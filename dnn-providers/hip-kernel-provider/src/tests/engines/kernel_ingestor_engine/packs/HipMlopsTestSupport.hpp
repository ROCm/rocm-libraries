// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file HipMlopsTestSupport.hpp
 * @brief What the HIP MLOps family tests share (ALMIOPEN-3066): the engines' names, their
 *        graph matches, device buffers, and a run of a graph through the provider as a plan
 *        build and execute would.
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

inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
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
/// would, and executes it on @p buffers with a workspace of the plan's size.
inline void executeThroughTheEngine(const MlopsEngine& engine,
                                    const flatbuffers::FlatBufferBuilder& builder,
                                    const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
                                    EngineRun* run = nullptr)
{
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(std::string(engine.name));
    core::Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto applicable = engineManager.getApplicableEngineIds(handle, graph);
    ASSERT_THAT(applicable, ::testing::Contains(engineId));

    hipdnn_test_sdk::utilities::MockEngineConfig engineConfig;
    EXPECT_CALL(engineConfig, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(engineConfig, engineId()).WillRepeatedly(::testing::Return(engineId));

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
