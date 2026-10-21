// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builds cuDNN v8 graphs, configs and plans with the v0.x builders, for the
// integration suites that run them against the real hipDNN backend.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_VariantPack.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hipdnn_shim_v8_test
{

namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;

inline std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

inline std::vector<std::int64_t> packedStrides(const std::vector<std::int64_t>& dims)
{
    std::vector<std::int64_t> strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// The decimal engine id a plan's JSON representation carries in "engine":{"id":"..."}.
inline std::string jsonEngineId(const std::string& json)
{
    const std::string key = "\"id\"";
    const auto keyPos = json.find(key);
    if(keyPos == std::string::npos)
    {
        return {};
    }
    const auto open = json.find('"', json.find(':', keyPos + key.size()));
    if(open == std::string::npos)
    {
        return {};
    }
    const auto close = json.find('"', open + 1);
    if(close == std::string::npos)
    {
        return {};
    }
    return json.substr(open + 1, close - open - 1);
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t bytes)
    {
        EXPECT_EQ(hipMalloc(&_ptr, std::max<std::size_t>(bytes, 1)), hipSuccess);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    ~DeviceBuffer()
    {
        if(_ptr != nullptr)
        {
            static_cast<void>(hipFree(_ptr));
        }
    }

    void* get() const
    {
        return _ptr;
    }

private:
    void* _ptr = nullptr;
};

// The status a builder call threw, or SUCCESS when it did not throw.
template <typename Action>
cudnnStatus_t statusOf(Action&& action)
{
    try
    {
        std::forward<Action>(action)();
    }
    catch(v8fe::cudnnException& e)
    {
        return e.getCudnnStatus();
    }
    return CUDNN_STATUS_SUCCESS;
}

inline v8fe::Tensor_v8 tensor(std::int64_t uid, const std::vector<std::int64_t>& dims)
{
    const std::vector<std::int64_t> strides = packedStrides(dims);
    return v8fe::TensorBuilder_v8()
        .setDataType(v8fe::DataType_t::FLOAT)
        .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
        .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .build();
}

// y = relu(x) on `handle`.
inline v8fe::OperationGraph_v8 reluGraph(cudnnHandle_t handle,
                                         std::int64_t xUid,
                                         std::int64_t yUid,
                                         const std::vector<std::int64_t>& dims)
{
    const v8fe::PointWiseDesc_v8 relu = v8fe::PointWiseDescBuilder_v8()
                                            .setMode(v8fe::PointwiseMode_t::RELU_FWD)
                                            .setComputeType(v8fe::DataType_t::FLOAT)
                                            .build();
    const v8fe::Operation_v8 op
        = v8fe::OperationBuilder_v8(v8fe::DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
              .setpwDesc(relu)
              .setxDesc(tensor(xUid, dims))
              .setyDesc(tensor(yUid, dims))
              .build();
    std::array<const v8fe::Operation_v8*, 1> ops = {&op};
    return v8fe::OperationGraphBuilder_v8()
        .setHandle(handle)
        .setOperationGraph(static_cast<std::int64_t>(ops.size()), ops.data())
        .build();
}

// Every engine config the heuristics return, in rank order.
inline v8fe::EngineConfigList heuristicConfigs(v8fe::OperationGraph_v8& graph,
                                               cudnnBackendHeurMode_t mode
                                               = CUDNN_HEUR_MODE_FALLBACK)
{
    v8fe::EngineHeuristics_v8 heuristics
        = v8fe::EngineHeuristicsBuilder_v8().setOperationGraph(graph).setHeurMode(mode).build();
    return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
}

inline v8fe::Engine_v8 engine(const v8fe::OperationGraph_v8& graph, std::int64_t index)
{
    return v8fe::EngineBuilder_v8().setOperationGraph(graph).setGlobalEngineIdx(index).build();
}

// Built through EngineConfigBuilder, as legacy samples do.
inline v8fe::ExecutionPlan_v8 planFor(const v8fe::Engine_v8& built)
{
    const v8fe::EngineConfig_v8 config = v8fe::EngineConfigBuilder_v8().setEngine(built).build();
    return v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
}

inline v8fe::ExecutionPlan_v8 planFor(const v8fe::OperationGraph_v8& graph, std::int64_t index)
{
    return planFor(engine(graph, index));
}

// The dense engine index a plan's tag ("<graph tag>eng<index>[_k...]") carries.
inline std::int64_t tagEngineIndex(const std::string& tag)
{
    const auto at = tag.rfind("eng");
    return at == std::string::npos ? -1 : std::stoll(tag.substr(at + 3));
}

// Appends a knob of `type` the engine did not report, as a caller can through
// getSupportedKnobs(). EngineConfigBuilder sends it only if a reported knob is set.
inline void appendKnob(v8fe::Engine_v8& built, cudnnBackendKnobType_t type, std::int64_t choice)
{
    using Knob = std::remove_reference_t<decltype(built.getSupportedKnobs().front())>;
    Knob knob(type, 64, 0, 1);
    knob.setChoice(static_cast<std::uint64_t>(choice));
    built.getSupportedKnobs().push_back(knob);
}

inline v8fe::VariantPack_v8
    variantPack(const std::vector<std::int64_t>& uids, std::vector<void*> pointers, void* workspace)
{
    return v8fe::VariantPackBuilder_v8()
        .setWorkspacePointer(workspace)
        .setDataPointers(static_cast<std::int64_t>(pointers.size()), pointers.data())
        .setUids(static_cast<std::int64_t>(uids.size()), uids.data())
        .build();
}

} // namespace hipdnn_shim_v8_test
