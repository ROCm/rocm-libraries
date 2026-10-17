// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Mock-backend fixture for the ported v0.x engine, plan and autotune
// builders: a conv graph over engines whose dense index equals their ID, and
// helpers to build configs, plans and variant packs through the builders.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Engine.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_VariantPack.h>

#include "ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;

struct BuilderFailure
{
    bool thrown = false;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    std::string what;
};

template <typename Action>
BuilderFailure builderFailureOf(Action&& action)
{
    BuilderFailure failure;
    try
    {
        std::forward<Action>(action)();
    }
    catch(v8fe::cudnnException& e)
    {
        failure.thrown = true;
        failure.status = e.getCudnnStatus();
        failure.what = e.what();
    }
    return failure;
}

inline std::string lastShimError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class ShimV8PlanFixture : public ShimV8MockBackendFixture
{
protected:
    using Dims = std::vector<std::int64_t>;

    static constexpr std::int64_t WORKSPACE_SIZE = 4096;
    static constexpr std::int64_t X_UID = 1;
    static constexpr std::int64_t W_UID = 2;
    static constexpr std::int64_t Y_UID = 3;

    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        lastShimError();
        installPlanWorkspace(WORKSPACE_SIZE);
        installEngines(3);
    }

    // Engines 0..count-1, ranked in ID order, so index i names engine i.
    void installEngines(std::int64_t count)
    {
        std::vector<std::int64_t> ids(static_cast<std::size_t>(count));
        std::iota(ids.begin(), ids.end(), std::int64_t{0});
        installHeuristicResults(std::move(ids));
    }

    // Dense index of `id` among `ids` (the shim numbers engines in ID order).
    static std::int64_t denseIndex(std::vector<std::int64_t> ids, std::int64_t id)
    {
        std::sort(ids.begin(), ids.end());
        return static_cast<std::int64_t>(
            std::distance(ids.begin(), std::lower_bound(ids.begin(), ids.end(), id)));
    }

    static Dims packedStrides(const Dims& dims)
    {
        Dims strides(dims.size(), 1);
        for(std::size_t i = dims.size() - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * dims[i];
        }
        return strides;
    }

    static v8fe::Tensor_v8 tensor(std::int64_t uid, const Dims& dims)
    {
        const Dims strides = packedStrides(dims);
        return v8fe::TensorBuilder_v8()
            .setDataType(v8fe::DataType_t::FLOAT)
            .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .build();
    }

    static v8fe::Operation_v8 convForward(const Dims& xDims, const Dims& wDims, const Dims& yDims)
    {
        const std::array<std::int64_t, 2> pads = {1, 1};
        const std::array<std::int64_t, 2> ones = {1, 1};
        const v8fe::ConvDesc_v8 conv = v8fe::ConvDescBuilder_v8()
                                           .setComputeType(v8fe::DataType_t::FLOAT)
                                           .setMathMode(CUDNN_CROSS_CORRELATION)
                                           .setSpatialDimCount(2)
                                           .setPrePadding(2, pads.data())
                                           .setPostPadding(2, pads.data())
                                           .setDilation(2, ones.data())
                                           .setSpatialStride(2, ones.data())
                                           .build();
        return v8fe::OperationBuilder_v8(
                   v8fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(tensor(X_UID, xDims))
            .setwDesc(tensor(W_UID, wDims))
            .setyDesc(tensor(Y_UID, yDims))
            .setcDesc(conv)
            .build();
    }

    v8fe::OperationGraph_v8 graphOf(const v8fe::Operation_v8& op)
    {
        const v8fe::Operation_v8* ops = &op;
        return v8fe::OperationGraphBuilder_v8()
            .setHandle(_handle)
            .setOperationGraph(1, &ops)
            .build();
    }

    // conv(x {2,4,8,8}, w {8,4,3,3}) -> y {2,8,8,8}; tag "ConvFwd_".
    v8fe::OperationGraph_v8 convGraph(std::int64_t channels = 4)
    {
        return graphOf(convForward({2, channels, 8, 8}, {8, channels, 3, 3}, {2, 8, 8, 8}));
    }

    static v8fe::EngineConfigList engineConfigs(v8fe::OperationGraph_v8& graph)
    {
        v8fe::EngineHeuristics_v8 heuristics = v8fe::EngineHeuristicsBuilder_v8()
                                                   .setOperationGraph(graph)
                                                   .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                                   .build();
        return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    }

    static v8fe::Engine_v8 engine(const v8fe::OperationGraph_v8& graph, std::int64_t index)
    {
        return v8fe::EngineBuilder_v8().setOperationGraph(graph).setGlobalEngineIdx(index).build();
    }

    static v8fe::ExecutionPlan_v8 planFor(const v8fe::ManagedOpaqueDescriptor& config,
                                          const std::string& tag)
    {
        return v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config, tag).build();
    }

    // Built through EngineBuilder and EngineConfigBuilder, as legacy samples do.
    static v8fe::ExecutionPlan_v8 planFor(const v8fe::OperationGraph_v8& graph, std::int64_t index)
    {
        const v8fe::Engine_v8 built = engine(graph, index);
        const v8fe::EngineConfig_v8 config
            = v8fe::EngineConfigBuilder_v8().setEngine(built).build();
        return v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    }

    // Engine index read back through ENGINECFG_ENGINE and ENGINE_GLOBAL_INDEX.
    static std::int64_t indexOf(cudnnBackendDescriptor_t config)
    {
        cudnnBackendDescriptor_t engineDescriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engineDescriptor),
                  CUDNN_STATUS_SUCCESS);
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(config,
                                           CUDNN_ATTR_ENGINECFG_ENGINE,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &count,
                                           static_cast<void*>(&engineDescriptor)),
                  CUDNN_STATUS_SUCCESS)
            << lastShimError();
        std::int64_t index = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(engineDescriptor,
                                           CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &count,
                                           static_cast<void*>(&index)),
                  CUDNN_STATUS_SUCCESS)
            << lastShimError();
        EXPECT_EQ(cudnnBackendDestroyDescriptor(engineDescriptor), CUDNN_STATUS_SUCCESS);
        return index;
    }

    static std::vector<std::int64_t> indicesOf(const v8fe::EngineConfigList& configs)
    {
        std::vector<std::int64_t> indices;
        indices.reserve(configs.size());
        for(const v8fe::ManagedOpaqueDescriptor& config : configs)
        {
            indices.push_back(indexOf(config->get_backend_descriptor()));
        }
        return indices;
    }

    static std::vector<std::string> tagsOf(const std::vector<v8fe::ExecutionPlan_v8>& plans)
    {
        std::vector<std::string> tags;
        tags.reserve(plans.size());
        for(const v8fe::ExecutionPlan_v8& plan : plans)
        {
            tags.push_back(plan.getTag());
        }
        return tags;
    }

    v8fe::VariantPack_v8 variantPack()
    {
        std::array<void*, 3> pointers = {&_x, &_w, &_y};
        const std::array<std::int64_t, 3> uids = {X_UID, W_UID, Y_UID};
        return v8fe::VariantPackBuilder_v8()
            .setWorkspacePointer(_workspace.data())
            .setDataPointers(static_cast<std::int64_t>(pointers.size()), pointers.data())
            .setUids(static_cast<std::int64_t>(uids.size()), uids.data())
            .build();
    }

private:
    float _x = 0.0F;
    float _w = 0.0F;
    float _y = 0.0F;
    std::array<std::byte, WORKSPACE_SIZE> _workspace{};
};

} // namespace hipdnn_shim_test
