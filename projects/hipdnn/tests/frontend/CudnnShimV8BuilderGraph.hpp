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
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hipdnn_shim_v8_test
{

namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;

using Dims = std::vector<std::int64_t>;

inline const Dims X_DIMS = {2, 4, 8, 8};
inline const Dims W_DIMS = {8, 4, 3, 3};
inline const Dims Y_DIMS = {2, 8, 8, 8};
inline const Dims CHANNEL_DIMS = {1, 4, 1, 1};
inline const Dims SCALAR_DIMS = {1, 1, 1, 1};
// Per-sample statistics of a norm over C, H and W.
inline const Dims SAMPLE_DIMS = {2, 1, 1, 1};
inline const Dims FEATURE_DIMS = {1, 4, 8, 8};
inline const Dims POOLED_DIMS = {2, 4, 4, 4};

inline std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

inline Dims packedStrides(const Dims& dims)
{
    return hipdnn_data_sdk::utilities::generateStrides(dims);
}

// Bytes of a dense FLOAT tensor of `dims`.
inline std::size_t floatTensorBytes(const Dims& dims)
{
    return sizeof(float)
           * static_cast<std::size_t>(
               std::accumulate(dims.begin(), dims.end(), std::int64_t{1}, std::multiplies<>()));
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

// Zero bytes allocate nothing: get() is null, as for a plan without workspace.
using DeviceBuffer = hipdnn_data_sdk::utilities::Workspace<>;

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

struct TensorOptions
{
    v8fe::DataType_t dataType = v8fe::DataType_t::FLOAT;
    bool isVirtual = false;
    bool isByValue = false;
};

inline v8fe::Tensor_v8 tensor(std::int64_t uid, const Dims& dims, const TensorOptions& options = {})
{
    const Dims strides = packedStrides(dims);
    return v8fe::TensorBuilder_v8()
        .setDataType(options.dataType)
        .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
        .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .setVirtual(options.isVirtual)
        .setByValue(options.isByValue)
        .build();
}

inline v8fe::Tensor_v8 virtualTensor(std::int64_t uid, const Dims& dims)
{
    TensorOptions options;
    options.isVirtual = true;
    return tensor(uid, dims, options);
}

inline v8fe::Tensor_v8 scalar(std::int64_t uid)
{
    TensorOptions options;
    options.isByValue = true;
    return tensor(uid, SCALAR_DIMS, options);
}

inline v8fe::ConvDesc_v8 convDesc()
{
    const std::array<std::int64_t, 2> ones = {1, 1};
    return v8fe::ConvDescBuilder_v8()
        .setComputeType(v8fe::DataType_t::FLOAT)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setSpatialDimCount(2)
        .setPrePadding(2, ones.data())
        .setPostPadding(2, ones.data())
        .setDilation(2, ones.data())
        .setSpatialStride(2, ones.data())
        .build();
}

inline v8fe::Operation_v8
    convForward(const v8fe::Tensor_v8& x, const v8fe::Tensor_v8& w, const v8fe::Tensor_v8& y)
{
    return v8fe::OperationBuilder_v8(
               v8fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .setxDesc(x)
        .setwDesc(w)
        .setyDesc(y)
        .setcDesc(convDesc())
        .build();
}

inline v8fe::PointWiseDesc_v8 pointwiseDesc(v8fe::PointwiseMode_t mode)
{
    return v8fe::PointWiseDescBuilder_v8()
        .setMode(mode)
        .setComputeType(v8fe::DataType_t::FLOAT)
        .build();
}

inline v8fe::Operation_v8 pointwise(v8fe::PointwiseMode_t mode,
                                    const v8fe::Tensor_v8& x,
                                    const v8fe::Tensor_v8* b,
                                    const v8fe::Tensor_v8& y)
{
    v8fe::OperationBuilder_v8 builder(v8fe::DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
    builder.setpwDesc(pointwiseDesc(mode)).setxDesc(x).setyDesc(y);
    if(b != nullptr)
    {
        builder.setbDesc(*b);
    }
    return builder.build();
}

// 2x2 window and stride, no padding; max pooling pads with -inf.
inline v8fe::ResampleDesc_v8 resampleDesc(cudnnResampleMode_t mode)
{
    const std::array<std::int64_t, 2> window = {2, 2};
    const std::array<std::int64_t, 2> zeros = {0, 0};
    return v8fe::ResampleDescBuilder_v8()
        .setResampleMode(mode)
        .setComputeType(v8fe::DataType_t::FLOAT)
        .setPaddingMode(mode == CUDNN_RESAMPLE_MAXPOOL ? CUDNN_NEG_INF_PAD : CUDNN_ZERO_PAD)
        .setSpatialDim(2, window.data())
        .setPrePadding(2, zeros.data())
        .setPostPadding(2, zeros.data())
        .setSpatialStride(2, window.data())
        .build();
}

// Lowers graphs against the real backend with the good test plugin. Graph lowering
// needs no device, so suites on this fixture do not skip on device-less hosts.
class DevicelessGraphFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(cudnnCreate(&_handle), CUDNN_STATUS_SUCCESS);
        lastError();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_handle), CUDNN_STATUS_SUCCESS);
        }
    }

    void expectGraphBuilds(const std::vector<v8fe::Operation_v8>& ops)
    {
        try
        {
            v8fe::OperationGraphBuilder_v8 builder;
            builder.setHandle(_handle).setOperationGraph(ops);
            const v8fe::OperationGraph_v8 graph = builder.build();
            EXPECT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
        }
        catch(const v8fe::cudnnException& e)
        {
            ADD_FAILURE() << e.what();
        }
    }

    cudnnHandle_t _handle = nullptr;
};

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

// The UIDs of the X_DIMS relu graph ReluGraphFixture builds.
constexpr std::int64_t RELU_X_UID = 1;
constexpr std::int64_t RELU_Y_UID = 2;

// IntegrationTestFixture with y = relu(x) built on its handle.
class ReluGraphFixture : public hipdnn_tests::IntegrationTestFixture
{
protected:
    void SetUp() override
    {
        IntegrationTestFixture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        _graph.emplace(reluGraph(_handle, RELU_X_UID, RELU_Y_UID, X_DIMS));
    }

    // The graph owns plugin state, so it goes before the handle that loaded the plugin.
    void TearDown() override
    {
        _graph.reset();
        IntegrationTestFixture::TearDown();
    }

    std::optional<v8fe::OperationGraph_v8> _graph;
};

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
