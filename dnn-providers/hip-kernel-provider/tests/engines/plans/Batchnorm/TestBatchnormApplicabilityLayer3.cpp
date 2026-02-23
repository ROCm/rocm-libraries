// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "TestBatchnormApplicability.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_plugin;
using hipdnn_data_sdk::utilities::TensorLayout;

namespace
{

// --- Tensor UID Constants ---

// Common UIDs for all batchnorm operations
struct BnCommonTensorIds
{
    [[maybe_unused]] static constexpr int64_t X = 1;
    [[maybe_unused]] static constexpr int64_t Y = 2;
    [[maybe_unused]] static constexpr int64_t SCALE = 3;
    [[maybe_unused]] static constexpr int64_t BIAS = 4;
    [[maybe_unused]] static constexpr int64_t EPSILON = 5;
    [[maybe_unused]] static constexpr int64_t MEAN = 6;
    [[maybe_unused]] static constexpr int64_t INV_VARIANCE = 7;
};

// Backward-specific
struct BnBackwardTensorIds : BnCommonTensorIds
{
    [[maybe_unused]] static constexpr int64_t DY = 8;
    [[maybe_unused]] static constexpr int64_t DX = 9;
    [[maybe_unused]] static constexpr int64_t DSCALE = 10;
    [[maybe_unused]] static constexpr int64_t DBIAS = 11;
};

// Fused operations (virtual tensors)
struct BnFusedTensorIds : BnBackwardTensorIds
{
    [[maybe_unused]] static constexpr int64_t BN_Y_VIRTUAL = 12;
    [[maybe_unused]] static constexpr int64_t DX_DRELU_VIRTUAL = 13;
};

// Training-specific
struct BnTrainingTensorIds : BnCommonTensorIds
{
    [[maybe_unused]] static constexpr int64_t PREV_RUNNING_MEAN = 8;
    [[maybe_unused]] static constexpr int64_t PREV_RUNNING_VARIANCE = 9;
    [[maybe_unused]] static constexpr int64_t MOMENTUM = 10;
    [[maybe_unused]] static constexpr int64_t NEXT_RUNNING_MEAN = 11;
    [[maybe_unused]] static constexpr int64_t NEXT_RUNNING_VARIANCE = 12;
};

// Inference with variance extension (standalone - uses variance instead of inv_variance, plus epsilon)
struct BnInferenceVarianceExtTensorIds
{
    [[maybe_unused]] static constexpr int64_t X = 1;
    [[maybe_unused]] static constexpr int64_t Y = 2;
    [[maybe_unused]] static constexpr int64_t SCALE = 3;
    [[maybe_unused]] static constexpr int64_t BIAS = 4;
    [[maybe_unused]] static constexpr int64_t EPSILON = 5;
    [[maybe_unused]] static constexpr int64_t MEAN = 6;
    [[maybe_unused]] static constexpr int64_t VARIANCE = 7;
};

[[maybe_unused]] inline TensorConfig createAffineTensor(int64_t uid,
                                                        const std::string& name,
                                                        hipdnn_data_sdk::data_objects::DataType dt,
                                                        const std::vector<int64_t>& derivedDims,
                                                        const std::vector<int64_t>& derivedStrides)
{
    return TensorConfigBuilder(uid, name, TensorRole::AFFINE)
        .withDataType(dt)
        .withDims(derivedDims)
        .withStrides(derivedStrides)
        .build();
}

[[maybe_unused]] inline TensorConfig createStatTensor(int64_t uid,
                                                      const std::string& name,
                                                      hipdnn_data_sdk::data_objects::DataType dt,
                                                      const std::vector<int64_t>& derivedDims,
                                                      const std::vector<int64_t>& derivedStrides)
{
    return TensorConfigBuilder(uid, name, TensorRole::STAT)
        .withDataType(dt)
        .withDims(derivedDims)
        .withStrides(derivedStrides)
        .build();
}

[[maybe_unused]] inline TensorConfig
    createScalarTensor(int64_t uid, const std::string& name, double value)
{
    return TensorConfigBuilder(uid, name, TensorRole::SCALAR).asScalar(value).build();
}

// --- Reusable Tensor Set Factories ---

inline std::vector<TensorConfig> createIoTensorPair(const BnTensorTypes& types,
                                                    const std::vector<int64_t>& dims,
                                                    const TensorLayout& layout,
                                                    int64_t xUid = BnCommonTensorIds::X,
                                                    int64_t yUid = BnCommonTensorIds::Y)
{
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);
    return {createIoTensor(xUid, "x", types.io, dims, strides),
            createIoTensor(yUid, "y", types.io, dims, strides)};
}

inline std::vector<TensorConfig> createAffineTensorPair(const BnTensorTypes& types,
                                                        const std::vector<int64_t>& baseDims,
                                                        const TensorLayout& layout,
                                                        int64_t scaleUid = BnCommonTensorIds::SCALE,
                                                        int64_t biasUid = BnCommonTensorIds::BIAS)
{
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(baseDims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    return {createAffineTensor(scaleUid, "scale", types.affine, derivedDims, derivedStrides),
            createAffineTensor(biasUid, "bias", types.affine, derivedDims, derivedStrides)};
}

inline std::vector<TensorConfig> createStatTensorPair(const BnTensorTypes& types,
                                                      const std::vector<int64_t>& baseDims,
                                                      const TensorLayout& layout,
                                                      int64_t meanUid = BnCommonTensorIds::MEAN,
                                                      int64_t invVarianceUid
                                                      = BnCommonTensorIds::INV_VARIANCE)
{
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(baseDims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    return {
        createStatTensor(meanUid, "mean", types.stat, derivedDims, derivedStrides),
        createStatTensor(invVarianceUid, "inv_variance", types.stat, derivedDims, derivedStrides)};
}

inline std::vector<TensorConfig> createBatchnormInferenceTensors(const BnTensorTypes& types,
                                                                 const std::vector<int64_t>& dims,
                                                                 const TensorLayout& layout)
{
    using UIDs = BnCommonTensorIds;
    std::vector<TensorConfig> configs;

    auto io = createIoTensorPair(types, dims, layout, UIDs::X, UIDs::Y);
    configs.insert(configs.end(), io.begin(), io.end());

    auto affine = createAffineTensorPair(types, dims, layout, UIDs::SCALE, UIDs::BIAS);
    configs.insert(configs.end(), affine.begin(), affine.end());

    auto stat = createStatTensorPair(types, dims, layout, UIDs::MEAN, UIDs::INV_VARIANCE);
    configs.insert(configs.end(), stat.begin(), stat.end());

    return configs;
}

inline std::vector<TensorConfig> createBatchnormTrainingTensors(const BnTensorTypes& types,
                                                                const std::vector<int64_t>& dims,
                                                                const TensorLayout& layout,
                                                                bool includeMeanOutput = false,
                                                                bool includeInvVarianceOutput
                                                                = false)
{
    using UIDs = BnTrainingTensorIds;
    std::vector<TensorConfig> configs;

    auto io = createIoTensorPair(types, dims, layout, UIDs::X, UIDs::Y);
    configs.insert(configs.end(), io.begin(), io.end());

    auto affine = createAffineTensorPair(types, dims, layout, UIDs::SCALE, UIDs::BIAS);
    configs.insert(configs.end(), affine.begin(), affine.end());

    configs.push_back(createScalarTensor(UIDs::EPSILON, "epsilon", 1e-5));

    if(includeMeanOutput || includeInvVarianceOutput)
    {
        auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
        auto derivedStrides
            = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);

        if(includeMeanOutput)
        {
            configs.push_back(
                createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
        }
        if(includeInvVarianceOutput)
        {
            configs.push_back(createStatTensor(
                UIDs::INV_VARIANCE, "inv_variance", types.stat, derivedDims, derivedStrides));
        }
    }

    return configs;
}

inline std::vector<TensorConfig> createBatchnormBackwardTensors(const BnTensorTypes& types,
                                                                const std::vector<int64_t>& dims,
                                                                const TensorLayout& layout,
                                                                bool includeMeanInput = false,
                                                                bool includeInvVarianceInput
                                                                = false)
{
    using UIDs = BnBackwardTensorIds;
    std::vector<TensorConfig> configs;
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    configs.push_back(createIoTensor(UIDs::X, "x", types.io, dims, strides));
    configs.push_back(createIoTensor(UIDs::DY, "dy", types.io, dims, strides));
    configs.push_back(createIoTensor(UIDs::DX, "dx", types.io, dims, strides));

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    configs.push_back(
        createAffineTensor(UIDs::SCALE, "scale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::DSCALE, "dscale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::DBIAS, "dbias", types.affine, derivedDims, derivedStrides));

    if(includeMeanInput)
    {
        configs.push_back(
            createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
    }
    if(includeInvVarianceInput)
    {
        configs.push_back(createStatTensor(
            UIDs::INV_VARIANCE, "inv_variance", types.stat, derivedDims, derivedStrides));
    }

    return configs;
}

inline std::vector<TensorConfig> createBatchnormFusedBackwardTensors(
    const BnTensorTypes& types, const std::vector<int64_t>& dims, const TensorLayout& layout)
{
    using UIDs = BnFusedTensorIds;
    std::vector<TensorConfig> configs;
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);

    // Forward inputs (x, scale, bias, mean, invVariance)
    configs.push_back(createIoTensor(UIDs::X, "x", types.io, dims, strides));
    configs.push_back(
        createAffineTensor(UIDs::SCALE, "scale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::BIAS, "bias", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
    configs.push_back(createStatTensor(
        UIDs::INV_VARIANCE, "inv_variance", types.stat, derivedDims, derivedStrides));

    // Backward inputs (dy)
    configs.push_back(createIoTensor(UIDs::DY, "dy", types.io, dims, strides));

    // Backward outputs (dx, dscale, dbias)
    configs.push_back(createIoTensor(UIDs::DX, "dx", types.io, dims, strides));
    configs.push_back(
        createAffineTensor(UIDs::DSCALE, "dscale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::DBIAS, "dbias", types.affine, derivedDims, derivedStrides));

    // Virtual tensors (BN_Y, DX_drelu)
    configs.push_back(
        createIoTensor(UIDs::BN_Y_VIRTUAL, "BN_Y", types.intermediate, dims, strides, true));
    configs.push_back(createIoTensor(
        UIDs::DX_DRELU_VIRTUAL, "DX_drelu", types.intermediate, dims, strides, true));

    return configs;
}

// --- BnTensorTypes Helpers ---

inline std::string dataTypeToString(hipdnn_data_sdk::data_objects::DataType dt)
{
    using DT = hipdnn_data_sdk::data_objects::DataType;
    switch(dt)
    {
    case DT::FLOAT:
        return "Float";
    case DT::HALF:
        return "Half";
    case DT::BFLOAT16:
        return "Bfloat16";
    default:
        return "Unknown";
    }
}

inline std::string toString(const BnTensorTypes& types)
{
    return "IO_" + dataTypeToString(types.io) + "_Affine_" + dataTypeToString(types.affine)
           + "_Stat_" + dataTypeToString(types.stat);
}

inline std::string activationModeToString(hipdnn_data_sdk::data_objects::PointwiseMode mode)
{
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;
    switch(mode)
    {
    case PM::IDENTITY:
        return "Identity";
    case PM::RELU_BWD:
        return "ReluBwd";
    case PM::SIGMOID_BWD:
        return "SigmoidBwd";
    case PM::TANH_BWD:
        return "TanhBwd";
    case PM::RELU_FWD:
        return "ReluFwd";
    default:
        return "Unknown";
    }
}

// --- Test Case Structs: Layer 3 (High-Level Validators) ---

struct BatchnormInferenceConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t yUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t meanUid;
    int64_t invVarianceUid;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormInferenceConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormTrainingConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t yUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t epsilonUid;
    flatbuffers::Optional<int64_t> meanUid;
    flatbuffers::Optional<int64_t> invVarianceUid;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormTrainingConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormBackwardConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t dyUid;
    int64_t dxUid;
    int64_t scaleUid;
    int64_t dscaleUid;
    int64_t dbiasUid;
    flatbuffers::Optional<int64_t> meanUid;
    flatbuffers::Optional<int64_t> invVarianceUid;

    friend std::ostream& operator<<(std::ostream& os, const BatchnormBackwardConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormFusedBackwardConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t meanUid;
    int64_t invVarianceUid;
    int64_t dyUid;
    int64_t dxUid;
    int64_t dscaleUid;
    int64_t dbiasUid;
    int64_t bnYVirtualUid;
    int64_t dxDreluVirtualUid;
    hipdnn_data_sdk::data_objects::PointwiseMode activationMode;

    friend std::ostream& operator<<(std::ostream& os,
                                    const BatchnormFusedBackwardConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct BatchnormInferenceVarianceExtFusedConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t yUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t epsilonUid;
    int64_t meanUid;
    int64_t varianceUid;
    hipdnn_data_sdk::data_objects::PointwiseMode activationMode;

    friend std::ostream& operator<<(std::ostream& os,
                                    const BatchnormInferenceVarianceExtFusedConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

struct ActivationModeTestCase
{
    std::string name;
    bool shouldPass;
    hipdnn_data_sdk::data_objects::PointwiseMode mode;
    flatbuffers::Optional<double> reluLowerClip;
    flatbuffers::Optional<double> reluUpperClip;
    flatbuffers::Optional<double> reluLowerClipSlope;

    friend std::ostream& operator<<(std::ostream& os, const ActivationModeTestCase& tc)
    {
        return os << tc.name;
    }
};
#if 0 //EAN
// --- Helper Functions ---

std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*> buildTensorMap(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>&
        tensorOffsets)
{
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        nullptr);

    builder.Finish(graphOffset);

    const auto* graph = hipdnn_data_sdk::data_objects::GetGraph(builder.GetBufferPointer());
    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*> tensorMap;

    if(graph->tensors() != nullptr)
    {
        for(const auto* tensorAttr : *graph->tensors())
        {
            tensorMap[tensorAttr->uid()] = tensorAttr;
        }
    }

    return tensorMap;
}

auto buildTensorMapFromConfigs(const std::vector<TensorConfig>& configs)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());

    for(const auto& config : configs)
    {
        if(config.isPassByValue)
        {
            // Create a pass-by-value tensor with embedded scalar value
            hipdnn_data_sdk::data_objects::Float64Value floatValue(config.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                config.uid,
                config.name.c_str(),
                config.dataType,
                &config.strides,
                &config.dims,
                config.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            config.uid,
                                                                            config.name.c_str(),
                                                                            config.dataType,
                                                                            &config.strides,
                                                                            &config.dims,
                                                                            config.isVirtual));
        }
    }

    auto tensorMap = buildTensorMap(builder, tensorOffsets);
    return std::make_pair(std::move(builder), std::move(tensorMap));
}
#endif

// --- Graph Builders ---

inline flatbuffers::FlatBufferBuilder
    buildBatchnormInferenceGraph(const std::vector<TensorConfig>& configs,
                                 int64_t xUid,
                                 int64_t yUid,
                                 int64_t scaleUid,
                                 int64_t biasUid,
                                 int64_t meanUid,
                                 int64_t invVarianceUid)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets from configs
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create BatchnormInferenceAttributes with specified UIDs
    auto bnInfAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, xUid, meanUid, invVarianceUid, scaleUid, biasUid, yUid);

    // Create node
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormTrainingGraph(const std::vector<TensorConfig>& configs,
                                int64_t xUid,
                                int64_t yUid,
                                int64_t scaleUid,
                                int64_t biasUid,
                                int64_t epsilonUid,
                                flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt,
                                flatbuffers::Optional<int64_t> invVarianceUid
                                = flatbuffers::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        if(cfg.isPassByValue)
        {
            // Create a pass-by-value tensor with embedded scalar value
            hipdnn_data_sdk::data_objects::Float64Value floatValue(cfg.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            cfg.uid,
                                                                            cfg.name.c_str(),
                                                                            cfg.dataType,
                                                                            &cfg.strides,
                                                                            &cfg.dims,
                                                                            cfg.isVirtual));
        }
    }

    // Create BatchnormAttributes (training mode) with specified UIDs
    auto bnAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        xUid,
        scaleUid,
        biasUid,
        epsilonUid,
        0, // peer_stats_tensor_uid
        flatbuffers::nullopt, // prev_running_mean_tensor_uid
        flatbuffers::nullopt, // prev_running_variance_tensor_uid
        flatbuffers::nullopt, // momentum_tensor_uid
        yUid,
        meanUid,
        invVarianceUid,
        flatbuffers::nullopt, // next_running_mean_tensor_uid
        flatbuffers::nullopt // next_running_variance_tensor_uid
    );

    // Create node
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormBackwardGraph(const std::vector<TensorConfig>& configs,
                                int64_t xUid,
                                int64_t dyUid,
                                int64_t dxUid,
                                int64_t scaleUid,
                                int64_t dscaleUid,
                                int64_t dbiasUid,
                                flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt,
                                flatbuffers::Optional<int64_t> invVarianceUid
                                = flatbuffers::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create BatchnormBackwardAttributes with specified UIDs
    auto bnBwdAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        dyUid,
        xUid,
        meanUid,
        invVarianceUid,
        scaleUid,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        dxUid,
        dscaleUid,
        dbiasUid);

    // Create node
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormFusedBackwardGraph(const std::vector<TensorConfig>& configs,
                                     int64_t xUid,
                                     int64_t scaleUid,
                                     int64_t biasUid,
                                     int64_t meanUid,
                                     int64_t invVarianceUid,
                                     int64_t dyUid,
                                     int64_t dxUid,
                                     int64_t dscaleUid,
                                     int64_t dbiasUid,
                                     int64_t bnYVirtualUid,
                                     int64_t dxDreluVirtualUid,
                                     hipdnn_data_sdk::data_objects::PointwiseMode activationMode
                                     = hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD)
{
    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attribute offsets
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create 3 nodes for the fusion
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder, xUid, meanUid, invVarianceUid, scaleUid, biasUid, bnYVirtualUid);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttrs.Union()));

    // Node 1: Activation (Backward mode, e.g., RELU_BWD)
    auto actAttrs = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        activationMode,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        bnYVirtualUid,
        flatbuffers::Optional<int64_t>(dyUid),
        flatbuffers::nullopt, // in_2_tensor_uid
        dxDreluVirtualUid);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "activation_bwd",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttrs.Union()));

    // Node 2: Batchnorm Backward
    auto bnBwdAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        dxDreluVirtualUid,
        xUid,
        flatbuffers::Optional<int64_t>(meanUid),
        flatbuffers::Optional<int64_t>(invVarianceUid),
        scaleUid,
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        dxUid,
        dscaleUid,
        dbiasUid);
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder buildBatchnormInferenceWithVarianceFusedGraph(
    const std::vector<TensorConfig>& configs,
    int64_t xUid,
    int64_t yUid,
    int64_t scaleUid,
    int64_t biasUid,
    int64_t epsilonUid,
    int64_t meanUid,
    int64_t varianceUid,
    int64_t bnYVirtualUid,
    hipdnn_data_sdk::data_objects::PointwiseMode activationMode
    = hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        if(cfg.isPassByValue)
        {
            hipdnn_data_sdk::data_objects::Float64Value floatValue(cfg.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            cfg.uid,
                                                                            cfg.name.c_str(),
                                                                            cfg.dataType,
                                                                            &cfg.strides,
                                                                            &cfg.dims,
                                                                            cfg.isVirtual));
        }
    }

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference with Variance
    auto bnAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
        builder, xUid, meanUid, varianceUid, scaleUid, biasUid, bnYVirtualUid, epsilonUid);

    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference_variance_ext",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt,
        bnAttrs.Union()));

    // Node 1: Activation
    auto actAttrs = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        activationMode,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        bnYVirtualUid, // in_0_tensor_uid
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        yUid); // out_0_tensor_uid

    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "activation",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttrs.Union()));

    // Build graph
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

// --- Test Data Providers: Layer 3 (High-Level Validators) ---

inline std::vector<BatchnormInferenceConfigTestCase>
    getCheckBatchnormInferenceConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnCommonTensorIds;

    std::vector<BatchnormInferenceConfigTestCase> cases;

    // Happy paths - all shapes × all 4D layouts × all valid type configurations
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                cases.push_back(
                    {"AcceptsInference_" + generateName(dims, *layout) + "_" + toString(typeConfig),
                     true,
                     createBatchnormInferenceTensors(typeConfig, dims, *layout),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::MEAN,
                     UIDs::INV_VARIANCE});
            }
        }
    }

    // Happy paths - 5D shapes × 5D layouts × all valid type configurations
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                cases.push_back(
                    {"AcceptsInference_" + generateName(dims, *layout) + "_" + toString(typeConfig),
                     true,
                     createBatchnormInferenceTensors(typeConfig, dims, *layout),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::MEAN,
                     UIDs::INV_VARIANCE});
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    auto sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        cases.push_back(
            {"RejectsInference_" + toString(invalidConfig),
             false,
             createBatchnormInferenceTensors(invalidConfig, sampleDims, TensorLayout::NCHW),
             UIDs::X,
             UIDs::Y,
             UIDs::SCALE,
             UIDs::BIAS,
             UIDs::MEAN,
             UIDs::INV_VARIANCE});
    }

    // Unhappy paths - mixed layouts (x is NCHW, y is NHWC)
    auto nchwStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NHWC.strideOrder);
    auto sampleDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleDims);
    auto sampleDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleDerivedDims, TensorLayout::NCHW.strideOrder);

    std::vector<TensorConfig> mixedLayoutConfigs
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, nchwStrides, ""},
           {UIDs::Y, "y", DT::FLOAT, sampleDims, nhwcStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::BIAS, "bias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::MEAN, "mean", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::INV_VARIANCE,
            "inv_variance",
            DT::FLOAT,
            sampleDerivedDims,
            sampleDerivedStrides,
            ""}};
    cases.push_back({"RejectsInference_MixedLayouts",
                     false,
                     mixedLayoutConfigs,
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::MEAN,
                     UIDs::INV_VARIANCE});

    return cases;
}

inline std::vector<BatchnormTrainingConfigTestCase>
    getCheckBatchnormTrainingConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnTrainingTensorIds;

    std::vector<BatchnormTrainingConfigTestCase> cases;

    // ========================================================================
    // Happy Paths: Valid Type Configurations from bn_type_configs::VALID
    // ========================================================================

    // All training shapes × all layouts × all valid type configs × 2 variants
    for(const auto& typeConfig : bn_type_configs::VALID)
    {
        for(const auto& dims : shapes::TRAINING_4D)
        {
            for(const auto* layout : LAYOUTS_4D)
            {
                // With mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // Happy paths - 5D training shapes × all valid type configs
    for(const auto& typeConfig : bn_type_configs::VALID)
    {
        for(const auto& dims : shapes::TRAINING_5D)
        {
            for(const auto* layout : LAYOUTS_5D)
            {
                // With mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without mean/variance
                cases.push_back(
                    {"AcceptsTraining_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutMeanVar",
                     true,
                     createBatchnormTrainingTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // ========================================================================
    // Unhappy Paths: Invalid Configurations
    // ========================================================================

    auto sampleTrainingDims = shapes::TRAINING_4D[0];

    // Unhappy paths - insufficient spatial dimensions (B × S ≤ 1)
    cases.push_back({"RejectsTraining_InsufficientSpatial_1x1",
                     false,
                     createBatchnormTrainingTensors(bn_type_configs::ALL_FLOAT,
                                                    shapes::INSUFFICIENT_SPATIAL_4D,
                                                    TensorLayout::NCHW,
                                                    false,
                                                    false),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - invalid IO data type (UINT8 instead of FLOAT)
    auto sampleTrainingStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleTrainingDims, TensorLayout::NCHW.strideOrder);
    auto derivedTrainingDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleTrainingDims);
    auto derivedTrainingStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedTrainingDims, TensorLayout::NCHW.strideOrder);

    std::vector<TensorConfig> invalidTypeConfigs = {
        {UIDs::X, "x", DT::UINT8, sampleTrainingDims, sampleTrainingStrides, "", false, false, 0.0},
        {UIDs::Y, "y", DT::UINT8, sampleTrainingDims, sampleTrainingStrides, "", false, false, 0.0},
        {UIDs::SCALE,
         "scale",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::BIAS,
         "bias",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::EPSILON, "epsilon", DT::FLOAT, {1}, {1}, "", false, true, 1e-5}};
    cases.push_back({"RejectsTraining_InvalidIoDataType",
                     false,
                     invalidTypeConfigs,
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - non-packed tensor (invalid strides)
    std::vector<TensorConfig> nonPackedConfigs = {
        {UIDs::X,
         "x",
         DT::FLOAT,
         sampleTrainingDims,
         {1000, 300, 20, 1},
         "",
         false,
         false,
         0.0}, // Non-packed! (intentionally invalid strides)
        {UIDs::Y, "y", DT::FLOAT, sampleTrainingDims, sampleTrainingStrides, "", false, false, 0.0},
        {UIDs::SCALE,
         "scale",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::BIAS,
         "bias",
         DT::FLOAT,
         derivedTrainingDims,
         derivedTrainingStrides,
         "",
         false,
         false,
         0.0},
        {UIDs::EPSILON, "epsilon", DT::FLOAT, {1}, {1}, "", false, true, 1e-5}};
    cases.push_back({"RejectsTraining_NonPackedTensor",
                     false,
                     nonPackedConfigs,
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - all invalid type configurations
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        const bool hasInvalidStatType = (invalidConfig.stat != DT::FLOAT);

        // Always test with mean/variance outputs (validates all types)
        cases.push_back({"RejectsTraining_" + toString(invalidConfig) + "_WithMeanVar",
                         false,
                         createBatchnormTrainingTensors(
                             invalidConfig, sampleTrainingDims, TensorLayout::NCHW, true, true),
                         UIDs::X,
                         UIDs::Y,
                         UIDs::SCALE,
                         UIDs::BIAS,
                         UIDs::EPSILON,
                         flatbuffers::Optional<int64_t>(UIDs::MEAN),
                         flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});

        // Only test without mean/variance if stat type is valid (FLOAT)
        // (Can't validate stat type if no stat tensors exist)
        if(!hasInvalidStatType)
        {
            cases.push_back(
                {"RejectsTraining_" + toString(invalidConfig) + "_NoMeanVar",
                 false,
                 createBatchnormTrainingTensors(
                     invalidConfig, sampleTrainingDims, TensorLayout::NCHW, false, false),
                 UIDs::X,
                 UIDs::Y,
                 UIDs::SCALE,
                 UIDs::BIAS,
                 UIDs::EPSILON,
                 flatbuffers::nullopt,
                 flatbuffers::nullopt});
        }
    }

    return cases;
}

inline std::vector<BatchnormBackwardConfigTestCase>
    getCheckBatchnormBackwardConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnBackwardTensorIds;

    std::vector<BatchnormBackwardConfigTestCase> cases;

    // Happy paths - all inference shapes × all 4D layouts × all valid type configs × 2 variants
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                // With optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // Happy paths - 5D inference shapes × 5D layouts × all valid type configs
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                // With optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, true, true),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::Optional<int64_t>(UIDs::MEAN),
                     flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});
                // Without optionals
                cases.push_back(
                    {"AcceptsBackward_" + generateName(dims, *layout) + "_" + toString(typeConfig)
                         + "_WithoutOptionals",
                     true,
                     createBatchnormBackwardTensors(typeConfig, dims, *layout, false, false),
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    auto sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        const bool hasInvalidStatType = (invalidConfig.stat != DT::FLOAT);

        // Always test with mean/variance inputs (validates all types)
        cases.push_back({"RejectsBackward_" + toString(invalidConfig) + "_WithMeanVar",
                         false,
                         createBatchnormBackwardTensors(
                             invalidConfig, sampleDims, TensorLayout::NCHW, true, true),
                         UIDs::X,
                         UIDs::DY,
                         UIDs::DX,
                         UIDs::SCALE,
                         UIDs::DSCALE,
                         UIDs::DBIAS,
                         flatbuffers::Optional<int64_t>(UIDs::MEAN),
                         flatbuffers::Optional<int64_t>(UIDs::INV_VARIANCE)});

        // Only test without mean/variance if stat type is valid (FLOAT)
        // (Can't validate stat type if no stat tensors exist)
        if(!hasInvalidStatType)
        {
            cases.push_back({"RejectsBackward_" + toString(invalidConfig) + "_NoMeanVar",
                             false,
                             createBatchnormBackwardTensors(
                                 invalidConfig, sampleDims, TensorLayout::NCHW, false, false),
                             UIDs::X,
                             UIDs::DY,
                             UIDs::DX,
                             UIDs::SCALE,
                             UIDs::DSCALE,
                             UIDs::DBIAS,
                             flatbuffers::nullopt,
                             flatbuffers::nullopt});
        }
    }

    // Unhappy paths - inconsistent IO shapes
    auto sampleStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NCHW.strideOrder);
    auto sampleDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleDims);
    auto sampleDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleDerivedDims, TensorLayout::NCHW.strideOrder);
    auto mediumDims = shapes::INFERENCE_4D[1]; // {1, 3, 112, 112}
    auto mediumStrides
        = hipdnn_data_sdk::utilities::generateStrides(mediumDims, TensorLayout::NCHW.strideOrder);

    std::vector<TensorConfig> inconsistentShapes
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::DY, "dy", DT::FLOAT, mediumDims, mediumStrides, ""}, // Different!
           {UIDs::DX, "dx", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DSCALE, "dscale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DBIAS, "dbias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""}};
    cases.push_back({"RejectsBackward_InconsistentShapes",
                     false,
                     inconsistentShapes,
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    // Unhappy paths - non-packed tensor
    std::vector<TensorConfig> nonPackedConfigs
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::DY,
            "dy",
            DT::FLOAT,
            sampleDims,
            {200000, 60000, 250, 1},
            ""}, // Non-packed! (intentionally invalid strides)
           {UIDs::DX, "dx", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DSCALE, "dscale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DBIAS, "dbias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""}};
    cases.push_back({"RejectsBackward_NonPackedTensor",
                     false,
                     nonPackedConfigs,
                     UIDs::X,
                     UIDs::DY,
                     UIDs::DX,
                     UIDs::SCALE,
                     UIDs::DSCALE,
                     UIDs::DBIAS,
                     flatbuffers::nullopt,
                     flatbuffers::nullopt});

    return cases;
}

inline std::vector<BatchnormFusedBackwardConfigTestCase>
    getCheckBatchnormFusedBackwardConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnFusedTensorIds;

    std::vector<BatchnormFusedBackwardConfigTestCase> cases;

    // Supported backward activation modes (from atomic validation tests)
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;
    const std::vector<PM> supportedBwdActivations = {PM::IDENTITY, PM::RELU_BWD};

    // Happy paths - all inference shapes × all 4D layouts × all valid type configs × all supported activations
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                for(const auto& activMode : supportedBwdActivations)
                {
                    cases.push_back({"AcceptsFused_" + generateName(dims, *layout) + "_"
                                         + toString(typeConfig) + "_"
                                         + activationModeToString(activMode),
                                     true,
                                     createBatchnormFusedBackwardTensors(typeConfig, dims, *layout),
                                     UIDs::X,
                                     UIDs::SCALE,
                                     UIDs::BIAS,
                                     UIDs::MEAN,
                                     UIDs::INV_VARIANCE,
                                     UIDs::DY,
                                     UIDs::DX,
                                     UIDs::DSCALE,
                                     UIDs::DBIAS,
                                     UIDs::BN_Y_VIRTUAL,
                                     UIDs::DX_DRELU_VIRTUAL,
                                     activMode});
                }
            }
        }
    }

    // Happy paths - 5D inference shapes × 5D layouts × all valid type configs × all supported activations
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                for(const auto& activMode : supportedBwdActivations)
                {
                    cases.push_back({"AcceptsFused_" + generateName(dims, *layout) + "_"
                                         + toString(typeConfig) + "_"
                                         + activationModeToString(activMode),
                                     true,
                                     createBatchnormFusedBackwardTensors(typeConfig, dims, *layout),
                                     UIDs::X,
                                     UIDs::SCALE,
                                     UIDs::BIAS,
                                     UIDs::MEAN,
                                     UIDs::INV_VARIANCE,
                                     UIDs::DY,
                                     UIDs::DX,
                                     UIDs::DSCALE,
                                     UIDs::DBIAS,
                                     UIDs::BN_Y_VIRTUAL,
                                     UIDs::DX_DRELU_VIRTUAL,
                                     activMode});
                }
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    auto sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_FUSED_ALL)
    {
        cases.push_back(
            {"RejectsFused_" + toString(invalidConfig),
             false,
             createBatchnormFusedBackwardTensors(invalidConfig, sampleDims, TensorLayout::NCHW),
             UIDs::X,
             UIDs::SCALE,
             UIDs::BIAS,
             UIDs::MEAN,
             UIDs::INV_VARIANCE,
             UIDs::DY,
             UIDs::DX,
             UIDs::DSCALE,
             UIDs::DBIAS,
             UIDs::BN_Y_VIRTUAL,
             UIDs::DX_DRELU_VIRTUAL,
             PM::RELU_BWD});
    }

    // Unhappy paths - mixed layouts
    auto sampleStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NCHW.strideOrder);
    auto sampleDerivedDims = hipdnn_data_sdk::utilities::getDerivedShape(sampleDims);
    auto sampleDerivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        sampleDerivedDims, TensorLayout::NCHW.strideOrder);
    auto nhwcStrides
        = hipdnn_data_sdk::utilities::generateStrides(sampleDims, TensorLayout::NHWC.strideOrder);

    std::vector<TensorConfig> mixedLayoutConfigs
        = {{UIDs::X, "x", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::SCALE, "scale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::BIAS, "bias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::MEAN, "mean", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::INV_VARIANCE,
            "inv_variance",
            DT::FLOAT,
            sampleDerivedDims,
            sampleDerivedStrides,
            ""},
           {UIDs::DY, "dy", DT::FLOAT, sampleDims, nhwcStrides, ""}, // NHWC - different!
           {UIDs::DX, "dx", DT::FLOAT, sampleDims, sampleStrides, ""},
           {UIDs::DSCALE, "dscale", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::DBIAS, "dbias", DT::FLOAT, sampleDerivedDims, sampleDerivedStrides, ""},
           {UIDs::BN_Y_VIRTUAL, "BN_Y", DT::FLOAT, sampleDims, sampleStrides, "", true},
           {UIDs::DX_DRELU_VIRTUAL, "DX_drelu", DT::FLOAT, sampleDims, sampleStrides, "", true}};
    cases.push_back({"RejectsFused_MixedLayouts",
                     false,
                     mixedLayoutConfigs,
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::RELU_BWD});

    // Unhappy paths - unsupported activation: SIGMOID_BWD
    cases.push_back({"RejectsFused_UnsupportedActivation_SigmoidBwd",
                     false,
                     createBatchnormFusedBackwardTensors(
                         bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW),
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::SIGMOID_BWD});

    // Unhappy paths - unsupported activation: TANH_BWD
    cases.push_back({"RejectsFused_UnsupportedActivation_TanhBwd",
                     false,
                     createBatchnormFusedBackwardTensors(
                         bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW),
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::TANH_BWD});

    // Unhappy paths - unsupported activation: RELU_FWD (wrong direction)
    cases.push_back({"RejectsFused_UnsupportedActivation_ReluFwdInBwdContext",
                     false,
                     createBatchnormFusedBackwardTensors(
                         bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW),
                     BnFusedTensorIds::X,
                     BnFusedTensorIds::SCALE,
                     BnFusedTensorIds::BIAS,
                     BnFusedTensorIds::MEAN,
                     BnFusedTensorIds::INV_VARIANCE,
                     BnFusedTensorIds::DY,
                     BnFusedTensorIds::DX,
                     BnFusedTensorIds::DSCALE,
                     BnFusedTensorIds::DBIAS,
                     BnFusedTensorIds::BN_Y_VIRTUAL,
                     BnFusedTensorIds::DX_DRELU_VIRTUAL,
                     PM::RELU_FWD});

    return cases;
}

inline std::vector<ActivationModeTestCase> getCheckBatchnormFwdActivationModeSupportedTestCases()
{
    return {
        // Happy paths - supported activation modes
        {"AcceptsIdentity",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsRelu",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsClippedRelu",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},
        {"AcceptsClamp",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::Optional<double>(0.0),
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},

        // Unhappy paths - unsupported activation modes
        {"RejectsLeakyRelu",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(0.01)},
        {"RejectsSigmoid",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsTanh",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::TANH_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsReluBwdInFwdContext",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
    };
}

inline std::vector<ActivationModeTestCase> getCheckBatchnormBwdActivationModeSupportedTestCases()
{
    return {
        // Happy paths - supported activation modes
        {"AcceptsIdentityBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsReluBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"AcceptsClippedReluBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},
        {"AcceptsClampBwd",
         true,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::Optional<double>(0.0),
         flatbuffers::Optional<double>(6.0),
         flatbuffers::nullopt},

        // Unhappy paths - unsupported activation modes
        {"RejectsLeakyReluBwd",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::Optional<double>(0.01)},
        {"RejectsSigmoidBwd",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsTanhBwd",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::TANH_BWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
        {"RejectsReluFwdInBwdContext",
         false,
         hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD,
         flatbuffers::nullopt,
         flatbuffers::nullopt,
         flatbuffers::nullopt},
    };
}

} // namespace

// --- Test Classes: Layer 3 (High-Level Configuration Validators) ---

class TestCheckBatchnormInferenceConfigSupported
    : public ::testing::TestWithParam<BatchnormInferenceConfigTestCase>
{
};

TEST_P(TestCheckBatchnormInferenceConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormInferenceGraph(
        tc.tensorConfigs, tc.xUid, tc.yUid, tc.scaleUid, tc.biasUid, tc.meanUid, tc.invVarianceUid);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW(
            { checkBatchnormInferenceTensorConfigSupported(*attrs, graph.getTensorMap()); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormInferenceTensorConfigSupported(*attrs, graph.getTensorMap()); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormInferenceConfigSupported,
                         testing::ValuesIn(getCheckBatchnormInferenceConfigSupportedTestCases()));

class TestCheckBatchnormTrainingConfigSupported
    : public ::testing::TestWithParam<BatchnormTrainingConfigTestCase>
{
};

TEST_P(TestCheckBatchnormTrainingConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormTrainingGraph(tc.tensorConfigs,
                                               tc.xUid,
                                               tc.yUid,
                                               tc.scaleUid,
                                               tc.biasUid,
                                               tc.epsilonUid,
                                               tc.meanUid,
                                               tc.invVarianceUid);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW(
            { checkBatchnormFwdTrainingTensorConfigSupported(*attrs, graph.getTensorMap()); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormFwdTrainingTensorConfigSupported(*attrs, graph.getTensorMap()); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormTrainingConfigSupported,
                         testing::ValuesIn(getCheckBatchnormTrainingConfigSupportedTestCases()));

class TestCheckBatchnormBackwardConfigSupported
    : public ::testing::TestWithParam<BatchnormBackwardConfigTestCase>
{
};

TEST_P(TestCheckBatchnormBackwardConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormBackwardGraph(tc.tensorConfigs,
                                               tc.xUid,
                                               tc.dyUid,
                                               tc.dxUid,
                                               tc.scaleUid,
                                               tc.dscaleUid,
                                               tc.dbiasUid,
                                               tc.meanUid,
                                               tc.invVarianceUid);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW(
            { checkBatchnormBackwardTensorConfigSupported(*attrs, graph.getTensorMap()); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormBackwardTensorConfigSupported(*attrs, graph.getTensorMap()); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormBackwardConfigSupported,
                         testing::ValuesIn(getCheckBatchnormBackwardConfigSupportedTestCases()));

class TestCheckBatchnormFusedBackwardConfigSupported
    : public ::testing::TestWithParam<BatchnormFusedBackwardConfigTestCase>
{
};

TEST_P(TestCheckBatchnormFusedBackwardConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormFusedBackwardGraph(tc.tensorConfigs,
                                                    tc.xUid,
                                                    tc.scaleUid,
                                                    tc.biasUid,
                                                    tc.meanUid,
                                                    tc.invVarianceUid,
                                                    tc.dyUid,
                                                    tc.dxUid,
                                                    tc.dscaleUid,
                                                    tc.dbiasUid,
                                                    tc.bnYVirtualUid,
                                                    tc.dxDreluVirtualUid,
                                                    tc.activationMode);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* actAttrs = actNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(actAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkBatchnormInferenceActivationBackwardTensorConfigSupported(
                *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkBatchnormInferenceActivationBackwardTensorConfigSupported(
                    *bnInfAttrs, *actAttrs, *bnBwdAttrs, graph.getTensorMap());
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    TestCheckBatchnormFusedBackwardConfigSupported,
    testing::ValuesIn(getCheckBatchnormFusedBackwardConfigSupportedTestCases()));

class TestCheckBatchnormFwdActivationModeSupported
    : public ::testing::TestWithParam<ActivationModeTestCase>
{
};

TEST_P(TestCheckBatchnormFwdActivationModeSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                            tc.mode,
                                                                            tc.reluLowerClip,
                                                                            tc.reluUpperClip,
                                                                            tc.reluLowerClipSlope,
                                                                            flatbuffers::nullopt,
                                                                            1,
                                                                            flatbuffers::nullopt,
                                                                            flatbuffers::nullopt,
                                                                            2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormFwdActivationModeSupported(*attr); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormFwdActivationModeSupported(*attr); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormFwdActivationModeSupported,
                         testing::ValuesIn(getCheckBatchnormFwdActivationModeSupportedTestCases()));

class TestCheckBatchnormBwdActivationModeSupported
    : public ::testing::TestWithParam<ActivationModeTestCase>
{
};

TEST_P(TestCheckBatchnormBwdActivationModeSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    flatbuffers::FlatBufferBuilder builder;
    auto actAttr = hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                            tc.mode,
                                                                            tc.reluLowerClip,
                                                                            tc.reluUpperClip,
                                                                            tc.reluLowerClipSlope,
                                                                            flatbuffers::nullopt,
                                                                            1,
                                                                            flatbuffers::nullopt,
                                                                            flatbuffers::nullopt,
                                                                            2);
    builder.Finish(actAttr);

    const auto* attr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::PointwiseAttributes>(
        builder.GetBufferPointer());

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({ checkBatchnormBwdActivationModeSupported(*attr); });
    }
    else
    {
        EXPECT_THROW(
            { checkBatchnormBwdActivationModeSupported(*attr); },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestCheckBatchnormBwdActivationModeSupported,
                         testing::ValuesIn(getCheckBatchnormBwdActivationModeSupportedTestCases()));

// --- Integration Tests: Peer Stats Validation ---

TEST(TestBatchnormApplicabilityChecks, RejectsBatchnormFwdTrainingWithPeerStats)
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnTrainingTensorIds;

    // Create valid tensor configs
    const auto& dims = shapes::TRAINING_4D[0];
    auto configs = createBatchnormTrainingTensors(
        bn_type_configs::ALL_FLOAT, dims, TensorLayout::NCHW, false, false);

    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attributes
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    for(const auto& cfg : configs)
    {
        if(cfg.isPassByValue)
        {
            hipdnn_data_sdk::data_objects::Float64Value floatValue(cfg.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            cfg.uid,
                                                                            cfg.name.c_str(),
                                                                            cfg.dataType,
                                                                            &cfg.strides,
                                                                            &cfg.dims,
                                                                            cfg.isVirtual));
        }
    }

    // Create peer_stats_tensor_uid with populated values (should be rejected)
    std::vector<int64_t> peerStatsUids = {100, 101, 102};
    auto peerStatsOffset = builder.CreateVector(peerStatsUids);

    // Create BatchnormAttributes WITH peer_stats populated
    auto bnAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        UIDs::X,
        UIDs::SCALE,
        UIDs::BIAS,
        UIDs::EPSILON,
        peerStatsOffset, // peer_stats populated - should be rejected
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        UIDs::Y,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt);

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        DT::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnAttrs.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder, "test_graph", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensorOffsets, &nodes);

    builder.Finish(graphOffset);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw because peer_stats is populated
    EXPECT_THROW(
        { checkBatchnormFwdTrainingTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// --- Variance Extension Support ---

// Creates tensors for BatchnormInferenceAttributesVarianceExt
inline std::vector<TensorConfig> createBatchnormInferenceWithVarianceTensors(
    const BnTensorTypes& types, const std::vector<int64_t>& dims, const TensorLayout& layout)
{
    using UIDs = BnInferenceVarianceExtTensorIds;
    std::vector<TensorConfig> configs;

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);
    configs.push_back(createIoTensor(UIDs::X, "x", types.io, dims, strides));
    configs.push_back(createIoTensor(UIDs::Y, "y", types.io, dims, strides));

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides
        = hipdnn_data_sdk::utilities::generateStrides(derivedDims, layout.strideOrder);
    configs.push_back(
        createAffineTensor(UIDs::SCALE, "scale", types.affine, derivedDims, derivedStrides));
    configs.push_back(
        createAffineTensor(UIDs::BIAS, "bias", types.affine, derivedDims, derivedStrides));

    configs.push_back(createScalarTensor(UIDs::EPSILON, "epsilon", 1e-5));

    configs.push_back(
        createStatTensor(UIDs::MEAN, "mean", types.stat, derivedDims, derivedStrides));
    configs.push_back(
        createStatTensor(UIDs::VARIANCE, "variance", types.stat, derivedDims, derivedStrides));

    return configs;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormInferenceWithVarianceGraph(const std::vector<TensorConfig>& configs,
                                             int64_t xUid,
                                             int64_t yUid,
                                             int64_t scaleUid,
                                             int64_t biasUid,
                                             int64_t epsilonUid,
                                             int64_t meanUid,
                                             int64_t varianceUid)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        if(cfg.isPassByValue)
        {
            hipdnn_data_sdk::data_objects::Float64Value floatValue(cfg.passedValue);
            auto valueOffset = builder.CreateStruct(floatValue).Union();
            tensorOffsets.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                cfg.uid,
                cfg.name.c_str(),
                cfg.dataType,
                &cfg.strides,
                &cfg.dims,
                cfg.isVirtual,
                hipdnn_data_sdk::data_objects::TensorValue::Float64Value,
                valueOffset));
        }
        else
        {
            tensorOffsets.push_back(
                hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                            cfg.uid,
                                                                            cfg.name.c_str(),
                                                                            cfg.dataType,
                                                                            &cfg.strides,
                                                                            &cfg.dims,
                                                                            cfg.isVirtual));
        }
    }

    auto bnAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
        builder, xUid, meanUid, varianceUid, scaleUid, biasUid, yUid, epsilonUid);

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference_variance_ext",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt,
        bnAttrs.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_graph",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorOffsets,
        &nodes);

    builder.Finish(graphOffset);
    return builder;
}

struct BatchnormInferenceVarianceExtConfigTestCase
{
    std::string name;
    bool shouldPass;
    std::vector<TensorConfig> tensorConfigs;
    int64_t xUid;
    int64_t yUid;
    int64_t scaleUid;
    int64_t biasUid;
    int64_t epsilonUid;
    int64_t meanUid;
    int64_t varianceUid;

    friend std::ostream& operator<<(std::ostream& os,
                                    const BatchnormInferenceVarianceExtConfigTestCase& tc)
    {
        return os << tc.name;
    }
};

inline std::vector<BatchnormInferenceVarianceExtConfigTestCase>
    getCheckBatchnormInferenceWithVarianceConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using UIDs = BnInferenceVarianceExtTensorIds;

    std::vector<BatchnormInferenceVarianceExtConfigTestCase> cases;

    // Happy paths - all shapes × all 4D layouts × all valid type configurations
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                cases.push_back(
                    {"AcceptsInferenceVarExt_" + generateName(dims, *layout) + "_"
                         + toString(typeConfig),
                     true,
                     createBatchnormInferenceWithVarianceTensors(typeConfig, dims, *layout),
                     UIDs::X,
                     UIDs::Y,
                     UIDs::SCALE,
                     UIDs::BIAS,
                     UIDs::EPSILON,
                     UIDs::MEAN,
                     UIDs::VARIANCE});
            }
        }
    }

    // Happy paths - 5D shapes
    for(const auto& dims : shapes::INFERENCE_5D)
    {
        for(const auto* layout : LAYOUTS_5D)
        {
            cases.push_back({"AcceptsInferenceVarExt_" + generateName(dims, *layout) + "_AllFloat",
                             true,
                             createBatchnormInferenceWithVarianceTensors(
                                 bn_type_configs::ALL_FLOAT, dims, *layout),
                             UIDs::X,
                             UIDs::Y,
                             UIDs::SCALE,
                             UIDs::BIAS,
                             UIDs::EPSILON,
                             UIDs::MEAN,
                             UIDs::VARIANCE});
        }
    }

    // Unhappy paths - invalid type configurations
    const auto& sampleDims = shapes::INFERENCE_4D[0];
    for(const auto& invalidConfig : type_configs::INVALID_ALL)
    {
        cases.push_back({"RejectsInferenceVarExt_" + toString(invalidConfig),
                         false,
                         createBatchnormInferenceWithVarianceTensors(
                             invalidConfig, sampleDims, TensorLayout::NCHW),
                         UIDs::X,
                         UIDs::Y,
                         UIDs::SCALE,
                         UIDs::BIAS,
                         UIDs::EPSILON,
                         UIDs::MEAN,
                         UIDs::VARIANCE});
    }

    return cases;
}

inline std::vector<BatchnormInferenceVarianceExtFusedConfigTestCase>
    getCheckBatchnormInferenceWithVarianceFusedConfigSupportedTestCases()
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnInferenceVarianceExtTensorIds;
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;

    std::vector<BatchnormInferenceVarianceExtFusedConfigTestCase> cases;
    const int64_t bnYVirtualUid = 100;

    // Happy paths - all shapes × all 4D layouts × all valid type configurations
    for(const auto& dims : shapes::INFERENCE_4D)
    {
        for(const auto* layout : LAYOUTS_4D)
        {
            for(const auto& typeConfig : bn_type_configs::VALID)
            {
                auto configs
                    = createBatchnormInferenceWithVarianceTensors(typeConfig, dims, *layout);
                // Add virtual tensor for fusion
                auto strides
                    = hipdnn_data_sdk::utilities::generateStrides(dims, layout->strideOrder);
                configs.push_back(createIoTensor(
                    bnYVirtualUid, "bn_y", typeConfig.intermediate, dims, strides, true));

                cases.push_back({"AcceptsInferenceVarExtFused_" + generateName(dims, *layout) + "_"
                                     + toString(typeConfig),
                                 true,
                                 configs,
                                 UIDs::X,
                                 UIDs::Y,
                                 UIDs::SCALE,
                                 UIDs::BIAS,
                                 UIDs::EPSILON,
                                 UIDs::MEAN,
                                 UIDs::VARIANCE,
                                 PM::RELU_FWD});
            }
        }
    }

    // Unhappy paths - all invalid type configurations
    for(const auto& invalidTypeConfig : type_configs::INVALID_FUSED_ALL)
    {
        const auto& dims = shapes::INFERENCE_4D[0];
        auto layout = LAYOUTS_4D[0];
        auto configs
            = createBatchnormInferenceWithVarianceTensors(invalidTypeConfig, dims, *layout);
        // Add virtual tensor for fusion
        auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout->strideOrder);
        configs.push_back(createIoTensor(
            bnYVirtualUid, "bn_y", invalidTypeConfig.intermediate, dims, strides, true));

        cases.push_back({"RejectsInferenceVarExtFused_" + generateName(dims, *layout) + "_"
                             + toString(invalidTypeConfig),
                         false,
                         configs,
                         UIDs::X,
                         UIDs::Y,
                         UIDs::SCALE,
                         UIDs::BIAS,
                         UIDs::EPSILON,
                         UIDs::MEAN,
                         UIDs::VARIANCE,
                         PM::RELU_FWD});
    }

    {
        // Unhappy paths - unsupported activation mode
        const auto& sampleDims = shapes::INFERENCE_4D[0];
        auto configs = createBatchnormInferenceWithVarianceTensors(
            bn_type_configs::ALL_FLOAT, sampleDims, TensorLayout::NCHW);
        auto strides = hipdnn_data_sdk::utilities::generateStrides(sampleDims,
                                                                   TensorLayout::NCHW.strideOrder);
        configs.push_back(
            createIoTensor(bnYVirtualUid, "bn_y", DT::FLOAT, sampleDims, strides, true));

        cases.push_back({"RejectsInferenceVarExtFused_UnsupportedActivation",
                         false,
                         configs,
                         UIDs::X,
                         UIDs::Y,
                         UIDs::SCALE,
                         UIDs::BIAS,
                         UIDs::EPSILON,
                         UIDs::MEAN,
                         UIDs::VARIANCE,
                         PM::SIGMOID_FWD});
    }

    return cases;
}

class TestCheckBatchnormInferenceWithVarianceConfigSupported
    : public ::testing::TestWithParam<BatchnormInferenceVarianceExtConfigTestCase>
{
};

TEST_P(TestCheckBatchnormInferenceWithVarianceConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();

    auto builder = buildBatchnormInferenceWithVarianceGraph(tc.tensorConfigs,
                                                            tc.xUid,
                                                            tc.yUid,
                                                            tc.scaleUid,
                                                            tc.biasUid,
                                                            tc.epsilonUid,
                                                            tc.meanUid,
                                                            tc.varianceUid);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(attrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkBatchnormInferenceVarianceExtTensorConfigSupported(*attrs, graph.getTensorMap());
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkBatchnormInferenceVarianceExtTensorConfigSupported(*attrs,
                                                                        graph.getTensorMap());
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    TestCheckBatchnormInferenceWithVarianceConfigSupported,
    testing::ValuesIn(getCheckBatchnormInferenceWithVarianceConfigSupportedTestCases()));

class TestCheckBatchnormInferenceWithVarianceFusedConfigSupported
    : public ::testing::TestWithParam<BatchnormInferenceVarianceExtFusedConfigTestCase>
{
};

TEST_P(TestCheckBatchnormInferenceWithVarianceFusedConfigSupported, ValidatesCorrectly)
{
    const auto& tc = GetParam();
    const int64_t bnYVirtualUid = 100;

    auto builder = buildBatchnormInferenceWithVarianceFusedGraph(tc.tensorConfigs,
                                                                 tc.xUid,
                                                                 tc.yUid,
                                                                 tc.scaleUid,
                                                                 tc.biasUid,
                                                                 tc.epsilonUid,
                                                                 tc.meanUid,
                                                                 tc.varianceUid,
                                                                 bnYVirtualUid,
                                                                 tc.activationMode);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);

    auto* bnAttrs = bnNode.attributes_as_BatchnormInferenceAttributesVarianceExt();
    auto* actAttrs = actNode.attributes_as_PointwiseAttributes();

    ASSERT_NE(bnAttrs, nullptr);
    ASSERT_NE(actAttrs, nullptr);

    if(tc.shouldPass)
    {
        EXPECT_NO_THROW({
            checkBatchnormInferenceVarianceExtActivationTensorConfigSupported(
                *bnAttrs, *actAttrs, graph.getTensorMap());
        });
    }
    else
    {
        EXPECT_THROW(
            {
                checkBatchnormInferenceVarianceExtActivationTensorConfigSupported(
                    *bnAttrs, *actAttrs, graph.getTensorMap());
            },
            hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    TestCheckBatchnormInferenceWithVarianceFusedConfigSupported,
    testing::ValuesIn(getCheckBatchnormInferenceWithVarianceFusedConfigSupportedTestCases()));

// --- Integration Tests: Peer Stats Validation ---

namespace
{

TEST(TestBatchnormApplicabilityChecks, RejectsBatchnormBackwardWithPeerStats)
{
    using namespace canonical_layouts;
    using DT = hipdnn_data_sdk::data_objects::DataType;
    using UIDs = BnBackwardTensorIds;

    // Create valid tensor configs
    const auto& dims = shapes::INFERENCE_4D[0];
    auto configs = createBatchnormBackwardTensors(
        bn_type_configs::ALL_FLOAT, dims, TensorLayout::NCHW, false, false);

    flatbuffers::FlatBufferBuilder builder;

    // Build tensor attributes
    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>> tensorOffsets;
    tensorOffsets.reserve(configs.size());
    for(const auto& cfg : configs)
    {
        tensorOffsets.push_back(
            hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                        cfg.uid,
                                                                        cfg.name.c_str(),
                                                                        cfg.dataType,
                                                                        &cfg.strides,
                                                                        &cfg.dims,
                                                                        cfg.isVirtual));
    }

    // Create peer_stats_tensor_uid with populated values (should be rejected)
    std::vector<int64_t> peerStatsUids = {200, 201};
    auto peerStatsOffset = builder.CreateVector(peerStatsUids);

    // Create BatchnormBackwardAttributes WITH peer_stats populated
    auto bnBwdAttrs = hipdnn_data_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        UIDs::DY,
        UIDs::X,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        UIDs::SCALE,
        peerStatsOffset, // peer_stats populated - should be rejected
        UIDs::DX,
        UIDs::DSCALE,
        UIDs::DBIAS);

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        DT::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttrs.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder, "test_graph", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensorOffsets, &nodes);

    builder.Finish(graphOffset);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw because peer_stats is populated
    EXPECT_THROW(
        { checkBatchnormBackwardTensorConfigSupported(*attrs, graph.getTensorMap()); },
        hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
