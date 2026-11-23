// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The fused two-node engine. It has one pack, so no operation matcher; the input and
/// output tokens are the batchnorm input and the activation output.
inline constexpr PackSymbols BATCHNORM_INFERENCE_POINTWISE{
    "hipkernel:BatchnormInferencePointwise",
    "hipkernel.batchnorm_inference_pointwise.graph_match",
    "",
    "hipkernel.batchnorm_inference_pointwise.kernel_match",
    "",
    "hipkernel.batchnorm_inference_pointwise.dispatch",
    "batchnorm_inference_pointwise.x.uid",
    "",
    "batchnorm_inference_pointwise.output.uid"};

/// Tensor uids the builder below uses.
constexpr int64_t BNPW_X_UID = 1;
constexpr int64_t BNPW_MEAN_UID = 2;
constexpr int64_t BNPW_INV_VARIANCE_UID = 3;
constexpr int64_t BNPW_SCALE_UID = 4;
constexpr int64_t BNPW_BIAS_UID = 5;
constexpr int64_t BNPW_INTERMEDIATE_UID = 6;
constexpr int64_t BNPW_OUTPUT_UID = 7;
/// A second Pointwise operand, added when `binaryPointwise` is set.
constexpr int64_t BNPW_EXTRA_UID = 8;

/// Everything the engine's graph match gates, defaulting to a graph it accepts: an NCHW
/// [2, 3, 4, 5] FLOAT batchnorm inference feeding a plain ReLU through a virtual tensor.
struct BatchnormInferencePointwiseGraph
{
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode
        = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD;
    std::optional<float> lowerClip;
    std::optional<float> upperClip;
    std::optional<float> lowerClipSlope;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    std::vector<int64_t> dims = {2, 3, 4, 5};
    /// The per-channel tensors' dims; empty means [1, C, 1, 1].
    std::vector<int64_t> channelDims;
    /// Builds the batchnorm node alone.
    bool batchnormOnly = false;
    /// Lists the Pointwise node before the batchnorm node.
    bool pointwiseFirst = false;
    /// The Pointwise node reads the batchnorm input rather than its output.
    bool pointwiseReadsInput = false;
    /// Gives the Pointwise node a second operand.
    bool binaryPointwise = false;
    bool intermediateVirtual = true;
    bool outputVirtual = false;
};

/// Packed NCHW strides for @p dims.
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

inline flatbuffers::FlatBufferBuilder
    buildBatchnormInferencePointwiseGraph(const BatchnormInferencePointwiseGraph& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> channelDims = spec.channelDims.empty()
                                                 ? std::vector<int64_t>{1, spec.dims.at(1), 1, 1}
                                                 : spec.channelDims;
    const auto strides = packedStrides(spec.dims);
    const auto channelStrides = packedStrides(channelDims);

    const auto tensor = [&](int64_t uid, bool channel, bool isVirtual) {
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          nullptr,
                                                          spec.dataType,
                                                          channel ? &channelStrides : &strides,
                                                          channel ? &channelDims : &spec.dims,
                                                          isVirtual);
    };

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(tensor(BNPW_X_UID, false, false));
    for(const auto uid : {BNPW_MEAN_UID, BNPW_INV_VARIANCE_UID, BNPW_SCALE_UID, BNPW_BIAS_UID})
    {
        tensors.push_back(tensor(uid, true, false));
    }
    tensors.push_back(tensor(BNPW_INTERMEDIATE_UID, false, spec.intermediateVirtual));
    tensors.push_back(tensor(BNPW_OUTPUT_UID, false, spec.outputVirtual));
    if(spec.binaryPointwise)
    {
        tensors.push_back(tensor(BNPW_EXTRA_UID, false, false));
    }

    const auto batchnormAttributes
        = data_objects::CreateBatchnormInferenceAttributes(builder,
                                                           BNPW_X_UID,
                                                           BNPW_MEAN_UID,
                                                           BNPW_INV_VARIANCE_UID,
                                                           BNPW_SCALE_UID,
                                                           BNPW_BIAS_UID,
                                                           BNPW_INTERMEDIATE_UID);
    const auto batchnorm
        = data_objects::CreateNodeDirect(builder,
                                         "batchnorm_inference",
                                         data_objects::DataType::FLOAT,
                                         data_objects::NodeAttributes::BatchnormInferenceAttributes,
                                         batchnormAttributes.Union());

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(spec.batchnormOnly)
    {
        nodes.push_back(batchnorm);
    }
    else
    {
        data_objects::PointwiseAttributesBuilder pointwiseBuilder(builder);
        pointwiseBuilder.add_operation(spec.mode);
        if(spec.lowerClip.has_value())
        {
            pointwiseBuilder.add_relu_lower_clip(*spec.lowerClip);
        }
        if(spec.upperClip.has_value())
        {
            pointwiseBuilder.add_relu_upper_clip(*spec.upperClip);
        }
        if(spec.lowerClipSlope.has_value())
        {
            pointwiseBuilder.add_relu_lower_clip_slope(*spec.lowerClipSlope);
        }
        pointwiseBuilder.add_in_0_tensor_uid(spec.pointwiseReadsInput ? BNPW_X_UID
                                                                      : BNPW_INTERMEDIATE_UID);
        if(spec.binaryPointwise)
        {
            pointwiseBuilder.add_in_1_tensor_uid(BNPW_EXTRA_UID);
        }
        pointwiseBuilder.add_out_0_tensor_uid(BNPW_OUTPUT_UID);
        const auto pointwiseAttributes = pointwiseBuilder.Finish();
        const auto pointwise
            = data_objects::CreateNodeDirect(builder,
                                             "activation",
                                             data_objects::DataType::FLOAT,
                                             data_objects::NodeAttributes::PointwiseAttributes,
                                             pointwiseAttributes.Union());

        nodes = spec.pointwiseFirst
                    ? std::vector<flatbuffers::Offset<data_objects::Node>>{pointwise, batchnorm}
                    : std::vector<flatbuffers::Offset<data_objects::Node>>{batchnorm, pointwise};
    }

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "batchnorm_inference_pointwise",
                                                   spec.dataType,
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   &tensors,
                                                   &nodes));
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
