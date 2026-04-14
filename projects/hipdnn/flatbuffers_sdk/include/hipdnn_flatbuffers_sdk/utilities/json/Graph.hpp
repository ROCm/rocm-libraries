// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/BatchnormAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/BatchnormBackwardAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/BatchnormInferenceAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/ConvolutionBwdAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/ConvolutionFwdAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/ConvolutionWrwAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/CustomOpAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/LayernormAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/MatmulAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/PointwiseAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/RMSNormAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/RMSNormBackwardAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/ReductionAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/SdpaAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/SdpaBackwardAttributes.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/TensorAttributes.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
NLOHMANN_JSON_SERIALIZE_ENUM(
    NodeAttributes,
    {{NodeAttributes::BatchnormInferenceAttributes, "BatchnormInferenceAttributes"},
     {NodeAttributes::BatchnormInferenceAttributesVarianceExt,
      "BatchnormInferenceAttributesVarianceExt"},
     {NodeAttributes::PointwiseAttributes, "PointwiseAttributes"},
     {NodeAttributes::BatchnormBackwardAttributes, "BatchnormBackwardAttributes"},
     {NodeAttributes::BatchnormAttributes, "BatchnormAttributes"},
     {NodeAttributes::ConvolutionFwdAttributes, "ConvolutionFwdAttributes"},
     {NodeAttributes::ConvolutionBwdAttributes, "ConvolutionBwdAttributes"},
     {NodeAttributes::ConvolutionWrwAttributes, "ConvolutionWrwAttributes"},
     {NodeAttributes::MatmulAttributes, "MatmulAttributes"},
     {NodeAttributes::SdpaAttributes, "SdpaAttributes"},
     {NodeAttributes::SdpaBackwardAttributes, "SdpaBackwardAttributes"},
     {NodeAttributes::LayernormAttributes, "LayernormAttributes"},
     {NodeAttributes::RMSNormAttributes, "RMSNormAttributes"},
     {NodeAttributes::RMSNormBackwardAttributes, "RMSNormBackwardAttributes"},
     {NodeAttributes::BlockScaleDequantizeAttributes, "BlockScaleDequantizeAttributes"},
     {NodeAttributes::BlockScaleQuantizeAttributes, "BlockScaleQuantizeAttributes"},
     {NodeAttributes::CustomOpAttributes, "CustomOpAttributes"},
     {NodeAttributes::ReductionAttributes, "ReductionAttributes"},
     {NodeAttributes::NONE, ""}})

NLOHMANN_JSON_SERIALIZE_ENUM(ConvMode,
                             {{ConvMode::UNSET, "UNSET"},
                              {ConvMode::CONVOLUTION, "CONVOLUTION"},
                              {ConvMode::CROSS_CORRELATION, "CROSS_CORRELATION"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& nodeJson, const data_objects::Node& node)
{
    auto type = node.attributes_type();

    switch(type)
    {
    case data_objects::NodeAttributes::BatchnormInferenceAttributes:
        nodeJson = *node.attributes_as_BatchnormInferenceAttributes();
        break;
    case data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt:
        nodeJson = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        break;
    case data_objects::NodeAttributes::BatchnormBackwardAttributes:
        nodeJson = *node.attributes_as_BatchnormBackwardAttributes();
        break;
    case data_objects::NodeAttributes::BatchnormAttributes:
        nodeJson = *node.attributes_as_BatchnormAttributes();
        break;
    case data_objects::NodeAttributes::PointwiseAttributes:
        nodeJson = *node.attributes_as_PointwiseAttributes();
        break;
    case data_objects::NodeAttributes::ConvolutionFwdAttributes:
        nodeJson = *node.attributes_as_ConvolutionFwdAttributes();
        break;
    case data_objects::NodeAttributes::ConvolutionBwdAttributes:
        nodeJson = *node.attributes_as_ConvolutionBwdAttributes();
        break;
    case data_objects::NodeAttributes::ConvolutionWrwAttributes:
        nodeJson = *node.attributes_as_ConvolutionWrwAttributes();
        break;
    case data_objects::NodeAttributes::MatmulAttributes:
        nodeJson = *node.attributes_as_MatmulAttributes();
        break;
    case data_objects::NodeAttributes::SdpaAttributes:
        nodeJson = *node.attributes_as_SdpaAttributes();
        break;
    case data_objects::NodeAttributes::SdpaBackwardAttributes:
        nodeJson = *node.attributes_as_SdpaBackwardAttributes();
        break;
    case data_objects::NodeAttributes::LayernormAttributes:
        nodeJson = *node.attributes_as_LayernormAttributes();
        break;
    case data_objects::NodeAttributes::RMSNormAttributes:
        nodeJson = *node.attributes_as_RMSNormAttributes();
        break;
    case data_objects::NodeAttributes::RMSNormBackwardAttributes:
        nodeJson = *node.attributes_as_RMSNormBackwardAttributes();
        break;
    case data_objects::NodeAttributes::BlockScaleDequantizeAttributes:
        nodeJson = *node.attributes_as_BlockScaleDequantizeAttributes();
        break;
    case data_objects::NodeAttributes::BlockScaleQuantizeAttributes:
        nodeJson = *node.attributes_as_BlockScaleQuantizeAttributes();
        break;
    case data_objects::NodeAttributes::CustomOpAttributes:
        nodeJson = *node.attributes_as_CustomOpAttributes();
        break;
    case data_objects::NodeAttributes::ReductionAttributes:
        nodeJson = *node.attributes_as_ReductionAttributes();
        break;
    default:
        throw std::runtime_error(
            "hipdnn_flatbuffers_sdk::data_objects::to_json(Node): Unsupported NodeAttributes type: "
            + std::to_string(static_cast<int8_t>(node.attributes_type())));
    }
    nodeJson["name"] = flatbuffers::safeStr(node.name());
    nodeJson["type"] = node.attributes_type();
    nodeJson["compute_data_type"] = node.compute_data_type();
}

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& graphJson, const data_objects::Graph& graph)
{
    graphJson["nodes"] = graph.nodes();
    graphJson["compute_data_type"] = graph.compute_data_type();
    graphJson["io_data_type"] = graph.io_data_type();
    graphJson["intermediate_data_type"] = graph.intermediate_data_type();
    graphJson["name"] = flatbuffers::safeStr(graph.name());
    graphJson["tensors"] = graph.tensors();
    if(graph.preferred_engine_id().has_value())
    {
        graphJson["preferred_engine_id"] = graph.preferred_engine_id().value();
    }
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::Node>(flatbuffers::FlatBufferBuilder& builder,
                                   const nlohmann::json& entry)
{
    auto type = entry.at("type").get<data_objects::NodeAttributes>();
    auto name = entry.at("name").get<std::string>();
    auto computeDataType = entry.at("compute_data_type").get<data_objects::DataType>();

    const flatbuffers::Offset<void> node = [&]() {
        switch(type)
        {
        case data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return to<data_objects::BatchnormInferenceAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt:
            return to<data_objects::BatchnormInferenceAttributesVarianceExt>(builder, entry)
                .Union();
        case data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return to<data_objects::BatchnormBackwardAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::BatchnormAttributes:
            return to<data_objects::BatchnormAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::PointwiseAttributes:
            return to<data_objects::PointwiseAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::ConvolutionFwdAttributes:
            return to<data_objects::ConvolutionFwdAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::ConvolutionBwdAttributes:
            return to<data_objects::ConvolutionBwdAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::ConvolutionWrwAttributes:
            return to<data_objects::ConvolutionWrwAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::MatmulAttributes:
            return to<data_objects::MatmulAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::SdpaAttributes:
            return to<data_objects::SdpaAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::SdpaBackwardAttributes:
            return to<data_objects::SdpaBackwardAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::LayernormAttributes:
            return to<data_objects::LayernormAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::RMSNormAttributes:
            return to<data_objects::RMSNormAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::RMSNormBackwardAttributes:
            return to<data_objects::RMSNormBackwardAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::BlockScaleDequantizeAttributes:
            return to<data_objects::BlockScaleDequantizeAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::BlockScaleQuantizeAttributes:
            return to<data_objects::BlockScaleQuantizeAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::CustomOpAttributes:
            return to<data_objects::CustomOpAttributes>(builder, entry).Union();
        case data_objects::NodeAttributes::ReductionAttributes:
            return to<data_objects::ReductionAttributes>(builder, entry).Union();
        default:
            throw std::runtime_error("hipdnn_flatbuffers_sdk::json::to<data_objects::Node>(): "
                                     "Unsupported NodeAttributes type: "
                                     + std::string{EnumNameNodeAttributes(type)});
        }
    }();

    return data_objects::CreateNodeDirect(builder, name.c_str(), computeDataType, type, node);
}

template <>
inline auto to<data_objects::Graph>(flatbuffers::FlatBufferBuilder& builder,
                                    const nlohmann::json& entry)
{
    using namespace data_objects;
    using namespace flatbuffers;

    auto name = entry.at("name").get<std::string>();
    auto computeType = entry.at("compute_data_type").get<data_objects::DataType>();
    auto ioType = entry.at("io_data_type").get<data_objects::DataType>();
    auto intermediateType = entry.at("intermediate_data_type").get<data_objects::DataType>();

    flatbuffers::Optional<int64_t> preferredEngineId = flatbuffers::nullopt;
    if(entry.contains("preferred_engine_id"))
    {
        preferredEngineId = entry["preferred_engine_id"].get<int64_t>();
    }

    auto nodes = toVector<Node>(builder, entry.at("nodes"));
    auto tensors = toVector<TensorAttributes>(builder, entry.at("tensors"));
    return data_objects::CreateGraphDirect(builder,
                                           name.c_str(),
                                           computeType,
                                           intermediateType,
                                           ioType,
                                           &tensors,
                                           &nodes,
                                           preferredEngineId);
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
