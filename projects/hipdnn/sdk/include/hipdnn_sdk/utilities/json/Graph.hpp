// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/json/BatchnormAttributes.hpp>
#include <hipdnn_sdk/utilities/json/BatchnormBackwardAttributes.hpp>
#include <hipdnn_sdk/utilities/json/BatchnormInferenceAttributes.hpp>
#include <hipdnn_sdk/utilities/json/Common.hpp>
#include <hipdnn_sdk/utilities/json/ConvolutionBwdAttributes.hpp>
#include <hipdnn_sdk/utilities/json/ConvolutionFwdAttributes.hpp>
#include <hipdnn_sdk/utilities/json/ConvolutionWrwAttributes.hpp>
#include <hipdnn_sdk/utilities/json/PointwiseAttributes.hpp>
#include <hipdnn_sdk/utilities/json/TensorAttributes.hpp>

namespace hipdnn_sdk::data_objects
{
NLOHMANN_JSON_SERIALIZE_ENUM(
    NodeAttributes,
    {{NodeAttributes::BatchnormInferenceAttributes, "BatchnormInferenceAttributes"},
     {NodeAttributes::PointwiseAttributes, "PointwiseAttributes"},
     {NodeAttributes::BatchnormBackwardAttributes, "BatchnormBackwardAttributes"},
     {NodeAttributes::BatchnormAttributes, "BatchnormAttributes"},
     {NodeAttributes::ConvolutionFwdAttributes, "ConvolutionFwdAttributes"},
     {NodeAttributes::ConvolutionBwdAttributes, "ConvolutionBwdAttributes"},
     {NodeAttributes::ConvolutionWrwAttributes, "ConvolutionWrwAttributes"},
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
    default:
        throw std::runtime_error(
            "hipdnn_sdk::data_objects::to_json(Node): Unsupported NodeAttributes type: "
            + std::to_string(static_cast<int8_t>(node.attributes_type())));
    }
    nodeJson["name"] = node.name()->c_str();
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
    graphJson["name"] = graph.name()->c_str();
    graphJson["tensors"] = graph.tensors();
}

}
namespace hipdnn_sdk::json
{
template <>
inline auto to<data_objects::Node>(flatbuffers::FlatBufferBuilder& builder,
                                   const nlohmann::json& entry)
{
    auto type = entry.at("type").get<data_objects::NodeAttributes>();
    auto name = entry.at("name").get<std::string>();
    auto computeDataType = entry.at("compute_data_type").get<data_objects::DataType>();

    flatbuffers::Offset<void> node = [&]() {
        switch(type)
        {
        case data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return to<data_objects::BatchnormInferenceAttributes>(builder, entry).Union();
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
        default:
            throw std::runtime_error(
                "hipdnn_sdk::json::to<data_objects::Node>(): Unsupported NodeAttributes type: "
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

    auto nodes = toVector<Node>(builder, entry.at("nodes"));
    auto tensors = toVector<TensorAttributes>(builder, entry.at("tensors"));
    return data_objects::CreateGraphDirect(
        builder, name.c_str(), computeType, intermediateType, ioType, &tensors, &nodes);
}

}
