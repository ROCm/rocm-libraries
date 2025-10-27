// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/json/BatchnormAttributes.hpp>
#include <hipdnn_sdk/utilities/json/BatchnormBackwardAttributes.hpp>
#include <hipdnn_sdk/utilities/json/BatchnormInferenceAttributes.hpp>
#include <hipdnn_sdk/utilities/json/Common.hpp>
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
     {NodeAttributes::NONE, ""}})

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
    default:
        throw std::runtime_error(
            "hipdnn_sdk::data_objects::to_json(Node): Unsupported NodeAttributes type: "
            + std::to_string(static_cast<int8_t>(node.attributes_type())));
    }
    nodeJson["name"] = node.name()->c_str();
    nodeJson["type"] = node.attributes_type();
}

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& graphJson, const data_objects::Graph& graph)
{
    graphJson["nodes"] = graph.nodes();
    graphJson["compute_type"] = graph.compute_type();
    graphJson["io_type"] = graph.io_type();
    graphJson["intermediate_type"] = graph.intermediate_type();
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
        default:
            throw std::runtime_error(
                "hipdnn_sdk::json::to<data_objects::Node>(): Unsupported NodeAttributes type: "
                + std::string{EnumNameNodeAttributes(type)});
        }
    }();

    return data_objects::CreateNodeDirect(builder, name.c_str(), type, node);
}

template <>
inline auto to<data_objects::Graph>(flatbuffers::FlatBufferBuilder& builder,
                                    const nlohmann::json& entry)
{
    using namespace data_objects;
    using namespace flatbuffers;

    auto name = entry.at("name").get<std::string>();
    auto computeType = entry.at("compute_type").get<data_objects::DataType>();
    auto ioType = entry.at("io_type").get<data_objects::DataType>();
    auto intermediateType = entry.at("intermediate_type").get<data_objects::DataType>();

    auto nodes = toVector<Node>(builder, entry.at("nodes"));
    auto tensors = toVector<TensorAttributes>(builder, entry.at("tensors"));
    return data_objects::CreateGraphDirect(
        builder, name.c_str(), computeType, intermediateType, ioType, &tensors, &nodes);
}

}
