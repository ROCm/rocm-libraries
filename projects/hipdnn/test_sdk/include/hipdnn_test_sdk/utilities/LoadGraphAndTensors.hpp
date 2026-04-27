// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Visitor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#endif
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/TensorFileUtils.hpp>
#include <type_traits>
#include <variant>

namespace hipdnn_test_sdk::utilities
{

inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor> tensorFromFileAndAttributes(
    const std::filesystem::path& filepath,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attributes)
{
    auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(attributes);
    hipdnn_test_sdk::detail::fillTensorFromFile(*tensor, filepath);

    return tensor;
}

struct GraphAndTensorMap
{
    flatbuffers::DetachedBuffer graphBuffer;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> tensorMap;
    std::vector<int64_t> outputTensorUids;

    const hipdnn_flatbuffers_sdk::data_objects::Graph& graph() const
    {
        return *hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuffer.data());
    }

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper createGraphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper{graphBuffer.data(),
                                                                          graphBuffer.size()};
    }

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers()
    {
        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        for(auto& [uid, tensor] : tensorMap)
        {
            const hipdnnPluginDeviceBuffer_t deviceBuffer{uid, tensor->rawDeviceData()};
            deviceBuffers.push_back(deviceBuffer);
        }
        return deviceBuffers;
    }

    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
        extractAndClearOutputTensorData()
    {
        std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
            outputTensorMap;

        auto tensorAttributeMap = createGraphWrapper().getTensorMap();
        for(const int64_t uid : outputTensorUids)
        {
            auto dataType = tensorAttributeMap[uid]->data_type();
            auto& outputTensorPtr = tensorMap[uid];

            auto zeroedTensorPtr = std::visit(
                [&](auto dataType) {
                    using DataType = decltype(dataType);
                    auto tensorPtr = std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>(
                        new hipdnn_data_sdk::utilities::Tensor<DataType>(
                            outputTensorPtr->dims(), outputTensorPtr->strides()));
                    tensorPtr->fillTensorWithValue(0.f);
                    return tensorPtr;
                },
                hipdnn_test_sdk::utilities::datatypeToNativeVariant(dataType));

            std::swap(zeroedTensorPtr, outputTensorPtr);

            outputTensorMap[uid] = std::move(zeroedTensorPtr);
        }

        return outputTensorMap;
    }

    bool validateTensors(
        std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>&
            referenceTensors,
        float absTolerance,
        float relTolerance)
    {
        auto tensorAttributeMap = createGraphWrapper().getTensorMap();

        for(auto& mapPair : referenceTensors)
        {
            auto uid = mapPair.first;
            auto& referenceTensorPtr = mapPair.second;

            auto dataType = tensorAttributeMap[uid]->data_type();
            auto validatorFunc = hipdnn_data_sdk::utilities::Visitor{
                [&](auto dataType) {
                    using DataType = decltype(dataType);

                    auto validator = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<DataType>{
                        absTolerance, relTolerance};
                    return validator.allClose(*referenceTensorPtr, *tensorMap.at(uid));
                },
                [&](int) {
                    throw std::runtime_error("validateTensors: Cannot validate integer tensors");
                    return false;
                }};
            const bool passedValidation = std::visit(
                validatorFunc, hipdnn_test_sdk::utilities::datatypeToNativeVariant(dataType));
            if(!passedValidation)
            {
                return false;
            }
        }

        return true;
    }

    std::unordered_map<int64_t, void*> hostBufferMap()
    {
        std::unordered_map<int64_t, void*> bufferMap;
        for(auto& [uid, tensor] : tensorMap)
        {
            bufferMap[uid] = tensor->rawHostData();
        }

        return bufferMap;
    }
};

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

inline std::vector<int64_t> getOutputTensorUidsFromGraph(nlohmann::json graph)
{
    std::vector<int64_t> outputTensorUids;

    for(const auto& node : graph.at("nodes"))
    {
        for(auto& [name, value] : node.at("outputs").items())
        {
            if(name.find("_tensor_uid") == std::string::npos)
            {
                continue;
            }

            outputTensorUids.push_back(value.get<int64_t>());
        }
    }

    return outputTensorUids;
}

inline GraphAndTensorMap loadGraphAndTensors(const std::filesystem::path& path)
{
    auto basePath = path;
    basePath.replace_extension();

    const nlohmann::json graphJson = [](const auto& path) {
        std::ifstream graphFileStream(path);
        if(!graphFileStream)
        {
            throw std::runtime_error("Error in loadGraphAndTensors(): file could not be opened "
                                     + path.string());
        }
        return nlohmann::json::parse(graphFileStream);
    }(path);

    flatbuffers::FlatBufferBuilder graphBuilder;
    auto graphOffset
        = hipdnn_flatbuffers_sdk::json::to<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            graphBuilder, graphJson);
    graphBuilder.Finish(graphOffset);

    auto graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

    auto outputTensorUids = getOutputTensorUidsFromGraph(graphJson);

    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> tensorMap;

    if(graph->tensors() == nullptr || graph->tensors()->empty())
    {
        throw std::runtime_error("Graph needs to include at least one tensor");
    }
    for(auto attributes : *graph->tensors())
    {
        auto tensorPath
            = basePath.string() + ".tensor" + std::to_string(attributes->uid()) + ".bin";
        tensorMap[attributes->uid()] = tensorFromFileAndAttributes(tensorPath, *attributes);
    }

    return {graphBuilder.Release(), std::move(tensorMap), outputTensorUids};
}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

}
