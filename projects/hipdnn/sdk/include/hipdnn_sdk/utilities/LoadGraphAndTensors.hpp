// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Visitor.hpp>
#include <hipdnn_sdk/utilities/json/Graph.hpp>
#include <type_traits>
#include <variant>

namespace hipdnn_sdk::utilities
{

namespace detail
{

template <class T>
struct DatatypeFromTensor
{
};

template <class T>
struct DatatypeFromTensor<Tensor<T>>
{
    using Type = T;
};

inline void fillTensorFromFile(ITensor& tensor, const std::filesystem::path& path)
{

    std::ifstream fileInputStream(path, std::ios::binary);
    if(!fileInputStream)
    {
        throw std::runtime_error("Error: could not load tensor " + path.string());
    }

    auto vec = std::vector<unsigned char>(std::istreambuf_iterator<char>(fileInputStream),
                                          std::istreambuf_iterator<char>{});

    tensor.fillWithData(vec.data(), vec.size());
}
}

template <class T>
using DataTypeFromTensor =
    typename detail::DatatypeFromTensor<std::remove_cv_t<std::remove_reference_t<T>>>::Type;

inline std::unique_ptr<ITensor>
    tensorFromFileAndAttributes(const std::filesystem::path& filepath,
                                const hipdnn_sdk::data_objects::TensorAttributes& attributes)
{
    auto tensor = hipdnn_sdk::test_utilities::createTensorFromAttribute(attributes);
    detail::fillTensorFromFile(*tensor, filepath);

    return tensor;
}

struct GraphAndTensorMap
{
    flatbuffers::DetachedBuffer graphBuffer;
    std::unordered_map<int64_t, std::unique_ptr<ITensor>> tensorMap;
    std::vector<int64_t> outputTensorUids;

    const data_objects::Graph& graph() const
    {
        return *data_objects::GetGraph(graphBuffer.data());
    }

    hipdnn_plugin::GraphWrapper createGraphWrapper() const
    {
        return hipdnn_plugin::GraphWrapper{graphBuffer.data(), graphBuffer.size()};
    }

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers()
    {
        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        for(auto& [uid, tensor] : tensorMap)
        {
            hipdnnPluginDeviceBuffer_t deviceBuffer{uid, tensor->rawDeviceData()};
            deviceBuffers.push_back(deviceBuffer);
        }
        return deviceBuffers;
    }

    std::unordered_map<int64_t, std::unique_ptr<ITensor>> extractAndClearOutputTensorData()
    {
        std::unordered_map<int64_t, std::unique_ptr<ITensor>> outputTensorMap;

        auto tensorAttributeMap = createGraphWrapper().getTensorMap();
        for(int64_t uid : outputTensorUids)
        {
            auto dataType = tensorAttributeMap[uid]->data_type();
            auto& outputTensorPtr = tensorMap[uid];

            auto zeroedTensorPtr = std::visit(
                [&](auto dataType) {
                    using DataType = decltype(dataType);
                    auto tensorPtr = std::unique_ptr<ITensor>(
                        new Tensor<DataType>(outputTensorPtr->dims(), outputTensorPtr->strides()));
                    tensorPtr->fillTensorWithValue(0.f);
                    return tensorPtr;
                },
                test_utilities::datatypeToNativeVariant(dataType));

            std::swap(zeroedTensorPtr, outputTensorPtr);

            outputTensorMap[uid] = std::move(zeroedTensorPtr);
        }

        return outputTensorMap;
    }

    bool validateTensors(std::unordered_map<int64_t, std::unique_ptr<ITensor>>& referenceTensors,
                         float absTolerance,
                         float relTolerance)
    {
        auto tensorAttributeMap = createGraphWrapper().getTensorMap();

        for(auto& mapPair : referenceTensors)
        {
            auto uid = mapPair.first;
            auto& referenceTensorPtr = mapPair.second;

            auto dataType = tensorAttributeMap[uid]->data_type();
            auto validatorFunc = Visitor{
                [&](auto dataType) {
                    using DataType = decltype(dataType);

                    auto validator = test_utilities::CpuFpReferenceValidation<DataType>{
                        static_cast<DataType>(absTolerance), static_cast<DataType>(relTolerance)};
                    return validator.allClose(*referenceTensorPtr, *tensorMap.at(uid));
                },
                [&](int) {
                    throw std::runtime_error("validateTensors: Cannot validate integer tensors");
                    return false;
                }};
            bool passedValidation
                = std::visit(validatorFunc, test_utilities::datatypeToNativeVariant(dataType));
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

    nlohmann::json graphJson = [](const auto& path) {
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
        = hipdnn_sdk::json::to<hipdnn_sdk::data_objects::Graph>(graphBuilder, graphJson);
    graphBuilder.Finish(graphOffset);

    auto graph = data_objects::GetGraph(graphBuilder.GetBufferPointer());

    auto outputTensorUids = getOutputTensorUidsFromGraph(graphJson);

    std::unordered_map<int64_t, std::unique_ptr<ITensor>> tensorMap;

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
}
