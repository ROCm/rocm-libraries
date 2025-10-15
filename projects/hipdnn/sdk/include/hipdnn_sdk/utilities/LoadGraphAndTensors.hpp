// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/json/Graph.hpp>
#include <type_traits>
#include <variant>

namespace hipdnn_sdk::utilities
{

namespace detail
{

template <class... Ts>
struct TensorVariant
{
    using Type = std::variant<std::unique_ptr<Tensor<Ts>>...>;
};

template <class T>
struct DatatypeFromTensor
{
};

template <class T>
struct DatatypeFromTensor<Tensor<T>>
{
    using Type = T;
};

template <class T>
void fillTensorFromFile(Tensor<T>& tensor, std::filesystem::path const& path)
{

    std::ifstream f(path, std::ios::binary);
    if(!f)
    {
        throw std::runtime_error("Error: could not load tensor " + path.string());
    }

    auto vec = std::vector<unsigned char>(std::istreambuf_iterator<char>(f),
                                          std::istreambuf_iterator<char>{});

    tensor.fillWithData(reinterpret_cast<T*>(vec.data()), vec.size() / sizeof(T));
}
}

using TensorVariant = detail::TensorVariant<float, double, half, hip_bfloat16, int32_t>::Type;

using TensorVariantMap = std::unordered_map<int64_t, TensorVariant>;

template <class T>
using DataTypeFromTensor =
    typename detail::DatatypeFromTensor<std::remove_cv_t<std::remove_reference_t<T>>>::Type;

inline TensorVariant
    tensorFromFileAndAttributes(std::filesystem::path const& filepath,
                                hipdnn_sdk::data_objects::TensorAttributes const& attributes)
{
    std::vector<int64_t> dims(attributes.dims()->begin(), attributes.dims()->end());
    std::vector<int64_t> strides(attributes.strides()->begin(), attributes.strides()->end());

    auto createTensor = [&](auto dataType) -> TensorVariant {
        using DataType = std::remove_const_t<decltype(dataType)>;

        auto tensor = std::make_unique<Tensor<DataType>>(dims, strides);

        detail::fillTensorFromFile(*tensor, filepath);

        return std::move(tensor);
    };

    return std::visit(createTensor,
                      test_utilities::datatypeToNativeVariant(attributes.data_type()));
}

struct GraphAndTensorMap
{
    flatbuffers::DetachedBuffer graphBuffer;
    TensorVariantMap tensorMap;
    std::vector<int64_t> outputTensorUids;

    const data_objects::Graph& graph() const
    {
        return *data_objects::GetGraph(graphBuffer.data());
    }

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers()
    {

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        // Iterating over this loop triggers the portability-template-virtual-member-function tidy for Allocator
        // Need to figure out why
        for(auto& [uid, tensorVariant] : tensorMap)
        {
            hipdnnPluginDeviceBuffer_t deviceBuffer;
            deviceBuffer.uid = uid;
            std::visit([&](auto& tensor) { deviceBuffer.ptr = tensor->memory().deviceData(); },
                       tensorVariant);
            deviceBuffers.push_back(deviceBuffer);
        }
        return deviceBuffers;
    }

    std::unordered_map<int64_t, void*> hostBufferMap()
    {
        std::unordered_map<int64_t, void*> bufferMap;
        for(auto& [uid, tensorVariant] : tensorMap)
        {
            bufferMap[uid] = std::visit(
                [](auto& tensor) -> void* { return tensor->memory().hostData(); }, tensorVariant);
        }

        return bufferMap;
    }
};

inline std::vector<int64_t> getOutputTensorUidsFromGraph(nlohmann::json graph)
{
    std::vector<int64_t> outputTensorUids;

    for(auto const& node : graph.at("nodes"))
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

inline GraphAndTensorMap loadGraphAndTensors(std::filesystem::path const& path)
{
    auto basePath = path;
    basePath.replace_extension();

    nlohmann::json graphJson = [](auto const& path) {
        std::ifstream f(path);
        if(!f)
        {
            throw std::runtime_error("Error in loadGraphAndTensors(): file could not be opened "
                                     + path.string());
        }
        return nlohmann::json::parse(f);
    }(path);

    flatbuffers::FlatBufferBuilder graphBuilder;
    auto graphOffset
        = hipdnn_sdk::json::to<hipdnn_sdk::data_objects::Graph>(graphBuilder, graphJson);
    graphBuilder.Finish(graphOffset);

    auto graph = data_objects::GetGraph(graphBuilder.GetBufferPointer());

    auto outputTensorUids = getOutputTensorUidsFromGraph(graphJson);

    std::unordered_map<int64_t, TensorVariant> tensorMap;
    for(auto attributes : *graph->tensors())
    {
        auto tensorPath
            = basePath.string() + ".tensor" + std::to_string(attributes->uid()) + ".bin";
        tensorMap[attributes->uid()] = tensorFromFileAndAttributes(tensorPath, *attributes);
    }

    return {graphBuilder.Release(), std::move(tensorMap), outputTensorUids};
}
}
