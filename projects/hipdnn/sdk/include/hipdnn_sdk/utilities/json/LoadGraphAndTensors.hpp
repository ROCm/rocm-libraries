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

namespace hipdnn_sdk::json
{

template <class T>
using TensorMap = std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<T>>>;

namespace detail
{

template <class... Ts>
struct TensorMapVariant
{
    using Type = std::variant<TensorMap<Ts>...>;
};

template <class T>
struct DataTypeFromTensorMap
{
};

template <class T>
struct DataTypeFromTensorMap<
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<T>>>>
{
    using Type = T;
};

template <class T>
void fillTensorFromFile(utilities::Tensor<T>& tensor, std::filesystem::path const& path)
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

template <class T>
std::unique_ptr<utilities::Tensor<T>>
    tensorFromFileAndAttributes(std::filesystem::path const& filepath,
                                hipdnn_sdk::data_objects::TensorAttributes const& attributes)
{
    std::vector<int64_t> dims(attributes.dims()->begin(), attributes.dims()->end());
    std::vector<int64_t> strides(attributes.strides()->begin(), attributes.strides()->end());
    auto tensor = std::make_unique<utilities::Tensor<T>>(dims, strides);

    detail::fillTensorFromFile(*tensor, filepath);

    return tensor;
}
}

using TensorMapVariant =
    typename detail::TensorMapVariant<float, double, half, hip_bfloat16, int32_t>::Type;
// using TensorMapVariant = std::variant<
//     std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<float>>>,
//     std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<double>>>,
//     std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<half>>>,
//     std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<hip_bfloat16>>>,
//     std::unordered_map<int64_t, std::unique_ptr<hipdnn_sdk::utilities::Tensor<int32_t>>>>;

template <class T>
using DataTypeFromTensorMap =
    typename detail::DataTypeFromTensorMap<std::remove_cv_t<std::remove_reference_t<T>>>::Type;

struct GraphAndTensorMap
{
    flatbuffers::DetachedBuffer graphBuffer;
    TensorMapVariant tensorMap;

    const data_objects::Graph& graph() const
    {
        return *data_objects::GetGraph(graphBuffer.data());
    }

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers()
    {
        return std::visit(
            []([[maybe_unused]] auto const& tensorMapIn) {
                std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

                // Iterating over this loop triggers the portability-template-virtual-member-function tidy for Allocator
                // Need to figure out why
                for(auto& [uid, tensor] : tensorMapIn)
                {
                    hipdnnPluginDeviceBuffer_t deviceBuffer;
                    deviceBuffer.uid = uid;
                    deviceBuffer.ptr = tensor->memory().deviceData();
                    deviceBuffers.push_back(deviceBuffer);
                }
                return deviceBuffers;
            },
            tensorMap);
    }
};

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

    auto ioType = graph->io_type();
    auto ret = std::visit(
        [&](auto type) -> TensorMapVariant {
            using DataType = decltype(type);
            std::unordered_map<int64_t, std::unique_ptr<utilities::Tensor<DataType>>> tensorMap;
            for(auto attributes : *graph->tensors())
            {
                auto tensorPath
                    = basePath.string() + ".tensor" + std::to_string(attributes->uid()) + ".bin";
                tensorMap[attributes->uid()]
                    = detail::tensorFromFileAndAttributes<DataType>(tensorPath, *attributes);
            }

            return tensorMap;
        },
        hipdnn_sdk::test_utilities::datatypeToNativeVariant(ioType));

    return {graphBuilder.Release(), std::move(ret)};
}
}
