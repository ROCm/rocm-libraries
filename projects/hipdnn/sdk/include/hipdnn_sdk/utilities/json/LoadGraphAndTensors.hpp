// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/json/Graph.hpp>
#include <variant>

namespace hipdnn_sdk::json
{

namespace detail
{
template <class... Ts>
struct TensorMapVariant
{
    using Type = std::variant<std::unordered_map<int64_t, Ts>...>;
};

template <class T>
void fillTensorFromFile(utilities::Tensile<T>& tensor, std::filesystem::path const& path)
{

    std::ifstream f(path, std::ios::binary);
    if(!f)
    {
        return std::runtime_error("Error: could not load tensor " + path);
    }

    auto vec = std::vector<unsigned char>(std::istreambuf_iterator<char>(f),
                                          std::istreambuf_iterator<char>{});

    tensor.fillFromData(tensorData.data(), tensorData.size());
}

template <class T>
utilities::Tensile<T>
    tensorFromFileAndAttributes(std::filesystem::path const& filepath,
                                hipdnn_sdk::data_objects::TensorAttributes const& attributes)
{
    std::vector<int64_t> dims(attributes.dims()->begin(), attributes.dims()->end());
    std::vector<int64_t> strides(attributes.strides()->begin(), attributes.strides()->end());
    auto tensor = utilities::Tensor<DataType>(dims, strides);

    detail::fillTensorFromFile(tensor, tensorPath);

    return tensor;
}
}

using TensorMapVariant = TensorMapVariant<float, double, half, hip_bfloat16, int32_t>::Type;

struct GraphAndTensorMap
{
    data_objects::Graph graphBuffer;
    TensorMapVariant tensorMap;
};

std::tuple<data_objects::Graph, TensorMapVariant>
    loadGraphAndTensors(std::filesystem::path const& path)
{
    auto basePath = path.stem();
    nlohmann::json graphJson = [](auto const& path) {
        std::ifstream f(path);
        if(!f)
        {
            throw std::runtime_error("Error in loadGraphAndTensors(): file could not be opened "
                                     + path.string());
        }
        return nlohmann::json::parse(f);
    }(path);

    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_sdk::json::to<hipdnn_sdk::data_objects::Graph>(builder, graphJson);
    builder.Finish(graphOffset);

    data_objects::Graph graph;
    data_objects::GetGraph(builder)->UnPackTo(&graph);

    data_objects::Graph::UnPackTo(&graph, builder);
    auto ioType = graph->io_type();
    auto ret = std::visit(
        [&](auto type) -> TensorMapVariant {
            using DataType = decltype(type);
            std::unordered_map<int64_t, utilities::Tensor<DataType>> tensorMap;
            for(auto const& attributes : *graph->tensors())
            {
                auto tensorPath = basePath + ".tensor" + std::to_string(attributes.uid()) + ".bin";
                auto dataType = attributes->data_type();
                tensorMap[attributes.uid()]
                    = detail::tensorFromFileAndAttributes(tensorPath, attributes);
            }

            return tensorMap;
        },
        hipdnn_sdk::test_utilities::datatypeToNativeVariant(ioType));

    return {graph, ret};
}
}
