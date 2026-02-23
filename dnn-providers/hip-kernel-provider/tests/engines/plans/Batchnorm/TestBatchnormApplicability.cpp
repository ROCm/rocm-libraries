// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestBatchnormApplicability.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>

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

std::pair<flatbuffers::FlatBufferBuilder,
          std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>>
    buildTensorMapFromConfigs(const std::vector<TensorConfig>& configs)
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

TensorConfig createIoTensor(int64_t uid,
                            const std::string& name,
                            hipdnn_data_sdk::data_objects::DataType dt,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            bool isVirtual)
{
    auto builder = TensorConfigBuilder(uid, name, TensorRole::IO)
                       .withDataType(dt)
                       .withDims(dims)
                       .withStrides(strides);

    if(isVirtual)
    {
        builder.asVirtual();
    }
    return builder.build();
}
