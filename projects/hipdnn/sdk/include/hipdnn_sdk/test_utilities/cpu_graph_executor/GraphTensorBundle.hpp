// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>
#include <vector>

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk::test_utilities
{

struct GraphTensorBundle
{
    GraphTensorBundle(
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_())
            {
                continue;
            }

            auto dims = convertFlatBufferVectorToStdVector(attr->dims());
            auto strides = convertFlatBufferVectorToStdVector(attr->strides());

            switch(attr->data_type())
            {
            case hipdnn_sdk::data_objects::DataType::FLOAT:
                tensors.emplace(id, std::make_unique<utilities::Tensor<float>>(dims, strides));
                break;
            case hipdnn_sdk::data_objects::DataType::HALF:
                tensors.emplace(id, std::make_unique<utilities::Tensor<half>>(dims, strides));
                break;
            case hipdnn_sdk::data_objects::DataType::BFLOAT16:
                tensors.emplace(id,
                                std::make_unique<utilities::Tensor<hip_bfloat16>>(dims, strides));
                break;
            default:
                throw std::runtime_error("Unsupported data type in GraphTensorBundle");
            }
        }
    }

    static void fillTensorWithRandomValues(std::unique_ptr<utilities::ITensor>& tensor,
                                           float minValue,
                                           float maxValue,
                                           unsigned int seed = 1.0f)
    {
        if(tensor->isType<float>())
        {
            auto* typedTensor = static_cast<utilities::TensorBase<float>*>(tensor.get());
            typedTensor->fillWithRandomValues(minValue, maxValue, seed);
        }
        else if(tensor->isType<half>())
        {
            auto* typedTensor = static_cast<utilities::TensorBase<half>*>(tensor.get());
            typedTensor->fillWithRandomValues(
                static_cast<half>(minValue), static_cast<half>(maxValue), seed);
        }
        else if(tensor->isType<hip_bfloat16>())
        {
            auto* typedTensor = static_cast<utilities::TensorBase<hip_bfloat16>*>(tensor.get());
            typedTensor->fillWithRandomValues(
                static_cast<hip_bfloat16>(minValue), static_cast<hip_bfloat16>(maxValue), seed);
        }
        else
        {
            throw std::runtime_error("Unsupported data type in fillTensorWithRandomValues");
        }
    }

    std::unordered_map<int64_t, void*> toVariantPack()
    {
        std::unordered_map<int64_t, void*> variantPack;
        for(auto& [id, tensorPtr] : tensors)
        {
            variantPack[id] = tensorPtr->rawHostData();
        }
        return variantPack;
    }

    std::unordered_map<int64_t, std::unique_ptr<utilities::ITensor>> tensors;
};

}
