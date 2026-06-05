// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

struct GraphTensorBundle
{
    GraphTensorBundle() = default;

    GraphTensorBundle(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_())
            {
                continue;
            }

            auto tensor = detail::createTensorFromAttribute(*attr);
            tensors.emplace(id, std::move(tensor));
        }
    }

    void randomizeTensor(int64_t uid, float min, float max, unsigned int seed)
    {
        auto it = tensors.find(uid);
        if(it == tensors.end())
        {
            throw std::runtime_error("Tensor with uid " + std::to_string(uid) + " not found");
        }
        it->second->fillTensorWithRandomValues(min, max, seed);
    }

    std::unordered_map<int64_t, void*> toHostVariantPack()
    {
        std::unordered_map<int64_t, void*> variantPack;
        for(auto& [id, tensorPtr] : tensors)
        {
            variantPack[id] = tensorPtr->rawHostData();
        }
        return variantPack;
    }

    std::unordered_map<int64_t, void*> toDeviceVariantPack()
    {
        std::unordered_map<int64_t, void*> variantPack;
        for(auto& [id, tensorPtr] : tensors)
        {
            variantPack[id] = tensorPtr->rawDeviceData();
        }
        return variantPack;
    }

    hipdnn_data_sdk::utilities::ITensor& getTensor(int64_t uid)
    {
        auto it = tensors.find(uid);
        if(it == tensors.end())
        {
            throw std::runtime_error(
                "GraphTensorBundle: tensor with uid " + std::to_string(uid)
                + " not found. The tensor may be marked as virtual in the graph"
                  " (virtual tensors are skipped during bundle construction).");
        }
        return *it->second;
    }

    const hipdnn_data_sdk::utilities::ITensor& getTensor(int64_t uid) const
    {
        auto it = tensors.find(uid);
        if(it == tensors.end())
        {
            throw std::runtime_error(
                "GraphTensorBundle: tensor with uid " + std::to_string(uid)
                + " not found. The tensor may be marked as virtual in the graph"
                  " (virtual tensors are skipped during bundle construction).");
        }
        return *it->second;
    }

    bool isOutput(int64_t id) const
    {
        return outputTensorIds.find(id) != outputTensorIds.end();
    }

    void sentinelFillOutputTensors()
    {
        for(auto id : outputTensorIds)
        {
            auto it = tensors.find(id);
            if(it != tensors.end())
            {
                it->second->fillWithSentinelValue();
            }
        }
    }

    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> tensors;
    std::unordered_set<int64_t> outputTensorIds;
};

}
