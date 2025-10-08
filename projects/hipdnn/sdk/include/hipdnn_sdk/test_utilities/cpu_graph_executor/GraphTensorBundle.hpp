// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>
#include <vector>

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
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

            auto tensor = createTensorFromAttribute(*attr);
            tensors.emplace(id, std::move(tensor));
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
