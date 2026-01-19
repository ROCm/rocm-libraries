// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
struct MatmulTensorBundle
{
    MatmulTensorBundle(const std::vector<int64_t>& aDims,
                       const std::vector<int64_t>& bDims,
                       const std::vector<int64_t>& cDims,
                       unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed())
        : aTensor(aDims)
        , bTensor(bDims)
        , cTensor(cDims)
    {
        aTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        bTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& aTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& bTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& cTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[aTensorAttr.get_uid()] = aTensor.memory().hostData();
        variantPack[bTensorAttr.get_uid()] = bTensor.memory().hostData();
        variantPack[cTensorAttr.get_uid()] = cTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<InputType> aTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> bTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> cTensor;
};

}
