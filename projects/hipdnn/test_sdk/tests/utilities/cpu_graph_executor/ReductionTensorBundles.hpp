// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
struct ReductionTensorBundle
{
    ReductionTensorBundle(const std::vector<int64_t>& xDims,
                          const std::vector<int64_t>& yDims,
                          unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed())
        : xTensor(xDims)
        , yTensor(yDims)
    {
        xTensor.fillWithRandomValues(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& yTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[yTensorAttr.get_uid()] = yTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<InputType> xTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> yTensor;
};

} // namespace hipdnn_sdk_test_utils
