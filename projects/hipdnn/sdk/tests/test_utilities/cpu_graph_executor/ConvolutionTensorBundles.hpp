// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
struct ConvolutionFwdTensorBundle
{
    ConvolutionFwdTensorBundle(const std::vector<int64_t>& xDims,
                               const std::vector<int64_t>& wDims,
                               const std::vector<int64_t>& yDims,
                               unsigned int seed = hipdnn_sdk::test_utilities::getGlobalTestSeed(),
                               const TensorLayout& layout = TensorLayout::NCHW)
        : xTensor(xDims, layout)
        , wTensor(wDims, layout)
        , yTensor(yDims, layout)
    {
        xTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        wTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& wTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& yTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[wTensorAttr.get_uid()] = wTensor.memory().hostData();
        variantPack[yTensorAttr.get_uid()] = yTensor.memory().hostData();
        return variantPack;
    }

    Tensor<InputType> xTensor;
    Tensor<InputType> wTensor;
    Tensor<InputType> yTensor;
};

template <typename InputDataType>
struct ConvolutionBwdTensorBundle
{
    ConvolutionBwdTensorBundle(const std::vector<int64_t>& dxDims,
                               const std::vector<int64_t>& wDims,
                               const std::vector<int64_t>& dyDims,
                               unsigned int seed = hipdnn_sdk::test_utilities::getGlobalTestSeed(),
                               const TensorLayout& layout = TensorLayout::NCHW)
        : dxTensor(dxDims, layout)
        , wTensor(wDims, layout)
        , dyTensor(dyDims, layout)
    {
        dyTensor.fillWithRandomValues(
            static_cast<InputDataType>(-1.0f), static_cast<InputDataType>(1.0f), seed);

        wTensor.fillWithRandomValues(
            static_cast<InputDataType>(-1.0f), static_cast<InputDataType>(1.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& dxTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& wTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dyTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;

        variantPack[dxTensorAttr.get_uid()] = dxTensor.memory().hostData();
        variantPack[wTensorAttr.get_uid()] = wTensor.memory().hostData();
        variantPack[dyTensorAttr.get_uid()] = dyTensor.memory().hostData();

        return variantPack;
    }

    std::vector<int64_t> derivedDims;
    Tensor<InputDataType> dxTensor;
    Tensor<InputDataType> wTensor;
    Tensor<InputDataType> dyTensor;
};

}
