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
struct ConvolutionFwdTensorBundle
{
    ConvolutionFwdTensorBundle(const std::vector<int64_t>& xDims,
                               const std::vector<int64_t>& wDims,
                               const std::vector<int64_t>& yDims,
                               unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed(),
                               const hipdnn_data_sdk::utilities::TensorLayout& layout
                               = hipdnn_data_sdk::utilities::TensorLayout::NCHW)
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

    hipdnn_data_sdk::utilities::Tensor<InputType> xTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> wTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> yTensor;
};

template <typename InputDataType>
struct ConvolutionBwdTensorBundle
{
    ConvolutionBwdTensorBundle(const std::vector<int64_t>& dxDims,
                               const std::vector<int64_t>& wDims,
                               const std::vector<int64_t>& dyDims,
                               unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed(),
                               const hipdnn_data_sdk::utilities::TensorLayout& layout
                               = hipdnn_data_sdk::utilities::TensorLayout::NCHW)
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
    hipdnn_data_sdk::utilities::Tensor<InputDataType> dxTensor;
    hipdnn_data_sdk::utilities::Tensor<InputDataType> wTensor;
    hipdnn_data_sdk::utilities::Tensor<InputDataType> dyTensor;
};

template <typename InputDataType>
struct ConvolutionWrwTensorBundle
{
    ConvolutionWrwTensorBundle(const std::vector<int64_t>& xDims,
                               const std::vector<int64_t>& dwDims,
                               const std::vector<int64_t>& dyDims,
                               unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed(),
                               const hipdnn_data_sdk::utilities::TensorLayout& layout
                               = hipdnn_data_sdk::utilities::TensorLayout::NCHW)
        : xTensor(xDims, layout)
        , dwTensor(dwDims, layout)
        , dyTensor(dyDims, layout)
    {
        dyTensor.fillWithRandomValues(
            static_cast<InputDataType>(-1.0f), static_cast<InputDataType>(1.0f), seed);

        xTensor.fillWithRandomValues(
            static_cast<InputDataType>(-1.0f), static_cast<InputDataType>(1.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dwTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dyTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;

        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[dwTensorAttr.get_uid()] = dwTensor.memory().hostData();
        variantPack[dyTensorAttr.get_uid()] = dyTensor.memory().hostData();

        return variantPack;
    }

    std::vector<int64_t> derivedDims;
    hipdnn_data_sdk::utilities::Tensor<InputDataType> xTensor;
    hipdnn_data_sdk::utilities::Tensor<InputDataType> dwTensor;
    hipdnn_data_sdk::utilities::Tensor<InputDataType> dyTensor;
};

}
