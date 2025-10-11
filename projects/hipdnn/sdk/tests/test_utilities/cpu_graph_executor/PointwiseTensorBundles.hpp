// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
struct PointwiseUnaryTensorBundle
{
    PointwiseUnaryTensorBundle(const std::vector<int64_t>& inputDims,
                               const std::vector<int64_t>& outputDims,
                               unsigned int seed = 1,
                               const TensorLayout& layout = TensorLayout::NCHW)
        : inputTensor(inputDims, layout)
        , outputTensor(outputDims, layout)
    {
        inputTensor.fillWithRandomValues(
            static_cast<InputType>(-2.0f), static_cast<InputType>(2.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& inputTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& outputTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[inputTensorAttr.get_uid()] = inputTensor.memory().hostData();
        variantPack[outputTensorAttr.get_uid()] = outputTensor.memory().hostData();
        return variantPack;
    }

    Tensor<InputType> inputTensor;
    Tensor<InputType> outputTensor;
};

template <typename InputType>
struct PointwiseBinaryTensorBundle
{
    PointwiseBinaryTensorBundle(const std::vector<int64_t>& input1Dims,
                                const std::vector<int64_t>& input2Dims,
                                const std::vector<int64_t>& outputDims,
                                unsigned int seed = 1,
                                const TensorLayout& layout = TensorLayout::NCHW)
        : input1Tensor(input1Dims, layout)
        , input2Tensor(input2Dims, layout)
        , outputTensor(outputDims, layout)
    {
        input1Tensor.fillWithRandomValues(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);
        input2Tensor.fillWithRandomValues(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed + 1);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& input1TensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& input2TensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& outputTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[input1TensorAttr.get_uid()] = input1Tensor.memory().hostData();
        variantPack[input2TensorAttr.get_uid()] = input2Tensor.memory().hostData();
        variantPack[outputTensorAttr.get_uid()] = outputTensor.memory().hostData();
        return variantPack;
    }

    Tensor<InputType> input1Tensor;
    Tensor<InputType> input2Tensor;
    Tensor<InputType> outputTensor;
};

}
