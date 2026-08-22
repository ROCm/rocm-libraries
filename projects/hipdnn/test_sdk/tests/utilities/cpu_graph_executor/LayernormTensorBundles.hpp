// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

namespace hipdnn_sdk_test_utils
{

struct LayernormFpropTensorBundle : public hipdnn_test_sdk::utilities::GraphTensorBundle
{
    LayernormFpropTensorBundle(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : hipdnn_test_sdk::utilities::GraphTensorBundle(tensorMap)
    {
        const auto& attributes
            = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();

        randomizeTensor(attributes.x_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.scale_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.bias_tensor_uid(), 0.0f, 1.0f, seed);
    }
};

struct LayernormBpropTensorBundle : public hipdnn_test_sdk::utilities::GraphTensorBundle
{
    LayernormBpropTensorBundle(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : hipdnn_test_sdk::utilities::GraphTensorBundle(tensorMap)
    {
        const auto& attributes = node.attributesAs<
            hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();

        randomizeTensor(attributes.dy_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.x_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.scale_tensor_uid(), 0.0f, 1.0f, seed);
        if(attributes.mean_tensor_uid().has_value())
        {
            randomizeTensor(attributes.mean_tensor_uid().value(), 0.0f, 1.0f, seed);
        }
        if(attributes.inv_variance_tensor_uid().has_value())
        {
            randomizeTensor(attributes.inv_variance_tensor_uid().value(), 0.0f, 1.0f, seed);
        }
    }
};

}
