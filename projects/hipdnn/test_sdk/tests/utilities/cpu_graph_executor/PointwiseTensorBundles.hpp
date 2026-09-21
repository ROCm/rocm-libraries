// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

namespace hipdnn_sdk_test_utils
{

struct PointwiseUnaryTensorBundle : public hipdnn_test_sdk::utilities::GraphTensorBundle
{
    PointwiseUnaryTensorBundle(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : hipdnn_test_sdk::utilities::GraphTensorBundle(tensorMap)
    {
        const auto& attributes
            = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

        randomizeTensor(attributes.in_0_tensor_uid(), -2.0f, 2.0f, seed);
        // Output tensor not randomized - computed by operation
    }
};

struct PointwiseBinaryTensorBundle : public hipdnn_test_sdk::utilities::GraphTensorBundle
{
    PointwiseBinaryTensorBundle(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : hipdnn_test_sdk::utilities::GraphTensorBundle(tensorMap)
    {
        const auto& attributes
            = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

        randomizeTensor(attributes.in_0_tensor_uid(), -1.0f, 1.0f, seed);
        randomizeTensor(attributes.in_1_tensor_uid().value(), -1.0f, 1.0f, seed + 1);
        // Output tensor not randomized - computed by operation
    }
};

}
