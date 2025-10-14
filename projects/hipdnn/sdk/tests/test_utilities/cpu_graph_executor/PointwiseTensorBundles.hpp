// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

namespace hipdnn_sdk_test_utils
{

struct PointwiseUnaryTensorBundle : public GraphTensorBundle
{
    PointwiseUnaryTensorBundle(
        const hipdnn_plugin::INodeWrapper& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : GraphTensorBundle(tensorMap)
    {
        const auto& attributes = node.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();

        randomizeTensor(attributes.in_0_tensor_uid(), -2.0f, 2.0f, seed);
        // Output tensor not randomized - computed by operation
    }
};

struct PointwiseBinaryTensorBundle : public GraphTensorBundle
{
    PointwiseBinaryTensorBundle(
        const hipdnn_plugin::INodeWrapper& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : GraphTensorBundle(tensorMap)
    {
        const auto& attributes = node.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();

        randomizeTensor(attributes.in_0_tensor_uid(), -1.0f, 1.0f, seed);
        randomizeTensor(attributes.in_1_tensor_uid().value(), -1.0f, 1.0f, seed + 1);
        // Output tensor not randomized - computed by operation
    }
};

}
