// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <hipdnn.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>

namespace hipdnn_tests
{

/// Validates a graph, lowers via build_operation_graph_via_descriptors(handle),
/// retrieves the serialized binary graph, and deserializes it into a GraphT.
/// Uses EXPECT macros internally; callers should verify the returned GraphT
/// is well-formed (e.g., check tensors.size()).
inline hipdnn_data_sdk::data_objects::GraphT lowerAndDeserialize(TestableGraphLowering& graph,
                                                                 hipdnnHandle_t handle)
{
    using hipdnn_frontend::ErrorCode;

    auto result = graph.validate();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph.build_operation_graph_via_descriptors(handle);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph.get_raw_graph_descriptor();
    EXPECT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<uint8_t> serializedData(serializedSize);
    EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    if(!serializedData.empty())
    {
        auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
        EXPECT_NE(graphFb, nullptr);
        if(graphFb != nullptr)
        {
            graphFb->UnPackTo(&graphT);
        }
    }
    return graphT;
}

/// Builds a UID-to-TensorAttributesT lookup map from a deserialized GraphT.
inline std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*>
    buildTensorMap(const hipdnn_data_sdk::data_objects::GraphT& graphT)
{
    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }
    return tensorMap;
}

} // namespace hipdnn_tests
