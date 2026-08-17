// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <hipdnn_backend.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>

namespace hipdnn_tests
{

/// Validates a graph, lowers via build_operation_graph_via_descriptors(handle),
/// retrieves the serialized binary graph, and deserializes it into a GraphT.
/// Uses EXPECT macros internally; callers should verify the returned GraphT
/// is well-formed (e.g., check tensors.size()).
inline hipdnn_flatbuffers_sdk::data_objects::GraphT
    lowerAndDeserialize(TestableGraphLowering& graph, hipdnnHandle_t handle)
{
    using hipdnn_frontend::ErrorCode;

    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;

    auto result = graph.validate();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return graphT;
    }

    result = graph.build_operation_graph_via_descriptors(handle);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(result.code != ErrorCode::OK)
    {
        return graphT;
    }

    auto rawDesc = graph.get_raw_graph_descriptor();
    EXPECT_NE(rawDesc, nullptr); // NOLINT(readability-implicit-bool-conversion)
    if(rawDesc == nullptr)
    {
        return graphT;
    }

    size_t serializedSize = 0;
    auto status = hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    if(status != HIPDNN_STATUS_SUCCESS || serializedSize == 0)
    {
        return graphT;
    }

    std::vector<uint8_t> serializedData(serializedSize);
    status = hipdnnBackendGetSerializedBinaryGraph_ext(
        rawDesc, serializedSize, &serializedSize, serializedData.data());
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return graphT;
    }

    auto graphFb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data());
    EXPECT_NE(graphFb, nullptr); // NOLINT(readability-implicit-bool-conversion)
    if(graphFb != nullptr)
    {
        graphFb->UnPackTo(&graphT);
    }

    return graphT;
}

/// Builds a UID-to-TensorAttributesT lookup map from a deserialized GraphT.
inline std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*>
    buildTensorMap(const hipdnn_flatbuffers_sdk::data_objects::GraphT& graphT)
{
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*>
        tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }
    return tensorMap;
}

} // namespace hipdnn_tests
