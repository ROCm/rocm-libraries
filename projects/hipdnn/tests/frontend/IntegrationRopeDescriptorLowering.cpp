// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/RopeConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationRopeDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(RopeAttributes& attrs,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("RopeIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            x->set_uid(K_ROPE_TENSOR_X_UID);
        }
        x->set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_ROPE_TENSOR_X_DIMS)).set_stride(toVec(K_ROPE_TENSOR_X_STRIDES));

        auto freqs = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            freqs->set_uid(K_ROPE_TENSOR_FREQS_UID);
        }
        freqs->set_name("freqs").set_data_type(DataType::FLOAT);
        freqs->set_dim(toVec(K_ROPE_TENSOR_FREQS_DIMS))
            .set_stride(toVec(K_ROPE_TENSOR_FREQS_STRIDES));

        auto y = graph->rope(x, freqs, attrs);
        if(assignUids)
        {
            y->set_uid(K_ROPE_TENSOR_Y_UID);
        }
        y->set_output(true).set_name("y");

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationRopeDescriptorLowering, RopeLoweringRoundTrip)
{
    RopeAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);
    attrs.set_rope_offset(0);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 3u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->dims, toVec(K_ROPE_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->strides, toVec(K_ROPE_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->name, "x");
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_FREQS_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->dims, toVec(K_ROPE_TENSOR_FREQS_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->strides, toVec(K_ROPE_TENSOR_FREQS_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->name, "freqs");
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->dims, toVec(K_ROPE_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->strides, toVec(K_ROPE_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->name, "y");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    // RoPE carries no compute-type attribute (cuDNN's rope node sets none).
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::UNSET);
    EXPECT_EQ(node->attributes.type, NodeAttrType::RopeAttributes);

    auto* opNode = node->attributes.AsRopeAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->x_tensor_uid, K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(opNode->freqs_tensor_uid, K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(opNode->y_tensor_uid, K_ROPE_TENSOR_Y_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    EXPECT_EQ(opNode->output_scale, 0.5);
    EXPECT_EQ(opNode->rope_dim, 4);
    EXPECT_EQ(opNode->rope_offset_ext, 0);
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationRopeDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    RopeAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);
    attrs.set_rope_offset(0);

    auto graphT = buildAndDeserialize(attrs, false);

    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 3u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsRopeAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->freqs_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->y_tensor_uid) > 0);
}

// Verifies that the optional rope_offset attribute survives lowering round-trip.
TEST_F(IntegrationRopeDescriptorLowering, RopeOffsetPreservedInRoundTrip)
{
    RopeAttributes attrs;
    attrs.set_name("test_rope_offset");
    attrs.set_rope_dim(4);
    attrs.set_rope_offset(2);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsRopeAttributes();
    ASSERT_NE(opNode, nullptr);

    ASSERT_TRUE(opNode->rope_offset_ext.has_value());
    EXPECT_EQ(opNode->rope_offset_ext.value(), 2);
}

} // namespace
