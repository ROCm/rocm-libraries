// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/gather_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/GatherConstants.hpp>
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
class IntegrationGatherDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(GatherAttributes& attrs,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("GatherIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            x->set_uid(K_GATHER_TENSOR_X_UID);
        }
        x->set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_GATHER_TENSOR_X_DIMS)).set_stride(toVec(K_GATHER_TENSOR_X_STRIDES));

        auto indices = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            indices->set_uid(K_GATHER_TENSOR_INDICES_UID);
        }
        indices->set_name("indices").set_data_type(DataType::INT32);
        indices->set_dim(toVec(K_GATHER_TENSOR_INDICES_DIMS))
            .set_stride(toVec(K_GATHER_TENSOR_INDICES_STRIDES));

        auto y = graph->gather(x, indices, attrs);
        if(assignUids)
        {
            y->set_uid(K_GATHER_TENSOR_Y_UID);
        }
        y->set_output(true).set_name("y");

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationGatherDescriptorLowering, GatherLoweringRoundTrip)
{
    GatherAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_axis(0);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 3u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_GATHER_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_X_UID]->dims, toVec(K_GATHER_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_X_UID]->strides, toVec(K_GATHER_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_X_UID]->name, "x");
    ASSERT_NE(tensorMap.count(K_GATHER_TENSOR_INDICES_UID), 0u);
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_INDICES_UID]->dims, toVec(K_GATHER_TENSOR_INDICES_DIMS));
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_INDICES_UID]->strides,
              toVec(K_GATHER_TENSOR_INDICES_STRIDES));
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_INDICES_UID]->data_type, DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_INDICES_UID]->name, "indices");
    ASSERT_NE(tensorMap.count(K_GATHER_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_Y_UID]->dims, toVec(K_GATHER_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_Y_UID]->strides, toVec(K_GATHER_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_Y_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_GATHER_TENSOR_Y_UID]->name, "y");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    // Gather has no compute-type attribute: it moves data without arithmetic.
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::UNSET);
    EXPECT_EQ(node->attributes.type, NodeAttrType::GatherAttributes);

    auto* opNode = node->attributes.AsGatherAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->x_tensor_uid, K_GATHER_TENSOR_X_UID);
    EXPECT_EQ(opNode->indices_tensor_uid, K_GATHER_TENSOR_INDICES_UID);
    EXPECT_EQ(opNode->y_tensor_uid, K_GATHER_TENSOR_Y_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    EXPECT_EQ(opNode->axis, 0);
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationGatherDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    GatherAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_axis(0);

    auto graphT = buildAndDeserialize(attrs, false);

    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 3u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsGatherAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->indices_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->y_tensor_uid) > 0);
}

} // namespace
