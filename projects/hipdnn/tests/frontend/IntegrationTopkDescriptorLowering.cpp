// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/TopkConstants.hpp>
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
class IntegrationTopkDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates x, calls topk(), marks both
    /// outputs as graph outputs, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(TopkAttributes& attrs,
                                                                     bool assignUids = true,
                                                                     DataType indicesType
                                                                     = DataType::NOT_SET)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("TopkIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            x->set_uid(K_TOPK_TENSOR_X_UID);
        }
        x->set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TOPK_TENSOR_X_DIMS)).set_stride(toVec(K_TOPK_TENSOR_X_STRIDES));

        auto [values, indices] = graph->topk(x, attrs);
        if(assignUids)
        {
            values->set_uid(K_TOPK_TENSOR_VALUES_UID);
            indices->set_uid(K_TOPK_TENSOR_INDICES_UID);
        }
        values->set_output(true).set_name("values");
        indices->set_output(true).set_name("indices");
        if(indicesType != DataType::NOT_SET)
        {
            indices->set_data_type(indicesType);
        }

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationTopkDescriptorLowering, TopkLoweringRoundTrip)
{
    TopkAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_k(4);
    attrs.set_largest(false);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_TOPK_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_X_UID]->dims, toVec(K_TOPK_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_X_UID]->strides, toVec(K_TOPK_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_X_UID]->name, "x");
    ASSERT_NE(tensorMap.count(K_TOPK_TENSOR_INDICES_UID), 0u);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_INDICES_UID]->dims, toVec(K_TOPK_TENSOR_INDICES_DIMS));
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_INDICES_UID]->strides, toVec(K_TOPK_TENSOR_INDICES_STRIDES));
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_INDICES_UID]->data_type, DataTypeSdk::INT64);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_INDICES_UID]->name, "indices");
    ASSERT_NE(tensorMap.count(K_TOPK_TENSOR_VALUES_UID), 0u);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_VALUES_UID]->dims, toVec(K_TOPK_TENSOR_VALUES_DIMS));
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_VALUES_UID]->strides, toVec(K_TOPK_TENSOR_VALUES_STRIDES));
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_VALUES_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_VALUES_UID]->name, "values");

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->attributes.type, NodeAttrType::TopkAttributes);

    auto* opNode = node->attributes.AsTopkAttributes();
    ASSERT_NE(opNode, nullptr);

    EXPECT_EQ(opNode->x_tensor_uid, K_TOPK_TENSOR_X_UID);
    EXPECT_EQ(opNode->indices_tensor_uid, K_TOPK_TENSOR_INDICES_UID);
    ASSERT_TRUE(opNode->values_tensor_uid.has_value());
    EXPECT_EQ(opNode->values_tensor_uid.value(), K_TOPK_TENSOR_VALUES_UID);

    EXPECT_EQ(node->name, "test_op");
    EXPECT_EQ(opNode->k, 4);
    EXPECT_FALSE(opNode->largest);
}

// The default largest = true survives lowering.
TEST_F(IntegrationTopkDescriptorLowering, LargestDefaultPreserved)
{
    TopkAttributes attrs;
    attrs.set_k(1);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsTopkAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->k, 1);
    EXPECT_TRUE(opNode->largest);
}

// An explicit INT32 indices data type survives lowering.
TEST_F(IntegrationTopkDescriptorLowering, Int32IndicesPreserved)
{
    TopkAttributes attrs;
    attrs.set_k(4);

    auto graphT = buildAndDeserialize(attrs, true, DataType::INT32);

    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_TOPK_TENSOR_INDICES_UID), 0u);
    EXPECT_EQ(tensorMap[K_TOPK_TENSOR_INDICES_UID]->data_type, DataTypeSdk::INT32);
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationTopkDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    TopkAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_k(4);

    auto graphT = buildAndDeserialize(attrs, false);

    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 3u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsTopkAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->indices_tensor_uid) > 0);
    ASSERT_TRUE(opNode->values_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(opNode->values_tensor_uid.value()) > 0);
}

} // namespace
