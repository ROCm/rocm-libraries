// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_bwd_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulBwdConstants.hpp>
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
class IntegrationMoeGroupedMatmulBwdDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT
        buildAndDeserialize(MoeGroupedMatmulBwdAttributes& attrs, bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("MoeGroupedMatmulBwdIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dOutput = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            dOutput->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
        }
        dOutput->set_name("doutput").set_data_type(DataType::FLOAT);
        dOutput->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));

        auto token = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            token->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
        }
        token->set_name("token").set_data_type(DataType::FLOAT);
        token->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));

        auto firstTokenOffset = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            firstTokenOffset->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
        }
        firstTokenOffset->set_name("first_token_offset").set_data_type(DataType::INT32);
        firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

        auto dweight = graph->moe_grouped_matmul_bwd(dOutput, token, firstTokenOffset, attrs);
        if(assignUids)
        {
            dweight->set_uid(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
        }
        dweight->set_output(true).set_name("dweight");
        dweight->set_dim(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLowering, MoeGroupedMatmulBwdLoweringRoundTrip)
{
    MoeGroupedMatmulBwdAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 4u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->data_type,
              DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID]->name, "doutput");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID]->name, "token");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->data_type,
              DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID]->name,
              "first_token_offset");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->data_type,
              DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID]->name, "dweight");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::MoeGroupedMatmulBwdAttributes);

    auto* opNode = node->attributes.AsMoeGroupedMatmulBwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->doutput_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    EXPECT_EQ(opNode->token_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    EXPECT_EQ(opNode->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(opNode->dweight_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");
}

// Verifies an operation-level compute type overrides the graph default during lowering.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLowering, OperationComputeDataTypeOverride)
{
    MoeGroupedMatmulBwdAttributes attrs;
    attrs.set_name("test_operation_compute_type");
    attrs.set_compute_data_type(DataType::HALF);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->compute_data_type, DataTypeSdk::HALF);
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationMoeGroupedMatmulBwdDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    MoeGroupedMatmulBwdAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto graphT = buildAndDeserialize(attrs, false);

    ASSERT_EQ(graphT.tensors.size(), 4u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 4u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsMoeGroupedMatmulBwdAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->doutput_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->token_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->first_token_offset_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->dweight_tensor_uid) > 0);
}

} // namespace
