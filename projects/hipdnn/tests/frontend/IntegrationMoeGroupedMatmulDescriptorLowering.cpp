// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulConstants.hpp>
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
using MoeGroupedMatmulModeSdk = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationMoeGroupedMatmulDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT
        buildAndDeserialize(MoeGroupedMatmulAttributes& attrs,
                            bool includeTokenIndex = false,
                            bool includeTokenKs = false,
                            bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("MoeGroupedMatmulIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto token = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            token->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
        }
        token->set_name("token").set_data_type(DataType::FLOAT);
        token->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));

        auto weight = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            weight->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
        }
        weight->set_name("weight").set_data_type(DataType::FLOAT);
        weight->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));

        auto firstTokenOffset = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            firstTokenOffset->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
        }
        firstTokenOffset->set_name("first_token_offset").set_data_type(DataType::INT32);
        firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

        std::shared_ptr<TensorAttributes> tokenIndex;
        if(includeTokenIndex)
        {
            tokenIndex = std::make_shared<TensorAttributes>();
            if(assignUids)
            {
                tokenIndex->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
            }
            tokenIndex->set_name("token_index").set_data_type(DataType::INT32);
            tokenIndex->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS))
                .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES));
        }

        std::shared_ptr<TensorAttributes> tokenKs;
        if(includeTokenKs)
        {
            tokenKs = std::make_shared<TensorAttributes>();
            if(assignUids)
            {
                tokenKs->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
            }
            tokenKs->set_name("token_ks").set_data_type(DataType::INT32);
            tokenKs->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS))
                .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES));
        }

        auto output = graph->moe_grouped_matmul(
            token, weight, firstTokenOffset, tokenIndex, tokenKs, attrs);
        if(assignUids)
        {
            output->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
        }
        output->set_output(true).set_name("output");

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLowering, MoeGroupedMatmulLoweringRoundTrip)
{
    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_mode(MoeGroupedMatmulMode::NONE);
    attrs.set_top_k(2);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 4u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID]->name, "token");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID]->name, "weight");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->data_type,
              DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID]->name,
              "first_token_offset");
    ASSERT_NE(tensorMap.count(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->dims,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->strides,
              toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID]->name, "output");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::MoeGroupedMatmulAttributes);

    auto* opNode = node->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->token_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    EXPECT_EQ(opNode->weight_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    EXPECT_EQ(opNode->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(opNode->output_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->mode, MoeGroupedMatmulModeSdk::NONE);

    EXPECT_EQ(opNode->top_k, 0);
}

// Verifies an operation-level compute type overrides the graph default during lowering.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLowering, OperationComputeDataTypeOverride)
{
    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_operation_compute_type");
    attrs.set_mode(MoeGroupedMatmulMode::NONE);
    attrs.set_top_k(0);
    attrs.set_compute_data_type(DataType::HALF);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->compute_data_type, DataTypeSdk::HALF);
}

// Verifies the canonical serialized footprint for NONE mode.
// Surplus frontend attributes are accepted for cuDNN compatibility and omitted
// when the selected mode does not serialize them.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLowering, ModeScenarioNoneCanonicalizesIgnoredRouting)
{
    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_none_canonicalizes_ignored_routing");
    attrs.set_mode(MoeGroupedMatmulMode::NONE);
    attrs.set_top_k(2);

    auto graphT = buildAndDeserialize(attrs, true, true);

    ASSERT_EQ(graphT.tensors.size(), 4u);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->mode, MoeGroupedMatmulModeSdk::NONE);
    EXPECT_FALSE(opNode->token_index_tensor_uid.has_value());
    EXPECT_FALSE(opNode->token_ks_tensor_uid.has_value());
    EXPECT_EQ(opNode->top_k, 0);
}
// Verifies the canonical serialized footprint for GATHER mode.
// Surplus frontend attributes are accepted for cuDNN compatibility and omitted
// when the selected mode does not serialize them.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLowering,
       ModeScenarioGatherCanonicalizesIgnoredScatterAttributes)
{
    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_gather_canonicalizes_ignored_scatter_attributes");
    attrs.set_mode(MoeGroupedMatmulMode::GATHER);
    attrs.set_top_k(2);

    auto graphT = buildAndDeserialize(attrs, true, true);

    ASSERT_EQ(graphT.tensors.size(), 5u);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->mode, MoeGroupedMatmulModeSdk::GATHER);
    ASSERT_TRUE(opNode->token_index_tensor_uid.has_value());
    EXPECT_EQ(*opNode->token_index_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    EXPECT_FALSE(opNode->token_ks_tensor_uid.has_value());
    EXPECT_EQ(opNode->top_k, 0);
}
// Verifies the canonical serialized footprint for SCATTER mode.
// Surplus frontend attributes are accepted for cuDNN compatibility and omitted
// when the selected mode does not serialize them.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLowering, ModeScenarioScatterPreservesRouting)
{
    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_scatter_preserves_routing");
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER);
    attrs.set_top_k(2);

    auto graphT = buildAndDeserialize(attrs, true, true);

    ASSERT_EQ(graphT.tensors.size(), 6u);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->mode, MoeGroupedMatmulModeSdk::SCATTER);
    ASSERT_TRUE(opNode->token_index_tensor_uid.has_value());
    EXPECT_EQ(*opNode->token_index_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    ASSERT_TRUE(opNode->token_ks_tensor_uid.has_value());
    EXPECT_EQ(*opNode->token_ks_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(opNode->top_k, 2);
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationMoeGroupedMatmulDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    MoeGroupedMatmulAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_mode(MoeGroupedMatmulMode::NONE);
    attrs.set_top_k(0);

    auto graphT = buildAndDeserialize(attrs, false, false, false);

    ASSERT_EQ(graphT.tensors.size(), 4u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 4u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->token_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->weight_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->first_token_offset_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->output_tensor_uid) > 0);
}

} // namespace
