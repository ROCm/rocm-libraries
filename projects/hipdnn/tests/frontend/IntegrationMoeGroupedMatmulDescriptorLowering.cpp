// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
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
        buildAndDeserialize(MoeGroupedMatmulAttributes& attrs)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("MoeGroupedMatmulIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto token = std::make_shared<TensorAttributes>();
        token->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID)
            .set_name("token")
            .set_data_type(DataType::FLOAT);
        token->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES));

        auto weight = std::make_shared<TensorAttributes>();
        weight->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID)
            .set_name("weight")
            .set_data_type(DataType::FLOAT);
        weight->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES));

        auto firstTokenOffset = std::make_shared<TensorAttributes>();
        firstTokenOffset->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID)
            .set_name("first_token_offset")
            .set_data_type(DataType::FLOAT);
        firstTokenOffset->set_dim(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS))
            .set_stride(toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));

        const std::shared_ptr<TensorAttributes> tokenIndex;

        const std::shared_ptr<TensorAttributes> tokenKs;

        auto output = graph->moe_grouped_matmul(
            token, weight, firstTokenOffset, tokenIndex, tokenKs, attrs);
        output->set_uid(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID).set_output(true).set_name("output");

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
              DataTypeSdk::FLOAT);
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

    EXPECT_EQ(opNode->top_k, 2);
}

} // namespace
