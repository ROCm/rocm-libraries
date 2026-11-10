// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/MoeRoutingConstants.hpp>
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
class IntegrationMoeRoutingDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(MoeRoutingAttributes& attrs,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("MoeRoutingIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto topkIds = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            topkIds->set_uid(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
        }
        topkIds->set_name("topk_ids").set_data_type(DataType::INT32);
        topkIds->set_dim(toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS))
            .set_stride(toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));

        auto results = graph->moe_routing(topkIds, attrs);
        const auto outputNames
            = std::array<const char*, 3>{"first_token_offset", "token_index", "token_ks"};
        const auto outputUids = std::array<int64_t, 3>{K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                                       K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                                       K_MOE_ROUTING_TENSOR_TOKEN_KS_UID};
        for(size_t i = 0; i < results.size(); ++i)
        {
            results[i]->set_output(true).set_name(outputNames[i]);
            if(assignUids)
            {
                results[i]->set_uid(outputUids[i]);
            }
        }

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationMoeRoutingDescriptorLowering, MoeRoutingLoweringRoundTrip)
{
    MoeRoutingAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_num_experts(4);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 4u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->dims,
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->strides,
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->data_type, DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->name, "topk_ids");
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->dims,
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->strides,
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->data_type,
              DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->name, "first_token_offset");
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->dims,
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->strides,
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->data_type, DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->name, "token_index");
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->dims,
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->strides,
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->data_type, DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->name, "token_ks");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->attributes.type, NodeAttrType::MoeRoutingAttributes);

    auto* opNode = node->attributes.AsMoeRoutingAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->topk_ids_tensor_uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(opNode->first_token_offset_tensor_uid, K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(opNode->token_index_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(opNode->token_ks_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    EXPECT_EQ(opNode->num_experts, 4);
}

} // namespace
