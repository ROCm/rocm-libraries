// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/SoftmaxConstants.hpp>
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
class IntegrationSoftmaxDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(SoftmaxAttributes& attrs,
                                                                     bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("SoftmaxIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto p = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            p->set_uid(K_SOFTMAX_TENSOR_P_UID);
        }
        p->set_name("p").set_data_type(DataType::FLOAT);
        p->set_dim(toVec(K_SOFTMAX_TENSOR_P_DIMS)).set_stride(toVec(K_SOFTMAX_TENSOR_P_STRIDES));

        auto [s, stats] = graph->softmax(p, attrs);
        if(assignUids)
        {
            s->set_uid(K_SOFTMAX_TENSOR_S_UID);
        }
        s->set_output(true).set_name("s");
        s->set_dim(toVec(K_SOFTMAX_TENSOR_S_DIMS)).set_stride(toVec(K_SOFTMAX_TENSOR_S_STRIDES));
        if(stats)
        {
            if(assignUids)
            {
                stats->set_uid(K_SOFTMAX_TENSOR_STATS_UID);
            }
            stats->set_output(true).set_name("stats").set_data_type(DataType::FLOAT);
            stats->set_dim(toVec(K_SOFTMAX_TENSOR_STATS_DIMS))
                .set_stride(toVec(K_SOFTMAX_TENSOR_STATS_STRIDES));
        }

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationSoftmaxDescriptorLowering, SoftmaxLoweringRoundTrip)
{
    SoftmaxAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_stats_use_log2(true);

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 2u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_P_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->dims, toVec(K_SOFTMAX_TENSOR_P_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->strides, toVec(K_SOFTMAX_TENSOR_P_STRIDES));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->name, "p");
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_S_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->dims, toVec(K_SOFTMAX_TENSOR_S_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->strides, toVec(K_SOFTMAX_TENSOR_S_STRIDES));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->name, "s");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::SoftmaxAttributes);

    auto* opNode = node->attributes.AsSoftmaxAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->p_tensor_uid, K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(opNode->s_tensor_uid, K_SOFTMAX_TENSOR_S_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    EXPECT_EQ(opNode->stats_use_log2, true);
}

// The optional stats output lowers to stats_tensor_uid with dims of p and last dim 1.
TEST_F(IntegrationSoftmaxDescriptorLowering, StatsOutputLowered)
{
    SoftmaxAttributes attrs;
    attrs.set_name("test_stats");
    attrs.set_generate_stats(true);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.tensors.size(), 3u);
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_STATS_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_STATS_UID]->dims, toVec(K_SOFTMAX_TENSOR_STATS_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_STATS_UID]->strides,
              toVec(K_SOFTMAX_TENSOR_STATS_STRIDES));

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsSoftmaxAttributes();
    ASSERT_NE(opNode, nullptr);
    ASSERT_TRUE(opNode->stats_tensor_uid.has_value());
    EXPECT_EQ(*opNode->stats_tensor_uid, K_SOFTMAX_TENSOR_STATS_UID);
    EXPECT_FALSE(opNode->stats_use_log2);
}

// Verifies an operation-level compute type overrides the graph default during lowering.
TEST_F(IntegrationSoftmaxDescriptorLowering, OperationComputeDataTypeOverride)
{
    SoftmaxAttributes attrs;
    attrs.set_name("test_operation_compute_type");
    attrs.set_stats_use_log2(true);
    attrs.set_compute_data_type(DataType::HALF);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    EXPECT_EQ(graphT.nodes[0]->compute_data_type, DataTypeSdk::HALF);
}

} // namespace
