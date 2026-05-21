// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/MatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationMatmulDescriptorLowering : public IntegrationTestFixture
{
};

// Builds a matmul graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationMatmulDescriptorLowering, MatmulGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestMatmulGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = std::make_shared<TensorAttributes>();
    a->set_uid(K_MATMUL_TENSOR_A_UID).set_name("A").set_data_type(DataType::FLOAT);
    a->set_dim(toVec(K_MATMUL_TENSOR_A_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_A_STRIDES));

    auto b = std::make_shared<TensorAttributes>();
    b->set_uid(K_MATMUL_TENSOR_B_UID).set_name("B").set_data_type(DataType::FLOAT);
    b->set_dim(toVec(K_MATMUL_TENSOR_B_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_B_STRIDES));

    MatmulAttributes matmulAttrs;
    matmulAttrs.set_name("matmul_op");

    auto c = graph->matmul(a, b, matmulAttrs);
    c->set_uid(K_MATMUL_TENSOR_C_UID).set_output(true).set_name("C");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify A tensor
    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_A_UID), 0u);
    auto* aT = tensorMap[K_MATMUL_TENSOR_A_UID];
    EXPECT_EQ(aT->name, "A");
    EXPECT_EQ(aT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(aT->dims, toVec(K_MATMUL_TENSOR_A_DIMS));
    EXPECT_EQ(aT->strides, toVec(K_MATMUL_TENSOR_A_STRIDES));
    EXPECT_FALSE(aT->virtual_);

    // Verify B tensor
    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_B_UID), 0u);
    auto* bT = tensorMap[K_MATMUL_TENSOR_B_UID];
    EXPECT_EQ(bT->name, "B");
    EXPECT_EQ(bT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(bT->dims, toVec(K_MATMUL_TENSOR_B_DIMS));
    EXPECT_EQ(bT->strides, toVec(K_MATMUL_TENSOR_B_STRIDES));
    EXPECT_FALSE(bT->virtual_);

    // Verify C tensor (dims/strides inferred by frontend to match shared constants)
    ASSERT_NE(tensorMap.count(K_MATMUL_TENSOR_C_UID), 0u);
    auto* cT = tensorMap[K_MATMUL_TENSOR_C_UID];
    EXPECT_EQ(cT->name, "C");
    EXPECT_EQ(cT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(cT->dims, toVec(K_MATMUL_TENSOR_C_DIMS));
    EXPECT_EQ(cT->strides, toVec(K_MATMUL_TENSOR_C_STRIDES));
    EXPECT_FALSE(cT->virtual_);

    // -- Verify matmul operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::MatmulAttributes);

    auto* matmul = node->attributes.AsMatmulAttributes();
    ASSERT_NE(matmul, nullptr);

    EXPECT_EQ(matmul->a_tensor_uid, K_MATMUL_TENSOR_A_UID);
    EXPECT_EQ(matmul->b_tensor_uid, K_MATMUL_TENSOR_B_UID);
    EXPECT_EQ(matmul->c_tensor_uid, K_MATMUL_TENSOR_C_UID);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationMatmulDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidMatmulGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = std::make_shared<TensorAttributes>();
    a->set_name("A").set_data_type(DataType::FLOAT);
    a->set_dim(toVec(K_MATMUL_TENSOR_A_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_A_STRIDES));

    auto b = std::make_shared<TensorAttributes>();
    b->set_name("B").set_data_type(DataType::FLOAT);
    b->set_dim(toVec(K_MATMUL_TENSOR_B_DIMS)).set_stride(toVec(K_MATMUL_TENSOR_B_STRIDES));

    const MatmulAttributes matmulAttrs;

    auto c = graph->matmul(a, b, matmulAttrs);
    c->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // All tensors should have been auto-assigned unique UIDs
    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 3u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The matmul operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* matmul = graphT.nodes[0]->attributes.AsMatmulAttributes();
    ASSERT_NE(matmul, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(matmul->a_tensor_uid) > 0)
        << "A tensor UID " << matmul->a_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(matmul->b_tensor_uid) > 0)
        << "B tensor UID " << matmul->b_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(matmul->c_tensor_uid) > 0)
        << "C tensor UID " << matmul->c_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All three tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids
        = {matmul->a_tensor_uid, matmul->b_tensor_uid, matmul->c_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 3u)
        << "Matmul node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
