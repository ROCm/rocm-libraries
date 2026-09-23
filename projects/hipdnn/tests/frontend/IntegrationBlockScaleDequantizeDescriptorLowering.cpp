// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationBlockScaleDequantizeDescriptorLowering : public IntegrationTestFixture
{
};

// Builds a block scale dequantize graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationBlockScaleDequantizeDescriptorLowering, BlockScaleDequantizeGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestBlockScaleDequantizeGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BSD_TENSOR_X_UID).set_name("X").set_data_type(DataType::FP8_E4M3);
    x->set_dim(toVec(K_BSD_TENSOR_X_DIMS)).set_stride(toVec(K_BSD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_BSD_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BSD_TENSOR_SCALE_DIMS)).set_stride(toVec(K_BSD_TENSOR_SCALE_STRIDES));

    BlockScaleDequantizeAttributes attrs;
    attrs.set_name("dequantize_op")
        .set_block_size(static_cast<int32_t>(K_BSD_BLOCK_SIZE))
        .set_is_negative_scale(true);

    auto y = graph->block_scale_dequantize(x, scale, attrs);
    y->set_uid(K_BSD_TENSOR_Y_UID).set_name("Y");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_BSD_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FP8_E4M3);
    EXPECT_EQ(xT->dims, toVec(K_BSD_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_BSD_TENSOR_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_BSD_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(scaleT->dims, toVec(K_BSD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_BSD_TENSOR_SCALE_STRIDES));
    EXPECT_FALSE(scaleT->virtual_);

    // Verify Y tensor (dims/strides inferred by frontend to match input)
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_BSD_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->dims, toVec(K_BSD_TENSOR_Y_DIMS));
    EXPECT_EQ(yT->strides, toVec(K_BSD_TENSOR_Y_STRIDES));
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_TRUE(yT->virtual_);

    // -- Verify block scale dequantize operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BlockScaleDequantizeAttributes);

    auto* dequant = node->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(dequant, nullptr);

    EXPECT_EQ(dequant->x_tensor_uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(dequant->scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(dequant->y_tensor_uid, K_BSD_TENSOR_Y_UID);
    ASSERT_EQ(dequant->block_size.size(), 1u);
    EXPECT_EQ(dequant->block_size[0], K_BSD_BLOCK_SIZE);
    EXPECT_TRUE(dequant->is_negative_scale);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationBlockScaleDequantizeDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidBlockScaleDequantizeGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FP8_E4M3);
    x->set_dim(toVec(K_BSD_TENSOR_X_DIMS)).set_stride(toVec(K_BSD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BSD_TENSOR_SCALE_DIMS)).set_stride(toVec(K_BSD_TENSOR_SCALE_STRIDES));

    BlockScaleDequantizeAttributes attrs;
    attrs.set_block_size(static_cast<int32_t>(K_BSD_BLOCK_SIZE));

    auto y = graph->block_scale_dequantize(x, scale, attrs);

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

    // The block scale dequantize operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* dequant = graphT.nodes[0]->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(dequant, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(dequant->x_tensor_uid) > 0)
        << "X tensor UID " << dequant->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(dequant->scale_tensor_uid) > 0)
        << "Scale tensor UID " << dequant->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(dequant->y_tensor_uid) > 0)
        << "Y tensor UID " << dequant->y_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All three tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids
        = {dequant->x_tensor_uid, dequant->scale_tensor_uid, dequant->y_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 3u)
        << "Block scale dequantize node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
