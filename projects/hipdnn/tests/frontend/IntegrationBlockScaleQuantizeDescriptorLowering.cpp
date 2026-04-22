// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/BlockScaleQuantizeConstants.hpp>
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
class IntegrationBlockScaleQuantizeDescriptorLowering : public IntegrationTestFixture
{
};

// Builds a block scale quantize graph via the frontend API, lowers it to the
// backend via build_operation_graph_via_descriptors, retrieves the serialized
// graph, and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLowering, BlockScaleQuantizeGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestBlockScaleQuantizeGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BSQ_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BSQ_TENSOR_X_DIMS)).set_stride(toVec(K_BSQ_TENSOR_X_STRIDES));

    BlockScaleQuantizeAttributes bsqAttrs;
    bsqAttrs.set_name("bsq_op");
    bsqAttrs.set_block_size(K_BSQ_BLOCK_SIZE);

    auto [y, scale] = graph->block_scale_quantize(x, std::move(bsqAttrs));
    y->set_uid(K_BSQ_TENSOR_Y_UID).set_output(true).set_name("Y");
    scale->set_uid(K_BSQ_TENSOR_SCALE_UID).set_output(true).set_name("Scale");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_BSQ_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_BSQ_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_BSQ_TENSOR_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_BSQ_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(yT->dims, toVec(K_BSQ_TENSOR_Y_DIMS));
    EXPECT_EQ(yT->strides, toVec(K_BSQ_TENSOR_Y_STRIDES));
    EXPECT_FALSE(yT->virtual_);

    // Verify Scale tensor (dims/strides inferred by frontend)
    ASSERT_NE(tensorMap.count(K_BSQ_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_BSQ_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_FALSE(scaleT->virtual_);

    // -- Verify block scale quantize operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BlockScaleQuantizeAttributes);

    auto* bsq = node->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(bsq, nullptr);

    EXPECT_EQ(bsq->x_tensor_uid, K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(bsq->y_tensor_uid, K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(bsq->scale_tensor_uid, K_BSQ_TENSOR_SCALE_UID);

    EXPECT_EQ(bsq->block_size, K_BSQ_BLOCK_SIZE);
    EXPECT_FALSE(bsq->axis.has_value());
    EXPECT_FALSE(bsq->transpose);
}

// Verifies that axis and transpose survive the descriptor-lowering round-trip
// and that Y strides are correctly inferred under transpose.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLowering,
       BlockScaleQuantizeGraphRoundTripWithAxisAndTranspose)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestBsqAxisTransposeGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Use dims where dim[axis] is divisible by block_size (64 / 32 = 2)
    const std::vector<int64_t> xDims = {2, 64, 32, 32};
    const std::vector<int64_t> xStrides = {65536, 1024, 32, 1};
    constexpr int32_t BLOCK_SIZE = 32;
    constexpr int64_t X_UID = 50;
    constexpr int64_t Y_UID = 51;
    constexpr int64_t SCALE_UID = 52;

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(xDims).set_stride(xStrides);

    BlockScaleQuantizeAttributes bsqAttrs;
    bsqAttrs.set_name("bsq_transpose_op");
    bsqAttrs.set_block_size(BLOCK_SIZE);
    bsqAttrs.set_axis(1);
    bsqAttrs.set_transpose(true);

    auto [y, scale] = graph->block_scale_quantize(x, std::move(bsqAttrs));
    y->set_uid(Y_UID).set_output(true).set_name("Y");
    scale->set_uid(SCALE_UID).set_output(true).set_name("Scale");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);

    // Y dims match X dims; strides are reordered by transpose
    ASSERT_NE(tensorMap.count(Y_UID), 0u);
    auto* yT = tensorMap[Y_UID];
    EXPECT_EQ(yT->dims, xDims);
    // Expected transposed strides: sort X stride indices ascending [3,2,1,0],
    // rotate axis=1 to front [1,0,3,2], inverse perm gives strideOrder [1,0,3,2]
    EXPECT_EQ(yT->strides, (std::vector<int64_t>{64, 1, 4096, 128}));

    // -- Verify BSQ operation attributes --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bsq = graphT.nodes[0]->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(bsq, nullptr);

    EXPECT_EQ(bsq->block_size, BLOCK_SIZE);
    EXPECT_TRUE(bsq->axis.has_value());
    EXPECT_EQ(bsq->axis.value(), 1);
    EXPECT_TRUE(bsq->transpose);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationBlockScaleQuantizeDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidBsqGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BSQ_TENSOR_X_DIMS)).set_stride(toVec(K_BSQ_TENSOR_X_STRIDES));

    BlockScaleQuantizeAttributes bsqAttrs;
    bsqAttrs.set_block_size(K_BSQ_BLOCK_SIZE);

    auto [y, scale] = graph->block_scale_quantize(x, std::move(bsqAttrs));
    y->set_output(true);
    scale->set_output(true);

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

    // The block scale quantize operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bsq = graphT.nodes[0]->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(bsq, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(bsq->x_tensor_uid) > 0)
        << "X tensor UID " << bsq->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bsq->y_tensor_uid) > 0)
        << "Y tensor UID " << bsq->y_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bsq->scale_tensor_uid) > 0)
        << "Scale tensor UID " << bsq->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All three tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids
        = {bsq->x_tensor_uid, bsq->y_tensor_uid, bsq->scale_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 3u)
        << "Block scale quantize node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
