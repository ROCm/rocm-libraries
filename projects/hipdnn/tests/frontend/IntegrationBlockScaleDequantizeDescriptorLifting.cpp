// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{
// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationBlockScaleDequantizeDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard BlockScaleDequantize graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("BlockScaleDequantizeLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_BSD_TENSOR_X_UID).set_name("X").set_data_type(DataType::FP8_E4M3);
        x->set_dim(toVec(K_BSD_TENSOR_X_DIMS)).set_stride(toVec(K_BSD_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_BSD_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_BSD_TENSOR_SCALE_DIMS))
            .set_stride(toVec(K_BSD_TENSOR_SCALE_STRIDES));

        BlockScaleDequantizeAttributes attrs;
        attrs.set_name("test_op")
            .set_block_size(static_cast<int32_t>(K_BSD_BLOCK_SIZE))
            .set_is_negative_scale(false);

        auto y = graph->block_scale_dequantize(x, scale, attrs);
        y->set_uid(K_BSD_TENSOR_Y_UID).set_name("Y");

        return graph;
    }
};

// Builds a standard BlockScaleDequantize graph, lowers via
// build_operation_graph(handle), lifts back with
// fromBackendDescriptor(), and performs comprehensive field-by-field
// validation.
TEST_F(IntegrationBlockScaleDequantizeDescriptorLifting, BasicBlockScaleDequantizeRoundTrip)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_uid(), K_BSD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_dim(), toVec(K_BSD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_stride(), toVec(K_BSD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_data_type(), DataType::FP8_E4M3);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_name(), "X");

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_uid(), K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_dim(), toVec(K_BSD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_stride(), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_name(), "Scale");

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID]->get_uid(), K_BSD_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID]->get_dim(), toVec(K_BSD_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID]->get_stride(), toVec(K_BSD_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID]->get_name(), "Y");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<BlockScaleDequantizeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a BlockScaleDequantizeNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify block_size
    ASSERT_EQ(opNode->attributes.get_block_size().size(), 1u);
    EXPECT_EQ(opNode->attributes.get_block_size()[0], K_BSD_BLOCK_SIZE);

    // Verify is_negative_scale
    EXPECT_EQ(opNode->attributes.get_is_negative_scale(), false);

    // Verify compute data type
    EXPECT_EQ(opNode->attributes.compute_data_type, DataType::FLOAT);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationBlockScaleDequantizeDescriptorLifting, BlockScaleDequantizeTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<BlockScaleDequantizeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify tensor sharing (pointer equality)
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_BSD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    EXPECT_EQ(opNode->attributes.get_scale()->get_uid(), K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID].get(), opNode->attributes.get_scale().get());
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_BSD_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a BlockScaleDequantize graph, serializes to binary, creates a
// backend descriptor from bytes (no handle, no finalize), calls
// fromBackendDescriptor(), and verifies all fields survive the
// FlatBuffer-direct path.
TEST_F(IntegrationBlockScaleDequantizeDescriptorLifting,
       BlockScaleDequantizeLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<BlockScaleDequantizeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify block_size
    ASSERT_EQ(opNode->attributes.get_block_size().size(), 1u);
    EXPECT_EQ(opNode->attributes.get_block_size()[0], K_BSD_BLOCK_SIZE);

    // Verify is_negative_scale
    EXPECT_EQ(opNode->attributes.get_is_negative_scale(), false);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_dim(), toVec(K_BSD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_X_UID]->get_stride(), toVec(K_BSD_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_dim(), toVec(K_BSD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_SCALE_UID]->get_stride(), toVec(K_BSD_TENSOR_SCALE_STRIDES));
    ASSERT_NE(tensorMap.count(K_BSD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID]->get_dim(), toVec(K_BSD_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_BSD_TENSOR_Y_UID]->get_stride(), toVec(K_BSD_TENSOR_Y_STRIDES));
}

// Verifies that the is_negative_scale attribute survives a lifting
// round-trip.
TEST_F(IntegrationBlockScaleDequantizeDescriptorLifting, IsNegativeScalePreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("BlockScaleDequantizeIsNegativeScaleLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BSD_TENSOR_X_UID).set_name("X").set_data_type(DataType::FP8_E4M3);
    x->set_dim(toVec(K_BSD_TENSOR_X_DIMS)).set_stride(toVec(K_BSD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_BSD_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BSD_TENSOR_SCALE_DIMS)).set_stride(toVec(K_BSD_TENSOR_SCALE_STRIDES));

    BlockScaleDequantizeAttributes attrs;
    attrs.set_name("test_is_negative_scale")
        .set_block_size(static_cast<int32_t>(K_BSD_BLOCK_SIZE))
        .set_is_negative_scale(true);

    auto y = graph->block_scale_dequantize(x, scale, attrs);
    y->set_uid(K_BSD_TENSOR_Y_UID).set_name("Y");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<BlockScaleDequantizeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a BlockScaleDequantizeNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(opNode->attributes.get_is_negative_scale(), true);
}

// Verifies that tensors without explicit UIDs are auto-assigned unique UIDs
// and that these UIDs are correctly preserved through the lifting round-trip.
TEST_F(IntegrationBlockScaleDequantizeDescriptorLifting, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("AutoUidBlockScaleDequantizeLiftingGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Create tensors without explicit UIDs
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FP8_E4M3);
    x->set_dim(toVec(K_BSD_TENSOR_X_DIMS)).set_stride(toVec(K_BSD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BSD_TENSOR_SCALE_DIMS)).set_stride(toVec(K_BSD_TENSOR_SCALE_STRIDES));

    BlockScaleDequantizeAttributes attrs;
    attrs.set_block_size(static_cast<int32_t>(K_BSD_BLOCK_SIZE));

    auto y = graph->block_scale_dequantize(x, scale, attrs);

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify all tensors have unique UIDs
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    std::unordered_set<int64_t> uids;
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.insert(uid);
    }
    EXPECT_EQ(uids.size(), 3u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // Verify the node references tensors from the map
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<BlockScaleDequantizeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify the node's tensors have UIDs that exist in the tensor map
    auto xUid = opNode->attributes.get_x()->get_uid();
    auto scaleUid = opNode->attributes.get_scale()->get_uid();
    auto yUid = opNode->attributes.get_y()->get_uid();

    EXPECT_TRUE(uids.count(xUid) > 0)
        << "X tensor UID " << xUid
        << " not found in lifted graph"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(scaleUid) > 0)
        << "Scale tensor UID " << scaleUid
        << " not found in lifted graph"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(yUid) > 0)
        << "Y tensor UID " << yUid
        << " not found in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    // All three tensor UIDs should be distinct
    const std::unordered_set<int64_t> nodeUids = {xUid, scaleUid, yUid};
    EXPECT_EQ(nodeUids.size(), 3u)
        << "Block scale dequantize node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
